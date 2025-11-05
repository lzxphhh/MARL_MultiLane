"""Multi-policy buffer for multi-head critic that uses Environment-Provided (EP) state."""
import torch
import numpy as np
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.utils.trans_tools import _flatten, _sa_cast


class MultiPolicyCriticBufferEP:
    """Multi-policy buffer for multi-head critic that uses Environment-Provided (EP) state.
    Each instance handles data for one specific scene class.
    Supports multi-head reward components and dynamic weights.
    """

    def __init__(self, args, share_obs_space):
        """Initialize multi-policy multi-head critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        self.episode_length = args["episode_length"]  # 每个环境的episode长度
        self.n_rollout_threads = args["n_rollout_threads"] # 多进程环境数量
        self.hidden_sizes = args["hidden_sizes"] # critic网络的隐藏层大小
        self.rnn_hidden_size = self.hidden_sizes[-1] # rnn隐藏层大小
        self.recurrent_n = args["recurrent_n"] # rnn的层数

        self.gamma = args["gamma"]  # 折扣因子
        self.gae_lambda = args["gae_lambda"]  # GAE的参数
        self.use_gae = args["use_gae"]  # 是否使用GAE

        self.use_proper_time_limits = args["use_proper_time_limits"]  # 是否考虑episode的提前结束

        share_obs_shape = get_shape_from_obs_space(share_obs_space)  # 获取单个智能体共享状态空间的形状，tuple of integer. eg: （54，）

        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        # 多头critic相关参数
        self.critic_head_num = args["critic_head_num"]
        self.critic_head_names = args["critic_head_names"]
        self.num_agents = args["num_CAVs"]

        # Buffer for share observations

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of critic
        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for value predictions made by this critic (for each head)
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.critic_head_num), dtype=np.float32
        )

        # Buffer for returns calculated at each timestep (for each head)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.critic_head_num), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for rewards received by agents at each timestep (for each component)
        self.reward_components = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.critic_head_num), dtype=np.float32
        )

        # Buffer for masks indicating whether an episode is done at each timestep
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for bad masks indicating truncation and termination
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
            self, share_obs, rnn_states_critic, value_preds, reward_components, masks, bad_masks
    ):
        """Insert data into buffer.
        Args:
            share_obs: (np.ndarray) share observations
            rnn_states_critic: (np.ndarray) RNN states for critic
            value_preds: (np.ndarray) value predictions for each head
            reward_components: (np.ndarray) reward components for each head
            masks: (np.ndarray) masks indicating done episodes
            bad_masks: (np.ndarray) masks indicating truncation vs termination
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        # 确保value_preds形状正确
        if len(value_preds.shape) == 3 and value_preds.shape[2] == self.critic_head_num:
            self.value_preds[self.step] = value_preds.copy()
        elif len(value_preds.shape) == 4 and value_preds.shape[2] == self.critic_head_num and value_preds.shape[3] == 1:
            self.value_preds[self.step] = value_preds[:, :, :, 0].copy()
        else:
            # 处理形状不匹配的情况
            if value_preds.shape[-1] == self.critic_head_num:
                self.value_preds[self.step] = value_preds.reshape(self.n_rollout_threads, self.critic_head_num)
            else:
                raise ValueError(f"Value preds shape {value_preds.shape} not compatible with critic_head_num {self.critic_head_num}")

        # 确保reward_components形状正确
        if len(reward_components.shape) == 2 and reward_components.shape[1] == self.critic_head_num:
            self.reward_components[self.step] = reward_components.copy()
        elif len(reward_components.shape) == 3 and reward_components.shape[2] == self.critic_head_num:
            self.reward_components[self.step] = reward_components.copy()
        else:
            # 处理形状不匹配的情况
            if reward_components.shape[-1] == self.critic_head_num:
                self.reward_components[self.step] = reward_components.reshape(self.n_rollout_threads, self.critic_head_num)
            else:
                raise ValueError(f"Reward components shape {reward_components.shape} not compatible with critic_head_num {self.critic_head_num}")

        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        """Get mean rewards for logging."""
        return np.mean(self.reward_components, axis=(0, 1))

    def get_reward_statistics(self):
        """Get reward statistics for dynamic weight computation.
        Returns:
            dict: reward statistics for each head
        """
        stats = {}
        for head_idx in range(self.critic_head_num):
            head_name = self.critic_head_names.get(head_idx, f"head_{head_idx}")
            head_rewards = self.reward_components[:, :, head_idx]

            stats[head_name] = {
                'mean': np.mean(head_rewards),
                'std': np.std(head_rewards),
                'min': np.min(head_rewards),
                'max': np.max(head_rewards),
                'sum': np.sum(head_rewards)
            }

        return stats

    def compute_returns(self, next_values, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_values: (np.ndarray) value predictions for the step after the last episode step. [n_rollout_threads, critic_head_num]
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        # 对每个头分别计算returns
        for head_idx in range(self.critic_head_num):
            # 提取当前头的值
            head_next_value = next_values[:, head_idx:head_idx + 1]
            head_values = self.value_preds[:, :, head_idx:head_idx + 1]
            head_rewards = self.reward_components[:, :, head_idx:head_idx + 1]
            head_returns = self.returns[:, :, head_idx:head_idx + 1]

            # 以下和mappo相同的计算逻辑
            if self.use_proper_time_limits:
                if self.use_gae:
                    head_values[-1] = head_next_value
                    gae = 0
                    for step in reversed(range(head_rewards.shape[0])):
                        if value_normalizer is not None:
                            delta = (
                                    head_rewards[step]
                                    + self.gamma
                                    * value_normalizer.denormalize(head_values[step + 1])
                                    * self.masks[step + 1]
                                    - value_normalizer.denormalize(head_values[step])
                            )
                            gae = (
                                    delta
                                    + self.gamma * self.gae_lambda
                                    * self.masks[step + 1]
                                    * gae
                            )
                            gae = self.bad_masks[step + 1] * gae
                            head_returns[step] = gae + value_normalizer.denormalize(head_values[step])
                        else:
                            delta = (
                                    head_rewards[step]
                                    + self.gamma
                                    * head_values[step + 1]
                                    * self.masks[step + 1]
                                    - head_values[step]
                            )
                            gae = (
                                    delta
                                    + self.gamma * self.gae_lambda
                                    * self.masks[step + 1]
                                    * gae
                            )
                            gae = self.bad_masks[step + 1] * gae
                            head_returns[step] = gae + head_values[step]
                else:
                    head_returns[-1] = head_next_value
                    for step in reversed(range(head_rewards.shape[0])):
                        if value_normalizer is not None:
                            discounted_cum_reward = (
                                    head_returns[step + 1] * self.gamma * self.masks[step + 1]
                                    + head_rewards[step]
                            )
                            discounted_value = (
                                    (1 - self.bad_masks[step + 1])
                                    * value_normalizer.denormalize(head_values[step])
                            )
                            head_returns[step] = (
                                    discounted_cum_reward * self.bad_masks[step + 1]
                                    + discounted_value
                            )
                        else:
                            head_returns[step] = (
                                    (
                                            head_returns[step + 1]
                                            * self.gamma
                                            * self.masks[step + 1]
                                            + head_rewards[step]
                                    )
                                    * self.bad_masks[step + 1]
                                    + (1 - self.bad_masks[step + 1])
                                    * head_values[step]
                            )
            else:
                if self.use_gae:
                    head_values[-1] = head_next_value
                    gae = 0
                    for step in reversed(range(head_rewards.shape[0])):
                        if value_normalizer is not None:
                            delta = (
                                    head_rewards[step]
                                    + self.gamma
                                    * value_normalizer.denormalize(head_values[step + 1])
                                    * self.masks[step + 1]
                                    - value_normalizer.denormalize(head_values[step])
                            )
                            gae = (
                                    delta
                                    + self.gamma * self.gae_lambda
                                    * self.masks[step + 1]
                                    * gae
                            )
                            head_returns[step] = (
                                    gae + value_normalizer.denormalize(head_values[step])
                            )
                        else:
                            delta = (
                                    head_rewards[step]
                                    + self.gamma
                                    * head_values[step + 1]
                                    * self.masks[step + 1]
                                    - head_values[step]
                            )
                            gae = (
                                    delta
                                    + self.gamma * self.gae_lambda
                                    * self.masks[step + 1]
                                    * gae
                            )
                            head_returns[step] = gae + head_values[step]
                else:
                    head_returns[-1] = head_next_value
                    for step in reversed(range(head_rewards.shape[0])):
                        head_returns[step] = (
                                head_returns[step + 1]
                                * self.gamma
                                * self.masks[step + 1]
                                + head_rewards[step]
                        )

    def compute_advantages(self, value_normalizer=None):
        """Compute advantages for each head separately.
        Args:
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        Returns:
            advantages: (np.ndarray) advantages for each head [episode_length, n_rollout_threads, critic_head_num]
        """
        advantages = np.zeros_like(self.reward_components)

        for head_idx in range(self.critic_head_num):
            head_values = self.value_preds[:-1, :, head_idx:head_idx + 1]
            head_returns = self.returns[:-1, :, head_idx:head_idx + 1]

            if value_normalizer is not None:
                head_advantages = head_returns - value_normalizer.denormalize(head_values)
            else:
                head_advantages = head_returns - head_values

            advantages[:, :, head_idx] = head_advantages[:, :, 0]

        return advantages

    def feed_forward_generator_critic(
        self, critic_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for critic that uses MLP network."""
        episode_length, n_rollout_threads = self.reward_components.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) = {n_rollout_threads * episode_length} "
                f"is required to be greater than or equal to the number of critic mini batches ({critic_num_mini_batch})."
            )
            mini_batch_size = batch_size // critic_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size: (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (episode_length * n_rollout_threads, *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )
        value_preds = self.value_preds[:-1].reshape(-1, self.critic_head_num)
        returns = self.returns[:-1].reshape(-1, self.critic_head_num)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def naive_recurrent_generator_critic(self, critic_num_mini_batch):
        """Training data generator for critic that uses RNN network."""
        n_rollout_threads = self.reward_components.shape[1]
        assert n_rollout_threads >= critic_num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({critic_num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // critic_num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(critic_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id: start_id + num_envs_per_batch]
            share_obs_batch = _flatten(T, N, self.share_obs[:-1, ids])
            value_preds_batch = _flatten(T, N, self.value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            rnn_states_critic_batch = self.rnn_states_critic[0, ids]

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network."""
        episode_length, n_rollout_threads = self.reward_components.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size: (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (n_rollout_threads, episode_length, *share_obs_shape) --> (n_rollout_threads * episode_length, *share_obs_shape)
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        value_preds = _sa_cast(self.value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind: ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind: ind + data_chunk_length])
                return_batch.append(returns[ind: ind + data_chunk_length])
                masks_batch.append(masks[ind: ind + data_chunk_length])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_critic_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)

            yield share_obs_batch, rnn_states_critic_batch, value_preds_batch, return_batch, masks_batch
