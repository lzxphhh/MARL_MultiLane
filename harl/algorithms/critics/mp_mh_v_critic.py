"""Multi-Head V Critic for Single Scene."""
import torch
import torch.nn as nn
import numpy as np
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.multihead_v_net import MHVNet


class MHVCritic:
    """Multi-Head V Critic for Single Scene.
    Critic that learns multiple V-functions for different reward components.
    Each instance handles data for one specific scene class.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.args = args  # yaml里model和algo的config打包作为args进入VCritic
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        self.clip_param = args["clip_param"]  # PPO的clip参数

        # TODO: PPO相关 在下面的update函数中用到对应
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]  # The maximum value for the gradient clipping
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.critic_lr = args["critic_lr"]  # critic的学习率
        self.opti_eps = args["opti_eps"]  # critic Adam优化器的eps
        self.weight_decay = args["weight_decay"]  # critic Adam优化器的weight_decay

        self.share_obs_space = cent_obs_space  # 共享观测空间/全局状态空间 eg. Box(-inf, inf, (54,), float32)

        # 多头critic相关参数
        self.critic_head_num = args["critic_head_num"]
        self.critic_head_names = args["critic_head_names"]
        self.num_agents = args["num_CAVs"]

        # 动态权重相关参数
        self.weight_update_frequency = args.get("weight_update_frequency", 20)  # 每20个episode更新一次权重
        self.weight_ema_decay = args.get("weight_ema_decay", 0.85)  # EMA衰减系数
        self.epsilon_stability = args.get("epsilon_stability", 0.1)  # 稳定性参数ε

        # 初始化动态权重 - 安全权重固定为1，其他权重平均分配
        self.dynamic_weights = np.ones(self.critic_head_num)
        safety_idx = None
        other_indices = []
        for idx, name in self.critic_head_names.items():
            if name == "safety":
                safety_idx = idx
                self.dynamic_weights[idx] = 1.0  # 安全权重固定为1
            else:
                other_indices.append(idx)

        # 其他权重平均分配且和为1
        if len(other_indices) > 0:
            for idx in other_indices:
                self.dynamic_weights[idx] = 1.0 / len(other_indices)

        # 奖励统计信息 - 用于计算动态权重
        self.reward_statistics = {
            'means': np.zeros(self.critic_head_num),
            'variances': np.zeros(self.critic_head_num),
            'sample_count': 0,
            'reward_history': []  # 存储最近的奖励样本
        }

        # 历史奖励缓存大小
        self.max_history_size = args.get("max_reward_history", 500)

        # 当前episode计数器 - 用于权重更新频率控制
        self.episode_count = 0

        # 初始化多头critic网络，输入为单个agent的共享观测空间，输出为K维分数R+rnn_state
        self.critic = MHVNet(args, self.share_obs_space, self.device)

        # 初始化critic网络的优化器
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions. (n_threads, critic_head_num)
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, rnn_states_critic

    def update_reward_statistics(self, reward_components):
        """Update reward statistics for dynamic weight computation.
        Args:
            reward_components: (np.ndarray) shape [batch_size, critic_head_num] or [critic_head_num]
        """
        # 确保输入是二维的
        if len(reward_components.shape) == 1:
            reward_components = reward_components.reshape(1, -1)

        # 批量添加到历史记录
        for i in range(len(reward_components)):
            reward_sample = reward_components[i]  # [critic_head_num]
            self.reward_statistics['reward_history'].append(reward_sample.copy())

        # 限制历史记录大小
        while len(self.reward_statistics['reward_history']) > self.max_history_size:
            self.reward_statistics['reward_history'].pop(0)

        # 更新样本计数
        self.reward_statistics['sample_count'] += len(reward_components)

    def compute_dynamic_weights(self):
        """Compute dynamic weights based on current reward statistics.
        Safety weight is fixed at 1.0, efficiency/stability/comfort weights are dynamic and sum to 1.0.

        Returns:
            weights: (np.ndarray) shape [critic_head_num] - safety=1.0, others sum to 1.0
        """
        # 初始化权重数组
        weights = np.zeros(self.critic_head_num)

        # 找到各个头的索引
        safety_idx = None
        other_indices = []
        for idx, name in self.critic_head_names.items():
            if name == "safety":
                safety_idx = idx
            else:
                other_indices.append(idx)

        # 安全权重固定为1
        if safety_idx is not None:
            weights[safety_idx] = 1.0

        # 如果没有足够的样本，其他权重平均分配
        if len(self.reward_statistics['reward_history']) < 10:
            if len(other_indices) > 0:
                for idx in other_indices:
                    weights[idx] = 1.0 / len(other_indices)
            return weights

        # 计算当前的均值和方差
        history = np.array(self.reward_statistics['reward_history'])  # [history_size, critic_head_num]
        current_means = np.mean(history, axis=0)  # [critic_head_num]
        current_variances = np.var(history, axis=0)  # [critic_head_num]

        # 使用EMA更新统计信息
        if self.reward_statistics['sample_count'] > 10:
            self.reward_statistics['means'] = (self.weight_ema_decay * self.reward_statistics['means'] +
                                               (1 - self.weight_ema_decay) * current_means)
            self.reward_statistics['variances'] = (self.weight_ema_decay * self.reward_statistics['variances'] +
                                                   (1 - self.weight_ema_decay) * current_variances)
        else:
            self.reward_statistics['means'] = current_means
            self.reward_statistics['variances'] = current_variances

        # 对非安全组件计算动态权重
        if len(other_indices) > 0:
            means = self.reward_statistics['means'][other_indices]
            variances = self.reward_statistics['variances'][other_indices]

            # 为了数值稳定性，对均值进行归一化到[0,1]
            if np.max(means) > np.min(means):
                normalized_means = (means - np.min(means)) / (np.max(means) - np.min(means))
            else:
                normalized_means = np.ones_like(means) * 0.5

            # 计算权重分子: m_k + ε*s_k²
            weight_numerators = normalized_means + self.epsilon_stability * variances

            # 计算权重分母: Σ(m_k + ε*s_k²)
            weight_denominator = np.sum(weight_numerators)

            # 避免除零
            if weight_denominator < 1e-8:
                # 平均分配
                for idx in other_indices:
                    weights[idx] = 1.0 / len(other_indices)
            else:
                # 计算非安全权重并归一化使其和为1
                other_weights = weight_numerators / weight_denominator
                for i, idx in enumerate(other_indices):
                    weights[idx] = other_weights[i]

        return weights

    def update_dynamic_weights(self):
        """Update dynamic weights based on current statistics."""
        new_weights = self.compute_dynamic_weights()
        self.dynamic_weights = new_weights
        return new_weights

    def get_dynamic_weights(self):
        """Get current dynamic weights."""
        return self.dynamic_weights.copy()

    def should_update_weights(self):
        """Check if weights should be updated based on episode frequency."""
        return self.episode_count % self.weight_update_frequency == 0

    def episode_done(self):
        """Call this when an episode is completed to update episode counter."""
        self.episode_count += 1
        if self.should_update_weights():
            self.update_dynamic_weights()

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch=None, value_normalizer=None):
        """Calculate value function loss for each head separately.
        Args:
            values: (torch.Tensor) value function predictions for each head.
            value_preds_batch: (torch.Tensor) "old" value predictions from data batch.
            return_batch: (torch.Tensor) reward to go returns.
            active_masks_batch: (torch.Tensor) masks indicating whether an agent is active.
            value_normalizer: (ValueNorm) normalize the rewards.
        Returns:
            value_loss: (torch.Tensor) value function loss for each head.
        """
        head_losses = []

        # 对每个头单独计算损失
        for head_idx in range(self.critic_head_num):
            head_values = values[:, head_idx]
            head_value_preds = value_preds_batch[:, head_idx]
            head_returns = return_batch[:, head_idx]

            # 确保维度正确
            head_values = head_values.unsqueeze(1)  # [batch_size, 1]
            head_value_preds = head_value_preds.unsqueeze(1)  # [batch_size, 1]
            head_returns = head_returns.unsqueeze(1)  # [batch_size, 1]

            # PPO clip loss计算
            value_pred_clipped = head_value_preds + (head_values - head_value_preds).clamp(
                -self.clip_param, self.clip_param
            )

            if value_normalizer is not None:
                value_normalizer.update(head_returns)
                error_clipped = value_normalizer.normalize(head_returns) - value_pred_clipped
                error_original = value_normalizer.normalize(head_returns) - head_values
            else:
                error_clipped = head_returns - value_pred_clipped
                error_original = head_returns - head_values

            if self.use_huber_loss:
                value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
                value_loss_original = huber_loss(error_original, self.huber_delta)
            else:
                value_loss_clipped = mse_loss(error_clipped)
                value_loss_original = mse_loss(error_original)

            if self.use_clipped_value_loss:
                head_value_loss = torch.max(value_loss_original, value_loss_clipped)
            else:
                head_value_loss = value_loss_original

            # 应用active masks
            if active_masks_batch is not None and self.use_policy_active_masks:
                active_sum = active_masks_batch.sum()
                if active_sum > 0:
                    head_value_loss = (head_value_loss * active_masks_batch).sum() / active_sum
                else:
                    head_value_loss = torch.zeros_like(head_value_loss).mean()
            else:
                head_value_loss = head_value_loss.mean()

            head_losses.append(head_value_loss)

        # 返回每个头的损失
        value_loss = torch.stack(head_losses)
        return value_loss

    def update(self, sample, value_normalizer=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        # 检查数据类型
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _ = self.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch
        )

        # 计算每个头的损失
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch,
            active_masks_batch=None,
            value_normalizer=value_normalizer
        )

        # 简单求和所有头的损失（权重在策略梯度中应用）
        value_loss_sum = torch.sum(value_loss)

        # 优化器步骤
        self.critic_optimizer.zero_grad()
        (value_loss_sum * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyMHCriticBufferEP) buffer containing training data.
            value_normalizer: (ValueNorm) normalize the rewards.
        Returns:
            train_info: (dict) contains information regarding training update.
        """

        train_info = {}
        train_info["value_loss"] = [0 for _ in range(self.critic_head_num)]
        train_info["critic_grad_norm"] = 0

        # 进行critic训练
        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(
                    self.critic_num_mini_batch
                )
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch
                )
            # sample出
            for sample in data_generator:
                try:
                    value_loss, critic_grad_norm = self.update(
                        sample, value_normalizer=value_normalizer
                    )

                    for i in range(self.critic_head_num):
                        train_info["value_loss"][i] += value_loss[i].item()

                    train_info["critic_grad_norm"] += critic_grad_norm

                except Exception as e:
                    print(f"Error during critic update: {e}")
                    continue

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k in train_info.keys():
            if k == "value_loss":
                for i in range(self.critic_head_num):
                    train_info["value_loss"][i] /= num_updates
            else:
                train_info[k] /= num_updates

        # 添加每个头的单独损失到训练信息
        for i in range(self.critic_head_num):
            head_key = f"value_loss_head_{i}"
            train_info[head_key] = train_info["value_loss"][i]

            # 如果有名称映射，也添加带名称的键
            if i in self.critic_head_names:
                name_key = f"value_loss_{self.critic_head_names[i]}"
                train_info[name_key] = train_info["value_loss"][i]

        # 计算平均损失
        train_info["value_loss_avg"] = sum(train_info["value_loss"]) / self.critic_head_num

        # 添加动态权重信息cc
        train_info["dynamic_weights"] = self.dynamic_weights.tolist()
        train_info["reward_sample_count"] = self.reward_statistics['sample_count']

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()

    def get_statistics_info(self):
        """Get current statistics information for logging."""
        info = {
            'sample_count': self.reward_statistics['sample_count'],
            'reward_means': self.reward_statistics['means'].tolist(),
            'reward_variances': self.reward_statistics['variances'].tolist(),
            'dynamic_weights': self.dynamic_weights.tolist(),
            'episode_count': self.episode_count
        }
        return info