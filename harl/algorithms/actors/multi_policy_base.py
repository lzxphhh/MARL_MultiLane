"""Base class for multi-policy on-policy algorithms."""

import torch
import numpy as np
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.models_tools import update_linear_schedule


class MultiPolicyBase:
    """Base class for multi-policy algorithms.
    Each instance handles data for one specific scene class.
    Supports dynamic weights from multi-head critic for policy gradient computation.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.  # yaml里面的model和algo的config打包作为args进入OnPolicyBase
            obs_space: (gym.spaces or list) observation space. # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.spaces) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        # "model" and "algo" sections in $Algorithm config file
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)   # dtype和device

        self.data_chunk_length = args["data_chunk_length"]  # TODO：这是什么 rnn相关
        self.use_recurrent_policy = args["use_recurrent_policy"]  # TODO：这两个的区别，基于rnn chunck是什么
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]  # TODO：这是什么
        self.action_aggregation = args["action_aggregation"]  # TODO：这是什么

        self.lr = args["lr"]  # actor学习率
        self.opti_eps = args["opti_eps"]  # optimizer的epsilon
        self.weight_decay = args["weight_decay"] # optimizer的权重衰减
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space

        # 多头critic相关参数
        self.critic_head_num = args["critic_head_num"]
        self.critic_head_names = args["critic_head_names"]
        self.num_reward_head = args["num_reward_head"]

        # 动态权重相关参数
        self.use_dynamic_weights = args.get("use_dynamic_weight", True)
        self.weight_ema_decay = args.get("weight_ema_decay", 0.85)

        # 当前场景的动态权重 - 由外部设置
        self.current_dynamic_weights = np.ones(self.critic_head_num) / self.critic_head_num

        # 建立actor网络结构
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.device)
        # 建立actor的优化器optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def set_dynamic_weights(self, weights):
        """Set current dynamic weights for this actor.
        Args:
            weights: (np.ndarray) dynamic weights from multi-head critic [critic_head_num]
        """
        if len(weights) != self.critic_head_num:
            raise ValueError(f"Weights length {len(weights)} must match critic_head_num {self.critic_head_num}")
        self.current_dynamic_weights = weights.copy()

    def get_dynamic_weights(self):
        """Get current dynamic weights."""
        return self.current_dynamic_weights.copy()

    def get_actions(
            self, obs, rnn_states_actor, reward_weights, masks, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs. 可以对应OnPolicyActorBuffer中在某一个step下的信息
        输入:
            obs: (np.ndarray) local agent inputs to the actor. 所有环境下当前时刻某个agent的obs 【thread_num, obs_dim】
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
                                上一时刻的rnn_state 【thread_num, rnn层数，rnn_state_dim】
            masks: (np.ndarray) denotes points at which RNN states should be reset. 【thread_num, 1】
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 当前智能体的可用动作 (if None, all actions available) 【thread_num, act_dim】
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
                                有没有available_actions
        输出:
            actions: (torch.Tensor) actions for the given inputs. 【thread_num, 1】
            action_log_probs: (torch.Tensor) log probabilities of actions. 【thread_num, 1】
            rnn_states_actor: (torch.Tensor) updated RNN states for actor. 【thread_num, rnn层数，rnn_state_dim】
        """
        actions, action_log_probs, rnn_states_actor, action_loss, reward_weights = self.actor(
            obs, rnn_states_actor, reward_weights, masks, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states_actor, action_loss, reward_weights

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
        self, obs, rnn_states_actor, reward_weights, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, action_loss, reward_weights = self.actor(
            obs, rnn_states_actor, reward_weights, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

    def compute_dynamic_policy_gradient(self, advantages_list):
        """Compute dynamically weighted policy gradients.
        Args:
            advantages_list: (list) list of advantages for each reward component [critic_head_num, ...]
        Returns:
            weighted_advantages: (np.ndarray) dynamically weighted advantages
        """
        if not self.use_dynamic_weights:
            # 如果不使用动态权重，简单平均
            return np.mean(advantages_list, axis=0)

        # 应用动态权重
        weighted_advantages = np.zeros_like(advantages_list[0])

        for head_idx in range(self.critic_head_num):
            if head_idx < len(advantages_list):
                weight = self.current_dynamic_weights[head_idx]
                weighted_advantages += weight * advantages_list[head_idx]

        return weighted_advantages

    def apply_dynamic_weights_to_loss(self, losses_list):
        """Apply dynamic weights to policy losses.
        Args:
            losses_list: (list) list of policy losses for each reward component
        Returns:
            weighted_loss: (torch.Tensor) dynamically weighted loss
        """
        if not self.use_dynamic_weights or len(losses_list) == 1:
            return torch.mean(torch.stack(losses_list))

        weighted_loss = torch.tensor(0.0, device=self.device)
        total_weight = 0.0

        for head_idx in range(min(len(losses_list), self.critic_head_num)):
            weight = self.current_dynamic_weights[head_idx]
            weighted_loss += weight * losses_list[head_idx]
            total_weight += weight

        # 归一化
        if total_weight > 0:
            weighted_loss = weighted_loss / total_weight

        return weighted_loss

    def update_reward_weights_ema(self, new_weights):
        """Update reward weights using EMA.
        Args:
            new_weights: (np.ndarray) new reward weights
        Returns:
            updated_weights: (np.ndarray) EMA updated weights
        """
        if self.use_dynamic_weights:
            # 使用EMA平滑权重更新
            self.current_dynamic_weights = (
                self.weight_ema_decay * self.current_dynamic_weights +
                (1 - self.weight_ema_decay) * new_weights
            )
        else:
            self.current_dynamic_weights = new_weights.copy()

        return self.current_dynamic_weights

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        """
        pass

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        # 测试actor网络结构
        self.actor.eval()

    def get_policy_info(self):
        """Get current policy information for logging."""
        info = {
            'dynamic_weights': self.current_dynamic_weights.tolist(),
            'use_dynamic_weights': self.use_dynamic_weights,
            'critic_head_num': self.critic_head_num,
            'critic_head_names': self.critic_head_names
        }
        return info

    def save_policy_weights(self, filepath):
        """Save actor policy weights.
        Args:
            filepath: (str) path to save the weights
        """
        torch.save(self.actor.state_dict(), filepath)

    def load_policy_weights(self, filepath):
        """Load actor policy weights.
        Args:
            filepath: (str) path to load the weights from
        """
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))

    def get_policy_parameters(self):
        """Get policy parameters for analysis.
        Returns:
            dict: parameter statistics
        """
        total_params = sum(p.numel() for p in self.actor.parameters())
        trainable_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }