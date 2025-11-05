import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
# from harl.models.base.MARL_CACC import CACC_rep
# from harl.models.base.simple_layers import MultiLayerMLP
# from harl.models.base.Reasoning_CACC import RCACC_rep
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space

from harl.models.base.multi_lane.improved_encoder import MultiLaneEncoder

class StochasticPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information. #yaml里面的model和algo的config打包
            obs_space: (gym.Space) observation space.  # 单个智能体的观测空间 eg: Box (18,)
            action_space: (gym.Space) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticPolicy, self).__init__()
        self.strategy = args['strategy']
        self.hidden_sizes = args["hidden_sizes"]  # MLP隐藏层神经元数量
        self.args = args  # yaml里面的model和algo的config打包
        self.gain = args["gain"]  # 激活函数的斜率或增益，增益较大的激活函数会更敏感地响应输入的小变化，而增益较小的激活函数则会对输入的小变化不那么敏感
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_policy_active_masks = args["use_policy_active_masks"]  # TODO：这是什么

        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]  # TODO：二者的区别是什么
        self.use_recurrent_policy = args["use_recurrent_policy"]
        # number of recurrent layers
        self.recurrent_n = args["recurrent_n"]  # RNN层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        obs_shape = get_shape_from_obs_space(obs_space)  # 获取观测空间的形状，tuple of integer. eg: （18，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.scene_name = args['scene_name']
        self.improved_encoder = MultiLaneEncoder(obs_shape[0], action_space.shape[0], self.hidden_sizes[-1], 'Continuous', args)
        # self.MARL_CACC = CACC_rep(obs_shape[0], action_space.shape[0], self.hidden_sizes[-1], 'Continuous', args)
        # self.improved_CACC = ICACC_rep(obs_shape[0], action_space.shape[0], self.hidden_sizes[-1], 'Continuous', args)
        # self.Reasoning_CACC = RCACC_rep(obs_shape[0], action_space.shape[0], self.hidden_sizes[-1], 'Continuous', args)
        # self.Reward_base = MultiLayerMLP(input_dim=obs_shape[0], output_dim=4, hidden_dims=[128, 256, 64], activation=nn.ReLU, dropout_rate=0.1)

        self.fixed_reward_weights = args["reward_weights"]
        # 设置动态权重生成的相关参数
        self.use_dynamic_weight = args["use_dynamic_weight"]
        self.use_weight_head = args["use_weight_head"]
        self.weight_ema_decay = args["weight_ema_decay"]   # 权重平滑系数
        self.reward_types = args["num_reward_head"]-1  # 奖励类型数量：安全、效率、稳定性、舒适性 -- 安全固定权重为1，其他项权重变化

        # 用于生成动态权重的网络层 - 输出3维权重
        # 可选激活函数: ReLU, Sigmoid, LeakyReLU, Tanh, SELU, Hardswish, Identity
        self.weight_head = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加稳定性
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, self.reward_types),  # 输出3维：效率、稳定性、舒适性
            nn.Softplus()  # 确保输出为正值
        )

        # 权重约束参数
        self.min_other_weight = 0.1  # 其他权重最小值
        self.max_other_weight = 0.7  # 其他权重最大值
        self.safety_weight_fixed = 1.0  # 固定安全权重

        # 存储上一次的权重，用于平滑
        self.register_buffer('prev_weights', torch.ones(1, self.reward_types) / self.reward_types)

        self.num_CAVs = args['num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids
        # 如果使用RNN，初始化RNN层
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 初始化ACT层, 用于输出动作(动作概率)，输入大小hidden_sizes[-1]，输出大小action_space.n
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,  # yaml里面的model和algo的config打包
        )

        self.to(device)
        self.example_extend_info = {
            'road_structure': torch.zeros(8),  # 0
            'next_node': torch.zeros(3),  # 1
            'self_stats': torch.zeros(1, 10),  # 2
            'surround_stats': torch.zeros(2, 6),  # 3
            'ego_lane_stats': torch.zeros(1, 6),  # 4
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['road_structure'], reconstructed['next_node'], reconstructed['self_stats'], \
            reconstructed['surround_stats'], reconstructed['ego_lane_stats']

    def generate_dynamic_weights(self, features, prev_weights):
        """
        根据提取的特征生成动态权重

        Args:
            features: Actor特征
            batch_size: 批次大小

        Returns:
            smoothed_weights: 平滑处理后的动态权重
        """
        # 通过权重生成头生成原始权重
        raw_weights = self.weight_head(features)

        # 归一化权重，确保总和为1
        normalized_weights = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 应用平滑更新 - 指数移动平均
        if torch.any(prev_weights != 0, dim=None).item():
            smoothed_weights = self.weight_ema_decay * prev_weights + (1 - self.weight_ema_decay) * normalized_weights
        else:
            smoothed_weights = normalized_weights

        return smoothed_weights

    def forward(
            self, obs, rnn_states, reward_weights, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        reward_weights = check(reward_weights).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        env_num = obs.size(0)
        # start = time.time()
        if self.strategy == 'base' or self.strategy == 'simple':
            actor_features = self.base(obs)
        elif self.strategy == 'improve':
            actor_features = self.improved_encoder(obs)
        # elif self.strategy == 'iMARL':
        #     # start = time.time()
        #     actor_features = self.MARL_CACC(obs, batch_size=obs.size(0))
            # actor_features = self.Reasoning_CACC(obs, batch_size=obs.size(0))
            # end = time.time()
            # print(f'Actor forward time: {end - start} second')

        # reward_weights = torch.tensor([1.0, 0.3, 0.3, 0.1], device=self.tpdv['device']).repeat(env_num, 1)
        # reward_weights = self.Reward_base(obs)

        # end = time.time()
        # print(f'forward time: {end - start} second')
        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_dynamic_weight:
            # 使用RNN输出的actor_features生成动态权重
            if isinstance(reward_weights, torch.Tensor):
                prew_weights = reward_weights[:, 1:].clone().detach().to(**self.tpdv)
            else:
                prew_weights = torch.from_numpy(reward_weights[:, 1:]).to(**self.tpdv)
            dynamic_reward_weights = self.generate_dynamic_weights(actor_features, prew_weights)
            safety_reward_weight = torch.ones(env_num,1).to(**self.tpdv)
            reward_weights = torch.cat([safety_reward_weight, dynamic_reward_weights], dim=1)
        else:
            # 使用固定权重，转换为正确的tensor格式
            if isinstance(self.fixed_reward_weights, dict):
                # 字典格式转数组
                weights_array = np.array([[
                    self.fixed_reward_weights['safety'],
                    self.fixed_reward_weights['efficiency'],
                    self.fixed_reward_weights['stability'],
                    self.fixed_reward_weights['comfort']
                ]])
            else:
                # 确保是 (1, 4) 形状
                weights_array = self.fixed_reward_weights.reshape(1, 4)

            # 转换为tensor并扩展到批次大小
            reward_weights = torch.from_numpy(weights_array).to(**self.tpdv).expand(env_num, 4)

        # 将特征和可用动作输入ACT层，得到动作，动作概率
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        # action_loss
        action_loss_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
        # last_actor_action = reconstruct_info[7]
        # last_actual_action = reconstruct_info[8]
        # action_mse_loss = torch.zeros(env_num, 1, device=self.tpdv['device'])
        # # action_cosine_loss = torch.zeros(env_num, 1, device=self.tpdv['device'])
        # action_mse_loss[:, 0] = torch.mean((last_actor_action[:, 0, 0] - last_actual_action[:, 0, 0]) ** 2)
        # # action_cosine_loss[:, 0] = 1 - torch.nn.functional.cosine_similarity(last_actor_action[:, 0, 0], last_actual_action[:, 0, 0], dim=-1).mean()
        # action_loss_output[:, 0] = action_mse_loss[:, 0]

        return actions, action_log_probs, rnn_states, action_loss_output, reward_weights

    def evaluate_actions(
            self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.strategy == 'base' or self.strategy == 'simple':
            actor_features = self.base(obs)
        elif self.strategy == 'improve':
            actor_features = self.improved_encoder(obs)
        # elif self.strategy == 'iMARL':
        #     actor_features = self.MARL_CACC(obs, batch_size=obs.size(0))
            # actor_features = self.Reasoning_CACC(obs, batch_size=obs.size(0))

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy, action_distribution

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device  # Get the device of obs_batch

        # Initialize the reconstructed_batch with the same structure as template_structure
        reconstructed_batch = {
            key: torch.empty((obs_batch.size(0),) + tensor.shape, device=device)
            for key, tensor in template_structure.items()
        }

        # Compute the cumulative sizes of each tensor in the template structure
        sizes = [tensor.numel() for tensor in template_structure.values()]
        cumulative_sizes = torch.cumsum(torch.tensor(sizes), dim=0)
        indices = [0] + cumulative_sizes.tolist()[:-1]

        # Split obs_batch into chunks based on the cumulative sizes
        split_tensors = torch.split(obs_batch, sizes, dim=1)

        # Assign the split tensors to the appropriate keys in the reconstructed_batch
        for key, split_tensor in zip(template_structure.keys(), split_tensors):
            reconstructed_batch[key] = split_tensor.view((-1,) + template_structure[key].shape)

        return reconstructed_batch
