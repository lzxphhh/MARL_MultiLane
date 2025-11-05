"""Multi-Head V Network for Single Scene."""
import torch
import torch.nn as nn
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method

class MHVNet(nn.Module):
    """Multi-Head V Network for Single Scene.
    Outputs multiple value function predictions for different reward components given global states.
    Each instance handles one specific scene class.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize MHVNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(MHVNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"] # MLP隐藏层神经元数量
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]  # RNN的层数
        self.tpdv = dict(dtype=torch.float32, device=device) # dtype和device
        self.device = device

        # 获取网络权重初始化方法函数
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space) # 获取观测空间的形状，tuple of integer. eg: （54，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小cent_obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        # 如果使用RNN，初始化RNN层
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 定义了一个初始化神经网络权重的函数 （特别是 nn.Linear 层的权重）
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 多头critic相关参数
        self.critic_head_num = args["critic_head_num"]
        self.critic_head_names = args["critic_head_names"]

        # 确保头数量和名称匹配
        assert self.critic_head_num == len(self.critic_head_names), \
            f"critic_head_num ({self.critic_head_num}) should match the length of critic_head_names ({len(self.critic_head_names)})"

        # 为每个奖励组件创建一个输出头
        # 每个头输出一个标量值函数
        self.v_heads = nn.ModuleList([
            init_(nn.Linear(self.hidden_sizes[-1], 1))
            for _ in range(self.critic_head_num)
        ])

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """Compute value function predictions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions for each head. Shape: [batch_size, critic_head_num]
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        critic_features = self.base(cent_obs)

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        batch_size = critic_features.size(0)

        # 预分配输出张量 [batch_size, critic_head_num]
        values = torch.zeros(batch_size, self.critic_head_num, device=self.device)

        # 为每个头计算值函数预测
        for head_idx, head in enumerate(self.v_heads):
            head_value = head(critic_features)  # [batch_size, 1]
            values[:, head_idx] = head_value.squeeze(1)  # [batch_size]

        return values, rnn_states

    def get_head_values(self, cent_obs, rnn_states, masks, head_indices=None):
        """Get values for specific heads.
        Args:
            cent_obs: observation inputs
            rnn_states: RNN states
            masks: mask tensor
            head_indices: (list, optional) list of head indices to compute. If None, compute all heads.
        Returns:
            values: (torch.Tensor) values for specified heads
            rnn_states: updated RNN states
        """
        # 检查输入
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 提取特征
        critic_features = self.base(cent_obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if head_indices is None:
            head_indices = list(range(self.critic_head_num))

        batch_size = critic_features.size(0)
        values = torch.zeros(batch_size, len(head_indices), device=self.device)

        for i, head_idx in enumerate(head_indices):
            if head_idx < self.critic_head_num:
                head_value = self.v_heads[head_idx](critic_features)
                values[:, i] = head_value.squeeze(1)

        return values, rnn_states

    def get_single_head_value(self, cent_obs, rnn_states, masks, head_idx):
        """Get value for a single specific head.
        Args:
            cent_obs: observation inputs
            rnn_states: RNN states
            masks: mask tensor
            head_idx: (int) index of the head to compute
        Returns:
            value: (torch.Tensor) value for the specified head [batch_size, 1]
            rnn_states: updated RNN states
        """
        if head_idx < 0 or head_idx >= self.critic_head_num:
            raise ValueError(f"head_idx {head_idx} is out of range [0, {self.critic_head_num})")

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        value = self.v_heads[head_idx](critic_features)

        return value, rnn_states

    def get_features(self, cent_obs, rnn_states, masks):
        """Extract features without computing head outputs.
        Args:
            cent_obs: observation inputs
            rnn_states: RNN states
            masks: mask tensor
        Returns:
            features: (torch.Tensor) extracted features
            rnn_states: updated RNN states
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        return critic_features, rnn_states

    def get_head_gradients(self):
        """Get gradients for each head.
        Returns:
            dict: gradients for each head
        """
        gradients = {}
        for head_idx, head in enumerate(self.v_heads):
            head_grads = {}
            for name, param in head.named_parameters():
                if param.grad is not None:
                    head_grads[name] = param.grad.clone()
                else:
                    head_grads[name] = None

            head_name = self.critic_head_names.get(head_idx, f"head_{head_idx}")
            gradients[head_name] = head_grads

        return gradients

    def freeze_heads(self, head_indices):
        """Freeze parameters for specific heads.
        Args:
            head_indices: (list) list of head indices to freeze
        """
        for head_idx in head_indices:
            if 0 <= head_idx < self.critic_head_num:
                for param in self.v_heads[head_idx].parameters():
                    param.requires_grad = False

    def unfreeze_heads(self, head_indices):
        """Unfreeze parameters for specific heads.
        Args:
            head_indices: (list) list of head indices to unfreeze
        """
        for head_idx in head_indices:
            if 0 <= head_idx < self.critic_head_num:
                for param in self.v_heads[head_idx].parameters():
                    param.requires_grad = True

    def reset_heads(self, head_indices):
        """Reset (reinitialize) parameters for specific heads.
        Args:
            head_indices: (list) list of head indices to reset
        """
        init_method = get_init_method(self.initialization_method)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        for head_idx in head_indices:
            if 0 <= head_idx < self.critic_head_num:
                self.v_heads[head_idx].apply(init_)

    def get_head_names_mapping(self):
        """Get mapping from head index to head name.
        Returns:
            dict: mapping from head_idx to head_name
        """
        return self.critic_head_names.copy()

    def get_head_parameters(self, head_idx):
        """Get parameters for a specific head.
        Args:
            head_idx: (int) head index
        Returns:
            parameters: iterator of parameters for the specified head
        """
        if 0 <= head_idx < self.critic_head_num:
            return self.v_heads[head_idx].parameters()
        else:
            raise ValueError(f"head_idx {head_idx} is out of range")

    def print_network_info(self):
        """Print network architecture information."""
        print(f"MHVNet Architecture Info:")
        print(f"  - Number of critic heads: {self.critic_head_num}")
        print(f"  - Head names: {self.critic_head_names}")
        print(f"  - Hidden sizes: {self.hidden_sizes}")
        print(f"  - Use RNN: {self.use_recurrent_policy or self.use_naive_recurrent_policy}")

        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        # 各部分参数数量
        base_params = sum(p.numel() for p in self.base.parameters())
        print(f"  - Base network parameters: {base_params:,}")

        if hasattr(self, 'rnn'):
            rnn_params = sum(p.numel() for p in self.rnn.parameters())
            print(f"  - RNN parameters: {rnn_params:,}")

        head_params = sum(p.numel() for p in self.v_heads.parameters())
        print(f"  - Multi-head parameters: {head_params:,}")

        # 每个头的参数数量
        single_head_params = sum(p.numel() for p in self.v_heads[0].parameters())
        print(f"  - Parameters per head: {single_head_params:,}")

    def save_head_weights(self, filepath, head_idx):
        """Save weights for a specific head.
        Args:
            filepath: (str) path to save the weights
            head_idx: (int) head index
        """
        if 0 <= head_idx < self.critic_head_num:
            torch.save(self.v_heads[head_idx].state_dict(), filepath)
        else:
            raise ValueError(f"Invalid head index: {head_idx}")

    def load_head_weights(self, filepath, head_idx):
        """Load weights for a specific head.
        Args:
            filepath: (str) path to load the weights from
            head_idx: (int) head index
        """
        if 0 <= head_idx < self.critic_head_num:
            self.v_heads[head_idx].load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            raise ValueError(f"Invalid head index: {head_idx}")

    def save_all_head_weights(self, filepath_prefix):
        """Save weights for all heads.
        Args:
            filepath_prefix: (str) prefix for the file paths
        """
        for head_idx in range(self.critic_head_num):
            head_name = self.critic_head_names.get(head_idx, f"head_{head_idx}")
            filepath = f"{filepath_prefix}_{head_name}.pt"
            self.save_head_weights(filepath, head_idx)

    def load_all_head_weights(self, filepath_prefix):
        """Load weights for all heads.
        Args:
            filepath_prefix: (str) prefix for the file paths
        """
        for head_idx in range(self.critic_head_num):
            head_name = self.critic_head_names.get(head_idx, f"head_{head_idx}")
            filepath = f"{filepath_prefix}_{head_name}.pt"
            try:
                self.load_head_weights(filepath, head_idx)
                print(f"Loaded weights for head {head_idx} ({head_name}) from {filepath}")
            except FileNotFoundError:
                print(f"Warning: Could not find weights file {filepath} for head {head_idx}")

    def get_output_shape(self):
        """Get the output shape of the network.
        Returns:
            tuple: (critic_head_num,) representing the output shape
        """
        return (self.critic_head_num,)

    def forward_with_head_weights(self, cent_obs, rnn_states, masks, head_weights):
        """Forward pass with weighted head outputs.
        Args:
            cent_obs: observation inputs
            rnn_states: RNN states
            masks: mask tensor
            head_weights: (torch.Tensor) weights for each head [critic_head_num]
        Returns:
            weighted_value: (torch.Tensor) weighted sum of head values [batch_size, 1]
            individual_values: (torch.Tensor) individual head values [batch_size, critic_head_num]
            rnn_states: updated RNN states
        """
        individual_values, rnn_states = self.forward(cent_obs, rnn_states, masks)

        # 确保权重在正确的设备上
        if not isinstance(head_weights, torch.Tensor):
            head_weights = torch.tensor(head_weights, device=self.device, dtype=torch.float32)
        else:
            head_weights = head_weights.to(self.device)

        # 计算加权值 [batch_size, 1]
        weighted_value = torch.sum(individual_values * head_weights.unsqueeze(0), dim=1, keepdim=True)

        return weighted_value, individual_values, rnn_states

    def compute_head_importance(self, cent_obs, rnn_states, masks):
        """Compute importance scores for each head based on output magnitude.
        Args:
            cent_obs: observation inputs
            rnn_states: RNN states
            masks: mask tensor
        Returns:
            importance_scores: (torch.Tensor) importance scores for each head
            values: (torch.Tensor) head values
            rnn_states: updated RNN states
        """
        values, rnn_states = self.forward(cent_obs, rnn_states, masks)

        # 计算每个头的重要性（基于输出的绝对值）
        importance_scores = torch.mean(torch.abs(values), dim=0)  # [critic_head_num]

        return importance_scores, values, rnn_states