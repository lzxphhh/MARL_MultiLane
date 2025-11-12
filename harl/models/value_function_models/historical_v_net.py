"""
Historical VNet for Multi-Lane MARL
根据harl/envs/00_structure.tex设计实现的Critic网络

Critic网络设计:
1. 使用MLP对每个车的历史轨迹信息进行处理
2. 使用图注意力网络对所有车辆的特征(历史轨迹特征+当前状态)进行融合得到车辆交互特征表示
3. 对交通信息进行MLP处理得到整体交通特征表示
4. 使用交叉注意力机制融合车辆交互特征表示与整体交通特征表示
5. 最终通过MLP处理输出状态价值V(s)

Author: Traffic Flow Research Team
Date: 2025-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import init, get_init_method


class HistoricalVNet(nn.Module):
    """
    Historical Value Network for Multi-Lane MARL

    符合00_structure.tex设计的Critic网络:
    - 历史轨迹编码
    - 图注意力网络聚合车辆交互
    - 交叉注意力融合交通与车辆特征
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """
        Initialize Historical VNet

        Args:
            args: (dict) arguments containing model configuration
            cent_obs_space: (gym.Space) centralized observation space
            device: (torch.device) device to run on (cpu/gpu)
        """
        super(HistoricalVNet, self).__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 获取初始化方法
        init_method = get_init_method(self.initialization_method)

        # Multi-lane specific parameters
        self.num_vehicles = args.get("num_vehicles", 15)  # CAV + 14 周围车辆
        self.num_lanes = args.get("num_lanes", 3)
        self.hist_steps = args.get("hist_steps", 20)
        self.state_dim = args.get("state_dim", 6)  # x, y, v, a, theta, type
        self.traffic_dim = args.get("traffic_dim", 6)  # lane statistics

        # 特征维度
        hidden_dim = self.hidden_sizes[-1]
        self.vehicle_feature_dim = hidden_dim
        self.traffic_feature_dim = hidden_dim // 2

        # 1. 历史轨迹编码器 (MLP对每个车的历史轨迹进行处理)
        self.trajectory_encoder = TrajectoryEncoder(
            hist_steps=self.hist_steps,
            state_dim=self.state_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=self.vehicle_feature_dim
        )

        # 2. 图注意力网络 (GAT聚合所有车辆特征)
        self.vehicle_gat = VehicleGraphAttentionNetwork(
            input_dim=self.vehicle_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=4,
            num_vehicles=self.num_vehicles
        )

        # 3. 交通特征处理 (MLP处理交通信息)
        self.traffic_encoder = nn.Sequential(
            nn.Linear(self.traffic_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, self.traffic_feature_dim),
            nn.ReLU()
        )

        # 4. 交叉注意力机制 (融合车辆交互特征与交通特征)
        self.cross_attention = CrossAttentionFusion(
            vehicle_dim=hidden_dim,
            traffic_dim=self.traffic_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4
        )

        # 5. 最终MLP输出状态价值
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            init_(nn.Linear(hidden_dim, 1))
        )

        self.to(device)

    def forward(self, cent_obs, rnn_states=None, masks=None):
        """
        Forward pass of Historical VNet

        Args:
            cent_obs: (torch.Tensor) centralized observation
                Expected structure: [batch, features]
                Features包含:
                    - 所有车辆的历史轨迹: [num_vehicles, hist_steps, state_dim]
                    - 交通信息: [traffic_dim]
            rnn_states: (torch.Tensor) not used in this implementation
            masks: (torch.Tensor) not used in this implementation

        Returns:
            values: (torch.Tensor) value function predictions [batch, 1]
            rnn_states: (torch.Tensor) dummy return for compatibility
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        batch_size = cent_obs.shape[0]

        # 解析centralized observation
        # 假设cent_obs的结构: [batch, num_vehicles * hist_steps * state_dim + traffic_dim]
        vehicle_hist_size = self.num_vehicles * self.hist_steps * self.state_dim

        # 提取车辆历史轨迹和交通信息
        vehicle_hist_flat = cent_obs[:, :vehicle_hist_size]
        traffic_info = cent_obs[:, vehicle_hist_size:vehicle_hist_size + self.traffic_dim]

        # Reshape车辆历史轨迹: [batch, num_vehicles, hist_steps * state_dim]
        vehicle_hist = vehicle_hist_flat.view(
            batch_size, self.num_vehicles, self.hist_steps * self.state_dim
        )

        # 1. 编码每个车辆的历史轨迹
        # [batch, num_vehicles, vehicle_feature_dim]
        vehicle_features = self.trajectory_encoder(vehicle_hist)

        # 2. 使用GAT聚合车辆交互特征
        # [batch, hidden_dim]
        aggregated_vehicle_features = self.vehicle_gat(vehicle_features)

        # 3. 编码交通信息
        # [batch, traffic_feature_dim]
        traffic_features = self.traffic_encoder(traffic_info)

        # 4. 交叉注意力融合
        # [batch, hidden_dim]
        fused_features = self.cross_attention(
            aggregated_vehicle_features, traffic_features
        )

        # 5. 输出状态价值
        # [batch, 1]
        values = self.value_head(fused_features)

        # 为兼容性返回dummy rnn_states
        if rnn_states is None:
            rnn_states = torch.zeros(batch_size, 1).to(**self.tpdv)

        return values, rnn_states


class TrajectoryEncoder(nn.Module):
    """
    历史轨迹编码器
    使用MLP对每个车的历史轨迹信息进行处理
    """

    def __init__(self, hist_steps, state_dim, hidden_dim, output_dim):
        super().__init__()

        input_dim = hist_steps * state_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, vehicle_hist):
        """
        Args:
            vehicle_hist: [batch, num_vehicles, hist_steps * state_dim]
        Returns:
            features: [batch, num_vehicles, output_dim]
        """
        batch_size, num_vehicles, _ = vehicle_hist.shape

        # Reshape for batch processing
        hist_flat = vehicle_hist.view(-1, vehicle_hist.shape[-1])

        # Encode
        features = self.encoder(hist_flat)

        # Reshape back
        features = features.view(batch_size, num_vehicles, -1)

        return features


class VehicleGraphAttentionNetwork(nn.Module):
    """
    图注意力网络
    对所有车辆的特征(历史轨迹特征+当前状态)进行融合得到车辆交互特征表示
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_vehicles):
        super().__init__()

        self.num_heads = num_heads
        self.num_vehicles = num_vehicles
        self.head_dim = hidden_dim // num_heads

        # Multi-head attention
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # Aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(output_dim * num_vehicles, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, vehicle_features):
        """
        Args:
            vehicle_features: [batch, num_vehicles, input_dim]
        Returns:
            aggregated_features: [batch, output_dim]
        """
        batch_size = vehicle_features.shape[0]

        # Compute Q, K, V
        Q = self.query(vehicle_features)  # [batch, num_vehicles, hidden_dim]
        K = self.key(vehicle_features)
        V = self.value(vehicle_features)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_vehicles, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_vehicles, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_vehicles, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # [batch, num_heads, num_vehicles, head_dim]
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, num_vehicles, head_dim]

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, self.num_vehicles, -1)

        # Output projection
        vehicle_features_updated = self.out_proj(attn_output)  # [batch, num_vehicles, output_dim]

        # Aggregate all vehicles
        vehicle_features_flat = vehicle_features_updated.view(batch_size, -1)
        aggregated = self.aggregation(vehicle_features_flat)

        return aggregated


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力机制
    融合车辆交互特征表示与整体交通特征表示
    """

    def __init__(self, vehicle_dim, traffic_dim, hidden_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project traffic features to match vehicle dimension
        self.traffic_proj = nn.Linear(traffic_dim, vehicle_dim)

        # Cross attention: vehicle as query, traffic as key/value
        self.query = nn.Linear(vehicle_dim, hidden_dim)
        self.key = nn.Linear(vehicle_dim, hidden_dim)
        self.value = nn.Linear(vehicle_dim, hidden_dim)

        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + vehicle_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vehicle_features, traffic_features):
        """
        Args:
            vehicle_features: [batch, vehicle_dim]
            traffic_features: [batch, traffic_dim]
        Returns:
            fused_features: [batch, hidden_dim]
        """
        batch_size = vehicle_features.shape[0]

        # Project traffic features
        traffic_proj = self.traffic_proj(traffic_features)  # [batch, vehicle_dim]

        # Compute Q from vehicle, K/V from traffic
        Q = self.query(vehicle_features).view(
            batch_size, self.num_heads, self.head_dim
        )  # [batch, num_heads, head_dim]

        K = self.key(traffic_proj).view(
            batch_size, self.num_heads, self.head_dim
        )

        V = self.value(traffic_proj).view(
            batch_size, self.num_heads, self.head_dim
        )

        # Attention scores
        scores = torch.sum(Q * K, dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=1)

        # Apply attention
        attn_output = attn_weights * V  # [batch, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, -1)  # [batch, hidden_dim]

        # Fuse with original vehicle features
        fused = torch.cat([attn_output, vehicle_features], dim=-1)
        fused_features = self.fusion(fused)

        return fused_features
