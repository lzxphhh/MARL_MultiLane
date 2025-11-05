import torch
import torch.nn as nn
import torch.nn.functional as F

class CAVCentricMotionEncoder(nn.Module):
    """
    针对单车道智能网联汽车场景的运动编码器
    提取以CAV为中心的局部交通运动特征
    """

    def __init__(self,
                 num_sur_veh=6,
                 num_state=4,  # 状态维度：位置、速度、加速度、航向角
                 num_steps=10,  # 历史时间步数
                 hidden_dim=64,  # 隐藏层维度
                 feature_dim=128,  # 输出特征维度
                 use_relative_states=True):  # 是否使用相对状态
        super().__init__()

        self.num_sur_veh = num_sur_veh
        self.num_state = num_state
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.use_relative_states = use_relative_states

        # CAV自身运动编码器
        self.cav_encoder = self._build_temporal_encoder(num_state, hidden_dim)

        # 周围车辆运动编码器（共享权重）
        self.surround_encoder = self._build_temporal_encoder(num_state, hidden_dim)

        # 位置感知注意力机制（区分前方和后方车辆）
        self.position_attention = PositionAwareAttention(num_sur_veh, hidden_dim, num_heads=4)

        # 相对运动特征提取器
        if use_relative_states:
            self.relative_encoder = RelativeMotionEncoder(num_sur_veh, num_state, hidden_dim)

        # 特征融合网络
        fusion_input_dim = hidden_dim  # CAV features
        fusion_input_dim += hidden_dim  # Aggregated surrounding features
        if use_relative_states:
            fusion_input_dim += hidden_dim  # Relative motion features

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def _build_temporal_encoder(self, input_dim, output_dim):
        """构建时序编码器"""
        return nn.Sequential(
            # 1D CNN for temporal pattern extraction
            nn.Conv1d(input_dim, output_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(),
            nn.Conv1d(output_dim // 2, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global temporal pooling
        )

    def forward(self, cav_hist_info, surround_hist_info):
        """
        Args:
            cav_hist_info: [num_envs, 1, num_steps*num_state] CAV历史轨迹
            surround_hist_info: [num_envs, 6, num_steps*num_state] 周围车辆历史轨迹

        Returns:
            motion_features: [num_envs, feature_dim] 运动特征
        """
        num_envs = cav_hist_info.shape[0]

        # 1. 处理CAV自身运动特征
        cav_hist_motion = cav_hist_info[:, :, 2:]  # 去除前两个标识位
        cav_features = self._encode_single_vehicle(cav_hist_motion.squeeze(1))  # [num_envs, hidden_dim]

        # 2. 批量处理周围车辆运动特征（优化点1：批量处理替代循环）
        surround_hist_motion = surround_hist_info[:, :, 2:]  # [num_envs, 6, num_steps*num_state]
        surround_features = self._encode_batch_vehicles(surround_hist_motion)  # [num_envs, 6, hidden_dim]

        # 3. 位置感知注意力聚合周围车辆特征
        aggregated_surround = self.position_attention(
            surround_features,
            cav_features.unsqueeze(1)
        )  # [num_envs, hidden_dim]

        # 4. 准备特征融合
        features_to_fuse = [cav_features, aggregated_surround]

        # 5. 提取相对运动特征（如果启用）
        if self.use_relative_states:
            relative_features = self.relative_encoder(cav_hist_motion, surround_hist_motion)
            features_to_fuse.append(relative_features)

        # 6. 特征融合
        fused_features = torch.cat(features_to_fuse, dim=-1)  # [num_envs, total_dim]
        motion_features = self.feature_fusion(fused_features)  # [num_envs, feature_dim]

        return motion_features

    def _encode_single_vehicle(self, hist_info):
        """编码单个车辆的历史轨迹"""
        # hist_info: [num_envs, num_steps*num_state]
        num_envs = hist_info.shape[0]

        # Reshape to [num_envs, num_steps, num_state]
        hist_reshaped = hist_info.view(num_envs, self.num_steps, self.num_state)

        # Transpose for conv1d: [num_envs, num_state, num_steps]
        hist_transposed = hist_reshaped.transpose(1, 2)

        # 修复：使用正确的编码器
        encoded = self.cav_encoder(hist_transposed)  # [num_envs, hidden_dim, 1]

        return encoded.squeeze(-1)  # [num_envs, hidden_dim]

    def _encode_batch_vehicles(self, surround_hist_motion):
        """批量编码周围车辆的历史轨迹（优化点1）"""
        # surround_hist_motion: [num_envs, num_sur_veh, num_steps*num_state]
        num_envs, num_sur_veh = surround_hist_motion.shape[:2]

        # 重塑为批量处理格式: [num_envs*num_sur_veh, num_steps*num_state]
        batch_hist = surround_hist_motion.reshape(num_envs * num_sur_veh, -1)

        # 批量编码
        batch_features = self._encode_single_vehicle(batch_hist)  # [num_envs*num_sur_veh, hidden_dim]

        # 重塑回原始格式: [num_envs, num_sur_veh, hidden_dim]
        surround_features = batch_features.view(num_envs, num_sur_veh, self.hidden_dim)

        return surround_features


class PositionAwareAttention(nn.Module):
    """位置感知注意力机制，区分前方和后方车辆的重要性（梯度友好版本）"""

    def __init__(self, num_sur_veh, hidden_dim, num_heads=4):
        super().__init__()
        self.num_sur_veh = num_sur_veh
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 位置编码：前方3辆车 (0,1,2) 和后方3辆车 (3,4,5)
        self.position_embedding = nn.Embedding(num_sur_veh, hidden_dim)

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 位置权重（前方车辆通常更重要）
        if num_sur_veh == 6:
            position_weights = torch.tensor([
                1.0, 0.8, 0.6,  # 前方车辆权重递减
                0.5, 0.4, 0.3  # 后方车辆权重更小
            ])
        elif num_sur_veh == 4:
            position_weights = torch.tensor([
                1.0, 0.6,  # 前方车辆权重递减
                0.8, 0.3,  # 后方车辆权重更小
            ])

        # 修复：注册为buffer而不是parameter，避免梯度计算问题
        self.register_buffer('position_weights', position_weights)

        # 预注册位置ID，避免重复创建
        position_ids = torch.arange(num_sur_veh)
        self.register_buffer('position_ids', position_ids)

    def forward(self, surround_features, cav_features):
        """
        Args:
            surround_features: [num_envs, 6, hidden_dim]
            cav_features: [num_envs, 1, hidden_dim]

        Returns:
            aggregated_features: [num_envs, hidden_dim]
        """
        num_envs = surround_features.shape[0]
        device = surround_features.device

        # 修复：动态计算位置编码，避免缓存问题
        position_emb = self.position_embedding(self.position_ids.to(device))  # [num_sur_veh, hidden_dim]
        position_emb = position_emb.unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, num_sur_veh, hidden_dim]

        # 特征 + 位置编码
        enhanced_features = surround_features + position_emb

        # 使用CAV特征作为query，周围车辆作为key和value
        attn_output, attn_weights = self.multihead_attn(
            query=cav_features,  # [num_envs, 1, hidden_dim]
            key=enhanced_features,  # [num_envs, num_sur_veh, hidden_dim]
            value=enhanced_features  # [num_envs, num_sur_veh, hidden_dim]
        )

        # 修复：动态计算位置权重，避免缓存问题
        position_weights = self.position_weights.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, num_sur_veh]
        weighted_attn = attn_weights * position_weights  # [num_envs, 1, num_sur_veh]
        weighted_attn = F.softmax(weighted_attn, dim=-1)

        # 加权聚合
        aggregated = torch.bmm(weighted_attn, surround_features)  # [num_envs, 1, hidden_dim]

        return aggregated.squeeze(1)  # [num_envs, hidden_dim]


class RelativeMotionEncoder(nn.Module):
    """相对运动特征编码器（梯度友好版本）"""

    def __init__(self, num_sur_veh, num_state, hidden_dim):
        super().__init__()
        self.num_sur_veh = num_sur_veh
        self.num_state = num_state
        self.hidden_dim = hidden_dim

        # 相对状态计算网络
        self.relative_state_net = nn.Sequential(
            nn.Linear(num_state * 2, hidden_dim),  # CAV + surrounding vehicle states
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 时序聚合网络
        self.temporal_aggregator = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 车辆间聚合网络
        self.vehicle_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * num_sur_veh, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, cav_hist_info, surround_hist_info):
        """
        计算CAV与周围车辆的相对运动特征（梯度友好版本）
        """
        num_envs = cav_hist_info.shape[0]
        num_steps = cav_hist_info.shape[-1] // self.num_state

        # Reshape历史信息
        cav_hist = cav_hist_info.view(num_envs, 1, num_steps, self.num_state)
        surround_hist = surround_hist_info.view(num_envs, self.num_sur_veh, num_steps, self.num_state)

        # 修复：使用更安全的广播操作
        # 广播CAV状态到所有周围车辆: [num_envs, num_sur_veh, num_steps, num_state]
        cav_hist_expanded = cav_hist.repeat(1, self.num_sur_veh, 1, 1)

        # 拼接CAV和周围车辆状态: [num_envs, num_sur_veh, num_steps, num_state*2]
        combined_states = torch.cat([cav_hist_expanded, surround_hist], dim=-1)

        # 重塑为批量处理: [num_envs*num_sur_veh*num_steps, num_state*2]
        batch_combined = combined_states.view(-1, self.num_state * 2)

        # 批量计算相对特征: [num_envs*num_sur_veh*num_steps, hidden_dim//2]
        batch_relative = self.relative_state_net(batch_combined)

        # 重塑回时序格式: [num_envs*num_sur_veh, hidden_dim//2, num_steps]
        relative_temporal = batch_relative.view(num_envs * self.num_sur_veh, self.hidden_dim // 2, num_steps)

        # 批量时序聚合: [num_envs*num_sur_veh, hidden_dim]
        aggregated_batch = self.temporal_aggregator(relative_temporal).squeeze(-1)

        # 重塑为车辆维度: [num_envs, num_sur_veh, hidden_dim]
        relative_features = aggregated_batch.view(num_envs, self.num_sur_veh, self.hidden_dim)

        # 展平并最终聚合: [num_envs, hidden_dim*num_sur_veh]
        flattened_features = relative_features.view(num_envs, -1)

        # 最终聚合: [num_envs, hidden_dim]
        final_relative = self.vehicle_aggregator(flattened_features)

        return final_relative