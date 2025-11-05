import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLaneMotionEncoder(nn.Module):
    """
    多车道场景下的运动编码器
    处理CAV与14辆周围车辆的交互特征
    """

    def __init__(self,
                 num_state=4,
                 num_steps=10,
                 hidden_dim=64,
                 feature_dim=128,
                 use_relative_states=True,
                 use_lane_aware=True):
        super().__init__()

        self.num_state = num_state
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.use_relative_states = use_relative_states
        self.use_lane_aware = use_lane_aware

        # CAV自身运动编码器
        self.cav_encoder = self._build_cnn_temporal_encoder(num_state, hidden_dim)

        # 周围车辆运动编码器
        self.surround_encoder = self._build_cnn_temporal_encoder(num_state, hidden_dim)

        # 多车道位置感知注意力机制
        self.multilane_attention = MultiLanePositionAwareAttention(
            hidden_dim,
            num_heads=8,
            num_vehicles=14
        )

        # 相对运动特征提取器
        if use_relative_states:
            self.relative_encoder = MultiLaneRelativeMotionEncoder(num_state, hidden_dim)

        # 车道感知特征提取器
        if use_lane_aware:
            self.lane_aware_encoder = LaneAwareFeatureEncoder(hidden_dim)

        # 特征融合网络
        fusion_input_dim = hidden_dim  # CAV features
        fusion_input_dim += hidden_dim  # Aggregated surrounding features
        if use_relative_states:
            fusion_input_dim += hidden_dim  # Relative motion features
        if use_lane_aware:
            fusion_input_dim += hidden_dim  # Lane-aware features

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def _build_cnn_temporal_encoder(self, input_dim, output_dim):
        """构建时序编码器"""
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(),
            nn.Conv1d(output_dim // 2, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, cav_hist_info, surround_hist_info):
        """
        Args:
            cav_hist_info: [batch_size, 1, hist_dim] CAV历史轨迹
            surround_hist_info: [batch_size, 14, hist_dim] 周围车辆历史轨迹
        Returns:
            motion_features: [batch_size, feature_dim] 运动特征
        """
        batch_size = cav_hist_info.shape[0]

        # 1. 处理CAV自身运动特征
        cav_hist_motion = cav_hist_info[:, :, 2:]  # 去除前两个标识位
        cav_features = self._encode_single_vehicle(cav_hist_motion.squeeze(1))

        # 2. 处理14辆周围车辆运动特征
        surround_features = []
        surround_hist_motion = []

        for i in range(14):
            vehicle_hist = surround_hist_info[:, i, 2:]  # 去除前两个标识位
            vehicle_feat = self._encode_single_vehicle(vehicle_hist)
            surround_features.append(vehicle_feat)
            surround_hist_motion.append(vehicle_hist)

        surround_features = torch.stack(surround_features, dim=1)  # [batch_size, 14, hidden_dim]
        surround_hist_motion = torch.stack(surround_hist_motion, dim=1)

        # 3. 多车道位置感知注意力聚合
        aggregated_surround = self.multilane_attention(
            surround_features,
            cav_features.unsqueeze(1)
        )

        # 4. 准备特征融合
        features_to_fuse = [cav_features, aggregated_surround]

        # 5. 相对运动特征
        if self.use_relative_states:
            relative_features = self.relative_encoder(
                cav_hist_motion.squeeze(1),
                surround_hist_motion
            )
            features_to_fuse.append(relative_features)

        # 6. 车道感知特征
        if self.use_lane_aware:
            lane_aware_features = self.lane_aware_encoder(surround_features)
            features_to_fuse.append(lane_aware_features)

        # 7. 特征融合
        fused_features = torch.cat(features_to_fuse, dim=-1)
        motion_features = self.feature_fusion(fused_features)

        return motion_features

    def _encode_single_vehicle(self, hist_info):
        """编码单个车辆的历史轨迹"""
        batch_size = hist_info.shape[0]

        # Reshape to [batch_size, num_steps, num_state]
        hist_reshaped = hist_info.view(batch_size, self.num_steps, self.num_state)

        # Transpose for conv1d: [batch_size, num_state, num_steps]
        hist_transposed = hist_reshaped.transpose(1, 2)

        # Temporal encoding
        encoded = self.surround_encoder(hist_transposed)

        return encoded.squeeze(-1)


class MultiLanePositionAwareAttention(nn.Module):
    """多车道位置感知注意力机制"""

    def __init__(self, hidden_dim, num_heads=8, num_vehicles=14):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_vehicles = num_vehicles

        # 车辆位置编码：
        # 0-2: 前方车道前方3辆车, 3-5: 前方车道后方3辆车
        # 6-7: 左侧车道前方2辆车, 8-9: 左侧车道后方2辆车
        # 10-11: 右侧车道前方2辆车, 12-13: 右侧车道后方2辆车
        self.position_embedding = nn.Embedding(num_vehicles, hidden_dim)

        # 车道类型编码
        self.lane_type_embedding = nn.Embedding(3, hidden_dim)  # 当前车道、左车道、右车道

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 多车道位置权重
        self.position_weights = nn.Parameter(torch.tensor([
            # 前方车道：前方车辆
            1.0, 0.8, 0.6,
            # 前方车道：后方车辆
            0.5, 0.4, 0.3,
            # 左侧车道：前方车辆
            0.7, 0.5,
            # 左侧车道：后方车辆
            0.4, 0.3,
            # 右侧车道：前方车辆
            0.7, 0.5,
            # 右侧车道：后方车辆
            0.4, 0.3
        ]))

        # 车道权重
        self.lane_weights = nn.Parameter(torch.tensor([
            1.0,  # 当前车道权重最高
            0.6,  # 左侧车道
            0.6  # 右侧车道
        ]))

    def forward(self, surround_features, cav_features):
        """
        Args:
            surround_features: [batch_size, 14, hidden_dim]
            cav_features: [batch_size, 1, hidden_dim]
        Returns:
            aggregated_features: [batch_size, hidden_dim]
        """
        batch_size = surround_features.shape[0]

        # 位置编码
        position_ids = torch.arange(self.num_vehicles, device=surround_features.device)
        position_emb = self.position_embedding(position_ids)
        position_emb = position_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # 车道类型编码
        lane_type_ids = torch.tensor([
            0, 0, 0, 0, 0, 0,  # 当前车道车辆 (前方3+后方3)
            1, 1, 1, 1,  # 左侧车道车辆 (前方2+后方2)
            2, 2, 2, 2  # 右侧车道车辆 (前方2+后方2)
        ], device=surround_features.device)
        lane_type_emb = self.lane_type_embedding(lane_type_ids)
        lane_type_emb = lane_type_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # 增强特征：位置编码 + 车道编码
        enhanced_features = surround_features + position_emb + lane_type_emb

        # 多头注意力
        attn_output, attn_weights = self.multihead_attn(
            query=cav_features,
            key=enhanced_features,
            value=enhanced_features
        )

        # 应用位置权重和车道权重
        position_weights = self.position_weights.unsqueeze(0).unsqueeze(0)

        # 计算车道权重
        lane_weights_expanded = torch.cat([
            self.lane_weights[0].repeat(6),  # 当前车道
            self.lane_weights[1].repeat(4),  # 左侧车道
            self.lane_weights[2].repeat(4)  # 右侧车道
        ]).unsqueeze(0).unsqueeze(0)

        # 综合权重
        combined_weights = position_weights * lane_weights_expanded
        weighted_attn = attn_weights * combined_weights
        weighted_attn = torch.softmax(weighted_attn, dim=-1)

        # 加权聚合
        aggregated = torch.bmm(weighted_attn, surround_features)

        return aggregated.squeeze(1)


class MultiLaneRelativeMotionEncoder(nn.Module):
    """多车道相对运动特征编码器"""

    def __init__(self, num_state, hidden_dim):
        super().__init__()
        self.num_state = num_state

        # 相对状态计算网络
        self.relative_state_net = nn.Sequential(
            nn.Linear(num_state * 2, hidden_dim),
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

        # 多车道车辆聚合网络
        self.vehicle_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 14, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, cav_hist_info, surround_hist_info):
        """计算CAV与14辆周围车辆的相对运动特征"""
        batch_size = cav_hist_info.shape[0]
        num_steps = cav_hist_info.shape[-1] // self.num_state

        # Reshape历史信息
        cav_hist = cav_hist_info.view(batch_size, 1, num_steps, self.num_state)
        surround_hist = surround_hist_info.view(batch_size, 14, num_steps, self.num_state)

        relative_features = []

        for i in range(14):  # 对每辆周围车辆
            # 获取CAV和第i辆车的状态序列
            cav_states = cav_hist.squeeze(1)
            vehicle_states = surround_hist[:, i, :, :]

            # 计算相对状态
            combined_states = torch.cat([cav_states, vehicle_states], dim=-1)

            # 通过网络提取相对特征
            relative_feat = []
            for t in range(num_steps):
                step_feat = self.relative_state_net(combined_states[:, t, :])
                relative_feat.append(step_feat)

            relative_feat = torch.stack(relative_feat, dim=-1)

            # 时序聚合
            aggregated = self.temporal_aggregator(relative_feat).squeeze(-1)
            relative_features.append(aggregated)

        # 所有车辆的相对特征
        all_relative = torch.cat(relative_features, dim=-1)

        # 最终聚合
        final_relative = self.vehicle_aggregator(all_relative)

        return final_relative


class LaneAwareFeatureEncoder(nn.Module):
    """车道感知特征编码器"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 车道内特征聚合
        self.current_lane_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),  # 当前车道6辆车
            nn.ReLU()
        )

        self.left_lane_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim // 2),  # 左侧车道4辆车
            nn.ReLU()
        )

        self.right_lane_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim // 2),  # 右侧车道4辆车
            nn.ReLU()
        )

        # 车道间交互特征提取 - 修复维度问题
        # 左右车道各输出 hidden_dim//2，拼接后为 hidden_dim
        self.lane_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 修改输入维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, surround_features):
        """
        Args:
            surround_features: [batch_size, 14, hidden_dim]
        Returns:
            lane_aware_features: [batch_size, hidden_dim]
        """
        # 分离不同车道的特征
        current_lane_features = surround_features[:, :6, :]  # 当前车道
        left_lane_features = surround_features[:, 6:10, :]  # 左侧车道
        right_lane_features = surround_features[:, 10:14, :]  # 右侧车道

        # 车道内聚合
        current_aggregated = self.current_lane_aggregator(
            current_lane_features.view(current_lane_features.size(0), -1)
        )
        left_aggregated = self.left_lane_aggregator(
            left_lane_features.view(left_lane_features.size(0), -1)
        )
        right_aggregated = self.right_lane_aggregator(
            right_lane_features.view(right_lane_features.size(0), -1)
        )

        # 相邻车道特征融合
        adjacent_lanes = torch.cat([left_aggregated, right_aggregated], dim=-1)

        # 车道间交互
        lane_interaction_features = self.lane_interaction(adjacent_lanes)

        # 最终车道感知特征
        lane_aware_features = current_aggregated + lane_interaction_features

        return lane_aware_features