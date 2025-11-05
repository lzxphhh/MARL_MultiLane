import torch
import torch.nn as nn
from harl.models.base.mlp import MLPBase
from harl.models.base.simple_layers import CrossAttention
from harl.models.base.multi_lane.interaction_encoder import MultiLaneMotionEncoder
from harl.models.base.multi_lane.motion_encoder import CAVCentricMotionEncoder

class MultiLaneEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(MultiLaneEncoder, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.para_env = args['n_rollout_threads']

        self.net_paras = args['actor_para']
        # 安全性阈值配置
        self.safety_threshold = args.get('lane_safety_threshold', 1.0)  # 默认阈值为2.0秒

        # 初始化网络模块 - 多车道运动编码器
        # self.multilane_motion_encoder = MultiLaneMotionEncoder(
        #     num_state=5,  # 位置、速度、加速度、航向角, TTC
        #     num_steps=10,  # 历史时间步数
        #     hidden_dim=64,
        #     feature_dim=128,  # 增加特征维度以处理更复杂的多车道场景
        #     use_relative_states=True,
        #     use_lane_aware=True  # 启用车道感知
        # )
        self.ego_lane_motion_encoder = CAVCentricMotionEncoder(
            num_sur_veh=6,
            num_state=6,
            num_steps=10,
            hidden_dim=64,
            feature_dim=64,
            use_relative_states=False
        )
        self.left_lane_motion_encoder = CAVCentricMotionEncoder(
            num_sur_veh=4,
            num_state=6,
            num_steps=10,
            hidden_dim=64,
            feature_dim=64,
            use_relative_states=False
        )
        self.right_lane_motion_encoder = CAVCentricMotionEncoder(
            num_sur_veh=4,
            num_state=6,
            num_steps=10,
            hidden_dim=64,
            feature_dim=64,
            use_relative_states=False
        )

        self.lane_fusion = EvenSimplerSafetyFusion(
            feature_dim=64,
            safety_threshold=self.safety_threshold
        )

        # # 特征维度对齐网络
        # self.motion_projection = nn.Linear(64*3, 64)  # motion encoder输出128维
        # self.current_projection = nn.Linear(64, 64)  # 将current state投影到128维

        # 交叉注意力机制 - 用于融合运动特征和当前状态特征
        self.cross_attention = CrossAttention(64, 8, 64, 0.1)

        # 多车道决策特征提取器
        self.multilane_decision_mlp = MLPBase(args, [64*2+3])  # 处理multilane_info

        # 当前状态特征提取器
        # road_info(2) + ego_cur_info(8) + surround_cur_info(14*7=98) + local_evaluation_info(6)
        self.current_state_mlp = MLPBase(args, [2 + 9 + 14 * 7 + 6])

        # 定义多车道观测信息模板结构
        self.multilane_obs_template = {
            'road_info': torch.zeros(2, dtype=torch.float32),  # 道路信息
            'ego_cur_info': torch.zeros(9, dtype=torch.float32),  # 自身当前状态
            'ego_hist_info': torch.zeros(1, 62, dtype=torch.float32),  # 自身历史状态
            # 'surround_cur_info': torch.zeros(14 * 7, dtype=torch.float32),  # 周围车辆当前状态
            'ego_lane_surround_cur_info': torch.zeros(6 * 7, dtype=torch.float32),
            'left_lane_surround_cur_info': torch.zeros(4 * 7, dtype=torch.float32),
            'right_lane_surround_cur_info': torch.zeros(4 * 7, dtype=torch.float32),
            # 'surround_hist_info': torch.zeros(14, 52, dtype=torch.float32),  # 周围车辆历史状态
            'ego_lane_surround_hist_info': torch.zeros(6, 62, dtype=torch.float32),
            'left_lane_surround_hist_info': torch.zeros(4, 62, dtype=torch.float32),
            'right_lane_surround_hist_info': torch.zeros(4, 62, dtype=torch.float32),
            'local_evaluation_info': torch.zeros(6, dtype=torch.float32),  # 局部评估指标
            'multilane_info': torch.zeros(3, dtype=torch.float32),  # 多车道决策信息
        }

    def reconstruct_multilane_obs(self, obs):
        """重构多车道观测信息"""
        reconstructed = self.reconstruct_obs_batch(obs, self.multilane_obs_template)
        return (reconstructed['road_info'],
                reconstructed['ego_cur_info'], reconstructed['ego_hist_info'],
                # reconstructed['surround_cur_info'],
                reconstructed['ego_lane_surround_cur_info'], reconstructed['left_lane_surround_cur_info'], reconstructed['right_lane_surround_cur_info'],
                # reconstructed['surround_hist_info'],
                reconstructed['ego_lane_surround_hist_info'], reconstructed['left_lane_surround_hist_info'], reconstructed['right_lane_surround_hist_info'],
                reconstructed['local_evaluation_info'],
                reconstructed['multilane_info'])

    def forward(self, obs, batch_size=20):
        """
        前向传播
        Args:
            obs: 观测数据 (n_rollout_thread, obs_dim)
            batch_size: 批次大小
        Returns:
            combined_embedding: 融合后的特征表示
        """
        # 重构观测信息
        multilane_info = self.reconstruct_multilane_obs(obs)
        road_info = multilane_info[0]
        ego_cur_info = multilane_info[1]
        ego_hist_info = multilane_info[2]
        # surround_cur_info = multilane_info[3]
        ego_lane_surround_cur_info = multilane_info[3]
        left_lane_surround_cur_info = multilane_info[4]
        right_lane_surround_cur_info = multilane_info[5]
        # surround_hist_info = multilane_info[4]
        ego_lane_surround_hist_info = multilane_info[6]
        left_lane_surround_hist_info = multilane_info[7]
        right_lane_surround_hist_info = multilane_info[8]
        local_evaluation_info = multilane_info[9]
        multilane_decision_info = multilane_info[10]

        # 1. 提取多车道运动交互特征
        # motion_features = self.multilane_motion_encoder(
        #     ego_hist_info,  # [batch_size, 1, hist_dim]
        #     surround_hist_info  # [batch_size, 14, hist_dim]
        # )

        ego_lane_motion_feature = self.ego_lane_motion_encoder(ego_hist_info, ego_lane_surround_hist_info)
        # left_lane_motion_feature = self.left_lane_motion_encoder(ego_hist_info, left_lane_surround_hist_info)
        # right_lane_motion_feature = self.right_lane_motion_encoder(ego_hist_info, right_lane_surround_hist_info)

        # 1.2 计算车道安全性掩码
        left_mask, right_mask = self._compute_lane_safety_masks(multilane_decision_info)
        # 批量处理左右车道特征提取
        left_lane_motion_feature = self._selective_lane_encoding(ego_hist_info, left_lane_surround_hist_info,
                                                                 self.left_lane_motion_encoder, left_mask)
        right_lane_motion_feature = self._selective_lane_encoding(ego_hist_info, right_lane_surround_hist_info,
                                                                  self.right_lane_motion_encoder, right_mask)

        # 简化的安全性引导融合
        motion_features, fusion_weights = self.lane_fusion(
            ego_lane_motion_feature,
            left_lane_motion_feature,
            right_lane_motion_feature,
            multilane_decision_info
        )

        # motion_features = torch.cat([ego_lane_motion_feature, left_lane_motion_feature, right_lane_motion_feature], dim=1)

        # 2. 提取当前状态特征
        current_state_features = torch.cat([
            road_info,
            ego_cur_info,
            # surround_cur_info,
            ego_lane_surround_cur_info, left_lane_surround_cur_info, right_lane_surround_cur_info,
            local_evaluation_info
        ], dim=1)
        current_state_features = self.current_state_mlp(current_state_features)

        # 4. 特征维度对齐
        # motion_features_aligned = self.motion_projection(motion_features)  # 确保128维
        # current_features_aligned = self.current_projection(current_state_features)  # 投影到128维

        # # 5. 使用交叉注意力融合运动特征和当前状态特征
        # motion_input = motion_features.unsqueeze(1)  # [batch_size, 1, motion_dim]
        # current_input = current_state_features.unsqueeze(1)  # [batch_size, 1, current_dim]

        # cross_attended_features = self.cross_attention(motion_input, current_input)
        # cross_attended_features = cross_attended_features.view(cross_attended_features.size(0), -1)

        # # 6. 最终特征融合
        # # 将交叉注意力特征与多车道决策特征拼接
        combined_feature = torch.cat([
            # cross_attended_features,
            motion_features,
            current_state_features,
            multilane_decision_info
        ], dim=1)

        combined_embedding = self.multilane_decision_mlp(combined_feature)

        return combined_embedding

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        """
        批量重构观测数据
        Args:
            obs_batch: 批量观测数据
            template_structure: 模板结构
        Returns:
            reconstructed_batch: 重构后的批量数据
        """
        device = obs_batch.device

        # 计算每个组件的大小并进行分割
        sizes = [tensor.numel() for tensor in template_structure.values()]
        split_tensors = torch.split(obs_batch, sizes, dim=1)

        # 重新构造观测数据
        reconstructed_batch = {
            key: split_tensor.view(obs_batch.size(0), *template_structure[key].shape).to(device)
            for key, split_tensor in zip(template_structure.keys(), split_tensors)
        }

        return reconstructed_batch

    def _compute_lane_safety_masks(self, multilane_decision_info):
        """
        计算车道安全性掩码
        Args:
            multilane_decision_info: [batch_size, 3] - [left_safetime, long_safetime, right_safetime]
        Returns:
            left_mask: [batch_size] - 左车道是否安全的布尔掩码
            right_mask: [batch_size] - 右车道是否安全的布尔掩码
        """
        # 提取各车道安全时间
        left_safetime = multilane_decision_info[:, 0]  # [batch_size]
        right_safetime = multilane_decision_info[:, 2]  # [batch_size]

        # 计算安全性掩码（安全时间大于阈值为True）
        left_mask = left_safetime > self.safety_threshold  # [batch_size]
        right_mask = right_safetime > self.safety_threshold  # [batch_size]

        return left_mask, right_mask

    def _selective_lane_encoding(self, ego_hist_info, lane_hist_info, lane_encoder, lane_mask):
        """
        选择性车道编码：仅对安全的车道进行特征提取
        Args:
            ego_hist_info: [batch_size, 1, hist_dim] 自车历史信息
            lane_hist_info: [batch_size, num_vehicles, hist_dim] 车道车辆历史信息
            lane_encoder: CAVCentricMotionEncoder 车道编码器
            lane_mask: [batch_size] 车道安全性掩码
        Returns:
            lane_features: [batch_size, feature_dim] 车道特征（不安全车道为零向量）
        """
        batch_size = ego_hist_info.shape[0]
        feature_dim = 64  # CAVCentricMotionEncoder的输出维度

        # 检查是否有任何环境需要计算该车道特征
        if not lane_mask.any():
            # 如果所有环境都不需要该车道特征，直接返回零向量
            return torch.zeros(batch_size, feature_dim, device=ego_hist_info.device)

        # 初始化输出特征
        lane_features = torch.zeros(batch_size, feature_dim, device=ego_hist_info.device)

        # 获取需要计算特征的环境索引
        valid_indices = torch.where(lane_mask)[0]

        if len(valid_indices) > 0:
            # 仅对安全的环境进行特征提取
            valid_ego_hist = ego_hist_info[valid_indices]  # [valid_batch, 1, hist_dim]
            valid_lane_hist = lane_hist_info[valid_indices]  # [valid_batch, num_vehicles, hist_dim]

            # 计算特征
            valid_features = lane_encoder(valid_ego_hist, valid_lane_hist)  # [valid_batch, feature_dim]

            # 将计算得到的特征放回对应位置
            lane_features[valid_indices] = valid_features

        return lane_features

    def _batch_lane_encoding(self, ego_hist_info, lane_hist_infos, lane_encoders, lane_masks):
        """
        批量处理多个车道的编码，进一步优化计算效率
        Args:
            ego_hist_info: [batch_size, 1, hist_dim]
            lane_hist_infos: list of [batch_size, num_vehicles, hist_dim]
            lane_encoders: list of CAVCentricMotionEncoder
            lane_masks: list of [batch_size] boolean masks
        Returns:
            lane_features: list of [batch_size, feature_dim]
        """
        batch_size = ego_hist_info.shape[0]
        feature_dim = 64

        lane_features = []

        for lane_hist_info, lane_encoder, lane_mask in zip(lane_hist_infos, lane_encoders, lane_masks):
            # 使用选择性编码
            features = self._selective_lane_encoding(ego_hist_info, lane_hist_info, lane_encoder, lane_mask)
            lane_features.append(features)

        return lane_features


class EvenSimplerSafetyFusion(nn.Module):
    """
    更加简化的版本：直接使用安全性掩码
    """

    def __init__(self, feature_dim=64, safety_threshold=2.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.safety_threshold = safety_threshold

    def forward(self, ego_feature, left_feature, right_feature, safety_times):
        """
        最简单的安全性引导融合：
        - 安全车道权重 = 1.0
        - 不安全车道权重 = 0.0
        - 本车道始终保持权重 = 1.0
        """
        batch_size = safety_times.shape[0]

        # 1. 计算安全性掩码
        safety_mask = (safety_times > self.safety_threshold).float()  # [batch_size, 3]
        safety_mask[:, 1] = 1.0  # 本车道始终安全

        # 2. 归一化权重
        weights = safety_mask / safety_mask.sum(dim=1, keepdim=True)

        # 3. 加权融合
        lane_features = torch.stack([left_feature, ego_feature, right_feature], dim=1)
        fused_features = (lane_features * weights.unsqueeze(-1)).sum(dim=1)

        return fused_features, weights

