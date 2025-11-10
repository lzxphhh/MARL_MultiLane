#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构因果模型 V2 - 完整实现三路径架构和五种融合策略
SCM Model V2 with Three-Path Architecture and Five Fusion Strategies

完整实现论文规格：
1. 三条因果路径: Necessity, Feasibility, Safety
2. 四种路径编码架构: Linear, Shallow, Medium, Deep
3. 五种融合策略: Minimum, Product, WeightedSum, Gated, Hierarchical
4. 环境层因果机制
5. 类型调制机制

基于论文: LCIntention/design_info/SCM_Intention_Prediction_Implementation.md

作者：交通流研究团队
日期：2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from enum import Enum


class PathArchitecture(Enum):
    """路径编码架构类型"""
    LINEAR = "linear"
    SHALLOW = "shallow"  # 推荐基线
    MEDIUM = "medium"
    DEEP = "deep"


class FusionStrategy(Enum):
    """融合策略类型"""
    MINIMUM = "minimum"
    PRODUCT = "product"
    WEIGHTED_SUM = "weighted_sum"
    GATED = "gated"
    HIERARCHICAL = "hierarchical"  # 推荐基线


class PathEncoder(nn.Module):
    """
    因果路径编码器
    实现四种架构: Linear, Shallow, Medium, Deep
    """

    def __init__(self, input_dim: int, architecture: PathArchitecture, dropout_p: float = 0.1):
        super(PathEncoder, self).__init__()
        self.input_dim = input_dim
        self.architecture = architecture

        if architecture == PathArchitecture.LINEAR:
            # 架构A: Linear - R^4 → R
            self.encoder = nn.Linear(input_dim, 1)

        elif architecture == PathArchitecture.SHALLOW:
            # 架构B: Shallow (推荐基线) - R^4 → R^8 → R
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 1)
            )

        elif architecture == PathArchitecture.MEDIUM:
            # 架构C: Medium - R^4 → R^12 → R^4 → R
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 12),
                nn.Tanh(),
                nn.Linear(12, 4),
                nn.Tanh(),
                nn.Linear(4, 1)
            )

        elif architecture == PathArchitecture.DEEP:
            # 架构D: Deep - R^4 → R^16 → R^8 → R^4 → R (with Dropout)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch, input_dim] 输入特征

        Returns:
            logit: [batch, 1] 输出logit值
        """
        return self.encoder(x)


class ThreePathSCM(nn.Module):
    """
    三路径结构因果模型
    实现论文第1.3.2节的个体层因果机制
    """

    def __init__(self,
                 path_architecture: PathArchitecture = PathArchitecture.SHALLOW,
                 fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
                 dropout_p: float = 0.1):
        super(ThreePathSCM, self).__init__()

        self.path_architecture = path_architecture
        self.fusion_strategy = fusion_strategy

        # 三条因果路径，每条输入维度为4
        self.necessity_path = PathEncoder(4, path_architecture, dropout_p)  # φ_N
        self.feasibility_path = PathEncoder(4, path_architecture, dropout_p)  # φ_F
        self.safety_path = PathEncoder(4, path_architecture, dropout_p)  # φ_S

        # 融合策略参数
        if fusion_strategy == FusionStrategy.MINIMUM:
            # Smooth-Min参数 α
            self.alpha = nn.Parameter(torch.tensor(10.0), requires_grad=True)

        elif fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            # Softmax归一化权重参数 θ
            self.theta = nn.Parameter(torch.randn(3))

        elif fusion_strategy == FusionStrategy.GATED:
            # 门控融合参数 β (归一化权重) + b (门控偏置)
            self.beta = nn.Parameter(torch.randn(3))
            self.gate_bias = nn.Parameter(torch.tensor(0.0))

        elif fusion_strategy == FusionStrategy.HIERARCHICAL:
            # 层次化融合参数
            self.theta_w = nn.Parameter(torch.randn(1) * 0.1)  # 必要性权重参数
            self.theta_S = nn.Parameter(torch.randn(1) * 0.1 - 0.5)  # 安全阈值
            self.beta_gate = nn.Parameter(torch.tensor(5.0))  # 门控锐度

    def extract_path_features(self, ind_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从个体特征中提取三条路径的输入

        个体特征(10维): [v_ego, v_rel, d_headway, T_headway,
                       Δv_adj_front, d_adj_front, T_adj_front,
                       Δv_adj_rear, d_adj_rear, T_adj_rear]

        Args:
            ind_features: [batch, 10] 个体特征

        Returns:
            φ_N: [batch, 4] 必要性路径输入
            φ_F: [batch, 4] 可行性路径输入
            φ_S: [batch, 4] 安全性路径输入
        """
        # 路径1: 必要性路径 (与本车道前车的交互)
        phi_N = ind_features[:, :4]  # [v_ego, v_rel, d_headway, T_headway]

        # 路径2: 可行性路径 (与目标车道前车的关系)
        phi_F = torch.cat([
            ind_features[:, 0:1],  # v_ego
            ind_features[:, 4:7]   # [Δv_adj_front, d_adj_front, T_adj_front]
        ], dim=1)

        # 路径3: 安全性路径 (与目标车道后车的关系)
        phi_S = torch.cat([
            ind_features[:, 0:1],  # v_ego
            ind_features[:, 7:10]  # [Δv_adj_rear, d_adj_rear, T_adj_rear]
        ], dim=1)

        return phi_N, phi_F, phi_S

    def forward(self, ind_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播 - 三路径融合

        Args:
            ind_features: [batch, 10] 个体特征

        Returns:
            Y_fused: [batch, 1] 融合后的logit值
            path_outputs: dict 包含三条路径的输出
        """
        # 提取三条路径的输入
        phi_N, phi_F, phi_S = self.extract_path_features(ind_features)

        # 三条路径分别计算logit
        N = self.necessity_path(phi_N)  # [batch, 1]
        F = self.feasibility_path(phi_F)  # [batch, 1]
        S = self.safety_path(phi_S)  # [batch, 1]

        # 根据融合策略融合
        Y_fused = self.fuse_paths(N, F, S)

        path_outputs = {
            'N': N,
            'F': F,
            'S': S,
            'Y_fused': Y_fused
        }

        return Y_fused, path_outputs

    def fuse_paths(self, N: torch.Tensor, F: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        融合三条路径的输出

        Args:
            N: [batch, 1] 必要性logit
            F: [batch, 1] 可行性logit
            S: [batch, 1] 安全性logit

        Returns:
            Y: [batch, 1] 融合后的logit
        """
        if self.fusion_strategy == FusionStrategy.MINIMUM:
            # 策略1: 最小值融合 (Smooth-Min)
            # Y_min = -(1/α) log(e^(-αN) + e^(-αF) + e^(-αS))
            Y = -(1.0 / self.alpha) * torch.logsumexp(
                torch.cat([-self.alpha * N, -self.alpha * F, -self.alpha * S], dim=1),
                dim=1, keepdim=True
            )

        elif self.fusion_strategy == FusionStrategy.PRODUCT:
            # 策略2: 乘积融合
            # P_N = σ(N), P_F = σ(F), P_S = σ(S)
            # Y_product = log(P_N · P_F · P_S)
            log_P_N = -torch.nn.functional.softplus(-N)
            log_P_F = -torch.nn.functional.softplus(-F)
            log_P_S = -torch.nn.functional.softplus(-S)
            Y = log_P_N + log_P_F + log_P_S

        elif self.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            # 策略3: 加权求和融合
            # w_i = e^θ_i / Σe^θ_j
            # Y = w_1·N + w_2·F + w_3·S
            weights = torch.softmax(self.theta, dim=0)
            Y = weights[0] * N + weights[1] * F + weights[2] * S

        elif self.fusion_strategy == FusionStrategy.GATED:
            # 策略4: 门控融合
            # α_i = softmax(β), W = Σα_i·Path_i, g = σ(W+b), Y = g·W
            alpha = torch.softmax(self.beta, dim=0)
            W = alpha[0] * N + alpha[1] * F + alpha[2] * S
            g = torch.sigmoid(W + self.gate_bias)
            Y = g * W

        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            # 策略5: 层次化融合 (推荐基线)
            # g_S = σ(β·(S - θ_S))  (安全门控)
            # w = σ(θ_w)  (必要性权重)
            # M = w·N + (1-w)·F  (动机组合)
            # Y = g_S · M  (层次化输出)
            g_S = torch.sigmoid(self.beta_gate * (S - self.theta_S))
            w = torch.sigmoid(self.theta_w)
            M = w * N + (1 - w) * F
            Y = g_S * M
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return Y


class HierarchicalSCMModelV2(nn.Module):
    """
    完整的分层结构因果模型 V2
    包含环境层 + 三路径个体层 + 类型调制
    """

    def __init__(self,
                 env_dim: int = 4,
                 ind_dim: int = 10,
                 path_architecture: PathArchitecture = PathArchitecture.SHALLOW,
                 fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
                 use_vehicle_type: bool = True,
                 sigma_hdv: float = 0.3,
                 sigma_cav: float = 0.05,
                 epsilon_min_hdv: float = 0.02,
                 epsilon_min_cav: float = 0.01,
                 enforce_causal_constraints: bool = True):
        """
        初始化完整SCM模型V2

        Args:
            env_dim: 环境特征维度
            ind_dim: 个体特征维度 (必须是10: 三路径各需4维输入)
            path_architecture: 路径编码架构
            fusion_strategy: 路径融合策略
            use_vehicle_type: 是否使用车辆类型调制
            sigma_hdv: HDV决策噪声标准差
            sigma_cav: CAV决策噪声标准差
            epsilon_min_hdv: HDV基础噪声概率
            epsilon_min_cav: CAV基础噪声概率
            enforce_causal_constraints: 是否强制因果约束
        """
        super(HierarchicalSCMModelV2, self).__init__()

        self.env_dim = env_dim
        self.ind_dim = ind_dim
        self.use_vehicle_type = use_vehicle_type
        self.enforce_causal_constraints = enforce_causal_constraints
        self.path_architecture = path_architecture
        self.fusion_strategy = fusion_strategy

        # 决策噪声参数
        self.sigma_hdv = sigma_hdv
        self.sigma_cav = sigma_cav
        self.epsilon_min_hdv = epsilon_min_hdv
        self.epsilon_min_cav = epsilon_min_cav

        # 环境层因果机制 (线性层)
        self.env_layer = nn.Linear(env_dim, 1, bias=True)

        # 个体层因果机制 (三路径)
        self.ind_layer = ThreePathSCM(path_architecture, fusion_strategy)

        # 类型调制机制
        if use_vehicle_type:
            self.w_env_base = nn.Parameter(torch.tensor(1.0))
            self.w_env_delta = nn.Parameter(torch.tensor(0.0))
            self.w_ind_base = nn.Parameter(torch.tensor(1.0))
            self.w_ind_delta = nn.Parameter(torch.tensor(0.0))

        # 门控参数 (用于最终融合)
        self.gamma = nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.delta = nn.Parameter(torch.tensor(0.4), requires_grad=False)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化参数并应用因果约束"""
        with torch.no_grad():
            if self.enforce_causal_constraints:
                # 环境层约束 (公式26)
                env_init = torch.randn(1, self.env_dim) * 0.1
                if self.env_dim >= 4:
                    env_init[0, 1] = -abs(env_init[0, 1])  # κ_avg ≤ 0
                    env_init[0, 2] = abs(env_init[0, 2])   # Δv ≥ 0
                    env_init[0, 3] = -abs(env_init[0, 3])  # Δκ ≤ 0

                self.env_layer.weight.data = env_init
                self.env_layer.bias.data.zero_()

    def apply_causal_constraints(self):
        """应用因果约束 (在训练时定期调用)"""
        if not self.enforce_causal_constraints:
            return

        with torch.no_grad():
            # 环境层约束
            if self.env_dim >= 4:
                self.env_layer.weight[0, 1] = -abs(self.env_layer.weight[0, 1])
                self.env_layer.weight[0, 2] = abs(self.env_layer.weight[0, 2])
                self.env_layer.weight[0, 3] = -abs(self.env_layer.weight[0, 3])

    def forward(self,
                env_features: torch.Tensor,
                ind_features: torch.Tensor,
                vehicle_type: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        完整前向传播

        Args:
            env_features: [batch, env_dim] 环境特征
            ind_features: [batch, ind_dim] 个体特征
            vehicle_type: [batch, 1] 车辆类型 (0=HDV, 1=CAV)

        Returns:
            P_final: [batch, 1] 最终变道概率
            intermediates: dict 中间结果
        """
        batch_size = env_features.size(0)

        # 1. 环境层因果机制
        psi_env = self.env_layer(env_features)  # [batch, 1]

        # 2. 个体层因果机制 (三路径融合)
        psi_ind, path_outputs = self.ind_layer(ind_features)  # [batch, 1]

        # 3. 类型调制权重
        if self.use_vehicle_type and vehicle_type is not None:
            w_env = self.w_env_base + vehicle_type * self.w_env_delta
            w_ind = self.w_ind_base + vehicle_type * self.w_ind_delta
        else:
            w_env = torch.ones_like(psi_env)
            w_ind = torch.ones_like(psi_ind)

        # 4. 构建完整因果机制
        g_intent = w_env * psi_env + w_ind * psi_ind

        # 5. 噪声参数
        if vehicle_type is not None:
            sigma = torch.where(vehicle_type == 0,
                              torch.tensor(self.sigma_hdv, device=vehicle_type.device),
                              torch.tensor(self.sigma_cav, device=vehicle_type.device))
            epsilon_min = torch.where(vehicle_type == 0,
                                    torch.tensor(self.epsilon_min_hdv, device=vehicle_type.device),
                                    torch.tensor(self.epsilon_min_cav, device=vehicle_type.device))
        else:
            sigma = torch.full((batch_size, 1), self.sigma_hdv, device=env_features.device)
            epsilon_min = torch.full((batch_size, 1), self.epsilon_min_hdv, device=env_features.device)

        # 6. 环境层概率
        P_env = torch.sigmoid(psi_env / sigma)

        # 7. 个体层门控
        gate = torch.sigmoid(self.gamma * (psi_ind - self.delta))

        # 8. 门控融合
        P_final = gate * P_env + (1 - gate) * epsilon_min

        # 收集中间结果
        intermediates = {
            'psi_env': psi_env,
            'psi_ind': psi_ind,
            'path_N': path_outputs['N'],
            'path_F': path_outputs['F'],
            'path_S': path_outputs['S'],
            'w_env': w_env,
            'w_ind': w_ind,
            'g_intent': g_intent,
            'gate': gate,
            'P_env': P_env,
            'sigma': sigma,
            'epsilon_min': epsilon_min
        }

        return P_final, intermediates

    def get_causal_parameters(self) -> Dict[str, torch.Tensor]:
        """获取因果参数"""
        params = {
            'alpha_env': self.env_layer.weight.data.clone(),
            'alpha_bias': self.env_layer.bias.data.clone()
        }

        if self.use_vehicle_type:
            params.update({
                'w_env_base': self.w_env_base.data.clone(),
                'w_env_delta': self.w_env_delta.data.clone(),
                'w_ind_base': self.w_ind_base.data.clone(),
                'w_ind_delta': self.w_ind_delta.data.clone()
            })

        return params


# SCM数据集类保持不变
class SCMDataset(torch.utils.data.Dataset):
    """SCM模型专用数据集"""

    def __init__(self, features_df, env_feature_cols, ind_feature_cols,
                 label_col='lane_change_label', vehicle_type_col=None):
        self.env_features = features_df[env_feature_cols].values.astype(np.float32)
        self.ind_features = features_df[ind_feature_cols].values.astype(np.float32)
        self.labels = features_df[label_col].values.astype(np.float32)

        if vehicle_type_col and vehicle_type_col in features_df.columns:
            self.vehicle_types = features_df[vehicle_type_col].values.astype(np.float32)
        else:
            self.vehicle_types = np.zeros(len(features_df), dtype=np.float32)

        self.env_dim = len(env_feature_cols)
        self.ind_dim = len(ind_feature_cols)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'env_features': torch.FloatTensor(self.env_features[idx]),
            'ind_features': torch.FloatTensor(self.ind_features[idx]),
            'label': torch.FloatTensor([self.labels[idx]]),
            'vehicle_type': torch.FloatTensor([self.vehicle_types[idx]])
        }
