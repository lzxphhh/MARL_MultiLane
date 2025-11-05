"""
Weight Computation Module for Multi-Objective RL
权重计算模块，实现基于局部特征的动态权重生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class DynamicWeightNetwork(nn.Module):
    """动态权重生成网络"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [64, 32],
                 output_dim: int = 3,  # efficiency, stability, comfort
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化权重网络

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（非安全目标数量）
            dropout_rate: Dropout比率
            activation: 激活函数类型
        """
        super(DynamicWeightNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 激活函数选择
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 输入特征 [batch_size, input_dim]

        Returns:
            权重logits [batch_size, output_dim]
        """
        logits = self.network(features)
        # 使用Softmax确保权重和为1
        weights = F.softmax(logits, dim=-1)
        return weights


class AdaptiveWeightManager:
    """自适应权重管理器，整合特征驱动权重生成和平滑化"""

    def __init__(self,
                 weight_network: DynamicWeightNetwork,
                 ema_decay: float = 0.9,
                 min_weight: float = 0.05,
                 max_weight: float = 0.8,
                 device: torch.device = torch.device("cpu")):
        """
        初始化权重管理器

        Args:
            weight_network: 动态权重网络
            ema_decay: 指数移动平均衰减系数
            min_weight: 最小权重限制
            max_weight: 最大权重限制
            device: 计算设备
        """
        self.weight_network = weight_network
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.device = device

        # 权重名称映射
        self.weight_names = ['efficiency', 'stability', 'comfort']

        # 历史权重存储 (用于EMA)
        self.prev_weights = None

    def extract_local_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        从观测中提取局部特征

        Args:
            obs: 观测张量 [batch_size, obs_dim]

        Returns:
            局部特征 [batch_size, feature_dim]
        """
        # 这里可以根据具体的观测结构进行特征提取
        # 目前简单使用全部观测作为特征
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        # 可以添加更复杂的特征提取逻辑
        # 例如：选择特定维度、归一化、特征工程等
        features = obs

        return features

    def apply_weight_constraints(self, weights: torch.Tensor) -> torch.Tensor:
        """
        应用权重约束

        Args:
            weights: 原始权重 [batch_size, num_objectives]

        Returns:
            约束后的权重
        """
        # 应用最小/最大权重限制
        constrained_weights = torch.clamp(weights, self.min_weight, self.max_weight)

        # 重新归一化确保和为1
        weight_sum = constrained_weights.sum(dim=-1, keepdim=True)
        constrained_weights = constrained_weights / (weight_sum + 1e-8)

        return constrained_weights

    def apply_ema_smoothing(self,
                            current_weights: torch.Tensor,
                            prev_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        应用指数移动平均平滑

        Args:
            current_weights: 当前权重
            prev_weights: 上一时刻权重

        Returns:
            平滑后的权重
        """
        if prev_weights is None:
            if self.prev_weights is not None:
                prev_weights = self.prev_weights
            else:
                # 第一次调用，无历史权重
                self.prev_weights = current_weights.clone()
                return current_weights

        # EMA平滑
        smoothed_weights = (self.ema_decay * prev_weights +
                            (1 - self.ema_decay) * current_weights)

        # 更新历史权重
        self.prev_weights = smoothed_weights.clone()

        return smoothed_weights

    def compute_weights(self,
                        obs: torch.Tensor,
                        prev_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算动态权重

        Args:
            obs: 观测张量
            prev_weights: 上一时刻权重（用于平滑）

        Returns:
            (最终权重, 计算信息)
        """
        # 提取局部特征
        features = self.extract_local_features(obs)

        # 生成原始权重
        with torch.no_grad():
            raw_weights = self.weight_network(features)

        # 应用约束
        constrained_weights = self.apply_weight_constraints(raw_weights)

        # 应用平滑
        final_weights = self.apply_ema_smoothing(constrained_weights, prev_weights)

        # 计算信息
        computation_info = {
            'features': features,
            'raw_weights': raw_weights,
            'constrained_weights': constrained_weights,
            'final_weights': final_weights,
            'weight_changes': None
        }

        if prev_weights is not None:
            computation_info['weight_changes'] = torch.abs(final_weights - prev_weights).mean()

        return final_weights, computation_info

    def get_full_weight_vector(self, non_safety_weights: torch.Tensor) -> torch.Tensor:
        """
        构建完整的权重向量（包含安全权重）

        Args:
            non_safety_weights: 非安全目标权重 [batch_size, 3]

        Returns:
            完整权重向量 [batch_size, 4] (safety=1, efficiency, stability, comfort)
        """
        batch_size = non_safety_weights.size(0)
        safety_weights = torch.ones(batch_size, 1, device=self.device)

        full_weights = torch.cat([safety_weights, non_safety_weights], dim=-1)
        return full_weights

    def update_weight_network(self,
                              obs: torch.Tensor,
                              target_weights: torch.Tensor,
                              optimizer: torch.optim.Optimizer) -> float:
        """
        更新权重网络参数

        Args:
            obs: 观测批次
            target_weights: 目标权重
            optimizer: 优化器

        Returns:
            权重预测损失
        """
        # 提取特征
        features = self.extract_local_features(obs)

        # 前向传播
        predicted_weights = self.weight_network(features)

        # 计算损失（MSE + KL散度）
        mse_loss = F.mse_loss(predicted_weights, target_weights)
        kl_loss = F.kl_div(predicted_weights.log(), target_weights, reduction='batchmean')

        total_loss = mse_loss + 0.1 * kl_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def get_weight_statistics(self) -> Dict:
        """获取权重统计信息"""
        if self.prev_weights is None:
            return {}

        stats = {
            'mean_weights': self.prev_weights.mean(dim=0),
            'std_weights': self.prev_weights.std(dim=0),
            'min_weights': self.prev_weights.min(dim=0)[0],
            'max_weights': self.prev_weights.max(dim=0)[0]
        }

        return stats


class WeightComputationSystem:
    """权重计算系统，整合所有权重相关功能"""

    def __init__(self,
                 obs_dim: int,
                 hidden_dims: list = [64, 32],
                 ema_decay: float = 0.9,
                 min_weight: float = 0.05,
                 max_weight: float = 0.8,
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 device: torch.device = torch.device("cpu")):
        """
        初始化权重计算系统

        Args:
            obs_dim: 观测维度
            hidden_dims: 隐藏层维度
            ema_decay: EMA衰减系数
            min_weight: 最小权重
            max_weight: 最大权重
            dropout_rate: Dropout比率
            activation: 激活函数类型
            device: 计算设备
        """
        # 参数验证
        assert 0 < ema_decay < 1, f"EMA decay must be in (0,1), got {ema_decay}"
        assert 0 < min_weight < max_weight < 1, f"Invalid weight bounds: min={min_weight}, max={max_weight}"

        # 创建权重网络
        self.weight_network = DynamicWeightNetwork(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=3,  # efficiency, stability, comfort
            dropout_rate=dropout_rate,
            activation=activation
        ).to(device)

        # 创建权重管理器
        self.weight_manager = AdaptiveWeightManager(
            weight_network=self.weight_network,
            ema_decay=ema_decay,
            min_weight=min_weight,
            max_weight=max_weight,
            device=device
        )

        self.device = device

    def compute_dynamic_weights(self,
                                obs: torch.Tensor,
                                prev_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算动态权重（主要接口）

        Args:
            obs: 观测张量
            prev_weights: 上一时刻权重

        Returns:
            (完整权重向量, 计算信息)
        """
        # 计算非安全权重
        non_safety_weights, computation_info = self.weight_manager.compute_weights(
            obs, prev_weights
        )

        # 构建完整权重向量
        full_weights = self.weight_manager.get_full_weight_vector(non_safety_weights)

        return full_weights, computation_info

    def to(self, device: torch.device):
        """移动到指定设备"""
        self.weight_network = self.weight_network.to(device)
        self.weight_manager.device = device
        self.device = device
        return self