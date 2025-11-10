#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征编码器模块
Feature Encoder Module for CQR-SCM

根据设计文档第4节实现特征提取和融合:
1. 自车历史编码 - GRU
2. 周车历史编码 - GRU + 意图调制的注意力机制
3. 交通状态编码 - MLP
4. 特征融合 - 拼接 + MLP

作者: 交通流研究团队
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class IntentionModulatedAttention(nn.Module):
    """
    意图调制的注意力机制

    根据横向意图(车道保持/变道)调制周车的注意力权重:
    - I=0 (车道保持): 主要关注本车道前车(EF)
    - I=1 (变道): 关注目标车道前后车(LF, LR 或 RF, RR)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 注意力权重网络(依赖于意图)
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for intention
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sur_features: torch.Tensor, intention: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sur_features: (batch, 6, hidden_dim) - 6个方向的周车特征
            intention: (batch, 1) - 横向意图
            masks: (batch, 6) - 周车存在掩码(1=存在, 0=不存在)

        Returns:
            fused_features: (batch, hidden_dim) - 融合后的周车特征
        """
        batch_size = sur_features.size(0)

        # 扩展intention以匹配周车数量
        intention_expanded = intention.unsqueeze(1).expand(batch_size, 6, 1)

        # 拼接周车特征和意图
        attention_input = torch.cat([sur_features, intention_expanded], dim=2)

        # 计算注意力得分
        attention_scores = self.attention_mlp(attention_input).squeeze(2)  # (batch, 6)

        # 应用掩码(不存在的周车得分设为-inf)
        attention_scores = attention_scores.masked_fill(masks == 0, float('-inf'))

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, 6)

        # 加权求和
        fused_features = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, 6)
            sur_features  # (batch, 6, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)

        return fused_features


class FeatureEncoder(nn.Module):
    """
    特征编码器

    将原始特征编码为融合特征向量 h_fused
    """

    def __init__(self, hidden_dim: int = 128, use_attention: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # 1. 自车历史GRU编码器
        # 输入: (20, 5) - [Δx, Δy, Δv, a, lane_id]
        self.ego_gru = nn.GRU(
            input_size=5,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 2. 周车历史GRU编码器(共享权重)
        self.sur_gru = nn.GRU(
            input_size=5,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 3. 意图调制注意力机制
        if use_attention:
            self.attention = IntentionModulatedAttention(hidden_dim)

        # 4. 交通状态MLP
        # 输入: (6,) - [v_left, κ_left, v_ego, κ_ego, v_right, κ_right]
        self.traffic_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 5. 自车当前状态MLP
        # 输入: (4,) - [v, a, lane_id, eta]
        self.ego_current_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU()
        )

        # 6. 特征融合MLP
        # 输入: h_ego + h_sur + h_traffic + ego_current + intention
        fusion_input_dim = hidden_dim + hidden_dim + hidden_dim + hidden_dim // 2 + 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, traffic_state: torch.Tensor, ego_current: torch.Tensor,
                ego_history: torch.Tensor, sur_current: torch.Tensor,
                sur_history: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            traffic_state: (batch, 6) - 交通状态
            ego_current: (batch, 4) - 自车当前状态
            ego_history: (batch, 20, 5) - 自车历史
            sur_current: (batch, 6, 8) - 周车当前状态
            sur_history: (batch, 6, 20, 5) - 周车历史
            intention: (batch, 1) - 横向意图

        Returns:
            h_fused: (batch, hidden_dim) - 融合特征向量
        """
        batch_size = traffic_state.size(0)

        # 1. 编码自车历史
        _, h_ego = self.ego_gru(ego_history)  # h: (2, batch, hidden_dim)
        h_ego = h_ego[-1]  # 取最后一层: (batch, hidden_dim)

        # 2. 编码周车历史(每个方向独立编码)
        sur_features = []
        sur_masks = []

        for i in range(6):  # 6个方向
            sur_hist_i = sur_history[:, i, :, :]  # (batch, 20, 5)

            # 检查周车是否存在(根据特征是否全为0判断)
            mask = (sur_hist_i.abs().sum(dim=(1, 2)) > 0).float()  # (batch,)
            sur_masks.append(mask)

            # GRU编码
            _, h_sur_i = self.sur_gru(sur_hist_i)
            h_sur_i = h_sur_i[-1]  # (batch, hidden_dim)
            sur_features.append(h_sur_i)

        # 堆叠周车特征
        sur_features = torch.stack(sur_features, dim=1)  # (batch, 6, hidden_dim)
        sur_masks = torch.stack(sur_masks, dim=1)  # (batch, 6)

        # 3. 注意力融合周车特征
        if self.use_attention:
            h_sur = self.attention(sur_features, intention, sur_masks)
        else:
            # 简单平均(考虑掩码)
            sur_masks_expanded = sur_masks.unsqueeze(2)  # (batch, 6, 1)
            masked_features = sur_features * sur_masks_expanded
            h_sur = masked_features.sum(dim=1) / (sur_masks.sum(dim=1, keepdim=True) + 1e-8)

        # 4. 编码交通状态
        h_traffic = self.traffic_mlp(traffic_state)  # (batch, hidden_dim)

        # 5. 编码自车当前状态
        h_ego_current = self.ego_current_mlp(ego_current)  # (batch, hidden_dim//2)

        # 6. 融合所有特征
        h_concat = torch.cat([
            h_ego,          # (batch, hidden_dim)
            h_sur,          # (batch, hidden_dim)
            h_traffic,      # (batch, hidden_dim)
            h_ego_current,  # (batch, hidden_dim//2)
            intention       # (batch, 1)
        ], dim=1)

        h_fused = self.fusion_mlp(h_concat)  # (batch, hidden_dim)

        return h_fused


def main():
    """测试特征编码器"""
    print("=" * 80)
    print("特征编码器测试")
    print("=" * 80)

    # 创建编码器
    encoder = FeatureEncoder(hidden_dim=128, use_attention=True)

    # 创建测试数据
    batch_size = 16
    traffic_state = torch.randn(batch_size, 6)
    ego_current = torch.randn(batch_size, 4)
    ego_history = torch.randn(batch_size, 20, 5)
    sur_current = torch.randn(batch_size, 6, 8)
    sur_history = torch.randn(batch_size, 6, 20, 5)
    intention = torch.randint(0, 2, (batch_size, 1)).float()

    print(f"\n输入形状:")
    print(f"  traffic_state: {traffic_state.shape}")
    print(f"  ego_current: {ego_current.shape}")
    print(f"  ego_history: {ego_history.shape}")
    print(f"  sur_current: {sur_current.shape}")
    print(f"  sur_history: {sur_history.shape}")
    print(f"  intention: {intention.shape}")

    # 前向传播
    h_fused = encoder(traffic_state, ego_current, ego_history,
                     sur_current, sur_history, intention)

    print(f"\n输出形状:")
    print(f"  h_fused: {h_fused.shape}")

    # 统计参数量
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n模型参数量: {num_params:,}")

    print("\n" + "=" * 80)
    print("✓ 特征编码器测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
