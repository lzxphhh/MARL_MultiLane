#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义模块
Models Module for CQR-SCM

根据设计文档实现四种方案:
1. 方案一: MLP-QR (不使用CQR校准)
2. 方案二: GRU-QR (不使用CQR校准)
3. 方案三: MLP-CQR (使用CQR校准)
4. 方案四: GRU-CQR (使用CQR校准,推荐方案)

所有方案均预测三个分位数: q_0.05, q_0.5, q_0.95

作者: 交通流研究团队
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import random
import math
from .feature_encoder import FeatureEncoder


class MLPQR(nn.Module):
    """
    方案一/三: MLP-QR / MLP-CQR

    根据设计文档公式(11-12)实现:
    q_τ(t_0 + kΔt | h_fused) = MLP_τ(h_fused)[k]

    特点:
    - 使用独立的MLP分别预测各个分位数
    - 不显式建模时序依赖
    - 推理速度快(并行计算所有时间步)
    - 支持2分位数和3分位数输出
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, prediction_length: int = 30,
                 use_attention: bool = True, num_quantiles: int = 3):
        """
        初始化模型

        Args:
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
            dropout: Dropout比例
            prediction_length: 预测时域长度
            use_attention: 是否使用注意力机制
            num_quantiles: 输出分位数个数 (2或3)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles

        # 特征编码器
        self.encoder = FeatureEncoder(hidden_dim=hidden_dim, use_attention=use_attention)

        # 独立的MLP分别预测各个分位数
        self.mlp_lo = self._make_mlp(hidden_dim, num_layers, dropout, prediction_length)
        self.mlp_hi = self._make_mlp(hidden_dim, num_layers, dropout, prediction_length)
        if num_quantiles == 3:
            self.mlp_median = self._make_mlp(hidden_dim, num_layers, dropout, prediction_length)

    def _make_mlp(self, hidden_dim: int, num_layers: int, dropout: float,
                  output_dim: int) -> nn.Sequential:
        """
        构建MLP网络

        Args:
            hidden_dim: 隐藏层维度
            num_layers: 层数
            dropout: Dropout比例
            output_dim: 输出维度 (预测时域长度)

        Returns:
            mlp: MLP网络
        """
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))  # 输出K_f个时间步
        return nn.Sequential(*layers)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        前向传播 - 预测分位数

        Args:
            features: 特征字典

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - (batch, 30)
            对于2分位数: (q_lo, None, q_hi) - (batch, 30)
                         中位数返回None以保持接口一致
        """
        # 特征编码
        h_fused = self.encoder(
            traffic_state=features['traffic_state'],
            ego_current=features['ego_current'],
            ego_history=features['ego_history'],
            sur_current=features['sur_current'],
            sur_history=features['sur_history'],
            intention=features['intention']
        )

        # 预测分位数
        q_lo = self.mlp_lo(h_fused)  # (batch, 30)
        q_hi = self.mlp_hi(h_fused)  # (batch, 30)

        if self.num_quantiles == 3:
            q_median = self.mlp_median(h_fused)  # (batch, 30)
            # 确保分位数顺序约束: q_lo <= q_median <= q_hi
            q_median = torch.max(q_median, q_lo + 1e-6)
            q_hi = torch.max(q_hi, q_median + 1e-6)
            return q_lo, q_median, q_hi
        else:
            # 确保分位数顺序约束: q_lo <= q_hi
            q_hi = torch.max(q_hi, q_lo + 1e-6)
            return q_lo, None, q_hi


class GRUQR(nn.Module):
    """
    方案二/四: GRU-QR / GRU-CQR

    根据设计文档公式(19-22)实现:
    - GRU解码器自回归生成未来位置序列
    - 显式建模时序因果链
    - 支持Teacher Forcing和Scheduled Sampling

    特点:
    - 显式时序依赖
    - 运动学合理性
    - 不确定性传播
    - 支持2分位数和3分位数输出
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, prediction_length: int = 30,
                 use_attention: bool = True, num_quantiles: int = 3,
                 time_encoding_dim: int = 32):
        """
        初始化模型

        Args:
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比例
            prediction_length: 预测时域长度
            use_attention: 是否使用注意力机制
            num_quantiles: 输出分位数个数 (2或3)
            time_encoding_dim: 时间编码维度（默认32）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles
        self.time_encoding_dim = time_encoding_dim

        # 特征编码器
        self.encoder = FeatureEncoder(hidden_dim=hidden_dim, use_attention=use_attention)


        # 独立/统一的GRU解码器（可选）
        # 输入: [x_t, v_t, a_t, time_encoding, h_fused] = [3 + time_encoding_dim + hidden_dim]
        gru_input_size = 3 + time_encoding_dim + hidden_dim
        # self.gru_lo = nn.GRU(
        #     input_size=gru_input_size,
        #     hidden_size=hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )
        # self.gru_hi = nn.GRU(
        #     input_size=gru_input_size,
        #     hidden_size=hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )

        # if num_quantiles == 3:
        #     self.gru_median = nn.GRU(
        #         input_size=gru_input_size,
        #         hidden_size=hidden_dim,
        #         num_layers=num_layers,
        #         batch_first=True,
        #         dropout=dropout if num_layers > 1 else 0
        #     )

        # # 初始隐状态MLP
        # self.init_mlp_lo = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim * num_layers)
        # )
        # self.init_mlp_hi = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim * num_layers)
        # )
        # if num_quantiles == 3:
        #     self.init_mlp_median = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, hidden_dim * num_layers)
        #     )

        # # 输出层
        # self.output_layer_lo = nn.Linear(hidden_dim, 1)
        # self.output_layer_hi = nn.Linear(hidden_dim, 1)
        # if num_quantiles == 3:
        #     self.output_layer_median = nn.Linear(hidden_dim, 1)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 统一的初始隐状态MLP（为共享GRU初始化）
        self.init_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
            nn.Tanh()  # 限制输出到[-1, 1]，防止初始状态过大
        )

        # 多头输出层：一次性输出所有分位数
        # 对于2分位数：输出[q_lo, q_hi]
        # 对于3分位数：输出[q_lo, q_median, q_hi]
        self.output_layer = nn.Linear(hidden_dim, num_quantiles)

    def _get_time_encoding(self, step: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成Transformer风格的时间步编码

        使用正弦余弦位置编码 + 归一化的时间比例

        Args:
            step: 当前时间步索引 (0到prediction_length-1)
            batch_size: 批次大小
            device: 设备

        Returns:
            time_encoding: (batch, time_encoding_dim) 时间编码向量
        """
        # 正弦余弦编码（占用time_encoding_dim-1维）
        position = torch.tensor([step], dtype=torch.float32, device=device)

        # 创建频率维度
        div_term = torch.exp(
            torch.arange(0, self.time_encoding_dim - 1, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / (self.time_encoding_dim - 1))
        )

        # 计算正弦余弦编码
        pe = torch.zeros(self.time_encoding_dim - 1, device=device)
        pe[0::2] = torch.sin(position * div_term)
        if self.time_encoding_dim > 1:
            pe[1::2] = torch.cos(position * div_term[:len(pe[1::2])])

        # 归一化的时间比例（占用最后1维）
        time_ratio = torch.tensor([step / self.prediction_length], dtype=torch.float32, device=device)

        # 合并编码
        encoding = torch.cat([pe, time_ratio], dim=0)  # (time_encoding_dim,)

        # 扩展到批次
        encoding = encoding.unsqueeze(0).expand(batch_size, -1)  # (batch, time_encoding_dim)

        return encoding

    def _smooth_state_estimation(self, history: torch.Tensor, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用历史窗口平滑估计速度和加速度

        Args:
            history: (batch, window_size) 历史位置序列
            dt: 时间间隔（默认0.1s）

        Returns:
            v_smooth: (batch,) 平滑后的速度
            a_smooth: (batch,) 平滑后的加速度
        """
        window_size = history.size(1)

        if window_size < 2:
            # 窗口太小，使用简单差分
            v_smooth = torch.zeros_like(history[:, 0])
            a_smooth = torch.zeros_like(history[:, 0])
            return v_smooth, a_smooth

        # 计算速度：使用中心差分或前向差分
        if window_size >= 3:
            # 使用3点平滑：v ≈ (x[t] - x[t-2]) / (2*dt)
            v_smooth = (history[:, -1] - history[:, -3]) / (2 * dt)
        else:
            # 使用2点差分：v ≈ (x[t] - x[t-1]) / dt
            v_smooth = (history[:, -1] - history[:, -2]) / dt

        # 计算加速度：使用速度序列
        if window_size >= 3:
            v_history = (history[:, 1:] - history[:, :-1]) / dt  # (batch, window_size-1)
            # 使用最后两个速度估计加速度
            a_smooth = (v_history[:, -1] - v_history[:, -2]) / dt
        else:
            # 窗口太小，加速度置零
            a_smooth = torch.zeros_like(v_smooth)

        return v_smooth, a_smooth

    def _decode_quantile(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                        gru: nn.GRU, init_mlp: nn.Module, output_layer: nn.Module,
                        y_true: torch.Tensor = None,
                        teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        单个分位数的自回归解码（改进版本）

        改进内容：
        1. 添加时间步编码（Transformer风格）
        2. 使用历史窗口平滑估计速度和加速度

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            gru: GRU解码器
            init_mlp: 初始隐状态MLP
            output_layer: 输出层
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            predictions: (batch, K) - 分位数预测序列
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 初始化GRU隐状态
        h_init = init_mlp(h_fused)  # (batch, hidden_dim * num_layers)
        h_t = h_init.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        # h_t: (num_layers, batch, hidden_dim)

        # 初始状态 [x(t0), v(t0), a(t0)]
        s_t = initial_state  # (batch, 3)

        # 自回归生成
        predictions = []

        # 位置历史窗口（用于平滑速度/加速度估计）
        # 初始化为当前位置的3个副本
        position_history = s_t[:, 0:1].expand(-1, 3).clone()  # (batch, 3)

        for k in range(self.prediction_length):
            # 生成时间步编码
            time_encoding = self._get_time_encoding(k, batch_size, device)  # (batch, time_encoding_dim)

            # 构造GRU输入: [s_t, time_encoding, h_fused]
            gru_input = torch.cat([s_t, time_encoding, h_fused], dim=1).unsqueeze(1)
            # gru_input: (batch, 1, 3 + time_encoding_dim + hidden_dim)

            # GRU解码
            out, h_t = gru(gru_input, h_t)
            # out: (batch, 1, hidden_dim)

            # 预测位置
            x_pred = output_layer(out.squeeze(1)).squeeze(1)  # (batch,)
            predictions.append(x_pred)

            # 更新状态 (用于下一步)
            if self.training and y_true is not None and random.random() < teacher_forcing_ratio:
                # Teacher Forcing: 使用真实值
                x_next = y_true[:, k]
            else:
                # 自回归: 使用预测值
                x_next = x_pred

            # 更新位置历史窗口
            position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)  # (batch, 3)

            # 使用平滑方法估计v, a
            dt = 0.1  # 10Hz采样
            v_next, a_next = self._smooth_state_estimation(position_history, dt)

            s_t = torch.stack([x_next, v_next, a_next], dim=1)

        return torch.stack(predictions, dim=1)  # (batch, K)

    def _decode_all_quantiles(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                              y_true: torch.Tensor = None,
                              teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        统一的自回归解码（所有分位数同时预测，带可微分约束）

        改进内容：
        1. 单个共享GRU，避免分位数独立演化
        2. 多头输出层一次性预测所有分位数
        3. 使用Softplus实现可微分的分位数顺序约束：
           - q_lo = q_lo_raw
           - q_median = q_lo + softplus(q_median_raw - q_lo)
           - q_hi = q_median + softplus(q_hi_raw - q_median)
        4. 添加时间步编码（Transformer风格）
        5. 使用历史窗口平滑估计速度和加速度

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - 每个都是(batch, K)
            对于2分位数: (q_lo, None, q_hi) - 每个都是(batch, K)
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 初始化GRU隐状态
        h_init = self.init_mlp(h_fused)  # (batch, hidden_dim * num_layers)
        h_t = h_init.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        # h_t: (num_layers, batch, hidden_dim)

        # 初始状态 [x(t0), v(t0), a(t0)]
        s_t = initial_state  # (batch, 3)

        # 自回归生成
        q_lo_predictions = []
        q_median_predictions = [] if self.num_quantiles == 3 else None
        q_hi_predictions = []

        # 位置历史窗口（用于平滑速度/加速度估计）
        # 初始化为当前位置的3个副本
        position_history = s_t[:, 0:1].expand(-1, 3).clone()  # (batch, 3)

        for k in range(self.prediction_length):
            # 生成时间步编码
            time_encoding = self._get_time_encoding(k, batch_size, device)  # (batch, time_encoding_dim)

            # 构造GRU输入: [s_t, time_encoding, h_fused]
            gru_input = torch.cat([s_t, time_encoding, h_fused], dim=1).unsqueeze(1)
            # gru_input: (batch, 1, 3 + time_encoding_dim + hidden_dim)

            # GRU解码
            out, h_t = self.gru(gru_input, h_t)
            # out: (batch, 1, hidden_dim)

            # 多头输出：一次性预测所有分位数
            quantiles_raw = self.output_layer(out.squeeze(1))  # (batch, num_quantiles)

            if self.num_quantiles == 3:
                # 提取原始分位数
                q_lo_raw = quantiles_raw[:, 0]      # (batch,)
                q_median_raw = quantiles_raw[:, 1]  # (batch,)
                q_hi_raw = quantiles_raw[:, 2]      # (batch,)

                # 可微分约束：使用Softplus保证 q_lo < q_median < q_hi
                q_lo = q_lo_raw
                q_median = q_lo + torch.nn.functional.softplus(q_median_raw - q_lo)
                q_hi = q_median + torch.nn.functional.softplus(q_hi_raw - q_median)

                q_lo_predictions.append(q_lo)
                q_median_predictions.append(q_median)
                q_hi_predictions.append(q_hi)

                # 状态更新使用中位数（最稳定）
                x_pred = q_median
            else:
                # 2分位数模式
                q_lo_raw = quantiles_raw[:, 0]  # (batch,)
                q_hi_raw = quantiles_raw[:, 1]  # (batch,)

                # 可微分约束：使用Softplus保证 q_lo < q_hi
                q_lo = q_lo_raw
                q_hi = q_lo + torch.nn.functional.softplus(q_hi_raw - q_lo)

                q_lo_predictions.append(q_lo)
                q_hi_predictions.append(q_hi)

                # 状态更新使用中点
                x_pred = (q_lo + q_hi) / 2.0

            # 更新状态 (用于下一步)
            if self.training and y_true is not None and random.random() < teacher_forcing_ratio:
                # Teacher Forcing: 使用真实值
                x_next = y_true[:, k]
            else:
                # 自回归: 使用预测值
                x_next = x_pred

            # 更新位置历史窗口
            position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)  # (batch, 3)

            # 使用平滑方法估计v, a
            dt = 0.1  # 10Hz采样
            v_next, a_next = self._smooth_state_estimation(position_history, dt)

            s_t = torch.stack([x_next, v_next, a_next], dim=1)

        # 组装结果
        q_lo = torch.stack(q_lo_predictions, dim=1)  # (batch, K)
        q_hi = torch.stack(q_hi_predictions, dim=1)  # (batch, K)

        if self.num_quantiles == 3:
            q_median = torch.stack(q_median_predictions, dim=1)  # (batch, K)
            return q_lo, q_median, q_hi
        else:
            return q_lo, None, q_hi

    def forward(self, features: Dict[str, torch.Tensor],
                y_true: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        前向传播 - 预测分位数（使用独立/统一的GRU解码器）

        Args:
            features: 特征字典
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - (batch, 30)
            对于2分位数: (q_lo, None, q_hi) - (batch, 30)
                         中位数返回None以保持接口一致
        """
        # 特征编码
        h_fused = self.encoder(
            traffic_state=features['traffic_state'],
            ego_current=features['ego_current'],
            ego_history=features['ego_history'],
            sur_current=features['sur_current'],
            sur_history=features['sur_history'],
            intention=features['intention']
        )

        # 初始状态 [x(t0), v(t0), a(t0)]
        initial_state = features['initial_state']  # (batch, 3)

        # # GRU解码各个分位数
        # q_lo = self._decode_quantile(h_fused, initial_state, self.gru_lo,
        #                              self.init_mlp_lo, self.output_layer_lo,
        #                              y_true, teacher_forcing_ratio)

        # q_hi = self._decode_quantile(h_fused, initial_state, self.gru_hi,
        #                              self.init_mlp_hi, self.output_layer_hi,
        #                              y_true, teacher_forcing_ratio)

        # if self.num_quantiles == 3:
        #     q_median = self._decode_quantile(h_fused, initial_state, self.gru_median,
        #                                      self.init_mlp_median, self.output_layer_median,
        #                                      y_true, teacher_forcing_ratio)
        #     # 确保分位数顺序约束
        #     q_median = torch.max(q_median, q_lo + 1e-6)
        #     q_hi = torch.max(q_hi, q_median + 1e-6)
        #     return q_lo, q_median, q_hi
        # else:
        #     # 确保分位数顺序约束
        #     q_hi = torch.max(q_hi, q_lo + 1e-6)
        #     return q_lo, None, q_hi

        # 统一解码所有分位数（自带可微分约束，无需事后调整）
        return self._decode_all_quantiles(h_fused, initial_state, y_true, teacher_forcing_ratio)


class LSTMQR(nn.Module):
    """
    方案三/六: LSTM-QR / LSTM-CQR

    类似于GRUQR,但使用LSTM单元代替GRU单元
    LSTM相比GRU的优势:
    - 更强的长期依赖建模能力 (通过cell state)
    - 更好的梯度传播 (门控机制更精细)

    特点:
    - 显式时序依赖
    - 运动学合理性
    - 不确定性传播
    - 支持2分位数和3分位数输出
    - 使用LSTM的cell state和hidden state
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, prediction_length: int = 30,
                 use_attention: bool = True, num_quantiles: int = 3,
                 time_encoding_dim: int = 32):
        """
        初始化模型

        Args:
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            prediction_length: 预测时域长度
            use_attention: 是否使用注意力机制
            num_quantiles: 输出分位数个数 (2或3)
            time_encoding_dim: 时间编码维度（默认32）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles
        self.time_encoding_dim = time_encoding_dim

        # 特征编码器
        self.encoder = FeatureEncoder(hidden_dim=hidden_dim, use_attention=use_attention)

        # 独立/统一的LSTM解码器（可选）
        # 输入: [x_t, v_t, a_t, time_encoding, h_fused] = [3 + time_encoding_dim + hidden_dim]
        lstm_input_size = 3 + time_encoding_dim + hidden_dim
        # self.lstm_lo = nn.LSTM(
        #     input_size=lstm_input_size,
        #     hidden_size=hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )
        # self.lstm_hi = nn.LSTM(
        #     input_size=lstm_input_size,
        #     hidden_size=hidden_dim,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     dropout=dropout if num_layers > 1 else 0
        # )

        # if num_quantiles == 3:
        #     self.lstm_median = nn.LSTM(
        #         input_size=lstm_input_size,
        #         hidden_size=hidden_dim,
        #         num_layers=num_layers,
        #         batch_first=True,
        #         dropout=dropout if num_layers > 1 else 0
        #     )

        # # 初始隐状态MLP (用于初始化h_0和c_0)
        # self.init_mlp_lo = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim * num_layers * 2)  # *2 for both h and c
        # )
        # self.init_mlp_hi = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim * num_layers * 2)
        # )
        # if num_quantiles == 3:
        #     self.init_mlp_median = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, hidden_dim * num_layers * 2)
        #     )

        # # 输出层
        # self.output_layer_lo = nn.Linear(hidden_dim, 1)
        # self.output_layer_hi = nn.Linear(hidden_dim, 1)
        # if num_quantiles == 3:
        #     self.output_layer_median = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 统一的初始隐状态MLP (用于初始化共享LSTM的h_0和c_0)
        # 添加Tanh限制初始化范围，防止cell state爆炸
        self.init_h_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
            nn.Tanh()  # 限制输出到[-1, 1]，防止h_0过大
        )
        self.init_c_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
            nn.Tanh()  # 限制输出到[-1, 1]，防止c_0过大
        )

        # 多头输出层：一次性输出所有分位数
        # 对于2分位数：输出[q_lo, q_hi]
        # 对于3分位数：输出[q_lo, q_median, q_hi]
        self.output_layer = nn.Linear(hidden_dim, num_quantiles)

    def _get_time_encoding(self, step: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成Transformer风格的时间步编码

        使用正弦余弦位置编码 + 归一化的时间比例

        Args:
            step: 当前时间步索引 (0到prediction_length-1)
            batch_size: 批次大小
            device: 设备

        Returns:
            time_encoding: (batch, time_encoding_dim) 时间编码向量
        """
        # 正弦余弦编码（占用time_encoding_dim-1维）
        position = torch.tensor([step], dtype=torch.float32, device=device)

        # 创建频率维度
        div_term = torch.exp(
            torch.arange(0, self.time_encoding_dim - 1, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / (self.time_encoding_dim - 1))
        )

        # 计算正弦余弦编码
        pe = torch.zeros(self.time_encoding_dim - 1, device=device)
        pe[0::2] = torch.sin(position * div_term)
        if self.time_encoding_dim > 1:
            pe[1::2] = torch.cos(position * div_term[:len(pe[1::2])])

        # 归一化的时间比例（占用最后1维）
        time_ratio = torch.tensor([step / self.prediction_length], dtype=torch.float32, device=device)

        # 合并编码
        encoding = torch.cat([pe, time_ratio], dim=0)  # (time_encoding_dim,)

        # 扩展到批次
        encoding = encoding.unsqueeze(0).expand(batch_size, -1)  # (batch, time_encoding_dim)

        return encoding

    def _smooth_state_estimation(self, history: torch.Tensor, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用历史窗口平滑估计速度和加速度

        Args:
            history: (batch, window_size) 历史位置序列
            dt: 时间间隔（默认0.1s）

        Returns:
            v_smooth: (batch,) 平滑后的速度
            a_smooth: (batch,) 平滑后的加速度
        """
        window_size = history.size(1)

        if window_size < 2:
            # 窗口太小，使用简单差分
            v_smooth = torch.zeros_like(history[:, 0])
            a_smooth = torch.zeros_like(history[:, 0])
            return v_smooth, a_smooth

        # 计算速度：使用中心差分或前向差分
        if window_size >= 3:
            # 使用3点平滑：v ≈ (x[t] - x[t-2]) / (2*dt)
            v_smooth = (history[:, -1] - history[:, -3]) / (2 * dt)
        else:
            # 使用2点差分：v ≈ (x[t] - x[t-1]) / dt
            v_smooth = (history[:, -1] - history[:, -2]) / dt

        # 计算加速度：使用速度序列
        if window_size >= 3:
            v_history = (history[:, 1:] - history[:, :-1]) / dt  # (batch, window_size-1)
            # 使用最后两个速度估计加速度
            a_smooth = (v_history[:, -1] - v_history[:, -2]) / dt
        else:
            # 窗口太小，加速度置零
            a_smooth = torch.zeros_like(v_smooth)

        return v_smooth, a_smooth

    def _decode_quantile(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                        lstm: nn.LSTM, init_mlp: nn.Module, output_layer: nn.Module,
                        y_true: torch.Tensor = None,
                        teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        单个分位数的自回归解码（LSTM版本）

        改进内容：
        1. 添加时间步编码（Transformer风格）
        2. 使用历史窗口平滑估计速度和加速度
        3. 使用LSTM的cell state和hidden state

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            lstm: LSTM解码器
            init_mlp: 初始隐状态MLP
            output_layer: 输出层
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            predictions: (batch, K) - 分位数预测序列
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 初始化LSTM隐状态和cell state
        init_state = init_mlp(h_fused)  # (batch, hidden_dim * num_layers * 2)
        init_state = init_state.view(batch_size, self.num_layers, self.hidden_dim * 2)
        # 分离h_0和c_0
        h_0 = init_state[:, :, :self.hidden_dim].transpose(0, 1).contiguous()  # (num_layers, batch, hidden_dim)
        c_0 = init_state[:, :, self.hidden_dim:].transpose(0, 1).contiguous()  # (num_layers, batch, hidden_dim)

        # 初始状态 [x(t0), v(t0), a(t0)]
        s_t = initial_state  # (batch, 3)

        # 自回归生成
        predictions = []

        # 位置历史窗口（用于平滑速度/加速度估计）
        # 初始化为当前位置的3个副本
        position_history = s_t[:, 0:1].expand(-1, 3).clone()  # (batch, 3)

        h_t, c_t = h_0, c_0

        for k in range(self.prediction_length):
            # 生成时间步编码
            time_encoding = self._get_time_encoding(k, batch_size, device)  # (batch, time_encoding_dim)

            # 构造LSTM输入: [s_t, time_encoding, h_fused]
            lstm_input = torch.cat([s_t, time_encoding, h_fused], dim=1).unsqueeze(1)
            # lstm_input: (batch, 1, 3 + time_encoding_dim + hidden_dim)

            # LSTM解码
            out, (h_t, c_t) = lstm(lstm_input, (h_t, c_t))
            # out: (batch, 1, hidden_dim)

            # 预测位置
            x_pred = output_layer(out.squeeze(1)).squeeze(1)  # (batch,)
            predictions.append(x_pred)

            # 更新状态 (用于下一步)
            if self.training and y_true is not None and random.random() < teacher_forcing_ratio:
                # Teacher Forcing: 使用真实值
                x_next = y_true[:, k]
            else:
                # 自回归: 使用预测值
                x_next = x_pred

            # 更新位置历史窗口
            position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)  # (batch, 3)

            # 使用平滑方法估计v, a
            dt = 0.1  # 10Hz采样
            v_next, a_next = self._smooth_state_estimation(position_history, dt)

            s_t = torch.stack([x_next, v_next, a_next], dim=1)

        return torch.stack(predictions, dim=1)  # (batch, K)

    def _decode_all_quantiles(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                              y_true: torch.Tensor = None,
                              teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        统一的自回归解码（所有分位数同时预测，带可微分约束）

        改进内容：
        1. 单个共享LSTM，避免分位数独立演化
        2. 多头输出层一次性预测所有分位数
        3. 使用Softplus实现可微分的分位数顺序约束：
           - q_lo = q_lo_raw
           - q_median = q_lo + softplus(q_median_raw - q_lo)
           - q_hi = q_median + softplus(q_hi_raw - q_median)
        4. 添加时间步编码（Transformer风格）
        5. 使用历史窗口平滑估计速度和加速度

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - 每个都是(batch, K)
            对于2分位数: (q_lo, None, q_hi) - 每个都是(batch, K)
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 初始化LSTM隐状态和cell state（使用独立的MLP）
        h_0_flat = self.init_h_mlp(h_fused)  # (batch, hidden_dim * num_layers)
        c_0_flat = self.init_c_mlp(h_fused)  # (batch, hidden_dim * num_layers)

        # 重塑为LSTM所需格式
        h_0 = h_0_flat.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        # (num_layers, batch, hidden_dim)
        c_0 = c_0_flat.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        # (num_layers, batch, hidden_dim)

        # 初始状态 [x(t0), v(t0), a(t0)]
        s_t = initial_state  # (batch, 3)

        # 自回归生成
        q_lo_predictions = []
        q_median_predictions = [] if self.num_quantiles == 3 else None
        q_hi_predictions = []

        # 位置历史窗口（用于平滑速度/加速度估计）
        # 初始化为当前位置的3个副本
        position_history = s_t[:, 0:1].expand(-1, 3).clone()  # (batch, 3)

        h_t, c_t = h_0, c_0

        for k in range(self.prediction_length):
            # 生成时间步编码
            time_encoding = self._get_time_encoding(k, batch_size, device)  # (batch, time_encoding_dim)

            # 构造LSTM输入: [s_t, time_encoding, h_fused]
            lstm_input = torch.cat([s_t, time_encoding, h_fused], dim=1).unsqueeze(1)
            # lstm_input: (batch, 1, 3 + time_encoding_dim + hidden_dim)

            # LSTM解码
            out, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            # out: (batch, 1, hidden_dim)

            # 多头输出：一次性预测所有分位数
            quantiles_raw = self.output_layer(out.squeeze(1))  # (batch, num_quantiles)

            if self.num_quantiles == 3:
                # 提取原始分位数
                q_lo_raw = quantiles_raw[:, 0]      # (batch,)
                q_median_raw = quantiles_raw[:, 1]  # (batch,)
                q_hi_raw = quantiles_raw[:, 2]      # (batch,)

                # 可微分约束：使用Softplus保证 q_lo < q_median < q_hi
                q_lo = q_lo_raw
                q_median = q_lo + torch.nn.functional.softplus(q_median_raw - q_lo)
                q_hi = q_median + torch.nn.functional.softplus(q_hi_raw - q_median)

                q_lo_predictions.append(q_lo)
                q_median_predictions.append(q_median)
                q_hi_predictions.append(q_hi)

                # 状态更新使用中位数（最稳定）
                x_pred = q_median
            else:
                # 2分位数模式
                q_lo_raw = quantiles_raw[:, 0]  # (batch,)
                q_hi_raw = quantiles_raw[:, 1]  # (batch,)

                # 可微分约束：使用Softplus保证 q_lo < q_hi
                q_lo = q_lo_raw
                q_hi = q_lo + torch.nn.functional.softplus(q_hi_raw - q_lo)

                q_lo_predictions.append(q_lo)
                q_hi_predictions.append(q_hi)

                # 状态更新使用中点
                x_pred = (q_lo + q_hi) / 2.0

            # 更新状态 (用于下一步)
            if self.training and y_true is not None and random.random() < teacher_forcing_ratio:
                # Teacher Forcing: 使用真实值
                x_next = y_true[:, k]
            else:
                # 自回归: 使用预测值
                x_next = x_pred

            # 更新位置历史窗口
            position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)  # (batch, 3)

            # 使用平滑方法估计v, a
            dt = 0.1  # 10Hz采样
            v_next, a_next = self._smooth_state_estimation(position_history, dt)

            s_t = torch.stack([x_next, v_next, a_next], dim=1)

        # 组装结果
        q_lo = torch.stack(q_lo_predictions, dim=1)  # (batch, K)
        q_hi = torch.stack(q_hi_predictions, dim=1)  # (batch, K)

        if self.num_quantiles == 3:
            q_median = torch.stack(q_median_predictions, dim=1)  # (batch, K)
            return q_lo, q_median, q_hi
        else:
            return q_lo, None, q_hi

    def forward(self, features: Dict[str, torch.Tensor],
                y_true: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        前向传播 - 预测分位数（使用独立/统一的LSTM解码器）

        Args:
            features: 特征字典
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - (batch, 30)
            对于2分位数: (q_lo, None, q_hi) - (batch, 30)
                         中位数返回None以保持接口一致
        """
        # 特征编码
        h_fused = self.encoder(
            traffic_state=features['traffic_state'],
            ego_current=features['ego_current'],
            ego_history=features['ego_history'],
            sur_current=features['sur_current'],
            sur_history=features['sur_history'],
            intention=features['intention']
        )

        # 初始状态 [x(t0), v(t0), a(t0)]
        initial_state = features['initial_state']  # (batch, 3)

        # # LSTM解码各个分位数
        # q_lo = self._decode_quantile(h_fused, initial_state, self.lstm_lo,
        #                              self.init_mlp_lo, self.output_layer_lo,
        #                              y_true, teacher_forcing_ratio)

        # q_hi = self._decode_quantile(h_fused, initial_state, self.lstm_hi,
        #                              self.init_mlp_hi, self.output_layer_hi,
        #                              y_true, teacher_forcing_ratio)

        # if self.num_quantiles == 3:
        #     q_median = self._decode_quantile(h_fused, initial_state, self.lstm_median,
        #                                      self.init_mlp_median, self.output_layer_median,
        #                                      y_true, teacher_forcing_ratio)
        #     # 确保分位数顺序约束
        #     q_median = torch.max(q_median, q_lo + 1e-6)
        #     q_hi = torch.max(q_hi, q_median + 1e-6)
        #     return q_lo, q_median, q_hi
        # else:
        #     # 确保分位数顺序约束
        #     q_hi = torch.max(q_hi, q_lo + 1e-6)
        #     return q_lo, None, q_hi
        
        # 统一解码所有分位数（自带可微分约束，无需事后调整）
        return self._decode_all_quantiles(h_fused, initial_state, y_true, teacher_forcing_ratio)


class TransformerQR(nn.Module):
    """
    方案四/八: Transformer-QR / Transformer-CQR

    使用Transformer Decoder进行自回归轨迹预测
    Transformer相比RNN的优势:
    - 自注意力机制: 可以关注任意历史时间步
    - 并行计算: 训练时可以并行处理所有时间步
    - 位置编码: 显式建模时间信息
    - 长期依赖: 避免梯度消失/爆炸问题

    特点:
    - 自注意力建模时序依赖
    - 因果掩码保证自回归特性
    - 支持Teacher Forcing
    - 支持2分位数和3分位数输出
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4,
                 num_heads: int = 4, dropout: float = 0.1,
                 prediction_length: int = 30, use_attention: bool = True,
                 num_quantiles: int = 3, ff_dim: int = 512):
        """
        初始化模型

        Args:
            hidden_dim: 隐藏层维度 (必须能被num_heads整除)
            num_layers: Transformer层数
            num_heads: 多头注意力头数
            dropout: Dropout比例
            prediction_length: 预测时域长度
            use_attention: 是否使用注意力机制(特征编码器)
            num_quantiles: 输出分位数个数 (2或3)
            ff_dim: 前馈网络维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles
        self.ff_dim = ff_dim

        # 确保hidden_dim能被num_heads整除
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"

        # 特征编码器
        self.encoder = FeatureEncoder(hidden_dim=hidden_dim, use_attention=use_attention)

        # 状态嵌入: [x_t, v_t, a_t] -> hidden_dim
        self.state_embedding = nn.Linear(3, hidden_dim)

        # 位置编码 (使用nn.Parameter使其可学习)
        self.positional_encoding = nn.Parameter(
            self._create_positional_encoding(prediction_length, hidden_dim),
            requires_grad=True
        )

        # 独立/统一的Transformer Decoder层（可选）
        # decoder_layer_lo = nn.TransformerDecoderLayer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=ff_dim,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.transformer_decoder_lo = nn.TransformerDecoder(
        #     decoder_layer_lo,
        #     num_layers=num_layers
        # )

        # decoder_layer_hi = nn.TransformerDecoderLayer(
        #     d_model=hidden_dim,
        #     nhead=num_heads,
        #     dim_feedforward=ff_dim,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.transformer_decoder_hi = nn.TransformerDecoder(
        #     decoder_layer_hi,
        #     num_layers=num_layers
        # )

        # if num_quantiles == 3:
        #     decoder_layer_median = nn.TransformerDecoderLayer(
        #         d_model=hidden_dim,
        #         nhead=num_heads,
        #         dim_feedforward=ff_dim,
        #         dropout=dropout,
        #         batch_first=True
        #     )
        #     self.transformer_decoder_median = nn.TransformerDecoder(
        #         decoder_layer_median,
        #         num_layers=num_layers
        #     )

        # # 输出层
        # self.output_layer_lo = nn.Linear(hidden_dim, 1)
        # self.output_layer_hi = nn.Linear(hidden_dim, 1)
        # if num_quantiles == 3:
        #     self.output_layer_median = nn.Linear(hidden_dim, 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 多头输出层：一次性输出所有分位数
        # 对于2分位数：输出[q_lo, q_hi]
        # 对于3分位数：输出[q_lo, q_median, q_hi]
        self.output_layer = nn.Linear(hidden_dim, num_quantiles)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建Transformer位置编码

        Args:
            max_len: 最大序列长度
            d_model: 模型维度

        Returns:
            pe: (max_len, d_model) 位置编码
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe  # (max_len, d_model)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码 (确保位置i只能看到位置<=i的信息)

        Args:
            seq_len: 序列长度
            device: 设备

        Returns:
            mask: (seq_len, seq_len) 因果掩码, True表示被掩盖
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def _smooth_state_estimation(self, history: torch.Tensor, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用历史窗口平滑估计速度和加速度

        Args:
            history: (batch, window_size) 历史位置序列
            dt: 时间间隔（默认0.1s）

        Returns:
            v_smooth: (batch,) 平滑后的速度
            a_smooth: (batch,) 平滑后的加速度
        """
        window_size = history.size(1)

        if window_size < 2:
            v_smooth = torch.zeros_like(history[:, 0])
            a_smooth = torch.zeros_like(history[:, 0])
            return v_smooth, a_smooth

        if window_size >= 3:
            v_smooth = (history[:, -1] - history[:, -3]) / (2 * dt)
        else:
            v_smooth = (history[:, -1] - history[:, -2]) / dt

        if window_size >= 3:
            v_history = (history[:, 1:] - history[:, :-1]) / dt
            a_smooth = (v_history[:, -1] - v_history[:, -2]) / dt
        else:
            a_smooth = torch.zeros_like(v_smooth)

        return v_smooth, a_smooth

    def _decode_quantile(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                        transformer_decoder: nn.TransformerDecoder, output_layer: nn.Module,
                        y_true: torch.Tensor = None,
                        teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        单个分位数的自回归解码（Transformer版本）

        实现方式:
        1. 训练模式: 使用Teacher Forcing并行处理所有时间步
        2. 推理模式: 自回归逐步生成

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            transformer_decoder: Transformer解码器
            output_layer: 输出层
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            predictions: (batch, K) - 分位数预测序列
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 将h_fused扩展为memory (用于cross-attention)
        memory = h_fused.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 训练模式 + Teacher Forcing
        if self.training and y_true is not None and teacher_forcing_ratio > 0:
            # 构造完整的状态序列 (使用真实值)
            positions = y_true  # (batch, K)

            # 构造状态序列
            # 初始状态
            states = [initial_state]  # List of (batch, 3)

            # 位置历史窗口
            position_history = initial_state[:, 0:1].expand(-1, 3).clone()

            for k in range(self.prediction_length):
                if random.random() < teacher_forcing_ratio:
                    x_next = y_true[:, k]
                else:
                    # 这里需要先做一次前向传播获取预测
                    # 为简化,在训练时全部使用teacher forcing
                    x_next = y_true[:, k]

                # 更新位置历史
                position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)
                v_next, a_next = self._smooth_state_estimation(position_history, dt=0.1)
                states.append(torch.stack([x_next, v_next, a_next], dim=1))

            # 堆叠状态序列: (batch, K, 3)
            states_seq = torch.stack(states[:-1], dim=1)  # 不包括最后一个状态

            # 状态嵌入
            tgt = self.state_embedding(states_seq)  # (batch, K, hidden_dim)

            # 添加位置编码
            tgt = tgt + self.positional_encoding[:self.prediction_length].unsqueeze(0)

            # 生成因果掩码
            causal_mask = self._generate_causal_mask(self.prediction_length, device)

            # Transformer解码
            decoder_output = transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask
            )  # (batch, K, hidden_dim)

            # 输出预测
            predictions = output_layer(decoder_output).squeeze(-1)  # (batch, K)

        else:
            # 推理模式: 自回归生成
            predictions = []
            s_t = initial_state  # (batch, 3)
            position_history = s_t[:, 0:1].expand(-1, 3).clone()

            # 存储已生成的状态序列
            generated_states = [s_t]  # List of (batch, 3)

            for k in range(self.prediction_length):
                # 当前状态序列
                states_seq = torch.stack(generated_states, dim=1)  # (batch, k+1, 3)

                # 状态嵌入
                tgt = self.state_embedding(states_seq)  # (batch, k+1, hidden_dim)

                # 添加位置编码
                tgt = tgt + self.positional_encoding[:k+1].unsqueeze(0)

                # 生成因果掩码
                causal_mask = self._generate_causal_mask(k+1, device)

                # Transformer解码
                decoder_output = transformer_decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=causal_mask
                )  # (batch, k+1, hidden_dim)

                # 取最后一个时间步的输出
                x_pred = output_layer(decoder_output[:, -1, :]).squeeze(-1)  # (batch,)
                predictions.append(x_pred)

                # 更新状态
                position_history = torch.cat([position_history[:, 1:], x_pred.unsqueeze(1)], dim=1)
                v_next, a_next = self._smooth_state_estimation(position_history, dt=0.1)
                s_t = torch.stack([x_pred, v_next, a_next], dim=1)
                generated_states.append(s_t)

            predictions = torch.stack(predictions, dim=1)  # (batch, K)

        return predictions

    def _decode_all_quantiles(self, h_fused: torch.Tensor, initial_state: torch.Tensor,
                              y_true: torch.Tensor = None,
                              teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        统一的自回归解码（所有分位数同时预测，带可微分约束）

        改进内容：
        1. 单个共享Transformer Decoder，避免分位数独立演化
        2. 多头输出层一次性预测所有分位数
        3. 使用Softplus实现可微分的分位数顺序约束
        4. 训练模式：使用Teacher Forcing并行处理所有时间步
        5. 推理模式：自回归逐步生成

        Args:
            h_fused: (batch, hidden_dim) - 融合特征
            initial_state: (batch, 3) - 初始状态 [x(t0), v(t0), a(t0)]
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - 每个都是(batch, K)
            对于2分位数: (q_lo, None, q_hi) - 每个都是(batch, K)
        """
        batch_size = h_fused.size(0)
        device = h_fused.device

        # 将h_fused扩展为memory (用于cross-attention)
        memory = h_fused.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 训练模式 + Teacher Forcing
        if self.training and y_true is not None and teacher_forcing_ratio > 0:
            # 构造完整的状态序列 (使用真实值)
            # 初始状态
            states = [initial_state]  # List of (batch, 3)

            # 位置历史窗口
            position_history = initial_state[:, 0:1].expand(-1, 3).clone()

            for k in range(self.prediction_length):
                if random.random() < teacher_forcing_ratio:
                    x_next = y_true[:, k]
                else:
                    # 为简化,在训练时全部使用teacher forcing
                    x_next = y_true[:, k]

                # 更新位置历史
                position_history = torch.cat([position_history[:, 1:], x_next.unsqueeze(1)], dim=1)
                v_next, a_next = self._smooth_state_estimation(position_history, dt=0.1)
                states.append(torch.stack([x_next, v_next, a_next], dim=1))

            # 堆叠状态序列: (batch, K, 3)
            states_seq = torch.stack(states[:-1], dim=1)  # 不包括最后一个状态

            # 状态嵌入
            tgt = self.state_embedding(states_seq)  # (batch, K, hidden_dim)

            # 添加位置编码
            tgt = tgt + self.positional_encoding[:self.prediction_length].unsqueeze(0)

            # 生成因果掩码
            causal_mask = self._generate_causal_mask(self.prediction_length, device)

            # Transformer解码
            decoder_output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask
            )  # (batch, K, hidden_dim)

            # 多头输出：一次性预测所有分位数
            quantiles_raw = self.output_layer(decoder_output)  # (batch, K, num_quantiles)

            if self.num_quantiles == 3:
                # 提取原始分位数
                q_lo_raw = quantiles_raw[:, :, 0]      # (batch, K)
                q_median_raw = quantiles_raw[:, :, 1]  # (batch, K)
                q_hi_raw = quantiles_raw[:, :, 2]      # (batch, K)

                # 可微分约束：使用Softplus保证 q_lo < q_median < q_hi
                q_lo = q_lo_raw
                q_median = q_lo + torch.nn.functional.softplus(q_median_raw - q_lo)
                q_hi = q_median + torch.nn.functional.softplus(q_hi_raw - q_median)

                return q_lo, q_median, q_hi
            else:
                # 2分位数模式
                q_lo_raw = quantiles_raw[:, :, 0]  # (batch, K)
                q_hi_raw = quantiles_raw[:, :, 1]  # (batch, K)

                # 可微分约束：使用Softplus保证 q_lo < q_hi
                q_lo = q_lo_raw
                q_hi = q_lo + torch.nn.functional.softplus(q_hi_raw - q_lo)

                return q_lo, None, q_hi

        else:
            # 推理模式: 自回归生成
            q_lo_predictions = []
            q_median_predictions = [] if self.num_quantiles == 3 else None
            q_hi_predictions = []

            s_t = initial_state  # (batch, 3)
            position_history = s_t[:, 0:1].expand(-1, 3).clone()

            # 存储已生成的状态序列
            generated_states = [s_t]  # List of (batch, 3)

            for k in range(self.prediction_length):
                # 当前状态序列
                states_seq = torch.stack(generated_states, dim=1)  # (batch, k+1, 3)

                # 状态嵌入
                tgt = self.state_embedding(states_seq)  # (batch, k+1, hidden_dim)

                # 添加位置编码
                tgt = tgt + self.positional_encoding[:k+1].unsqueeze(0)

                # 生成因果掩码
                causal_mask = self._generate_causal_mask(k+1, device)

                # Transformer解码
                decoder_output = self.transformer_decoder(
                    tgt=tgt,
                    memory=memory,
                    tgt_mask=causal_mask
                )  # (batch, k+1, hidden_dim)

                # 取最后一个时间步的输出，多头输出所有分位数
                quantiles_raw = self.output_layer(decoder_output[:, -1, :])  # (batch, num_quantiles)

                if self.num_quantiles == 3:
                    # 提取原始分位数
                    q_lo_raw = quantiles_raw[:, 0]      # (batch,)
                    q_median_raw = quantiles_raw[:, 1]  # (batch,)
                    q_hi_raw = quantiles_raw[:, 2]      # (batch,)

                    # 可微分约束：使用Softplus保证 q_lo < q_median < q_hi
                    q_lo = q_lo_raw
                    q_median = q_lo + torch.nn.functional.softplus(q_median_raw - q_lo)
                    q_hi = q_median + torch.nn.functional.softplus(q_hi_raw - q_median)

                    q_lo_predictions.append(q_lo)
                    q_median_predictions.append(q_median)
                    q_hi_predictions.append(q_hi)

                    # 状态更新使用中位数（最稳定）
                    x_pred = q_median
                else:
                    # 2分位数模式
                    q_lo_raw = quantiles_raw[:, 0]  # (batch,)
                    q_hi_raw = quantiles_raw[:, 1]  # (batch,)

                    # 可微分约束：使用Softplus保证 q_lo < q_hi
                    q_lo = q_lo_raw
                    q_hi = q_lo + torch.nn.functional.softplus(q_hi_raw - q_lo)

                    q_lo_predictions.append(q_lo)
                    q_hi_predictions.append(q_hi)

                    # 状态更新使用中点
                    x_pred = (q_lo + q_hi) / 2.0

                # 更新状态
                position_history = torch.cat([position_history[:, 1:], x_pred.unsqueeze(1)], dim=1)
                v_next, a_next = self._smooth_state_estimation(position_history, dt=0.1)
                s_t = torch.stack([x_pred, v_next, a_next], dim=1)
                generated_states.append(s_t)

            # 组装结果
            q_lo = torch.stack(q_lo_predictions, dim=1)  # (batch, K)
            q_hi = torch.stack(q_hi_predictions, dim=1)  # (batch, K)

            if self.num_quantiles == 3:
                q_median = torch.stack(q_median_predictions, dim=1)  # (batch, K)
                return q_lo, q_median, q_hi
            else:
                return q_lo, None, q_hi

    def forward(self, features: Dict[str, torch.Tensor],
                y_true: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.0) -> Tuple[torch.Tensor, ...]:
        """
        前向传播 - 预测分位数（使用独立/统一的Transformer Decoder）

        Args:
            features: 特征字典
            y_true: (batch, K) - 真实值 (用于Teacher Forcing)
            teacher_forcing_ratio: Teacher Forcing比例

        Returns:
            对于3分位数: (q_lo, q_median, q_hi) - (batch, 30)
            对于2分位数: (q_lo, None, q_hi) - (batch, 30)
                         中位数返回None以保持接口一致
        """
        # 特征编码
        h_fused = self.encoder(
            traffic_state=features['traffic_state'],
            ego_current=features['ego_current'],
            ego_history=features['ego_history'],
            sur_current=features['sur_current'],
            sur_history=features['sur_history'],
            intention=features['intention']
        )

        # 初始状态 [x(t0), v(t0), a(t0)]
        initial_state = features['initial_state']  # (batch, 3)

        # # Transformer解码各个分位数
        # q_lo = self._decode_quantile(h_fused, initial_state, self.transformer_decoder_lo,
        #                              self.output_layer_lo, y_true, teacher_forcing_ratio)

        # q_hi = self._decode_quantile(h_fused, initial_state, self.transformer_decoder_hi,
        #                              self.output_layer_hi, y_true, teacher_forcing_ratio)

        # if self.num_quantiles == 3:
        #     q_median = self._decode_quantile(h_fused, initial_state, self.transformer_decoder_median,
        #                                      self.output_layer_median, y_true, teacher_forcing_ratio)
        #     # 确保分位数顺序约束
        #     q_median = torch.max(q_median, q_lo + 1e-6)
        #     q_hi = torch.max(q_hi, q_median + 1e-6)
        #     return q_lo, q_median, q_hi
        # else:
        #     # 确保分位数顺序约束
        #     q_hi = torch.max(q_hi, q_lo + 1e-6)
        #     return q_lo, None, q_hi
        
        # 统一解码所有分位数（自带可微分约束，无需事后调整）
        return self._decode_all_quantiles(h_fused, initial_state, y_true, teacher_forcing_ratio)


def main():
    """测试模型"""
    print("=" * 80)
    print("模型测试")
    print("=" * 80)

    # 测试数据
    batch_size = 8
    features = {
        'traffic_state': torch.randn(batch_size, 6),
        'ego_current': torch.randn(batch_size, 4),
        'ego_history': torch.randn(batch_size, 20, 5),
        'sur_current': torch.randn(batch_size, 6, 8),
        'sur_history': torch.randn(batch_size, 6, 20, 5),
        'intention': torch.randint(0, 2, (batch_size, 1)).float(),
        'initial_state': torch.randn(batch_size, 3)
    }
    y_true = torch.randn(batch_size, 30) * 10 + 50

    # 测试MLP-QR
    print("\n测试MLP-QR:")
    print("-" * 80)
    mlp_model = MLPQR(hidden_dim=256, num_layers=4)
    num_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    print(f"参数量: {num_params_mlp:,}")

    import time
    start = time.time()
    q_lo, q_median, q_hi = mlp_model(features)
    elapsed = time.time() - start

    print(f"输出形状:")
    print(f"  q_lo: {q_lo.shape}")
    print(f"  q_median: {q_median.shape}")
    print(f"  q_hi: {q_hi.shape}")
    print(f"推理时间: {elapsed*1000:.2f}ms ({elapsed*1000/batch_size:.2f}ms/样本)")

    # 验证分位数约束
    assert torch.all(q_median >= q_lo), "q_median应该 >= q_lo"
    assert torch.all(q_hi >= q_median), "q_hi应该 >= q_median"
    print("✓ 分位数顺序约束满足")

    # 测试GRU-QR
    print("\n测试GRU-QR:")
    print("-" * 80)
    gru_model = GRUQR(hidden_dim=128, num_layers=2)
    num_params_gru = sum(p.numel() for p in gru_model.parameters())
    print(f"参数量: {num_params_gru:,}")

    start = time.time()
    q_lo, q_median, q_hi = gru_model(features, y_true=None, teacher_forcing_ratio=0.0)
    elapsed = time.time() - start

    print(f"输出形状:")
    print(f"  q_lo: {q_lo.shape}")
    print(f"  q_median: {q_median.shape}")
    print(f"  q_hi: {q_hi.shape}")
    print(f"推理时间: {elapsed*1000:.2f}ms ({elapsed*1000/batch_size:.2f}ms/样本)")

    # 验证分位数约束
    assert torch.all(q_median >= q_lo), "q_median应该 >= q_lo"
    assert torch.all(q_hi >= q_median), "q_hi应该 >= q_median"
    print("✓ 分位数顺序约束满足")

    # 测试Teacher Forcing
    print("\n测试Teacher Forcing:")
    gru_model.train()
    q_lo_tf, q_median_tf, q_hi_tf = gru_model(features, y_true=y_true, teacher_forcing_ratio=0.9)
    print(f"✓ Teacher Forcing模式正常")

    # 测试LSTM-QR
    print("\n测试LSTM-QR:")
    print("-" * 80)
    lstm_model = LSTMQR(hidden_dim=128, num_layers=2)
    num_params_lstm = sum(p.numel() for p in lstm_model.parameters())
    print(f"参数量: {num_params_lstm:,}")

    start = time.time()
    q_lo, q_median, q_hi = lstm_model(features, y_true=None, teacher_forcing_ratio=0.0)
    elapsed = time.time() - start

    print(f"输出形状:")
    print(f"  q_lo: {q_lo.shape}")
    print(f"  q_median: {q_median.shape}")
    print(f"  q_hi: {q_hi.shape}")
    print(f"推理时间: {elapsed*1000:.2f}ms ({elapsed*1000/batch_size:.2f}ms/样本)")

    # 验证分位数约束
    assert torch.all(q_median >= q_lo), "q_median应该 >= q_lo"
    assert torch.all(q_hi >= q_median), "q_hi应该 >= q_median"
    print("✓ 分位数顺序约束满足")

    # 测试Teacher Forcing
    print("\n测试LSTM Teacher Forcing:")
    lstm_model.train()
    q_lo_tf, q_median_tf, q_hi_tf = lstm_model(features, y_true=y_true, teacher_forcing_ratio=0.9)
    print(f"✓ Teacher Forcing模式正常")

    # 测试Transformer-QR
    print("\n测试Transformer-QR:")
    print("-" * 80)
    transformer_model = TransformerQR(hidden_dim=128, num_layers=2, num_heads=4, ff_dim=256)
    num_params_transformer = sum(p.numel() for p in transformer_model.parameters())
    print(f"参数量: {num_params_transformer:,}")

    start = time.time()
    q_lo, q_median, q_hi = transformer_model(features, y_true=None, teacher_forcing_ratio=0.0)
    elapsed = time.time() - start

    print(f"输出形状:")
    print(f"  q_lo: {q_lo.shape}")
    print(f"  q_median: {q_median.shape}")
    print(f"  q_hi: {q_hi.shape}")
    print(f"推理时间: {elapsed*1000:.2f}ms ({elapsed*1000/batch_size:.2f}ms/样本)")

    # 验证分位数约束
    assert torch.all(q_median >= q_lo), "q_median应该 >= q_lo"
    assert torch.all(q_hi >= q_median), "q_hi应该 >= q_median"
    print("✓ 分位数顺序约束满足")

    # 测试Transformer Teacher Forcing
    print("\n测试Transformer Teacher Forcing:")
    transformer_model.train()
    q_lo_tf, q_median_tf, q_hi_tf = transformer_model(features, y_true=y_true, teacher_forcing_ratio=0.9)
    print(f"✓ Teacher Forcing模式正常")

    print("\n" + "=" * 80)
    print("✓ 所有模型测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
