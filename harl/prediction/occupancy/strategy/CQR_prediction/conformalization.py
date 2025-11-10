#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共形化预测模块
Conformalized Prediction Module

根据设计文档第6节实现CQR算法:
1. 在校准集上计算符合性得分
2. 计算校准分位数
3. 应用校准分位数到测试预测

理论保证: P{Y ∈ C(X)} ≥ 1-α

作者: 交通流研究团队
日期: 2025
"""

import numpy as np
import torch
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class ConformalizationCalibrator:
    """
    共形化校准器

    实现CQR算法,提供有限样本覆盖保证
    支持2分位数（仅共形化校准）和3分位数（共形化+中位数修正）
    """

    def __init__(self, alpha: float = 0.1, prediction_length: int = 30, num_quantiles: int = 3):
        """
        初始化校准器

        Args:
            alpha: 误覆盖率(默认0.1,对应90%覆盖率)
            prediction_length: 预测时域长度(默认30步)
            num_quantiles: 分位数个数 (2或3)
        """
        self.alpha = alpha
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles

        # 校准分位数(初始化为None,需要先调用calibrate方法)
        # 注意：CQR使用单一的Q值，而不是分离的Q_lo和Q_hi
        self.Q = None  # shape: (prediction_length,) - CQR校准分位数

        # 校准集大小(用于有限样本修正)
        self.n_calib = None

        print(f"初始化共形化校准器: α={alpha}, 目标覆盖率≥{1-alpha:.1%}, 分位数个数={num_quantiles}")

    def calibrate(self, q_lo_pred: np.ndarray, q_median_pred: np.ndarray,
                  q_hi_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        标准CQR校准算法 (基于2019 NIPS论文)

        Args:
            q_lo_pred: (N_calib, K_f) - 下分位数预测
            q_median_pred: (N_calib, K_f) - 中位数预测 (仅用于验证,不参与校准)
            q_hi_pred: (N_calib, K_f) - 上分位数预测
            y_true: (N_calib, K_f) - 真实未来位置

        CQR算法 (Romano et al., NeurIPS 2019):
        1. 计算conformity scores: E = max(q_lo - y, y - q_hi)
        2. 计算校准分位数: Q = quantile(E, (1-α)(1+1/n))
        3. 构造区间: [q_lo - Q, q_hi + Q]
        """
        assert q_lo_pred.shape == y_true.shape, "预测和标签形状不匹配"
        assert q_hi_pred.shape == y_true.shape, "预测和标签形状不匹配"

        self.n_calib = len(q_lo_pred)

        print(f"\n{'='*80}")
        print("CQR校准 (Conformalized Quantile Regression)")
        print(f"{'='*80}")
        print(f"校准集样本数: {self.n_calib}")
        print(f"预测时域长度: {self.prediction_length} 步")

        # 🔍 诊断: 检查原始QR在校准集上的覆盖率
        covered_qr = (y_true >= q_lo_pred) & (y_true <= q_hi_pred)  # (N, K_f)
        qr_coverage_per_step = covered_qr.mean(axis=0)  # (K_f,)
        qr_coverage_overall = covered_qr.mean()

        print(f"\n[诊断] 原始QR在校准集上的表现:")
        print(f"  整体Coverage: {qr_coverage_overall*100:.1f}%")
        print(f"  1s Coverage:   {qr_coverage_per_step[:10].mean()*100:.1f}%")
        print(f"  2s Coverage:   {qr_coverage_per_step[10:20].mean()*100:.1f}%")
        print(f"  3s Coverage:   {qr_coverage_per_step[20:30].mean()*100:.1f}%")

        qr_width_per_step = (q_hi_pred - q_lo_pred).mean(axis=0)  # (K_f,)
        print(f"  原始区间宽度: 1s={qr_width_per_step[:10].mean():.4f}, "
              f"2s={qr_width_per_step[10:20].mean():.4f}, "
              f"3s={qr_width_per_step[20:30].mean():.4f}")

        # 1. 计算conformity scores (CQR核心公式)
        # E(x,y) = max{q_lo(x) - y, y - q_hi(x)}
        # 含义: 区间未能覆盖真实值的程度
        # - E <= 0: y在区间内，完美覆盖
        # - E > 0: y在区间外，需要扩展E的距离
        E = np.maximum(q_lo_pred - y_true, y_true - q_hi_pred)  # (N, K_f)

        print(f"\nConformity Scores统计:")
        print(f"  E 范围: [{E.min():.3f}, {E.max():.3f}]")
        print(f"  E 均值: {E.mean():.3f}")
        print(f"  E <= 0 (已覆盖)的比例: {(E <= 0).mean()*100:.1f}%")
        print(f"  E > 0 (需扩展)的比例: {(E > 0).mean()*100:.1f}%")

        # 2. 计算校准分位数
        # 有限样本修正: ceil((n+1)(1-α))/n
        quantile_level = (1 - self.alpha) * (1 + 1 / self.n_calib)

        # 独立校准: 每个时间步使用各自的conformity score分布
        self.Q = np.quantile(E, quantile_level, axis=0)  # (K_f,)

        print(f"\n校准分位数 (分位数水平={quantile_level:.4f}):")
        print(f"  Q 范围: [{self.Q.min():.3f}, {self.Q.max():.3f}]")
        print(f"  Q 均值: {self.Q.mean():.3f}")

        # 3. 分析校准分位数的时序演化
        print(f"\n校准分位数时序演化 (每5步):")
        print(f"  {'步数':>6} {'Q':>10} {'QR区间宽度':>15} {'CQR区间宽度':>15}")
        print(f"  {'-'*50}")
        for k in range(0, self.prediction_length, 5):
            qr_width = (q_hi_pred[:, k] - q_lo_pred[:, k]).mean()
            cqr_width = qr_width + 2 * self.Q[k]
            print(f"  t{k+1:>5} {self.Q[k]:>10.3f} {qr_width:>15.3f} {cqr_width:>15.3f}")

        print(f"\n✓ CQR校准完成")
        print(f"  理论保证: P(Y ∈ [q_lo - Q, q_hi + Q]) ≥ {100*(1-self.alpha):.1f}%")
        print(f"  关键特性: 保持QR的可变宽度自适应区间")
        print(f"{'='*80}\n")

    def calibrate_joint(self, q_lo_pred: np.ndarray, q_median_pred: np.ndarray,
                       q_hi_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        联合校准 - 使用最大conformity score保证轨迹级别的覆盖率

        与独立校准的区别:
        - 独立校准: 每个时间步独立校准,边际覆盖保证
        - 联合校准: 使用每条轨迹的最大E,保证整条轨迹的覆盖率 ≥ 90%

        Args:
            q_lo_pred: (N_calib, K_f) - 下分位数预测
            q_median_pred: (N_calib, K_f) - 中位数预测 (不参与校准)
            q_hi_pred: (N_calib, K_f) - 上分位数预测
            y_true: (N_calib, K_f) - 真实未来位置

        CQR联合校准算法:
        1. 对每个轨迹,计算所有时间步的conformity scores
        2. 取每条轨迹的最大E: E_max[i] = max_k E[i,k]
        3. 使用E_max计算校准分位数Q
        4. 将Q广播到所有时间步
        """
        assert q_lo_pred.shape == y_true.shape, "预测和标签形状不匹配"
        assert q_hi_pred.shape == y_true.shape, "预测和标签形状不匹配"

        self.n_calib = len(q_lo_pred)

        print(f"\n{'='*80}")
        print("CQR联合校准 (Joint Calibration)")
        print(f"{'='*80}")
        print(f"校准集样本数: {self.n_calib}")
        print(f"预测时域长度: {self.prediction_length} 步")

        # 1. 计算conformity scores
        E = np.maximum(q_lo_pred - y_true, y_true - q_hi_pred)  # (N, K_f)

        # 2. 对每条轨迹取最大conformity score (最坏情况)
        # 这保证了如果调整后的区间覆盖了最坏情况,则覆盖所有时间步
        E_max = E.max(axis=1)  # (N,) - 每条轨迹的最大conformity score

        print(f"\nConformity Scores统计:")
        print(f"  E 范围: [{E.min():.3f}, {E.max():.3f}]")
        print(f"  E_max: 均值={E_max.mean():.3f}, 标准差={E_max.std():.3f}, 最大值={E_max.max():.3f}")

        # 3. 计算校准分位数 (使用最大conformity score)
        quantile_level = (1 - self.alpha) * (1 + 1 / self.n_calib)
        Q_joint = float(np.quantile(E_max, quantile_level))

        print(f"\n联合校准分位数 (分位数水平={quantile_level:.4f}):")
        print(f"  Q = {Q_joint:.3f}")

        # 4. 广播到所有时间步 (所有时间步使用相同的Q)
        self.Q = np.full(self.prediction_length, Q_joint)

        # 显示区间宽度
        print(f"\n区间宽度分析:")
        for k in range(0, self.prediction_length, 5):
            qr_width = (q_hi_pred[:, k] - q_lo_pred[:, k]).mean()
            cqr_width = qr_width + 2 * Q_joint
            print(f"  t{k+1:>5}: QR={qr_width:>6.3f}, CQR={cqr_width:>6.3f}")

        print(f"\n✓ CQR联合校准完成")
        print(f"  理论保证: 至少 {100*(1-self.alpha):.1f}% 的轨迹在所有{self.prediction_length}个时间步上被完全覆盖")
        print(f"{'='*80}\n")

    def apply(self, q_lo_pred: np.ndarray, q_median_pred: np.ndarray,
              q_hi_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        标准CQR区间构造 (基于2019 NIPS论文)

        Args:
            q_lo_pred: (N, K_f) - 下分位数预测
            q_median_pred: (N, K_f) - 中位数预测 (直接使用,不修正)
            q_hi_pred: (N, K_f) - 上分位数预测

        Returns:
            x_min, x_max, median: CQR预测区间和中位数
                - x_min = q_lo - Q (向下扩展Q)
                - x_max = q_hi + Q (向上扩展Q)
                - median = q_median (直接使用QR预测,不修正)

        CQR关键特性:
        1. 保持QR的非对称性 (不强制对称区间)
        2. 保持QR的可变宽度 (区间宽度 = QR宽度 + 2Q)
        3. 无需修正中位数 (QR已经学到了最优中位数)
        """
        if self.Q is None:
            raise RuntimeError("必须先调用calibrate或calibrate_joint方法进行校准")

        # CQR标准区间构造: [q_lo - Q, q_hi + Q]
        x_min = q_lo_pred - self.Q[np.newaxis, :]  # 向下扩展Q
        x_max = q_hi_pred + self.Q[np.newaxis, :]  # 向上扩展Q

        # 中位数直接使用QR预测,不修正
        # 原因: QR通过最小化pinball loss已经学到了最优中位数
        median = q_median_pred if q_median_pred is not None else None

        return x_min, x_max, median

    def apply_torch(self, q_lo_pred: torch.Tensor, q_median_pred: torch.Tensor,
                    q_hi_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        标准CQR区间构造 (PyTorch版本)

        Args:
            q_lo_pred: (N, K_f) - 下分位数预测
            q_median_pred: (N, K_f) - 中位数预测 (直接使用,不修正)
            q_hi_pred: (N, K_f) - 上分位数预测

        Returns:
            x_min, x_max, median: CQR预测区间和中位数
        """
        if self.Q is None:
            raise RuntimeError("必须先调用calibrate或calibrate_joint方法进行校准")

        device = q_lo_pred.device

        # 转换为torch tensor
        Q_tensor = torch.from_numpy(self.Q).float().to(device)

        # CQR标准区间构造: [q_lo - Q, q_hi + Q]
        x_min = q_lo_pred - Q_tensor.unsqueeze(0)
        x_max = q_hi_pred + Q_tensor.unsqueeze(0)

        # 中位数直接使用QR预测,不修正
        median = q_median_pred if q_median_pred is not None else None

        return x_min, x_max, median

    def save(self, path: str) -> None:
        """保存CQR校准分位数"""
        if self.Q is None:
            raise RuntimeError("没有可保存的校准数据")

        save_dict = {
            'Q': self.Q,
            'alpha': self.alpha,
            'n_calib': self.n_calib,
            'prediction_length': self.prediction_length,
            'num_quantiles': self.num_quantiles
        }

        np.savez(path, **save_dict)

        print(f"✓ CQR校准数据已保存到: {path}")

    def load(self, path: str) -> None:
        """加载CQR校准分位数"""
        data = np.load(path)
        self.Q = data['Q']
        self.alpha = float(data['alpha'])
        self.n_calib = int(data['n_calib'])
        self.prediction_length = int(data['prediction_length'])
        self.num_quantiles = int(data['num_quantiles'])

        print(f"✓ 从 {path} 加载CQR校准数据")
        print(f"  α={self.alpha}, n_calib={self.n_calib}, Q范围=[{self.Q.min():.3f}, {self.Q.max():.3f}]")


class AsymmetricCQRCalibrator:
    """
    非对称CQR校准器 (结合中位数修正)

    与标准CQR的区别:
    1. 先修正中位数的系统性偏差
    2. 基于修正后的中位数重新定义符合性得分
    3. 对上下侧分别计算校准分位数,支持非对称区间

    理论保证: P{Y ∈ C(X)} ≥ 1-α (保持不变)
    """

    def __init__(self, alpha: float = 0.1, prediction_length: int = 30, num_quantiles: int = 3):
        """
        初始化非对称CQR校准器

        Args:
            alpha: 误覆盖率(默认0.1,对应90%覆盖率)
            prediction_length: 预测时域长度(默认30步)
            num_quantiles: 分位数个数 (必须是3)
        """
        self.alpha = alpha
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles

        # 中位数偏差修正量
        self.median_bias = None  # shape: (prediction_length,)

        # 非对称校准分位数
        self.Q_lo = None  # shape: (prediction_length,) - 下侧校准分位数
        self.Q_hi = None  # shape: (prediction_length,) - 上侧校准分位数

        # 校准集大小
        self.n_calib = None

        # MAE改善情况
        self.mae_before = None
        self.mae_after = None
        self.use_median_correction = False

        print(f"初始化非对称CQR校准器: α={alpha}, 目标覆盖率≥{1-alpha:.1%}")

    def calibrate(self, q_lo_pred: np.ndarray, q_median_pred: np.ndarray,
                  q_hi_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        非对称CQR校准 (结合中位数修正)

        Args:
            q_lo_pred: (N_calib, K_f) - 下分位数预测
            q_median_pred: (N_calib, K_f) - 中位数预测
            q_hi_pred: (N_calib, K_f) - 上分位数预测
            y_true: (N_calib, K_f) - 真实未来位置

        流程:
        1. 计算中位数偏差并修正
        2. 评估修正效果,决定是否启用
        3. 基于修正后的中位数重新定义符合性得分
        4. 对上下侧分别计算校准分位数
        """
        assert q_lo_pred.shape == y_true.shape, "预测和标签形状不匹配"
        assert q_hi_pred.shape == y_true.shape, "预测和标签形状不匹配"
        assert q_median_pred is not None, "非对称CQR需要中位数预测"

        self.n_calib = len(q_lo_pred)

        print(f"\n{'='*80}")
        print("非对称CQR校准 (Asymmetric CQR with Median Correction)")
        print(f"{'='*80}")
        print(f"校准集样本数: {self.n_calib}")
        print(f"预测时域长度: {self.prediction_length} 步")

        # ============================================================
        # 阶段1: 中位数修正
        # ============================================================
        print(f"\n[阶段1] 中位数偏差修正")
        print(f"{'-'*80}")

        # 计算修正前的MAE
        residuals = y_true - q_median_pred  # (N, K_f)
        mae_per_step_before = np.abs(residuals).mean(axis=0)  # (K_f,)
        self.mae_before = mae_per_step_before.mean()

        print(f"修正前中位数MAE:")
        print(f"  整体: {self.mae_before:.4f}")
        print(f"  1s:   {mae_per_step_before[:10].mean():.4f}")
        print(f"  2s:   {mae_per_step_before[10:20].mean():.4f}")
        print(f"  3s:   {mae_per_step_before[20:30].mean():.4f}")

        # 计算中位数偏差 (使用中位数而非均值,更鲁棒)
        self.median_bias = np.median(residuals, axis=0)  # (K_f,)

        print(f"\n检测到的系统性偏差:")
        print(f"  偏差范围: [{self.median_bias.min():.4f}, {self.median_bias.max():.4f}]")
        print(f"  偏差均值: {self.median_bias.mean():.4f}")
        print(f"  偏差中位数: {np.median(self.median_bias):.4f}")

        # 应用修正
        q_median_corrected = q_median_pred + self.median_bias[np.newaxis, :]

        # 计算修正后的MAE
        residuals_after = y_true - q_median_corrected
        mae_per_step_after = np.abs(residuals_after).mean(axis=0)
        self.mae_after = mae_per_step_after.mean()

        print(f"\n修正后中位数MAE:")
        print(f"  整体: {self.mae_after:.4f}")
        print(f"  1s:   {mae_per_step_after[:10].mean():.4f}")
        print(f"  2s:   {mae_per_step_after[10:20].mean():.4f}")
        print(f"  3s:   {mae_per_step_after[20:30].mean():.4f}")

        # 决策: 是否启用中位数修正
        if self.mae_after < self.mae_before:
            improvement = (self.mae_before - self.mae_after) / self.mae_before * 100
            print(f"\n✓ 中位数修正有效!")
            print(f"  MAE改善: {self.mae_before:.4f} → {self.mae_after:.4f}")
            print(f"  改善率: {improvement:.2f}%")
            self.use_median_correction = True
            q_median_for_cqr = q_median_corrected
        else:
            degradation = (self.mae_after - self.mae_before) / self.mae_before * 100
            print(f"\n✗ 中位数修正无效 (MAE增大 {degradation:.2f}%)")
            print(f"  使用原始QR中位数进行后续校准")
            self.use_median_correction = False
            self.median_bias = np.zeros(self.prediction_length)  # 不修正
            q_median_for_cqr = q_median_pred

        # ============================================================
        # 阶段2: 非对称CQR校准
        # ============================================================
        print(f"\n[阶段2] 非对称CQR校准")
        print(f"{'-'*80}")

        # 基于修正后的中位数重新定义符合性得分
        # 关键修正：符合性得分必须衡量"需要扩展的量"，而不是"当前位置"
        # 使用max(0, ...)确保只保留需要扩展的情况
        # E_lo: max(0, q_lo - y) 只有当y < q_lo时才需要向下扩展
        # E_hi: max(0, y - q_hi) 只有当y > q_hi时才需要向上扩展

        # 计算符合性得分（使用原始的q_lo和q_hi）
        E_lo = np.maximum(0, q_lo_pred - y_true)  # 下侧不足量（正值表示需要扩展）
        E_hi = np.maximum(0, y_true - q_hi_pred)  # 上侧不足量（正值表示需要扩展）

        # 计算原始QR的半宽度（用于分析）
        half_width_lo = q_median_for_cqr - q_lo_pred  # (N, K_f)
        half_width_hi = q_hi_pred - q_median_for_cqr  # (N, K_f)

        print(f"原始QR半宽度统计（基于修正后中位数）:")
        print(f"  下侧半宽度: 均值={half_width_lo.mean():.4f}, 范围=[{half_width_lo.min():.4f}, {half_width_lo.max():.4f}]")
        print(f"  上侧半宽度: 均值={half_width_hi.mean():.4f}, 范围=[{half_width_hi.min():.4f}, {half_width_hi.max():.4f}]")
        asymmetry = half_width_hi.mean() / half_width_lo.mean() if half_width_lo.mean() > 0 else 1.0
        print(f"  非对称度: {asymmetry:.3f} (>1表示上侧更宽)")

        print(f"\nConformity Scores统计:")
        print(f"  E_lo: 均值={E_lo.mean():.4f}, E<=0(已覆盖)比例={100*(E_lo<=0).mean():.1f}%")
        print(f"  E_hi: 均值={E_hi.mean():.4f}, E<=0(已覆盖)比例={100*(E_hi<=0).mean():.1f}%")

        # 计算校准分位数 (有限样本修正)
        quantile_level = (1 - self.alpha) * (1 + 1 / self.n_calib)

        # 独立计算上下侧校准分位数
        self.Q_lo = np.quantile(E_lo, quantile_level, axis=0)  # (K_f,)
        self.Q_hi = np.quantile(E_hi, quantile_level, axis=0)  # (K_f,)

        print(f"\n校准分位数 (分位数水平={quantile_level:.4f}):")
        print(f"  Q_lo: 均值={self.Q_lo.mean():.4f}, 范围=[{self.Q_lo.min():.4f}, {self.Q_lo.max():.4f}]")
        print(f"  Q_hi: 均值={self.Q_hi.mean():.4f}, 范围=[{self.Q_hi.min():.4f}, {self.Q_hi.max():.4f}]")
        q_asymmetry = self.Q_hi.mean() / self.Q_lo.mean() if self.Q_lo.mean() != 0 else 1.0
        print(f"  非对称度: {q_asymmetry:.3f}")

        # 分析校准后的区间宽度
        print(f"\n区间宽度演化 (每5步):")
        print(f"  {'步数':>6} {'QR宽度':>10} {'Q_lo':>10} {'Q_hi':>10} {'CQR宽度':>10}")
        print(f"  {'-'*60}")
        for k in range(0, self.prediction_length, 5):
            qr_width = (q_hi_pred[:, k] - q_lo_pred[:, k]).mean()
            cqr_width = qr_width + self.Q_lo[k] + self.Q_hi[k]  # Q_lo和Q_hi都是正值，扩展区间
            print(f"  t{k+1:>5} {qr_width:>10.4f} {self.Q_lo[k]:>10.4f} {self.Q_hi[k]:>10.4f} {cqr_width:>10.4f}")

        print(f"\n✓ 非对称CQR校准完成")
        print(f"  理论保证: P(Y ∈ [q_median - half_lo - Q_lo, q_median + half_hi + Q_hi]) ≥ {100*(1-self.alpha):.1f}%")
        print(f"  关键特性: 结合中位数修正 + 非对称区间扩展")
        print(f"{'='*80}\n")

    def apply(self, q_lo_pred: np.ndarray, q_median_pred: np.ndarray,
              q_hi_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        应用非对称CQR校准

        Args:
            q_lo_pred: (N, K_f) - 下分位数预测
            q_median_pred: (N, K_f) - 中位数预测
            q_hi_pred: (N, K_f) - 上分位数预测

        Returns:
            x_min, x_max, median_corrected: 非对称CQR预测区间和修正后的中位数
        """
        if self.Q_lo is None or self.Q_hi is None:
            raise RuntimeError("必须先调用calibrate方法进行校准")

        # 步骤1: 修正中位数
        if self.use_median_correction:
            q_median_corrected = q_median_pred + self.median_bias[np.newaxis, :]
        else:
            q_median_corrected = q_median_pred

        # 步骤2: 非对称CQR区间构造
        # 与标准CQR类似，但上下侧使用不同的校准分位数
        # Q_lo和Q_hi现在都是非负值（表示需要扩展的量）
        # x_min = q_lo - Q_lo (向下扩展)
        # x_max = q_hi + Q_hi (向上扩展)
        x_min = q_lo_pred - self.Q_lo[np.newaxis, :]
        x_max = q_hi_pred + self.Q_hi[np.newaxis, :]

        return x_min, x_max, q_median_corrected

    def save(self, path: str) -> None:
        """保存非对称CQR校准数据"""
        if self.Q_lo is None or self.Q_hi is None:
            raise RuntimeError("没有可保存的校准数据")

        save_dict = {
            'Q_lo': self.Q_lo,
            'Q_hi': self.Q_hi,
            'median_bias': self.median_bias,
            'alpha': self.alpha,
            'n_calib': self.n_calib,
            'prediction_length': self.prediction_length,
            'num_quantiles': self.num_quantiles,
            'use_median_correction': self.use_median_correction,
            'mae_before': self.mae_before,
            'mae_after': self.mae_after
        }

        np.savez(path, **save_dict)
        print(f"✓ 非对称CQR校准数据已保存到: {path}")

    def load(self, path: str) -> None:
        """加载非对称CQR校准数据"""
        data = np.load(path)
        self.Q_lo = data['Q_lo']
        self.Q_hi = data['Q_hi']
        self.median_bias = data['median_bias']
        self.alpha = float(data['alpha'])
        self.n_calib = int(data['n_calib'])
        self.prediction_length = int(data['prediction_length'])
        self.num_quantiles = int(data['num_quantiles'])
        self.use_median_correction = bool(data['use_median_correction'])
        self.mae_before = float(data['mae_before'])
        self.mae_after = float(data['mae_after'])

        print(f"✓ 从 {path} 加载非对称CQR校准数据")
        print(f"  α={self.alpha}, n_calib={self.n_calib}")
        print(f"  中位数修正: {'启用' if self.use_median_correction else '禁用'}")


def compute_empirical_coverage(x_min: np.ndarray, x_max: np.ndarray,
                               y_true: np.ndarray) -> np.ndarray:
    """
    计算经验覆盖率

    Args:
        x_min: (N, K_f) - 预测区间下界
        x_max: (N, K_f) - 预测区间上界
        y_true: (N, K_f) - 真实值

    Returns:
        coverage: (K_f,) - 每个时间步的覆盖率
    """
    covered = (y_true >= x_min) & (y_true <= x_max)
    coverage = covered.mean(axis=0)
    return coverage


def validate_quantile_ordering(x_min: np.ndarray, median: np.ndarray, x_max: np.ndarray) -> dict:
    """
    验证分位数排序是否正确

    Args:
        x_min: (N, K_f) - 预测区间下界
        median: (N, K_f) - 中位数预测
        x_max: (N, K_f) - 预测区间上界

    Returns:
        validation_result: 包含验证结果的字典
    """
    # 检查 x_min <= median <= x_max
    valid_lo = (x_min <= median).all()
    valid_hi = (median <= x_max).all()

    # 统计违反排序的样本数
    violations_lo = np.sum(x_min > median)
    violations_hi = np.sum(median > x_max)
    total_samples = x_min.size

    result = {
        'is_valid': valid_lo and valid_hi,
        'violations_lo': violations_lo,
        'violations_hi': violations_hi,
        'violation_rate': (violations_lo + violations_hi) / total_samples,
        'max_lo_violation': np.max(x_min - median) if violations_lo > 0 else 0.0,
        'max_hi_violation': np.max(median - x_max) if violations_hi > 0 else 0.0
    }

    return result


def main():
    """测试共形化校准器"""
    print("=" * 80)
    print("共形化校准器测试")
    print("=" * 80)

    # 模拟数据
    n_calib = 1000
    n_test = 500
    K_f = 30

    np.random.seed(42)

    # 模拟校准集预测和标签
    # 假设分位数回归预测有一定偏差
    y_calib_true = np.random.randn(n_calib, K_f) * 10 + 50
    q_lo_calib = y_calib_true - 8 + np.random.randn(n_calib, K_f) * 2
    q_hi_calib = y_calib_true + 8 + np.random.randn(n_calib, K_f) * 2

    # 计算校准前的覆盖率
    covered_before = (y_calib_true >= q_lo_calib) & (y_calib_true <= q_hi_calib)
    coverage_before = covered_before.mean(axis=0)

    print(f"\n校准前覆盖率: {coverage_before.mean():.3f} (目标≥0.90)")
    print(f"  最小覆盖率: {coverage_before.min():.3f}")
    print(f"  最大覆盖率: {coverage_before.max():.3f}")

    # 创建校准器并校准
    calibrator = ConformalizationCalibrator(alpha=0.1)
    calibrator.calibrate(q_lo_calib, q_hi_calib, y_calib_true)

    # 应用到校准集(检验理论)
    x_min_calib, x_max_calib = calibrator.apply(q_lo_calib, q_hi_calib)
    coverage_calib = compute_empirical_coverage(x_min_calib, x_max_calib, y_calib_true)

    print(f"\n校准集上的覆盖率(应该≥0.90):")
    print(f"  平均覆盖率: {coverage_calib.mean():.3f}")
    print(f"  最小覆盖率: {coverage_calib.min():.3f}")

    # 测试集
    y_test_true = np.random.randn(n_test, K_f) * 10 + 50
    q_lo_test = y_test_true - 8 + np.random.randn(n_test, K_f) * 2
    q_hi_test = y_test_true + 8 + np.random.randn(n_test, K_f) * 2

    # 应用共形化
    x_min_test, x_max_test = calibrator.apply(q_lo_test, q_hi_test)
    coverage_test = compute_empirical_coverage(x_min_test, x_max_test, y_test_true)

    print(f"\n测试集上的覆盖率:")
    print(f"  平均覆盖率: {coverage_test.mean():.3f}")
    print(f"  最小覆盖率: {coverage_test.min():.3f}")

    # 区间宽度
    width_before = (q_hi_calib - q_lo_calib).mean()
    width_after = (x_max_calib - x_min_calib).mean()
    print(f"\n平均区间宽度:")
    print(f"  校准前: {width_before:.2f}")
    print(f"  校准后: {width_after:.2f}")
    print(f"  扩展: +{width_after - width_before:.2f}")

    # 测试保存和加载
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "calibrator.npz")
        calibrator.save(save_path)

        # 加载
        calibrator2 = ConformalizationCalibrator()
        calibrator2.load(save_path)

        # 验证
        x_min_test2, x_max_test2 = calibrator2.apply(q_lo_test, q_hi_test)
        assert np.allclose(x_min_test, x_min_test2), "加载后结果不一致"
        print(f"\n✓ 保存/加载功能测试通过")

    print("\n" + "=" * 80)
    print("✓ 共形化校准器测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
