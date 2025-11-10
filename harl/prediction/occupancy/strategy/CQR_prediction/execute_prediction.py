#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CQR轨迹占用预测执行模块
用于加载训练好的QR/CQR模型并进行实时轨迹预测

功能：
1. 加载QR模型（GRU-based Quantile Regression）
2. 加载CQR校准器（Conformal Quantile Regression calibration）
3. 执行实时轨迹占用预测
4. 提供批量推理和单样本推理接口

支持的模型：
- QR-GRU-uncertainty: 仅QR预测
- QR-GRU-uncertainty_pcgrad: 仅QR预测（带梯度投影）
- CQR-GRU-uncertainty: QR + CQR校准
- CQR-GRU-uncertainty_pcgrad: QR + CQR校准（带梯度投影）

作者：交通流研究团队
日期：2025
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Handle both package import and standalone execution
try:
    from .models import GRUQR
    from .conformalization import ConformalizationCalibrator
except ImportError:
    from models import GRUQR
    from conformalization import ConformalizationCalibrator


class CQRPredictor:
    """
    CQR轨迹占用预测器
    用于加载和执行训练好的QR/CQR模型进行实时预测
    """

    def __init__(self,
                 model_type: str = "QR-GRU-uncertainty",
                 model_base_path: Optional[str] = None,
                 use_conformal: bool = False,
                 calibrator_type: str = "standard",
                 device: str = "cpu"):
        """
        初始化CQR预测器

        Args:
            model_type: 模型类型
                - "QR-GRU-uncertainty": 基础QR模型
                - "QR-GRU-uncertainty_pcgrad": QR模型（梯度投影）
                - "CQR-GRU-uncertainty": CQR模型（需读取QR+校准器）
                - "CQR-GRU-uncertainty_pcgrad": CQR模型（梯度投影+校准器）
            model_base_path: 模型文件夹基础路径
            use_conformal: 是否使用共形化校准（自动检测）
            calibrator_type: 校准器类型 ("standard" 或 "asymmetric")
            device: 运行设备，"cpu" 或 "cuda"
        """
        self.model_type = model_type
        self.device = torch.device(device)
        self.calibrator_type = calibrator_type

        # 设置模型路径
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent / "models"

        # 自动判断是否使用CQR
        self.use_conformal = model_type.startswith("CQR-") or use_conformal

        if self.use_conformal:
            # CQR模型：需要读取QR子模型 + 校准器
            self.model_path = Path(model_base_path) / "CQR_models" / model_type
            self.qr_model_path = self.model_path / "00-QR-GRU"
        else:
            # QR模型：直接读取
            self.model_path = Path(model_base_path) / "QR_models" / model_type
            self.qr_model_path = self.model_path

        if not self.qr_model_path.exists():
            raise FileNotFoundError(f"QR模型路径不存在: {self.qr_model_path}")

        # 初始化模型和校准器
        self.model = None
        self.calibrator = None
        self.model_config = None

        # 加载模型
        self._load_qr_model()

        # 加载CQR校准器（如果需要）
        if self.use_conformal:
            self._load_calibrator()

        print(f"[CQRPredictor] 成功初始化预测器")
        print(f"[CQRPredictor] 模型类型: {model_type}")
        print(f"[CQRPredictor] 使用共形化: {self.use_conformal}")
        if self.use_conformal:
            print(f"[CQRPredictor] 校准器类型: {self.calibrator_type}")
        print(f"[CQRPredictor] 运行设备: {self.device}")

    def _load_qr_model(self):
        """加载QR模型参数"""
        model_file = self.qr_model_path / "best_model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        print(f"[CQRPredictor] 加载QR模型: {model_file}")

        # 加载checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # 获取模型配置
        if 'hyperparameters' in checkpoint:
            self.model_config = checkpoint['hyperparameters']
        else:
            # 使用默认配置
            self.model_config = {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.1,
                'prediction_length': 30,
                'num_quantiles': 3,
                'time_encoding_dim': 32
            }

        # 创建模型实例
        self.model = GRUQR(
            hidden_dim=self.model_config.get('hidden_dim', 128),
            num_layers=self.model_config.get('num_layers', 2),
            dropout=self.model_config.get('dropout', 0.1),
            prediction_length=self.model_config.get('prediction_length', 30),
            use_attention=self.model_config.get('use_attention', True),
            num_quantiles=self.model_config.get('num_quantiles', 3),
            time_encoding_dim=self.model_config.get('time_encoding_dim', 32)
        )

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"[CQRPredictor] QR模型加载完成")
        print(f"  - 隐藏维度: {self.model_config.get('hidden_dim')}")
        print(f"  - GRU层数: {self.model_config.get('num_layers')}")
        print(f"  - 预测长度: {self.model_config.get('prediction_length')} 步")

    def _load_calibrator(self):
        """加载CQR校准器"""
        if self.calibrator_type == "standard":
            calibrator_file = self.model_path / "calibrator_standard.npz"
        elif self.calibrator_type == "asymmetric":
            calibrator_file = self.model_path / "calibrator_asymmetric.npz"
        else:
            calibrator_file = self.model_path / "calibrator.npz"

        if not calibrator_file.exists():
            print(f"[WARNING] 校准器文件不存在: {calibrator_file}")
            print(f"[WARNING] 将不使用共形化校准")
            self.use_conformal = False
            return

        print(f"[CQRPredictor] 加载CQR校准器: {calibrator_file}")

        # 创建校准器实例
        self.calibrator = ConformalizationCalibrator(
            alpha=0.1,
            prediction_length=self.model_config.get('prediction_length', 30),
            num_quantiles=self.model_config.get('num_quantiles', 3)
        )

        # 加载校准参数
        self.calibrator.load(str(calibrator_file))

        print(f"[CQRPredictor] CQR校准器加载完成")
        print(f"  - 目标覆盖率: ≥ {(1-self.calibrator.alpha)*100:.0f}%")
        print(f"  - 校准样本数: {self.calibrator.n_calib}")

    def predict_qr(self,
                   features: Dict[str, Union[np.ndarray, torch.Tensor]],
                   return_tensors: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行QR预测（仅分位数回归）

        Args:
            features: 特征字典，包含:
                - 'traffic_state': [batch, 6] 交通状态
                - 'ego_current': [batch, 4] 自车当前状态
                - 'ego_history': [batch, 20, 5] 自车历史轨迹
                - 'sur_current': [batch, 6, 8] 周边车辆当前状态
                - 'sur_history': [batch, 6, 20, 5] 周边车辆历史轨迹
                - 'intention': [batch, 1] 横向意图
                - 'initial_state': [batch, 3] 初始状态 [x, v, a]
            return_tensors: 是否返回torch张量（默认返回numpy）

        Returns:
            q_lo: [batch, 30] 下分位数（5th percentile）
            q_median: [batch, 30] 中位数（50th percentile）
            q_hi: [batch, 30] 上分位数（95th percentile）
        """
        # 转换为tensor
        features_tensor = self._prepare_features(features)

        # 推理
        with torch.no_grad():
            q_lo, q_median, q_hi = self.model(
                features_tensor,
                y_true=None,
                teacher_forcing_ratio=0.0
            )

        if return_tensors:
            return q_lo, q_median, q_hi
        else:
            return q_lo.cpu().numpy(), q_median.cpu().numpy(), q_hi.cpu().numpy()

    def predict_cqr(self,
                    features: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行CQR预测（分位数回归 + 共形化校准）

        Args:
            features: 特征字典（同predict_qr）

        Returns:
            x_min: [batch, 30] 校准后的下界
            median: [batch, 30] 中位数（可能经过校正）
            x_max: [batch, 30] 校准后的上界
        """
        if not self.use_conformal or self.calibrator is None:
            raise ValueError("CQR校准器未加载，请使用predict_qr或初始化时指定use_conformal=True")

        # 先执行QR预测
        q_lo, q_median, q_hi = self.predict_qr(features, return_tensors=False)

        # 应用CQR校准
        x_min, x_max, median = self.calibrator.apply(q_lo, q_median, q_hi)

        return x_min, median, x_max

    def predict(self,
                features: Dict[str, Union[np.ndarray, torch.Tensor]],
                return_intervals: bool = True) -> Union[
                    Tuple[np.ndarray, np.ndarray, np.ndarray],
                    np.ndarray
                ]:
        """
        统一预测接口（自动选择QR或CQR）

        Args:
            features: 特征字典
            return_intervals: 是否返回区间（默认True）

        Returns:
            如果return_intervals=True:
                - 对于QR: (q_lo, q_median, q_hi)
                - 对于CQR: (x_min, median, x_max)
            如果return_intervals=False:
                - 仅返回中位数预测: median [batch, 30]
        """
        if self.use_conformal:
            x_min, median, x_max = self.predict_cqr(features)
            if return_intervals:
                return x_min, median, x_max
            else:
                return median
        else:
            q_lo, q_median, q_hi = self.predict_qr(features, return_tensors=False)
            if return_intervals:
                return q_lo, q_median, q_hi
            else:
                return q_median

    def predict_single(self,
                      traffic_state: Union[np.ndarray, List],
                      ego_current: Union[np.ndarray, List],
                      ego_history: Union[np.ndarray, List],
                      sur_current: Union[np.ndarray, List],
                      sur_history: Union[np.ndarray, List],
                      intention: float = 0.0) -> Dict[str, np.ndarray]:
        """
        单样本预测（便捷接口）

        Args:
            traffic_state: [6] 交通状态 [EL_v, EL_k, LL_v, LL_k, RL_v, RL_k]
            ego_current: [4] 自车状态 [v, a, lane_id, eta]
            ego_history: [20, 5] 自车历史 [Δx, Δy, Δv, Δa, lane_id]
            sur_current: [6, 8] 周边车辆当前状态
            sur_history: [6, 20, 5] 周边车辆历史轨迹
            intention: 横向意图 (0或1)

        Returns:
            结果字典:
                - 'median': [30] 中位数预测
                - 'lower': [30] 下界
                - 'upper': [30] 上界
                - 'interval_width': [30] 区间宽度
        """
        # 构造特征字典
        features = {
            'traffic_state': np.array(traffic_state).reshape(1, -1),
            'ego_current': np.array(ego_current).reshape(1, -1),
            'ego_history': np.array(ego_history).reshape(1, 20, 5),
            'sur_current': np.array(sur_current).reshape(1, 6, 8),
            'sur_history': np.array(sur_history).reshape(1, 6, 20, 5),
            'intention': np.array([[intention]])
        }

        # 添加initial_state
        features['initial_state'] = features['ego_current'][:, :3]  # [x, v, a]

        # 执行预测
        lower, median, upper = self.predict(features, return_intervals=True)

        return {
            'median': median[0],
            'lower': lower[0],
            'upper': upper[0],
            'interval_width': upper[0] - lower[0]
        }

    def _prepare_features(self, features: Dict) -> Dict[str, torch.Tensor]:
        """准备输入特征（转换为tensor并移动到设备）"""
        features_tensor = {}

        for key, value in features.items():
            if isinstance(value, np.ndarray):
                features_tensor[key] = torch.FloatTensor(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                features_tensor[key] = value.to(self.device)
            else:
                features_tensor[key] = torch.FloatTensor(np.array(value)).to(self.device)

        # 确保有initial_state
        if 'initial_state' not in features_tensor:
            # 从ego_current提取 [x, v, a]
            # 注意：ego_current是 [v, a, lane_id, eta]，需要重新组织
            # 假设x从历史轨迹的最后一个位置获取
            if 'ego_history' in features_tensor:
                last_x = features_tensor['ego_history'][:, -1, 0]  # 最后一个Δx
            else:
                last_x = torch.zeros(features_tensor['ego_current'].size(0), device=self.device)

            v = features_tensor['ego_current'][:, 0]
            a = features_tensor['ego_current'][:, 1]
            features_tensor['initial_state'] = torch.stack([last_x, v, a], dim=1)

        return features_tensor

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'model_type': self.model_type,
            'use_conformal': self.use_conformal,
            'device': str(self.device),
            'model_config': self.model_config
        }

        if self.calibrator:
            info['calibrator'] = {
                'type': self.calibrator_type,
                'alpha': self.calibrator.alpha,
                'target_coverage': f">= {(1-self.calibrator.alpha)*100:.0f}%",
                'n_calib': self.calibrator.n_calib
            }

        return info


class CQRPredictorFactory:
    """
    CQR预测器工厂类
    用于统一管理和创建不同类型的预测器
    """

    _instances = {}  # 单例缓存

    @classmethod
    def create_predictor(cls,
                        model_type: str = "QR-GRU-uncertainty",
                        model_base_path: Optional[str] = None,
                        calibrator_type: str = "standard",
                        device: str = "cpu",
                        use_cache: bool = True) -> CQRPredictor:
        """
        创建预测器（支持单例模式）

        Args:
            model_type: 模型类型
                - "QR-GRU-uncertainty"
                - "QR-GRU-uncertainty_pcgrad"
                - "CQR-GRU-uncertainty"
                - "CQR-GRU-uncertainty_pcgrad"
            model_base_path: 模型基础路径
            calibrator_type: 校准器类型 ("standard" 或 "asymmetric")
            device: 运行设备
            use_cache: 是否使用缓存的实例

        Returns:
            CQRPredictor实例
        """
        cache_key = f"{model_type}_{calibrator_type}_{device}"

        if use_cache and cache_key in cls._instances:
            print(f"[CQRPredictorFactory] 使用缓存的预测器: {cache_key}")
            return cls._instances[cache_key]

        # 判断是否使用CQR
        use_conformal = model_type.startswith("CQR-")

        predictor = CQRPredictor(
            model_type=model_type,
            model_base_path=model_base_path,
            use_conformal=use_conformal,
            calibrator_type=calibrator_type,
            device=device
        )

        if use_cache:
            cls._instances[cache_key] = predictor

        return predictor

    @classmethod
    def get_available_models(cls, model_base_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        获取可用的模型列表

        Args:
            model_base_path: 模型基础路径

        Returns:
            字典: {'QR': [...], 'CQR': [...]}
        """
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent / "models"

        model_base_path = Path(model_base_path)

        available = {'QR': [], 'CQR': []}

        # 检查QR模型
        qr_path = model_base_path / "QR_models"
        if qr_path.exists():
            for item in qr_path.iterdir():
                if item.is_dir() and (item / "best_model.pt").exists():
                    available['QR'].append(item.name)

        # 检查CQR模型
        cqr_path = model_base_path / "CQR_models"
        if cqr_path.exists():
            for item in cqr_path.iterdir():
                if item.is_dir() and (item / "00-QR-GRU" / "best_model.pt").exists():
                    available['CQR'].append(item.name)

        return available


def demo_prediction():
    """演示如何使用预测器"""
    print("=" * 80)
    print("CQR轨迹占用预测演示")
    print("=" * 80)

    # 查看可用模型
    print("\n1. 查看可用模型")
    print("-" * 80)
    available = CQRPredictorFactory.get_available_models()
    print(f"QR模型: {available['QR']}")
    print(f"CQR模型: {available['CQR']}")

    # 创建QR预测器
    print("\n2. 创建QR预测器")
    print("-" * 80)
    qr_predictor = CQRPredictorFactory.create_predictor(
        model_type="QR-GRU-uncertainty",
        device="cpu"
    )

    # 准备测试数据（随机生成，实际使用时应该是真实数据）
    np.random.seed(42)
    batch_size = 2

    features = {
        'traffic_state': np.random.randn(batch_size, 6).astype(np.float32),
        'ego_current': np.random.randn(batch_size, 4).astype(np.float32),
        'ego_history': np.random.randn(batch_size, 20, 5).astype(np.float32),
        'sur_current': np.random.randn(batch_size, 6, 8).astype(np.float32),
        'sur_history': np.random.randn(batch_size, 6, 20, 5).astype(np.float32),
        'intention': np.array([[0], [1]], dtype=np.float32)
    }

    # 3. QR预测
    print("\n3. QR批量预测")
    print("-" * 80)
    q_lo, q_median, q_hi = qr_predictor.predict(features, return_intervals=True)
    print(f"预测形状: q_lo={q_lo.shape}, q_median={q_median.shape}, q_hi={q_hi.shape}")
    print(f"样本1 - 第1秒中位数: {q_median[0, :10].mean():.3f}")
    print(f"样本1 - 第3秒中位数: {q_median[0, 20:30].mean():.3f}")
    print(f"样本1 - 区间宽度: {(q_hi[0] - q_lo[0]).mean():.3f}")

    # 4. 创建CQR预测器
    print("\n4. 创建CQR预测器")
    print("-" * 80)
    cqr_predictor = CQRPredictorFactory.create_predictor(
        model_type="CQR-GRU-uncertainty",
        calibrator_type="standard",
        device="cpu"
    )

    # 5. CQR预测
    print("\n5. CQR批量预测")
    print("-" * 80)
    x_min, median, x_max = cqr_predictor.predict(features, return_intervals=True)
    print(f"预测形状: x_min={x_min.shape}, median={median.shape}, x_max={x_max.shape}")
    print(f"样本1 - 第1秒中位数: {median[0, :10].mean():.3f}")
    print(f"样本1 - 第3秒中位数: {median[0, 20:30].mean():.3f}")
    print(f"样本1 - 校准后区间宽度: {(x_max[0] - x_min[0]).mean():.3f}")

    # 6. 对比QR和CQR
    print("\n6. QR vs CQR 对比")
    print("-" * 80)
    qr_width = (q_hi[0] - q_lo[0]).mean()
    cqr_width = (x_max[0] - x_min[0]).mean()
    print(f"QR区间宽度: {qr_width:.3f}")
    print(f"CQR区间宽度: {cqr_width:.3f}")
    print(f"区间扩展: {cqr_width - qr_width:.3f} (+{(cqr_width/qr_width-1)*100:.1f}%)")

    # 7. 模型信息
    print("\n7. 模型信息")
    print("-" * 80)
    info = cqr_predictor.get_model_info()
    print(f"模型类型: {info['model_type']}")
    print(f"使用共形化: {info['use_conformal']}")
    if 'calibrator' in info:
        print(f"目标覆盖率: {info['calibrator']['target_coverage']}")
        print(f"校准样本数: {info['calibrator']['n_calib']}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_prediction()
