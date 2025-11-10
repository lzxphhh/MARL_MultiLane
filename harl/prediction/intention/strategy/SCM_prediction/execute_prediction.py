#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM意图预测执行模块
用于加载训练好的模型参数并进行实时意图预测

功能：
1. 加载预训练的SCM模型（支持medium_hierarchical和shallow_hierarchical）
2. 处理MARL环境的实时观测数据
3. 生成周边车辆的换道意图预测
4. 提供批量推理和单样本推理接口

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
    from .scm_model_v2 import HierarchicalSCMModelV2, PathArchitecture, FusionStrategy
except ImportError:
    from scm_model_v2 import HierarchicalSCMModelV2, PathArchitecture, FusionStrategy


class SCMPredictor:
    """
    SCM意图预测器
    用于加载和执行训练好的SCM模型进行实时预测
    """

    def __init__(self,
                 model_type: str = "shallow_hierarchical",
                 model_base_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        初始化SCM预测器

        Args:
            model_type: 模型类型，可选 "shallow_hierarchical" 或 "medium_hierarchical"
            model_base_path: 模型文件夹基础路径，默认为当前模块的../../models/SCM_models/
            device: 运行设备，"cpu" 或 "cuda"
        """
        self.model_type = model_type
        self.device = torch.device(device)

        # 设置模型路径
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent / "models" / "SCM_models"

        self.model_path = Path(model_base_path) / model_type

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        # 根据模型类型确定架构
        if "shallow" in model_type:
            self.path_architecture = PathArchitecture.SHALLOW
        elif "medium" in model_type:
            self.path_architecture = PathArchitecture.MEDIUM
        elif "deep" in model_type:
            self.path_architecture = PathArchitecture.DEEP
        else:
            self.path_architecture = PathArchitecture.LINEAR

        # 融合策略（从模型类型推断）
        if "hierarchical" in model_type:
            self.fusion_strategy = FusionStrategy.HIERARCHICAL
        elif "gated" in model_type:
            self.fusion_strategy = FusionStrategy.GATED
        elif "weighted" in model_type:
            self.fusion_strategy = FusionStrategy.WEIGHTED_SUM
        elif "product" in model_type:
            self.fusion_strategy = FusionStrategy.PRODUCT
        elif "minimum" in model_type:
            self.fusion_strategy = FusionStrategy.MINIMUM
        else:
            self.fusion_strategy = FusionStrategy.HIERARCHICAL

        # 初始化模型
        self.model = None
        self.feature_stats = None

        # 加载模型
        self._load_model()

        print(f"[SCMPredictor] 成功加载模型: {model_type}")
        print(f"[SCMPredictor] 路径架构: {self.path_architecture.value}")
        print(f"[SCMPredictor] 融合策略: {self.fusion_strategy.value}")
        print(f"[SCMPredictor] 运行设备: {self.device}")

    def _load_model(self):
        """加载模型参数"""
        # 加载模型checkpoint
        model_file = self.model_path / "best_model.pth"
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        checkpoint = torch.load(model_file, map_location=self.device)

        # 创建模型实例
        self.model = HierarchicalSCMModelV2(
            env_dim=4,
            ind_dim=10,
            path_architecture=self.path_architecture,
            fusion_strategy=self.fusion_strategy,
            use_vehicle_type=True,
            enforce_causal_constraints=True
        )

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 尝试加载因果参数（用于验证）
        causal_params_file = self.model_path / "causal_parameters.json"
        if causal_params_file.exists():
            with open(causal_params_file, 'r') as f:
                self.causal_params = json.load(f)
        else:
            self.causal_params = None

        print(f"[SCMPredictor] 模型参数加载完成")
        if self.causal_params:
            print(f"[SCMPredictor] 环境层权重: {self.causal_params['alpha_env']}")

    def normalize_features(self,
                          env_features: np.ndarray,
                          ind_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化特征（如果有预存的统计信息）

        注意：当前版本假设输入已经归一化，如果需要归一化，需要提供训练时的统计信息

        Args:
            env_features: [N, 4] 环境特征
            ind_features: [N, 10] 个体特征

        Returns:
            归一化后的特征
        """
        # TODO: 如果需要，从训练数据中加载归一化统计信息
        # 当前假设输入已经归一化
        return env_features, ind_features

    def predict(self,
                env_features: Union[np.ndarray, torch.Tensor],
                ind_features: Union[np.ndarray, torch.Tensor],
                vehicle_types: Optional[Union[np.ndarray, torch.Tensor]] = None,
                return_intermediates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        执行意图预测

        Args:
            env_features: [N, 4] 环境特征
                - v_avg_norm: 平均速度（归一化）
                - kappa_avg_norm: 平均密度（归一化）
                - delta_v_norm: 车道速度差（归一化）
                - delta_kappa_norm: 车道密度差（归一化）

            ind_features: [N, 10] 个体特征（归一化）
                - v_ego_norm: 自车速度
                - v_rel_norm: 与前车相对速度
                - d_headway_norm: 车头间距
                - T_headway_norm: 时间间距
                - delta_v_adj_front_norm: 与目标车道前车速度差
                - d_adj_front_norm: 与目标车道前车距离
                - T_adj_front_norm: 与目标车道前车时间间隙
                - delta_v_adj_rear_norm: 与目标车道后车速度差
                - d_adj_rear_norm: 与目标车道后车距离
                - T_adj_rear_norm: 与目标车道后车时间间隙

            vehicle_types: [N, 1] 车辆类型 (0=HDV, 1=CAV)，可选
            return_intermediates: 是否返回中间结果

        Returns:
            predictions: [N, 1] 换道意图概率
            intermediates: (可选) 中间结果字典
        """
        # 转换为tensor
        if isinstance(env_features, np.ndarray):
            env_features = torch.FloatTensor(env_features)
        if isinstance(ind_features, np.ndarray):
            ind_features = torch.FloatTensor(ind_features)
        if vehicle_types is not None and isinstance(vehicle_types, np.ndarray):
            vehicle_types = torch.FloatTensor(vehicle_types)

        # 确保形状正确
        if env_features.dim() == 1:
            env_features = env_features.unsqueeze(0)
        if ind_features.dim() == 1:
            ind_features = ind_features.unsqueeze(0)
        if vehicle_types is not None and vehicle_types.dim() == 1:
            vehicle_types = vehicle_types.unsqueeze(1)

        # 移动到设备
        env_features = env_features.to(self.device)
        ind_features = ind_features.to(self.device)
        if vehicle_types is not None:
            vehicle_types = vehicle_types.to(self.device)

        # 推理
        with torch.no_grad():
            P_final, intermediates = self.model(env_features, ind_features, vehicle_types)

        # 转换为numpy
        predictions = P_final.cpu().numpy()

        if return_intermediates:
            # 转换中间结果为numpy
            intermediates_np = {}
            for key, value in intermediates.items():
                if isinstance(value, torch.Tensor):
                    intermediates_np[key] = value.cpu().numpy()
                else:
                    intermediates_np[key] = value
            return predictions, intermediates_np
        else:
            return predictions

    def predict_single(self,
                      env_feature: Union[np.ndarray, List],
                      ind_feature: Union[np.ndarray, List],
                      vehicle_type: int = 0) -> float:
        """
        单样本预测（便捷接口）

        Args:
            env_feature: [4] 环境特征向量
            ind_feature: [10] 个体特征向量
            vehicle_type: 车辆类型 (0=HDV, 1=CAV)

        Returns:
            probability: 换道意图概率 (0-1之间的浮点数)
        """
        env_feature = np.array(env_feature).reshape(1, -1)
        ind_feature = np.array(ind_feature).reshape(1, -1)
        vehicle_type = np.array([[vehicle_type]], dtype=np.float32)

        prediction = self.predict(env_feature, ind_feature, vehicle_type)
        return float(prediction[0, 0])

    def predict_batch_from_dict(self,
                                observations: List[Dict],
                                return_intermediates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]:
        """
        从观测字典批量预测

        Args:
            observations: 观测字典列表，每个字典包含:
                - 'env_features': [4] 环境特征
                - 'ind_features': [10] 个体特征
                - 'vehicle_type': (可选) 车辆类型
            return_intermediates: 是否返回中间结果

        Returns:
            predictions: [N, 1] 预测概率
            intermediates_list: (可选) 中间结果列表
        """
        env_features = np.array([obs['env_features'] for obs in observations])
        ind_features = np.array([obs['ind_features'] for obs in observations])

        vehicle_types = None
        if 'vehicle_type' in observations[0]:
            vehicle_types = np.array([[obs['vehicle_type']] for obs in observations])

        if return_intermediates:
            predictions, intermediates = self.predict(
                env_features, ind_features, vehicle_types,
                return_intermediates=True
            )
            # 将批量中间结果拆分为列表
            intermediates_list = []
            for i in range(len(observations)):
                inter_i = {k: v[i] for k, v in intermediates.items()}
                intermediates_list.append(inter_i)
            return predictions, intermediates_list
        else:
            return self.predict(env_features, ind_features, vehicle_types)

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'path_architecture': self.path_architecture.value,
            'fusion_strategy': self.fusion_strategy.value,
            'device': str(self.device),
            'causal_parameters': self.causal_params
        }


class SCMPredictorFactory:
    """
    SCM预测器工厂类
    用于统一管理和创建不同类型的预测器
    """

    _instances = {}  # 单例缓存

    @classmethod
    def create_predictor(cls,
                        model_type: str = "shallow_hierarchical",
                        model_base_path: Optional[str] = None,
                        device: str = "cpu",
                        use_cache: bool = True) -> SCMPredictor:
        """
        创建预测器（支持单例模式）

        Args:
            model_type: 模型类型
            model_base_path: 模型基础路径
            device: 运行设备
            use_cache: 是否使用缓存的实例

        Returns:
            SCMPredictor实例
        """
        cache_key = f"{model_type}_{device}"

        if use_cache and cache_key in cls._instances:
            print(f"[SCMPredictorFactory] 使用缓存的预测器: {cache_key}")
            return cls._instances[cache_key]

        predictor = SCMPredictor(
            model_type=model_type,
            model_base_path=model_base_path,
            device=device
        )

        if use_cache:
            cls._instances[cache_key] = predictor

        return predictor

    @classmethod
    def get_available_models(cls, model_base_path: Optional[str] = None) -> List[str]:
        """
        获取可用的模型列表

        Args:
            model_base_path: 模型基础路径

        Returns:
            可用模型类型列表
        """
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent / "models" / "SCM_models"

        model_base_path = Path(model_base_path)
        if not model_base_path.exists():
            return []

        available_models = []
        for item in model_base_path.iterdir():
            if item.is_dir() and (item / "best_model.pth").exists():
                available_models.append(item.name)

        return available_models


def demo_prediction():
    """演示如何使用预测器"""
    print("=" * 80)
    print("SCM意图预测演示")
    print("=" * 80)

    # 1. 创建预测器
    predictor = SCMPredictorFactory.create_predictor(
        model_type="shallow_hierarchical",
        device="cpu"
    )

    # 2. 准备测试数据（随机生成，实际使用时应该是真实的归一化特征）
    np.random.seed(42)

    # 环境特征 [v_avg, kappa_avg, delta_v, delta_kappa]
    env_features = np.random.randn(5, 4).astype(np.float32)

    # 个体特征 [v_ego, v_rel, d_headway, T_headway,
    #          delta_v_adj_front, d_adj_front, T_adj_front,
    #          delta_v_adj_rear, d_adj_rear, T_adj_rear]
    ind_features = np.random.randn(5, 10).astype(np.float32)

    # 车辆类型
    vehicle_types = np.array([[0], [1], [0], [1], [0]], dtype=np.float32)

    # 3. 批量预测
    print("\n批量预测:")
    predictions, intermediates = predictor.predict(
        env_features, ind_features, vehicle_types,
        return_intermediates=True
    )

    for i in range(len(predictions)):
        print(f"  样本 {i}: 换道概率 = {predictions[i, 0]:.4f}, "
              f"车辆类型 = {'CAV' if vehicle_types[i, 0] == 1 else 'HDV'}")

    # 4. 单样本预测
    print("\n单样本预测:")
    single_prob = predictor.predict_single(
        env_feature=env_features[0],
        ind_feature=ind_features[0],
        vehicle_type=0
    )
    print(f"  换道概率 = {single_prob:.4f}")

    # 5. 显示模型信息
    print("\n模型信息:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if key != 'causal_parameters':
            print(f"  {key}: {value}")

    # 6. 显示可用模型
    print("\n可用模型:")
    available = SCMPredictorFactory.get_available_models()
    for model in available:
        print(f"  - {model}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_prediction()
