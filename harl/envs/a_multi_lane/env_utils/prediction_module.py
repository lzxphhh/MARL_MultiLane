#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测模块集成

功能:
1. 调用SCM意图预测模型，预测周边车辆换道意图
2. 调用CQR占用预测模型，基于意图预测结果预测未来轨迹占用
3. 为环境提供统一的预测接口

Author: 交通流研究团队
Date: 2025-01
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 添加预测模块路径
harl_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(harl_root))

try:
    from harl.prediction.intention.strategy.SCM_prediction import SCMPredictorFactory
    from harl.prediction.occupancy.strategy.CQR_prediction import CQRPredictorFactory
    from harl.envs.a_multi_lane.env_utils.feature_extractor import FeatureExtractor
except ImportError:
    # 当直接运行时的导入
    import os
    os.chdir(str(harl_root))
    sys.path.insert(0, str(harl_root))
    from harl.prediction.intention.strategy.SCM_prediction import SCMPredictorFactory
    from harl.prediction.occupancy.strategy.CQR_prediction import CQRPredictorFactory
    from feature_extractor import FeatureExtractor


class PredictionModule:
    """预测模块 - 集成意图预测和占用预测"""

    def __init__(self,
                 intention_model_type: str = "shallow_hierarchical",
                 occupancy_model_type: str = "CQR-GRU-uncertainty",
                 use_conformal: bool = True,
                 device: str = "cpu",
                 use_cache: bool = True):
        """
        初始化预测模块

        Args:
            intention_model_type: 意图预测模型类型 ("shallow_hierarchical" 或 "medium_hierarchical")
            occupancy_model_type: 占用预测模型类型
                - "QR-GRU-uncertainty"
                - "QR-GRU-uncertainty_pcgrad"
                - "CQR-GRU-uncertainty"
                - "CQR-GRU-uncertainty_pcgrad"
            use_conformal: 是否使用共形化预测（CQR模型）
            device: 计算设备
            use_cache: 是否使用单例缓存
        """
        print(f"[PredictionModule] 初始化预测模块...")

        # 创建特征提取器
        self.feature_extractor = FeatureExtractor()

        # 创建意图预测器
        print(f"  - 加载意图预测模型: {intention_model_type}")
        self.intention_predictor = SCMPredictorFactory.create_predictor(
            model_type=intention_model_type,
            device=device,
            use_cache=use_cache
        )

        # 创建占用预测器
        print(f"  - 加载占用预测模型: {occupancy_model_type}")
        if use_conformal and "CQR" in occupancy_model_type:
            self.occupancy_predictor = CQRPredictorFactory.create_predictor(
                model_type=occupancy_model_type,
                calibrator_type="standard",
                device=device,
                use_cache=use_cache
            )
        else:
            self.occupancy_predictor = CQRPredictorFactory.create_predictor(
                model_type=occupancy_model_type,
                device=device,
                use_cache=use_cache
            )

        self.device = device
        print(f"[PredictionModule] ✓ 预测模块初始化完成")

    def predict_intentions(self,
                          vehicle_ids: List[str],
                          state: Dict,
                          lane_statistics: Dict) -> Dict[str, float]:
        """
        批量预测车辆换道意图

        Args:
            vehicle_ids: 需要预测的车辆ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            intentions: {vehicle_id: intention_probability}
                intention_probability ∈ [0, 1]，表示换道概率
        """
        if len(vehicle_ids) == 0:
            return {}

        # 批量提取特征
        env_features, ind_features, vehicle_types = self.feature_extractor.batch_extract_intention_features(
            vehicle_ids, state, lane_statistics
        )

        if env_features.shape[0] == 0:
            return {}

        # 调用意图预测模型
        predictions = self.intention_predictor.predict(
            env_features, ind_features, vehicle_types
        )  # shape: [N,]

        # 构建结果字典
        intentions = {}
        for i, veh_id in enumerate(vehicle_ids):
            if i < len(predictions):
                intentions[veh_id] = float(predictions[i])

        return intentions

    def predict_single_intention(self,
                                vehicle_id: str,
                                state: Dict,
                                lane_statistics: Dict) -> float:
        """
        预测单个车辆的换道意图

        Args:
            vehicle_id: 车辆ID
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            intention_probability: 换道概率 [0, 1]
        """
        intentions = self.predict_intentions([vehicle_id], state, lane_statistics)
        return intentions.get(vehicle_id, 0.0)

    def predict_occupancy(self,
                         vehicle_id: str,
                         state: Dict,
                         lane_statistics: Dict,
                         intention: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预测单个车辆的未来轨迹占用

        Args:
            vehicle_id: 车辆ID
            state: 环境状态字典
            lane_statistics: 车道统计信息
            intention: 意图预测结果（如果为None，则自动调用意图预测）

        Returns:
            lower: 下界轨迹 [30,] (3秒，每步0.1s)
            median: 中位轨迹 [30,]
            upper: 上界轨迹 [30,]
        """
        # 如果没有提供意图，先预测意图
        if intention is None:
            intention = self.predict_single_intention(vehicle_id, state, lane_statistics)

        # 提取占用预测特征
        occ_features = self.feature_extractor.extract_occupancy_features(
            vehicle_id, state, lane_statistics, intention
        )

        # 添加batch维度
        features_batch = {
            key: value[np.newaxis, ...] for key, value in occ_features.items()
        }

        # 调用占用预测模型
        lower, median, upper = self.occupancy_predictor.predict(
            features_batch, return_intervals=True
        )  # shapes: [1, 30]

        # 移除batch维度
        return lower[0], median[0], upper[0]

    def predict_intentions_and_occupancy(self,
                                        vehicle_ids: List[str],
                                        state: Dict,
                                        lane_statistics: Dict) -> Dict[str, Dict]:
        """
        批量预测车辆的意图和占用（端到端）

        Args:
            vehicle_ids: 车辆ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            predictions: {
                vehicle_id: {
                    'intention': float,
                    'occupancy': {
                        'lower': np.ndarray [30,],
                        'median': np.ndarray [30,],
                        'upper': np.ndarray [30,]
                    }
                }
            }
        """
        predictions = {}

        # 步骤1: 批量预测意图
        intentions = self.predict_intentions(vehicle_ids, state, lane_statistics)

        # 步骤2: 逐个预测占用（使用意图作为输入）
        for veh_id in vehicle_ids:
            if veh_id not in intentions:
                continue

            try:
                intention = intentions[veh_id]
                lower, median, upper = self.predict_occupancy(
                    veh_id, state, lane_statistics, intention
                )

                predictions[veh_id] = {
                    'intention': intention,
                    'occupancy': {
                        'lower': lower,
                        'median': median,
                        'upper': upper
                    }
                }
            except (ValueError, KeyError) as e:
                # 车辆状态不完整，跳过
                continue

        return predictions

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'intention_model': self.intention_predictor.get_model_info(),
            'occupancy_model': self.occupancy_predictor.get_model_info()
        }


class PredictionModuleFactory:
    """预测模块工厂"""

    _cached_modules = {}

    @classmethod
    def create_module(cls,
                     intention_model_type: str = "shallow_hierarchical",
                     occupancy_model_type: str = "CQR-GRU-uncertainty",
                     use_conformal: bool = True,
                     device: str = "cpu",
                     use_cache: bool = True) -> PredictionModule:
        """
        创建预测模块（支持单例缓存）

        Args:
            intention_model_type: 意图预测模型类型
            occupancy_model_type: 占用预测模型类型
            use_conformal: 是否使用共形化
            device: 计算设备
            use_cache: 是否使用缓存

        Returns:
            PredictionModule实例
        """
        if use_cache:
            cache_key = f"{intention_model_type}_{occupancy_model_type}_{device}"
            if cache_key not in cls._cached_modules:
                cls._cached_modules[cache_key] = PredictionModule(
                    intention_model_type=intention_model_type,
                    occupancy_model_type=occupancy_model_type,
                    use_conformal=use_conformal,
                    device=device,
                    use_cache=True
                )
            return cls._cached_modules[cache_key]
        else:
            return PredictionModule(
                intention_model_type=intention_model_type,
                occupancy_model_type=occupancy_model_type,
                use_conformal=use_conformal,
                device=device,
                use_cache=False
            )


if __name__ == "__main__":
    # 测试预测模块
    print("=" * 80)
    print("测试预测模块集成")
    print("=" * 80)

    # 创建模拟状态
    state = {
        'HDV_0': {
            'lane_index': 1,
            'speed': 25.0,
            'acceleration': 0.5,
            'position_x': 100.0,
            'position_y': 3.5,
            'surrounding_vehicles': {
                'front_1': {
                    'id': 'HDV_1',
                    'long_dist': 50.0,
                    'long_rel_v': -2.0,
                    'speed': 23.0,
                    'acceleration': 0.0,
                    'lane_index': 1,
                    'length': 5.0,
                    'width': 2.0
                },
                'adj_front': {
                    'id': 'HDV_2',
                    'long_dist': 60.0,
                    'long_rel_v': 1.0,
                    'speed': 26.0,
                    'lane_index': 2,
                },
            }
        },
        'CAV_0': {
            'lane_index': 2,
            'speed': 28.0,
            'acceleration': 0.2,
            'position_x': 120.0,
            'position_y': 7.0,
            'surrounding_vehicles': {
                'front_1': {'long_dist': 80.0, 'long_rel_v': 0.0},
            }
        }
    }

    lane_statistics = {
        0: {'mean_speed': 20.0, 'density': 0.08},
        1: {'mean_speed': 25.0, 'density': 0.05},
        2: {'mean_speed': 28.0, 'density': 0.04}
    }

    # 创建预测模块
    pred_module = PredictionModuleFactory.create_module(
        intention_model_type="shallow_hierarchical",
        occupancy_model_type="CQR-GRU-uncertainty",
        use_conformal=True,
        device="cpu",
        use_cache=True
    )

    # 测试1: 批量意图预测
    print("\n1. 批量意图预测:")
    vehicle_ids = ['HDV_0', 'CAV_0']
    intentions = pred_module.predict_intentions(vehicle_ids, state, lane_statistics)
    for veh_id, intention in intentions.items():
        print(f"  {veh_id}: 换道概率 = {intention:.4f}")

    # 测试2: 单个占用预测
    print("\n2. 占用预测 (HDV_0):")
    lower, median, upper = pred_module.predict_occupancy('HDV_0', state, lane_statistics)
    print(f"  下界轨迹: {lower[:5]}... (前5步)")
    print(f"  中位轨迹: {median[:5]}...")
    print(f"  上界轨迹: {upper[:5]}...")
    print(f"  区间宽度: {(upper - lower).mean():.4f}m (平均)")

    # 测试3: 端到端预测
    print("\n3. 端到端预测 (意图+占用):")
    all_predictions = pred_module.predict_intentions_and_occupancy(
        vehicle_ids, state, lane_statistics
    )
    for veh_id, pred in all_predictions.items():
        print(f"  {veh_id}:")
        print(f"    意图: {pred['intention']:.4f}")
        print(f"    占用区间宽度: {(pred['occupancy']['upper'] - pred['occupancy']['lower']).mean():.4f}m")

    # 模型信息
    print("\n4. 模型信息:")
    info = pred_module.get_model_info()
    print(f"  意图预测模型: {info['intention_model']['model_type']}")
    print(f"  占用预测模型: {info['occupancy_model']['model_type']}")
    print(f"  使用共形化: {info['occupancy_model']['use_conformal']}")

    print("\n✓ 预测模块测试完成!")
