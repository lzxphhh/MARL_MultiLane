#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境配置

定义环境的核心参数和配置
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class EnvironmentConfig:
    """环境配置类"""

    # ==================== 基础配置 ====================
    num_CAVs: int = 5  # CAV数量
    num_HDVs: int = 10  # HDV数量
    CAV_penetration: float = 0.5  # CAV渗透率
    delta_t: float = 0.1  # 时间步长 (s)
    warmup_steps: int = 50  # 预热步数

    # ==================== 车道配置 ====================
    num_lanes: int = 3  # 车道数量
    lane_width: float = 3.5  # 车道宽度 (m)
    road_length: float = 1000.0  # 道路长度 (m)

    # ==================== 车辆参数 ====================
    v_max: float = 33.33  # 最大速度 (m/s) ~120 km/h
    v_min: float = 0.0  # 最小速度
    a_max: float = 3.0  # 最大加速度 (m/s^2)
    a_min: float = -4.0  # 最大减速度 (m/s^2)
    vehicle_length: float = 5.0  # 车辆长度 (m)
    vehicle_width: float = 2.0  # 车辆宽度 (m)

    # ==================== 安全参数 ====================
    gap_threshold: float = 2.0  # 碰撞距离阈值 (m)
    gap_warn_threshold: float = 5.0  # 警告距离阈值 (m)
    TTC_collision_threshold: float = 0.5  # TTC碰撞阈值 (s)
    TTC_warning_threshold: float = 1.5  # TTC警告阈值 (s)

    # ==================== 观测空间配置 ====================
    obs_size: int = 1034  # 局部观测维度
    shared_obs_size: int = 500  # 共享观测维度（可变）
    history_length: int = 20  # 历史轨迹长度

    # ==================== 动作空间配置 ====================
    action_dim: int = 2  # 动作维度 [横向, 纵向]
    lateral_action_range: Tuple[float, float] = (-1.5, 1.5)  # 横向动作范围
    longitudinal_action_range: Tuple[float, float] = (-1.5, 1.5)  # 纵向动作范围

    # ==================== 奖励权重 ====================
    reward_weights: Dict[str, float] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.reward_weights is None:
            self.reward_weights = {
                'safety': 0.4,
                'efficiency': 0.3,
                'stability': 0.2,
                'comfort': 0.1
            }

    # ==================== 预测模型配置 ====================
    intention_model_type: str = "shallow_hierarchical"
    occupancy_model_type: str = "CQR-GRU-uncertainty"
    use_conformal: bool = True

    # ==================== 决策模型配置 ====================
    decision_model_type: str = "shallow_hierarchical"
    freeze_base_model: bool = True  # 初期冻结
    enable_decision_training: bool = True  # 启用决策微调

    # ==================== 微调策略配置 ====================
    fine_tune_lr: float = 1e-4  # 微调学习率
    fine_tune_stage_thresholds: Tuple[int, int] = (1000, 2000)  # 阶段切换阈值


# IDM参数
IDM_PARAMS = {
    'a': 1.0,  # 最大加速度 (m/s^2)
    'b': 1.5,  # 舒适减速度 (m/s^2)
    'delta': 4.0,  # 加速度指数
    's0': 2.0,  # 最小间距 (m)
    'T': 1.5,  # 期望时间间隙 (s)
    'v_desired': 30.0  # 期望速度 (m/s)
}


# 场景中心（用于场景分类）
SCENE_CENTERS = {
    'class_0': [0.053, 26.1, 0.054, 26.2, 0.052, 25.9],
    'class_1': [0.082, 20.5, 0.080, 20.8, 0.081, 20.6],
    'class_2': [0.065, 23.0, 0.063, 23.5, 0.064, 23.2],
    'class_3': [0.045, 28.0, 0.044, 28.5, 0.046, 27.8],
    'class_4': [0.095, 18.0, 0.093, 18.5, 0.094, 18.2],
    'class_5': [0.070, 22.0, 0.068, 22.5, 0.069, 22.2],
    'class_6': [0.055, 25.0, 0.053, 25.5, 0.054, 25.2],
    'class_7': [0.088, 19.0, 0.086, 19.5, 0.087, 19.2],
    'class_8': [0.060, 24.0, 0.058, 24.5, 0.059, 24.2],
    'class_9': [0.075, 21.0, 0.073, 21.5, 0.074, 21.2],
}


def get_default_config() -> EnvironmentConfig:
    """获取默认配置"""
    return EnvironmentConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_default_config()
    print("环境配置:")
    print(f"  CAV数量: {config.num_CAVs}")
    print(f"  HDV数量: {config.num_HDVs}")
    print(f"  CAV渗透率: {config.CAV_penetration}")
    print(f"  观测维度: {config.obs_size}")
    print(f"  动作维度: {config.action_dim}")
    print(f"  奖励权重: {config.reward_weights}")
    print(f"  意图模型: {config.intention_model_type}")
    print(f"  占用模型: {config.occupancy_model_type}")
    print(f"  决策模型: {config.decision_model_type}")
    print(f"  微调学习率: {config.fine_tune_lr}")
