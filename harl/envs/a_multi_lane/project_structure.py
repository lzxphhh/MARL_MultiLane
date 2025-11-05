# project_structure.py
"""
混合交通系统协同控制框架 - 项目结构定义
Project Structure for Hybrid Traffic System Cooperative Control Framework
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class VehicleType(Enum):
    """车辆类型枚举"""
    HDV = 0  # Human Driven Vehicle
    CAV = 1  # Connected Automated Vehicle


class Direction(Enum):
    """相对方向枚举"""
    EF = "EF"  # Ego Front
    ER = "ER"  # Ego Rear
    LF = "LF"  # Left Front
    LR = "LR"  # Left Rear
    RF = "RF"  # Right Front
    RR = "RR"  # Right Rear


class LateralDecision(Enum):
    """横向决策枚举"""
    LEFT = -1  # 向左变道
    KEEP = 0  # 保持车道
    RIGHT = 1  # 向右变道


class ScenarioType(Enum):
    """场景类型枚举"""
    FREE_FLOW = 1  # 自由流
    TRANSITION_BALANCED = 2  # 过渡均衡流
    TRANSITION_UNBALANCED = 3  # 过渡非均衡流
    CONGESTED = 4  # 拥堵流


@dataclass
class VehicleState:
    """车辆状态信息"""
    x: float  # 纵向位置 [m]
    y: float  # 横向位置 [m]
    v: float  # 速度 [m/s]
    theta: float  # 航向角 [rad]
    a: float  # 加速度 [m/s²]
    omega: float  # 横摆角速度 [rad/s]
    lane_id: int  # 车道ID {0,1,2}
    vehicle_type: VehicleType  # 车辆类型
    front_v: float # 前方车辆速度
    front_spacing: float # 与前方车辆的间距


@dataclass
class TrafficScenarioParams:
    """交通场景参数"""
    mean_vector: np.ndarray  # 4维均值向量 [κ, σκ, v̄, σv̄]
    covariance_matrix: np.ndarray  # 4x4协方差矩阵


@dataclass
class WeightConfig:
    """权重配置"""
    # 横向决策模块权重
    efficiency: float  # 效率性权重
    equilibrium: float  # 均衡性权重

    # 协同控制器权重
    ws: float  # 安全权重
    we: float  # 效率权重
    wc: float  # 舒适权重
    wt: float  # 任务权重
    wcoop: float  # 协同权重


@dataclass
class SystemParameters:
    """系统参数配置"""
    # 时间参数
    dt: float = 0.1  # 时间步长 [s]
    prediction_horizon: float = 3.0  # 预测周期 [s]

    # 车辆参数
    vehicle_length: float = 5 # 车辆长度 [m]
    safe_distance: float = 2.0  # 安全间距 [m]
    lane_width: float = 3.2  # 车道宽度 [m]

    # 动力学约束
    a_max: float = 4.0  # 最大加速度 [m/s²]
    a_min: float = -6.0  # 最大制动加速度 [m/s²]
    a_dot_max: float = 4.0  # 最大加速度变化率 [m/s³]
    v_max: float = 35.0  # 最大速度 [m/s]

    # 舒适性参数
    alpha_c: float = 0.5  # 加速度舒适性系数
    beta_c: float = 0.5  # 加速度变化率舒适性系数

    # 不确定性参数
    sigma_a_cav: float = 0.5  # CAV加速度标准偏差
    sigma_a_hdv: float = 1.0  # HDV加速度标准偏差

    # 概率阈值
    prob_threshold_1: float = 0.05  # 主邻接车辆概率阈值
    prob_threshold_2: float = 0.05  # 次邻接车辆概率阈值

    # 调制参数
    alpha_ext: float = 2.0  # 外部调制参数
    beta_ext: float = 1.5  # 外部调制参数β
    beta_lane: float = 2.0  # 横向影响衰减系数
    alpha_time: float = 1.0  # 时间影响调节系数

    # 决策阈值
    necessity_threshold: float = 0.5  # 必要性阈值
    decision_threshold: float = 0.5  # 决策阈值
    confidence_threshold: float = 0.5  # 置信度阈值

    # 场景识别参数
    temperature_param: float = 0.5  # 温度参数β
    smooth_factor: float = 0.8  # 平滑因子α


# 预设场景参数
NORMALIZATION_COEFFICIENTS = np.array([1/97.306, 1/33.694, 1/17.312, 1/4.493])
# 预设场景参数
SCENARIO_PARAMS = {
    ScenarioType.FREE_FLOW: TrafficScenarioParams(
        mean_vector=np.array([33.898, 4.350, 14.504, 0.930]),
        covariance_matrix=np.array([
            [31.436, 2.361, -12.293, 0.080],
            [2.361, 3.356, -1.257, 0.136],
            [-12.293, -1.257, 5.954, -0.142],
            [0.080, 0.136, -0.142, 0.163]
        ])
    ),
    ScenarioType.TRANSITION_BALANCED: TrafficScenarioParams(
        mean_vector=np.array([43.279, 5.679, 10.489, 0.951]),
        covariance_matrix=np.array([
            [36.956, 3.463, -8.802, 0.182],
            [3.463, 4.398, -0.848, 0.222],
            [-8.802, -0.848, 3.306, -0.074],
            [0.182, 0.222, -0.074, 0.179]
        ])
    ),
    ScenarioType.TRANSITION_UNBALANCED: TrafficScenarioParams(
        mean_vector=np.array([50.381, 11.799, 8.532, 2.354]),
        covariance_matrix=np.array([
            [43.553, 10.136, -9.985, 0.594],
            [10.136, 10.630, -1.956, 1.342],
            [-9.985, -1.956, 2.959, 0.079],
            [0.594, 1.342, 0.079, 0.407]
        ])
    ),
    ScenarioType.CONGESTED: TrafficScenarioParams(
        mean_vector=np.array([65.830, 9.206, 4.658, 1.093]),
        covariance_matrix=np.array([
            [91.469, 15.518, -8.889, -0.307],
            [15.518, 14.709, -1.506, 1.349],
            [-8.889, -1.506, 1.247, 0.076],
            [-0.307, 1.349, 0.076, 0.291]
        ])
    )
}
# 预设权重配置
DEFAULT_WEIGHTS = {
    ScenarioType.FREE_FLOW: WeightConfig(
        efficiency=0.7, equilibrium=0.3,
        ws=0.3, we=0.3, wc=0.2, wt=0.1, wcoop=0.1
    ),
    ScenarioType.TRANSITION_BALANCED: WeightConfig(
        efficiency=0.5, equilibrium=0.5,
        ws=0.35, we=0.25, wc=0.2, wt=0.1, wcoop=0.1
    ),
    ScenarioType.TRANSITION_UNBALANCED: WeightConfig(
        efficiency=0.4, equilibrium=0.6,
        ws=0.4, we=0.2, wc=0.2, wt=0.1, wcoop=0.1
    ),
    ScenarioType.CONGESTED: WeightConfig(
        efficiency=0.2, equilibrium=0.8,
        ws=0.5, we=0.15, wc=0.2, wt=0.1, wcoop=0.05
    )
}