#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横向控制器：基于Bezier曲线的轨迹规划

实现4阶Bezier曲线用于平滑换道轨迹规划，包括：
- 横向位置轨迹 y_ref(t)
- 横向速度轨迹 v_y_ref(t)
- 横向加速度轨迹 a_y_ref(t)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import math


class LateralController:
    """
    横向轨迹控制器

    使用4阶Bezier曲线生成平滑的换道轨迹
    """

    def __init__(self, lane_width: float = 3.75, default_lane_change_time: float = 3.0):
        """
        初始化横向控制器

        Args:
            lane_width: 车道宽度（米），默认3.75m
            default_lane_change_time: 默认换道时间（秒），默认3.0s
        """
        self.lane_width = lane_width
        self.default_t_LC = default_lane_change_time

        # Bezier曲线阶数
        self.bezier_order = 4

        # 当前换道状态
        self.active_maneuvers = {}  # {veh_id: maneuver_info}

    def bernstein_poly(self, i: int, n: int, u: float) -> float:
        """
        计算Bernstein多项式基函数

        B_i^n(u) = C(n,i) * u^i * (1-u)^(n-i)

        Args:
            i: 索引 (0 <= i <= n)
            n: 阶数
            u: 参数 (0 <= u <= 1)

        Returns:
            Bernstein多项式值
        """
        from math import comb
        return comb(n, i) * (u ** i) * ((1 - u) ** (n - i))

    def compute_bezier_trajectory(self,
                                   y_0: float,
                                   y_target: float,
                                   t_LC: float,
                                   n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算4阶Bezier曲线的横向轨迹（位置、速度、加速度）

        根据00_structure.tex中的公式：
        - y_ref(t) = Σ_{i=0}^4 B_i^4(u) * P_i
        - v_y_ref(t) = (4/t_LC) * Σ_{i=0}^3 B_i^3(u) * (P_{i+1} - P_i)
        - a_y_ref(t) = (12/t_LC^2) * Σ_{i=0}^2 B_i^2(u) * (P_{i+2} - 2*P_{i+1} + P_i)

        Args:
            y_0: 初始横向位置（米）
            y_target: 目标横向位置（米）
            t_LC: 换道时间（秒）
            n_samples: 采样点数（默认30，对应3秒@10Hz）

        Returns:
            time_array: 时间序列 [n_samples]
            position_array: 横向位置 [n_samples]
            velocity_array: 横向速度 [n_samples]
            acceleration_array: 横向加速度 [n_samples]
        """
        # 定义5个控制点
        P = np.array([
            y_0,                        # P_0
            y_0,                        # P_1
            (y_0 + y_target) / 2.0,    # P_2
            y_target,                   # P_3
            y_target                    # P_4
        ])

        # 时间序列
        time_array = np.linspace(0, t_LC, n_samples)
        u_array = time_array / t_LC  # 归一化时间 [0, 1]

        # 初始化输出数组
        position_array = np.zeros(n_samples)
        velocity_array = np.zeros(n_samples)
        acceleration_array = np.zeros(n_samples)

        # 逐点计算
        for idx, u in enumerate(u_array):
            # 1. 位置：y_ref(t) = Σ_{i=0}^4 B_i^4(u) * P_i
            position = 0.0
            for i in range(5):
                position += self.bernstein_poly(i, 4, u) * P[i]
            position_array[idx] = position

            # 2. 速度：v_y_ref(t) = (4/t_LC) * Σ_{i=0}^3 B_i^3(u) * (P_{i+1} - P_i)
            velocity = 0.0
            for i in range(4):
                delta_P = P[i+1] - P[i]
                velocity += self.bernstein_poly(i, 3, u) * delta_P
            velocity *= (4.0 / t_LC)
            velocity_array[idx] = velocity

            # 3. 加速度：a_y_ref(t) = (12/t_LC^2) * Σ_{i=0}^2 B_i^2(u) * (P_{i+2} - 2*P_{i+1} + P_i)
            acceleration = 0.0
            for i in range(3):
                delta2_P = P[i+2] - 2*P[i+1] + P[i]
                acceleration += self.bernstein_poly(i, 2, u) * delta2_P
            acceleration *= (12.0 / (t_LC ** 2))
            acceleration_array[idx] = acceleration

        return time_array, position_array, velocity_array, acceleration_array

    def start_lane_change(self,
                          veh_id: str,
                          current_lane: int,
                          target_lane: int,
                          current_y_position: float,
                          t_LC: Optional[float] = None) -> Dict:
        """
        开始换道操作

        Args:
            veh_id: 车辆ID
            current_lane: 当前车道索引
            target_lane: 目标车道索引
            current_y_position: 当前横向位置（米）
            t_LC: 换道时间（秒），如果为None则使用默认值

        Returns:
            maneuver_info: 换道信息字典
        """
        if t_LC is None:
            t_LC = self.default_t_LC

        # 计算目标横向位置
        lane_change_direction = target_lane - current_lane
        target_y_position = current_y_position + lane_change_direction * self.lane_width

        # 生成Bezier轨迹
        time_array, position_array, velocity_array, acceleration_array = \
            self.compute_bezier_trajectory(current_y_position, target_y_position, t_LC)

        # 创建换道信息
        maneuver_info = {
            'veh_id': veh_id,
            'start_time': 0.0,  # 由外部设置
            'duration': t_LC,
            'current_lane': current_lane,
            'target_lane': target_lane,
            'y_start': current_y_position,
            'y_target': target_y_position,
            'trajectory': {
                'time': time_array,
                'position': position_array,
                'velocity': velocity_array,
                'acceleration': acceleration_array
            },
            'current_index': 0,
            'completed': False
        }

        # 存储到活动换道字典
        self.active_maneuvers[veh_id] = maneuver_info

        return maneuver_info

    def get_reference_at_time(self,
                              veh_id: str,
                              elapsed_time: float) -> Optional[Dict]:
        """
        获取指定车辆在给定时间的参考轨迹点

        Args:
            veh_id: 车辆ID
            elapsed_time: 从换道开始经过的时间（秒）

        Returns:
            reference_point: {
                'y_ref': 横向位置,
                'v_y_ref': 横向速度,
                'a_y_ref': 横向加速度,
                'completed': 是否完成
            }
            如果车辆不在换道中，返回None
        """
        if veh_id not in self.active_maneuvers:
            return None

        maneuver = self.active_maneuvers[veh_id]

        if maneuver['completed']:
            return None

        # 检查是否超过换道时间
        if elapsed_time >= maneuver['duration']:
            # 返回终点值并标记完成
            maneuver['completed'] = True
            return {
                'y_ref': maneuver['y_target'],
                'v_y_ref': 0.0,
                'a_y_ref': 0.0,
                'completed': True
            }

        # 在轨迹中查找最接近的时间点
        time_array = maneuver['trajectory']['time']
        idx = np.argmin(np.abs(time_array - elapsed_time))
        maneuver['current_index'] = idx

        return {
            'y_ref': maneuver['trajectory']['position'][idx],
            'v_y_ref': maneuver['trajectory']['velocity'][idx],
            'a_y_ref': maneuver['trajectory']['acceleration'][idx],
            'completed': False
        }

    def cancel_lane_change(self, veh_id: str):
        """
        取消换道操作

        Args:
            veh_id: 车辆ID
        """
        if veh_id in self.active_maneuvers:
            del self.active_maneuvers[veh_id]

    def is_lane_changing(self, veh_id: str) -> bool:
        """
        检查车辆是否正在换道

        Args:
            veh_id: 车辆ID

        Returns:
            True if lane changing, False otherwise
        """
        return veh_id in self.active_maneuvers and not self.active_maneuvers[veh_id]['completed']

    def get_maneuver_progress(self, veh_id: str) -> Optional[float]:
        """
        获取换道进度

        Args:
            veh_id: 车辆ID

        Returns:
            进度百分比 [0, 1]，如果不在换道返回None
        """
        if veh_id not in self.active_maneuvers:
            return None

        maneuver = self.active_maneuvers[veh_id]
        if maneuver['completed']:
            return 1.0

        return maneuver['current_index'] / (len(maneuver['trajectory']['time']) - 1)

    def reset(self):
        """重置所有活动换道"""
        self.active_maneuvers.clear()


class LateralControllerWithPID:
    """
    带PID跟踪的横向控制器

    生成Bezier参考轨迹后，使用PID控制器跟踪
    """

    def __init__(self,
                 lane_width: float = 3.75,
                 default_lane_change_time: float = 3.0,
                 kp: float = 1.0,
                 ki: float = 0.1,
                 kd: float = 0.5):
        """
        初始化带PID的横向控制器

        Args:
            lane_width: 车道宽度
            default_lane_change_time: 默认换道时间
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
        """
        self.trajectory_planner = LateralController(lane_width, default_lane_change_time)

        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID状态
        self.pid_states = {}  # {veh_id: {'error_integral': 0, 'last_error': 0}}

    def compute_control_command(self,
                                veh_id: str,
                                current_y: float,
                                current_v_y: float,
                                reference: Dict,
                                dt: float = 0.1) -> float:
        """
        计算PID控制命令

        Args:
            veh_id: 车辆ID
            current_y: 当前横向位置
            current_v_y: 当前横向速度
            reference: 参考轨迹点（来自get_reference_at_time）
            dt: 时间步长

        Returns:
            control_command: 横向加速度命令（m/s^2）
        """
        if veh_id not in self.pid_states:
            self.pid_states[veh_id] = {
                'error_integral': 0.0,
                'last_error': 0.0
            }

        # 位置误差
        error_y = reference['y_ref'] - current_y

        # 速度误差
        error_v = reference['v_y_ref'] - current_v_y

        # 积分项
        self.pid_states[veh_id]['error_integral'] += error_y * dt

        # 微分项
        error_derivative = (error_y - self.pid_states[veh_id]['last_error']) / dt
        self.pid_states[veh_id]['last_error'] = error_y

        # PID控制
        a_y_command = (
            self.kp * error_y +
            self.ki * self.pid_states[veh_id]['error_integral'] +
            self.kd * error_derivative +
            reference['a_y_ref']  # 前馈项
        )

        return a_y_command

    def reset_pid(self, veh_id: str):
        """重置PID状态"""
        if veh_id in self.pid_states:
            del self.pid_states[veh_id]


def map_lateral_decision_to_lane_change(lateral_decision: int,
                                        current_lane: int,
                                        num_lanes: int = 5) -> int:
    """
    将横向决策映射为目标车道

    Args:
        lateral_decision: 0=保持, 1=左换道, 2=右换道
        current_lane: 当前车道索引 (0-indexed)
        num_lanes: 总车道数

    Returns:
        target_lane: 目标车道索引
    """
    if lateral_decision == 0:  # 保持
        return current_lane
    elif lateral_decision == 1:  # 左换道
        return min(current_lane + 1, num_lanes - 1)
    elif lateral_decision == 2:  # 右换道
        return max(current_lane - 1, 0)
    else:
        return current_lane


if __name__ == "__main__":
    # 测试横向控制器
    print("=" * 80)
    print("测试横向控制器")
    print("=" * 80)

    # 创建控制器
    controller = LateralController(lane_width=3.75, default_lane_change_time=3.0)

    # 测试1：生成Bezier轨迹
    print("\n测试1: 生成Bezier轨迹 (从0米到3.75米)")
    time_array, pos_array, vel_array, acc_array = controller.compute_bezier_trajectory(
        y_0=0.0, y_target=3.75, t_LC=3.0, n_samples=30
    )

    print(f"  轨迹点数: {len(time_array)}")
    print(f"  初始位置: {pos_array[0]:.4f} m")
    print(f"  最终位置: {pos_array[-1]:.4f} m")
    print(f"  最大横向速度: {np.max(np.abs(vel_array)):.4f} m/s")
    print(f"  最大横向加速度: {np.max(np.abs(acc_array)):.4f} m/s^2")

    # 测试2：换道操作
    print("\n测试2: 开始换道操作")
    maneuver = controller.start_lane_change(
        veh_id='CAV_0',
        current_lane=2,
        target_lane=3,
        current_y_position=7.5,  # 第2车道中心
        t_LC=3.0
    )

    print(f"  车辆ID: {maneuver['veh_id']}")
    print(f"  当前车道: {maneuver['current_lane']}, 目标车道: {maneuver['target_lane']}")
    print(f"  起始位置: {maneuver['y_start']:.2f} m")
    print(f"  目标位置: {maneuver['y_target']:.2f} m")
    print(f"  换道时间: {maneuver['duration']:.2f} s")

    # 测试3：获取参考轨迹点
    print("\n测试3: 获取不同时间的参考轨迹点")
    for t in [0.0, 1.0, 2.0, 3.0]:
        ref = controller.get_reference_at_time('CAV_0', t)
        if ref:
            print(f"  t={t:.1f}s: y={ref['y_ref']:.3f}m, v_y={ref['v_y_ref']:.3f}m/s, a_y={ref['a_y_ref']:.3f}m/s^2")

    # 测试4：PID控制器
    print("\n测试4: 测试带PID的横向控制器")
    pid_controller = LateralControllerWithPID(kp=1.0, ki=0.1, kd=0.5)

    # 模拟跟踪
    current_y = 0.0
    current_v_y = 0.0
    dt = 0.1

    pid_controller.trajectory_planner.start_lane_change('CAV_1', 0, 1, current_y, t_LC=3.0)

    print("  前10步的PID控制:")
    for step in range(10):
        t = step * dt
        ref = pid_controller.trajectory_planner.get_reference_at_time('CAV_1', t)
        if ref:
            a_y_cmd = pid_controller.compute_control_command('CAV_1', current_y, current_v_y, ref, dt)

            # 简单积分更新状态
            current_v_y += a_y_cmd * dt
            current_y += current_v_y * dt

            print(f"    步骤{step}: y={current_y:.3f}m, v_y={current_v_y:.3f}m/s, a_y={a_y_cmd:.3f}m/s^2")

    print("\n✓ 横向控制器测试完成!")
