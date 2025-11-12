#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全评估模块：基于占用网格的碰撞风险评估

实现功能：
1. 占用网格表示 (10cm × 10cm精度)
2. 占用重叠率计算
3. 安全惩罚计算
4. TTC (Time To Collision) 评估
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class OccupancyGrid:
    """
    占用网格类

    使用离散网格表示车辆占用空间，用于碰撞检测
    """

    def __init__(self,
                 resolution: float = 0.1,
                 x_range: Tuple[float, float] = (0.0, 1000.0),
                 y_range: Tuple[float, float] = (-20.0, 20.0)):
        """
        初始化占用网格

        Args:
            resolution: 网格分辨率（米），默认0.1m (10cm)
            x_range: 纵向范围（米）
            y_range: 横向范围（米）
        """
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range

        # 计算网格尺寸
        self.nx = int((x_range[1] - x_range[0]) / resolution)
        self.ny = int((y_range[1] - y_range[0]) / resolution)

        # 创建网格 (sparse representation for efficiency)
        self.grid = {}  # {(i, j): vehicle_id}

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        将世界坐标转换为网格索引

        Args:
            x: 纵向坐标（米）
            y: 横向坐标（米）

        Returns:
            (i, j): 网格索引
        """
        i = int((x - self.x_range[0]) / self.resolution)
        j = int((y - self.y_range[0]) / self.resolution)
        return i, j

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """
        将网格索引转换为世界坐标（网格中心）

        Args:
            i, j: 网格索引

        Returns:
            (x, y): 世界坐标（米）
        """
        x = self.x_range[0] + (i + 0.5) * self.resolution
        y = self.y_range[0] + (j + 0.5) * self.resolution
        return x, y

    def add_vehicle(self,
                    vehicle_id: str,
                    x_center: float,
                    y_center: float,
                    length: float = 5.0,
                    width: float = 2.0,
                    heading: float = 0.0):
        """
        将车辆添加到占用网格

        Args:
            vehicle_id: 车辆ID
            x_center, y_center: 车辆中心坐标（米）
            length: 车辆长度（米）
            width: 车辆宽度（米）
            heading: 航向角（弧度）
        """
        # 计算车辆四个角点（矩形）
        half_l = length / 2.0
        half_w = width / 2.0

        corners = [
            (-half_l, -half_w),
            (-half_l, half_w),
            (half_l, half_w),
            (half_l, -half_w)
        ]

        # 旋转和平移
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        rotated_corners = []
        for dx, dy in corners:
            x_rot = x_center + dx * cos_h - dy * sin_h
            y_rot = y_center + dx * sin_h + dy * cos_h
            rotated_corners.append((x_rot, y_rot))

        # 找到包围盒
        x_coords = [c[0] for c in rotated_corners]
        y_coords = [c[1] for c in rotated_corners]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 填充网格（保守估计）
        i_min, j_min = self.world_to_grid(x_min, y_min)
        i_max, j_max = self.world_to_grid(x_max, y_max)

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    self.grid[(i, j)] = vehicle_id

    def get_occupancy_cells(self, vehicle_id: str) -> List[Tuple[int, int]]:
        """
        获取指定车辆占用的网格单元

        Args:
            vehicle_id: 车辆ID

        Returns:
            cells: 网格单元列表 [(i, j), ...]
        """
        return [cell for cell, vid in self.grid.items() if vid == vehicle_id]

    def clear(self):
        """清空网格"""
        self.grid.clear()

    def get_occupied_vehicles(self) -> Dict[str, int]:
        """
        获取所有车辆的占用单元数

        Returns:
            {vehicle_id: num_cells}
        """
        count = defaultdict(int)
        for cell, vehicle_id in self.grid.items():
            count[vehicle_id] += 1
        return dict(count)


class SafetyAssessor:
    """
    安全评估器

    计算占用重叠率和安全惩罚
    """

    def __init__(self,
                 resolution: float = 0.1,
                 vehicle_length: float = 5.0,
                 vehicle_width: float = 2.0):
        """
        初始化安全评估器

        Args:
            resolution: 占用网格分辨率（米）
            vehicle_length: 标准车辆长度（米）
            vehicle_width: 标准车辆宽度（米）
        """
        self.resolution = resolution
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        # 创建占用网格
        self.grid = OccupancyGrid(resolution=resolution)

        # 计算标准车辆占用面积（用于归一化）
        self.standard_area = vehicle_length * vehicle_width

    def compute_overlap_ratio(self,
                              ego_trajectory: np.ndarray,
                              other_trajectory: np.ndarray,
                              ego_id: str = 'ego',
                              other_id: str = 'other') -> float:
        """
        计算两条轨迹的占用重叠率

        根据00_structure.tex公式 (Eq. 19-20):
        Overlap(τ_ego, τ_other) = |O_ego ∩ O_other| / |O_ego ∪ O_other|

        Args:
            ego_trajectory: ego车辆轨迹 [N, 2] (x, y)
            other_trajectory: 其他车辆轨迹 [M, 2] (x, y)
            ego_id: ego车辆ID
            other_id: 其他车辆ID

        Returns:
            overlap_ratio: 重叠率 ∈ [0, 1]
        """
        # 清空网格
        self.grid.clear()

        # 添加ego轨迹到网格
        for point in ego_trajectory:
            x, y = point[0], point[1]
            self.grid.add_vehicle(
                ego_id, x, y,
                self.vehicle_length, self.vehicle_width, heading=0.0
            )

        # 获取ego占用单元
        ego_cells = set(self.grid.get_occupancy_cells(ego_id))

        # 清空网格，添加other轨迹
        self.grid.clear()
        for point in other_trajectory:
            x, y = point[0], point[1]
            self.grid.add_vehicle(
                other_id, x, y,
                self.vehicle_length, self.vehicle_width, heading=0.0
            )

        # 获取other占用单元
        other_cells = set(self.grid.get_occupancy_cells(other_id))

        # 计算交集和并集
        intersection = ego_cells & other_cells
        union = ego_cells | other_cells

        if len(union) == 0:
            return 0.0

        overlap_ratio = len(intersection) / len(union)

        return overlap_ratio

    def compute_safety_penalty(self,
                               ego_trajectory: np.ndarray,
                               predicted_occupancies: Dict[str, np.ndarray],
                               theta_safe: float = 0.3) -> Tuple[float, Dict]:
        """
        计算安全惩罚

        根据00_structure.tex公式 (Eq. 21):
        L_safety = Σ_j max(0, Overlap(τ_ego, τ_j^pred) - θ_safe)^2

        Args:
            ego_trajectory: ego车辆预测轨迹 [N, 2]
            predicted_occupancies: 其他车辆预测占用 {veh_id: trajectory [M, 2]}
            theta_safe: 安全阈值（默认0.3）

        Returns:
            total_penalty: 总安全惩罚
            overlap_details: 重叠详情 {veh_id: overlap_ratio}
        """
        total_penalty = 0.0
        overlap_details = {}

        for other_id, other_traj in predicted_occupancies.items():
            # 计算重叠率
            overlap = self.compute_overlap_ratio(
                ego_trajectory, other_traj,
                ego_id='ego', other_id=other_id
            )

            overlap_details[other_id] = overlap

            # 计算惩罚
            if overlap > theta_safe:
                penalty = (overlap - theta_safe) ** 2
                total_penalty += penalty

        return total_penalty, overlap_details

    def compute_ttc(self,
                   ego_x: float,
                   ego_y: float,
                   ego_vx: float,
                   ego_vy: float,
                   other_x: float,
                   other_y: float,
                   other_vx: float,
                   other_vy: float) -> float:
        """
        计算两车辆的TTC (Time To Collision)

        Args:
            ego_x, ego_y: ego车辆位置
            ego_vx, ego_vy: ego车辆速度
            other_x, other_y: 其他车辆位置
            other_vx, other_vy: 其他车辆速度

        Returns:
            ttc: 碰撞时间（秒），如果不会碰撞返回np.inf
        """
        # 相对位置
        dx = other_x - ego_x
        dy = other_y - ego_y

        # 相对速度
        dvx = other_vx - ego_vx
        dvy = other_vy - ego_vy

        # 检查是否接近
        rel_speed_sq = dvx**2 + dvy**2
        if rel_speed_sq < 1e-6:
            # 相对速度接近0
            distance = np.sqrt(dx**2 + dy**2)
            return distance / 1e-3 if distance < 10.0 else np.inf

        # 最近接近时间
        t_cpa = -(dx * dvx + dy * dvy) / rel_speed_sq

        if t_cpa < 0:
            # 正在远离
            return np.inf

        # 最近接近距离
        x_cpa = dx + dvx * t_cpa
        y_cpa = dy + dvy * t_cpa
        d_cpa = np.sqrt(x_cpa**2 + y_cpa**2)

        # 碰撞阈值（车辆宽度+长度的一半）
        collision_threshold = (self.vehicle_length + self.vehicle_width) / 2.0

        if d_cpa < collision_threshold:
            return t_cpa
        else:
            return np.inf

    def batch_ttc_assessment(self,
                            ego_state: Dict,
                            surrounding_states: Dict[str, Dict]) -> Dict[str, float]:
        """
        批量计算ego车辆与周边车辆的TTC

        Args:
            ego_state: ego车辆状态 {'x', 'y', 'vx', 'vy'}
            surrounding_states: 周边车辆状态 {veh_id: {'x', 'y', 'vx', 'vy'}}

        Returns:
            ttc_dict: {veh_id: ttc}
        """
        ttc_dict = {}

        for other_id, other_state in surrounding_states.items():
            ttc = self.compute_ttc(
                ego_state['x'], ego_state['y'],
                ego_state['vx'], ego_state['vy'],
                other_state['x'], other_state['y'],
                other_state['vx'], other_state['vy']
            )
            ttc_dict[other_id] = ttc

        return ttc_dict


def compute_trajectory_safety_score(ego_trajectory: np.ndarray,
                                    predicted_occupancies: Dict[str, np.ndarray],
                                    theta_safe: float = 0.3,
                                    resolution: float = 0.1) -> Dict:
    """
    计算轨迹安全得分（便捷函数）

    Args:
        ego_trajectory: ego轨迹 [N, 2]
        predicted_occupancies: 周边车辆预测占用 {veh_id: traj [M, 2]}
        theta_safe: 安全阈值
        resolution: 网格分辨率

    Returns:
        safety_info: {
            'penalty': 安全惩罚,
            'max_overlap': 最大重叠率,
            'num_conflicts': 冲突车辆数,
            'overlap_details': 重叠详情
        }
    """
    assessor = SafetyAssessor(resolution=resolution)

    penalty, overlap_details = assessor.compute_safety_penalty(
        ego_trajectory, predicted_occupancies, theta_safe
    )

    max_overlap = max(overlap_details.values()) if overlap_details else 0.0
    num_conflicts = sum(1 for overlap in overlap_details.values() if overlap > theta_safe)

    return {
        'penalty': penalty,
        'max_overlap': max_overlap,
        'num_conflicts': num_conflicts,
        'overlap_details': overlap_details
    }


if __name__ == "__main__":
    # 测试安全评估模块
    print("=" * 80)
    print("测试安全评估模块")
    print("=" * 80)

    # 测试1: 占用网格
    print("\n测试1: 占用网格")
    grid = OccupancyGrid(resolution=0.1)

    # 添加两辆车
    grid.add_vehicle('CAV_0', x_center=100.0, y_center=3.75, length=5.0, width=2.0)
    grid.add_vehicle('HDV_0', x_center=105.0, y_center=3.75, length=5.0, width=2.0)

    occupancy = grid.get_occupied_vehicles()
    print(f"  CAV_0 占用单元数: {occupancy.get('CAV_0', 0)}")
    print(f"  HDV_0 占用单元数: {occupancy.get('HDV_0', 0)}")

    # 测试2: 重叠率计算
    print("\n测试2: 重叠率计算")
    assessor = SafetyAssessor(resolution=0.1)

    # 情况1: 不重叠
    ego_traj_1 = np.array([
        [100.0, 3.75],
        [101.0, 3.75],
        [102.0, 3.75]
    ])
    other_traj_1 = np.array([
        [110.0, 3.75],
        [111.0, 3.75],
        [112.0, 3.75]
    ])

    overlap_1 = assessor.compute_overlap_ratio(ego_traj_1, other_traj_1)
    print(f"  情况1 (不重叠): 重叠率 = {overlap_1:.4f}")

    # 情况2: 完全重叠
    ego_traj_2 = np.array([[100.0, 3.75]])
    other_traj_2 = np.array([[100.0, 3.75]])

    overlap_2 = assessor.compute_overlap_ratio(ego_traj_2, other_traj_2)
    print(f"  情况2 (完全重叠): 重叠率 = {overlap_2:.4f}")

    # 情况3: 部分重叠
    ego_traj_3 = np.array([
        [100.0, 3.75],
        [102.0, 3.75],
        [104.0, 3.75]
    ])
    other_traj_3 = np.array([
        [103.0, 3.75],
        [105.0, 3.75],
        [107.0, 3.75]
    ])

    overlap_3 = assessor.compute_overlap_ratio(ego_traj_3, other_traj_3)
    print(f"  情况3 (部分重叠): 重叠率 = {overlap_3:.4f}")

    # 测试3: 安全惩罚
    print("\n测试3: 安全惩罚计算")
    ego_trajectory = np.array([
        [100.0 + i, 3.75] for i in range(10)
    ])

    predicted_occupancies = {
        'HDV_0': np.array([[105.0 + i, 3.75] for i in range(10)]),  # 部分重叠
        'HDV_1': np.array([[120.0 + i, 3.75] for i in range(10)])   # 不重叠
    }

    penalty, details = assessor.compute_safety_penalty(
        ego_trajectory, predicted_occupancies, theta_safe=0.3
    )

    print(f"  总安全惩罚: {penalty:.6f}")
    for veh_id, overlap in details.items():
        print(f"    {veh_id}: 重叠率 = {overlap:.4f}")

    # 测试4: TTC计算
    print("\n测试4: TTC计算")

    # 情况1: 迎面接近
    ttc1 = assessor.compute_ttc(
        ego_x=100.0, ego_y=3.75, ego_vx=20.0, ego_vy=0.0,
        other_x=150.0, other_y=3.75, other_vx=-20.0, other_vy=0.0
    )
    print(f"  情况1 (迎面接近): TTC = {ttc1:.2f} s")

    # 情况2: 追尾接近
    ttc2 = assessor.compute_ttc(
        ego_x=100.0, ego_y=3.75, ego_vx=25.0, ego_vy=0.0,
        other_x=120.0, other_y=3.75, other_vx=20.0, other_vy=0.0
    )
    print(f"  情况2 (追尾接近): TTC = {ttc2:.2f} s")

    # 情况3: 远离
    ttc3 = assessor.compute_ttc(
        ego_x=100.0, ego_y=3.75, ego_vx=20.0, ego_vy=0.0,
        other_x=120.0, other_y=3.75, other_vx=25.0, other_vy=0.0
    )
    print(f"  情况3 (远离): TTC = {ttc3:.2f} s" if ttc3 < 1e6 else "  情况3 (远离): TTC = ∞")

    # 测试5: 批量TTC
    print("\n测试5: 批量TTC评估")
    ego_state = {'x': 100.0, 'y': 3.75, 'vx': 20.0, 'vy': 0.0}
    surrounding_states = {
        'HDV_0': {'x': 110.0, 'y': 3.75, 'vx': 18.0, 'vy': 0.0},
        'HDV_1': {'x': 90.0, 'y': 3.75, 'vx': 22.0, 'vy': 0.0},
        'HDV_2': {'x': 120.0, 'y': 7.5, 'vx': 20.0, 'vy': 0.0}
    }

    ttc_dict = assessor.batch_ttc_assessment(ego_state, surrounding_states)
    for veh_id, ttc in ttc_dict.items():
        ttc_str = f"{ttc:.2f} s" if ttc < 100.0 else "∞"
        print(f"  {veh_id}: TTC = {ttc_str}")

    # 测试6: 轨迹安全得分
    print("\n测试6: 轨迹安全得分")
    safety_info = compute_trajectory_safety_score(
        ego_trajectory, predicted_occupancies, theta_safe=0.3
    )
    print(f"  安全惩罚: {safety_info['penalty']:.6f}")
    print(f"  最大重叠率: {safety_info['max_overlap']:.4f}")
    print(f"  冲突车辆数: {safety_info['num_conflicts']}")

    print("\n✓ 安全评估模块测试完成!")
