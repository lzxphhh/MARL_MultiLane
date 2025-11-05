# probability_generator.py
"""
概率分布生成器模块
Probability Distribution Generator Module
负责生成车辆位置的概率分布
"""

import numpy as np
from typing import Dict, Tuple

from numpy import ndarray

from harl.envs.a_multi_lane.project_structure import SystemParameters, VehicleType, Direction


class ProbabilityDistributionGenerator:
    """概率分布生成器"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化概率分布生成器

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params

    def discretize_space_time(self,
                              prediction_horizon: float,
                              x_min: float,
                              x_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        离散化时空网格

        Args:
            prediction_horizon: 预测周期 [s]
            x_min: 最小位置 [m]
            x_max: 最大位置 [m]

        Returns:
            (时间网格, 空间网格)
        """
        # 时间离散化
        time_steps = int(prediction_horizon / self.params.dt) + 1
        time_grid = np.linspace(0, prediction_horizon, time_steps)

        # 空间离散化
        dx = 2.5  # 空间分辨率 [m]
        x_steps = int((x_max - x_min) / dx) + 1
        space_grid = np.linspace(x_min, x_max, x_steps)

        return time_grid, space_grid

    def _check_interval_overlap(self, grid_start: float, grid_end: float,
                                condition_start: float, condition_end: float) -> bool:
        """
        检查两个区间是否有重叠

        Args:
            grid_start: 网格区间起始点
            grid_end: 网格区间结束点
            condition_start: 条件区间起始点
            condition_end: 条件区间结束点

        Returns:
            True if 有重叠, False otherwise
        """
        return not (grid_end < condition_start or grid_start > condition_end)

    def _compute_overlap_ratio(self, grid_start: float, grid_end: float,
                               condition_start: float, condition_end: float) -> float:
        """
        计算网格区间与条件区间的重叠比例

        Args:
            grid_start: 网格区间起始点
            grid_end: 网格区间结束点
            condition_start: 条件区间起始点
            condition_end: 条件区间结束点

        Returns:
            重叠比例 [0, 1]
        """
        if not self._check_interval_overlap(grid_start, grid_end, condition_start, condition_end):
            return 0.0

        overlap_start = max(grid_start, condition_start)
        overlap_end = min(grid_end, condition_end)
        overlap_length = overlap_end - overlap_start
        grid_length = grid_end - grid_start

        return overlap_length / grid_length if grid_length > 0 else 0.0

    def compute_comfort_probability_distribution(self,
                                                 x_comfort_min: np.ndarray,
                                                 x_comfort_max: np.ndarray,
                                                 x_center: np.ndarray,
                                                 time_grid: np.ndarray,
                                                 space_grid: np.ndarray) -> np.ndarray:
        """
        计算舒适性概率分布

        Args:
            x_comfort_min: 舒适性最小位置数组
            x_comfort_max: 舒适性最大位置数组
            x_center: 中心位置数组
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            概率分布矩阵 [时间, 空间]
        """
        alpha_ext = self.params.alpha_ext
        beta_ext = self.params.beta_ext
        dx = space_grid[1] - space_grid[0] if len(space_grid) > 1 else 0.5

        prob_matrix = np.zeros((len(time_grid), len(space_grid)))

        for t_idx, t in enumerate(time_grid):
            if t_idx >= len(x_comfort_min):
                continue

            x_min_t = x_comfort_min[t_idx]
            x_max_t = x_comfort_max[t_idx]
            x_center_t = x_center[t_idx]

            # 舒适性区域范围
            comfort_range = (x_max_t - x_min_t) / 2

            if comfort_range < 0:
                continue

            # 计算舒适性概率分布
            for x_idx, x_h in enumerate(space_grid):
                # 定义当前网格的区间范围
                if x_idx == 0:
                    grid_start = x_h - dx / 2
                    grid_end = x_h + dx / 2
                else:
                    grid_start = space_grid[x_idx - 1] + dx / 2
                    grid_end = x_h + dx / 2

                # 检查网格区间与舒适性区间是否有重叠
                if self._check_interval_overlap(grid_start, grid_end, x_min_t, x_max_t):
                    # 计算重叠比例
                    overlap_ratio = self._compute_overlap_ratio(grid_start, grid_end, x_min_t, x_max_t)

                    # 使用网格中心点计算概率密度
                    grid_center = (grid_start + grid_end) / 2
                    if comfort_range == 0:
                        normalized_distance = 1
                    else:
                        distance = abs(grid_center - x_center_t) if abs(grid_center - x_center_t) < comfort_range else comfort_range
                        normalized_distance = distance / comfort_range
                    phi_value = np.exp(-alpha_ext * (normalized_distance ** beta_ext))

                    # 按重叠比例调整概率
                    prob_matrix[t_idx, x_idx] = phi_value

        # 按时间步归一化
        total_probs = np.zeros(len(time_grid))
        for t_idx in range(len(time_grid)):
            total_prob = np.sum(prob_matrix[t_idx, :])
            total_probs[t_idx] = total_prob
            if total_prob > 0:
                prob_matrix[t_idx, :] *= 0.95 / total_prob  # 舒适性区域占95%概率

        return prob_matrix

    def compute_extended_probability_distribution(self,
                                                  x_physical_min: np.ndarray,
                                                  x_physical_max: np.ndarray,
                                                  x_comfort_min: np.ndarray,
                                                  x_comfort_max: np.ndarray,
                                                  time_grid: np.ndarray,
                                                  space_grid: np.ndarray) -> np.ndarray:
        """
        计算扩展区域概率分布（物理包络 - 舒适性包络）

        Args:
            x_physical_min: 物理最小位置数组
            x_physical_max: 物理最大位置数组
            x_comfort_min: 舒适性最小位置数组
            x_comfort_max: 舒适性最大位置数组
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            扩展区域概率分布矩阵
        """
        dx = space_grid[1] - space_grid[0] if len(space_grid) > 1 else 0.5
        prob_matrix = np.zeros((len(time_grid), len(space_grid)))

        for t_idx, t in enumerate(time_grid):
            if t_idx >= len(x_physical_min):
                continue

            x_phys_min_t = x_physical_min[t_idx]
            x_phys_max_t = x_physical_max[t_idx]
            x_comfort_min_t = x_comfort_min[t_idx]
            x_comfort_max_t = x_comfort_max[t_idx]

            # 计算扩展区域长度
            extended_length = ((x_phys_max_t - x_phys_min_t) -
                               (x_comfort_max_t - x_comfort_min_t))

            if extended_length <= 0:
                continue

            # 均匀分布在扩展区域
            uniform_prob = 0.05 / extended_length  # 扩展区域占5%概率

            for x_idx, x_h in enumerate(space_grid):
                # 定义当前网格的区间范围
                if x_idx == 0:
                    grid_start = x_h - dx / 2
                    grid_end = x_h + dx / 2
                else:
                    grid_start = space_grid[x_idx - 1] + dx / 2 if x_idx > 0 else x_h - dx / 2
                    grid_end = x_h + dx / 2

                # 检查网格区间是否在物理包络内
                in_physical = self._check_interval_overlap(grid_start, grid_end, x_phys_min_t, x_phys_max_t)

                # 检查网格区间是否在舒适性包络内
                in_comfort = self._check_interval_overlap(grid_start, grid_end, x_comfort_min_t, x_comfort_max_t)

                if in_physical and not in_comfort:
                    # 在物理包络内但不在舒适性包络内（扩展区域）
                    # 计算与物理包络的重叠比例
                    physical_overlap = self._compute_overlap_ratio(grid_start, grid_end, x_phys_min_t, x_phys_max_t)

                    # 如果与舒适性包络有部分重叠，需要减去重叠部分
                    if in_comfort:
                        comfort_overlap = self._compute_overlap_ratio(grid_start, grid_end, x_comfort_min_t,
                                                                      x_comfort_max_t)
                        extended_overlap = physical_overlap - comfort_overlap
                    else:
                        extended_overlap = physical_overlap

                    if extended_overlap > 0:
                        prob_matrix[t_idx, x_idx] = uniform_prob * extended_overlap * dx

        return prob_matrix

    def compute_total_probability_distribution(self,
                                               envelope_data: Dict,
                                               time_grid: ndarray,
                                               space_grid: ndarray,
                                               vehicle_type: VehicleType = VehicleType.HDV) -> Dict:
        """
        计算总概率分布（舒适性 + 扩展区域）

        Args:
            envelope_data: 包络线数据
            vehicle_type: 车辆类型

        Returns:
            概率分布字典
        """
        x_ref = envelope_data['reference']['position']

        # # 确定空间范围
        # time_array = envelope_data['time']
        # x_min = min(np.min(envelope_data['physical']['min']),
        #             np.min(envelope_data['comfort']['min']))
        # x_max = max(np.max(envelope_data['physical']['max']),
        #             np.max(envelope_data['comfort']['max']))
        #
        # # 离散化时空网格
        # time_grid, space_grid = self.discretize_space_time(
        #     time_array[-1], x_min - 10, x_max + 10
        # )

        # 计算舒适性概率分布
        comfort_prob = self.compute_comfort_probability_distribution(
            envelope_data['comfort']['min'],
            envelope_data['comfort']['max'],
            x_ref,
            time_grid,
            space_grid
        )

        # 计算扩展区域概率分布
        extended_prob = self.compute_extended_probability_distribution(
            envelope_data['physical']['min'],
            envelope_data['physical']['max'],
            envelope_data['comfort']['min'],
            envelope_data['comfort']['max'],
            time_grid,
            space_grid
        )

        # 总概率分布
        total_prob = comfort_prob + extended_prob

        return {
            'time_grid': time_grid,
            'space_grid': space_grid,
            'comfort_probability': comfort_prob,
            'extended_probability': extended_prob,
            'total_probability': total_prob,
            'envelope_data': envelope_data
        }

    def compute_ego_vehicle_distribution(self,
                                         x_ego_min: np.ndarray,
                                         x_ego_max: np.ndarray,
                                         time_grid: np.ndarray,
                                         space_grid: np.ndarray) -> np.ndarray:
        """
        计算ego车辆的占用概率分布（确定性）

        Args:
            x_ego_min: ego车辆最小占用位置
            x_ego_max: ego车辆最大占用位置
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            ego车辆概率分布矩阵
        """
        prob_matrix = np.zeros((len(time_grid), len(space_grid)))

        for t_idx in range(len(time_grid)):
            if t_idx >= len(x_ego_min):
                continue

            x_min_t = x_ego_min[t_idx]
            x_max_t = x_ego_max[t_idx]

            for x_idx, x in enumerate(space_grid):
                if x_min_t <= x <= x_max_t:
                    prob_matrix[t_idx, x_idx] = 1.0

        return prob_matrix

    def apply_causal_modulation(self,
                                probability_distribution: np.ndarray,
                                cooperation_capability: float,
                                conflict_regions: Dict,
                                time_grid: np.ndarray,
                                space_grid: np.ndarray) -> np.ndarray:
        """
        应用因果调制

        Args:
            probability_distribution: 原始概率分布
            cooperation_capability: 协同能力
            conflict_regions: 冲突区域字典
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            调制后的概率分布
        """
        modulated_prob = probability_distribution.copy()

        # 避让调制因子
        avoidance_factor = 1.0 - cooperation_capability

        for t_idx, t in enumerate(time_grid):
            if t in conflict_regions:
                conflict_region = conflict_regions[t]

                for x_idx, x in enumerate(space_grid):
                    if conflict_region['x_min'] <= x <= conflict_region['x_max']:
                        # 在冲突区域内，降低概率
                        modulated_prob[t_idx, x_idx] *= avoidance_factor

        # 重新归一化
        for t_idx in range(len(time_grid)):
            total_prob = np.sum(modulated_prob[t_idx, :])
            if total_prob > 0:
                modulated_prob[t_idx, :] /= total_prob

        return modulated_prob

    def compute_lateral_influence_coefficient(self,
                                              ego_y: float,
                                              target_lane_y: float,
                                              lane_width: float) -> float:
        """
        计算横向位置影响系数

        Args:
            ego_y: ego车辆横向位置
            target_lane_y: 目标车道中心线位置
            lane_width: 车道宽度

        Returns:
            横向影响系数
        """
        beta_lane = self.params.beta_lane

        distance_to_center = abs(ego_y - target_lane_y)

        if distance_to_center < lane_width / 2:
            return 1.0
        else:
            excess_distance = distance_to_center - lane_width / 2
            normalized_distance = excess_distance / lane_width
            return np.exp(-beta_lane * normalized_distance)

    def compute_temporal_influence_coefficient(self,
                                               current_time: float,
                                               prediction_horizon: float) -> float:
        """
        计算时间影响系数

        Args:
            current_time: 当前时间
            prediction_horizon: 预测周期

        Returns:
            时间影响系数
        """
        alpha_time = self.params.alpha_time

        # Sigmoid函数：预测时间越长，置信度越低
        return 1.0 / (1.0 + np.exp(alpha_time * (current_time - prediction_horizon)))

    def extract_significant_overlap_regions(self,
                                            prob_distribution_1: np.ndarray,
                                            prob_distribution_2: np.ndarray,
                                            time_grid: np.ndarray,
                                            space_grid: np.ndarray,
                                            threshold_1: float = 0.05,
                                            threshold_2: float = 0.05) -> Dict:
        """
        提取概率显著性重叠区域

        Args:
            prob_distribution_1: 第一个车辆的概率分布
            prob_distribution_2: 第二个车辆的概率分布
            time_grid: 时间网格
            space_grid: 空间网格
            threshold_1: 车辆1概率阈值
            threshold_2: 车辆2概率阈值

        Returns:
            重叠区域字典
        """
        overlap_regions = {}

        for t_idx, t in enumerate(time_grid):
            significant_overlap_indices = []

            for x_idx, x in enumerate(space_grid):
                if (prob_distribution_1[t_idx, x_idx] > threshold_1 and
                        prob_distribution_2[t_idx, x_idx] > threshold_2):
                    significant_overlap_indices.append(x_idx)

            if significant_overlap_indices:
                overlap_regions[t] = {
                    'x_indices': significant_overlap_indices,
                    'x_positions': space_grid[significant_overlap_indices],
                    'x_min': space_grid[significant_overlap_indices[0]],
                    'x_max': space_grid[significant_overlap_indices[-1]]
                }

        return overlap_regions

    def compute_vehicle_cooperation_intent(self,
                                           vehicle_type: VehicleType,
                                           direction: Direction) -> float:
        """
        计算车辆协同意图

        Args:
            vehicle_type: 车辆类型
            direction: 相对方向

        Returns:
            协同意图值 [0, 1]
        """
        if vehicle_type == VehicleType.CAV:
            # CAV始终具有协同意图
            return 1.0
        elif vehicle_type == VehicleType.HDV:
            # HDV根据位置决定协同意图
            if direction in [Direction.LR, Direction.ER, Direction.RR]:
                # 后方HDV会根据前方车辆行为做适应性调整
                return 0.5
            else:
                # 前方HDV不会主动避让
                return 0.0
        else:
            return 0.0

    def generate_multi_vehicle_probability_distributions(self,
                                                         vehicle_envelopes: Dict[str, Dict],
                                                         vehicle_types: Dict[str, VehicleType],
                                                         vehicle_directions: Dict[str, Direction],
                                                         ego_trajectory: Dict,
                                                         time_grid: ndarray,
                                                         space_grid: ndarray) -> Dict:
        """
        生成多车辆概率分布

        Args:
            vehicle_envelopes: 各车辆的包络线数据
            vehicle_types: 各车辆的类型
            vehicle_directions: 各车辆的相对方向
            ego_envelope: ego车辆包络线数据

        Returns:
            多车辆概率分布字典
        """
        results = {}

        # 生成每个车辆的概率分布
        for vehicle_id, envelope_data in vehicle_envelopes.items():
            vehicle_type = vehicle_types[vehicle_id]
            direction = vehicle_directions[vehicle_id]

            # 计算协同意图
            cooperation_intent = self.compute_vehicle_cooperation_intent(vehicle_type, direction)

            # 生成概率分布
            prob_dist = self.compute_total_probability_distribution(
                envelope_data, time_grid, space_grid, vehicle_type
            )

            # 应用因果调制（如果有协同意图）
            if cooperation_intent > 0:
                # 这里需要冲突区域信息，暂时使用空字典
                conflict_regions = {}
                modulated_prob = self.apply_causal_modulation(
                    prob_dist['total_probability'],
                    cooperation_intent,
                    conflict_regions,
                    prob_dist['time_grid'],
                    prob_dist['space_grid']
                )
                prob_dist['modulated_probability'] = modulated_prob

            results[vehicle_id] = {
                'probability_distribution': prob_dist,
                'cooperation_intent': cooperation_intent,
                'vehicle_type': vehicle_type,
                'direction': direction
            }

        # 生成ego车辆分布
        ego_time_grid = ego_trajectory['time']
        ego_x_ref = ego_trajectory['longitudinal']['position']

        # 计算ego车辆占用区域
        vehicle_length = self.params.vehicle_length
        safe_distance = self.params.safe_distance

        ego_x_min = ego_x_ref - vehicle_length - safe_distance
        ego_x_max = ego_x_ref + vehicle_length + safe_distance

        # 使用第一个邻接车辆的空间网格（假设所有车辆使用相同网格）
        if results:
            first_vehicle_data = list(results.values())[0]
            space_grid = first_vehicle_data['probability_distribution']['space_grid']
            time_grid = first_vehicle_data['probability_distribution']['time_grid']
        else:
            # 如果没有邻接车辆，创建默认网格
            time_grid, space_grid = self.discretize_space_time(
                ego_time_grid[-1],
                np.min(ego_x_min) - 10,
                np.max(ego_x_max) + 10
            )

        ego_prob_dist = self.compute_ego_vehicle_distribution(
            ego_x_min, ego_x_max, time_grid, space_grid
        )

        results['ego'] = {
            'probability_distribution': {
                'time_grid': time_grid,
                'space_grid': space_grid,
                'total_probability': ego_prob_dist
            },
            'cooperation_intent': 1.0,  # ego车始终有协同意图
            'vehicle_type': VehicleType.CAV,
            'direction': 'ego'
        }

        return results