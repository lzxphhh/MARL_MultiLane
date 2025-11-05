# cooperation_evaluator.py
"""
协同能力评估器模块
Cooperation Capability Evaluator Module
负责评估周围车辆的协同能力和避让行为
"""

import numpy as np
from typing import Dict
from harl.envs.a_multi_lane.project_structure import SystemParameters


class CooperationCapabilityEvaluator:
    """协同能力评估器"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化协同能力评估器

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params

    def compute_overlap_regions(self,
                                ego_prob_dist: np.ndarray,
                                neighbor_prob_dist: np.ndarray,
                                space_grid: np.ndarray,
                                vehicle_length: float = None,
                                safe_distance: float = None) -> Dict:
        """
        计算重叠区域

        Args:
            ego_prob_dist: ego车辆概率分布
            neighbor_prob_dist: 邻接车辆概率分布
            space_grid: 空间网格
            vehicle_length: 车辆长度
            safe_distance: 安全距离

        Returns:
            重叠区域字典
        """
        if vehicle_length is None:
            vehicle_length = self.params.vehicle_length
        if safe_distance is None:
            safe_distance = self.params.safe_distance

        overlap_regions = {}

        # 对每个时间步计算重叠区域
        for t_idx in range(ego_prob_dist.shape[0]):
            overlap_indices = []

            for x_idx in range(len(space_grid)):
                # 考虑车辆体积和安全间距
                ego_prob = ego_prob_dist[t_idx, x_idx]
                neighbor_prob = neighbor_prob_dist[t_idx, x_idx]

                # 如果两个概率都显著，则认为存在重叠
                if (ego_prob > self.params.prob_threshold_1 and
                        neighbor_prob > self.params.prob_threshold_2):
                    overlap_indices.append(x_idx)

            if overlap_indices:
                overlap_regions[t_idx] = {
                    'x_indices': overlap_indices,
                    'x_positions': space_grid[overlap_indices],
                    'x_min': space_grid[overlap_indices[0]] - vehicle_length - safe_distance,
                    'x_max': space_grid[overlap_indices[-1]] + vehicle_length + safe_distance,
                    'center': np.mean(space_grid[overlap_indices])
                }

        return overlap_regions

    def compute_secondary_overlap_regions(self,
                                          primary_neighbor_prob: np.ndarray,
                                          secondary_neighbor_prob: np.ndarray,
                                          space_grid: np.ndarray) -> Dict:
        """
        计算次重叠区域（主邻接车辆与次邻接车辆之间）

        Args:
            primary_neighbor_prob: 主邻接车辆概率分布
            secondary_neighbor_prob: 次邻接车辆概率分布
            space_grid: 空间网格

        Returns:
            次重叠区域字典
        """
        vehicle_length = self.params.vehicle_length
        safe_distance = self.params.safe_distance

        secondary_overlap_regions = {}

        for t_idx in range(primary_neighbor_prob.shape[0]):
            overlap_indices = []

            for x_idx in range(len(space_grid)):
                primary_prob = primary_neighbor_prob[t_idx, x_idx]
                secondary_prob = secondary_neighbor_prob[t_idx, x_idx]

                if (primary_prob > self.params.prob_threshold_1 and
                        secondary_prob > self.params.prob_threshold_2):
                    overlap_indices.append(x_idx)

            if overlap_indices:
                # 主邻接车辆的影响区域需要考虑车辆体积
                x_min_extended = space_grid[overlap_indices[0]] - vehicle_length - safe_distance
                x_max_extended = space_grid[overlap_indices[-1]] + vehicle_length + safe_distance

                secondary_overlap_regions[t_idx] = {
                    'x_indices': overlap_indices,
                    'x_positions': space_grid[overlap_indices],
                    'x_min': x_min_extended,
                    'x_max': x_max_extended,
                    'center': np.mean(space_grid[overlap_indices])
                }

        return secondary_overlap_regions

    def compute_conflict_probability(self,
                                     ego_prob_dist: np.ndarray,
                                     neighbor_prob_dist: np.ndarray,
                                     overlap_regions: Dict,
                                     cooperation_intent: float,
                                     time_grid: np.ndarray,
                                     space_grid: np.ndarray) -> Dict:
        """
        计算冲突概率

        Args:
            ego_prob_dist: ego车辆概率分布
            neighbor_prob_dist: 邻接车辆概率分布
            overlap_regions: 重叠区域
            cooperation_intent: 协同意图
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            冲突概率字典
        """
        conflict_probs = {}

        for t_idx, t in enumerate(time_grid):
            if t_idx in overlap_regions:
                overlap_region = overlap_regions[t_idx]
                conflict_prob = 0.0

                # 在重叠区域内计算冲突概率
                for x_idx in overlap_region['x_indices']:
                    if x_idx < len(space_grid):
                        ego_prob = ego_prob_dist[t_idx, x_idx]
                        neighbor_prob = neighbor_prob_dist[t_idx, x_idx]

                        # 冲突概率 = ego概率 × 邻接车辆概率 × (1 - 协同意图)
                        local_conflict_prob = ego_prob * neighbor_prob * (1 - cooperation_intent)
                        conflict_prob += local_conflict_prob

                # 应用时间影响系数
                time_coeff = self.compute_temporal_influence_coefficient(t, time_grid[-1])
                conflict_prob *= time_coeff

                conflict_probs[t_idx] = {
                    'time': t,
                    'conflict_probability': conflict_prob,
                    'overlap_region': overlap_region,
                    'time_coefficient': time_coeff
                }

        return conflict_probs

    def compute_temporal_influence_coefficient(self,
                                               current_time: float,
                                               prediction_horizon: float) -> float:
        """
        计算时间影响系数

        Args:
            current_time: 当前时间
            prediction_horizon: 预测周期

        Returns:
            时间影响系数 [0.5, 1]
        """
        alpha_time = self.params.alpha_time
        return 1.0 / (1.0 + np.exp(alpha_time * (current_time - prediction_horizon)))

    def evaluate_primary_neighbor_cooperation(self,
                                              ego_prob_dist: np.ndarray,
                                              primary_prob_dist: np.ndarray,
                                              secondary_prob_dist: np.ndarray,
                                              primary_cooperation_intent: float,
                                              secondary_cooperation_intent: float,
                                              time_grid: np.ndarray,
                                              space_grid: np.ndarray) -> Dict:
        """
        评估主邻接车辆的协同能力

        Args:
            ego_prob_dist: ego车辆概率分布
            primary_prob_dist: 主邻接车辆概率分布
            secondary_prob_dist: 次邻接车辆概率分布
            primary_cooperation_intent: 主邻接车辆协同意图
            secondary_cooperation_intent: 次邻接车辆协同意图
            time_grid: 时间网格
            space_grid: 空间网格

        Returns:
            协同能力评估结果
        """
        # 1. 计算主重叠区域（ego与主邻接车辆）
        primary_overlap_regions = self.compute_overlap_regions(
            ego_prob_dist, primary_prob_dist, space_grid
        )

        # 2. 计算次重叠区域（主邻接车辆与次邻接车辆）
        secondary_overlap_regions = self.compute_secondary_overlap_regions(
            primary_prob_dist, secondary_prob_dist, space_grid
        )

        # 3. 计算主重叠冲突概率
        primary_conflicts = self.compute_conflict_probability(
            ego_prob_dist, primary_prob_dist, primary_overlap_regions,
            primary_cooperation_intent, time_grid, space_grid
        )

        # 4. 计算次重叠冲突概率
        secondary_conflicts = self.compute_conflict_probability(
            primary_prob_dist, secondary_prob_dist, secondary_overlap_regions,
            secondary_cooperation_intent, time_grid, space_grid
        )

        # 5. 计算每个时刻的协同能力
        cooperation_capability_timeline = {}

        for t_idx, t in enumerate(time_grid):
            primary_conflict_prob = 0.0
            secondary_conflict_prob = 0.0

            if t_idx in primary_conflicts:
                primary_conflict_prob = primary_conflicts[t_idx]['conflict_probability']

            if t_idx in secondary_conflicts:
                secondary_conflict_prob = secondary_conflicts[t_idx]['conflict_probability']

            # 协同能力 = 1 - 总冲突概率
            total_conflict_prob = primary_conflict_prob + secondary_conflict_prob
            cooperation_capability = 1.0 - min(total_conflict_prob, 1.0)

            cooperation_capability_timeline[t_idx] = {
                'time': t,
                'cooperation_capability': cooperation_capability,
                'primary_conflict_prob': primary_conflict_prob,
                'secondary_conflict_prob': secondary_conflict_prob,
                'total_conflict_prob': total_conflict_prob
            }

        # 6. 计算整个预测周期的综合协同能力
        if cooperation_capability_timeline:
            min_capability = min([
                data['cooperation_capability']
                for data in cooperation_capability_timeline.values()
            ])
        else:
            min_capability = 1.0  # 无冲突情况

        return {
            'overall_cooperation_capability': min_capability,
            'timeline_cooperation_capability': cooperation_capability_timeline,
            'primary_overlap_regions': primary_overlap_regions,
            'secondary_overlap_regions': secondary_overlap_regions,
            'primary_conflicts': primary_conflicts,
            'secondary_conflicts': secondary_conflicts
        }

    def evaluate_multi_vehicle_cooperation(self,
                                           vehicle_probability_data: Dict) -> Dict:
        """
        评估多车辆协同能力

        Args:
            vehicle_probability_data: 多车辆概率分布数据

        Returns:
            多车辆协同能力评估结果
        """
        results = {}

        ego_data = vehicle_probability_data.get('ego', {})
        ego_prob_dist = ego_data.get('probability_distribution', {}).get('total_probability')

        if ego_prob_dist is None:
            return results

        time_grid = ego_data['probability_distribution']['time_grid']
        space_grid = ego_data['probability_distribution']['space_grid']

        # 按方向分组车辆
        direction_groups = {}
        for vehicle_id, vehicle_data in vehicle_probability_data.items():
            if vehicle_id == 'ego':
                continue

            direction = vehicle_data['direction']
            if direction not in direction_groups:
                direction_groups[direction] = []
            direction_groups[direction].append((vehicle_id, vehicle_data))

        # 评估每个方向的协同能力
        for direction, vehicles in direction_groups.items():
            if len(vehicles) == 0:
                continue

            # 找出主邻接车辆和次邻接车辆
            primary_vehicle = None
            secondary_vehicle = None

            for vehicle_id, vehicle_data in vehicles:
                if vehicle_id[3:] == '1':
                    primary_vehicle = (vehicle_id, vehicle_data)
                elif vehicle_id[3:] == '2':
                    secondary_vehicle = (vehicle_id, vehicle_data)

            # 如果只有一个车辆，将其视为主邻接车辆
            if primary_vehicle is None and vehicles:
                primary_vehicle = vehicles[0]

            if primary_vehicle:
                primary_id, primary_data = primary_vehicle
                primary_prob_dist = primary_data['probability_distribution']['total_probability']
                primary_cooperation_intent = primary_data['cooperation_intent']

                # 次邻接车辆数据
                if secondary_vehicle:
                    secondary_id, secondary_data = secondary_vehicle
                    secondary_prob_dist = secondary_data['probability_distribution']['total_probability']
                    secondary_cooperation_intent = secondary_data['cooperation_intent']
                else:
                    # 如果没有次邻接车辆，创建零概率分布
                    secondary_prob_dist = np.zeros_like(primary_prob_dist)
                    secondary_cooperation_intent = 0.0

                # 评估协同能力
                cooperation_result = self.evaluate_primary_neighbor_cooperation(
                    ego_prob_dist, primary_prob_dist, secondary_prob_dist,
                    primary_cooperation_intent, secondary_cooperation_intent,
                    time_grid, space_grid
                )

                results[direction] = {
                    'primary_vehicle': primary_id if primary_vehicle else None,
                    'secondary_vehicle': secondary_vehicle[0] if secondary_vehicle else None,
                    'cooperation_evaluation': cooperation_result
                }

        return results
