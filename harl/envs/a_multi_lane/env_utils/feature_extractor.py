#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取模块

功能:
1. 将环境状态转换为意图预测所需的特征（以被预测车辆为ego）
2. 将环境状态转换为占用预测所需的特征（以被预测车辆为ego）
3. 将环境状态转换为决策模型所需的特征（以本车为ego）

Author: 交通流研究团队
Date: 2025-01
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class FeatureExtractor:
    """特征提取器 - 为预测和决策模块准备输入特征"""

    def __init__(self,
                 v_max: float = 33.33,  # 最大速度 (m/s) ~120 km/h
                 kappa_max: float = 0.2,  # 最大密度 (veh/m)
                 d_max: float = 200.0,  # 最大距离 (m)
                 T_max: float = 10.0,  # 最大时间间隙 (s)
                 a_max: float = 4.0):  # 最大加速度 (m/s^2)
        """
        初始化特征提取器

        Args:
            v_max: 速度归一化最大值
            kappa_max: 密度归一化最大值
            d_max: 距离归一化最大值
            T_max: 时间间隙归一化最大值
            a_max: 加速度归一化最大值
        """
        self.v_max = v_max
        self.kappa_max = kappa_max
        self.d_max = d_max
        self.T_max = T_max
        self.a_max = a_max

    def _normalize_value(self, value: float, max_value: float, min_value: float = 0.0) -> float:
        """归一化单个值到 [0, 1] 或 [-1, 1]"""
        if min_value < 0:
            # 双向归一化 [-max, max] -> [-1, 1]
            return np.clip(value / max_value, -1.0, 1.0)
        else:
            # 单向归一化 [0, max] -> [0, 1]
            return np.clip((value - min_value) / (max_value - min_value), 0.0, 1.0)

    def extract_intention_features(self,
                                   target_vehicle_id: str,
                                   state: Dict,
                                   lane_statistics: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        提取意图预测特征（以目标车辆为ego）

        Args:
            target_vehicle_id: 被预测车辆ID
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            env_features: 环境特征 [4,] - [v_avg, kappa_avg, delta_v, delta_kappa]
            ind_features: 个体特征 [10,] - [v_ego, v_rel, d_headway, T_headway, ...]
            vehicle_type: 车辆类型 (0=HDV, 1=CAV)
        """
        if target_vehicle_id not in state:
            raise ValueError(f"Vehicle {target_vehicle_id} not found in state")

        ego_info = state[target_vehicle_id]
        ego_lane = ego_info['lane_index']
        ego_v = ego_info['speed']
        ego_a = ego_info.get('acceleration', 0.0)

        # =============== 环境特征 (4维) ===============
        # 当前车道和目标车道（假设目标是相邻车道）
        current_lane_id = ego_lane
        target_lane_id = ego_lane + 1 if ego_lane < 2 else ego_lane - 1  # 简化：选择相邻车道

        # 车道平均速度和密度
        v_current = lane_statistics.get(current_lane_id, {}).get('mean_speed', ego_v)
        kappa_current = lane_statistics.get(current_lane_id, {}).get('density', 0.05)
        v_target = lane_statistics.get(target_lane_id, {}).get('mean_speed', ego_v)
        kappa_target = lane_statistics.get(target_lane_id, {}).get('density', 0.05)

        # 归一化
        v_avg_norm = self._normalize_value((v_current + v_target) / 2, self.v_max)
        kappa_avg_norm = self._normalize_value((kappa_current + kappa_target) / 2, self.kappa_max)
        delta_v_norm = self._normalize_value(v_target - v_current, self.v_max, -self.v_max)
        delta_kappa_norm = self._normalize_value(kappa_target - kappa_current, self.kappa_max, -self.kappa_max)

        env_features = np.array([v_avg_norm, kappa_avg_norm, delta_v_norm, delta_kappa_norm], dtype=np.float32)

        # =============== 个体特征 (10维) ===============
        # 提取周围车辆信息
        surround_info = ego_info.get('surrounding_vehicles', {})

        # 当前车道前车
        front_1 = surround_info.get('front_1', None)
        if front_1 and front_1 != 0:
            v_rel = self._normalize_value(front_1.get('long_rel_v', 0.0), self.v_max, -self.v_max)
            d_headway = self._normalize_value(front_1.get('long_dist', 100.0), self.d_max)
            T_headway = self._normalize_value(front_1.get('long_dist', 100.0) / max(ego_v, 0.1), self.T_max)
        else:
            v_rel, d_headway, T_headway = 0.0, 1.0, 1.0

        # 目标车道前车
        adj_front = surround_info.get('adj_front', None)
        if adj_front and adj_front != 0:
            delta_v_adj_front = self._normalize_value(adj_front.get('long_rel_v', 0.0), self.v_max, -self.v_max)
            d_adj_front = self._normalize_value(adj_front.get('long_dist', 100.0), self.d_max)
            T_adj_front = self._normalize_value(adj_front.get('long_dist', 100.0) / max(ego_v, 0.1), self.T_max)
        else:
            delta_v_adj_front, d_adj_front, T_adj_front = 0.0, 1.0, 1.0

        # 目标车道后车
        adj_rear = surround_info.get('adj_rear', None)
        if adj_rear and adj_rear != 0:
            delta_v_adj_rear = self._normalize_value(adj_rear.get('long_rel_v', 0.0), self.v_max, -self.v_max)
            d_adj_rear = self._normalize_value(abs(adj_rear.get('long_dist', 100.0)), self.d_max)
            T_adj_rear = self._normalize_value(abs(adj_rear.get('long_dist', 100.0)) / max(ego_v, 0.1), self.T_max)
        else:
            delta_v_adj_rear, d_adj_rear, T_adj_rear = 0.0, 1.0, 1.0

        v_ego_norm = self._normalize_value(ego_v, self.v_max)

        ind_features = np.array([
            v_ego_norm, v_rel, d_headway, T_headway,
            delta_v_adj_front, d_adj_front, T_adj_front,
            delta_v_adj_rear, d_adj_rear, T_adj_rear
        ], dtype=np.float32)

        # =============== 车辆类型 ===============
        vehicle_type = 1 if 'CAV' in target_vehicle_id else 0

        return env_features, ind_features, vehicle_type

    def extract_occupancy_features(self,
                                   target_vehicle_id: str,
                                   state: Dict,
                                   lane_statistics: Dict,
                                   intention: float = 0.0) -> Dict[str, np.ndarray]:
        """
        提取占用预测特征（以目标车辆为ego）

        Args:
            target_vehicle_id: 被预测车辆ID
            state: 环境状态字典
            lane_statistics: 车道统计信息
            intention: 意图预测结果 (0-1之间，换道概率)

        Returns:
            features: 特征字典
                - traffic_state: [6,] 交通状态 [EL_v, EL_k, LL_v, LL_k, RL_v, RL_k]
                - ego_current: [4,] 自车当前状态 [v, a, lane_id, eta]
                - ego_history: [20, 5] 自车历史轨迹 [Δx, Δy, Δv, Δa, lane_id]
                - sur_current: [6, 8] 周边车辆当前状态
                - sur_history: [6, 20, 5] 周边车辆历史轨迹
                - intention: [1,] 横向意图
        """
        if target_vehicle_id not in state:
            raise ValueError(f"Vehicle {target_vehicle_id} not found in state")

        ego_info = state[target_vehicle_id]
        ego_lane = ego_info['lane_index']
        ego_v = ego_info['speed']
        ego_a = ego_info.get('acceleration', 0.0)
        ego_x = ego_info['position_x']
        ego_y = ego_info['position_y']

        # =============== 交通状态 (6维) ===============
        # 提取三条车道的平均速度和密度
        lane_ids = [0, 1, 2]  # EL(应急车道), LL(左车道), RL(右车道)
        traffic_state = []
        for lane_id in lane_ids:
            lane_info = lane_statistics.get(lane_id, {})
            v_lane = lane_info.get('mean_speed', 25.0) / self.v_max  # 归一化
            kappa_lane = lane_info.get('density', 0.05) / self.kappa_max  # 归一化
            traffic_state.extend([v_lane, kappa_lane])
        traffic_state = np.array(traffic_state, dtype=np.float32)

        # =============== 自车当前状态 (4维) ===============
        # eta: 预计到达时间 (简化为基于速度的估计)
        eta = 10.0 if ego_v > 1.0 else 20.0
        ego_current = np.array([
            ego_v / self.v_max,
            ego_a / self.a_max,
            ego_lane / 2.0,  # 归一化到 [0, 1]
            eta / 30.0  # 归一化
        ], dtype=np.float32)

        # =============== 自车历史轨迹 (20, 5) ===============
        # 简化：使用历史状态（如果有）或填充零
        ego_history = ego_info.get('history_trajectory', np.zeros((20, 5), dtype=np.float32))
        if ego_history.shape[0] < 20:
            # 填充不足的历史
            pad_length = 20 - ego_history.shape[0]
            ego_history = np.vstack([np.zeros((pad_length, 5)), ego_history])
        elif ego_history.shape[0] > 20:
            # 截取最近20步
            ego_history = ego_history[-20:]

        # =============== 周边车辆当前状态 (6, 8) ===============
        surround_keys = ['front_1', 'front_2', 'rear_1', 'rear_2', 'adj_front', 'adj_rear']
        sur_current = np.zeros((6, 8), dtype=np.float32)

        surround_info = ego_info.get('surrounding_vehicles', {})
        for i, key in enumerate(surround_keys):
            sur_veh = surround_info.get(key, None)
            if sur_veh and sur_veh != 0:
                # [Δx, Δy, v, a, lane_id, length, width, type]
                sur_current[i] = [
                    sur_veh.get('long_dist', 0.0) / self.d_max,
                    sur_veh.get('lat_dist', 0.0) / 10.0,  # 横向距离归一化
                    sur_veh.get('speed', ego_v) / self.v_max,
                    sur_veh.get('acceleration', 0.0) / self.a_max,
                    sur_veh.get('lane_index', ego_lane) / 2.0,
                    sur_veh.get('length', 5.0) / 10.0,
                    sur_veh.get('width', 2.0) / 5.0,
                    1.0 if 'CAV' in sur_veh.get('id', '') else 0.0
                ]

        # =============== 周边车辆历史轨迹 (6, 20, 5) ===============
        sur_history = np.zeros((6, 20, 5), dtype=np.float32)
        for i, key in enumerate(surround_keys):
            sur_veh = surround_info.get(key, None)
            if sur_veh and sur_veh != 0:
                veh_history = sur_veh.get('history_trajectory', np.zeros((20, 5)))
                if veh_history.shape[0] < 20:
                    pad_length = 20 - veh_history.shape[0]
                    veh_history = np.vstack([np.zeros((pad_length, 5)), veh_history])
                elif veh_history.shape[0] > 20:
                    veh_history = veh_history[-20:]
                sur_history[i] = veh_history

        # =============== 横向意图 (1维) ===============
        intention_feature = np.array([intention], dtype=np.float32)

        return {
            'traffic_state': traffic_state,
            'ego_current': ego_current,
            'ego_history': ego_history,
            'sur_current': sur_current,
            'sur_history': sur_history,
            'intention': intention_feature
        }

    def extract_decision_features(self,
                                  ego_vehicle_id: str,
                                  state: Dict,
                                  lane_statistics: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        提取决策特征（以本车CAV为ego）

        Args:
            ego_vehicle_id: 本车CAV的ID
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            env_features: 环境特征 [4,]
            ind_features: 个体特征 [10,]
            vehicle_type: 车辆类型 (固定为1=CAV)
        """
        # 决策特征提取与意图预测相同，但确保是CAV
        env_features, ind_features, _ = self.extract_intention_features(
            ego_vehicle_id, state, lane_statistics
        )

        # 决策模块的ego必须是CAV
        vehicle_type = 1

        return env_features, ind_features, vehicle_type

    def batch_extract_intention_features(self,
                                         vehicle_ids: List[str],
                                         state: Dict,
                                         lane_statistics: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量提取意图预测特征

        Args:
            vehicle_ids: 车辆ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息

        Returns:
            env_features_batch: [N, 4]
            ind_features_batch: [N, 10]
            vehicle_types_batch: [N, 1]
        """
        env_features_list = []
        ind_features_list = []
        vehicle_types_list = []

        for veh_id in vehicle_ids:
            try:
                env_feat, ind_feat, veh_type = self.extract_intention_features(
                    veh_id, state, lane_statistics
                )
                env_features_list.append(env_feat)
                ind_features_list.append(ind_feat)
                vehicle_types_list.append([veh_type])
            except (ValueError, KeyError):
                # 车辆不在状态中，跳过
                continue

        if len(env_features_list) == 0:
            # 返回空数组
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 10), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.float32))

        return (np.array(env_features_list, dtype=np.float32),
                np.array(ind_features_list, dtype=np.float32),
                np.array(vehicle_types_list, dtype=np.float32))


if __name__ == "__main__":
    # 测试特征提取器
    print("=" * 80)
    print("测试特征提取器")
    print("=" * 80)

    # 创建模拟状态
    state = {
        'CAV_0': {
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
                'adj_rear': {
                    'id': 'HDV_3',
                    'long_dist': -30.0,
                    'long_rel_v': -3.0,
                    'speed': 22.0,
                    'lane_index': 2,
                }
            }
        }
    }

    lane_statistics = {
        0: {'mean_speed': 20.0, 'density': 0.08},
        1: {'mean_speed': 25.0, 'density': 0.05},
        2: {'mean_speed': 28.0, 'density': 0.04}
    }

    # 创建特征提取器
    extractor = FeatureExtractor()

    # 测试意图预测特征
    print("\n1. 意图预测特征:")
    env_feat, ind_feat, veh_type = extractor.extract_intention_features(
        'CAV_0', state, lane_statistics
    )
    print(f"  环境特征 shape: {env_feat.shape}, 值: {env_feat}")
    print(f"  个体特征 shape: {ind_feat.shape}, 值: {ind_feat}")
    print(f"  车辆类型: {veh_type} (1=CAV)")

    # 测试占用预测特征
    print("\n2. 占用预测特征:")
    occ_feat = extractor.extract_occupancy_features(
        'CAV_0', state, lane_statistics, intention=0.8
    )
    for key, value in occ_feat.items():
        print(f"  {key:15s}: shape={value.shape}")

    # 测试决策特征
    print("\n3. 决策特征:")
    env_feat_dec, ind_feat_dec, veh_type_dec = extractor.extract_decision_features(
        'CAV_0', state, lane_statistics
    )
    print(f"  环境特征 shape: {env_feat_dec.shape}")
    print(f"  个体特征 shape: {ind_feat_dec.shape}")
    print(f"  车辆类型: {veh_type_dec} (1=CAV)")

    print("\n✓ 特征提取器测试完成!")
