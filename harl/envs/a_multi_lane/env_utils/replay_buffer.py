#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经验回放缓冲区

用于存储和采样CAV和HDV的历史轨迹数据，支持：
1. CAV轨迹存储（用于决策模型微调）
2. HDV轨迹存储（用于CQR校准）
3. 分类场景存储（换道/保持）
4. 批量采样
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import pickle


class TrajectorySegment:
    """
    轨迹片段

    存储单个车辆的一段轨迹及相关信息
    """

    def __init__(self,
                 vehicle_id: str,
                 vehicle_type: str,
                 trajectory: np.ndarray,
                 features: Optional[Dict] = None,
                 label: Optional[int] = None,
                 reward: Optional[float] = None,
                 timestamp: Optional[float] = None):
        """
        初始化轨迹片段

        Args:
            vehicle_id: 车辆ID
            vehicle_type: 车辆类型 ('CAV' 或 'HDV')
            trajectory: 轨迹数据 [T, D] (时间步 × 维度)
            features: 特征字典（环境特征、个体特征等）
            label: 标签（换道决策等）
            reward: 奖励值
            timestamp: 时间戳
        """
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.trajectory = trajectory
        self.features = features or {}
        self.label = label
        self.reward = reward
        self.timestamp = timestamp


class ReplayBuffer:
    """
    经验回放缓冲区

    存储轨迹片段，支持分类存储和采样
    """

    def __init__(self, capacity: int = 10000):
        """
        初始化回放缓冲区

        Args:
            capacity: 缓冲区容量（每类）
        """
        self.capacity = capacity

        # 分类存储
        self.cav_buffer = deque(maxlen=capacity)  # CAV轨迹
        self.hdv_buffer = deque(maxlen=capacity)  # HDV轨迹

        # 分场景存储（用于平衡采样）
        self.cav_lane_change = deque(maxlen=capacity // 2)  # CAV换道
        self.cav_keep_lane = deque(maxlen=capacity // 2)    # CAV保持

        self.hdv_lane_change = deque(maxlen=capacity // 2)  # HDV换道
        self.hdv_keep_lane = deque(maxlen=capacity // 2)    # HDV保持

        # 统计信息
        self.stats = {
            'total_cav': 0,
            'total_hdv': 0,
            'cav_lane_change_count': 0,
            'cav_keep_lane_count': 0,
            'hdv_lane_change_count': 0,
            'hdv_keep_lane_count': 0
        }

    def add_cav_trajectory(self,
                          vehicle_id: str,
                          trajectory: np.ndarray,
                          features: Dict,
                          decision: int,
                          reward: float,
                          timestamp: float):
        """
        添加CAV轨迹

        Args:
            vehicle_id: CAV ID
            trajectory: 轨迹 [T, D]
            features: 特征字典
            decision: 横向决策 (0=保持, 1/2=换道)
            reward: 奖励
            timestamp: 时间戳
        """
        segment = TrajectorySegment(
            vehicle_id=vehicle_id,
            vehicle_type='CAV',
            trajectory=trajectory,
            features=features,
            label=decision,
            reward=reward,
            timestamp=timestamp
        )

        # 添加到总缓冲区
        self.cav_buffer.append(segment)
        self.stats['total_cav'] += 1

        # 分类存储
        if decision == 0:  # 保持
            self.cav_keep_lane.append(segment)
            self.stats['cav_keep_lane_count'] += 1
        else:  # 换道
            self.cav_lane_change.append(segment)
            self.stats['cav_lane_change_count'] += 1

    def add_hdv_trajectory(self,
                          vehicle_id: str,
                          trajectory: np.ndarray,
                          features: Dict,
                          intention: int,
                          timestamp: float):
        """
        添加HDV轨迹

        Args:
            vehicle_id: HDV ID
            trajectory: 轨迹 [T, D]
            features: 特征字典
            intention: 换道意图 (0=保持, 1=换道)
            timestamp: 时间戳
        """
        segment = TrajectorySegment(
            vehicle_id=vehicle_id,
            vehicle_type='HDV',
            trajectory=trajectory,
            features=features,
            label=intention,
            reward=None,
            timestamp=timestamp
        )

        # 添加到总缓冲区
        self.hdv_buffer.append(segment)
        self.stats['total_hdv'] += 1

        # 分类存储
        if intention == 0:  # 保持
            self.hdv_keep_lane.append(segment)
            self.stats['hdv_keep_lane_count'] += 1
        else:  # 换道
            self.hdv_lane_change.append(segment)
            self.stats['hdv_lane_change_count'] += 1

    def sample_cav_batch(self,
                        batch_size: int,
                        balanced: bool = True) -> List[TrajectorySegment]:
        """
        采样CAV批次

        Args:
            batch_size: 批次大小
            balanced: 是否平衡采样（换道/保持各半）

        Returns:
            batch: TrajectorySegment列表
        """
        if len(self.cav_buffer) == 0:
            return []

        if balanced and len(self.cav_lane_change) > 0 and len(self.cav_keep_lane) > 0:
            # 平衡采样
            half = batch_size // 2
            indices_lc = np.random.choice(len(self.cav_lane_change), size=min(half, len(self.cav_lane_change)), replace=False)
            indices_kl = np.random.choice(len(self.cav_keep_lane), size=min(batch_size - half, len(self.cav_keep_lane)), replace=False)

            batch = [self.cav_lane_change[i] for i in indices_lc] + [self.cav_keep_lane[i] for i in indices_kl]
        else:
            # 随机采样
            indices = np.random.choice(len(self.cav_buffer), size=min(batch_size, len(self.cav_buffer)), replace=False)
            batch = [self.cav_buffer[i] for i in indices]

        return batch

    def sample_hdv_batch(self,
                        batch_size: int,
                        balanced: bool = False) -> List[TrajectorySegment]:
        """
        采样HDV批次

        Args:
            batch_size: 批次大小
            balanced: 是否平衡采样

        Returns:
            batch: TrajectorySegment列表
        """
        if len(self.hdv_buffer) == 0:
            return []

        if balanced and len(self.hdv_lane_change) > 0 and len(self.hdv_keep_lane) > 0:
            half = batch_size // 2
            indices_lc = np.random.choice(len(self.hdv_lane_change), size=min(half, len(self.hdv_lane_change)), replace=False)
            indices_kl = np.random.choice(len(self.hdv_keep_lane), size=min(batch_size - half, len(self.hdv_keep_lane)), replace=False)

            batch = [self.hdv_lane_change[i] for i in indices_lc] + [self.hdv_keep_lane[i] for i in indices_kl]
        else:
            indices = np.random.choice(len(self.hdv_buffer), size=min(batch_size, len(self.hdv_buffer)), replace=False)
            batch = [self.hdv_buffer[i] for i in indices]

        return batch

    def get_recent_cav_trajectories(self, n: int = 100) -> List[TrajectorySegment]:
        """
        获取最近的N条CAV轨迹

        Args:
            n: 数量

        Returns:
            轨迹列表
        """
        return list(self.cav_buffer)[-n:]

    def get_recent_hdv_trajectories(self, n: int = 100) -> List[TrajectorySegment]:
        """
        获取最近的N条HDV轨迹

        Args:
            n: 数量

        Returns:
            轨迹列表
        """
        return list(self.hdv_buffer)[-n:]

    def get_statistics(self) -> Dict:
        """
        获取缓冲区统计信息

        Returns:
            stats: 统计字典
        """
        stats = self.stats.copy()
        stats['current_cav_buffer_size'] = len(self.cav_buffer)
        stats['current_hdv_buffer_size'] = len(self.hdv_buffer)
        stats['cav_lane_change_buffer'] = len(self.cav_lane_change)
        stats['cav_keep_lane_buffer'] = len(self.cav_keep_lane)
        stats['hdv_lane_change_buffer'] = len(self.hdv_lane_change)
        stats['hdv_keep_lane_buffer'] = len(self.hdv_keep_lane)

        if stats['current_cav_buffer_size'] > 0:
            stats['cav_lane_change_ratio'] = len(self.cav_lane_change) / stats['current_cav_buffer_size']
        else:
            stats['cav_lane_change_ratio'] = 0.0

        if stats['current_hdv_buffer_size'] > 0:
            stats['hdv_lane_change_ratio'] = len(self.hdv_lane_change) / stats['current_hdv_buffer_size']
        else:
            stats['hdv_lane_change_ratio'] = 0.0

        return stats

    def clear_cav(self):
        """清空CAV缓冲区"""
        self.cav_buffer.clear()
        self.cav_lane_change.clear()
        self.cav_keep_lane.clear()

    def clear_hdv(self):
        """清空HDV缓冲区"""
        self.hdv_buffer.clear()
        self.hdv_lane_change.clear()
        self.hdv_keep_lane.clear()

    def clear_all(self):
        """清空所有缓冲区"""
        self.clear_cav()
        self.clear_hdv()
        self.stats = {
            'total_cav': 0,
            'total_hdv': 0,
            'cav_lane_change_count': 0,
            'cav_keep_lane_count': 0,
            'hdv_lane_change_count': 0,
            'hdv_keep_lane_count': 0
        }

    def save(self, filepath: str):
        """
        保存缓冲区到文件

        Args:
            filepath: 文件路径
        """
        data = {
            'cav_buffer': list(self.cav_buffer),
            'hdv_buffer': list(self.hdv_buffer),
            'cav_lane_change': list(self.cav_lane_change),
            'cav_keep_lane': list(self.cav_keep_lane),
            'hdv_lane_change': list(self.hdv_lane_change),
            'hdv_keep_lane': list(self.hdv_keep_lane),
            'stats': self.stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """
        从文件加载缓冲区

        Args:
            filepath: 文件路径
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.cav_buffer = deque(data['cav_buffer'], maxlen=self.capacity)
        self.hdv_buffer = deque(data['hdv_buffer'], maxlen=self.capacity)
        self.cav_lane_change = deque(data['cav_lane_change'], maxlen=self.capacity // 2)
        self.cav_keep_lane = deque(data['cav_keep_lane'], maxlen=self.capacity // 2)
        self.hdv_lane_change = deque(data['hdv_lane_change'], maxlen=self.capacity // 2)
        self.hdv_keep_lane = deque(data['hdv_keep_lane'], maxlen=self.capacity // 2)
        self.stats = data['stats']


class CQRCalibrationBuffer:
    """
    CQR校准缓冲区

    专门用于存储HDV轨迹以进行CQR模型的校准
    """

    def __init__(self, target_size: int = 500):
        """
        初始化校准缓冲区

        Args:
            target_size: 目标样本数（根据00_structure.tex，校准集大小为500）
        """
        self.target_size = target_size
        self.buffer = deque(maxlen=target_size)

    def add_sample(self,
                   features: Dict,
                   ground_truth_trajectory: np.ndarray):
        """
        添加校准样本

        Args:
            features: 输入特征（交通状态、车辆状态等）
            ground_truth_trajectory: 真实轨迹 [T, 2] (x, y)
        """
        sample = {
            'features': features,
            'ground_truth': ground_truth_trajectory
        }
        self.buffer.append(sample)

    def is_ready(self) -> bool:
        """
        检查是否达到目标样本数

        Returns:
            True if ready for calibration
        """
        return len(self.buffer) >= self.target_size

    def get_calibration_set(self) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        获取校准数据集

        Returns:
            features_list: 特征列表
            ground_truth_list: 真实轨迹列表
        """
        features_list = [sample['features'] for sample in self.buffer]
        ground_truth_list = [sample['ground_truth'] for sample in self.buffer]

        return features_list, ground_truth_list

    def clear(self):
        """清空校准缓冲区"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # 测试经验回放缓冲区
    print("=" * 80)
    print("测试经验回放缓冲区")
    print("=" * 80)

    # 创建缓冲区
    replay_buffer = ReplayBuffer(capacity=1000)

    # 测试1: 添加CAV轨迹
    print("\n测试1: 添加CAV轨迹")
    for i in range(50):
        trajectory = np.random.randn(30, 2)  # 30步，2维(x, y)
        features = {
            'env_features': np.random.randn(4),
            'ind_features': np.random.randn(10)
        }
        decision = np.random.choice([0, 1, 2])  # 0=保持, 1/2=换道
        reward = np.random.randn()

        replay_buffer.add_cav_trajectory(
            vehicle_id=f'CAV_{i}',
            trajectory=trajectory,
            features=features,
            decision=decision,
            reward=reward,
            timestamp=i * 0.1
        )

    stats = replay_buffer.get_statistics()
    print(f"  总CAV轨迹数: {stats['total_cav']}")
    print(f"  CAV缓冲区大小: {stats['current_cav_buffer_size']}")
    print(f"  CAV换道轨迹数: {stats['cav_lane_change_buffer']}")
    print(f"  CAV保持轨迹数: {stats['cav_keep_lane_buffer']}")
    print(f"  CAV换道比例: {stats['cav_lane_change_ratio']:.2%}")

    # 测试2: 添加HDV轨迹
    print("\n测试2: 添加HDV轨迹")
    for i in range(100):
        trajectory = np.random.randn(30, 2)
        features = {
            'env_features': np.random.randn(4),
            'ind_features': np.random.randn(10)
        }
        intention = np.random.choice([0, 1])  # 0=保持, 1=换道

        replay_buffer.add_hdv_trajectory(
            vehicle_id=f'HDV_{i}',
            trajectory=trajectory,
            features=features,
            intention=intention,
            timestamp=i * 0.1
        )

    stats = replay_buffer.get_statistics()
    print(f"  总HDV轨迹数: {stats['total_hdv']}")
    print(f"  HDV缓冲区大小: {stats['current_hdv_buffer_size']}")
    print(f"  HDV换道轨迹数: {stats['hdv_lane_change_buffer']}")
    print(f"  HDV保持轨迹数: {stats['hdv_keep_lane_buffer']}")
    print(f"  HDV换道比例: {stats['hdv_lane_change_ratio']:.2%}")

    # 测试3: 批量采样
    print("\n测试3: 批量采样")
    cav_batch = replay_buffer.sample_cav_batch(batch_size=16, balanced=True)
    print(f"  CAV批次大小: {len(cav_batch)}")
    print(f"  CAV批次决策分布:")
    decisions = [seg.label for seg in cav_batch]
    for decision in [0, 1, 2]:
        count = decisions.count(decision)
        print(f"    决策{decision}: {count} ({count/len(cav_batch)*100:.1f}%)")

    hdv_batch = replay_buffer.sample_hdv_batch(batch_size=32, balanced=False)
    print(f"\n  HDV批次大小: {len(hdv_batch)}")
    print(f"  HDV批次意图分布:")
    intentions = [seg.label for seg in hdv_batch]
    for intention in [0, 1]:
        count = intentions.count(intention)
        print(f"    意图{intention}: {count} ({count/len(hdv_batch)*100:.1f}%)")

    # 测试4: 获取最近轨迹
    print("\n测试4: 获取最近轨迹")
    recent_cav = replay_buffer.get_recent_cav_trajectories(n=10)
    print(f"  最近10条CAV轨迹数: {len(recent_cav)}")

    recent_hdv = replay_buffer.get_recent_hdv_trajectories(n=20)
    print(f"  最近20条HDV轨迹数: {len(recent_hdv)}")

    # 测试5: CQR校准缓冲区
    print("\n测试5: CQR校准缓冲区")
    calib_buffer = CQRCalibrationBuffer(target_size=500)

    for i in range(600):
        features = {
            'traffic_state': np.random.randn(6),
            'ego_current': np.random.randn(4)
        }
        gt_trajectory = np.random.randn(30, 2)

        calib_buffer.add_sample(features, gt_trajectory)

    print(f"  校准缓冲区大小: {len(calib_buffer)}")
    print(f"  是否准备就绪: {calib_buffer.is_ready()}")

    if calib_buffer.is_ready():
        feat_list, gt_list = calib_buffer.get_calibration_set()
        print(f"  校准集特征数: {len(feat_list)}")
        print(f"  校准集轨迹数: {len(gt_list)}")

    # 测试6: 保存和加载
    print("\n测试6: 保存和加载")
    import tempfile
    import os

    temp_file = tempfile.mktemp(suffix='.pkl')
    replay_buffer.save(temp_file)
    print(f"  缓冲区已保存到: {temp_file}")

    # 创建新缓冲区并加载
    new_buffer = ReplayBuffer(capacity=1000)
    new_buffer.load(temp_file)

    new_stats = new_buffer.get_statistics()
    print(f"  加载后CAV轨迹数: {new_stats['current_cav_buffer_size']}")
    print(f"  加载后HDV轨迹数: {new_stats['current_hdv_buffer_size']}")

    # 清理
    os.remove(temp_file)

    print("\n✓ 经验回放缓冲区测试完成!")
