#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策模块集成

功能:
1. 调用SCM决策模型，为CAV生成横向换道决策
2. 支持MARL训练过程中的在线微调
3. 渐进式微调策略：冻结→解冻个体层→全局微调

Author: 交通流研究团队
Date: 2025-01
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 添加决策模块路径
harl_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(harl_root))

try:
    from harl.decisions.lateral_decisions.SCM_decisions import SCMDecisionMakerFactory
    from harl.envs.a_multi_lane.env_utils.feature_extractor import FeatureExtractor
except ImportError:
    # 当直接运行时的导入
    import os
    os.chdir(str(harl_root))
    sys.path.insert(0, str(harl_root))
    from harl.decisions.lateral_decisions.SCM_decisions import SCMDecisionMakerFactory
    from feature_extractor import FeatureExtractor


class DecisionModule:
    """决策模块 - 集成SCM横向决策并支持MARL微调"""

    def __init__(self,
                 model_type: str = "shallow_hierarchical",
                 freeze_base_model: bool = True,
                 enable_training: bool = True,
                 device: str = "cpu",
                 use_cache: bool = True):
        """
        初始化决策模块

        Args:
            model_type: 决策模型类型 ("shallow_hierarchical" 或 "medium_hierarchical")
            freeze_base_model: 是否冻结基础SCM模型（初期建议True）
            enable_training: 是否启用训练模式（MARL训练时设为True）
            device: 计算设备
            use_cache: 是否使用单例缓存
        """
        print(f"[DecisionModule] 初始化决策模块...")

        # 创建特征提取器
        self.feature_extractor = FeatureExtractor()

        # 创建决策生成器
        print(f"  - 加载决策模型: {model_type}")
        print(f"  - 冻结基础模型: {freeze_base_model}")
        print(f"  - 训练模式: {enable_training}")

        self.decision_maker = SCMDecisionMakerFactory.create_decision_maker(
            model_type=model_type,
            freeze_base_model=freeze_base_model,
            enable_training=enable_training,
            device=device,
            use_cache=use_cache
        )

        self.device = device
        self.enable_training = enable_training

        # 获取决策模型（用于微调）
        self.decision_model = self.decision_maker.get_model()

        # 微调相关
        self.optimizer = None
        self.fine_tune_stage = 0  # 0=冻结, 1=解冻个体层, 2=全局微调
        self.total_episodes = 0
        self.fine_tune_losses = []

        print(f"[DecisionModule] ✓ 决策模块初始化完成")

    def make_decision(self,
                     ego_vehicle_id: str,
                     state: Dict,
                     lane_statistics: Dict,
                     return_prob: bool = False) -> Tuple[int, Optional[float]]:
        """
        为单个CAV生成横向决策

        Args:
            ego_vehicle_id: CAV的ID
            state: 环境状态字典
            lane_statistics: 车道统计信息
            return_prob: 是否返回决策概率

        Returns:
            decision: 0=保持车道, 1=换道
            probability: 换道概率（如果return_prob=True）
        """
        # 提取决策特征
        env_features, ind_features, vehicle_type = self.feature_extractor.extract_decision_features(
            ego_vehicle_id, state, lane_statistics
        )

        # 添加batch维度
        env_features = env_features[np.newaxis, :]  # [1, 4]
        ind_features = ind_features[np.newaxis, :]  # [1, 10]
        vehicle_type = np.array([[vehicle_type]], dtype=np.float32)  # [1, 1]

        # 调用决策模型
        if return_prob:
            decisions, probs = self.decision_maker.decide(
                env_features, ind_features, vehicle_type, return_prob=True
            )
            return int(decisions[0, 0]), float(probs[0, 0])
        else:
            decisions = self.decision_maker.decide(
                env_features, ind_features, vehicle_type, return_prob=False
            )
            return int(decisions[0, 0]), None

    def make_batch_decisions(self,
                            ego_vehicle_ids: List[str],
                            state: Dict,
                            lane_statistics: Dict,
                            return_prob: bool = False) -> Dict[str, Tuple[int, Optional[float]]]:
        """
        批量为多个CAV生成横向决策

        Args:
            ego_vehicle_ids: CAV的ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息
            return_prob: 是否返回决策概率

        Returns:
            decisions: {vehicle_id: (decision, probability)}
        """
        decisions = {}

        for veh_id in ego_vehicle_ids:
            try:
                decision, prob = self.make_decision(
                    veh_id, state, lane_statistics, return_prob
                )
                decisions[veh_id] = (decision, prob)
            except (ValueError, KeyError):
                # 车辆状态不完整，默认保持车道
                decisions[veh_id] = (0, None)

        return decisions

    # ==================== MARL微调相关 ====================

    def setup_fine_tuning(self,
                         learning_rate: float = 1e-4,
                         stage_thresholds: Tuple[int, int] = (1000, 2000)):
        """
        配置微调策略

        Args:
            learning_rate: 学习率
            stage_thresholds: 阶段切换阈值 (episode数)
                - stage_thresholds[0]: 解冻个体层的episode数
                - stage_thresholds[1]: 全局微调的episode数
        """
        if not self.enable_training:
            print("[DecisionModule] ⚠️ 训练模式未启用，无法配置微调")
            return

        print(f"[DecisionModule] 配置渐进式微调策略:")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - 阶段1 (冻结): 0 - {stage_thresholds[0]} episodes")
        print(f"  - 阶段2 (解冻个体层): {stage_thresholds[0]} - {stage_thresholds[1]} episodes")
        print(f"  - 阶段3 (全局微调): {stage_thresholds[1]}+ episodes")

        self.stage_thresholds = stage_thresholds

        # 创建优化器（只优化可训练参数）
        trainable_params = self.decision_model.get_trainable_parameters()
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        # 打印参数状态
        self.decision_model.print_parameter_status()

    def update_fine_tune_stage(self):
        """根据episode数更新微调阶段"""
        if not self.enable_training or self.optimizer is None:
            return

        old_stage = self.fine_tune_stage

        if self.total_episodes >= self.stage_thresholds[1]:
            # 阶段3: 全局微调
            self.fine_tune_stage = 2
            if old_stage != 2:
                print(f"\n[DecisionModule] 切换到阶段3: 全局微调 (episode {self.total_episodes})")
                # 解冻环境层
                self.decision_model.unfreeze_layer(self.decision_model.scm_model.env_layer)
                # 重新创建优化器
                trainable_params = self.decision_model.get_trainable_parameters()
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
                self.decision_model.print_parameter_status()

        elif self.total_episodes >= self.stage_thresholds[0]:
            # 阶段2: 解冻个体层
            self.fine_tune_stage = 1
            if old_stage != 1:
                print(f"\n[DecisionModule] 切换到阶段2: 解冻个体层 (episode {self.total_episodes})")
                # 解冻个体层
                self.decision_model.unfreeze_layer(self.decision_model.scm_model.ind_layer)
                # 重新创建优化器
                trainable_params = self.decision_model.get_trainable_parameters()
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
                self.decision_model.print_parameter_status()

    def compute_decision_loss(self,
                             ego_vehicle_ids: List[str],
                             state: Dict,
                             lane_statistics: Dict,
                             rewards: Dict[str, float]) -> torch.Tensor:
        """
        计算决策损失（用于策略梯度）

        Args:
            ego_vehicle_ids: CAV的ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息
            rewards: {vehicle_id: reward} 奖励字典

        Returns:
            loss: 决策损失
        """
        if not self.enable_training:
            raise RuntimeError("训练模式未启用")

        total_loss = 0.0
        valid_count = 0

        for veh_id in ego_vehicle_ids:
            if veh_id not in rewards:
                continue

            try:
                # 提取特征
                env_features, ind_features, vehicle_type = self.feature_extractor.extract_decision_features(
                    veh_id, state, lane_statistics
                )

                # 转换为tensor
                env_feat_tensor = torch.from_numpy(env_features[np.newaxis, :]).to(self.device)
                ind_feat_tensor = torch.from_numpy(ind_features[np.newaxis, :]).to(self.device)
                veh_type_tensor = torch.from_numpy(np.array([[vehicle_type]], dtype=np.float32)).to(self.device)

                # 前向传播（保持梯度）
                P_final, _ = self.decision_model(env_feat_tensor, ind_feat_tensor, veh_type_tensor)

                # 策略梯度损失: -reward * log(P)
                reward_tensor = torch.tensor(rewards[veh_id], dtype=torch.float32, device=self.device)

                # 对于二值决策，使用交叉熵形式
                # decision = 1 时，损失 = -reward * log(P_final)
                # decision = 0 时，损失 = -reward * log(1 - P_final)
                # 这里我们假设reward越大越好，P_final是换道概率
                loss = -reward_tensor * torch.log(P_final + 1e-8)

                total_loss = total_loss + loss
                valid_count += 1

            except (ValueError, KeyError):
                continue

        if valid_count == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / valid_count

    def update_decision_model(self,
                             ego_vehicle_ids: List[str],
                             state: Dict,
                             lane_statistics: Dict,
                             rewards: Dict[str, float]) -> float:
        """
        更新决策模型（MARL训练步骤）

        Args:
            ego_vehicle_ids: CAV的ID列表
            state: 环境状态字典
            lane_statistics: 车道统计信息
            rewards: {vehicle_id: reward} 奖励字典

        Returns:
            loss_value: 损失值
        """
        if not self.enable_training or self.optimizer is None:
            raise RuntimeError("微调未配置，请先调用 setup_fine_tuning()")

        # 计算损失
        loss = self.compute_decision_loss(ego_vehicle_ids, state, lane_statistics, rewards)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（避免梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.decision_model.get_trainable_parameters(), max_norm=1.0)

        # 更新参数
        self.optimizer.step()

        # 应用因果约束（保持因果结构）
        self.decision_model.apply_causal_constraints()

        # 记录损失
        loss_value = loss.item()
        self.fine_tune_losses.append(loss_value)

        return loss_value

    def on_episode_end(self, save_dir: Optional[str] = None):
        """
        Episode结束时的处理

        Args:
            save_dir: 模型保存目录（可选）
        """
        self.total_episodes += 1

        # 更新微调阶段
        self.update_fine_tune_stage()

        # 重置决策统计
        self.decision_maker.reset_episode_stats()

        # 定期保存模型
        if save_dir and self.total_episodes % 100 == 0:
            save_path = f"{save_dir}/scm_decision_ep{self.total_episodes}.pth"
            self.decision_maker.save_model(save_path)
            print(f"[DecisionModule] 保存模型: {save_path}")

    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'total_episodes': self.total_episodes,
            'fine_tune_stage': self.fine_tune_stage,
            'recent_losses': self.fine_tune_losses[-100:] if self.fine_tune_losses else [],
            'avg_loss': np.mean(self.fine_tune_losses[-100:]) if self.fine_tune_losses else 0.0,
            'decision_stats': self.decision_maker.get_decision_stats()
        }

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = self.decision_maker.get_model_info()
        info.update({
            'enable_training': self.enable_training,
            'fine_tune_stage': self.fine_tune_stage,
            'total_episodes': self.total_episodes
        })
        return info


class DecisionModuleFactory:
    """决策模块工厂"""

    _cached_modules = {}

    @classmethod
    def create_module(cls,
                     model_type: str = "shallow_hierarchical",
                     freeze_base_model: bool = True,
                     enable_training: bool = True,
                     device: str = "cpu",
                     use_cache: bool = True) -> DecisionModule:
        """
        创建决策模块（支持单例缓存）

        Args:
            model_type: 决策模型类型
            freeze_base_model: 是否冻结基础模型
            enable_training: 是否启用训练模式
            device: 计算设备
            use_cache: 是否使用缓存

        Returns:
            DecisionModule实例
        """
        if use_cache:
            cache_key = f"{model_type}_{freeze_base_model}_{enable_training}_{device}"
            if cache_key not in cls._cached_modules:
                cls._cached_modules[cache_key] = DecisionModule(
                    model_type=model_type,
                    freeze_base_model=freeze_base_model,
                    enable_training=enable_training,
                    device=device,
                    use_cache=True
                )
            return cls._cached_modules[cache_key]
        else:
            return DecisionModule(
                model_type=model_type,
                freeze_base_model=freeze_base_model,
                enable_training=enable_training,
                device=device,
                use_cache=False
            )


if __name__ == "__main__":
    # 测试决策模块
    print("=" * 80)
    print("测试决策模块集成")
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
                },
                'adj_front': {
                    'id': 'HDV_2',
                    'long_dist': 60.0,
                    'long_rel_v': 1.0,
                },
            }
        },
        'CAV_1': {
            'lane_index': 2,
            'speed': 28.0,
            'acceleration': 0.2,
            'position_x': 120.0,
            'position_y': 7.0,
            'surrounding_vehicles': {}
        }
    }

    lane_statistics = {
        0: {'mean_speed': 20.0, 'density': 0.08},
        1: {'mean_speed': 25.0, 'density': 0.05},
        2: {'mean_speed': 28.0, 'density': 0.04}
    }

    # 创建决策模块
    dec_module = DecisionModuleFactory.create_module(
        model_type="shallow_hierarchical",
        freeze_base_model=True,
        enable_training=True,
        device="cpu",
        use_cache=True
    )

    # 配置微调策略
    dec_module.setup_fine_tuning(learning_rate=1e-4, stage_thresholds=(10, 20))

    # 测试1: 单个决策
    print("\n1. 单个CAV决策:")
    decision, prob = dec_module.make_decision('CAV_0', state, lane_statistics, return_prob=True)
    print(f"  CAV_0: 决策={decision} (0=保持/1=换道), 概率={prob:.4f}")

    # 测试2: 批量决策
    print("\n2. 批量CAV决策:")
    decisions = dec_module.make_batch_decisions(['CAV_0', 'CAV_1'], state, lane_statistics, return_prob=True)
    for veh_id, (dec, prob) in decisions.items():
        print(f"  {veh_id}: 决策={dec}, 概率={prob:.4f if prob else 'N/A'}")

    # 测试3: 模拟MARL训练
    print("\n3. 模拟MARL微调:")
    rewards = {'CAV_0': 0.8, 'CAV_1': 0.6}
    for ep in range(25):
        loss = dec_module.update_decision_model(['CAV_0', 'CAV_1'], state, lane_statistics, rewards)
        dec_module.on_episode_end()

        if ep in [0, 9, 19, 24]:
            stats = dec_module.get_training_stats()
            print(f"  Episode {ep}: Loss={loss:.6f}, Stage={stats['fine_tune_stage']}")

    # 测试4: 训练统计
    print("\n4. 训练统计:")
    stats = dec_module.get_training_stats()
    print(f"  总Episodes: {stats['total_episodes']}")
    print(f"  微调阶段: {stats['fine_tune_stage']}")
    print(f"  平均损失: {stats['avg_loss']:.6f}")

    print("\n✓ 决策模块测试完成!")
