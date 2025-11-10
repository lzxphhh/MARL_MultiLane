#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM横向意图决策执行模块
用于在MARL环境中生成CAV的换道决策

功能：
1. 加载预训练的SCM模型作为决策基准
2. 生成与人类驾驶逻辑一致的换道决策
3. 支持MARL训练过程中的在线微调
4. 提供批量决策和单样本决策接口
5. 支持决策策略的评估和分析

作者：交通流研究团队
日期：2025
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Handle both package import and standalone execution
try:
    from .scm_decision_model import SCMDecisionModel, SCMDecisionModelFactory, PathArchitecture, FusionStrategy
except ImportError:
    from scm_decision_model import SCMDecisionModel, SCMDecisionModelFactory, PathArchitecture, FusionStrategy


class SCMDecisionMaker:
    """
    SCM横向决策生成器
    用于在MARL环境中为CAV生成换道决策
    """

    def __init__(self,
                 model_type: str = "shallow_hierarchical",
                 model_base_path: Optional[str] = None,
                 freeze_base_model: bool = True,
                 device: str = "cpu",
                 enable_training: bool = False):
        """
        初始化SCM决策生成器

        Args:
            model_type: 预训练模型类型
                - "shallow_hierarchical": Shallow架构
                - "medium_hierarchical": Medium架构
            model_base_path: 模型文件夹基础路径
            freeze_base_model: 是否冻结基础SCM模型
            device: 运行设备
            enable_training: 是否启用训练模式（用于MARL微调）
        """
        self.model_type = model_type
        self.device = torch.device(device)
        self.enable_training = enable_training
        self.freeze_base_model = freeze_base_model

        # 设置模型路径
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent.parent / "prediction" / "intention" / "models" / "SCM_models"

        self.model_path = Path(model_base_path) / model_type / "best_model.pth"

        if not self.model_path.exists():
            raise FileNotFoundError(f"预训练模型不存在: {self.model_path}")

        # 确定模型架构
        if "shallow" in model_type:
            self.path_architecture = PathArchitecture.SHALLOW
        elif "medium" in model_type:
            self.path_architecture = PathArchitecture.MEDIUM
        else:
            self.path_architecture = PathArchitecture.LINEAR

        self.fusion_strategy = FusionStrategy.HIERARCHICAL

        # 初始化模型
        self.model = None
        self._load_model()

        # 决策统计
        self.episode_stats = []
        self.current_episode_decisions = []

        print(f"[SCMDecisionMaker] 成功初始化决策生成器")
        print(f"[SCMDecisionMaker] 模型类型: {model_type}")
        print(f"[SCMDecisionMaker] 冻结基础模型: {freeze_base_model}")
        print(f"[SCMDecisionMaker] 训练模式: {enable_training}")
        print(f"[SCMDecisionMaker] 运行设备: {self.device}")

    def _load_model(self):
        """加载决策模型"""
        print(f"[SCMDecisionMaker] 加载预训练模型: {self.model_path}")

        if self.enable_training:
            # 训练模式：创建用于微调的模型
            self.model = SCMDecisionModelFactory.create_for_fine_tuning(
                pretrained_model_path=str(self.model_path),
                path_architecture=self.path_architecture,
                fusion_strategy=self.fusion_strategy,
                freeze_base_model=self.freeze_base_model,
                device=str(self.device)
            )
            self.model.train()
        else:
            # 推理模式：创建固定的决策模型
            self.model = SCMDecisionModelFactory.create_from_pretrained(
                pretrained_model_path=str(self.model_path),
                path_architecture=self.path_architecture,
                fusion_strategy=self.fusion_strategy,
                freeze_encoder=True,
                freeze_env_layer=True,
                freeze_ind_layer=True,
                device=str(self.device)
            )
            self.model.eval()

    def decide(self,
               env_features: Union[np.ndarray, torch.Tensor],
               ind_features: Union[np.ndarray, torch.Tensor],
               vehicle_types: Optional[Union[np.ndarray, torch.Tensor]] = None,
               return_prob: bool = False,
               return_intermediates: bool = False) -> Union[np.ndarray, Tuple]:
        """
        生成换道决策

        Args:
            env_features: [N, 4] 环境特征
                - v_avg_norm: 平均速度（归一化）
                - kappa_avg_norm: 平均密度（归一化）
                - delta_v_norm: 车道速度差（归一化）
                - delta_kappa_norm: 车道密度差（归一化）

            ind_features: [N, 10] 个体特征（归一化）
                - v_ego_norm: 自车速度
                - v_rel_norm: 与前车相对速度
                - d_headway_norm: 车头间距
                - T_headway_norm: 时间间距
                - delta_v_adj_front_norm: 与目标车道前车速度差
                - d_adj_front_norm: 与目标车道前车距离
                - T_adj_front_norm: 与目标车道前车时间间隙
                - delta_v_adj_rear_norm: 与目标车道后车速度差
                - d_adj_rear_norm: 与目标车道后车距离
                - T_adj_rear_norm: 与目标车道后车时间间隙

            vehicle_types: [N, 1] 车辆类型 (0=HDV, 1=CAV)，可选
            return_prob: 是否返回概率（默认返回二值决策）
            return_intermediates: 是否返回中间结果

        Returns:
            如果return_intermediates=False:
                decisions: [N, 1] 决策
                    - 如果return_prob=True: 换道概率 [0, 1]
                    - 如果return_prob=False: 二值决策 {0, 1}
            如果return_intermediates=True:
                (decisions, intermediates_dict)
        """
        # 转换为tensor
        if isinstance(env_features, np.ndarray):
            env_features = torch.FloatTensor(env_features)
        if isinstance(ind_features, np.ndarray):
            ind_features = torch.FloatTensor(ind_features)
        if vehicle_types is not None and isinstance(vehicle_types, np.ndarray):
            vehicle_types = torch.FloatTensor(vehicle_types)

        # 确保形状正确
        if env_features.dim() == 1:
            env_features = env_features.unsqueeze(0)
        if ind_features.dim() == 1:
            ind_features = ind_features.unsqueeze(0)
        if vehicle_types is not None and vehicle_types.dim() == 1:
            vehicle_types = vehicle_types.unsqueeze(1)

        # 移动到设备
        env_features = env_features.to(self.device)
        ind_features = ind_features.to(self.device)
        if vehicle_types is not None:
            vehicle_types = vehicle_types.to(self.device)

        # 生成决策
        if self.enable_training:
            # 训练模式：保持梯度
            decisions = self.model.make_decision(
                env_features, ind_features, vehicle_types,
                use_threshold=True, return_prob=return_prob
            )
        else:
            # 推理模式：不计算梯度
            with torch.no_grad():
                decisions = self.model.make_decision(
                    env_features, ind_features, vehicle_types,
                    use_threshold=True, return_prob=return_prob
                )

        # 转换为numpy
        decisions_np = decisions.cpu().numpy()

        # 记录决策
        if not self.enable_training:
            for i in range(len(decisions_np)):
                self.current_episode_decisions.append({
                    'decision': int(decisions_np[i, 0]),
                    'prob': float(decisions_np[i, 0]) if return_prob else None
                })

        if return_intermediates:
            with torch.no_grad():
                _, intermediates = self.model(env_features, ind_features, vehicle_types, return_intermediates=True)

            # 转换中间结果为numpy
            intermediates_np = {}
            for key, value in intermediates.items():
                if isinstance(value, torch.Tensor):
                    intermediates_np[key] = value.detach().cpu().numpy()
                else:
                    intermediates_np[key] = value

            return decisions_np, intermediates_np
        else:
            return decisions_np

    def decide_single(self,
                     env_feature: Union[np.ndarray, List],
                     ind_feature: Union[np.ndarray, List],
                     vehicle_type: int = 1) -> Dict[str, float]:
        """
        单样本决策（便捷接口）

        Args:
            env_feature: [4] 环境特征向量
            ind_feature: [10] 个体特征向量
            vehicle_type: 车辆类型 (0=HDV, 1=CAV)

        Returns:
            结果字典:
                - 'decision': 二值决策 {0, 1}
                - 'probability': 换道概率 [0, 1]
                - 'confidence': 决策置信度
        """
        env_feature = np.array(env_feature).reshape(1, -1)
        ind_feature = np.array(ind_feature).reshape(1, -1)
        vehicle_type_arr = np.array([[vehicle_type]], dtype=np.float32)

        # 获取概率
        prob = self.decide(env_feature, ind_feature, vehicle_type_arr, return_prob=True)

        # 获取二值决策
        decision = self.decide(env_feature, ind_feature, vehicle_type_arr, return_prob=False)

        # 计算置信度（距离阈值的远近）
        threshold = self.model.decision_threshold.item()
        confidence = abs(float(prob[0, 0]) - threshold)

        return {
            'decision': int(decision[0, 0]),
            'probability': float(prob[0, 0]),
            'confidence': confidence,
            'threshold': threshold
        }

    def update_decision_threshold(self, new_threshold: float):
        """更新决策阈值（用于在线调整）"""
        with torch.no_grad():
            self.model.decision_threshold.fill_(new_threshold)
        print(f"[SCMDecisionMaker] 决策阈值更新为: {new_threshold:.4f}")

    def get_model(self) -> SCMDecisionModel:
        """获取底层模型（用于MARL训练）"""
        return self.model

    def set_training_mode(self, mode: bool):
        """设置训练模式"""
        self.enable_training = mode
        if mode:
            self.model.train()
        else:
            self.model.eval()

    def get_decision_stats(self) -> Dict:
        """获取当前episode的决策统计"""
        if len(self.current_episode_decisions) == 0:
            return {
                'total_decisions': 0,
                'lane_change_count': 0,
                'keep_lane_count': 0,
                'lane_change_rate': 0.0
            }

        total = len(self.current_episode_decisions)
        lane_changes = sum(1 for d in self.current_episode_decisions if d['decision'] == 1)

        return {
            'total_decisions': total,
            'lane_change_count': lane_changes,
            'keep_lane_count': total - lane_changes,
            'lane_change_rate': lane_changes / total if total > 0 else 0.0
        }

    def reset_episode_stats(self):
        """重置episode统计（在episode开始时调用）"""
        if len(self.current_episode_decisions) > 0:
            self.episode_stats.append(self.get_decision_stats())
        self.current_episode_decisions = []

    def get_all_episode_stats(self) -> List[Dict]:
        """获取所有episode的统计"""
        return self.episode_stats

    def save_model(self, save_path: str):
        """保存微调后的模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'path_architecture': self.path_architecture.value,
            'fusion_strategy': self.fusion_strategy.value,
            'decision_threshold': self.model.decision_threshold.item(),
            'episode_stats': self.episode_stats
        }, save_path)
        print(f"[SCMDecisionMaker] 模型已保存: {save_path}")

    def load_finetuned_model(self, model_path: str):
        """加载微调后的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'episode_stats' in checkpoint:
            self.episode_stats = checkpoint['episode_stats']
        print(f"[SCMDecisionMaker] 微调模型已加载: {model_path}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'path_architecture': self.path_architecture.value,
            'fusion_strategy': self.fusion_strategy.value,
            'device': str(self.device),
            'training_mode': self.enable_training,
            'freeze_base_model': self.freeze_base_model,
            'decision_threshold': self.model.decision_threshold.item(),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in self.model.parameters())
        }


class SCMDecisionMakerFactory:
    """
    SCM决策生成器工厂类
    """

    _instances = {}  # 单例缓存

    @classmethod
    def create_decision_maker(cls,
                             model_type: str = "shallow_hierarchical",
                             model_base_path: Optional[str] = None,
                             freeze_base_model: bool = True,
                             device: str = "cpu",
                             enable_training: bool = False,
                             use_cache: bool = True) -> SCMDecisionMaker:
        """
        创建决策生成器（支持单例模式）

        Args:
            model_type: 模型类型
            model_base_path: 模型基础路径
            freeze_base_model: 是否冻结基础模型
            device: 运行设备
            enable_training: 是否启用训练
            use_cache: 是否使用缓存

        Returns:
            SCMDecisionMaker实例
        """
        cache_key = f"{model_type}_{freeze_base_model}_{device}_{enable_training}"

        if use_cache and cache_key in cls._instances:
            print(f"[SCMDecisionMakerFactory] 使用缓存的决策生成器: {cache_key}")
            return cls._instances[cache_key]

        decision_maker = SCMDecisionMaker(
            model_type=model_type,
            model_base_path=model_base_path,
            freeze_base_model=freeze_base_model,
            device=device,
            enable_training=enable_training
        )

        if use_cache:
            cls._instances[cache_key] = decision_maker

        return decision_maker

    @classmethod
    def get_available_models(cls, model_base_path: Optional[str] = None) -> List[str]:
        """获取可用的预训练模型列表"""
        if model_base_path is None:
            current_dir = Path(__file__).parent
            model_base_path = current_dir.parent.parent.parent / "prediction" / "intention" / "models" / "SCM_models"

        model_base_path = Path(model_base_path)
        if not model_base_path.exists():
            return []

        available_models = []
        for item in model_base_path.iterdir():
            if item.is_dir() and (item / "best_model.pth").exists():
                available_models.append(item.name)

        return available_models


def demo_decision_maker():
    """演示如何使用决策生成器"""
    print("=" * 80)
    print("SCM横向意图决策生成器演示")
    print("=" * 80)

    # 1. 查看可用模型
    print("\n1. 查看可用模型")
    print("-" * 80)
    available = SCMDecisionMakerFactory.get_available_models()
    print(f"可用模型: {available}")

    # 2. 创建决策生成器（推理模式）
    print("\n2. 创建决策生成器（推理模式）")
    print("-" * 80)
    decision_maker = SCMDecisionMakerFactory.create_decision_maker(
        model_type="shallow_hierarchical",
        freeze_base_model=True,
        device="cpu",
        enable_training=False
    )

    # 3. 准备测试数据
    np.random.seed(42)
    batch_size = 5

    env_features = np.random.randn(batch_size, 4).astype(np.float32)
    ind_features = np.random.randn(batch_size, 10).astype(np.float32)
    vehicle_types = np.ones((batch_size, 1), dtype=np.float32)  # 全部是CAV

    # 4. 批量决策
    print("\n3. 批量决策")
    print("-" * 80)
    decisions = decision_maker.decide(env_features, ind_features, vehicle_types, return_prob=False)
    probs = decision_maker.decide(env_features, ind_features, vehicle_types, return_prob=True)

    for i in range(batch_size):
        print(f"  CAV {i+1}: 决策={int(decisions[i, 0])}, 概率={probs[i, 0]:.4f}")

    # 5. 单样本决策
    print("\n4. 单样本决策")
    print("-" * 80)
    result = decision_maker.decide_single(
        env_feature=env_features[0],
        ind_feature=ind_features[0],
        vehicle_type=1
    )
    print(f"  决策: {result['decision']}")
    print(f"  概率: {result['probability']:.4f}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  阈值: {result['threshold']:.4f}")

    # 6. 获取中间结果
    print("\n5. 获取中间结果")
    print("-" * 80)
    decisions, intermediates = decision_maker.decide(
        env_features[:2], ind_features[:2], vehicle_types[:2],
        return_prob=False, return_intermediates=True
    )
    print(f"  环境贡献: {intermediates['psi_env'][:, 0]}")
    print(f"  个体贡献: {intermediates['psi_ind'][:, 0]}")
    print(f"  门控值: {intermediates['gate'][:, 0]}")

    # 7. 决策统计
    print("\n6. 决策统计")
    print("-" * 80)
    stats = decision_maker.get_decision_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 8. 模型信息
    print("\n7. 模型信息")
    print("-" * 80)
    info = decision_maker.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_decision_maker()
