#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM横向意图决策模型
基于预训练的SCM意图预测模型，用于CAV的换道决策

功能：
1. 加载预训练的SCM模型作为初始化
2. 生成与人类驾驶意图逻辑相同的换道决策
3. 支持在MARL训练过程中进行微调
4. 可切换为纯决策模式或预测模式

设计思路：
- 复用SCM预测模型的结构（HierarchicalSCMModelV2）
- 初始化时加载预训练权重
- 训练时可以冻结部分层或全部微调
- 输出换道意图概率，用于决策

作者：交通流研究团队
日期：2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import sys

# 导入SCM预测模型
try:
    # 尝试从prediction模块导入
    from harl.prediction.intention.strategy.SCM_prediction.scm_model_v2 import (
        HierarchicalSCMModelV2,
        PathArchitecture,
        FusionStrategy
    )
except ImportError:
    # 如果失败，添加路径后导入
    current_dir = Path(__file__).parent
    scm_pred_path = current_dir.parent.parent.parent / "prediction" / "intention" / "strategy" / "SCM_prediction"
    sys.path.insert(0, str(scm_pred_path))
    from scm_model_v2 import (
        HierarchicalSCMModelV2,
        PathArchitecture,
        FusionStrategy
    )


class SCMDecisionModel(nn.Module):
    """
    SCM横向意图决策模型

    继承自HierarchicalSCMModelV2，添加决策相关功能：
    1. 支持从预训练模型加载权重
    2. 支持选择性冻结层
    3. 支持MARL训练时的微调
    4. 输出决策概率和中间结果
    """

    def __init__(self,
                 env_dim: int = 4,
                 ind_dim: int = 10,
                 path_architecture: PathArchitecture = PathArchitecture.SHALLOW,
                 fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
                 pretrained_model_path: Optional[str] = None,
                 freeze_encoder: bool = False,
                 freeze_env_layer: bool = False,
                 freeze_ind_layer: bool = False,
                 use_vehicle_type: bool = True,
                 enforce_causal_constraints: bool = True):
        """
        初始化SCM决策模型

        Args:
            env_dim: 环境特征维度
            ind_dim: 个体特征维度
            path_architecture: 路径编码架构
            fusion_strategy: 路径融合策略
            pretrained_model_path: 预训练模型路径
            freeze_encoder: 是否冻结编码器层
            freeze_env_layer: 是否冻结环境层
            freeze_ind_layer: 是否冻结个体层（三路径）
            use_vehicle_type: 是否使用车辆类型调制
            enforce_causal_constraints: 是否强制因果约束
        """
        # 不调用super().__init__，而是手动创建SCM模型
        super(SCMDecisionModel, self).__init__()

        self.env_dim = env_dim
        self.ind_dim = ind_dim
        self.path_architecture = path_architecture
        self.fusion_strategy = fusion_strategy
        self.use_vehicle_type = use_vehicle_type
        self.enforce_causal_constraints = enforce_causal_constraints

        # 创建基础SCM模型
        self.scm_model = HierarchicalSCMModelV2(
            env_dim=env_dim,
            ind_dim=ind_dim,
            path_architecture=path_architecture,
            fusion_strategy=fusion_strategy,
            use_vehicle_type=use_vehicle_type,
            enforce_causal_constraints=enforce_causal_constraints
        )

        # 加载预训练权重（如果提供）
        if pretrained_model_path is not None:
            self.load_pretrained_weights(pretrained_model_path)

        # 冻结指定的层
        if freeze_env_layer:
            self.freeze_layer(self.scm_model.env_layer)

        if freeze_ind_layer:
            self.freeze_layer(self.scm_model.ind_layer)

        # 决策相关参数
        self.decision_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 统计信息
        self.training_stats = {
            'total_decisions': 0,
            'lane_change_decisions': 0,
            'keep_lane_decisions': 0
        }

    def load_pretrained_weights(self, model_path: str):
        """
        加载预训练的SCM模型权重

        Args:
            model_path: 预训练模型路径（.pth文件）
        """
        print(f"[SCMDecisionModel] 加载预训练权重: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')

        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 加载到scm_model
        self.scm_model.load_state_dict(state_dict, strict=True)

        print(f"[SCMDecisionModel] 预训练权重加载完成")

        # 显示加载的参数信息
        if 'causal_parameters' in checkpoint or hasattr(checkpoint, 'get'):
            causal_params = checkpoint.get('causal_parameters', None)
            if causal_params:
                print(f"  环境层权重: {causal_params.get('alpha_env', 'N/A')}")

    def freeze_layer(self, layer: nn.Module):
        """冻结指定层的参数"""
        for param in layer.parameters():
            param.requires_grad = False
        print(f"[SCMDecisionModel] 冻结层: {layer.__class__.__name__}")

    def unfreeze_layer(self, layer: nn.Module):
        """解冻指定层的参数"""
        for param in layer.parameters():
            param.requires_grad = True
        print(f"[SCMDecisionModel] 解冻层: {layer.__class__.__name__}")

    def forward(self,
                env_features: torch.Tensor,
                ind_features: torch.Tensor,
                vehicle_type: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播 - 生成换道决策

        Args:
            env_features: [batch, env_dim] 环境特征
            ind_features: [batch, ind_dim] 个体特征
            vehicle_type: [batch, 1] 车辆类型 (0=HDV, 1=CAV)
            return_intermediates: 是否返回中间结果

        Returns:
            decision_prob: [batch, 1] 换道决策概率
            intermediates: (可选) 中间结果字典
        """
        # 使用SCM模型前向传播
        P_final, intermediates = self.scm_model(env_features, ind_features, vehicle_type)

        # 决策概率就是P_final
        decision_prob = P_final

        if return_intermediates:
            # 添加决策相关的中间结果
            intermediates['decision_threshold'] = self.decision_threshold
            intermediates['decision'] = (decision_prob > self.decision_threshold).float()
            return decision_prob, intermediates
        else:
            return decision_prob, None

    def make_decision(self,
                     env_features: torch.Tensor,
                     ind_features: torch.Tensor,
                     vehicle_type: Optional[torch.Tensor] = None,
                     use_threshold: bool = True,
                     return_prob: bool = False) -> torch.Tensor:
        """
        做出换道决策

        Args:
            env_features: [batch, env_dim] 环境特征
            ind_features: [batch, ind_dim] 个体特征
            vehicle_type: [batch, 1] 车辆类型
            use_threshold: 是否使用阈值进行二值化决策
            return_prob: 是否返回概率而非二值决策

        Returns:
            decision: [batch, 1]
                - 如果return_prob=True: 换道概率 [0, 1]
                - 如果return_prob=False: 二值决策 {0, 1}
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            decision_prob, _ = self.forward(env_features, ind_features, vehicle_type)

        if return_prob:
            return decision_prob
        else:
            if use_threshold:
                decision = (decision_prob > self.decision_threshold).float()
            else:
                decision = (decision_prob > 0.5).float()

            # 更新统计
            if not self.training:
                self.training_stats['total_decisions'] += decision.size(0)
                self.training_stats['lane_change_decisions'] += decision.sum().item()
                self.training_stats['keep_lane_decisions'] += (decision.size(0) - decision.sum().item())

            return decision

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练的参数列表"""
        return [p for p in self.parameters() if p.requires_grad]

    def get_frozen_parameters(self) -> List[nn.Parameter]:
        """获取冻结的参数列表"""
        return [p for p in self.parameters() if not p.requires_grad]

    def print_parameter_status(self):
        """打印参数冻结状态"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n[SCMDecisionModel] 参数状态:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

        # 显示各层状态
        print(f"\n  各层状态:")
        print(f"    环境层: {'冻结' if not next(self.scm_model.env_layer.parameters()).requires_grad else '可训练'}")
        print(f"    个体层: {'冻结' if not next(self.scm_model.ind_layer.parameters()).requires_grad else '可训练'}")
        if self.use_vehicle_type:
            print(f"    类型调制: {'冻结' if not self.scm_model.w_env_base.requires_grad else '可训练'}")
        print(f"    决策阈值: {self.decision_threshold.item():.4f} ({'可训练' if self.decision_threshold.requires_grad else '冻结'})")

    def get_decision_stats(self) -> Dict:
        """获取决策统计信息"""
        stats = self.training_stats.copy()
        if stats['total_decisions'] > 0:
            stats['lane_change_rate'] = stats['lane_change_decisions'] / stats['total_decisions']
            stats['keep_lane_rate'] = stats['keep_lane_decisions'] / stats['total_decisions']
        else:
            stats['lane_change_rate'] = 0.0
            stats['keep_lane_rate'] = 0.0
        return stats

    def reset_decision_stats(self):
        """重置决策统计"""
        self.training_stats = {
            'total_decisions': 0,
            'lane_change_decisions': 0,
            'keep_lane_decisions': 0
        }

    def apply_causal_constraints(self):
        """应用因果约束（训练时调用）"""
        self.scm_model.apply_causal_constraints()

    def get_causal_parameters(self) -> Dict:
        """获取因果参数"""
        return self.scm_model.get_causal_parameters()


class SCMDecisionModelFactory:
    """
    SCM决策模型工厂类
    用于创建和管理决策模型
    """

    @staticmethod
    def create_from_pretrained(
        pretrained_model_path: str,
        path_architecture: PathArchitecture = PathArchitecture.SHALLOW,
        fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
        freeze_encoder: bool = False,
        freeze_env_layer: bool = False,
        freeze_ind_layer: bool = False,
        device: str = "cpu"
    ) -> SCMDecisionModel:
        """
        从预训练模型创建决策模型

        Args:
            pretrained_model_path: 预训练模型路径
            path_architecture: 路径架构（必须与预训练模型一致）
            fusion_strategy: 融合策略（必须与预训练模型一致）
            freeze_encoder: 是否冻结编码器
            freeze_env_layer: 是否冻结环境层
            freeze_ind_layer: 是否冻结个体层
            device: 运行设备

        Returns:
            SCMDecisionModel实例
        """
        model = SCMDecisionModel(
            env_dim=4,
            ind_dim=10,
            path_architecture=path_architecture,
            fusion_strategy=fusion_strategy,
            pretrained_model_path=pretrained_model_path,
            freeze_encoder=freeze_encoder,
            freeze_env_layer=freeze_env_layer,
            freeze_ind_layer=freeze_ind_layer,
            use_vehicle_type=True,
            enforce_causal_constraints=True
        )

        model.to(device)
        model.print_parameter_status()

        return model

    @staticmethod
    def create_for_fine_tuning(
        pretrained_model_path: str,
        path_architecture: PathArchitecture = PathArchitecture.SHALLOW,
        fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
        freeze_base_model: bool = True,
        device: str = "cpu"
    ) -> SCMDecisionModel:
        """
        创建用于微调的模型（推荐配置）

        Args:
            pretrained_model_path: 预训练模型路径
            path_architecture: 路径架构
            fusion_strategy: 融合策略
            freeze_base_model: 是否冻结基础模型（仅训练决策阈值）
            device: 运行设备

        Returns:
            SCMDecisionModel实例
        """
        model = SCMDecisionModel(
            env_dim=4,
            ind_dim=10,
            path_architecture=path_architecture,
            fusion_strategy=fusion_strategy,
            pretrained_model_path=pretrained_model_path,
            freeze_encoder=freeze_base_model,
            freeze_env_layer=freeze_base_model,
            freeze_ind_layer=freeze_base_model,
            use_vehicle_type=True,
            enforce_causal_constraints=True
        )

        model.to(device)

        print(f"\n[SCMDecisionModelFactory] 创建微调模型")
        print(f"  基础模型: {'冻结' if freeze_base_model else '可训练'}")
        print(f"  决策阈值: 可训练")
        model.print_parameter_status()

        return model


def test_scm_decision_model():
    """测试SCM决策模型"""
    print("=" * 80)
    print("测试SCM横向意图决策模型")
    print("=" * 80)

    # 查找预训练模型
    current_dir = Path(__file__).parent
    model_base = current_dir.parent.parent.parent / "prediction" / "intention" / "models" / "SCM_models"

    # 尝试两个模型
    for model_name in ["shallow_hierarchical", "medium_hierarchical"]:
        model_path = model_base / model_name / "best_model.pth"

        if not model_path.exists():
            print(f"\n跳过 {model_name}: 模型文件不存在")
            continue

        print(f"\n{'=' * 80}")
        print(f"测试模型: {model_name}")
        print('=' * 80)

        # 确定架构
        arch = PathArchitecture.SHALLOW if "shallow" in model_name else PathArchitecture.MEDIUM

        # 创建决策模型
        decision_model = SCMDecisionModelFactory.create_from_pretrained(
            pretrained_model_path=str(model_path),
            path_architecture=arch,
            fusion_strategy=FusionStrategy.HIERARCHICAL,
            freeze_env_layer=True,
            freeze_ind_layer=False,  # 个体层可训练
            device="cpu"
        )

        # 准备测试数据
        batch_size = 5
        env_features = torch.randn(batch_size, 4)
        ind_features = torch.randn(batch_size, 10)
        vehicle_types = torch.randint(0, 2, (batch_size, 1)).float()

        # 测试前向传播
        print(f"\n1. 测试前向传播")
        decision_prob, intermediates = decision_model(
            env_features, ind_features, vehicle_types,
            return_intermediates=True
        )
        print(f"  决策概率形状: {decision_prob.shape}")
        print(f"  决策概率范围: [{decision_prob.min():.4f}, {decision_prob.max():.4f}]")

        # 测试决策
        print(f"\n2. 测试决策生成")
        decisions = decision_model.make_decision(
            env_features, ind_features, vehicle_types,
            use_threshold=True, return_prob=False
        )
        print(f"  决策结果: {decisions.squeeze().numpy()}")
        print(f"  换道决策数: {decisions.sum().item()}/{batch_size}")

        # 显示统计
        print(f"\n3. 决策统计")
        stats = decision_model.get_decision_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 测试中间结果
        print(f"\n4. 中间结果")
        print(f"  决策阈值: {intermediates['decision_threshold'].item():.4f}")
        print(f"  环境贡献: {intermediates['psi_env'][:3].squeeze().numpy()}")
        print(f"  个体贡献: {intermediates['psi_ind'][:3].squeeze().numpy()}")
        print(f"  门控值: {intermediates['gate'][:3].squeeze().numpy()}")

        break  # 只测试第一个找到的模型

    print(f"\n{'=' * 80}")
    print("✓ 测试完成!")
    print('=' * 80)


if __name__ == "__main__":
    test_scm_decision_model()
