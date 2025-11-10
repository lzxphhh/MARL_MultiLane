#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM横向意图决策模块

基于预训练的SCM意图预测模型生成CAV的换道决策
支持MARL训练过程中的在线微调

主要组件:
- scm_decision_model: SCM决策模型定义
- execute_decision: 决策执行器

使用示例:
    from harl.decisions.lateral_decisions.SCM_decisions import SCMDecisionMakerFactory

    # 创建决策生成器
    decision_maker = SCMDecisionMakerFactory.create_decision_maker(
        model_type="shallow_hierarchical",
        enable_training=True  # 启用MARL微调
    )

    # 生成决策
    decisions = decision_maker.decide(env_features, ind_features, vehicle_types)
"""

from .scm_decision_model import (
    SCMDecisionModel,
    SCMDecisionModelFactory,
    PathArchitecture,
    FusionStrategy
)

from .execute_decision import (
    SCMDecisionMaker,
    SCMDecisionMakerFactory
)

__all__ = [
    # 决策模型
    'SCMDecisionModel',
    'SCMDecisionModelFactory',

    # 枚举类型
    'PathArchitecture',
    'FusionStrategy',

    # 决策生成器
    'SCMDecisionMaker',
    'SCMDecisionMakerFactory',
]

__version__ = '1.0.0'
__author__ = '交通流研究团队'
