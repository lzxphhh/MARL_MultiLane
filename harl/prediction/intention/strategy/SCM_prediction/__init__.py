#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCM意图预测模块

提供基于结构因果模型(SCM)的车辆换道意图预测功能
支持多种模型架构和融合策略

主要组件:
- scm_model_v2: SCM模型定义
- execute_prediction: 预测执行器

使用示例:
    from harl.prediction.intention.strategy.SCM_prediction import SCMPredictorFactory

    # 创建预测器
    predictor = SCMPredictorFactory.create_predictor(
        model_type="shallow_hierarchical",
        device="cpu"
    )

    # 执行预测
    predictions = predictor.predict(env_features, ind_features, vehicle_types)
"""

from .scm_model_v2 import (
    HierarchicalSCMModelV2,
    ThreePathSCM,
    PathEncoder,
    PathArchitecture,
    FusionStrategy,
    SCMDataset
)

from .execute_prediction import (
    SCMPredictor,
    SCMPredictorFactory
)

__all__ = [
    # 模型类
    'HierarchicalSCMModelV2',
    'ThreePathSCM',
    'PathEncoder',

    # 枚举类型
    'PathArchitecture',
    'FusionStrategy',

    # 数据集
    'SCMDataset',

    # 预测器
    'SCMPredictor',
    'SCMPredictorFactory',
]

__version__ = '2.0.0'
__author__ = '交通流研究团队'
