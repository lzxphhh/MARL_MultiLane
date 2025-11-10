#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CQR轨迹占用预测模块

提供基于Conformal Quantile Regression (CQR)的车辆轨迹占用预测功能
支持QR和CQR两种模式

主要组件:
- models: GRU-QR模型定义
- conformalization: CQR校准器
- execute_prediction: 预测执行器
- feature_encoder: 特征编码器

使用示例:
    from harl.prediction.occupancy.strategy.CQR_prediction import CQRPredictorFactory

    # 创建预测器
    predictor = CQRPredictorFactory.create_predictor(
        model_type="CQR-GRU-uncertainty",
        device="cpu"
    )

    # 执行预测
    x_min, median, x_max = predictor.predict(features)
"""

from .models import GRUQR, MLPQR
from .conformalization import ConformalizationCalibrator
from .execute_prediction import CQRPredictor, CQRPredictorFactory

__all__ = [
    # 模型类
    'GRUQR',
    'MLPQR',

    # 校准器
    'ConformalizationCalibrator',

    # 预测器
    'CQRPredictor',
    'CQRPredictorFactory',
]

__version__ = '1.0.0'
__author__ = '交通流研究团队'
