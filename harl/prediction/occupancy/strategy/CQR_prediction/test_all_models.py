#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有四个预训练模型的性能和输出

测试内容:
1. 加载QR-GRU-uncertainty模型
2. 加载QR-GRU-uncertainty_pcgrad模型
3. 加载CQR-GRU-uncertainty模型
4. 加载CQR-GRU-uncertainty_pcgrad模型
5. 对比四个模型的预测结果
6. 验证CQR校准的有效性
"""

import numpy as np
import torch
try:
    from execute_prediction import CQRPredictorFactory
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from execute_prediction import CQRPredictorFactory


def create_test_features(batch_size=3, seed=42):
    """创建测试数据"""
    np.random.seed(seed)

    features = {
        # 交通状态 [batch, 6]: [EL_v, EL_k, LL_v, LL_k, RL_v, RL_k]
        'traffic_state': np.array([
            [25.0, 0.05, 28.0, 0.04, 26.0, 0.045],  # 中等密度
            [15.0, 0.12, 18.0, 0.10, 16.0, 0.11],   # 高密度
            [30.0, 0.02, 32.0, 0.018, 31.0, 0.019]  # 低密度
        ], dtype=np.float32),

        # 自车当前状态 [batch, 4]: [v, a, lane_id, eta]
        'ego_current': np.array([
            [27.0, 0.5, 1.0, 10.0],
            [17.0, -0.3, 1.0, 15.0],
            [31.0, 0.1, 2.0, 8.0]
        ], dtype=np.float32),

        # 自车历史轨迹 [batch, 20, 5]: [Δx, Δy, Δv, Δa, lane_id]
        'ego_history': np.random.randn(batch_size, 20, 5).astype(np.float32) * 0.1,

        # 周边车辆当前状态 [batch, 6, 8]
        'sur_current': np.random.randn(batch_size, 6, 8).astype(np.float32) * 0.5,

        # 周边车辆历史轨迹 [batch, 6, 20, 5]
        'sur_history': np.random.randn(batch_size, 6, 20, 5).astype(np.float32) * 0.1,

        # 横向意图 [batch, 1]: 0=保持, 1=换道
        'intention': np.array([[0], [1], [0]], dtype=np.float32)
    }

    return features


def test_all_models():
    """测试所有四个模型"""
    print("=" * 80)
    print("测试CQR轨迹占用预测模型")
    print("=" * 80)

    # 准备测试数据
    features = create_test_features(batch_size=3)
    print(f"\n测试数据形状:")
    for key, value in features.items():
        print(f"  {key:15s}: {value.shape}")

    # 定义要测试的模型
    models_to_test = [
        {
            'name': 'QR-GRU-uncertainty',
            'type': 'QR',
            'use_calibrator': False
        },
        {
            'name': 'QR-GRU-uncertainty_pcgrad',
            'type': 'QR',
            'use_calibrator': False
        },
        {
            'name': 'CQR-GRU-uncertainty',
            'type': 'CQR',
            'use_calibrator': True,
            'calibrator_type': 'standard'
        },
        {
            'name': 'CQR-GRU-uncertainty_pcgrad',
            'type': 'CQR',
            'use_calibrator': True,
            'calibrator_type': 'standard'
        }
    ]

    # 存储结果
    results = {}

    # 测试每个模型
    for model_config in models_to_test:
        model_name = model_config['name']
        print(f"\n{'=' * 80}")
        print(f"测试模型: {model_name}")
        print('=' * 80)

        try:
            # 创建预测器
            if model_config['use_calibrator']:
                predictor = CQRPredictorFactory.create_predictor(
                    model_type=model_name,
                    calibrator_type=model_config.get('calibrator_type', 'standard'),
                    device="cpu",
                    use_cache=False
                )
            else:
                predictor = CQRPredictorFactory.create_predictor(
                    model_type=model_name,
                    device="cpu",
                    use_cache=False
                )

            # 执行预测
            lower, median, upper = predictor.predict(features, return_intervals=True)

            # 保存结果
            results[model_name] = {
                'lower': lower,
                'median': median,
                'upper': upper,
                'interval_width': upper - lower
            }

            # 显示预测结果
            print(f"\n预测结果:")
            for i in range(3):
                print(f"  样本 {i+1}:")
                print(f"    横向意图: {'换道' if features['intention'][i, 0] == 1 else '保持'}")
                print(f"    第1秒中位数: {median[i, :10].mean():.4f}m")
                print(f"    第2秒中位数: {median[i, 10:20].mean():.4f}m")
                print(f"    第3秒中位数: {median[i, 20:30].mean():.4f}m")
                print(f"    平均区间宽度: {(upper[i] - lower[i]).mean():.4f}m")
                print(f"    第1秒区间: [{lower[i, 0]:.4f}, {upper[i, 0]:.4f}]")
                print(f"    第30步区间: [{lower[i, 29]:.4f}, {upper[i, 29]:.4f}]")

            # 显示模型信息
            info = predictor.get_model_info()
            print(f"\n模型配置:")
            print(f"  模型类型: {info['model_type']}")
            print(f"  使用共形化: {info['use_conformal']}")
            if 'calibrator' in info:
                print(f"  目标覆盖率: {info['calibrator']['target_coverage']}")
                print(f"  校准样本数: {info['calibrator']['n_calib']}")

        except Exception as e:
            print(f"\n❌ 模型 {model_name} 测试失败:")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 模型对比
    print(f"\n{'=' * 80}")
    print("模型对比分析")
    print('=' * 80)

    if len(results) >= 2:
        # 1. QR模型对比
        print(f"\n1. QR模型对比 (uncertainty vs pcgrad)")
        print('-' * 80)
        if 'QR-GRU-uncertainty' in results and 'QR-GRU-uncertainty_pcgrad' in results:
            qr_basic = results['QR-GRU-uncertainty']
            qr_pcgrad = results['QR-GRU-uncertainty_pcgrad']

            median_diff = np.abs(qr_basic['median'] - qr_pcgrad['median']).mean()
            width_basic = qr_basic['interval_width'].mean()
            width_pcgrad = qr_pcgrad['interval_width'].mean()

            print(f"  中位数预测差异: {median_diff:.4f}m")
            print(f"  Basic区间宽度: {width_basic:.4f}m")
            print(f"  PCGrad区间宽度: {width_pcgrad:.4f}m")
            print(f"  区间宽度差异: {abs(width_basic - width_pcgrad):.4f}m")

        # 2. CQR模型对比
        print(f"\n2. CQR模型对比 (uncertainty vs pcgrad)")
        print('-' * 80)
        if 'CQR-GRU-uncertainty' in results and 'CQR-GRU-uncertainty_pcgrad' in results:
            cqr_basic = results['CQR-GRU-uncertainty']
            cqr_pcgrad = results['CQR-GRU-uncertainty_pcgrad']

            median_diff = np.abs(cqr_basic['median'] - cqr_pcgrad['median']).mean()
            width_basic = cqr_basic['interval_width'].mean()
            width_pcgrad = cqr_pcgrad['interval_width'].mean()

            print(f"  中位数预测差异: {median_diff:.4f}m")
            print(f"  Basic区间宽度: {width_basic:.4f}m")
            print(f"  PCGrad区间宽度: {width_pcgrad:.4f}m")
            print(f"  区间宽度差异: {abs(width_basic - width_pcgrad):.4f}m")

        # 3. QR vs CQR对比
        print(f"\n3. QR vs CQR 校准效果")
        print('-' * 80)
        for base_name in ['GRU-uncertainty', 'GRU-uncertainty_pcgrad']:
            qr_name = f'QR-{base_name}'
            cqr_name = f'CQR-{base_name}'

            if qr_name in results and cqr_name in results:
                print(f"\n  {base_name}:")
                qr_width = results[qr_name]['interval_width'].mean()
                cqr_width = results[cqr_name]['interval_width'].mean()
                expansion = cqr_width - qr_width
                expansion_pct = (expansion / qr_width) * 100

                print(f"    QR区间宽度:  {qr_width:.4f}m")
                print(f"    CQR区间宽度: {cqr_width:.4f}m")
                print(f"    区间扩展:    {expansion:.4f}m (+{expansion_pct:.1f}%)")

                # 分析不同时间步的扩展
                qr_w = results[qr_name]['interval_width']
                cqr_w = results[cqr_name]['interval_width']
                print(f"    第1秒扩展: {(cqr_w[:, :10] - qr_w[:, :10]).mean():.4f}m")
                print(f"    第2秒扩展: {(cqr_w[:, 10:20] - qr_w[:, 10:20]).mean():.4f}m")
                print(f"    第3秒扩展: {(cqr_w[:, 20:30] - qr_w[:, 20:30]).mean():.4f}m")

    # 4. 预测轨迹演化分析
    print(f"\n4. 预测轨迹时序演化")
    print('-' * 80)
    for model_name, model_results in results.items():
        print(f"\n  {model_name}:")
        median = model_results['median']
        width = model_results['interval_width']

        # 按时间段统计
        for t_start, t_end, label in [(0, 10, '1s'), (10, 20, '2s'), (20, 30, '3s')]:
            med_mean = median[:, t_start:t_end].mean()
            med_std = median[:, t_start:t_end].std()
            width_mean = width[:, t_start:t_end].mean()
            print(f"    {label}: 中位数={med_mean:.4f}±{med_std:.4f}m, 区间宽度={width_mean:.4f}m")

    print(f"\n{'=' * 80}")
    print("✓ 所有模型测试完成!")
    print('=' * 80)


if __name__ == "__main__":
    test_all_models()
