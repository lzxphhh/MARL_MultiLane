#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测和决策包装器

在原有analyze_traffic基础上增加预测和决策功能
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加项目根路径
harl_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(harl_root))


def enhance_traffic_analysis_with_predictions(cav_statistics: Dict,
                                              hdv_statistics: Dict,
                                              state: Dict,
                                              lane_statistics: Dict,
                                              pred_module=None,
                                              dec_module=None) -> Tuple[Dict, Dict]:
    """
    在原有traffic analysis基础上增加预测和决策信息

    Args:
        cav_statistics: CAV统计信息（来自analyze_traffic）
        hdv_statistics: HDV统计信息（来自analyze_traffic）
        state: 原始车辆状态
        lane_statistics: 车道统计信息
        pred_module: 预测模块
        dec_module: 决策模块

    Returns:
        enhanced_cav_statistics: 增强的CAV统计（包含决策）
        enhanced_hdv_statistics: 增强的HDV统计（包含预测）
    """
    enhanced_cav_statistics = cav_statistics.copy()
    enhanced_hdv_statistics = hdv_statistics.copy()

    # 1. 预测HDV意图和占用
    if pred_module is not None and len(hdv_statistics) > 0:
        hdv_ids = list(hdv_statistics.keys())
        try:
            hdv_predictions = pred_module.predict_intentions_and_occupancy(
                hdv_ids, state, lane_statistics
            )

            # 将预测结果添加到HDV统计中
            for veh_id, pred in hdv_predictions.items():
                if veh_id in enhanced_hdv_statistics:
                    enhanced_hdv_statistics[veh_id]['predicted_intention'] = pred['intention']
                    enhanced_hdv_statistics[veh_id]['predicted_occupancy'] = pred['occupancy']

        except Exception as e:
            print(f"[Warning] HDV prediction failed: {e}")

    # 2. 生成CAV决策
    if dec_module is not None and len(cav_statistics) > 0:
        cav_ids = list(cav_statistics.keys())
        try:
            cav_decisions = dec_module.make_batch_decisions(
                cav_ids, state, lane_statistics, return_prob=True
            )

            # 将决策结果添加到CAV统计中
            for veh_id, (decision, prob) in cav_decisions.items():
                if veh_id in enhanced_cav_statistics:
                    enhanced_cav_statistics[veh_id]['lateral_decision'] = decision
                    enhanced_cav_statistics[veh_id]['decision_probability'] = prob

        except Exception as e:
            print(f"[Warning] CAV decision failed: {e}")

    return enhanced_cav_statistics, enhanced_hdv_statistics


def extract_prediction_features_batch(vehicle_ids: List[str],
                                     state: Dict,
                                     lane_statistics: Dict) -> Dict[str, Dict]:
    """
    批量提取预测特征（用于模型输入）

    Args:
        vehicle_ids: 车辆ID列表
        state: 车辆状态字典
        lane_statistics: 车道统计信息

    Returns:
        features_dict: {vehicle_id: {'env_features': array, 'ind_features': array, ...}}
    """
    try:
        from harl.envs.a_multi_lane.env_utils.feature_extractor import FeatureExtractor
    except:
        # 如果导入失败，返回空字典
        return {}

    extractor = FeatureExtractor()
    features_dict = {}

    for veh_id in vehicle_ids:
        try:
            # 提取意图预测特征
            env_feat, ind_feat, veh_type = extractor.extract_intention_features(
                veh_id, state, lane_statistics
            )
            features_dict[veh_id] = {
                'env_features': env_feat,
                'ind_features': ind_feat,
                'vehicle_type': veh_type
            }
        except Exception as e:
            continue

    return features_dict


def integrate_predictions_into_observations(obs_dict: Dict,
                                          cav_statistics: Dict,
                                          hdv_statistics: Dict) -> Dict:
    """
    将预测和决策信息整合到观测中

    Args:
        obs_dict: 原始观测字典
        cav_statistics: CAV统计（包含决策）
        hdv_statistics: HDV统计（包含预测）

    Returns:
        enhanced_obs_dict: 增强的观测字典
    """
    enhanced_obs = {}

    for ego_id, obs in obs_dict.items():
        # 复制原始观测
        if isinstance(obs, dict):
            enhanced_obs[ego_id] = obs.copy()
        else:
            enhanced_obs[ego_id] = obs

        # 添加周边车辆预测信息
        if ego_id in cav_statistics:
            ego_info = cav_statistics[ego_id]

            # 获取周边车辆
            surroundings = ego_info.get('surroundings', {})

            # 收集周边车辆的预测意图
            surround_predictions = {}
            for sur_key, sur_content in surroundings.items():
                if isinstance(sur_content, dict):
                    sur_veh_id = sur_content.get('veh_id', None)
                    if sur_veh_id and sur_veh_id in hdv_statistics:
                        hdv_info = hdv_statistics[sur_veh_id]
                        if 'predicted_intention' in hdv_info:
                            surround_predictions[sur_key] = hdv_info['predicted_intention']

            # 添加到观测中
            if isinstance(enhanced_obs[ego_id], dict):
                enhanced_obs[ego_id]['surround_predictions'] = surround_predictions
                # 添加自己的决策信息
                enhanced_obs[ego_id]['own_decision'] = ego_info.get('lateral_decision', 0)
                enhanced_obs[ego_id]['decision_prob'] = ego_info.get('decision_probability', 0.0)

    return enhanced_obs


def update_rewards_with_decision_consistency(reward_dict: Dict,
                                            cav_statistics: Dict,
                                            rl_actions: Dict,
                                            consistency_weight: float = 0.1) -> Dict:
    """
    基于决策一致性更新奖励

    Args:
        reward_dict: 原始奖励字典
        cav_statistics: CAV统计（包含决策）
        rl_actions: RL动作字典 {veh_id: [lateral_action, longitudinal_action]}
        consistency_weight: 一致性奖励权重

    Returns:
        updated_reward_dict: 更新后的奖励字典
    """
    updated_rewards = reward_dict.copy()

    for veh_id in cav_statistics.keys():
        if veh_id not in rl_actions or veh_id not in updated_rewards:
            continue

        # 获取决策信息
        lateral_decision = cav_statistics[veh_id].get('lateral_decision', 0)
        decision_prob = cav_statistics[veh_id].get('decision_probability', 0.0)

        # 从RL动作提取横向决策（假设action[0]是横向动作）
        rl_lateral_action = rl_actions[veh_id][0]

        # 将连续动作映射为决策
        # 假设 [-1.5, -0.5) -> 右换道(2), [-0.5, 0.5] -> 保持(0), (0.5, 1.5] -> 左换道(1)
        if rl_lateral_action < -0.5:
            rl_decision = 2  # 右换道
        elif rl_lateral_action <= 0.5:
            rl_decision = 0  # 保持
        else:
            rl_decision = 1  # 左换道

        # 简化：只区分换道(1)和保持(0)
        rl_decision_binary = 0 if rl_decision == 0 else 1

        # 计算一致性奖励
        if lateral_decision == rl_decision_binary:
            # 决策一致，奖励与概率成正比
            consistency_reward = consistency_weight * decision_prob
        else:
            # 决策不一致，惩罚
            consistency_reward = -consistency_weight * decision_prob

        # 更新总奖励
        updated_rewards[veh_id] += consistency_reward

    return updated_rewards


def create_prediction_decision_info(cav_statistics: Dict,
                                   hdv_statistics: Dict) -> Dict:
    """
    创建预测和决策信息字典（用于记录和分析）

    Args:
        cav_statistics: CAV统计
        hdv_statistics: HDV统计

    Returns:
        info_dict: 信息字典
    """
    info = {
        'cav_decisions': {},
        'hdv_predictions': {},
        'statistics': {
            'num_cavs': len(cav_statistics),
            'num_hdvs': len(hdv_statistics),
            'num_lane_changes': 0,
            'avg_decision_prob': 0.0,
            'avg_prediction_prob': 0.0
        }
    }

    # CAV决策信息
    decision_probs = []
    for veh_id, stats in cav_statistics.items():
        if 'lateral_decision' in stats:
            info['cav_decisions'][veh_id] = {
                'decision': stats['lateral_decision'],
                'probability': stats.get('decision_probability', 0.0)
            }
            decision_probs.append(stats.get('decision_probability', 0.0))
            if stats['lateral_decision'] == 1:
                info['statistics']['num_lane_changes'] += 1

    if decision_probs:
        info['statistics']['avg_decision_prob'] = np.mean(decision_probs)

    # HDV预测信息
    prediction_probs = []
    for veh_id, stats in hdv_statistics.items():
        if 'predicted_intention' in stats:
            info['hdv_predictions'][veh_id] = {
                'intention': stats['predicted_intention'],
                'has_occupancy': stats.get('predicted_occupancy') is not None
            }
            prediction_probs.append(stats['predicted_intention'])

    if prediction_probs:
        info['statistics']['avg_prediction_prob'] = np.mean(prediction_probs)

    return info


if __name__ == "__main__":
    # 测试包装器
    print("=" * 80)
    print("测试预测决策包装器")
    print("=" * 80)

    # 模拟统计数据
    cav_stats = {
        'CAV_0': {
            'speed': 25.0,
            'acceleration': 0.5,
            'surroundings': {
                'front_1': {'veh_id': 'HDV_0', 'long_dist': 50.0}
            }
        }
    }

    hdv_stats = {
        'HDV_0': {
            'speed': 23.0,
            'acceleration': 0.0
        }
    }

    # 模拟增强（无实际模块）
    enhanced_cav, enhanced_hdv = enhance_traffic_analysis_with_predictions(
        cav_stats, hdv_stats, {}, {}, pred_module=None, dec_module=None
    )

    print("\n✓ 包装器创建成功")
    print(f"  CAV统计: {len(enhanced_cav)} 辆")
    print(f"  HDV统计: {len(enhanced_hdv)} 辆")

    # 创建信息字典
    info = create_prediction_decision_info(enhanced_cav, enhanced_hdv)
    print(f"\n信息统计: {info['statistics']}")

    print("\n✓ 预测决策包装器测试完成!")
