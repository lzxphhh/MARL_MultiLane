# lateral_decision.py
"""
横向决策模块实现
Lateral Decision Module Implementation
基于变道必要性预筛选和变道可行性评估，决定车辆的横向机动策略
"""

import numpy as np
from typing import Dict
from harl.envs.a_multi_lane.project_structure import (
    LateralDecision, WeightConfig, SystemParameters
)


class LateralDecisionModule:
    """横向决策模块"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化横向决策模块

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params

    def clip(self, value: float, min_val: float, max_val: float) -> float:
        """限幅函数"""
        return max(min_val, min(value, max_val))

    def sigmoid(self, x: float) -> float:
        """Sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def compute_efficiency_probability_linear(self,
                                              target_lane_speed: float,
                                              ego_lane_speed: float,
                                              speed_threshold: float = 3.0) -> float:
        """
        计算效率性概率 - 线性隶属度函数

        Args:
            target_lane_speed: 目标车道平均速度 [m/s]
            ego_lane_speed: 自身车道平均速度 [m/s]
            speed_threshold: 速度差阈值 [m/s]

        Returns:
            效率性概率 [0, 1]
        """
        speed_diff = target_lane_speed - ego_lane_speed
        return self.clip(speed_diff / speed_threshold, 0.0, 1.0)

    def compute_efficiency_probability_sigmoid(self,
                                               target_lane_speed: float,
                                               ego_lane_speed: float,
                                               speed_threshold: float = 3.0,
                                               gamma: float = 4.0,
                                               speed_bias: float = 1.5) -> float:
        """
        计算效率性概率 - Sigmoid隶属度函数

        Args:
            target_lane_speed: 目标车道平均速度 [m/s]
            ego_lane_speed: 自身车道平均速度 [m/s]
            speed_threshold: 速度差阈值 [m/s]
            gamma: 陡峭程度控制参数
            speed_bias: 偏移量 [m/s]

        Returns:
            效率性概率 [0, 1]
        """
        speed_diff = target_lane_speed - ego_lane_speed
        normalized_diff = (speed_diff - speed_bias) / speed_threshold
        return self.sigmoid(gamma * normalized_diff)

    def compute_equilibrium_probability_linear(self,
                                               ego_lane_density: float,
                                               target_lane_density: float,
                                               density_threshold: float = 10.0) -> float:
        """
        计算交通流均衡性概率 - 线性隶属度函数

        Args:
            ego_lane_density: 自身车道交通密度 [veh/km]
            target_lane_density: 目标车道交通密度 [veh/km]
            density_threshold: 密度差阈值 [veh/km]

        Returns:
            均衡性概率 [0, 1]
        """
        density_diff = ego_lane_density - target_lane_density
        return self.clip(density_diff / density_threshold, 0.0, 1.0)

    def compute_equilibrium_probability_sigmoid(self,
                                                ego_lane_density: float,
                                                target_lane_density: float,
                                                density_threshold: float = 10.0,
                                                gamma: float = 4.0,
                                                density_bias: float = 5.0) -> float:
        """
        计算交通流均衡性概率 - Sigmoid隶属度函数

        Args:
            ego_lane_density: 自身车道交通密度 [veh/km]
            target_lane_density: 目标车道交通密度 [veh/km]
            density_threshold: 密度差阈值 [veh/km]
            gamma: 陡峭程度控制参数
            density_bias: 偏移量 [veh/km]

        Returns:
            均衡性概率 [0, 1]
        """
        density_diff = ego_lane_density - target_lane_density
        normalized_diff = (density_diff - density_bias) / density_threshold
        return self.sigmoid(gamma * normalized_diff)

    def evaluate_lane_change_necessity(self,
                                       traffic_states: Dict[str, float],
                                       weights: WeightConfig,
                                       use_sigmoid: bool = True) -> Dict[str, float]:
        """
        评估变道必要性

        Args:
            traffic_states: 交通状态信息
                - 'v_left': 左侧车道平均速度 [m/s]
                - 'v_ego': 自身车道平均速度 [m/s]
                - 'v_right': 右侧车道平均速度 [m/s]
                - 'density_left': 左侧车道交通密度 [veh/km]
                - 'density_ego': 自身车道交通密度 [veh/km]
                - 'density_right': 右侧车道交通密度 [veh/km]
            weights: 权重配置
            use_sigmoid: 是否使用Sigmoid函数

        Returns:
            各方向变道必要性概率字典
        """
        results = {}

        # 计算左变道必要性
        if traffic_states['v_left'] != None and traffic_states['density_left'] != None:
            if use_sigmoid:
                p_efficiency_left = self.compute_efficiency_probability_sigmoid(
                    traffic_states['v_left'], traffic_states['v_ego']
                )
                p_equilibrium_left = self.compute_equilibrium_probability_sigmoid(
                    traffic_states['density_ego'], traffic_states['density_left']
                )
            else:
                p_efficiency_left = self.compute_efficiency_probability_linear(
                    traffic_states['v_left'], traffic_states['v_ego']
                )
                p_equilibrium_left = self.compute_equilibrium_probability_linear(
                    traffic_states['density_ego'], traffic_states['density_left']
                )
        else:
            p_efficiency_left = 0
            p_equilibrium_left = 0

        results['necessity_left'] = (
                weights.efficiency * p_efficiency_left +
                weights.equilibrium * p_equilibrium_left
        )

        # 计算右变道必要性
        if traffic_states['v_right'] != None and traffic_states['density_right'] != None:
            if use_sigmoid:
                p_efficiency_right = self.compute_efficiency_probability_sigmoid(
                    traffic_states['v_right'], traffic_states['v_ego']
                )
                p_equilibrium_right = self.compute_equilibrium_probability_sigmoid(
                    traffic_states['density_ego'], traffic_states['density_right']
                )
            else:
                p_efficiency_right = self.compute_efficiency_probability_linear(
                    traffic_states['v_right'], traffic_states['v_ego']
                )
                p_equilibrium_right = self.compute_equilibrium_probability_linear(
                    traffic_states['density_ego'], traffic_states['density_right']
                )
        else:
            p_efficiency_right = 0
            p_equilibrium_right = 0

        results['necessity_right'] = (
                weights.efficiency * p_efficiency_right +
                weights.equilibrium * p_equilibrium_right
        )

        # 预筛选条件
        threshold = self.params.necessity_threshold
        results['pre_change_left'] = 1 if results['necessity_left'] >= threshold else 0
        results['pre_change_right'] = 1 if results['necessity_right'] >= threshold else 0

        return results

    def make_final_lateral_decision(self,
                                    necessity_probs: Dict[str, float],
                                    feasibility_probs: Dict[str, float]) -> Dict:
        """
        做出最终横向决策

        Args:
            necessity_probs: 变道必要性概率字典
            feasibility_probs: 变道可行性概率字典
                - 'feasibility_left': 左变道可行性概率
                - 'feasibility_right': 右变道可行性概率

        Returns:
            决策结果字典
        """
        # 计算综合概率
        p_left = necessity_probs['necessity_left'] * feasibility_probs.get('feasibility_left', 0.0)
        p_right = necessity_probs['necessity_right'] * feasibility_probs.get('feasibility_right', 0.0)
        p_keep = 1.0 - max(p_left, p_right)  # 保持车道概率

        # 决策阈值
        decision_threshold = self.params.decision_threshold

        # 决策逻辑
        if max(p_left, p_right) < decision_threshold:
            # 都不满足阈值，保持车道
            final_decision = LateralDecision.KEEP
            decision_confidence = p_keep
        elif p_left > p_right and p_left >= decision_threshold:
            # 左变道
            final_decision = LateralDecision.LEFT
            decision_confidence = p_left
        elif p_right > p_left and p_right >= decision_threshold:
            # 右变道
            final_decision = LateralDecision.RIGHT
            decision_confidence = p_right
        else:
            # 默认保持车道
            final_decision = LateralDecision.KEEP
            decision_confidence = p_keep

        return {
            'decision': final_decision,
            'confidence': decision_confidence,
            'probabilities': {
                'left': p_left,
                'right': p_right,
                'keep': p_keep
            }
        }

    def cav_negotiation_priority(self,
                                 cav_decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        CAV变道决策协商 - 优先策略
        前车优先，后车等待前车变道完成后再评估

        Args:
            cav_decisions: CAV决策字典，键为车辆ID，值为决策结果

        Returns:
            协商后的决策结果
        """
        # 按车辆位置排序（假设x坐标越大车辆越前）
        sorted_cavs = sorted(cav_decisions.items(),
                             key=lambda item: item[1].get('position_x', 0),
                             reverse=True)

        negotiated_decisions = {}
        occupied_directions = set()

        for vehicle_id, decision_info in sorted_cavs:
            decision = decision_info['decision']

            # 如果该方向已被前车占用，后车保持车道
            if decision != LateralDecision.KEEP and decision in occupied_directions:
                negotiated_decisions[vehicle_id] = {
                    **decision_info,
                    'decision': LateralDecision.KEEP,
                    'negotiated': True,
                    'reason': 'Waiting for front vehicle'
                }
            else:
                negotiated_decisions[vehicle_id] = {
                    **decision_info,
                    'negotiated': False
                }
                if decision != LateralDecision.KEEP:
                    occupied_directions.add(decision)

        return negotiated_decisions

    def cav_negotiation_competitive(self,
                                    cav_decisions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        CAV变道决策协商 - 竞争策略
        概率高者优先，低者可选择反向变道

        Args:
            cav_decisions: CAV决策字典，键为车辆ID，值为决策结果

        Returns:
            协商后的决策结果
        """
        negotiated_decisions = {}

        # 按同方向变道分组
        left_candidates = []
        right_candidates = []
        keep_vehicles = []

        for vehicle_id, decision_info in cav_decisions.items():
            if decision_info['decision'] == LateralDecision.LEFT:
                left_candidates.append((vehicle_id, decision_info))
            elif decision_info['decision'] == LateralDecision.RIGHT:
                right_candidates.append((vehicle_id, decision_info))
            else:
                keep_vehicles.append((vehicle_id, decision_info))

        # 处理左变道冲突
        if len(left_candidates) > 1:
            # 按概率排序，最高者获得优先权
            left_candidates.sort(key=lambda x: x[1]['confidence'], reverse=True)
            winner_id, winner_info = left_candidates[0]

            negotiated_decisions[winner_id] = {**winner_info, 'negotiated': False}

            # 其他车辆考虑右变道或保持车道
            for vehicle_id, decision_info in left_candidates[1:]:
                right_prob = decision_info['probabilities'].get('right', 0.0)
                if right_prob >= self.params.decision_threshold:
                    negotiated_decisions[vehicle_id] = {
                        **decision_info,
                        'decision': LateralDecision.RIGHT,
                        'negotiated': True,
                        'reason': 'Alternative direction due to conflict'
                    }
                else:
                    negotiated_decisions[vehicle_id] = {
                        **decision_info,
                        'decision': LateralDecision.KEEP,
                        'negotiated': True,
                        'reason': 'No feasible alternative'
                    }
        else:
            for vehicle_id, decision_info in left_candidates:
                negotiated_decisions[vehicle_id] = {**decision_info, 'negotiated': False}

        # 处理右变道冲突（同理）
        if len(right_candidates) > 1:
            right_candidates.sort(key=lambda x: x[1]['confidence'], reverse=True)
            winner_id, winner_info = right_candidates[0]

            negotiated_decisions[winner_id] = {**winner_info, 'negotiated': False}

            for vehicle_id, decision_info in right_candidates[1:]:
                left_prob = decision_info['probabilities'].get('left', 0.0)
                if left_prob >= self.params.decision_threshold:
                    negotiated_decisions[vehicle_id] = {
                        **decision_info,
                        'decision': LateralDecision.LEFT,
                        'negotiated': True,
                        'reason': 'Alternative direction due to conflict'
                    }
                else:
                    negotiated_decisions[vehicle_id] = {
                        **decision_info,
                        'decision': LateralDecision.KEEP,
                        'negotiated': True,
                        'reason': 'No feasible alternative'
                    }
        else:
            for vehicle_id, decision_info in right_candidates:
                negotiated_decisions[vehicle_id] = {**decision_info, 'negotiated': False}

        # 保持车道的车辆不需要协商
        for vehicle_id, decision_info in keep_vehicles:
            negotiated_decisions[vehicle_id] = {**decision_info, 'negotiated': False}

        return negotiated_decisions
#
#     def make_lateral_decision(self,
#                               traffic_states: Dict[str, float],
#                               feasibility_probs: Dict[str, float],
#                               weights: WeightConfig,
#                               use_sigmoid: bool = True) -> Dict:
#         """
#         横向决策主函数
#
#         Args:
#             traffic_states: 交通状态信息
#             feasibility_probs: 变道可行性概率
#             weights: 权重配置
#             use_sigmoid: 是否使用Sigmoid函数
#
#         Returns:
#             横向决策结果
#         """
#         # 1. 评估变道必要性
#         necessity_results = self.evaluate_lane_change_necessity(
#             traffic_states, weights, use_sigmoid
#         )
#
#         # 2. 做出最终决策
#         decision_result = self.make_final_lateral_decision(
#             necessity_results, feasibility_probs
#         )
#
#         # 3. 整合结果
#         result = {
#             **decision_result,
#             'necessity_probs': necessity_results,
#             'feasibility_probs': feasibility_probs,
#             'pre_decisions': {
#                 'left': necessity_results['pre_decision_left'],
#                 'right': necessity_results['pre_decision_right']
#             }
#         }
#
#         return result
#
#
# # 使用示例
# def example_usage():
#     """横向决策模块使用示例"""
#
#     # 初始化系统参数
#     system_params = SystemParameters()
#
#     # 创建横向决策模块
#     lateral_decision = LateralDecisionModule(system_params)
#
#     # 模拟交通状态信息
#     traffic_states = {
#         'v_left': 15.0,  # 左侧车道平均速度
#         'v_ego': 12.0,  # 自身车道平均速度
#         'v_right': 13.0,  # 右侧车道平均速度
#         'density_left': 45.0,  # 左侧车道交通密度
#         'density_ego': 55.0,  # 自身车道交通密度
#         'density_right': 50.0  # 右侧车道交通密度
#     }
#
#     # 模拟变道可行性概率（通常由冲突评估模块提供）
#     feasibility_probs = {
#         'feasibility_left': 0.8,  # 左变道可行性
#         'feasibility_right': 0.7  # 右变道可行性
#     }
#
#     # 模拟权重配置（通常由场景识别模块提供）
#     weights = WeightConfig(
#         efficiency=0.6, equilibrium=0.4,
#         ws=0.3, we=0.3, wc=0.2, wt=0.1, wcoop=0.1
#     )
#
#     # 执行横向决策
#     result = lateral_decision.make_lateral_decision(
#         traffic_states, feasibility_probs, weights
#     )
#
#     print("横向决策结果:")
#     print(f"最终决策: {result['decision']}")
#     print(f"决策置信度: {result['confidence']:.3f}")
#     print(f"各方向概率: 左={result['probabilities']['left']:.3f}, "
#           f"右={result['probabilities']['right']:.3f}, "
#           f"保持={result['probabilities']['keep']:.3f}")
#     print(f"预决策: 左={result['pre_decisions']['left']}, "
#           f"右={result['pre_decisions']['right']}")
#
#     # CAV协商示例
#     print("\nCAV协商示例:")
#     cav_decisions = {
#         'CAV_1': {
#             'decision': LateralDecision.LEFT,
#             'confidence': 0.8,
#             'probabilities': {'left': 0.8, 'right': 0.3, 'keep': 0.2},
#             'position_x': 100.0
#         },
#         'CAV_2': {
#             'decision': LateralDecision.LEFT,
#             'confidence': 0.6,
#             'probabilities': {'left': 0.6, 'right': 0.7, 'keep': 0.4},
#             'position_x': 95.0
#         }
#     }
#
#     negotiated = lateral_decision.cav_negotiation_competitive(cav_decisions)
#     for vehicle_id, decision_info in negotiated.items():
#         print(f"{vehicle_id}: {decision_info['decision']}, "
#               f"协商={decision_info['negotiated']}")
#
#
# if __name__ == "__main__":
#     example_usage()