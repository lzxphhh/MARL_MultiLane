import itertools
import math
from typing import List, Dict, Tuple, Union
import copy

import numpy as np
import collections.abc
from itertools import chain
import time

LANE_LENGTHS = {
    'E0': 500,
    'E1': 2000,
}

state_divisors = np.array([2000, 3, 20, 3, 360, 3, 20])
# Lane_start = {
#     'E0': (0, 0),
#     'E1': (100, 0),
#     'E2': (100, 100),
#     'E3': (0, 100),
# }
# Lane_end = {
#     'E0': (100, 0),
#     'E1': (100, 100),
#     'E2': (0, 100),
#     'E3': (0, 0),
# }
Lane_start = {
    'E0': (-500, 0),
    'E1': (0, 0),
}
Lane_end = {
    'E0': (0, 0),
    'E1': (2000, 0),
}


def analyze_traffic(state, lane_ids, max_veh_num, parameters, vehicles_hist, hist_length, TTC_assessment):
    stability_Q = np.array([[1, 0], [0, 0.5]])
    comfort_R = 0.5

    def calculate_safety_metrics(long_dist, long_rel_v, long_rel_a, parameters):
        """计算安全指标：TTC, ACT, DRAC"""
        if long_dist <= 0:
            return parameters['min_TTC'], parameters['min_ACT'], parameters['max_DRAC']

        if long_rel_v > 0:
            long_TTC = np.clip(long_dist / long_rel_v, parameters['min_TTC'], parameters['max_TTC'], dtype=np.float64)
            long_DRAC = np.clip(long_rel_v ** 2 / long_dist, parameters['min_DRAC'], parameters['max_DRAC'], dtype=np.float64)
        else:
            long_TTC = parameters['max_TTC']
            long_DRAC = parameters['min_DRAC']

        if long_rel_a == 0:
            long_ACT = long_TTC
        else:
            D = long_rel_v ** 2 + 2 * long_dist * long_rel_a
            if D < 0:
                long_ACT = parameters['max_ACT']
            else:
                t_c_1 = (long_rel_v + np.sqrt(D)) / (0 - long_rel_a)
                t_c_2 = (long_rel_v - np.sqrt(D)) / (0 - long_rel_a)
                if t_c_1 <= 0 and t_c_2 <= 0:
                    long_ACT = parameters['max_ACT']
                elif t_c_1 > 0 and t_c_2 > 0:
                    long_ACT = min(t_c_1, t_c_2)
                else:
                    long_ACT = max(t_c_1, t_c_2)

        return long_TTC, long_ACT, long_DRAC

    def get_surrounding_safety_metrics(surroundings, lane_index, parameters, direction):
        """获取周围车辆的安全指标"""
        leader_key = f'{direction}_leader_1'
        follower_key = f'{direction}_follower_1'

        # 前车指标
        if leader_key in surroundings:
            front_TTC = surroundings[leader_key]['long_TTC']
            front_ACT = surroundings[leader_key]['long_ACT']
        elif (direction == 'left' and lane_index == '2') or (direction == 'right' and lane_index == '0'):
            front_TTC = parameters['min_TTC']
            front_ACT = parameters['min_ACT']
        else:
            front_TTC = parameters['max_TTC']
            front_ACT = parameters['max_ACT']

        # 后车指标
        if follower_key in surroundings:
            back_TTC = surroundings[follower_key]['long_TTC']
            back_ACT = surroundings[follower_key]['long_ACT']
        elif (direction == 'left' and lane_index == '2') or (direction == 'right' and lane_index == '0'):
            back_TTC = parameters['min_TTC']
            back_ACT = parameters['min_ACT']
        else:
            back_TTC = parameters['max_TTC']
            back_ACT = parameters['max_ACT']

        return min(front_TTC, back_TTC), min(front_ACT, back_ACT)

    def process_vehicle_history(vehicle_id, lane_index, position, speed, acceleration, heading, spacing, relative_speed,
                                jerk, safety_time, vehicles_hist, hist_length):
        """处理车辆的历史信息"""
        for i in range(hist_length - 1, 0, -1):
            if vehicles_hist[f'hist_{i}'][vehicle_id] == [0.0] * 10:
                vehicles_hist[f'hist_{i + 1}'][vehicle_id] = [lane_index, position[0], position[1], speed, acceleration,
                                                              heading, spacing, relative_speed, jerk, safety_time]
            else:
                vehicles_hist[f'hist_{i + 1}'][vehicle_id] = vehicles_hist[f'hist_{i}'][vehicle_id].copy()
        vehicles_hist['hist_1'][vehicle_id] = [lane_index, position[0], position[1], speed, acceleration, heading,
                                               spacing, relative_speed, jerk, safety_time]

    def init_lane_statistics():
        """初始化车道统计数据结构"""
        return {
            'vehicle_count': 0, 'lane_num_CAV': 0, 'lanechange_count': 0,
            'positions': [], 'speeds': [], 'accelerations': [], 'waiting_times': [],
            'accumulated_waiting_times': [], 'jerks': [], 'long_TTCs': [], 'long_ACTs': [],
            'time_headways': [], 'relative_speeds': [], 'aggressivenesss': [], 'lanechange_gaps': []
        }

    def init_flow_statistics():
        """初始化流量统计数据结构"""
        return {
            'vehicle_count': 0, 'num_CAV': 0, 'lanechange_count': 0,
            'speeds': [], 'accelerations': [], 'long_TTCs': [], 'long_ACTs': [],
            'relative_speeds': [], 'jerks': [], 'aggressivenesss': [], 'lanechange_gaps': [],
            'lane_densities': [], 'lane_flow_rates': [], 'time_headways': [],
        }

    def init_evaluation():
        """初始化评估数据结构"""
        return {
            'safety_TSIs': [], 'safety_ASIs': [], 'efficiency_ASRs': [], 'efficiency_TFRs': [],
            'stability_SSVDs': [], 'stability_SSSDs': [], 'comfort_AIs': [], 'comfort_JIs': [],
            'control_efficiencies': [], 'comfort_costs': [], 'stability_indices': [], 'comfort_indices': []
        }

    def create_vehicle_stats_dict(basic_info, safety_metrics, behavior_metrics, control_metrics, additional_info=None, hierarchical_info=None):
        """创建车辆统计字典"""
        stats = {**basic_info, **safety_metrics, **behavior_metrics, **control_metrics}
        if additional_info:
            stats.update(additional_info)
        if hierarchical_info:
            stats.update(hierarchical_info)
        return stats

    def calculate_mean_stats(stats_dict, vehicle_count):
        """计算统计指标的均值"""
        if vehicle_count == 0:
            return {}

        results = {}

        # 基础统计
        if stats_dict['speeds']:
            mean_speed = np.mean(stats_dict['speeds'])
            std_speed = np.std(stats_dict['speeds'])
            results.update({
                'mean_speed': mean_speed,
                'std_speed': std_speed,
                'coef_var_speed': std_speed / mean_speed if mean_speed > 0 else 0
            })

        if stats_dict['accelerations']:
            mean_acceleration = np.mean(stats_dict['accelerations'])
            std_acceleration = np.std(stats_dict['accelerations'])
            results.update({
                'mean_acceleration': mean_acceleration,
                'std_acceleration': std_acceleration,
                'coef_var_acceleration': std_acceleration / mean_acceleration if mean_acceleration > 0 else 0
            })

        # 其他统计指标
        stat_keys = ['jerks', 'long_TTCs', 'long_ACTs', 'time_headways', 'relative_speeds', 'aggressivenesss']
        for key in stat_keys:
            if stats_dict[key]:
                mean_key = f"mean_{key[:-1] if key.endswith('s') else key}"
                results[mean_key] = np.mean(stats_dict[key])
                if key == 'aggressivenesss':
                    results['std_aggresiveness'] = np.std(stats_dict[key])

        # 等待时间和换道间隙
        if 'waiting_times' in stats_dict and stats_dict['waiting_times']:
            results['mean_waiting_time'] = np.mean(stats_dict['waiting_times'])
        else:
            results['mean_waiting_time'] = 0
        if 'accumulated_waiting_times' in stats_dict and stats_dict['accumulated_waiting_times']:
            results['mean_accumulated_waiting_time'] = np.mean(stats_dict['accumulated_waiting_times'])
        else:
            results['mean_accumulated_waiting_time'] = 0
        if stats_dict['lanechange_gaps']:
            results['mean_lanechange_gap'] = np.mean(stats_dict['lanechange_gaps'])
        else:
            results['mean_lanechange_gap'] = 0

        return results

    # 初始化统计数据结构
    lane_statistics = {lane_id: init_lane_statistics() for lane_id in lane_ids}
    cav_statistics = {}
    hdv_statistics = {}
    reward_statistics = {}
    flow_statistics = init_flow_statistics()
    evaluation = init_evaluation()

    # 处理每辆车
    for vehicle_id, vehicle in state.items():
        # # 跳过leader车辆
        # if vehicle['vehicle_type'] == 'leader':
        #     continue

        # 基础信息提取和处理
        lane_id = vehicle['lane_id']
        lane_index = str(vehicle['lane_index'])
        road_id = vehicle['road_id']

        # 处理特殊道路ID
        if road_id[:3] in [':J1', ':J2']:
            road_id = 'E1'
            lane_id = f'E1_{int(lane_index)}'

        # 车辆状态信息
        basic_info = {
            'position': vehicle['position'],
            'speed': vehicle['speed'],
            'acceleration': vehicle['acceleration'],
            'heading': vehicle['heading'],
            'road_id': road_id,
            'lane_index': lane_index
        }

        # 前车信息处理
        leader_info = vehicle['leader']
        if not leader_info or leader_info[0] not in state:
            # 如果没有前车，使用默认值
            long_dist = parameters['minGap'] + basic_info['speed'] * parameters['tau']  # 默认跟车距离为理想距离
            time_headway = parameters['max_time_headway']
            long_rel_v = 0
            long_rel_a = 0
            long_TTC = parameters['max_TTC']
            long_ACT = parameters['max_ACT']
            long_DRAC = parameters['min_DRAC']
        else:
            leader_id = leader_info[0]
            leader_state = state[leader_id]
            leader_long_pos = leader_state['position'][0]
            leader_speed = leader_state['speed']
            leader_acceleration = leader_state['acceleration']

            long_dist = leader_long_pos - basic_info['position'][0] - 5
            time_headway = long_dist / basic_info['speed'] if basic_info['speed'] > 0 else parameters[
                'max_time_headway']
            long_rel_v = basic_info['speed'] - leader_speed
            long_rel_a = basic_info['acceleration'] - leader_acceleration

            # 计算安全指标
            long_TTC, long_ACT, long_DRAC = calculate_safety_metrics(long_dist, long_rel_v, long_rel_a, parameters)

        # 安全和行为指标
        safety_metrics = {
            'long_TTC': long_TTC,
            'long_ACT': long_ACT,
            'long_DRAC': long_DRAC,
            'spcing': long_dist,
            'relative_speed': long_rel_v,
            'time_headway': time_headway
        }

        # 控制指标
        delta_v = long_rel_v
        desired_s = parameters['minGap'] + basic_info['speed'] * parameters['tau']
        delta_s = long_dist - desired_s
        stability_matric = np.array([delta_s, delta_v])
        control_efficiency = stability_matric @ stability_Q @ stability_matric.T
        comfort_cost = comfort_R * (basic_info['acceleration'] ** 2)
        stability_index = np.exp(-control_efficiency)
        comfort_index = np.exp(-comfort_cost)

        control_metrics = {
            'delta_v': delta_v,
            'delta_s': delta_s,
            'control_efficiency': control_efficiency,
            'comfort_cost': comfort_cost
        }

        # 历史信息和行为分析
        last_acceleration = vehicles_hist['hist_1'][vehicle_id][4]
        jerk = (basic_info['acceleration'] - last_acceleration) / 0.1
        # 更新历史记录
        process_vehicle_history(vehicle_id, int(lane_index), basic_info['position'], basic_info['speed'],
                                basic_info['acceleration'], basic_info['heading'], long_dist, long_rel_v,
                                jerk, long_ACT, vehicles_hist, hist_length)

        last_lane_index = vehicles_hist['hist_2'][vehicle_id][0]
        is_lanechange = 0 if last_lane_index == int(lane_index) else 1
        lanechange_gap = 0 if not is_lanechange else long_dist

        # 激进程度计算
        acc_aggressiveness = min(abs(basic_info['acceleration']) / 3.0, 1.0)
        following_aggressiveness = max(0, 1.0 - time_headway / 3.0) if time_headway > 0 else 0.5
        aggressiveness = (0.4 * acc_aggressiveness + 0.4 * following_aggressiveness + 0.2 * is_lanechange)

        behavior_metrics = {
            'jerk': jerk,
            'is_lanechange': is_lanechange,
            'lanechange_gap': lanechange_gap,
            'aggressiveness': aggressiveness
        }

        # 评估指标
        veh_TSI = 1 - max(0, 1 - long_TTC / parameters['safe_warn_threshold']['TTC']) ** 2
        veh_ASI = 1 - max(0, 1 - long_ACT / parameters['safe_warn_threshold']['ACT']) ** 2
        veh_ASR = basic_info['speed'] / parameters['max_v']

        # 处理除零错误
        leader_speed = state[leader_info[0]]['speed'] if leader_info and leader_info[0] in state else 1.0
        veh_SSVD = 1 - abs(long_rel_v) / leader_speed if leader_speed != 0 else 0
        veh_SSSD = 1 - abs(delta_s) / desired_s if desired_s != 0 else 0
        veh_AI = 1 - min(1, abs(basic_info['acceleration']) / parameters['max_a'])
        veh_JI = 1 - min(1, abs(jerk) / parameters['max_jerk'])

        # 附加信息
        additional_info = {
            'distance': vehicle['distance'],
            'waiting_time': vehicle['waiting_time'],
            'accumulated_waiting_time': vehicle['accumulated_waiting_time'],
            'fuel_consumption': vehicle['fuel_consumption'],
            'co2_emission': vehicle['co2_emission'],
            'safety_TSI': veh_TSI,
            'safety_ASI': veh_ASI,
            'efficiency_ASR': veh_ASR,
            'stability_SSVD': veh_SSVD,
            'stability_SSSD': veh_SSSD,
            'comfort_AI': veh_AI,
            'comfort_JI': veh_JI,
            'control_efficiency': control_efficiency,
            'comfort_cost': comfort_cost,
        }

        tau = parameters.get('tau', 1.0)
        a_ideal = (-long_rel_v) / tau

        # 分层奖励信息
        hierarchical_info = {
            'a_ideal': a_ideal,
            's_baseline': desired_s,
            'v_baseline': leader_speed,
        }

        # 处理ego车和HDV车
        is_ego = vehicle['vehicle_type'] == 'ego'

        if is_ego:
            surroundings = vehicle['surroundings']
            surroundings_expand = vehicle['surroundings_expand']

            # 获取左右车道安全指标
            left_TTC, left_ACT = get_surrounding_safety_metrics(surroundings, lane_index, parameters, 'left')
            right_TTC, right_ACT = get_surrounding_safety_metrics(surroundings, lane_index, parameters, 'right')

            ego_additional = {
                'surroundings': surroundings,
                'surroundings_expand': surroundings_expand,
                'left_TTC': left_TTC,
                'left_ACT': left_ACT,
                'right_TTC': right_TTC,
                'right_ACT': right_ACT
            }

            TTC_assessment[vehicle_id]['left'] = left_TTC
            TTC_assessment[vehicle_id]['right'] = right_TTC
            TTC_assessment[vehicle_id]['current'] = long_TTC

            cav_statistics[vehicle_id] = create_vehicle_stats_dict(
                basic_info, safety_metrics, behavior_metrics, control_metrics, ego_additional
            )

            # 更新评估指标
            evaluation['safety_TSIs'].append(veh_TSI)
            evaluation['safety_ASIs'].append(veh_ASI)
        else:
            hdv_statistics[vehicle_id] = create_vehicle_stats_dict(
                basic_info, safety_metrics, behavior_metrics, control_metrics
            )

        # 更新通用评估指标
        evaluation['efficiency_ASRs'].append(veh_ASR)
        evaluation['stability_SSVDs'].append(veh_SSVD)
        evaluation['stability_SSSDs'].append(veh_SSSD)
        evaluation['comfort_AIs'].append(veh_AI)
        evaluation['comfort_JIs'].append(veh_JI)
        evaluation['control_efficiencies'].append(control_efficiency)
        evaluation['comfort_costs'].append(comfort_cost)
        evaluation['stability_indices'].append(stability_index)
        evaluation['comfort_indices'].append(comfort_index)

        # 奖励统计
        reward_statistics[vehicle_id] = create_vehicle_stats_dict(
            basic_info, safety_metrics, behavior_metrics, control_metrics, additional_info, hierarchical_info
        )

        # 更新车道统计
        if lane_index in lane_ids:
            lane_stats = lane_statistics[lane_index]
            lane_stats['lane_length'] = LANE_LENGTHS[road_id]
            lane_stats['vehicle_count'] += 1
            if is_ego:
                lane_stats['lane_num_CAV'] += 1
            lane_stats['lanechange_count'] += is_lanechange

            # 添加各项指标
            lane_stats['positions'].append(basic_info['position'][0])
            lane_stats['speeds'].append(basic_info['speed'])
            lane_stats['accelerations'].append(basic_info['acceleration'])
            lane_stats['waiting_times'].append(additional_info['waiting_time'])
            lane_stats['accumulated_waiting_times'].append(additional_info['accumulated_waiting_time'])
            lane_stats['jerks'].append(jerk)
            lane_stats['long_TTCs'].append(long_TTC)
            lane_stats['long_ACTs'].append(long_ACT)
            lane_stats['time_headways'].append(time_headway)
            lane_stats['relative_speeds'].append(long_rel_v)
            lane_stats['aggressivenesss'].append(aggressiveness)

            if is_lanechange:
                lane_stats['lanechange_gaps'].append(lanechange_gap)

        # 更新流量统计
        flow_statistics['vehicle_count'] += 1
        if is_ego:
            flow_statistics['num_CAV'] += 1
        flow_statistics['lanechange_count'] += is_lanechange

        flow_statistics['speeds'].append(basic_info['speed'])
        flow_statistics['accelerations'].append(basic_info['acceleration'])
        flow_statistics['long_TTCs'].append(long_TTC)
        flow_statistics['long_ACTs'].append(long_ACT)
        flow_statistics['relative_speeds'].append(long_rel_v)
        flow_statistics['time_headways'].append(time_headway)
        flow_statistics['jerks'].append(jerk)
        flow_statistics['aggressivenesss'].append(aggressiveness)

        if is_lanechange:
            flow_statistics['lanechange_gaps'].append(lanechange_gap)

    # 计算车道级别统计
    for lane_index in lane_ids:
        lane_stats = lane_statistics[lane_index]
        vehicle_count = lane_stats['vehicle_count']

        if vehicle_count > 0:
            # 计算基础统计
            mean_stats = calculate_mean_stats(lane_stats, vehicle_count)
            lane_stats.update(mean_stats)

            # 计算车道特有指标
            lane_num_CAV = lane_stats['lane_num_CAV']
            lane_stats['lane_CAV_penetration'] = lane_num_CAV / vehicle_count

            # 位置相关计算
            head_pos = max(lane_stats['positions'])
            end_pos = min(lane_stats['positions'])
            platoon_length = head_pos - end_pos

            if platoon_length > 0:
                lane_density = vehicle_count / platoon_length * 1000
                lane_flow_rate = vehicle_count * lane_stats['mean_speed'] / platoon_length
            else:
                lane_density = 0
                lane_flow_rate = 0

            lane_stats.update({
                'platoon_length': platoon_length,
                'lane_density': lane_density,
                'lane_flow_rate': lane_flow_rate
            })

            # 更新流量统计
            flow_statistics['lane_densities'].append(lane_density)
            flow_statistics['lane_flow_rates'].append(lane_flow_rate)
            evaluation['efficiency_TFRs'].append(lane_flow_rate)

    # 计算流量级别统计
    vehicle_count = flow_statistics['vehicle_count']
    if vehicle_count > 0:
        # 计算基础统计
        mean_stats = calculate_mean_stats(flow_statistics, vehicle_count)
        flow_statistics.update(mean_stats)

        # 计算CAV渗透率
        flow_statistics['CAV_penetration'] = flow_statistics['num_CAV'] / vehicle_count

        # 计算平均密度和流量
        if flow_statistics['lane_densities']:
            flow_statistics['mean_density'] = np.mean(flow_statistics['lane_densities'])
        if flow_statistics['lane_flow_rates']:
            flow_statistics['mean_flow_rate'] = np.mean(flow_statistics['lane_flow_rates'])

        # 计算最终评估指标
        evaluation.update({
            'safety_TSI': np.mean(evaluation['safety_TSIs']) if evaluation['safety_TSIs'] else 0,
            'safety_ASI': np.mean(evaluation['safety_ASIs']) if evaluation['safety_ASIs'] else 0,
            'efficiency_ASR': np.mean(evaluation['efficiency_ASRs']),
            'efficiency_TFR': np.mean(evaluation['efficiency_TFRs']) if evaluation['efficiency_TFRs'] else 0,
            'stability_SSVD': np.mean(evaluation['stability_SSVDs']),
            'stability_SSSD': np.mean(evaluation['stability_SSSDs']),
            'comfort_AI': np.mean(evaluation['comfort_AIs']),
            'comfort_JI': np.mean(evaluation['comfort_JIs']),
            'control_efficiency': np.mean(evaluation['control_efficiencies']),
            'comfort_cost': np.mean(evaluation['comfort_costs']),
            'stability_index': np.mean(evaluation['stability_indices']),
            'comfort_index': np.mean(evaluation['comfort_indices'])
        })

    return cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, TTC_assessment

def check_collisions_based_pos(vehicles, gap_threshold: float):
    """输出距离过小的车辆的 ID, 直接根据 pos 来进行计算是否碰撞 (比较简单)

    Args:
        vehicles: 包含车辆部分的位置信息
        gap_threshold (float): 最小的距离限制
    """
    collisions = []
    collisions_info = []

    _distance = {}  # 记录每辆车之间的距离
    for (id1, v1), (id2, v2) in itertools.combinations(vehicles.items(), 2):
        dist = math.sqrt(
            (v1['position'][0] - v2['position'][0]) ** 2 \
            + (v1['position'][1] - v2['position'][1]) ** 2
        )
        _distance[f'{id1}-{id2}'] = dist
        if dist < gap_threshold:
            collisions.append((id1, id2))
            collisions_info.append({'collision': True,
                                    'CAV_key': id1,
                                    'surround_key': id2,
                                    'distance': dist,
                                    })

    return collisions, collisions_info

def check_collisions(vehicles, ego_ids, gap_threshold: float, gap_warn_collision: float):
    ego_collision = []
    ego_warn = []

    info = []

    # 把ego的state专门拿出来
    def filter_vehicles(vehicles, ego_ids):
        # Using dictionary comprehension to create a new dictionary
        # by iterating over all key-value pairs in the original dictionary
        # and including only those whose keys are in ego_ids
        filtered_vehicles = {key: value for key, value in vehicles.items() if key in ego_ids}
        return filtered_vehicles

    filtered_ego_vehicles = filter_vehicles(vehicles, ego_ids)

    for ego_key, ego_value in filtered_ego_vehicles.items():
        for surround_direction, content in filtered_ego_vehicles[ego_key]['surroundings'].items():
            c_info = None
            w_info = None

            # print(ego_key, 'is surrounded by:', content[0], 'with direction', surround_direction,
            # 'at distance', content[1])
            # TODO: 同一个车道和不同车道的车辆的warn gap应该是不一样！！！！11
            distance = math.sqrt(content['long_dist'] ** 2 + content['lat_dist'] ** 2)
            # print('distance:', distance)
            if distance < gap_threshold:
                # print(ego_key, 'collision with', content[0])
                ego_collision.append((ego_key, content['veh_id']))
                c_info = {'collision': True,
                          'CAV_key': ego_key,
                          'surround_key': content['veh_id'],
                          'distance': distance,
                          'relative_speed': content['long_rel_v'],
                          }

            elif gap_threshold <= distance < gap_warn_collision:
                ego_warn.append((ego_key, content['veh_id']))
                w_info = {'warn': True,
                          'CAV_key': ego_key,
                          'surround_key': content['veh_id'],
                          'distance': distance,
                          'relative_speed': content['long_rel_v'],
                          }
            if c_info:
                info.append(c_info)
            elif w_info:
                info.append(w_info)

    return ego_collision, ego_warn, info

def check_prefix(a: str, B: List[str]) -> bool:
    """检查 B 中元素是否有以 a 开头的

    Args:
        a (str): lane_id
        B (List[str]): bottle_neck ids

    Returns:
        bool: 返回 lane_id 是否是 bottleneck
    """
    return any(a.startswith(prefix) for prefix in B)

def calculate_congestion(vehicles: int, length: float, num_lane: int, ratio: float = 1) -> float:
    """计算 bottle neck 的占有率, 我们假设一辆车算上车间距是 10m, 那么一段路的。占有率是
        占有率 = 车辆数/(车道长度*车道数/10)
    于是可以根据占有率计算拥堵系数为:
        拥堵程度 = min(占有率, 1)

    Args:
        vehicles (int): 在 bottle neck 处车辆的数量
        length (float): bottle neck 的长度, 单位是 m
        num_lane (int): bottle neck 的车道数

    Returns:
        float: 拥堵系数 in (0,1)
    """
    capacity_used = ratio * vehicles / (length * num_lane / 10)  # 占有率
    congestion_level = min(capacity_used, 1)  # Ensuring congestion level does not exceed 100%
    return congestion_level

def calculate_speed(congestion_level: float, speed: int) -> float:
    """根据拥堵程度来计算车辆的速度

    Args:
        congestion_level (float): 拥堵的程度, 通过 calculate_congestion 计算得到
        speed (int): 车辆当前的速度

    Returns:
        float: 车辆新的速度
    """
    if congestion_level > 0.2:
        speed = speed * (1 - congestion_level)
        speed = max(speed, 1)
    else:
        speed = -1  # 不控制速度
    return speed

def one_hot_encode(value, unique_values):
    """Create an array with zeros and set the corresponding index to 1
    """
    one_hot = np.zeros(len(unique_values))
    index = unique_values.index(value)
    one_hot[index] = 1
    return one_hot.tolist()

def euclidean_distance(point1, point2):
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point1 - point2)
    return distance

def compute_centralized_vehicle_features(lane_statistics, feature_vectors, bottle_neck_positions):
    shared_features = {}

    # ############################## 所有车的速度 位置 转向信息 ##############################
    all_vehicle = []
    for _, ego_feature in feature_vectors.items():
        all_vehicle += ego_feature[:4]

    # ############################## lane_statistics 的信息 ##############################
    # Initialize a list to hold all lane statistics
    all_lane_stats = []

    # Iterate over all possible lanes to get their statistics
    for _, lane_info in lane_statistics.items():
        # - vehicle_count: 当前车道的车辆数 1
        # - lane_density: 当前车道的车辆密度 1
        # - lane_length: 这个车道的长度 1
        # - speeds: 在这个车道上车的速度 1 (mean)
        # - waiting_times: 一旦车辆开始行驶，等待时间清零 1  (mean)
        # - accumulated_waiting_times: 车的累积等待时间 1 (mean, max)

        all_lane_stats += lane_info[:4] + lane_info[6:7] + lane_info[9:10]

    # ############################## bottleneck 的信息 ##############################
    # 车辆距离bottle_neck
    bottle_neck_position_x = bottle_neck_positions[0] / 700
    bottle_neck_position_y = bottle_neck_positions[1]

    for ego_id in feature_vectors.keys():
        shared_features[ego_id] = [bottle_neck_position_x, bottle_neck_position_y] + all_vehicle + all_lane_stats

    # assert all(len(shared_feature) == 130 for shared_feature in shared_features.values())

    return shared_features

def compute_centralized_vehicle_features_hierarchical_version(
        obs_size, shared_obs_size, lane_statistics,
        feature_vectors_current, feature_vectors_current_flatten,
        feature_vectors, feature_vectors_flatten, ego_ids):
    shared_features = {}
    actor_features = {}

    for ego_id in feature_vectors.keys():
        actor_features[ego_id] = feature_vectors[ego_id].copy() # shared_features--actor / critic , can there output different obs to actor and critic?
    for ego_id in feature_vectors_current.keys():
        shared_features[ego_id] = feature_vectors_current[ego_id].copy()
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    shared_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               shared_features.items()}
    actor_features_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                               actor_features.items()}
    if len(shared_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in shared_features_flatten:
                shared_features_flatten[ego_id] = np.zeros(shared_obs_size)
    if len(actor_features_flatten) != len(ego_ids):
        for ego_id in feature_vectors_flatten.keys():
            if ego_id not in actor_features_flatten:
                actor_features_flatten[ego_id] = np.zeros(obs_size)
    if len(shared_features_flatten) != len(ego_ids):
        print("Error: len(shared_features_flatten) != len(ego_ids)")
    if len(feature_vectors_flatten) != len(ego_ids):
        print("Error: len(feature_vectors_flatten) != len(ego_ids)")
    return actor_features, actor_features_flatten, shared_features, shared_features_flatten
