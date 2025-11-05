import math
from typing import List, Dict, Tuple, Union
import copy

import numpy as np
from tensorflow.python.training import evaluation


def feature_selection_simple_version(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[float]],
        flow_statistics: Dict[str, List[float]],
        reward_statistics: Dict[str, List[float]],
        evaluation: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        node_positions: Dict[str, List[int]],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    Parameters = self.parameters.copy()
    observation_vectors = {
        vehicle_id: {
            'road_info': [],
            'ego_info': [],
            'surround_info': [],
            'local_evaluation_info': [],
        } for vehicle_id in ego_ids
    }
    observation_vectors_flatten = {}
    shared_state_vectors = {
        vehicle_id: {
            'road_structure': [],
            'platoon_info': [],
            'flow_info': [],
            'global_evaluation_info': [],
        } for vehicle_id in ego_ids
    }
    shared_state_vectors_flatten = {}

    # A function to flatten a dictionary structure into 1D array
    def flatten_to_1d(data_dict):
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        size_obs = np.size(np.array(flat_list))
        return np.array(flat_list)

    def flatten_and_pad(stats, pad_length):
        """展平嵌套列表并填充到指定长度"""
        flat = [item for sublist in stats for item in sublist]
        if len(flat) < pad_length:
            flat += [0] * (pad_length - len(flat))
        else:
            flat = flat[:pad_length]
        return flat

    max_x = node_positions['J2'][0]

    for vehicle_id in ego_ids:
        norm_ego_position = ego_statistics[vehicle_id]['position'][0] / max_x
        distance_target = 1 - norm_ego_position
        ego_speed = ego_statistics[vehicle_id]['speed']
        norm_ego_speed = ego_speed / Parameters['max_v']
        ego_acc = ego_statistics[vehicle_id]['acceleration']
        norm_ego_acc = ego_acc / Parameters['max_a']
        ego_heading = ego_statistics[vehicle_id]['heading']
        norm_ego_heading = ego_heading / Parameters['max_heading']
        if 'front_1' in ego_statistics[vehicle_id]['surroundings'].keys():
            desire_speed = ego_statistics[vehicle_id]['surroundings']['front_1']['veh_vel']
            ego_spacing = ego_statistics[vehicle_id]['surroundings']['front_1']['long_dist']
            ego_safetime = ego_statistics[vehicle_id]['surroundings']['front_1']['long_ACT'] if self.use_V2V_info \
                else ego_statistics[vehicle_id]['surroundings']['front_1']['long_TTC']
            relative_speed = ego_statistics[vehicle_id]['surroundings']['front_1']['long_rel_v']
        else:
            desire_speed = Parameters['max_v']
            ego_spacing = Parameters['minGap'] + ego_speed * Parameters['tau']
            ego_safetime = Parameters['max_TTC']
            relative_speed = ego_speed - desire_speed

        desire_spacing = Parameters['minGap'] + ego_speed * Parameters['tau']
        norm_spacing = ego_spacing / desire_spacing
        norm_speed_deviation = relative_speed / Parameters[
            'max_v']
        norm_spacing_deviation = (ego_spacing - desire_spacing) / desire_spacing
        norm_safetime = ego_safetime / Parameters['max_TTC']

        ego_safety = reward_statistics[vehicle_id]['safety_ASI'] if self.use_V2V_info else Parameters['safety_TSI']
        ego_efficiency = reward_statistics[vehicle_id]['efficiency_ASR']
        ego_stability = 0.4 * reward_statistics[vehicle_id]['stability_SSVD'] + 0.6 * reward_statistics[vehicle_id]['stability_SSSD']
        ego_comfort = reward_statistics[vehicle_id]['comfort_JI'] if self.use_V2V_info else reward_statistics[vehicle_id]['comfort_AI']
        ego_control_efficiency = reward_statistics[vehicle_id]['control_efficiency']
        ego_comfort_cost = reward_statistics[vehicle_id]['comfort_cost']

        long_safetime = ego_safetime
        left_safetime = ego_statistics[vehicle_id]['left_ACT'] if self.use_V2V_info else ego_statistics[vehicle_id]['left_TTC']
        right_safetime = ego_statistics[vehicle_id]['right_ACT'] if self.use_V2V_info else ego_statistics[vehicle_id]['right_TTC']

        surround_infos = []
        speeds = [ego_speed]
        accelerations = [ego_acc]
        safety_indices = [ego_safety]
        efficiency_indices = [ego_efficiency]
        stability_indices = [ego_stability]
        comfort_indices = [ego_comfort]
        control_efficiency_indices = [ego_control_efficiency]
        comfort_cost_indices = [ego_comfort_cost]
        if self.use_V2V_info:
            for sur_key, surrounding in ego_statistics[vehicle_id]['surroundings_expand'].items():
                sur_id = surrounding['veh_id']
                if sur_key in ['back_1', 'back_2', 'back_3']:
                    speeds.append(reward_statistics[sur_id]['speed'])
                    accelerations.append(abs(reward_statistics[sur_id]['acceleration']))
                sur_type = surrounding['veh_type']
                sur_pos_mark = surrounding['sur_type']
                norm_relat_pos = surrounding['long_dist'] / max_x
                norm_relat_speed = surrounding['long_rel_v'] / Parameters['max_v']
                norm_relat_acc = surrounding['long_rel_a'] / Parameters['max_a']
                norm_relat_heading = surrounding['rel_heading'] / Parameters['max_heading']
                norm_safetime = surrounding['long_ACT'] / Parameters['max_ACT']
                sur_info = [sur_pos_mark, sur_type, norm_relat_pos, norm_relat_speed, norm_relat_acc, norm_relat_heading, norm_safetime]
                surround_infos.append(sur_info)

                sur_safety = reward_statistics[sur_id]['safety_ASI'] if self.use_V2V_info else reward_statistics[sur_id]['safety_TSI']
                sur_efficiency = reward_statistics[sur_id]['efficiency_ASR']
                sur_stability = 0.4 * reward_statistics[sur_id]['stability_SSVD'] + 0.6 * reward_statistics[vehicle_id]['stability_SSSD']
                sur_comfort = reward_statistics[sur_id]['comfort_JI'] if self.use_V2V_info else reward_statistics[vehicle_id]['comfort_AI']
                sur_control_efficiency = reward_statistics[sur_id]['control_efficiency']
                sur_comfort_cost = reward_statistics[sur_id]['comfort_cost']
                safety_indices.append(sur_safety)
                efficiency_indices.append(sur_efficiency)
                stability_indices.append(sur_stability)
                comfort_indices.append(sur_comfort)
                control_efficiency_indices.append(sur_control_efficiency)
                comfort_cost_indices.append(sur_comfort_cost)

        else:
            for sur_key, surrounding in ego_statistics[vehicle_id]['surroundings'].items():
                sur_id = surrounding['veh_id']
                if sur_key == 'back_1':
                    speeds.append(reward_statistics[sur_id]['speed'])
                    accelerations.append(abs(reward_statistics[sur_id]['acceleration']))
                sur_type = 1 if surrounding['veh_type'] == 'ego' else 0
                sur_pos_mark = surrounding['sur_type']
                norm_relat_pos = surrounding['long_dist'] / max_x
                norm_relat_speed = surrounding['long_rel_v'] / Parameters['max_v']
                norm_relat_acc = 0
                norm_relat_heading = 0
                norm_safetime = surrounding['long_TTC'] / Parameters['max_TTC']
                sur_info = [sur_pos_mark, sur_type, norm_relat_pos, norm_relat_speed, norm_relat_acc, norm_relat_heading, norm_safetime]
                surround_infos.append(sur_info)

                sur_safety = reward_statistics[sur_id]['safety_ASI'] if self.use_V2V_info else reward_statistics[sur_id]['safety_TSI']
                sur_efficiency = reward_statistics[sur_id]['efficiency_ASR']
                sur_stability = 0.4 * reward_statistics[sur_id]['stability_SSVD'] + 0.6 * reward_statistics[vehicle_id]['stability_SSSD']
                sur_comfort = reward_statistics[sur_id]['comfort_JI'] if self.use_V2V_info else reward_statistics[vehicle_id]['comfort_AI']
                sur_control_efficiency = reward_statistics[sur_id]['control_efficiency']
                sur_comfort_cost = reward_statistics[sur_id]['comfort_cost']
                safety_indices.append(sur_safety)
                efficiency_indices.append(sur_efficiency)
                stability_indices.append(sur_stability)
                comfort_indices.append(sur_comfort)
                control_efficiency_indices.append(sur_control_efficiency)
                comfort_cost_indices.append(sur_comfort_cost)

        surround_infos_pad = flatten_and_pad(surround_infos, 14 * 7)

        mean_safety = np.mean(safety_indices)
        mean_efficiency = np.mean(efficiency_indices)
        mean_stability = np.mean(stability_indices)
        mean_comfort = np.mean(comfort_indices)
        mean_control_efficiency = np.mean(control_efficiency_indices)
        mean_comfort_cost = np.mean(comfort_cost_indices)

        observation_vectors[vehicle_id]['road_info'] = [1, distance_target]
        observation_vectors[vehicle_id]['ego_info'] = [norm_ego_position, norm_ego_speed, norm_ego_acc, norm_ego_heading,
                                                       norm_speed_deviation, norm_spacing, norm_spacing_deviation, norm_safetime]  # TODO: add last_action IDM_action
        observation_vectors[vehicle_id]['surround_info'] = surround_infos_pad
        observation_vectors[vehicle_id]['local_evaluation_info'] = [mean_safety, mean_efficiency, mean_stability, mean_comfort, mean_control_efficiency, mean_comfort_cost]
        observation_vectors[vehicle_id]['multilane_info'] = [left_safetime, long_safetime, right_safetime]

    flow = []
    for i_veh in range(self.num_CAVs + self.num_HDVs):
        i_order = int(i_veh / 5) * 10 + i_veh % 5
        ego_id = self.vehicle_names[i_order]
        veh_type = 1 if ego_id in ego_ids else 0
        norm_pos = reward_statistics[ego_id]['position'][0] / max_x
        norm_speed = reward_statistics[ego_id]['speed'] / Parameters['max_v']
        norm_acc = reward_statistics[ego_id]['acceleration'] / Parameters['max_a']
        norm_safe_time = reward_statistics[ego_id]['long_ACT'] / Parameters['max_ACT'] if reward_statistics[ego_id]['long_ACT'] != None else 1
        norm_DRAC = reward_statistics[ego_id]['long_DRAC'] / Parameters['max_DRAC'] if reward_statistics[ego_id]['long_DRAC'] != None else 0
        norm_speed_deviation = reward_statistics[ego_id]['delta_v'] / Parameters['max_v'] if reward_statistics[ego_id]['delta_v'] != None else 0
        norm_spacing_deviation = reward_statistics[ego_id]['delta_s'] / max_x if reward_statistics[ego_id]['delta_s'] != None else 0
        veh_info = [veh_type, norm_pos, norm_speed, norm_acc, norm_safe_time, norm_DRAC, norm_speed_deviation,
                    norm_spacing_deviation]
        flow.append(veh_info)
    flow_pad = flatten_and_pad(flow, (self.num_CAVs + self.num_HDVs) * 8)

    flow_CAV_penetration = flow_statistics['CAV_penetration']
    flow_mean_speed = flow_statistics['mean_speed']
    flow_mean_acc = flow_statistics['mean_acceleration']
    flow_mean_rel_v = flow_statistics['mean_relative_speed']
    flow_mean_jerk = flow_statistics['mean_jerk']
    flow_lanechange =flow_statistics['lanechange_count']
    flow_mean_lanechange_gap = flow_statistics['mean_lanechange_gap']
    flow_mean_TTC = flow_statistics['mean_long_TTC']
    flow_mean_aggressiveness = flow_statistics['mean_aggressiveness']
    flow_mean_flow_rate = flow_statistics['mean_flow_rate']
    flow_mean_density = flow_statistics['mean_density']


    global_safety = evaluation['safety_ASI']
    global_efficiency = evaluation['efficiency_ASR']
    global_flowrate = evaluation['efficiency_TFR']
    global_stability = 0.6 * evaluation['stability_SSVD'] + 0.4 * evaluation['stability_SSSD']
    global_comfort_AI = evaluation['comfort_AI']
    global_comfort_JI = evaluation['comfort_JI']
    global_control_efficiency = evaluation['control_efficiency']
    global_comfort_cost = evaluation['comfort_cost']

    state = {}
    state['road_structure'] = [1]
    state['flow_info'] = flow_pad
    state['flow_infos'] = [flow_CAV_penetration, flow_mean_speed, flow_mean_acc, flow_mean_rel_v, flow_mean_jerk,
                           flow_lanechange, flow_mean_lanechange_gap, flow_mean_TTC, flow_mean_aggressiveness,
                           flow_mean_flow_rate, flow_mean_density]
    state['global_evaluation_info'] = [global_safety, global_efficiency, global_flowrate, global_stability,
                                       global_comfort_AI, global_comfort_JI, global_control_efficiency, global_comfort_cost]

    for vehicle_id in ego_ids:
        shared_state_vectors[vehicle_id] = state

    observation_vectors_flatten = {ego_id: flatten_to_1d(feature_vector) for ego_id, feature_vector in
                                   observation_vectors.items()}

    shared_state_vectors_flatten = {ego_id: flatten_to_1d(shared_feature_vector) for ego_id, shared_feature_vector in
                                    shared_state_vectors.items()}

    return shared_state_vectors, shared_state_vectors_flatten, observation_vectors, observation_vectors_flatten


def feature_selection_improved_version(
        self,
        hdv_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        ego_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        lane_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        flow_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        reward_statistics: Dict[str, List[float]],
        evaluation: Dict[str, List[float]],
        unique_edges: List[str],
        edge_lane_num: Dict[str, int],
        node_positions: Dict[str, List[int]],
        ego_ids: List[str]
) -> Dict[str, List[float]]:
    """
    改进版特征选择函数，提取多车道场景下的观测向量和状态向量

    Returns:
        tuple: (improved_state_vectors, improved_state_vectors_flatten,
                improved_observation_vectors, improved_observation_vectors_flatten)
    """

    # ====================== 初始化参数和数据结构 ======================
    Parameters = self.parameters.copy()

    # 初始化观测向量结构
    observation_vectors = {
        vehicle_id: {
            'road_info': [],
            'ego_info': [],
            'surround_info': [],
            'local_evaluation_info': [],
        } for vehicle_id in ego_ids
    }

    improved_observation_vectors = {
        vehicle_id: {
            'road_info': [],
            'ego_cur_info': [],
            'ego_hist_info': [],
            # 'surround_cur_info': [],
            # 'surround_hist_info': [],
            'local_evaluation_info': [],
        } for vehicle_id in ego_ids
    }

    # 初始化状态向量结构
    observation_vectors_flatten = {}
    shared_state_vectors = {
        vehicle_id: {
            'road_structure': [],
            'flow_info': [],
            'flow_infos': [],
            'global_evaluation_info': [],
        } for vehicle_id in ego_ids
    }

    improved_state_vectors = {
        vehicle_id: {
            'road_structure': [],
            'flow_info': [],
            # 'flow_hist_info': [],
            'flow_infos': [],
            'global_evaluation_info': [],
        } for vehicle_id in ego_ids
    }
    shared_state_vectors_flatten = {}

    # ====================== 工具函数定义 ======================
    def flatten_to_1d(data_dict):
        """将字典结构展平为1D数组"""
        flat_list = []
        for key, item in data_dict.items():
            if isinstance(item, list):
                flat_list.extend(item)
            elif isinstance(item, np.ndarray):
                flat_list.extend(item.flatten())
        return np.array(flat_list)

    def flatten_and_pad(stats, pad_length):
        """展平嵌套列表并填充到指定长度"""
        flat = [item for sublist in stats for item in sublist]
        if len(flat) < pad_length:
            flat += [0] * (pad_length - len(flat))
        else:
            flat = flat[:pad_length]
        return flat

    def compute_ego_basic_states(vehicle_id):
        """计算ego车辆基础状态信息"""
        ego_position_x = ego_statistics[vehicle_id]['position'][0]
        norm_ego_position_x = ego_position_x / max_x
        distance_target = 1 - norm_ego_position_x
        ego_position_y = ego_statistics[vehicle_id]['position'][1]
        norm_ego_position_y = ego_position_y / 3.2

        ego_speed = ego_statistics[vehicle_id]['speed']
        norm_ego_speed = ego_speed / Parameters['max_v']

        ego_acc = ego_statistics[vehicle_id]['acceleration']
        norm_ego_acc = ego_acc / Parameters['max_a']

        ego_heading = ego_statistics[vehicle_id]['heading']
        norm_ego_heading = ego_heading / Parameters['max_heading']

        # 处理前车信息
        if 'front_1' in ego_statistics[vehicle_id]['surroundings'].keys():
            front_info = ego_statistics[vehicle_id]['surroundings']['front_1']
            desire_speed = front_info['veh_vel']
            ego_spacing = front_info['long_dist']
            ego_safetime = (front_info['long_ACT'] if self.use_V2V_info
                            else front_info['long_TTC'])
            relative_speed = front_info['long_rel_v']
        else:
            desire_speed = Parameters['max_v']
            ego_spacing = Parameters['minGap'] + ego_speed * Parameters['tau']
            ego_safetime = Parameters['max_TTC']
            relative_speed = ego_speed - desire_speed

        # 计算归一化状态
        desire_spacing = Parameters['minGap'] + ego_speed * Parameters['tau']
        norm_spacing = ego_spacing / desire_spacing
        norm_speed_deviation = relative_speed / Parameters['max_v']
        norm_spacing_deviation = (ego_spacing - desire_spacing) / desire_spacing
        norm_safetime = ego_safetime / Parameters['max_TTC']

        ego_info = [norm_ego_position_x, norm_ego_position_y, norm_ego_speed, norm_ego_acc, norm_ego_heading,
                    norm_speed_deviation, norm_spacing, norm_spacing_deviation, norm_safetime]

        # 计算历史信息
        ego_hist_info = compute_ego_history_info(vehicle_id, ego_position_x, ego_position_y, ego_speed, ego_acc, ego_heading)

        return {
            'position': [ego_position_x, ego_position_y],
            'speed': ego_speed,
            'acceleration': ego_acc,
            'heading': ego_heading,
            'distance_target': distance_target,
            'ego_info': ego_info,
            'ego_hist_info': ego_hist_info,
            'ego_safetime': ego_safetime
        }

    def compute_ego_history_info(vehicle_id, ego_position_x, ego_position_y, ego_speed, ego_acc, ego_heading):
        """计算ego车辆历史信息"""
        if self.use_hist_info:
            ego_hist_info = [2, 2]  # 与sur_pos_mark, sur_type对齐
            for i in range(self.hist_length, 0, -1):
                hist_data = self.vehicles_hist[f'hist_{i}'][vehicle_id]
                hist_pos_x = hist_data[1]
                hist_pos_y = hist_data[2]
                hist_vel = hist_data[3]
                hist_acc = hist_data[4]
                hist_heading = hist_data[5]
                hist_safety_time = hist_data[9]

                hist_relat_pos_x = (hist_pos_x - ego_position_x) / max_x
                hist_relat_pos_y = (hist_pos_y - ego_position_y) / 3.2
                hist_relat_vel = (hist_vel - ego_speed) / Parameters['max_v']
                hist_relat_acc = (hist_acc - ego_acc) / Parameters['max_a']
                hist_relat_heading = (hist_heading - ego_heading) / Parameters['max_heading']
                hist_safety_time_norm = hist_safety_time / Parameters['max_ACT']

                hist_info = [hist_relat_pos_x, hist_relat_pos_y, hist_relat_vel, hist_relat_acc,
                             hist_relat_heading, hist_safety_time_norm]
                ego_hist_info.extend(hist_info)
        else:
            ego_hist_info = [0.0] * (2 + self.hist_length * 6)

        return ego_hist_info

    def compute_ego_evaluation_metrics(vehicle_id):
        """计算ego车辆评估指标"""
        ego_safety = (reward_statistics[vehicle_id]['safety_ASI'] if self.use_V2V_info
                      else Parameters['safety_TSI'])
        ego_efficiency = reward_statistics[vehicle_id]['efficiency_ASR']
        ego_stability = (0.4 * reward_statistics[vehicle_id]['stability_SSVD'] +
                         0.6 * reward_statistics[vehicle_id]['stability_SSSD'])
        ego_comfort = (reward_statistics[vehicle_id]['comfort_JI'] if self.use_V2V_info
                       else reward_statistics[vehicle_id]['comfort_AI'])
        ego_control_efficiency = reward_statistics[vehicle_id]['control_efficiency']
        ego_comfort_cost = reward_statistics[vehicle_id]['comfort_cost']

        return {
            'safety': ego_safety,
            'efficiency': ego_efficiency,
            'stability': ego_stability,
            'comfort': ego_comfort,
            'control_efficiency': ego_control_efficiency,
            'comfort_cost': ego_comfort_cost
        }

    def compute_multilane_safetimes(vehicle_id):
        """计算多车道安全时间"""
        ego_stats = ego_statistics[vehicle_id]

        long_safetime = (ego_stats['surroundings']['front_1']['long_ACT'] if self.use_V2V_info
                         else ego_stats['surroundings']['front_1']['long_TTC']
        if 'front_1' in ego_stats['surroundings'] else None)

        left_safetime = (ego_stats['left_ACT'] if self.use_V2V_info else ego_stats['left_TTC'])
        right_safetime = (ego_stats['right_ACT'] if self.use_V2V_info else ego_stats['right_TTC'])

        return {
            'left': left_safetime,
            'long': long_safetime,
            'right': right_safetime
        }

    def process_single_surrounding_vehicle(sur_key, surrounding, ego_states, metrics_collector):
        """处理单个周围车辆"""
        if self.use_V2V_info:
            return process_v2v_vehicle(sur_key, surrounding, ego_states, metrics_collector)
        else:
            return process_non_v2v_vehicle(sur_key, surrounding, ego_states, metrics_collector)

    def process_v2v_vehicle(sur_key, surrounding, ego_states, metrics_collector):
        """处理V2V模式下的周围车辆"""
        sur_id = surrounding['veh_id']

        # 收集速度和加速度信息
        if sur_key in ['back_1', 'back_2', 'back_3']:
            metrics_collector['speeds'].append(reward_statistics[sur_id]['speed'])
            metrics_collector['accelerations'].append(abs(reward_statistics[sur_id]['acceleration']))

        # 计算当前状态
        sur_type = surrounding['veh_type']
        sur_pos_mark = surrounding['sur_type']
        norm_relat_pos = surrounding['long_dist'] / max_x
        norm_relat_speed = surrounding['long_rel_v'] / Parameters['max_v']
        norm_relat_acc = surrounding['long_rel_a'] / Parameters['max_a']
        norm_relat_heading = surrounding['rel_heading'] / Parameters['max_heading']
        norm_safetime = surrounding['long_ACT'] / Parameters['max_ACT']

        sur_info = [sur_pos_mark, sur_type, norm_relat_pos, norm_relat_speed,
                    norm_relat_acc, norm_relat_heading, norm_safetime]

        # 计算评估指标并更新收集器
        vehicle_metrics = compute_vehicle_evaluation_metrics(sur_id)
        update_metrics_collector(metrics_collector, vehicle_metrics)

        # 计算历史信息
        sur_hist_info = compute_vehicle_history_info(sur_id, sur_pos_mark, sur_type, ego_states)

        return sur_info, sur_hist_info

    def process_non_v2v_vehicle(sur_key, surrounding, ego_states, metrics_collector):
        """处理非V2V模式下的周围车辆"""
        if sur_key in ['front_1', 'back_1']:
            sur_id = surrounding['id']

            # 收集速度和加速度信息
            if sur_key in ['back_1', 'back_2', 'back_3']:
                metrics_collector['speeds'].append(reward_statistics[sur_id]['speed'])
                metrics_collector['accelerations'].append(abs(reward_statistics[sur_id]['acceleration']))

            # 计算当前状态
            sur_type = surrounding['veh_type']
            sur_pos_mark = surrounding['sur_type']
            norm_relat_pos = (surrounding['long_dist'] + 5) / max_x
            norm_relat_speed = surrounding['long_rel_v'] / Parameters['max_v']
            norm_relat_acc = 0
            norm_relat_heading = 0
            norm_safetime = surrounding['long_TTC'] / Parameters['max_TTC']

            sur_info = [sur_pos_mark, sur_type, norm_relat_pos, norm_relat_speed,
                        norm_relat_acc, norm_relat_heading, norm_safetime]

            # 计算评估指标并更新收集器
            vehicle_metrics = compute_vehicle_evaluation_metrics(sur_id)
            update_metrics_collector(metrics_collector, vehicle_metrics)

            # 计算历史信息
            sur_hist_info = compute_vehicle_history_info(sur_id, sur_pos_mark, sur_type, ego_states)
        else:
            sur_info = [0.0] * 7
            sur_hist_info = [0.0] * (2 + self.hist_length * 5)

        return sur_info, sur_hist_info

    def compute_vehicle_evaluation_metrics(vehicle_id):
        """计算车辆评估指标"""
        sur_safety = (reward_statistics[vehicle_id]['safety_ASI'] if self.use_V2V_info
                      else reward_statistics[vehicle_id]['safety_TSI'])
        sur_efficiency = reward_statistics[vehicle_id]['efficiency_ASR']
        sur_stability = (0.4 * reward_statistics[vehicle_id]['stability_SSVD'] +
                         0.6 * reward_statistics[vehicle_id]['stability_SSSD'])
        sur_comfort = (reward_statistics[vehicle_id]['comfort_JI'] if self.use_V2V_info
                       else reward_statistics[vehicle_id]['comfort_AI'])
        sur_control_efficiency = reward_statistics[vehicle_id]['control_efficiency']
        sur_comfort_cost = reward_statistics[vehicle_id]['comfort_cost']

        return {
            'safety': sur_safety,
            'efficiency': sur_efficiency,
            'stability': sur_stability,
            'comfort': sur_comfort,
            'control_efficiency': sur_control_efficiency,
            'comfort_cost': sur_comfort_cost
        }

    def update_metrics_collector(metrics_collector, vehicle_metrics):
        """更新指标收集器"""
        metrics_collector['safety_indices'].append(vehicle_metrics['safety'])
        metrics_collector['efficiency_indices'].append(vehicle_metrics['efficiency'])
        metrics_collector['stability_indices'].append(vehicle_metrics['stability'])
        metrics_collector['comfort_indices'].append(vehicle_metrics['comfort'])
        metrics_collector['control_efficiency_indices'].append(vehicle_metrics['control_efficiency'])
        metrics_collector['comfort_cost_indices'].append(vehicle_metrics['comfort_cost'])

    def compute_vehicle_history_info(vehicle_id, sur_pos_mark, sur_type, ego_states):
        """计算车辆历史信息"""
        if self.use_hist_info:
            sur_hist_info = [sur_pos_mark, sur_type]
            for i in range(self.hist_length, 0, -1):
                hist_data = self.vehicles_hist[f'hist_{i}'][vehicle_id]
                hist_pos_x = hist_data[1]
                hist_pos_y = hist_data[2]
                hist_vel = hist_data[3]
                hist_acc = hist_data[4]
                hist_heading = hist_data[5]
                hist_safety_time = hist_data[9]

                hist_relat_pos_x = (hist_pos_x - ego_states['position'][0]) / max_x
                hist_relat_pos_y = (hist_pos_y - ego_states['position'][1]) / 3.2
                hist_relat_vel = (hist_vel - ego_states['speed']) / Parameters['max_v']
                hist_relat_acc = (hist_acc - ego_states['acceleration']) / Parameters['max_a']
                hist_relat_heading = (hist_heading - ego_states['heading']) / Parameters['max_heading']
                hist_safety_time_norm = hist_safety_time / Parameters['max_ACT']

                hist_info = [hist_relat_pos_x, hist_relat_pos_y, hist_relat_vel, hist_relat_acc,
                             hist_relat_heading, hist_safety_time_norm]
                sur_hist_info.extend(hist_info)
        else:
            sur_hist_info = [0.0] * (2 + self.hist_length * 6)

        return sur_hist_info

    def process_surrounding_vehicles(vehicle_id, ego_states, ego_metrics):
        """处理周围车辆信息"""
        # 初始化数据容器
        surround_infos = []
        surround_hist_infos = []
        ego_lane_surround_infos = []
        ego_lane_surround_hist_infos = []
        left_lane_surround_infos = []
        left_lane_surround_hist_infos = []
        right_lane_surround_infos = []
        right_lane_surround_hist_infos = []

        # 初始化指标收集器
        metrics_collector = {
            'speeds': [ego_states['speed']],
            'accelerations': [ego_states['acceleration']],
            'safety_indices': [ego_metrics['safety']],
            'efficiency_indices': [ego_metrics['efficiency']],
            'stability_indices': [ego_metrics['stability']],
            'comfort_indices': [ego_metrics['comfort']],
            'control_efficiency_indices': [ego_metrics['control_efficiency']],
            'comfort_cost_indices': [ego_metrics['comfort_cost']]
        }

        # 处理14辆周围车辆
        for i_sur in range(14):
            sur_key = surround_key[i_sur]

            if sur_key in ego_statistics[vehicle_id]['surroundings_expand'].keys():
                sur_info, sur_hist_info = process_single_surrounding_vehicle(
                    sur_key, ego_statistics[vehicle_id]['surroundings_expand'][sur_key],
                    ego_states, metrics_collector
                )
            else:
                sur_info = [0.0] * 7
                sur_hist_info = [0.0] * (2 + self.hist_length * 6)

            # 分配到对应车道
            surround_infos.append(sur_info)
            surround_hist_infos.append(sur_hist_info)

            if i_sur < 6:  # ego车道
                ego_lane_surround_infos.append(sur_info)
                ego_lane_surround_hist_infos.append(sur_hist_info)
            elif i_sur < 10:  # 左车道
                left_lane_surround_infos.append(sur_info)
                left_lane_surround_hist_infos.append(sur_hist_info)
            else:  # 右车道
                right_lane_surround_infos.append(sur_info)
                right_lane_surround_hist_infos.append(sur_hist_info)

        return {
            'surround_infos': surround_infos,
            'surround_hist_infos': surround_hist_infos,
            'ego_lane': {
                'current': ego_lane_surround_infos,
                'history': ego_lane_surround_hist_infos
            },
            'left_lane': {
                'current': left_lane_surround_infos,
                'history': left_lane_surround_hist_infos
            },
            'right_lane': {
                'current': right_lane_surround_infos,
                'history': right_lane_surround_hist_infos
            },
            'metrics': metrics_collector
        }

    def update_local_evaluation_history(vehicle_id, local_evaluation):
        """更新局部评估历史"""
        if self.use_hist_info:
            for i_step in range(self.hist_length - 1, 0, -1):
                if self.local_evaluation_hist[f'hist_{i_step}'][vehicle_id] == [0.0] * 6:
                    self.local_evaluation_hist[f'hist_{i_step + 1}'][vehicle_id] = local_evaluation
                else:
                    self.local_evaluation_hist[f'hist_{i_step + 1}'][vehicle_id] = \
                        self.local_evaluation_hist[f'hist_{i_step}'][vehicle_id].copy()
            self.local_evaluation_hist[f'hist_1'][vehicle_id] = local_evaluation

    # ====================== 全局参数计算 ======================
    max_x = node_positions['J2'][0]
    min_x = node_positions['J0'][0]
    max_desire_spacing = Parameters['minGap'] + Parameters['max_v'] * Parameters['tau']

    surround_key = {
        0: 'front_1', 1: 'front_2', 2: 'front_3',
        3: 'back_1', 4: 'back_2', 5: 'back_3',
        6: 'left_leader_1', 7: 'left_leader_2',
        8: 'left_follower_1', 9: 'left_follower_2',
        10: 'right_leader_1', 11: 'right_leader_2',
        12: 'right_follower_1', 13: 'right_follower_2',
    }

    # ====================== 主循环：处理每个ego车辆 ======================
    for vehicle_id in ego_ids:
        # 计算ego车辆基础状态
        ego_states = compute_ego_basic_states(vehicle_id)

        # 计算ego车辆评估指标
        ego_metrics = compute_ego_evaluation_metrics(vehicle_id)

        # 计算多车道安全时间
        multilane_safetimes = compute_multilane_safetimes(vehicle_id)

        # 处理周围车辆信息
        surrounding_data = process_surrounding_vehicles(vehicle_id, ego_states, ego_metrics)

        # 计算局部评估指标
        mean_safety = np.mean(surrounding_data['metrics']['safety_indices'])
        mean_efficiency = np.mean(surrounding_data['metrics']['efficiency_indices'])
        mean_stability = np.mean(surrounding_data['metrics']['stability_indices'])
        mean_comfort = np.mean(surrounding_data['metrics']['comfort_indices'])
        mean_control_efficiency = np.mean(surrounding_data['metrics']['control_efficiency_indices'])
        mean_comfort_cost = np.mean(surrounding_data['metrics']['comfort_cost_indices'])

        local_evaluation = [mean_safety, mean_efficiency, mean_stability,
                            mean_comfort, mean_control_efficiency, mean_comfort_cost]

        # 展平周围车辆信息
        surround_infos_pad = flatten_and_pad(surrounding_data['surround_infos'], 14 * 7)
        surround_hist_infos_pad = flatten_and_pad(surrounding_data['surround_hist_infos'],
                                                  14 * (2 + self.hist_length * 6))

        # 展平各车道信息
        ego_lane_surround_infos_pad = flatten_and_pad(surrounding_data['ego_lane']['current'], 6 * 7)
        ego_lane_surround_hist_infos_pad = flatten_and_pad(surrounding_data['ego_lane']['history'],
                                                           6 * (2 + self.hist_length * 6))

        left_lane_surround_infos_pad = flatten_and_pad(surrounding_data['left_lane']['current'], 4 * 7)
        left_lane_surround_hist_infos_pad = flatten_and_pad(surrounding_data['left_lane']['history'],
                                                            4 * (2 + self.hist_length * 6))

        right_lane_surround_infos_pad = flatten_and_pad(surrounding_data['right_lane']['current'], 4 * 7)
        right_lane_surround_hist_infos_pad = flatten_and_pad(surrounding_data['right_lane']['history'],
                                                             4 * (2 + self.hist_length * 6))

        # 计算安全性指标（原有逻辑保持不变）
        safe_time = ego_statistics[vehicle_id]['surroundings']['front_1']['long_ACT'] if self.use_V2V_info \
            else ego_statistics[vehicle_id]['surroundings']['front_1']['long_TTC']
        threshold = Parameters['safe_warn_threshold']['ACT'] if self.use_V2V_info \
            else Parameters['safe_warn_threshold']['TTC']
        safety = 1 - max(0, 1 - safe_time / threshold) ** 2

        # 组装基础观测向量
        observation_vectors[vehicle_id]['road_info'] = [1, ego_states['distance_target']]
        observation_vectors[vehicle_id]['ego_info'] = ego_states['ego_info']
        observation_vectors[vehicle_id]['surround_info'] = surround_infos_pad
        observation_vectors[vehicle_id]['local_evaluation_info'] = local_evaluation
        observation_vectors[vehicle_id]['multilane_info'] = [
            multilane_safetimes['left'], multilane_safetimes['long'], multilane_safetimes['right']
        ]

        # 组装改进观测向量
        improved_observation_vectors[vehicle_id]['road_info'] = [1, ego_states['distance_target']]
        improved_observation_vectors[vehicle_id]['ego_cur_info'] = ego_states['ego_info']
        improved_observation_vectors[vehicle_id]['ego_hist_info'] = ego_states['ego_hist_info']
        improved_observation_vectors[vehicle_id]['ego_lane_surround_cur_info'] = ego_lane_surround_infos_pad
        improved_observation_vectors[vehicle_id]['left_lane_surround_cur_info'] = left_lane_surround_infos_pad
        improved_observation_vectors[vehicle_id]['right_lane_surround_cur_info'] = right_lane_surround_infos_pad
        improved_observation_vectors[vehicle_id]['ego_lane_surround_hist_info'] = ego_lane_surround_hist_infos_pad
        improved_observation_vectors[vehicle_id]['left_lane_surround_hist_info'] = left_lane_surround_hist_infos_pad
        improved_observation_vectors[vehicle_id]['right_lane_surround_hist_info'] = right_lane_surround_hist_infos_pad
        improved_observation_vectors[vehicle_id]['local_evaluation_info'] = local_evaluation
        improved_observation_vectors[vehicle_id]['multilane_info'] = [
            multilane_safetimes['left'], multilane_safetimes['long'], multilane_safetimes['right']
        ]

        # 更新历史评估信息
        update_local_evaluation_history(vehicle_id, local_evaluation)

    # ====================== 处理全局流量信息 ======================
    flow = []
    flow_hist_info = []

    for i_veh in range(self.num_CAVs + self.num_HDVs):
        i_order = int(i_veh / 5) * 10 + i_veh % 5
        ego_id = self.vehicle_names[i_order]
        veh_type = 1 if ego_id in ego_ids else 0

        # 计算当前状态
        norm_pos_x = reward_statistics[ego_id]['position'][0] / max_x
        norm_pos_y = reward_statistics[ego_id]['position'][1] / 3.2
        norm_speed = reward_statistics[ego_id]['speed'] / Parameters['max_v']
        norm_acc = reward_statistics[ego_id]['acceleration'] / Parameters['max_a']

        # 处理可能为None的值
        norm_safe_time = (reward_statistics[ego_id]['long_ACT'] / Parameters['max_ACT']
                          if reward_statistics[ego_id]['long_ACT'] is not None else 1)
        norm_DRAC = (reward_statistics[ego_id]['long_DRAC'] / Parameters['max_DRAC']
                     if reward_statistics[ego_id]['long_DRAC'] is not None else 0)
        norm_speed_deviation = (reward_statistics[ego_id]['delta_v'] / Parameters['max_v']
                                if reward_statistics[ego_id]['delta_v'] is not None else 0)
        norm_spacing_deviation = (reward_statistics[ego_id]['delta_s'] / max_x
                                  if reward_statistics[ego_id]['delta_s'] is not None else 0)

        veh_info = [veh_type, norm_pos_x, norm_pos_y, norm_speed, norm_acc, norm_safe_time,
                    norm_DRAC, norm_speed_deviation, norm_spacing_deviation]
        flow.append(veh_info)

        # 计算历史信息
        veh_hist_info = [veh_type]
        for i in range(self.hist_length, 0, -1):
            if self.use_hist_info:
                hist_data = self.vehicles_hist[f'hist_{i}'][ego_id]
                hist_info = [hist_data[1]/max_x, hist_data[2]/3.2, hist_data[3]/Parameters['max_v'],
                             hist_data[4]/Parameters['max_a'], hist_data[5]/Parameters['max_heading'], hist_data[9]/Parameters['max_ACT']]
            else:
                hist_info = [0.0] * 6
            veh_hist_info.extend(hist_info)

        flow_hist_info.append(veh_hist_info)

    flow_pad = flatten_and_pad(flow, (self.num_CAVs + self.num_HDVs) * 9)
    flow_hist_pad = flatten_and_pad(flow_hist_info, (self.num_CAVs + self.num_HDVs) * (1 + 6 * 10))

    # ====================== 处理全局评估信息 ======================
    flow_CAV_penetration = flow_statistics['CAV_penetration']
    flow_mean_speed = flow_statistics['mean_speed']
    flow_mean_acc = flow_statistics['mean_acceleration']
    flow_mean_rel_v = flow_statistics['mean_relative_speed']
    flow_mean_jerk = flow_statistics['mean_jerk']
    flow_lanechange = flow_statistics['lanechange_count']
    flow_mean_lanechange_gap = flow_statistics['mean_lanechange_gap']
    flow_mean_TTC = flow_statistics['mean_long_TTC']
    flow_mean_aggressiveness = flow_statistics['mean_aggressiveness']
    flow_mean_flow_rate = flow_statistics['mean_flow_rate']
    flow_mean_density = flow_statistics['mean_density']

    global_safety = evaluation['safety_ASI']
    global_efficiency = evaluation['efficiency_ASR']
    global_flowrate = evaluation['efficiency_TFR']
    global_stability = 0.6 * evaluation['stability_SSVD'] + 0.4 * evaluation['stability_SSSD']
    global_comfort_AI = evaluation['comfort_AI']
    global_comfort_JI = evaluation['comfort_JI']
    global_control_efficiency = evaluation['control_efficiency']
    global_comfort_cost = evaluation['comfort_cost']

    # ====================== 组装最终状态向量 ======================
    state = {
        'road_structure': [1],
        'flow_info': flow_pad,
        'flow_infos': [flow_CAV_penetration, flow_mean_speed, flow_mean_acc, flow_mean_rel_v, flow_mean_jerk,
                       flow_lanechange, flow_mean_lanechange_gap, flow_mean_TTC, flow_mean_aggressiveness,
                       flow_mean_flow_rate, flow_mean_density],
        'global_evaluation_info': [global_safety, global_efficiency, global_flowrate, global_stability,
                                   global_comfort_AI, global_comfort_JI, global_control_efficiency, global_comfort_cost]
    }

    improved_state = {
        'road_structure': [1],
        'flow_info': flow_pad,
        # 'flow_hist_info': flow_hist_pad,
        'flow_infos': [flow_CAV_penetration, flow_mean_speed, flow_mean_acc, flow_mean_rel_v, flow_mean_jerk,
                       flow_lanechange, flow_mean_lanechange_gap, flow_mean_TTC, flow_mean_aggressiveness,
                       flow_mean_flow_rate, flow_mean_density],
        'global_evaluation_info': [global_safety, global_efficiency, global_flowrate, global_stability,
                                   global_comfort_AI, global_comfort_JI, global_control_efficiency, global_comfort_cost]
    }

    # 为每个ego车辆分配状态向量
    for vehicle_id in ego_ids:
        shared_state_vectors[vehicle_id] = state
        improved_state_vectors[vehicle_id] = improved_state

    # ====================== 展平所有向量 ======================
    observation_vectors_flatten = {
        ego_id: flatten_to_1d(feature_vector)
        for ego_id, feature_vector in observation_vectors.items()
    }

    shared_state_vectors_flatten = {
        ego_id: flatten_to_1d(shared_feature_vector)
        for ego_id, shared_feature_vector in shared_state_vectors.items()
    }

    improved_observation_vectors_flatten = {
        ego_id: flatten_to_1d(feature_vector)
        for ego_id, feature_vector in improved_observation_vectors.items()
    }

    improved_state_vectors_flatten = {
        ego_id: flatten_to_1d(shared_feature_vector)
        for ego_id, shared_feature_vector in improved_state_vectors.items()
    }

    # ====================== 返回结果 ======================
    return improved_state_vectors, improved_state_vectors_flatten, improved_observation_vectors, improved_observation_vectors_flatten

# def dynamic_reward_computation(
#     self,
#     ego_ids: List[str],
#     reward_infos: Dict[str, List[Union[float, str, Tuple[int]]]],
#     reward_statistics: Dict[str, Tuple[float]],
#     flow_infos: Dict[str, List[Union[float, str, Tuple[int]]]]
# ):
#     rewards_dict = {}
#     evaluations_dict = {}
#     parameters = self.parameters
#
#     max_speed = parameters['max_v']  # ego vehicle 的最快的速度
#     max_acceleration = parameters['max_a']
#     TTC_max = parameters['max_TTC']
#     TTC_warning_threshold = parameters['safe_warn_threshold']['TTC']
#     TTC_collision_threshold = parameters['crash_threshold']['TTC']
#     K_TTC = 20 / (TTC_warning_threshold - TTC_collision_threshold)
#     T_exp = parameters['tau']
#     L_exp = parameters['minGap']
#     stability_Q = np.array([[1, 0], [0, 0.5]])
#     # stability_Q = np.array([[0.5, 0], [0, 1]])
#     comfort_R = 0.5
#     efficiency_R = 0.005
#
#     # 先把reward_statistics中所有的车辆的信息都全局记录下来self.vehicles_info
#     for veh_id, (road_id, distance, position_x, position_y, speed, acceleration, heading, waiting_time,
#                  accumulated_waiting_time) in reward_statistics.items():
#         self.vehicles_info[veh_id] = [
#             (self.vehicles_info.get(veh_id, [0, None])[0] + self.delta_t),  # travel time
#             road_id,
#             distance,
#             position_x,
#             position_y,
#             speed,
#             acceleration,
#             heading,
#             waiting_time,
#             accumulated_waiting_time
#         ]
#     # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
#     if len(self.out_of_road) > 0:
#         for veh_id in self.out_of_road:
#             if veh_id in self.vehicles_info:
#                 del self.vehicles_info[veh_id]
#
#     # ######################### 开始计算reward  # #########################
#     inidividual_rew_ego = {key: {} for key in list(set(self.ego_ids) - set(self.out_of_road))}
#
#     # ######################## 初始化 for group reward ########################
#     all_ego_vehicle_speed = []  # CAV车辆的平均速度 - 使用target speed
#     # all_ego_vehicle_mean_speed = []  # CAV车辆的累积平均速度 - 使用速度/时间
#     all_ego_vehicle_acceleration = []  # CAV车辆的平均加速度
#     all_ego_vehicle_accumulated_waiting_time = []  # # CAV车辆的累积平均等待时间
#     all_ego_vehicle_waiting_time = []  # CAV车辆的等待时间
#     all_ego_vehicle_safety_rewards = []
#     all_ego_vehicle_efficiency_rewards = []
#     all_ego_vehicle_stability_rewards = []
#     all_ego_vehicle_comfort_rewards = []
#     all_ego_vehicle_control_efficiency = []
#     all_ego_vehicle_comfort_cost = []
#
#     all_vehicle_speed = []  # CAV和HDV车辆的平均速度 - 使用target speed
#     # all_vehicle_mean_speed = []  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
#     all_vehicle_acceleration = []  # CAV和HDV车辆的平均加速度
#     all_vehicle_accumulated_waiting_time = []  # CAV和HDV车辆的累积平均等待时间
#     all_vehicle_waiting_time = []  # CAV和HDV车辆的等待时间
#     all_vehicle_safety_rewards = []
#     all_vehicle_efficiency_rewards = []
#     all_vehicle_stability_rewards = []
#     all_vehicle_comfort_rewards = []
#     all_vehicle_control_efficiency = []
#     all_vehicle_comfort_cost = []
#
#     v_lead = reward_infos['Leader']['speed']
#
#     for veh_id, veh_info in reward_infos.items():
#         if veh_id == 'Leader':
#             continue
#         else:
#             speed = veh_info['speed']
#             acceleration = veh_info['acceleration']
#             TTC = veh_info['TTC']
#             delta_s = veh_info['spacing_deviation']
#             delta_v = veh_info['speed_deviation']
#             waiting_time = veh_info['waiting_time']
#             accumulated_waiting_time = veh_info['accumulated_waiting_time']
#             relate_vel = veh_info['speed_deviation']
#             front_speed = speed - relate_vel
#
#             if TTC_collision_threshold < TTC and TTC <= TTC_warning_threshold:
#                 safety_r = (1 / (np.exp(-(K_TTC * (TTC - (TTC_collision_threshold + TTC_warning_threshold) / 2))) + 1)) - 1
#             elif TTC <= TTC_collision_threshold:
#                 safety_r = -1
#             else:
#                 safety_r = 0
#
#             efficiency_r = -abs(speed - max_speed) / max_speed * 1 + 1  # [0, 1]
#             # efficiency_r = (-(speed - max_speed)**2 / max_speed**2) + 1  # [0, 1] -- Marginal_Utility
#             # efficiency_r = -abs(speed - front_speed) / front_speed * 1 + 1
#             # efficiency_r = 0.2 * (-abs(speed - v_lead) / front_speed * 1 + 1) + 0.8 * (-abs(speed - max_speed) / max_speed * 1 + 1)
#             # effi_cost = efficiency_R * ((speed - max_speed) ** 2)
#             # efficiency_r = np.exp(-effi_cost)
#             # efficiency_r = speed / max_speed
#
#             stability_matric = np.array([delta_s, delta_v])
#             e_ss = stability_matric @ stability_Q @ stability_matric.T
#             # stability_r = np.exp(-e_ss)  # [0, 1] -- TVT2023-Li
#             stability_r = 1 / (1 + (0.1*e_ss)**4)  # [0, 1] -- Marginal_Utility
#
#             acc_cost = comfort_R*((acceleration) ** 2)
#             # comfort_r = np.exp(-acc_cost)  # [0, 1] -- TVT2023-Li
#             comfort_r = 1 / (1 + (1* ((acceleration) ** 2))**2)   # [0, 1] -- Marginal_Utility
#
#             all_vehicle_speed.append(speed)
#             all_vehicle_acceleration.append(acceleration)
#             all_vehicle_waiting_time.append(waiting_time)
#             all_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
#
#             all_vehicle_safety_rewards.append(safety_r)
#             all_vehicle_efficiency_rewards.append(efficiency_r)
#             all_vehicle_stability_rewards.append(stability_r)
#             all_vehicle_comfort_rewards.append(comfort_r)
#             all_vehicle_control_efficiency.append(e_ss)
#             all_vehicle_comfort_cost.append(acc_cost)
#
#             # 把CAV单独取出来
#             if veh_id in self.ego_ids:
#                 inidividual_rew_ego[veh_id]['safety'] = safety_r
#                 inidividual_rew_ego[veh_id]['efficiency'] = efficiency_r
#                 inidividual_rew_ego[veh_id]['comfort'] = stability_r
#                 inidividual_rew_ego[veh_id]['stability'] = comfort_r
#
#                 all_ego_vehicle_speed.append(speed)
#                 all_ego_vehicle_acceleration.append(acceleration)
#                 all_ego_vehicle_waiting_time.append(waiting_time)
#                 all_ego_vehicle_accumulated_waiting_time.append(efficiency_r)
#
#                 all_ego_vehicle_safety_rewards.append(safety_r)
#                 all_ego_vehicle_efficiency_rewards.append(efficiency_r)
#                 all_ego_vehicle_stability_rewards.append(stability_r)
#                 all_ego_vehicle_comfort_rewards.append(comfort_r)
#                 all_ego_vehicle_control_efficiency.append(e_ss)
#                 all_ego_vehicle_comfort_cost.append(acc_cost)
#
#     all_vehicle_mean_speed = np.mean(all_vehicle_speed)
#     al_vehicle_efficiency_r = np.mean(all_vehicle_efficiency_rewards)   # -abs(all_vehicle_mean_speed - max_speed) / max_speed * 1 + 1  # [0, 1]
#     all_vehicle_mean_control_efficiency = np.mean(all_vehicle_control_efficiency)
#     all_vehicle_stability_r = np.mean(all_vehicle_stability_rewards)   # np.exp(-all_vehicle_mean_control_efficiency)
#     all_vehicle_mean_comfort_cost = np.mean(all_vehicle_comfort_cost)
#     all_vehicle_comfort_r = np.mean(all_vehicle_comfort_rewards)   # np.exp(-all_vehicle_mean_comfort_cost)
#
#     all_ego_vehicle_mean_speed = np.mean(all_ego_vehicle_speed)
#     all_ego_vehicle_efficiency_r = np.mean(all_ego_vehicle_efficiency_rewards)   # -abs(all_ego_vehicle_mean_speed - max_speed) / max_speed * 1 + 1  # [0, 1]
#     all_ego_vehicle_mean_control_efficiency = np.mean(all_ego_vehicle_control_efficiency)
#     all_ego_vehicle_stability_r = np.mean(all_ego_vehicle_stability_rewards)   # np.exp(-all_ego_vehicle_mean_control_efficiency)
#     all_ego_vehicle_mean_comfort_cost = np.mean(all_ego_vehicle_comfort_cost)
#     all_ego_vehicle_comfort_r = np.mean(all_ego_vehicle_comfort_rewards)   # np.exp(-all_ego_vehicle_mean_comfort_cost)
#
#     individual_weight = 1 - self.CAV_penetration
#     global_weight = self.CAV_penetration
#     # individual_weight = 0
#     # global_weight = 1
#
#     safety_rewards = {
#         key: 1 * inidividual_rew_ego[key]['safety']
#         for key in inidividual_rew_ego
#     }
#     efficiency_rewards = {
#         key: individual_weight * inidividual_rew_ego[key]['efficiency'] + global_weight * all_ego_vehicle_efficiency_r  # al_vehicle_efficiency_r, all_ego_vehicle_efficiency_r
#         for key in inidividual_rew_ego
#     }
#     stability_rewards = {
#         key: individual_weight * inidividual_rew_ego[key]['stability'] + global_weight * all_ego_vehicle_stability_r  # all_vehicle_stability_r, all_ego_vehicle_stability_r
#         for key in inidividual_rew_ego
#     }
#     comfort_rewards = {
#         key: individual_weight * inidividual_rew_ego[key]['comfort'] + global_weight * all_ego_vehicle_comfort_r  # all_vehicle_comfort_r, all_ego_vehicle_comfort_r
#         for key in inidividual_rew_ego
#     }
#
#     weight = self.rew_weights
#     rewards = {
#         key: weight['safety'] * safety_rewards[key] \
#            + weight['efficiency'] * efficiency_rewards[key] \
#            + weight['stability'] * stability_rewards[key] \
#            + weight['comfort'] * comfort_rewards[key]
#         for key in inidividual_rew_ego
#         }
#
#     for ego_id in ego_ids:
#         rewards_dict[ego_id] = {
#             'local_safety': inidividual_rew_ego[ego_id]['safety'],
#             'local_efficiency': inidividual_rew_ego[ego_id]['efficiency'],
#             'local_stability': inidividual_rew_ego[ego_id]['stability'],
#             'local_comfort': inidividual_rew_ego[ego_id]['comfort'],
#             'global_safety': safety_rewards[ego_id],
#             'global_efficiency': efficiency_rewards[ego_id],
#             'global_stability': stability_rewards[ego_id],
#             'global_comfort': comfort_rewards[ego_id],
#             'weights': [weight['safety'], weight['efficiency'], weight['stability'], weight['comfort']],
#             'sum_reward': rewards[ego_id],
#         }
#
#     evaluations_dict = {
#         'safety_SI': flow_infos['safety_SI'],
#         'efficiency_ASR': flow_infos['efficiency_ASR'],
#         'efficiency_TFR': flow_infos['efficiency_TFR'],
#         'stability_SSVD': flow_infos['stability_SSVD'],
#         'stability_SSSD': flow_infos['stability_SSSD'],
#         'stability_VSS': flow_infos['stability_VSS'],
#         'comfort_AI': flow_infos['comfort_AI'],
#         'comfort_JI': flow_infos['comfort_JI'],
#         'control_efficiency': all_ego_vehicle_mean_control_efficiency,   # all_vehicle_mean_control_efficiency, all_ego_vehicle_mean_control_efficiency -- refer to the 2023TVT-Li
#         'control_reward': all_ego_vehicle_stability_r,  # all_vehicle_stability_r, all_ego_vehicle_stability_r
#         'mean_stability_reward': all_vehicle_stability_r,  # all_vehicle_stability_r, all_ego_vehicle_stability_r
#         'comfort_cost': all_ego_vehicle_mean_comfort_cost,   # all_vehicle_mean_comfort_cost, all_ego_vehicle_mean_comfort_cost -- refer to the 2023TVT-Li
#         'comfort_reward': all_ego_vehicle_comfort_r,  # all_vehicle_comfort_r, all_ego_vehicle_comfort_r
#         'mean_comfort_reward': all_vehicle_comfort_r  # all_vehicle_comfort_r, all_ego_vehicle_comfort_r
#     }
#
#     return rewards_dict, evaluations_dict


def hierarchical_reward_computation(
        self,
        ego_ids: List[str],
        reward_statistics: Dict[str, List[Union[float, str]]],
        flow_statistics: Dict[str, List[Union[float, str, Tuple[int]]]],
        evaluation: Dict[str, List[Union[float, str]]],
):
    """
    分层奖励计算函数
    """
    rewards_dict = {}
    evaluations_dict = {}
    parameters = self.parameters

    # 基础参数
    max_speed = parameters['max_v']
    max_acceleration = parameters['max_a']
    TTC_max = parameters['max_TTC']
    TTC_warning_threshold = parameters['safe_warn_threshold']['TTC']
    TTC_collision_threshold = parameters['crash_threshold']['TTC']
    K_TTC = 20 / (TTC_warning_threshold - TTC_collision_threshold)
    T_exp = parameters['tau']
    L_exp = parameters['minGap']
    stability_Q = np.array([[1, 0], [0, 0.5]])
    max_jerk = parameters['max_jerk']

    # 分层奖励权重参数
    hierarchical_weights = self.hierarchical_weights
    anti_fake_params = self.anti_fake_params

    gamma_passive = anti_fake_params['gamma_passive']
    delta_fake = anti_fake_params['delta_fake']
    theta_passive = anti_fake_params['theta_passive']
    phi_distance = anti_fake_params['phi_distance']
    lambda_speed = anti_fake_params['lambda_speed']

    # 更新车辆信息记录
    for veh_id, veh_info in reward_statistics.items():
        self.vehicles_info[veh_id] = [
            (self.vehicles_info.get(veh_id, [0, None])[0] + self.delta_t),
            veh_info['road_id'], veh_info['position'][0], veh_info['position'][1], veh_info['speed'],
            veh_info['acceleration'], veh_info['heading'], veh_info['waiting_time'],
            veh_info['accumulated_waiting_time']
        ]

    # 清理离开道路的车辆信息
    if len(self.out_of_road) > 0:
        for veh_id in self.out_of_road:
            if veh_id in self.vehicles_info:
                del self.vehicles_info[veh_id]

    # ================== 计算原有评估指标所需的统计量 ==================
    all_ego_vehicle_speed = []
    all_ego_vehicle_acceleration = []
    all_ego_vehicle_safety_rewards = []
    all_ego_vehicle_efficiency_rewards = []
    all_ego_vehicle_stability_rewards = []
    all_ego_vehicle_comfort_rewards = []
    all_ego_vehicle_control_efficiency = []
    all_ego_vehicle_comfort_cost = []

    all_vehicle_speed = []
    all_vehicle_acceleration = []
    all_vehicle_safety_rewards = []
    all_vehicle_efficiency_rewards = []
    all_vehicle_stability_rewards = []
    all_vehicle_comfort_rewards = []
    all_vehicle_control_efficiency = []
    all_vehicle_comfort_cost = []

    # 初始化个体奖励
    individual_rew_ego = {key: {} for key in list(set(self.ego_ids) - set(self.out_of_road))}

    # 计算车队效率基准
    all_speeds = [info['speed'] for info in reward_statistics.values() if info != reward_statistics.get('Leader', {})]
    fleet_efficiency_baseline = np.mean(all_speeds) if all_speeds else max_speed

    # 为每个车辆计算奖励
    for veh_id, veh_info in reward_statistics.items():
        # if veh_id == 'Leader':
        #     continue

        speed = veh_info['speed']
        acceleration = veh_info['acceleration']
        TTC = veh_info['long_TTC']
        ACT = veh_info['long_ACT']
        delta_s = veh_info['delta_v']
        delta_v = veh_info['delta_s']
        spacing = veh_info.get('spacing', delta_s + L_exp + speed * T_exp)
        waiting_time = veh_info['waiting_time']
        accumulated_waiting_time = veh_info['accumulated_waiting_time']
        jerk = veh_info['jerk']
        a_ideal = veh_info['a_ideal']
        s_baseline = veh_info['s_baseline']
        v_baseline = veh_info['v_baseline']

        # ================== 1. 基础奖励计算 ==================
        # 安全奖励
        if TTC_collision_threshold < TTC and TTC <= TTC_warning_threshold:
            safety_r = (1 / (np.exp(-(K_TTC * (TTC - (TTC_collision_threshold + TTC_warning_threshold) / 2))) + 1)) - 1
        elif TTC <= TTC_collision_threshold:
            safety_r = -1
        else:
            safety_r = 0

        # 效率奖励
        efficiency_r = -abs(speed - max_speed) / max_speed * 1 + 1

        # 稳定性奖励
        stability_matric = np.array([delta_s, delta_v])
        e_ss = stability_matric @ stability_Q @ stability_matric.T
        stability_r = 1 / (1 + (0.1 * e_ss) ** 4)

        # 舒适性奖励
        comfort_R = 0.5
        acc_cost = comfort_R * (acceleration ** 2)
        comfort_r = 1 / (1 + (1 * (acceleration ** 2)) ** 2)

        # ================== 2. 动态协调奖励计算 ==================
        # 控制平滑性
        smoothness_r = np.exp(-jerk ** 2 / max_jerk ** 2)

        # 响应质量
        response_r = np.exp(-abs(acceleration - a_ideal) / max_acceleration)

        # 动态协调总奖励
        coordination_r = (hierarchical_weights['smooth'] * smoothness_r +
                          hierarchical_weights['response'] * response_r)

        # ================== 3. 系统贡献奖励计算 ==================
        # 效率贡献度
        efficiency_contrib_r = (speed - fleet_efficiency_baseline) / max_speed

        # 稳定性贡献度
        efficiency_retention = min(1, speed / v_baseline) if v_baseline > 0 else 1
        stability_improvement = stability_r - 0.5  # 假设基准稳定性为0.5
        stability_contrib_r = stability_improvement * efficiency_retention

        # 系统贡献总奖励
        contribution_r = (hierarchical_weights['eff_contrib'] * efficiency_contrib_r +
                          hierarchical_weights['stab_contrib'] * stability_contrib_r)

        # ================== 4. 反虚假优化奖励计算 ==================
        # 消极行为评分
        s_desired = L_exp + speed * T_exp
        passivity_score = (spacing / s_desired - 1) + lambda_speed * (v_baseline - speed) / v_baseline if v_baseline > 0 else (spacing / s_desired - 1)
        passivity_penalty = -gamma_passive * max(0, passivity_score - theta_passive) ** 2

        # 虚假稳定性惩罚
        distance_inflation_ratio = max(0, (spacing - s_baseline) / s_baseline) if s_baseline > 0 else 0
        fake_stability_penalty = -delta_fake * max(0, distance_inflation_ratio - phi_distance)

        # 反虚假优化总奖励
        anti_fake_r = passivity_penalty + fake_stability_penalty

        # ================== 统计信息收集 ==================
        # 所有车辆统计
        all_vehicle_speed.append(speed)
        all_vehicle_acceleration.append(acceleration)
        all_vehicle_safety_rewards.append(safety_r)
        all_vehicle_efficiency_rewards.append(efficiency_r)
        all_vehicle_stability_rewards.append(stability_r)
        all_vehicle_comfort_rewards.append(comfort_r)
        all_vehicle_control_efficiency.append(e_ss)
        all_vehicle_comfort_cost.append(acc_cost)

        # CAV车辆单独统计
        if veh_id in self.ego_ids:
            all_ego_vehicle_speed.append(speed)
            all_ego_vehicle_acceleration.append(acceleration)
            all_ego_vehicle_safety_rewards.append(safety_r)
            all_ego_vehicle_efficiency_rewards.append(efficiency_r)
            all_ego_vehicle_stability_rewards.append(stability_r)
            all_ego_vehicle_comfort_rewards.append(comfort_r)
            all_ego_vehicle_control_efficiency.append(e_ss)
            all_ego_vehicle_comfort_cost.append(acc_cost)

            # 存储个体奖励
            individual_rew_ego[veh_id] = {
                'safety': safety_r,
                'efficiency': efficiency_r,
                'stability': stability_r,
                'comfort': comfort_r,
                'coordination': coordination_r,
                'contribution': contribution_r,
                'anti_fake': anti_fake_r,
                'smoothness': smoothness_r,
                'response': response_r,
                'efficiency_contrib': efficiency_contrib_r,
                'stability_contrib': stability_contrib_r,
                'passivity_penalty': passivity_penalty,
                'fake_stability_penalty': fake_stability_penalty
            }

    # ================== 计算全局统计指标 ==================
    all_vehicle_efficiency_r = np.mean(all_vehicle_efficiency_rewards) if all_vehicle_efficiency_rewards else 0
    all_vehicle_mean_control_efficiency = np.mean(
        all_vehicle_control_efficiency) if all_vehicle_control_efficiency else 0
    all_vehicle_stability_r = np.mean(all_vehicle_stability_rewards) if all_vehicle_stability_rewards else 0
    all_vehicle_mean_comfort_cost = np.mean(all_vehicle_comfort_cost) if all_vehicle_comfort_cost else 0
    all_vehicle_comfort_r = np.mean(all_vehicle_comfort_rewards) if all_vehicle_comfort_rewards else 0

    all_ego_vehicle_efficiency_r = np.mean(
        all_ego_vehicle_efficiency_rewards) if all_ego_vehicle_efficiency_rewards else 0
    all_ego_vehicle_mean_control_efficiency = np.mean(
        all_ego_vehicle_control_efficiency) if all_ego_vehicle_control_efficiency else 0
    all_ego_vehicle_stability_r = np.mean(all_ego_vehicle_stability_rewards) if all_ego_vehicle_stability_rewards else 0
    all_ego_vehicle_mean_comfort_cost = np.mean(all_ego_vehicle_comfort_cost) if all_ego_vehicle_comfort_cost else 0
    all_ego_vehicle_comfort_r = np.mean(all_ego_vehicle_comfort_rewards) if all_ego_vehicle_comfort_rewards else 0

    # 计算分层奖励的全局统计
    all_ego_coordination = [individual_rew_ego[key]['coordination'] for key in individual_rew_ego]
    all_ego_contribution = [individual_rew_ego[key]['contribution'] for key in individual_rew_ego]
    all_ego_anti_fake = [individual_rew_ego[key]['anti_fake'] for key in individual_rew_ego]

    global_coordination = np.mean(all_ego_coordination) if all_ego_coordination else 0
    global_contribution = np.mean(all_ego_contribution) if all_ego_contribution else 0
    global_anti_fake = np.mean(all_ego_anti_fake) if all_ego_anti_fake else 0

    # ================== 计算最终奖励 ==================
    individual_weight = 1 - self.CAV_penetration
    global_weight = self.CAV_penetration
    weight = self.rew_weights

    for ego_id in ego_ids:
        if ego_id not in individual_rew_ego:
            continue

        # 全局奖励（结合个体和全局）
        global_safety = individual_rew_ego[ego_id]['safety']
        global_efficiency = (individual_weight * individual_rew_ego[ego_id]['efficiency'] +
                             global_weight * all_ego_vehicle_efficiency_r)
        global_stability = (individual_weight * individual_rew_ego[ego_id]['stability'] +
                            global_weight * all_ego_vehicle_stability_r)
        global_comfort = (individual_weight * individual_rew_ego[ego_id]['comfort'] +
                          global_weight * all_ego_vehicle_comfort_r)

        # 分层奖励
        final_coordination = (individual_weight * individual_rew_ego[ego_id]['coordination'] +
                              global_weight * global_coordination)
        final_contribution = (individual_weight * individual_rew_ego[ego_id]['contribution'] +
                              global_weight * global_contribution)
        final_anti_fake = (individual_weight * individual_rew_ego[ego_id]['anti_fake'] +
                           global_weight * global_anti_fake)

        # 计算综合奖励
        basic_reward = (weight['safety'] * global_safety +
                        weight['efficiency'] * global_efficiency +
                        weight['stability'] * global_stability +
                        weight['comfort'] * global_comfort)

        hierarchical_reward = (hierarchical_weights['coordination'] * final_coordination +
                               hierarchical_weights['contribution'] * final_contribution +
                               hierarchical_weights['anti_fake'] * final_anti_fake)

        sum_reward = (hierarchical_weights['basic'] * basic_reward + hierarchical_reward)

        # 存储奖励结果
        rewards_dict[ego_id] = {
            'local_safety': individual_rew_ego[ego_id]['safety'],
            'local_efficiency': individual_rew_ego[ego_id]['efficiency'],
            'local_stability': individual_rew_ego[ego_id]['stability'],
            'local_comfort': individual_rew_ego[ego_id]['comfort'],
            'global_safety': global_safety,
            'global_efficiency': global_efficiency,
            'global_stability': global_stability,
            'global_comfort': global_comfort,
            'weights': [weight['safety'], weight['efficiency'], weight['stability'], weight['comfort']],
            'sum_reward': sum_reward,
            # 新增分层奖励详情
            'coordination': final_coordination,
            'contribution': final_contribution,
            'anti_fake': final_anti_fake,
            'smoothness': individual_rew_ego[ego_id]['smoothness'],
            'response': individual_rew_ego[ego_id]['response'],
            'efficiency_contrib': individual_rew_ego[ego_id]['efficiency_contrib'],
            'stability_contrib': individual_rew_ego[ego_id]['stability_contrib'],
            'passivity_penalty': individual_rew_ego[ego_id]['passivity_penalty'],
            'fake_stability_penalty': individual_rew_ego[ego_id]['fake_stability_penalty'],
            'hierarchical_weights': hierarchical_weights
        }

    # ================== 构建评估字典 ==================
    evaluations_dict = {
        # 原有flow_infos指标
        'safety_SI': evaluation['safety_TSI'],
        'efficiency_ASR': evaluation['efficiency_ASR'],
        'efficiency_TFR': evaluation['efficiency_TFR'],
        'stability_SSVD': evaluation['stability_SSVD'],
        'stability_SSSD': evaluation['stability_SSSD'],
        'comfort_AI': evaluation['comfort_AI'],
        'comfort_JI': evaluation['comfort_JI'],

        # 原有需要保留的评估指标
        'control_efficiency': all_ego_vehicle_mean_control_efficiency,
        'control_reward': all_ego_vehicle_stability_r,
        'mean_stability_reward': all_vehicle_stability_r,
        'comfort_cost': all_ego_vehicle_mean_comfort_cost,
        'comfort_reward': all_ego_vehicle_comfort_r,
        'mean_comfort_reward': all_vehicle_comfort_r,

        # 新增分层评估指标
        'mean_coordination': global_coordination,
        'mean_contribution': global_contribution,
        'mean_anti_fake': global_anti_fake,
        'fleet_efficiency_baseline': fleet_efficiency_baseline,
        'mean_smoothness': np.mean(
            [individual_rew_ego[k]['smoothness'] for k in individual_rew_ego]) if individual_rew_ego else 0,
        'mean_response': np.mean(
            [individual_rew_ego[k]['response'] for k in individual_rew_ego]) if individual_rew_ego else 0,
        'mean_efficiency_contrib': np.mean(
            [individual_rew_ego[k]['efficiency_contrib'] for k in individual_rew_ego]) if individual_rew_ego else 0,
        'mean_stability_contrib': np.mean(
            [individual_rew_ego[k]['stability_contrib'] for k in individual_rew_ego]) if individual_rew_ego else 0,
        'mean_passivity_penalty': np.mean(
            [individual_rew_ego[k]['passivity_penalty'] for k in individual_rew_ego]) if individual_rew_ego else 0,
        'mean_fake_stability_penalty': np.mean(
            [individual_rew_ego[k]['fake_stability_penalty'] for k in individual_rew_ego]) if individual_rew_ego else 0,
    }

    return rewards_dict, evaluations_dict
