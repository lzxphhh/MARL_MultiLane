import math
from typing import List, Dict, Tuple, Union
import copy
import os
import csv

import numpy as np

LANE_LENGTHS = {
    'E0': 500,
    'E1': 2000,
}
Lane_start = {
    'E0': (-500, 0),
    'E1': (0, 0),
}
Lane_end = {
    'E0': (0, 0),
    'E1': (2000, 0),
}


def crash_evaluation(state, ego_ids, Parameters):
    safety_collisions = []
    safety_warnings = []
    info = []

    # 把ego的state专门拿出来
    def filter_vehicles(vehicles, ego_ids):
        # Using dictionary comprehension to create a new dictionary
        # by iterating over all key-value pairs in the original dictionary
        # and including only those whose keys are in ego_ids
        filtered_vehicles = {key: value for key, value in vehicles.items() if key in ego_ids}
        return filtered_vehicles

    filtered_ego_vehicles = filter_vehicles(state, ego_ids)

    for ego_key, ego_value in filtered_ego_vehicles.items():
        c_info = None
        w_info = None

        front_info = ego_value['surroundings']['front_1']
        target_id = front_info['veh_id']
        spacing = front_info['long_dist']
        relative_speed = front_info['long_rel_v']
        TTC = front_info['long_TTC']
        ACT = front_info['long_ACT']

        if spacing < Parameters['crash_threshold']['spacing'] \
            or TTC < Parameters['crash_threshold']['TTC'] or \
            ACT < Parameters['crash_threshold']['ACT']:
            safety_collisions.append((ego_key, target_id))
            c_info = {
                'collision': True,
                'CAV_key': ego_key,
                'surround_key': target_id,
                'spacing': spacing,
                'relative_speed': relative_speed,
                'TTC': TTC,
                'ACT': ACT,
            }
        elif Parameters['crash_threshold']['spacing'] <= spacing < Parameters['safe_warn_threshold']['spacing'] \
                or Parameters['crash_threshold']['TTC'] <= TTC < Parameters['safe_warn_threshold']['TTC'] \
                or Parameters['crash_threshold']['ACT'] <= ACT < Parameters['safe_warn_threshold']['ACT']:
            safety_warnings.append((ego_key, target_id))
            w_info = {
                'warn': True,
                'CAV_key': ego_key,
                'surround_key': target_id,
                'spacing': spacing,
                'relative_speed': relative_speed,
                'TTC': TTC,
                'ACT': ACT,
            }
        if c_info:
            info.append(c_info)
        elif w_info:
            info.append(w_info)

    return safety_collisions, safety_warnings, info

def observation_analysis(state, reward_statistics, flow_statistics,
                         vehicle_names,
                         is_evaluation, save_csv_dir, sim_time,
                         ttc_hist, act_hist, crash_count,
                         scene_centers):
    for vehicle_id, vehicle in state.items():
        if vehicle['vehicle_type'] == 'ego':
            now_TTC = reward_statistics[vehicle_id]['long_TTC']
            now_ACT = reward_statistics[vehicle_id]['long_ACT']
            ttc_hist[vehicle_id].append(now_TTC)
            act_hist[vehicle_id].append(now_ACT)
            if now_TTC < 0.5 or now_ACT < 0.5:
                crash_count[vehicle_id] += 1

    if is_evaluation:
        for veh_order, veh_id in vehicle_names.items():
            # Create dictionary with all the data you want to save
            veh_info = {
                't': sim_time,
                'Position_x': reward_statistics[veh_id]['position'][0],
                # 'Position_y': platoon_infos[veh_id]['position'][1],  # Uncommented this line
                'v_Vel': reward_statistics[veh_id]['speed'],
                'v_Acc': reward_statistics[veh_id]['acceleration']
            }

            csv_path = save_csv_dir + '/' + f'{veh_order}' + '_' + veh_id + '.csv'

            # Check if file exists to write headers only once
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a', newline='') as csvfile:
                # Use DictWriter instead of writer to handle dictionary data properly
                writer = csv.DictWriter(csvfile, fieldnames=veh_info.keys())

                # Write header if file is being created for the first time
                if not file_exists:
                    writer.writeheader()

                # Write the row of data
                writer.writerow(veh_info)

    now_v_mean = flow_statistics['mean_speed']
    now_v_std = flow_statistics['std_speed']
    now_a_mean = flow_statistics['mean_acceleration']
    now_a_std = flow_statistics['std_acceleration']
    now_jerk_mean = flow_statistics['mean_jerk']
    now_density = flow_statistics['mean_density']
    now_thw_mean = flow_statistics['mean_time_headway']
    now_rel_v_mean = flow_statistics['mean_relative_speed']
    now_lc_count = flow_statistics['lanechange_count']
    now_lc_gap_mean = flow_statistics['mean_lanechange_gap']
    now_aggressiveness_mean = flow_statistics['mean_aggressiveness']
    now_ttc_mean = flow_statistics['mean_long_TTC']
    now_act_mean = flow_statistics['mean_long_ACT']
    now_flow_rate = flow_statistics['mean_flow_rate']

    current_features = {
        'v_mean': now_v_mean,
        'a_mean': now_a_mean,
        'a_std': now_a_std,
        'density': now_density
    }

    # Calculate distance to each scene center
    min_distance = float('inf')
    classified_scene = None
    distances = {}

    for scene_class, center in scene_centers.items():
        # Calculate Euclidean distance
        distance = np.sqrt(
            (current_features['v_mean'] - center['v_mean']) ** 2 +
            (current_features['a_mean'] - center['a_mean']) ** 2 +
            (current_features['a_std'] - center['a_std']) ** 2 +
            (current_features['density'] - center['density']) ** 2
        )

        distances[scene_class] = distance

        if distance < min_distance:
            min_distance = distance
            classified_scene = scene_class

    return ttc_hist, act_hist, crash_count, classified_scene




