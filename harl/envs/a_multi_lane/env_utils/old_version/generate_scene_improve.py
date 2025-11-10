import os
import random

import pandas as pd
import traci
import libsumo as ls
from tshub.utils.get_abs_path import get_abs_path
import numpy as np
from typing import List, Dict, Tuple, Union

def read_csv_files(folder_path):
    """
    读取文件夹中的所有CSV文件，并按照序号0-10进行排序
    返回一个字典，键为序号，值为DataFrame
    """
    trajectory_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            base_name, ext = os.path.splitext(filename)
            parts = base_name.split('_')
            if len(parts) < 2:
                continue  # 跳过不符合命名格式的文件
            index_str = parts[1]
            try:
                index = int(index_str)
                if 0 <= index <= 29:
                    filepath = os.path.join(folder_path, filename)
                    df = pd.read_csv(filepath)
                    # 确保必要的列存在
                    required_columns = {'t', 'Position', 'v_Vel', 'v_Acc'}
                    if not required_columns.issubset(df.columns):
                        raise ValueError(f"文件 {filename} 缺少必要的列。")
                    # 仅保留必要的列
                    df = df[['t', 'Position', 'v_Vel', 'v_Acc']]
                    trajectory_dict[index] = df
            except ValueError:
                continue  # 跳过序号不是整数的文件
    # 确保所有序号0-10都有对应的文件
    for i in range(10):
        if i not in trajectory_dict:
            raise ValueError(f"缺少序号为{i}的CSV文件。")
    return trajectory_dict

def get_distribution_and_vehicle_names(platoon_data_list, CAV_penetration, veh_num=10, aggressive=0.2, cautious=0.2, normal=0.6):
    """
    根据给定的 CAV_penetration 返回:
      1) distribution: 长度为10的列表(0=HDV, 1=CAV)
      2) vehicle_names: dict, 其中 key=车辆id(10~19)，value=车辆名称
                       10 为 'Leader'(相当于HDV_0)，其余根据 distribution 生成。
    """

    # 如果不在字典里，就默认全部是 CAV
    lane_num_veh = int(veh_num / 3)
    lanes_distribution = ([0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    num_HDV = int((1-CAV_penetration) * veh_num)
    # 2. 为车辆分配名称
    # Leader 固定 vehicle_id, 不算在HDV中，特征统计不涉及Leader
    vehicle_names = {0: 'Leader_0_0', 1: 'Leader_0_1', 2: 'Leader_0_2', 3: 'Leader_0_3', 4: 'Leader_0_4',
                     5: 'Leader_0_5', 6: 'Leader_0_6', 7: 'Leader_0_7', 8: 'Leader_0_8', 9: 'Leader_0_9',
                     100: 'Leader_1_0', 101: 'Leader_1_1', 102: 'Leader_1_2', 103: 'Leader_1_3', 104: 'Leader_1_4',
                     105: 'Leader_1_5', 106: 'Leader_1_6', 107: 'Leader_1_7', 108: 'Leader_1_8', 109: 'Leader_1_9',
                     200: 'Leader_2_0', 201: 'Leader_2_1', 202: 'Leader_2_2', 203: 'Leader_2_3', 204: 'Leader_2_4',
                     205: 'Leader_2_5', 206: 'Leader_2_6', 207: 'Leader_2_7', 208: 'Leader_2_8', 209: 'Leader_2_9',
                     20: 'Follower_0_20', 21: 'Follower_0_21', 22: 'Follower_0_22', 23: 'Follower_0_23', 24: 'Follower_0_24',
                     25: 'Follower_0_25', 26: 'Follower_0_26', 27: 'Follower_0_27', 28: 'Follower_0_28', 29: 'Follower_0_29',
                     120: 'Follower_1_20', 121: 'Follower_1_21', 122: 'Follower_1_22', 123: 'Follower_1_23', 124: 'Follower_1_24',
                     125: 'Follower_1_25', 126: 'Follower_1_26', 127: 'Follower_1_27', 128: 'Follower_1_28', 129: 'Follower_1_29',
                     220: 'Follower_2_20', 221: 'Follower_2_21', 222: 'Follower_2_22', 223: 'Follower_2_23', 224: 'Follower_2_24',
                     225: 'Follower_2_25', 226: 'Follower_2_26', 227: 'Follower_2_27', 228: 'Follower_2_28', 229: 'Follower_2_29',
                     }

    # Leader 不计入车队
    platoon_rows = []
    distribution = {}
    cav_count = 0
    hdv_count = 0
    create_veh_num = 0
    for i in range(3):
        [mean_v, std_v] = platoon_data_list[i]
        for j in range(10):
            minGap = 1.5
            tau = 2.5
            vehicle_id = i * 100 + j
            vehicle_name = f'Leader_{i}_{j}'
            des_vel = mean_v + random.uniform(-1,1) * std_v /2
            des_acc = 0
            time_headway = tau + random.random() * 0.2
            headway = minGap + des_vel * time_headway
            if j == 0:
                des_vel = mean_v
                des_pos = 0.0 - random.random() * 10
            else:
                front_pos = platoon_rows[create_veh_num-1]['Position']
                des_pos = front_pos - headway - 5
            platoon_info = {
                'id': vehicle_id,
                'name': vehicle_name,
                'type': 'leader',
                'lane_index': str(i),
                'Position': des_pos,
                'v_Vel': des_vel,
                'v_Acc': des_acc,
                'spacing': headway,
                'time_headway': time_headway,
            }
            platoon_rows.append(platoon_info)
            create_veh_num += 1

        leader_id = vehicle_id
        for j in range(10, 10+lane_num_veh):
            front_id = leader_id + j - 10
            follower_id = front_id + 1
            leader_pos = platoon_rows[create_veh_num-1]['Position']
            follower_vel = mean_v + random.uniform(-1,1) * std_v / 4
            follower_acc = 0
            if lanes_distribution[i][j-10] == 1:
                minGap = 1.0
                tau = 1.5
                distribution[follower_id] = 0   # CAV
                vehicle_names[follower_id] = f'CAV_{cav_count}'
                veh_type = 'ego'
                cav_count += 1
            else:    # HDV
                vehicle_names[follower_id] = f'HDV_{hdv_count}'
                hdv_type = random.random()
                if hdv_type < aggressive:
                    distribution[follower_id] = 1
                    veh_type = 'HDV_0'
                    minGap = 0.8
                    tau = 1.5
                elif hdv_type < aggressive+normal:
                    distribution[follower_id] = 2
                    veh_type = 'HDV_1'
                    minGap = 1.0
                    tau = 2.2
                else:
                    distribution[follower_id] = 3
                    veh_type = 'HDV_2'
                    minGap = 1.5
                    tau = 2.7
                hdv_count += 1
            time_headway = tau + random.random() * 0.2
            headway = minGap + follower_vel * time_headway
            follower_pos = leader_pos - headway - 5
            new_row = {
                'id': follower_id,
                'name': vehicle_names[follower_id],
                'type': veh_type,
                'lane_index': str(i),
                'Position': follower_pos,
                'v_Vel': follower_vel,
                'v_Acc': follower_acc,
                'spacing': headway,
                'time_headway': time_headway,
            }
            platoon_rows.append(new_row)
            create_veh_num += 1
        for j in range(10+lane_num_veh, 20+lane_num_veh):
            front_id = leader_id + j - 10
            vehicle_id = front_id + 1
            minGap = 1.5
            tau = 2.5
            vehicle_name = f'Follower_{i}_{j+10-lane_num_veh}'
            des_vel = mean_v + random.uniform(-1,1) * std_v / 4
            des_acc = 0
            time_headway = tau + random.random() * 0.2
            headway = minGap + des_vel * time_headway
            front_pos = platoon_rows[create_veh_num-1]['Position']
            des_pos = front_pos - headway - 5
            platoon_info = {
                'id': vehicle_id,
                'name': vehicle_name,
                'type': 'follower',
                'lane_index': str(i),
                'Position': des_pos,
                'v_Vel': des_vel,
                'v_Acc': des_acc,
                'spacing': headway,
                'time_headway': time_headway,
            }
            platoon_rows.append(platoon_info)
            create_veh_num += 1

    return distribution, vehicle_names, platoon_rows


def generate_scenario_NGSIM(
        aggressive, cautious, normal,
        use_gui: bool, sce_name: str,
        CAV_num: int, HDV_num: int,
        CAV_penetration: float, distribution: str,
        scene_info: Dict[int, List[int]]):
    """
    Mixed Traffic Flow (MTF) scenario generation: v_0 = 10 m/s, v_max = 20 m/s
    use_gui: false for libsumo, true for traci
    sce_name: scenario name, e.g., "Env_Bottleneck"
    CAV_num: number of CAVs
    CAV_penetration: CAV penetration rate, only 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 can be used
    - the number of vehicles in the scenario is determined by veh_num=CAV_num/CAV_penetration
    - the number of HDVs is determined by HDV_num=veh_num-CAV_num
    - 3 types of HDVs are randomly generated, and the parameters of each type of HDVs are defined in .rou.xml
    --- HDV_0: aggressive, 0.2 probability
    --- HDV_1: cautious, 0.2 probability
    --- HDV_2: normal, 0.6 probability
    distribution: "random" or "uniform"
    """
    if use_gui:
        scene_change = traci.vehicle
    else:
        scene_change = ls.vehicle

    veh_num = CAV_num + HDV_num

    lane_0_data = scene_info[0]
    lane_1_data = scene_info[1]
    lane_2_data = scene_info[2]

    platoon_data_list = [lane_0_data, lane_1_data, lane_2_data]

    veh_distribution, vehicle_names, platoon_rows = get_distribution_and_vehicle_names(platoon_data_list, CAV_penetration, veh_num, aggressive, cautious, normal)

    quene_start = {veh_name: [] for veh_name in vehicle_names.values()}
    leader_pos = 1000
    num_all_veh = veh_num+3*20
    for i_veh in range(num_all_veh):
        lane_index = platoon_rows[i_veh]['lane_index']
        veh_pos = platoon_rows[i_veh]['Position']+leader_pos
        veh_speed = platoon_rows[i_veh]['v_Vel']
        veh_accel = platoon_rows[i_veh]['v_Acc']
        quene_start[platoon_rows[i_veh]['name']].append([veh_pos, veh_speed, veh_accel])
        scene_change.add(
            vehID=platoon_rows[i_veh]['name'],
            typeID=platoon_rows[i_veh]['type'],
            routeID=f'route_0',
            depart="now",
            departPos=f'{veh_pos}',
            departLane=lane_index,
            departSpeed=f'{veh_speed}',
        )
        # scene_change.setAcceleration(platoon_rows[i_veh]['name'], veh_accel, 0.1)

    start_speed = {key: quene_start[key][0][1] for key in quene_start.keys()}
    return quene_start, start_speed, vehicle_names

