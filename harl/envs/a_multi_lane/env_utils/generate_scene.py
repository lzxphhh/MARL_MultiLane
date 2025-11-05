import os
import pandas as pd
import traci
import libsumo as ls
from tshub.utils.get_abs_path import get_abs_path
import numpy as np

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
    for i in range(30):
        if i not in trajectory_dict:
            raise ValueError(f"缺少序号为{i}的CSV文件。")
    return trajectory_dict

def get_distribution_and_vehicle_names(quene_data, CAV_penetration, veh_num=10):
    """
    根据给定的 CAV_penetration 返回:
      1) distribution: 长度为10的列表(0=HDV, 1=CAV)
      2) vehicle_names: dict, 其中 key=车辆id(10~19)，value=车辆名称
                       10 为 'Leader'(相当于HDV_0)，其余根据 distribution 生成。
    """
    # 1. 预先定义各渗透率对应的 distribution
    lane_1_distributions_map = {
        0.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        0.1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        0.2: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        0.3: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        0.4: [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        0.5: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        0.6: [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        0.7: [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        0.8: [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        0.9: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        1.0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    lane_2_distributions_map = {
        0.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        0.1: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        0.2: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        0.3: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        0.4: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        0.5: [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        0.6: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        0.7: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        0.8: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        0.9: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        1.0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    }
    lane_3_distributions_map = {
        0.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        0.1: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        0.2: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        0.3: [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        0.4: [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        0.5: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        0.6: [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        0.7: [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        0.8: [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        0.9: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        1.0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    }

    # 如果不在字典里，就默认全部是 CAV
    lane_num_veh = int(veh_num / 3)
    lanes_distribution = [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1]
    num_HDV = int((1-CAV_penetration) * veh_num)
    # 2. 为车辆分配名称
    # Leader 固定为 vehicle_id=10, 相当于 HDV_0
    vehicle_names = {0: 'Leader_0', 10: 'Leader_1', 20: 'Leader_2'}

    # Leader 不计入车队
    platoon_rows = []
    distribution = {}
    cav_count = 0
    hdv_count = 0
    for i in range(3):
        leader_id = i*10
        platoon_info = {
            'id': leader_id,
            'name': f'Leader_{i}',
            'type': 'leader',
            'lane_index': str(i),
            'Position': quene_data[leader_id].loc[0, 'Position'],
            'v_Vel': quene_data[leader_id].loc[0, 'v_Vel'],
            'v_Acc': quene_data[leader_id].loc[0, 'v_Acc'],
            'spacing': 0,
            'time_headway': 0,
        }
        platoon_rows.append(platoon_info)
        for j in range(1, lane_num_veh):
            front_id = leader_id + j - 1
            follower_id = front_id + 1
            leader_df = quene_data[front_id]
            follower_df = quene_data[follower_id]
            leader_pos = leader_df.loc[0, 'Position']
            follower_pos = follower_df.loc[0, 'Position']
            follower_vel = follower_df.loc[0, 'v_Vel']
            follower_acc = follower_df.loc[0, 'v_Acc']
            headway = leader_pos - follower_pos - 5.0

            if lanes_distribution[i][j] == 1:
                minGap = 0.5
                tau = 0.8
                time_headway = (headway - minGap) / follower_vel
                distribution[follower_id] = 0   # CAV
                vehicle_names[follower_id] = f'CAV_{cav_count}'
                veh_type = 'ego'
                cav_count += 1
            else:    # HDV
                minGap = 1.5
                tau = 1.0
                time_headway = (headway - minGap) / follower_vel
                vehicle_names[follower_id] = f'HDV_{hdv_count}'
                if time_headway < 1.2:
                    distribution[follower_id] = 1
                    veh_type = 'HDV_0'
                elif time_headway < 2.0:
                    distribution[follower_id] = 2
                    veh_type = 'HDV_1'
                else:
                    distribution[follower_id] = 3
                    veh_type = 'HDV_2'
                hdv_count += 1
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

    return distribution, vehicle_names, platoon_rows


def generate_scenario(
        aggressive, cautious, normal,
        use_gui: bool, sce_name: str,
        CAV_num: int, HDV_num: int,
        CAV_penetration: float, distribution: str,
        group_id: int):
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

    folder_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_data/multi_lane_platoons/multi_lane_group_{group_id}'
    queues_data = read_csv_files(folder_path)

    veh_distribution, vehicle_names, platoon_rows = get_distribution_and_vehicle_names(queues_data, CAV_penetration, veh_num)

    quene_start = {veh_name: [] for veh_name in vehicle_names.values()}
    leader_pos = 500
    for i_veh in range(veh_num):
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

