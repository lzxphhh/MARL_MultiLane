import numpy as np
from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType
)

def format_conversion(pos_x, pos_y, speed, heading, acc, yaw_rate, lane_id, veh_type, front_v, front_spacing):
    if veh_type == 0:
        type = VehicleType.HDV
    else:
        type = VehicleType.CAV
    return VehicleState(
        x=pos_x, y=pos_y, v=speed, theta=heading, a=acc, omega=yaw_rate,
        lane_id=lane_id, vehicle_type=type, front_v=front_v, front_spacing=front_spacing
    )

def get_traffic_state(obs, lane_ids):
    traffic_state = {
        'lane_densities': [],
        'density_mean': 0,
        'density_std': 0,
        'lane_velocities': [],
        'velocity_mean': 0,
        'velocity_std': 0,
    }
    lane_info = {
        lane_id: {
            'positions': [],
            'speeds': [],
        } for lane_id in lane_ids
    }
    for veh_id, veh_info in obs.items():
        # if veh_id[:3] == 'CAV' or veh_id[:3] == 'HDV':
        position = veh_info['position'][0]
        speed = veh_info['speed']
        lane_id = str(veh_info['lane_index'])
        if lane_id in lane_ids:
            lane_info[lane_id]['positions'].append(position)
            lane_info[lane_id]['speeds'].append(speed)

    for lane_id in lane_ids:
        positions = lane_info[lane_id]['positions']
        speeds = lane_info[lane_id]['speeds']
        platoon_length = (np.max(positions) - np.min(positions)) / 1000
        vehicle_num = len(positions)
        lane_density = vehicle_num / platoon_length
        mean_speed = np.mean(speeds)
        traffic_state['lane_densities'].append(lane_density)
        traffic_state['lane_velocities'].append(mean_speed)

    lane_densities = traffic_state['lane_densities']
    density_mean = np.mean(lane_densities)
    density_std = np.std(lane_densities)
    lane_velocities = traffic_state['lane_velocities']
    velocity_mean = np.mean(lane_velocities)
    velocity_std = np.std(lane_velocities)

    traffic_state['density_mean'] = density_mean
    traffic_state['density_std'] = density_std
    traffic_state['velocity_mean'] = velocity_mean
    traffic_state['velocity_std'] = velocity_std

    return traffic_state


def get_surrounding_state(obs, vehicle_id, use_gui):
    if use_gui:
        import traci as traci
    else:
        import libsumo as traci

    def get_vehicle_type(target_id):
        """获取车辆类型"""
        if target_id[:3] == 'HDV' or target_id[:6] == 'Leader' or target_id[:8] == 'Follower':
            return 0
        elif target_id[:3] == 'CAV':
            return 1
        else:
            raise ValueError('Unknown vehicle type')

    def create_vehicle_info(sur_type, target_id, veh_type, target_pos, target_v, target_a, target_heading):
        """创建车辆信息字典"""
        return {
            'sur_type': sur_type,
            'veh_id': target_id,
            'veh_type': veh_type,
            'veh_pos': target_pos,
            'veh_vel': target_v,
            'veh_acc': target_a,
            'veh_heading': target_heading,
        }

    def get_surrounding_type(is_leader, is_left, key_suffix):
        if is_leader:
            long_mark = 0 * 10
        else:
            long_mark = 1 * 10
        if is_left:
            lat_mark = 1 * 100
        else:
            lat_mark = 2 * 100
        num_mark = int(key_suffix)
        return lat_mark + long_mark + num_mark

    def process_neighbor_vehicle(target_id, is_leader, is_left, key_suffix):
        """处理邻近车辆"""
        target_pos = traci.vehicle.getPosition(target_id)
        pos_x = target_pos[0]
        pos_y = target_pos[1]
        target_v = traci.vehicle.getSpeed(target_id)
        target_a = traci.vehicle.getAcceleration(target_id)
        target_heading = traci.vehicle.getAngle(target_id)
        lane_id = traci.vehicle.getLaneIndex(target_id)

        # 获取车辆类型
        veh_type = get_vehicle_type(target_id)
        sur_mark = get_surrounding_type(is_leader, is_left, key_suffix)

        # 创建车辆信息
        sur_key = f"{'left' if is_left else 'right'}_{'leader' if is_leader else 'follower'}_{key_suffix}"
        vehicle_info = create_vehicle_info(sur_mark, target_id, veh_type, target_pos, target_v, target_a, target_heading)

        sur_mark = f"{'L' if is_left else 'R'}{'F' if is_leader else 'R'}_{key_suffix}"

        front_vehicle = traci.vehicle.getLeader(target_id, 300)
        if front_vehicle not in [None, ()] and front_vehicle[0] != '':
            front_id = front_vehicle[0]
            front_v = traci.vehicle.getSpeed(front_id)
            front_pos_x = traci.vehicle.getPosition(front_id)[0]
            front_spacing = front_pos_x - pos_x - 5
        else:
            front_v = -1
            front_spacing = -1

        sur_state = format_conversion(pos_x, pos_y, target_v, target_heading, target_a, 0, lane_id, veh_type, front_v, front_spacing)

        return sur_key, vehicle_info, sur_mark, sur_state

    def process_longitudinal_vehicle(target_id, is_leader, index):
        """处理纵向车辆（前车或后车）"""
        target_pos = traci.vehicle.getPosition(target_id)
        pos_x = target_pos[0]
        pos_y = target_pos[1]
        target_v = traci.vehicle.getSpeed(target_id)
        target_a = traci.vehicle.getAcceleration(target_id)
        target_heading = traci.vehicle.getAngle(target_id)
        lane_id = traci.vehicle.getLaneIndex(target_id)

        # 获取车辆类型
        veh_type = get_vehicle_type(target_id)
        if is_leader:
            long_mark = 0 * 10
            sur_key = f'ego_leader_{index}'
        else:
            long_mark = 1 * 10
            sur_key = f'ego_follower_{index}'
        num_mark = int(index)
        sur_mark = long_mark + num_mark

        vehicle_info = create_vehicle_info(sur_mark, target_id, veh_type, target_pos, target_v, target_a, target_heading)

        sur_mark = f"{'EF' if is_leader else 'ER'}_{index}"

        front_vehicle = traci.vehicle.getLeader(target_id, 300)
        if front_vehicle not in [None, ()] and front_vehicle[0] != '':
            front_id = front_vehicle[0]
            front_v = traci.vehicle.getSpeed(front_id)
            front_pos_x = traci.vehicle.getPosition(front_id)[0]
            front_spacing = front_pos_x - pos_x - 5
        else:
            front_v = -1
            front_spacing = -1

        sur_state = format_conversion(pos_x, pos_y, target_v, target_heading, target_a, 0, lane_id, veh_type, front_v, front_spacing)

        return sur_key, vehicle_info, sur_mark, sur_state


    surrounding_vehicle = {}
    surrounding_state = {}
    ego_pos = traci.vehicle.getPosition(vehicle_id)
    # 处理左右车道的后车和前车
    modes_config = {
        'left_followers': (0b000, False, True),  # (mode, is_leader, is_left)
        'right_followers': (0b001, False, False),
        'left_leaders': (0b010, True, True),
        'right_leaders': (0b011, True, False)
    }
    modes_vehicles = {
        'left_followers': [],
        'right_followers': [],
        'left_leaders': [],
        'right_leaders': []
    }

    for key, (mode, is_leader, is_left) in modes_config.items():
        neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
        for n in neighbors:
            if n not in [None, ()] and n[0] != '':
                target_id = n[0]
                modes_vehicles[key].append(target_id)

                # 处理第一辆邻近车辆
                sur_key, vehicle_info, sur_mark, sur_state = process_neighbor_vehicle(target_id, is_leader, is_left, "1")
                surrounding_vehicle[sur_key] = vehicle_info
                surrounding_state[sur_mark] = sur_state

                # 处理第二辆邻近车辆
                if is_leader:
                    neighbors_expand = traci.vehicle.getLeader(target_id)
                else:
                    neighbors_expand = traci.vehicle.getFollower(target_id)
                if neighbors_expand not in [None, ()] and neighbors_expand[0] != '':
                    target_id_2 = neighbors_expand[0]
                    modes_vehicles[key].append(target_id_2)
                    sur_key_2, vehicle_info_2, sur_mark, sur_state = process_neighbor_vehicle(target_id_2, is_leader, is_left, "2")
                    surrounding_vehicle[sur_key_2] = vehicle_info_2
                    surrounding_state[sur_mark] = sur_state
                else:
                    break
    # if len(surrounding_vehicle) != 4 and len(surrounding_vehicle) != 8:
    #     print('surrounding_vehicles is not a multiple of 4')

    if modes_vehicles['left_followers'] != [] and modes_vehicles['left_leaders'] == []:
        current_id = modes_vehicles['left_followers'][0]
        for i in range(2):
            front_vehicle = traci.vehicle.getLeader(current_id, 300)
            if front_vehicle not in [None, ()] and front_vehicle[0] != '':
                front_id = front_vehicle[0]
                sur_key, vehicle_info, sur_mark, sur_state = process_neighbor_vehicle(front_id, True, True, i+1)
                surrounding_vehicle[sur_key] = vehicle_info
                surrounding_state[sur_mark] = sur_state
                current_id = front_id
            else:
                break
    if modes_vehicles['right_followers'] != [] and modes_vehicles['right_leaders'] == []:
        current_id = modes_vehicles['right_followers'][0]
        for i in range(2):
            front_vehicle = traci.vehicle.getLeader(current_id, 300)
            if front_vehicle not in [None, ()] and front_vehicle[0] != '':
                front_id = front_vehicle[0]
                sur_key, vehicle_info, sur_mark, sur_state = process_neighbor_vehicle(front_id, True, False, i + 1)
                surrounding_vehicle[sur_key] = vehicle_info
                surrounding_state[sur_mark] = sur_state
                current_id = front_id
            else:
                break

    # 处理同车道前车
    current_id = vehicle_id
    for i in range(2):
        front_vehicle = traci.vehicle.getLeader(current_id, 300)
        if front_vehicle not in [None, ()] and front_vehicle[0] != '':
            front_id = front_vehicle[0]
            sur_key, vehicle_info, sur_mark, sur_state = process_longitudinal_vehicle( front_id, True, i + 1)
            surrounding_vehicle[sur_key] = vehicle_info
            surrounding_state[sur_mark] = sur_state
            current_id = front_id
        else:
            break

    # 处理同车道后车
    current_id = vehicle_id
    for i in range(2):
        back_vehicle = traci.vehicle.getFollower(current_id, 300)
        if back_vehicle not in [None, ()] and back_vehicle[0] != '':
            back_id = back_vehicle[0]
            sur_key, vehicle_info, sur_mark, sur_state = process_longitudinal_vehicle(back_id, False, i + 1)
            surrounding_vehicle[sur_key] = vehicle_info
            surrounding_state[sur_mark] = sur_state
            current_id = back_id
        else:
            break

    return surrounding_vehicle, surrounding_state

