"""
基于配置文件的场景生成器 - 改进版
以HDV+CAV段为基准对齐，Leader段反向推算位置
"""

import os
import json
import random
import pandas as pd
import traci
import libsumo as ls
import numpy as np
from typing import List, Dict, Tuple, Union

class ConfigBasedSceneGenerator:
    """基于配置文件的场景生成器"""

    def __init__(self, config_file: str = "vehicle_config.json"):
        """
        初始化场景生成器

        Args:
            config_file: 车辆配置文件路径
        """
        self.config_file = config_file
        self.config = None
        self.speed_ranges = {}
        self.vehicle_types = {}
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            raise Exception(f"加载配置文件失败: {e}")

        self._parse_config()
        print(f"已加载配置文件: {self.config_file}")

    def _parse_config(self):
        """解析配置文件内容"""
        self.speed_ranges = self.config.get("speed_ranges", {})

        # 构建车辆类型参数字典
        for base_type, type_config in self.config.get("vehicle_types", {}).items():
            self.vehicle_types[base_type] = {}

            base_params = type_config.get("base_params", {})
            speed_variants = type_config.get("speed_variants", {})

            for speed_level, params in speed_variants.items():
                # 合并基础参数和速度相关参数
                full_params = {**base_params, **params}
                self.vehicle_types[base_type][speed_level] = full_params

    def get_speed_level(self, speed: float) -> str:
        """根据速度获取速度等级"""
        for level, range_info in self.speed_ranges.items():
            if range_info["min"] <= speed <= range_info["max"]:
                return level

        # 默认返回medium
        return "medium"

    def get_vehicle_type_by_speed(self, base_type: str, speed: float) -> Tuple[str, Dict]:
        """
        根据速度范围选择对应的车辆类型并返回相应的参数

        Args:
            base_type: 基础类型 ('ego', 'leader', 'HDV_0', 'HDV_1', 'HDV_2')
            speed: 车辆速度

        Returns:
            tuple: (vehicle_type_id, parameters_dict)
        """
        if base_type not in self.vehicle_types:
            raise ValueError(f"未知的车辆类型: {base_type}")

        # ego类型特殊处理：返回统一的ego类型
        if base_type == "ego":
            # 使用medium等级的参数作为ego的统一参数
            speed_level = "medium"
            if speed_level not in self.vehicle_types[base_type]:
                speed_level = "low"  # 备选方案

            vehicle_type_id = "ego"  # 统一使用ego作为类型ID
            params = self.vehicle_types[base_type][speed_level]

            return vehicle_type_id, params

        # 其他类型按原逻辑处理
        speed_level = self.get_speed_level(speed)

        if speed_level not in self.vehicle_types[base_type]:
            # 如果没有对应的速度等级，使用medium作为默认值
            speed_level = "medium"
            if speed_level not in self.vehicle_types[base_type]:
                raise ValueError(f"车辆类型 {base_type} 没有定义速度等级 {speed_level}")

        # 生成类型ID（映射到原有的命名方式）
        level_mapping = {"low": "_0", "medium": "_1", "high": "_2"}
        suffix = level_mapping.get(speed_level, "_1")
        vehicle_type_id = f"{base_type}{suffix}"

        params = self.vehicle_types[base_type][speed_level]

        return vehicle_type_id, params


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


def generate_leader_parameters(scene_gen: ConfigBasedSceneGenerator, leader_file: str, mean_vel: float, std_vel: float,
                               num_leaders: int = 10):
    """
    生成Leader段车辆的速度和车头时距参数

    Args:
        scene_gen: 场景生成器实例
        mean_vel: 平均速度
        std_vel: 速度标准差
        num_leaders: Leader车辆数量

    Returns:
        list: 包含每辆Leader车参数的列表 [{'speed': x, 'time_headway': y, 'spacing': z, 'params': {...}}, ...]
    """
    leader_params = []
    first_leader_path = leader_file
    leader_data = pd.read_csv(first_leader_path)
    speed_command = leader_data.loc[0, 'Velocity']

    for i in range(num_leaders):
        # 生成随机速度
        if i == 0:
            # 第一辆Leader车使用平均速度
            des_vel = speed_command
        else:
            # 其他Leader车在平均速度基础上添加随机扰动
            des_vel = mean_vel + random.uniform(-1, 1) * std_vel / 2

        # 根据速度获取车辆类型和参数
        vehicle_type_id, params = scene_gen.get_vehicle_type_by_speed('leader', des_vel)

        tau = float(params['tau'])
        minGap = float(params['minGap'])

        # 生成车头时距（比tau大0.2以上）
        time_headway = tau + 0.3 + random.random() * 0.2

        # 计算跟车距离
        spacing = minGap + des_vel * time_headway

        leader_info = {
            'speed': des_vel,
            'time_headway': time_headway,
            'spacing': spacing,
            'vehicle_type_id': vehicle_type_id,
            'params': params
        }
        leader_params.append(leader_info)

    return leader_params


def calculate_leader_start_position(leader_params: List[Dict], hdv_cav_baseline: float = 0.0, num_leaders: int = 10):
    """
    根据Leader段参数反推第一辆Leader车的起始位置

    Args:
        leader_params: Leader段车辆参数列表
        hdv_cav_baseline: HDV+CAV段基准位置

    Returns:
        float: 第一辆Leader车的起始位置
    """
    # 计算Leader段所需的总长度
    total_leader_length = sum(param['spacing'] for param in leader_params[1:])  # 跳过第一辆车

    # 添加额外的安全间距（用于与HDV+CAV段的过渡）
    safety_margin = 20.0  # 额外安全距离

    # 第一辆Leader车的起始位置
    leader_start_pos = hdv_cav_baseline + total_leader_length + 5*(num_leaders-1) + safety_margin

    return leader_start_pos


def determine_leader_count(platoon_data_list):
    """
    根据各车道速度差异确定Leader数量
    速度低的车道生成15辆Leader，速度高的车道生成10辆Leader

    Args:
        platoon_data_list: 各车道的速度数据 [[mean_vel, std_vel], ...]

    Returns:
        list: 各车道的Leader数量 [count_lane0, count_lane1, count_lane2]
    """
    # 获取所有车道的平均速度
    lane_speeds = [data[0] for data in platoon_data_list]
    max_speed = max(lane_speeds)
    min_speed = min(lane_speeds)

    # 判断是否存在显著速度差异（差异大于1 m/s认为是显著差异）
    speed_diff_threshold = 1.0
    has_speed_diff = (max_speed - min_speed) > speed_diff_threshold

    leader_counts = []

    if not has_speed_diff:
        # 速度差异不大，所有车道使用相同的Leader数量
        leader_counts = [10, 10, 10]
        print(f"速度差异较小（{max_speed - min_speed:.1f} m/s），所有车道使用10辆Leader")
    else:
        # 存在显著速度差异，根据速度分配Leader数量
        for i, speed in enumerate(lane_speeds):
            if speed <= (min_speed + speed_diff_threshold / 2):
                # 低速车道：15辆Leader
                leader_counts.append(15)
                print(f"车道{i} 低速（{speed:.1f} m/s）：使用15辆Leader")
            else:
                # 高速车道：10辆Leader
                leader_counts.append(10)
                print(f"车道{i} 高速（{speed:.1f} m/s）：使用10辆Leader")

    return leader_counts


def get_distribution_and_vehicle_names(scene_gen: ConfigBasedSceneGenerator,
                                       leaders_path,
                                       platoon_data_list, CAV_penetration,
                                       veh_num=10, aggressive=0.2, cautious=0.2, normal=0.6):
    """
    改进版场景生成：以HDV+CAV段为基准对齐，Leader段反向推算

    Args:
        scene_gen: 场景生成器实例
        platoon_data_list: 各车道的速度数据 [[mean_vel, std_vel], ...]
        CAV_penetration: CAV渗透率
        veh_num: 每车道车辆数
        aggressive: 激进型HDV比例
        cautious: 谨慎型HDV比例
        normal: 正常型HDV比例

    Returns:
        tuple: (distribution, vehicle_names, platoon_rows)
    """

    # 车道分布配置
    lane_num_veh = int(veh_num / 3)
    lanes_distribution = ([0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                          [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                          [0, 0, 1, 0, 1, 0, 0, 1, 0, 1])

    # HDV+CAV段基准位置（设定为0）
    hdv_cav_baseline = 0.0

    # 根据速度差异确定各车道的Leader数量
    leader_counts = determine_leader_count(platoon_data_list)

    # 更新车辆名称字典以适应动态Leader数量
    vehicle_names = {}

    # 为各车道动态生成Leader车辆名称
    for lane_idx in range(3):
        for leader_idx in range(leader_counts[lane_idx]):
            vehicle_id = lane_idx * 100 + leader_idx
            vehicle_names[vehicle_id] = f'Leader_{lane_idx}_{leader_idx}'

    # HDV+CAV段和Follower段的名称保持原有逻辑
    for lane_idx in range(3):
        base_id = lane_idx * 100
        # 跟随车辆ID从20开始，避免与Leader ID冲突
        for j in range(10):
            follower_id = base_id + leader_counts[lane_idx] + 10 + j
            vehicle_names[follower_id] = f'Follower_{lane_idx}_{j}'

    platoon_rows = []
    distribution = {}
    cav_count = 0
    hdv_count = 0

    print(f"开始生成场景，HDV+CAV段基准位置: {hdv_cav_baseline}")

    # 为3个车道分别生成车辆
    for lane_idx in range(3):
        [mean_v, std_v] = platoon_data_list[lane_idx]
        print(f"\n=== 车道 {lane_idx} 生成 (平均速度: {mean_v:.1f} m/s) ===")

        # 阶段1: 生成Leader段车辆参数
        num_leaders = leader_counts[lane_idx]
        leader_file = leaders_path[lane_idx]
        leader_params = generate_leader_parameters(scene_gen, leader_file, mean_v, std_v, num_leaders)

        # 阶段2: 计算Leader段起始位置
        leader_start_pos = calculate_leader_start_position(leader_params, hdv_cav_baseline, num_leaders)
        print(f"Leader段起始位置: {leader_start_pos:.1f} m (Leader数量: {num_leaders})")

        # 阶段3: 生成Leader段车辆
        current_pos = leader_start_pos
        for j in range(num_leaders):
            vehicle_id = lane_idx * 100 + j
            vehicle_name = f'Leader_{lane_idx}_{j}'

            leader_param = leader_params[j]
            des_vel = leader_param['speed']
            vehicle_type_id = leader_param['vehicle_type_id']
            time_headway = leader_param['time_headway']

            if j == 0:
                # 第一辆Leader车位置
                des_pos = current_pos
            else:
                # 后续Leader车位置：前车位置 - 当前车的跟车距离
                spacing = leader_param['spacing']
                des_pos = current_pos - spacing - 5  # 5m为车长余量

            platoon_info = {
                'id': vehicle_id,
                'name': vehicle_name,
                'type': vehicle_type_id,
                'lane_index': str(lane_idx),
                'Position': des_pos,
                'v_Vel': des_vel,
                'v_Acc': 0.0,
                'spacing': leader_param['spacing'] if j > 0 else 0,
                'time_headway': time_headway,
            }
            platoon_rows.append(platoon_info)
            current_pos = des_pos

        leader_end_pos = current_pos
        print(f"Leader段结束位置: {leader_end_pos:.1f} m")

        # 阶段4: 生成HDV+CAV段车辆（紧跟Leader段）
        for j in range(lane_num_veh):
            vehicle_id = lane_idx * 100 + num_leaders + j  # 对应follower_id的生成逻辑
            follower_vel = mean_v + random.uniform(-1, 1) * std_v / 4

            # 根据车道分布确定车辆类型
            if lanes_distribution[lane_idx][j] == 1:
                # CAV 车辆
                distribution[vehicle_id] = 0
                vehicle_names[vehicle_id] = f'CAV_{cav_count}'
                vehicle_type_id, params = scene_gen.get_vehicle_type_by_speed('ego', follower_vel)
                cav_count += 1
            else:
                # HDV 车辆
                vehicle_names[vehicle_id] = f'HDV_{hdv_count}'
                hdv_type = random.random()
                if hdv_type < aggressive:
                    distribution[vehicle_id] = 1
                    base_type = 'HDV_0'  # aggressive
                elif hdv_type < aggressive + normal:
                    distribution[vehicle_id] = 2
                    base_type = 'HDV_1'  # normal
                else:
                    distribution[vehicle_id] = 3
                    base_type = 'HDV_2'  # cautious

                vehicle_type_id, params = scene_gen.get_vehicle_type_by_speed(base_type, follower_vel)
                hdv_count += 1

            # 计算跟车参数
            minGap = float(params['minGap'])
            if follower_vel > 12 and vehicle_type_id == 'ego':
                tau = float(params['tau']) + 1.5
            else:
                tau = float(params['tau'])
            time_headway = tau + 0.7 + random.random() * 0.2
            spacing = minGap + follower_vel * time_headway

            # 位置：基于前车位置
            follower_pos = current_pos - spacing - 5

            hdv_cav_info = {
                'id': vehicle_id,
                'name': vehicle_names[vehicle_id],
                'type': vehicle_type_id,
                'lane_index': str(lane_idx),
                'Position': follower_pos,
                'v_Vel': follower_vel,
                'v_Acc': 0.0,
                'spacing': spacing,
                'time_headway': time_headway,
            }
            platoon_rows.append(hdv_cav_info)
            current_pos = follower_pos

        hdv_cav_end_pos = current_pos
        print(f"HDV+CAV段结束位置: {hdv_cav_end_pos:.1f} m")

        # 阶段5: 生成Follower段车辆（紧跟HDV+CAV段）
        remaining_followers = 10  # 剩余的Follower数量
        for j in range(remaining_followers):
            vehicle_id = lane_idx * 100 + num_leaders + 10 + j
            vehicle_name = f'Follower_{lane_idx}_{j}'
            des_vel = mean_v + random.uniform(-1, 1) * std_v / 4

            # 使用Follower类型
            vehicle_type_id, params = scene_gen.get_vehicle_type_by_speed('follower', des_vel)
            minGap = float(params['minGap'])
            tau = float(params['tau'])
            time_headway = tau + 0.3 + random.random() * 0.2
            spacing = minGap + des_vel * time_headway

            # 位置：基于前车位置
            follower_pos = current_pos - spacing - 5

            follower_info = {
                'id': vehicle_id,
                'name': vehicle_name,
                'type': vehicle_type_id,
                'lane_index': str(lane_idx),
                'Position': follower_pos,
                'v_Vel': des_vel,
                'v_Acc': 0.0,
                'spacing': spacing,
                'time_headway': time_headway,
            }
            platoon_rows.append(follower_info)
            current_pos = follower_pos

        print(f"Follower段结束位置: {current_pos:.1f} m")

    print(f"\n场景生成完成:")
    print(f"  总车辆数: {len(platoon_rows)}")
    print(f"  CAV数量: {cav_count}")
    print(f"  HDV数量: {hdv_count}")

    return distribution, vehicle_names, platoon_rows, leader_counts


def generate_scenario_NGSIM(
        config_file: str,
        aggressive, cautious, normal,
        use_gui: bool, sce_name: str,
        CAV_num: int, HDV_num: int,
        CAV_penetration: float, distribution: str,
        scene_info: Dict[int, List[int]],
        leaders_path: List[str],):
    """
    改进版混合交通流场景生成

    Args:
        config_file: 车辆配置文件路径
        aggressive: 激进型HDV比例
        cautious: 谨慎型HDV比例
        normal: 正常型HDV比例
        use_gui: false for libsumo, true for traci
        sce_name: scenario name
        CAV_num: CAV数量
        HDV_num: HDV数量
        CAV_penetration: CAV渗透率
        distribution: 分布类型
        scene_info: 场景信息字典

    Returns:
        tuple: (quene_start, start_speed, vehicle_names)
    """
    # 创建配置管理器
    scene_gen = ConfigBasedSceneGenerator(config_file)

    if use_gui:
        scene_change = traci.vehicle
    else:
        scene_change = ls.vehicle

    veh_num = CAV_num + HDV_num

    lane_0_data = scene_info[0]
    lane_1_data = scene_info[1]
    lane_2_data = scene_info[2]

    platoon_data_list = [lane_0_data, lane_1_data, lane_2_data]

    # 生成车辆分布和参数
    veh_distribution, vehicle_names, platoon_rows, leader_counts = get_distribution_and_vehicle_names(
        scene_gen, leaders_path, platoon_data_list, CAV_penetration, veh_num, aggressive, cautious, normal)

    # 初始化车辆队列
    quene_start = {veh_name: [] for veh_name in vehicle_names.values()}
    leader_pos_offset = 2000  # 整体位置偏移量
    num_all_veh = len(platoon_rows)

    print(f"\n开始添加 {num_all_veh} 辆车到SUMO仿真环境...")

    # 添加车辆到仿真环境
    success_count = 0
    for i_veh, veh_info in enumerate(platoon_rows):
        lane_index = veh_info['lane_index']
        veh_pos = veh_info['Position']
        veh_speed = veh_info['v_Vel']
        veh_accel = veh_info['v_Acc']
        veh_name = veh_info['name']
        veh_type = veh_info['type']

        # 记录车辆初始状态
        quene_start[veh_name].append([veh_pos, veh_speed, veh_accel])

        try:
            if veh_type in ["leader_0", "leader_1", "leader_2"]:
                scene_change.add(
                    vehID=veh_name,
                    typeID=veh_info['type'],
                    routeID='route_1',
                    depart="now",
                    departPos=f'{veh_pos}',
                    departLane=lane_index,
                    departSpeed=f'{veh_speed}',
                )
            else:
                scene_change.add(
                    vehID=veh_name,
                    typeID=veh_info['type'],
                    routeID='route_0',
                    depart="now",
                    departPos=f'{veh_pos + leader_pos_offset}',
                    departLane=lane_index,
                    departSpeed=f'{veh_speed}',
                )
            success_count += 1

            # 可选：设置初始加速度
            # scene_change.setAcceleration(veh_name, veh_accel, 0.1)

        except Exception as e:
            print(f"添加车辆 {veh_name} 失败: {e}")
            continue

    start_speed = {key: quene_start[key][0][1] if quene_start[key] else 0
                   for key in quene_start.keys()}

    print(f"\n场景初始化完成:")
    print(f"  成功添加车辆: {success_count}/{num_all_veh}")
    print(f"  CAV数量: {sum(1 for dist in veh_distribution.values() if dist == 0)}")
    print(f"  HDV数量: {len([dist for dist in veh_distribution.values() if dist > 0])}")

    return quene_start, start_speed, vehicle_names, leader_counts
