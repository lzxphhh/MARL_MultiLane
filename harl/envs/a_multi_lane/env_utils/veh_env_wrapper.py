import time
import os
from typing import Any, SupportsFloat, Tuple, Dict, List
import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from loguru import logger
import pandas as pd
import random

from .generate_scene_NGSIM import generate_scenario_NGSIM
from .wrapper_utils import (
    analyze_traffic,
    compute_centralized_vehicle_features_hierarchical_version,
    check_collisions_based_pos,
    check_collisions
)
from .state_utils import (
    crash_evaluation,
    observation_analysis,
)
from .state_selection import (
    feature_selection_simple_version,
    feature_selection_improved_version,
    # dynamic_reward_computation,
    hierarchical_reward_computation,
)
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from .test_baseline_flow.evaluation_single_method import evaluate_method

# 获得全局路径
path_convert = get_abs_path(__file__)
# 设置日志 -- tshub自带的给环境的
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level='ERROR')

GAP_THRESHOLD = 1.5
WARN_GAP_THRESHOLD = 3.0

# 定义IDM模型参数
IDM_PARAMS = {
    'a': 2.6,  # 最大加速度 (m/s^2)
    'b': 4.5,  # 舒适减速度 (m/s^2)
    'delta': 4,  # 加速度指数
    's0': 2.5,  # 最小安全距离 (m)
    'T': 1.0,  # 安全时间头距 (s)
    'v_desired': 30.0  # 期望速度 (m/s)
}

class VehEnvWrapper(gym.Wrapper):
    """Vehicle Env Wrapper for vehicle info
    """

    def __init__(self, env: Env,
                 name_scenario: str,  # 场景的名称
                 max_num_CAVs: int,  # 最大的 CAV 数量
                 max_num_HDVs: int,  # 最大的 HDV 数量
                 CAV_penetration: float,  # HDV 的数量
                 num_CAVs: int,  # CAV 的数量
                 num_HDVs: int,  # HDV 的数量
                 lane_max_num_vehs: int, # 每条车道车辆最大数量
                 edge_ids: List[str],  # 路网中所有路段的 id
                 edge_lane_num: Dict[str, int],  # 每个 edge 的车道数
                 calc_features_lane_ids: List[str],  # 需要统计特征的 lane id
                 node_positions: Dict[str, float],  # node的坐标
                 filepath: str,  # 日志文件的路径
                 delta_t: float,  # 动作之间的间隔时间
                 warmup_steps: int,  # reset 的时候仿真的步数, 确保 ego vehicle 可以全部出现
                 use_gui: bool,  # 是否使用 GUI
                 aggressive: float,  # aggressive 的概率
                 cautious: float,  # cautious 的概率
                 normal: float,  # normal 的概率
                 strategy: str, # MARL 的策略- feature extraction
                 use_hist_info: bool,  # 是否使用历史信息
                 hist_length: int,  # 历史信息的长度
                 num_seconds: int,  # 仿真的时间长度
                 rew_weights: Dict[str, float],  # 奖励的权重
                 scene_specify: bool,
                 scene_id: int, # leader在数据集中的位置
                 use_V2V_info: str, # 是否使用V2V
                 parameters: Dict[str, float], # 场景以及模型中的基础参数
                 is_evaluation: bool,  #开启测试与评估
                 save_model_name: str,   #模型名称（训练自动生成）--用于存储测试评估结果
                 eval_weights: Dict[str, float],
                 scene_centers: Dict[str, Dict[str, float]],
                 scene_definition: Dict[int, Dict[int, List[int]]],
                 scene_length: int,
                 ) -> None:
        super().__init__(env)
        self.name_scenario = name_scenario
        self.max_num_CAVs = max_num_CAVs
        self.max_num_HDVs = max_num_HDVs
        self.CAV_penetration = CAV_penetration
        self.num_CAVs = num_CAVs
        self.num_HDVs = num_HDVs
        self.lane_max_num_vehs = lane_max_num_vehs

        # random generate self.num_CAVs CAVs from range (4, 6)
        # self.num_CAVs = random.choice([4, 5, 6])
        # self.CAV_penetration = random.choice([0.4, 0.5, 0.6])

        self.is_evaluation = is_evaluation
        self.save_model_name = save_model_name
        self.eval_weights = eval_weights

        self.edge_ids = edge_ids
        self.edge_lane_num = edge_lane_num
        self.calc_features_lane_ids = calc_features_lane_ids  # 需要统计特征的 lane id
        self.node_positions = node_positions  # bottle neck 的坐标
        self.warmup_steps = warmup_steps
        self.use_gui = use_gui
        self.delta_t = delta_t
        self.max_num_seconds = num_seconds
        self.aggressive = aggressive
        self.cautious = cautious
        self.normal = normal
        self.strategy = strategy
        self.use_hist_info = use_hist_info
        self.hist_length = hist_length
        self.rew_weights = rew_weights
        self.use_V2V_info = use_V2V_info
        self.parameters = parameters
        self.scene_centers = scene_centers
        self.realtime_class = []

        self.ego_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.mixed_hdv_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.leader_ids = [f'Leader_{i}_{j}' for i in range(3) for j in range(20)]
        self.follower_ids = [f'Follower_{i}_{j}' for i in range(3) for j in range(10)]
        self.veh_ids = self.ego_ids + self.mixed_hdv_ids + self.leader_ids + self.follower_ids
        self.vehs_start = {}

        # 记录当前速度
        self.current_speed = {key: 0 for key in self.ego_ids}
        # 记录当前的lane
        self.current_lane = {key: 0 for key in self.ego_ids}

        self.congestion_level = 0  # 初始是不堵车的
        self.vehicles_info = {}  # 记录仿真内车辆的 (初始 lane index, travel time)
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}  # RL控制的车辆是否在路网上

        self.total_timesteps = 0  # 记录总的时间步数
        # #######
        # Writer
        # #######
        logger.info(f'RL: Log Path, {filepath}')
        self.t_start = time.time()
        # self.results_writer = ResultsWriter(
        #     filepath,
        #     header={"t_start": self.t_start},
        # )
        self.rewards_writer = list()
        if self.strategy == 'base':
            self.self_obs_size = 2 + 8 + 14 * 7 + 6 + 3
            self.shared_obs_size = 1 + (self.num_CAVs+self.num_HDVs) * 8 + 11 + 8
        elif self.strategy == 'improve':
            # with hist_info version
            self.self_obs_size = 2 + 9 + (2+6*10) + 14*7 + 14*(2+6*10) + 6 + 3
            self.shared_obs_size = 1 + (self.num_CAVs+self.num_HDVs)*9 + 11 + 8 # + (self.num_CAVs+self.num_HDVs)*(1+6*10)
        self.vehicles_hist = {}
        # self.lanes_hist = {}
        self.flow_hist = {}
        self.local_evaluation_hist = {}
        self.global_evaluation_hist = {}
        self.veh_distribution_hist = {}
        if self.use_hist_info:
            self.obs_size = self.self_obs_size
            for i in range(self.hist_length):
                self.vehicles_hist[f'hist_{i+1}'] = {veh_id: [0.0]*10 for veh_id in self.veh_ids}
                # self.lanes_hist[f'hist_{i+1}'] = {lane_id: np.zeros(26) for lane_id in self.calc_features_lane_ids}
                self.flow_hist[f'hist_{i+1}'] = [0.0]*6
                self.local_evaluation_hist[f'hist_{i+1}'] = {veh_id: [0.0]*6 for veh_id in self.ego_ids}
                self.global_evaluation_hist[f'hist_{i+1}'] = [0.0]*6
                self.veh_distribution_hist[f'hist_{i+1}'] = np.zeros(self.lane_max_num_vehs * 6)
        else:
            self.obs_size = self.self_obs_size
        self.surround_vehicles = {ego_id: {} for ego_id in self.ego_ids}
        self.surround_vehicles_expand = {ego_id: {} for ego_id in self.ego_ids}
        self.required_surroundings = ['front_1', 'back_1']
        self.TTC_assessment = {ego_id: {key: self.parameters['max_TTC'] for key in ['left', 'right', 'current']} for ego_id in self.ego_ids}
        self.change_action_mark = {ego_id: [] for ego_id in self.ego_ids}
        self.ttc_record = {ego_id: [] for ego_id in self.ego_ids}
        self.act_record = {ego_id: [] for ego_id in self.ego_ids}
        self.crash_record = {ego_id: 0 for ego_id in self.ego_ids}

        self.actor_action = {ego_id: [] for ego_id in self.ego_ids}
        self.scene_definition = scene_definition
        if scene_specify:
            self.scene_class_id = scene_id
        else:
            self.scene_class_id = random.randint(0, 11)
        self.scene_length = scene_length
        self.scene_info = scene_definition[self.scene_class_id]
        self.config_file = "/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/vehicle_config.json"

        # # 保存车辆运动信息
        if self.is_evaluation:
            self.save_csv_dir = "/home/lzx/00_Projects/00_platoon_MARL/HARL/examples/results/a_multi_lane/multi-lane/results_analysis/" + save_model_name + "/" + f"multi_lane_group_{self.group_id}" # 在测试完成后更改文件夹文字为config的模型名称如 seed-00042-2025-05-21-10-18-38
            # Check if folder exists
            if os.path.exists(self.save_csv_dir):
                # Folder exists, so empty it
                for file in os.listdir(self.save_csv_dir):
                    file_path = os.path.join(self.save_csv_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
            else:
                # Folder doesn't exist, so create it
                os.makedirs(self.save_csv_dir)
        else:
            self.save_csv_dir = None
        self.save_motion_dir = None
        self.output_dir = None

        self.vehicle_names = {}
        self.leader_counts = []

    # #####################
    # Obs and Action Space
    # #####################
    @property
    def action_space(self):
        """定义连续的动作空间，加速度范围为 [-3, 3]"""
        # 一维动作空间
        # return {_ego_id: gym.spaces.Box(
        #     low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        # ) for _ego_id in self.ego_ids}
        # 二维动作空间
        return {_ego_id: gym.spaces.Box(
            low=np.array([-1.5, -3]), high=np.array([1.5, 3]), shape=(2,), dtype=np.float64
        ) for _ego_id in
                self.ego_ids}

    @property
    def observation_space(self):
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64,
        )
        return {_ego_id: obs_space for _ego_id in self.ego_ids}

    @property
    def share_observation_space(self):
        share_obs_space = gym.spaces.Box(
            low = -np.inf, high = np.inf, shape = (self.shared_obs_size,), dtype=np.float64,
        )
        return {_ego_id: share_obs_space for _ego_id in self.ego_ids}

    # ##################
    # Tools for observations
    # ##################
    def append_surrounding(self, state):
        # 车辆的长和宽
        v_length = 5
        v_width = 1.8
        surrounding_vehicles = {}
        sorrounding_vehicles_expand = {}

        def calculate_safety_metrics(long_dist, long_rel_v, long_rel_a):
            """计算安全指标：TTC, ACT, DRAC"""
            if long_dist <= 0:
                long_TTC = self.parameters['min_TTC']
                long_DRAC = self.parameters['max_DRAC']
                long_ACT = self.parameters['min_ACT']
            else:
                if long_rel_v > 0:
                    long_TTC = np.clip(long_dist / long_rel_v, self.parameters['min_TTC'], self.parameters['max_TTC'], dtype=np.float64)
                    long_DRAC = np.clip(long_rel_v ** 2 / long_dist, self.parameters['min_DRAC'], self.parameters['max_DRAC'], dtype=np.float64)
                else:
                    long_TTC = self.parameters['max_TTC']
                    long_DRAC = self.parameters['min_DRAC']

                if long_rel_a == 0:
                    long_ACT = long_TTC
                else:
                    D = long_rel_v ** 2 + 2 * long_dist * long_rel_a
                    if D < 0:
                        long_ACT = self.parameters['max_ACT']
                    else:
                        t_c_1 = (long_rel_v + np.sqrt(D)) / (0 - long_rel_a)
                        t_c_2 = (long_rel_v - np.sqrt(D)) / (0 - long_rel_a)
                        if t_c_1 <= 0 and t_c_2 <= 0:
                            long_ACT = self.parameters['max_ACT']
                        elif t_c_1 > 0 and t_c_2 > 0:
                            long_ACT = min(t_c_1, t_c_2)
                        else:
                            long_ACT = max(t_c_1, t_c_2)

            return long_TTC, long_ACT, long_DRAC

        def get_vehicle_type(target_id):
            """获取车辆类型"""
            if target_id[:3] == 'HDV' or target_id[:6] == 'Leader' or target_id[:8] == 'Follower':
                return 0
            elif target_id[:3] == 'CAV':
                return 1
            else:
                raise ValueError('Unknown vehicle type')

        def create_vehicle_info(sur_type, target_id, veh_type, target_pos, target_v, target_a, target_heading, long_dist, lat_dist,
                                long_rel_v, long_rel_a, rel_heading, long_TTC, long_ACT, long_DRAC):
            """创建车辆信息字典"""
            return {
                'sur_type': sur_type,
                'veh_id': target_id,
                'veh_type': veh_type,
                'veh_pos': target_pos,
                'veh_vel': target_v,
                'veh_acc': target_a,
                'veh_heading': target_heading,
                'long_dist': long_dist,
                'lat_dist': lat_dist,
                'long_rel_v': long_rel_v,
                'long_rel_a': long_rel_a,
                'rel_heading': rel_heading,
                'long_TTC': long_TTC,
                'long_ACT': long_ACT,
                'long_DRAC': long_DRAC,
            }

        def process_neighbor_vehicle(vehicle_id, target_id, ego_pos, is_leader, is_left, key_suffix):
            """处理邻近车辆"""
            if self.use_gui:
                import traci as traci
            else:
                import libsumo as traci

            target_pos = traci.vehicle.getPosition(target_id)
            target_v = traci.vehicle.getSpeed(target_id)
            target_a = traci.vehicle.getAcceleration(target_id)
            target_heading = traci.vehicle.getAngle(target_id)

            ego_v = traci.vehicle.getSpeed(vehicle_id)
            ego_a = traci.vehicle.getAcceleration(vehicle_id)
            ego_heading = traci.vehicle.getAngle(vehicle_id)
            rel_heading = ego_heading - target_heading

            if is_leader:
                # 处理前车
                back_pos = ego_pos[0]
                back_v = ego_v
                back_a = ego_a
                front_pos = target_pos[0]
                front_v = target_v
                front_a = target_a
                sur_type = 10 if is_left else 20
            else:
                # 处理后车
                front_pos = ego_pos[0]
                front_v = ego_v
                front_a = ego_a
                back_pos = target_pos[0]
                back_v = target_v
                back_a = target_a
                sur_type = 11 if is_left else 21

            long_dist = front_pos - back_pos - v_length
            long_rel_v = back_v - front_v
            long_rel_a = back_a - front_a

            # 计算横向距离
            if is_left:
                left_pos = target_pos[1]
                right_pos = ego_pos[1]
            else:
                left_pos = ego_pos[1]
                right_pos = target_pos[1]
            lat_dist = left_pos - right_pos - v_width

            # 计算安全指标
            long_TTC, long_ACT, long_DRAC = calculate_safety_metrics(long_dist, long_rel_v, long_rel_a)

            # 获取车辆类型
            veh_type = get_vehicle_type(target_id)

            # 创建车辆信息
            sur_key = f"{'left' if is_left else 'right'}_{'leader' if is_leader else 'follower'}_{key_suffix}"
            vehicle_info = create_vehicle_info(sur_type, target_id, veh_type, target_pos, target_v, target_a, target_heading, long_dist, lat_dist,
                                               long_rel_v, long_rel_a, rel_heading, long_TTC, long_ACT, long_DRAC)

            return sur_key, vehicle_info

        def process_longitudinal_vehicle(vehicle_id, target_id, is_leader, index):
            """处理纵向车辆（前车或后车）"""
            if self.use_gui:
                import traci as traci
            else:
                import libsumo as traci


            ego_pos = traci.vehicle.getPosition(vehicle_id)
            ego_heading = traci.vehicle.getAngle(vehicle_id)
            ego_v = traci.vehicle.getSpeed(vehicle_id)
            ego_a = traci.vehicle.getAcceleration(vehicle_id)

            target_pos = traci.vehicle.getPosition(target_id)
            target_v = traci.vehicle.getSpeed(target_id)
            target_a = traci.vehicle.getAcceleration(target_id)
            target_heading = traci.vehicle.getAngle(target_id)
            rel_heading = ego_heading - target_heading

            if is_leader:
                # 处理前车
                back_pos = ego_pos[0]
                back_v = traci.vehicle.getSpeed(vehicle_id)
                back_a = traci.vehicle.getAcceleration(vehicle_id)
                front_pos = target_pos[0]
                front_v = traci.vehicle.getSpeed(target_id)
                front_a = traci.vehicle.getAcceleration(target_id)
                sur_type = 0
                sur_key = f'front_{index}'
            else:
                # 处理后车
                front_pos = ego_pos[0]
                front_v = traci.vehicle.getSpeed(vehicle_id)
                front_a = traci.vehicle.getAcceleration(vehicle_id)
                back_pos = target_pos[0]
                back_v = traci.vehicle.getSpeed(target_id)
                back_a = traci.vehicle.getAcceleration(target_id)
                sur_type = 1
                sur_key = f'back_{index}'

            long_dist = front_pos - back_pos - v_length
            long_rel_v = back_v - front_v
            long_rel_a = back_a - front_a
            lat_dist = 0

            # 计算安全指标
            long_TTC, long_ACT, long_DRAC = calculate_safety_metrics(long_dist, long_rel_v, long_rel_a)

            # 获取车辆类型
            veh_type = get_vehicle_type(target_id)

            # 创建车辆信息
            vehicle_info = create_vehicle_info(sur_type, target_id, veh_type, target_pos, target_v, target_a, target_heading, long_dist, lat_dist,
                                               long_rel_v, long_rel_a, rel_heading, long_TTC, long_ACT, long_DRAC)

            return sur_key, vehicle_info

        for vehicle_id in state['vehicle'].keys():
            # 对于所有RL控制的车辆
            if vehicle_id in self.ego_ids:
                if self.use_gui:
                    import traci as traci
                else:
                    import libsumo as traci

                surrounding_vehicle = {}
                sorrounding_vehicle_expand = {}
                ego_pos = traci.vehicle.getPosition(vehicle_id)

                # 处理左右车道的后车和前车
                modes_config = {
                    'left_followers': (0b000, False, True),  # (mode, is_leader, is_left)
                    'right_followers': (0b001, False, False),
                    'left_leaders': (0b010, True, True),
                    'right_leaders': (0b011, True, False)
                }

                for key, (mode, is_leader, is_left) in modes_config.items():
                    neighbors = traci.vehicle.getNeighbors(vehicle_id, mode)
                    for n in neighbors:
                        if n not in [None, ()] and n[0] != '':
                            target_id = n[0]

                            # 处理第一辆邻近车辆
                            sur_key, vehicle_info = process_neighbor_vehicle(vehicle_id, target_id, ego_pos,
                                                                             is_leader, is_left, "1")
                            surrounding_vehicle[sur_key] = vehicle_info
                            sorrounding_vehicle_expand[sur_key] = vehicle_info

                            # 处理第二辆邻近车辆
                            if is_leader:
                                neighbors_expand = traci.vehicle.getLeader(target_id)
                            else:
                                neighbors_expand = traci.vehicle.getFollower(target_id)
                            if neighbors_expand not in [None, ()] and neighbors_expand[0] != '':
                                target_id_2 = neighbors_expand[0]
                                sur_key_2, vehicle_info_2 = process_neighbor_vehicle(vehicle_id, target_id_2, ego_pos,
                                                                                     is_leader, is_left, "2")
                                sorrounding_vehicle_expand[sur_key_2] = vehicle_info_2

                # 处理同车道前车
                current_id = vehicle_id
                for i in range(3):
                    front_vehicle = traci.vehicle.getLeader(current_id, 300)
                    if front_vehicle not in [None, ()] and front_vehicle[0] != '':
                        front_id = front_vehicle[0]
                        sur_key, vehicle_info = process_longitudinal_vehicle(vehicle_id, front_id, True, i + 1)

                        if i == 0:
                            surrounding_vehicle[sur_key] = vehicle_info
                        sorrounding_vehicle_expand[sur_key] = vehicle_info
                        current_id = front_id
                    else:
                        break

                # 处理同车道后车
                current_id = vehicle_id
                for i in range(3):
                    back_vehicle = traci.vehicle.getFollower(current_id, 300)
                    if back_vehicle not in [None, ()] and back_vehicle[0] != '':
                        back_id = back_vehicle[0]
                        sur_key, vehicle_info = process_longitudinal_vehicle(vehicle_id, back_id, False, i + 1)

                        if i == 0:
                            surrounding_vehicle[sur_key] = vehicle_info
                        sorrounding_vehicle_expand[sur_key] = vehicle_info
                        current_id = back_id
                    else:
                        break

                surrounding_vehicles[vehicle_id] = surrounding_vehicle
                sorrounding_vehicles_expand[vehicle_id] = sorrounding_vehicle_expand

        for vehicle_id in surrounding_vehicles.keys():
            state['vehicle'][vehicle_id]['surroundings'] = surrounding_vehicles[vehicle_id]
            state['vehicle'][vehicle_id]['surroundings_expand'] = sorrounding_vehicles_expand[vehicle_id]
            self.surround_vehicles[vehicle_id] = surrounding_vehicles[vehicle_id]
            self.surround_vehicles_expand[vehicle_id] = sorrounding_vehicles_expand[vehicle_id]

        return state

    # ##################
    # Tools for actions
    # ##################
    def __init_actions(self, raw_state):
        """初始化所有车辆(CAV+HDV)的速度:
        1. 所有车辆的速度保持不变, (0, -1) --> 0 表示不换道, -1 表示速度不变 [(-1, -1)表示HDV不接受RL输出]
        """
        self.actions = dict()
        for _veh_id, veh_info in raw_state['vehicle'].items():
            self.actions[_veh_id] = (-1, -1, -1)

    def compute_idm_acceleration(self, v_ego, v_lead, p_ego, p_lead, params=IDM_PARAMS):
        """
        计算IDM模型的加速度
        """
        a = params['a']
        b = params['b']
        delta = params['delta']
        s0 = params['s0']
        T = params['T']
        v0 = params['v_desired']

        if v_lead != np.nan and p_lead != np.nan:
            delta_v = v_ego - v_lead
            s = p_lead - p_ego - 5

            s_star = s0 + max(0, v_ego * T + (v_ego * delta_v) / (2 * np.sqrt(a * b)))
            acceleration = a * (1 - (v_ego / v0) ** delta - (s_star / s) ** 2)
        else:
            acceleration = a * (1 - (v_ego / v0) ** delta)

        return acceleration

    def __update_actions(self, raw_action):
        """更新 ego 车辆的速度
        """
        self.actual_actions = {ego_id: [] for ego_id in self.ego_ids}
        for vehicle_id in self.surround_vehicles.keys(): # self.surround_vehicle_expand.keys():
            for surround_key in self.required_surroundings:
                if surround_key not in self.surround_vehicles[vehicle_id]:
                    self.surround_vehicles[vehicle_id][surround_key] = 0
        # leader_speed = np.sin(self.total_timesteps / 5) * 3 + 10
        step = self.vehicles_info['Leader_0_0'][0] if self.vehicles_info != {} else 0.0
        step = 0.0 if step > 20.0 else step
        for i in range(3):
            [mean_v, std_v] = self.scene_info[i]
            for j in range(self.leader_counts[i]):
                vehicle_name = f'Leader_{i}_{j}'
                if self.use_gui:
                    import traci as traci
                else:
                    import libsumo as traci
                if self.vehicles_info != {}:
                    leader = traci.vehicle.getLeader(vehicle_name, 100)
                    if leader and leader[0] in self.vehicles_info:
                        self.actions[vehicle_name] = (0, -1, -1)
                    else:
                        if j == 0:
                            next_speed = mean_v + std_v * np.sin(step * np.pi * 0.4 + i * np.pi)
                        else:
                            now_speed = self.vehicles_info[vehicle_name]['speed']
                            next_speed = now_speed + random.uniform(-1, 1) * 0.2
                        self.actions[vehicle_name] = (0, -1, next_speed)
            for j in range(10):
                vehicle_name = f'Follower_{i}_{j}'
                self.actions[vehicle_name] = (0, -1, -1)

        def map_action_to_lane(a):
            if a < -0.5:
                return 2
            elif a <= 0.5:
                return 0
            else:
                return 1
        # raw_action: lateral_action, longitudinal_action
        # excute_action: lane_change, delta_y, target_speed
        for _veh_id in raw_action:
            if _veh_id in self.actions:  # 只更新 ego vehicle 的速度
                # 更新换道
                raw_lateral_action = raw_action[_veh_id][0]
                lateral_action = raw_lateral_action
                if self.TTC_assessment[_veh_id]['right'] < 1.0:
                    lateral_action = np.clip(lateral_action, -0.5, 1.5, dtype=np.float64)
                if self.TTC_assessment[_veh_id]['left'] < 1.0:
                    lateral_action = np.clip(lateral_action, -1.5, 0.5, dtype=np.float64)
                actual_lanechange = map_action_to_lane(lateral_action)
                # IDM model
                ego_x = self.vehicles_info[_veh_id][2] if self.vehicles_info != {} else self.vehs_start[_veh_id][0][0]
                ego_v = self.vehicles_info[_veh_id][4] if self.vehicles_info != {} else self.vehs_start[_veh_id][0][1]
                ego_a = self.vehicles_info[_veh_id][5] if self.vehicles_info != {} else 0
                if self.surround_vehicles[_veh_id] != {}:
                    if self.surround_vehicles[_veh_id]['front_1'] != 0:
                        front_1 = self.surround_vehicles[_veh_id]['front_1']
                        delta_v = front_1['long_rel_v']
                        delta_s = front_1['long_dist']
                        delta_a = front_1['long_rel_a']
                        veh_lead_speed = ego_v - delta_v
                        veh_lead_pos = ego_x + delta_s + 5
                        veh_lead_acc = ego_a - delta_a
                    else:
                        veh_lead_speed = np.nan
                        veh_lead_pos = np.nan
                        veh_lead_acc = np.nan
                else:
                    veh_lead_speed = np.nan
                    veh_lead_pos = np.nan
                    veh_lead_acc = np.nan
                a_IDM = self.compute_idm_acceleration(ego_v, veh_lead_speed, ego_x, veh_lead_pos)
                # 更新速度
                control_input = np.clip(raw_action[_veh_id][1] + a_IDM, -4, 3, dtype=np.float64)
                actual_acceleration = control_input # current_acceleration + delta_acceleration
                speed_command = np.clip(self.current_speed[_veh_id] + actual_acceleration * self.delta_t, 0, 20, dtype=np.float64)
                self.actual_actions[_veh_id].append([actual_lanechange, control_input, speed_command])
                self.actions[_veh_id] = (actual_lanechange, -1, speed_command)

                self.actor_action[_veh_id].append([raw_action[_veh_id], (actual_lanechange, control_input), speed_command])

        return self.actions

    # ##########################
    # State and Reward Wrappers
    # ##########################
    def state_wrapper(self, state, sim_time):
        """对原始信息的 state 进行处理, 分别得到:
        - 车道的信息
        - ego vehicle 的属性
        """
        state = state['vehicle'].copy()  # 只需要分析车辆的信息

        # 计算车辆和地图的数据
        cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, self.TTC_assessment = analyze_traffic(
            state=state, lane_ids=self.calc_features_lane_ids, max_veh_num=self.lane_max_num_vehs,
            parameters=self.parameters, vehicles_hist=self.vehicles_hist, hist_length=self.hist_length,
            TTC_assessment=self.TTC_assessment
        )
        if self.is_evaluation:
            if sim_time==0.0:
                now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                self.output_dir = os.path.join(self.save_csv_dir, now_time)
                self.save_motion_dir = os.path.join(self.output_dir, "trajectory_csv")
                os.makedirs(self.save_motion_dir)
                self.ttc_record = {ego_id: [] for ego_id in self.ego_ids}
                self.act_record = {ego_id: [] for ego_id in self.ego_ids}
                self.crash_record = {ego_id: 0 for ego_id in self.ego_ids}
        save_motion_dir = self.save_motion_dir
        # ttc_record, act_record, crash_record, classified_scene = observation_analysis(
        #     state=state, reward_statistics=reward_statistics, flow_statistics=flow_statistics,
        #     vehicle_names=self.vehicle_names, is_evaluation=self.is_evaluation, save_csv_dir=save_motion_dir, sim_time=sim_time,
        #     ttc_hist=self.ttc_record, act_hist=self.act_record, crash_count=self.crash_record, scene_centers=self.scene_centers,
        # )
        # # Convert 'class_X' to X
        # classified_scene_mark = int(classified_scene.split('_')[1])
        # self.realtime_class.append(classified_scene_mark)
        # self.ttc_record = ttc_record
        # self.act_record = act_record
        # self.crash_record = crash_record
        start_time = time.time()

        # 计算每个 ego vehicle 的 state 拼接为向量
        if self.strategy == 'base':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = feature_selection_simple_version(
                self,
                hdv_statistics=hdv_statistics,
                ego_statistics=cav_statistics,
                lane_statistics=lane_statistics,
                flow_statistics=flow_statistics,
                reward_statistics=reward_statistics,
                evaluation=evaluation,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                node_positions=self.node_positions,
                ego_ids=self.ego_ids,
            )
        elif self.strategy == 'improve':
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = feature_selection_improved_version(
                self,
                hdv_statistics=hdv_statistics,
                ego_statistics=cav_statistics,
                lane_statistics=lane_statistics,
                flow_statistics=flow_statistics,
                reward_statistics=reward_statistics,
                evaluation=evaluation,
                unique_edges=self.edge_ids,
                edge_lane_num=self.edge_lane_num,
                node_positions=self.node_positions,
                ego_ids=self.ego_ids,
            )
        # end_time = time.time()
        # print(f'Time for feature extraction: {end_time - start_time} second')

        return feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, classified_scene_mark

    def reward_wrapper(self, lane_statistics, ego_statistics, reward_statistics, rew_weights) -> float:
        """
        根据 ego vehicle 的状态信息和 reward 的统计信息计算 reward
        我希望：
            global reward
                1. 所有车CAV+HDV都能够尽可能时间短的到达终点
                2. 所有CAV的平均速度尽可能的接近最快速度
                3.

            special reward (near bottleneck area)
                1. 在通过bottleneck的时候尽量全速通过
                2. CAV车辆尽可能的减速
                2. CAV车辆尽可能的保持距离


            local reward
                1. 每个单独的CAV尽可能不要和其他车辆碰撞
                    a. 警告距离
                    b. 碰撞距离
                2. 每个单独的CAV尽可能的保持最快速度
                3. 每个单独的CAV离开路网的奖励

        """
        max_speed = 20  # ego vehicle 的最快的速度
        max_acceleration = 3
        TTC_warning_threshold = 1.5
        TTC_collision_threshold = 0.5
        K_TTC = 20 / (TTC_warning_threshold - TTC_collision_threshold)
        T_exp = 1.5
        L_exp = 2.0

        # 先把reward_statistics中所有的车辆的信息都全局记录下来self.vehicles_info
        for veh_id, (road_id, distance, position_x, position_y, speed, acceleration, heading, waiting_time,
                     accumulated_waiting_time) in reward_statistics.items():
            self.vehicles_info[veh_id] = [
                (self.vehicles_info.get(veh_id, [0, None])[0] + self.delta_t),  # travel time
                road_id,
                distance,
                position_x,
                position_y,
                speed,
                acceleration,
                heading,
                waiting_time,
                accumulated_waiting_time
            ]
        # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
        if len(self.out_of_road) > 0:
            for veh_id in self.out_of_road:
                if veh_id in self.vehicles_info:
                    del self.vehicles_info[veh_id]

        # ######################### 开始计算reward  # #########################
        inidividual_rew_ego = {key: {} for key in list(set(self.ego_ids) - set(self.out_of_road))}

        # ######################## 初始化 for group reward ########################
        all_ego_vehicle_speed = []  # CAV车辆的平均速度 - 使用target speed
        all_ego_vehicle_mean_speed = []  # CAV车辆的累积平均速度 - 使用速度/时间
        all_ego_vehicle_acceleration = []  # CAV车辆的平均加速度
        all_ego_vehicle_accumulated_waiting_time = []  # # CAV车辆的累积平均等待时间
        all_ego_vehicle_waiting_time = []  # CAV车辆的等待时间

        all_vehicle_speed = []  # CAV和HDV车辆的平均速度 - 使用target speed
        all_vehicle_mean_speed = []  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        all_vehicle_acceleration = []  # CAV和HDV车辆的平均加速度
        all_vehicle_accumulated_waiting_time = []  # CAV和HDV车辆的累积平均等待时间
        all_vehicle_waiting_time = []  # CAV和HDV车辆的等待时间

        for veh_id, (veh_travel_time, road_id, distance, position_x,
                     position_y, speed, acceleration, heading,
                     waiting_time, accumulated_waiting_time) in list(self.vehicles_info.items()):

            # CAV和HDV车辆的
            all_vehicle_speed.append(speed)
            all_vehicle_mean_speed.append(distance / veh_travel_time)
            all_vehicle_acceleration.append(abs(acceleration))
            all_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
            all_vehicle_waiting_time.append(waiting_time)

            # 把CAV单独取出来
            if veh_id in self.ego_ids:

                # for group reward 计算CAV车辆的累积平均速度
                all_ego_vehicle_speed.append(speed)
                all_ego_vehicle_mean_speed.append(distance / veh_travel_time)
                all_ego_vehicle_acceleration.append(abs(acceleration))
                all_ego_vehicle_accumulated_waiting_time.append(accumulated_waiting_time)
                all_ego_vehicle_waiting_time.append(waiting_time)

                # ######################## for individual reward ########################
                # # CAV车辆的累积平均速度越靠近最大速度，reward越高 - [0, 5]
                # individual_speed_r = -abs(distance / veh_travel_time - max_speed) / max_speed * 5 + 5
                # inidividual_rew_ego[veh_id] += 1 * individual_speed_r

                # CAV车辆的target速度越靠近最大速度，reward越高 - [0, 1]
                individual_speed_r_simple = -abs(speed - max_speed) / max_speed * 1 + 1
                # individual_speed_r_simple = speed / max_speed
                inidividual_rew_ego[veh_id]['efficiency'] = individual_speed_r_simple
                # CAV车辆的加速度绝对值越小，reward越高 - [-1, 0]
                # individual_acceleration_r = -(acceleration ** 2) * 0.5
                individual_acceleration_r = -((acceleration / max_acceleration) ** 2)  # + 1
                # acc_value = acceleration ** 2
                # individual_acceleration_r = np.exp(-acc_value) # + 1
                inidividual_rew_ego[veh_id]['comfort'] = individual_acceleration_r

                # 警告距离和碰撞距离, stability-related values
                max_TTC = 20.0
                if 'front' not in ego_statistics[veh_id][6].keys():
                    TTC = 20.0
                    s_act = L_exp
                    v_act = 0
                    delta_v = 0
                    DRAC = 0
                else:
                    front_info = ego_statistics[veh_id][6]['front']
                    TTC = front_info[7]
                    s_act = front_info[2]
                    v_act = ego_statistics[veh_id][1]
                    delta_v = -front_info[4]
                    DRAC = front_info[8]
                # stability reward
                delta_s = s_act - (v_act * T_exp + L_exp)
                stability_matric = np.array([delta_s, delta_v])
                stability_weight = np.array([[1, 0], [0, 0.5]])
                e_ss = stability_matric @ stability_weight @ stability_matric.T
                inidividual_rew_ego[veh_id]['stability'] = np.exp(-e_ss)

                if TTC_collision_threshold < TTC and TTC <= TTC_warning_threshold:
                    safety_r = (1 / (np.exp(-(K_TTC * (TTC - (TTC_collision_threshold + TTC_warning_threshold)/2))) + 1)) - 1
                elif TTC <= TTC_collision_threshold:
                    safety_r = -1
                else:
                    safety_r = 0
                # safety_r = min(TTC, max_TTC) / max_TTC
                # if TTC <= TTC_collision_threshold:
                #     safety_r = safety_r-4
                inidividual_rew_ego[veh_id]['safety'] = safety_r  # [-1, 1]
        all_rew_ego = inidividual_rew_ego.copy()
        rew_safety = {key: inidividual_rew_ego[key]['safety'] for key in inidividual_rew_ego}
        rew_stability = {key: inidividual_rew_ego[key]['stability'] for key in inidividual_rew_ego}
        rew_efficiency = {key: inidividual_rew_ego[key]['efficiency'] for key in inidividual_rew_ego}
        rew_comfort = {key: inidividual_rew_ego[key]['comfort'] for key in inidividual_rew_ego}

        # 计算全局reward
        all_ego_vehicle_speed = np.mean(all_ego_vehicle_speed)  # CAV车辆的平均速度 - 使用target speed
        # all_ego_mean_speed = np.mean(all_ego_vehicle_mean_speed)  # CAV车辆的累积平均速度 - 使用速度/时间
        all_ego_vehicle_acceleration = np.mean(all_ego_vehicle_acceleration)  # CAV车辆的平均加速度

        all_vehicle_speed = np.mean(all_vehicle_speed)  # CAV和HDV车辆的平均速度 - 使用target speed
        # all_vehicle_mean_speed = np.mean(all_vehicle_mean_speed)  # CAV和HDV车辆的累积平均速度 - 使用速度/时间
        all_vehicle_acceleration = np.mean(all_vehicle_acceleration)  # CAV和HDV车辆的平均加速度
        global_ego_speed_r = -abs(all_ego_vehicle_speed - max_speed) / max_speed * 1 + 1  # [0, 5]
        # global_ego_speed_r = all_ego_vehicle_speed / max_speed
        # global_ego_mean_speed_r = -abs(all_ego_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_ego_acceleration_r = -abs(all_ego_vehicle_acceleration) + 6  # [0, 6]
        global_ego_acceleration_r = -((all_ego_vehicle_acceleration/max_acceleration) ** 2) # + 1
        # acc_value = all_ego_vehicle_acceleration ** 2
        # global_ego_acceleration_r = np.exp(-acc_value)  # + 1

        # global_all_speed_r = -abs(all_vehicle_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_mean_speed_r = -abs(all_vehicle_mean_speed - max_speed) / max_speed * 5 + 5  # [0, 5]
        # global_all_acceleration_r = -abs(all_vehicle_acceleration) + 6   # [0, 6]
        for veh_id in all_rew_ego.keys():
            all_rew_ego[veh_id]['global_efficiency'] = global_ego_speed_r
            all_rew_ego[veh_id]['global_comfort'] = global_ego_acceleration_r

        time_penalty = 0
        # weight = {'efficiency': 1, 'comfort': 1, 'safety': 10, 'stability': 1}
        weight = rew_weights

        rewards = {key: weight['safety'] * inidividual_rew_ego[key]['safety'] \
                        + weight['stability'] * inidividual_rew_ego[key]['stability'] \
                        + weight['efficiency'] * (1-self.CAV_penetration) * inidividual_rew_ego[key]['efficiency'] \
                        + weight['comfort'] * (1-self.CAV_penetration) * inidividual_rew_ego[key]['comfort'] \
                        # + time_penalty_ego[key] \
                        + weight['efficiency'] * self.CAV_penetration * all_rew_ego[key]['global_efficiency'] \
                        # + global_ego_mean_speed_r \
                        + weight['comfort'] * self.CAV_penetration * all_rew_ego[key]['global_comfort'] \
                        # + global_ego_waiting_time_r \
                        # + global_ego_accumulated_waiting_time_r \
                        # + global_all_speed_r \
                        # + global_all_mean_speed_r \
                        # + global_all_acceleration_r \
                        + time_penalty
                   for key in inidividual_rew_ego}

        return rewards, all_vehicle_speed, all_vehicle_acceleration, rew_safety, rew_stability, rew_efficiency, rew_comfort
    # ############
    # Collision
    # #############

    def check_collisions(self, init_state):

        ################# 碰撞检查 ###########################################
        # 简单版本 - 根据车头的两两位置计算是否碰撞
        collisions_head_vehs, collisions_head_info = check_collisions_based_pos(init_state['vehicle'],
                                                                                gap_threshold=GAP_THRESHOLD)

        # print('point to point collision:', collisions_head_vehs, collisions_head_info)

        # 稍微复杂的版本 - 根据neighbour位置计算是否碰撞
        collisions_neigh_vehs, warn_neigh_vehs, collisions_neigh_info = check_collisions(init_state['vehicle'],
                                                                                         self.ego_ids,
                                                                                         gap_threshold=GAP_THRESHOLD,
                                                                                         gap_warn_collision=WARN_GAP_THRESHOLD
                                                                                         # 给reward的警告距离
                                                                                         )
        # print('neighbour collision:', collisions_neigh_vehs, collisions__neigh_info)

        collisions_for_reward = {
            'collision': collisions_head_vehs + collisions_neigh_vehs,
            'warn': warn_neigh_vehs,
            'info': collisions_neigh_info + collisions_head_info
        }

        self.warn_ego_ids = {}
        self.coll_ego_ids = {}

        for key, value in collisions_for_reward.items():
            if key == 'warn' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'warn' in element:
                        if not element['CAV_key'] in self.warn_ego_ids:
                            self.warn_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            # append the distance
                            self.warn_ego_ids[element['CAV_key']].append(element['distance'])

            if key == 'collision' and len(value) != 0:
                for element in collisions_for_reward['info']:
                    if 'collision' in element:
                        if not element['CAV_key'] in self.coll_ego_ids:
                            self.coll_ego_ids.update({element['CAV_key']: [element['distance']]})
                        else:
                            self.coll_ego_ids[element['CAV_key']].append(element['distance'])

    # ############
    # reset & step
    # #############

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化
        """
        # 初始化超参数
        # bottleneck 处的拥堵程度 # TODO: 根据lane statastics来计算
        self.congestion_level = 0
        # 记录仿真内所有车辆的信息 - 在reward wrapper中更新
        self.vehicles_info = {}
        # 记录行驶出路网的车辆
        self.out_of_road = []
        # 假设这些车初始化都在路网上 活着
        self.agent_mask = {ego_id: True for ego_id in self.ego_ids}
        self.current_speed = {key: 10 for key in self.ego_ids}

        # 初始化环境
        init_state = self.env.reset()
        leader_0_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{self.scene_length}s/{self.scene_definition[self.scene_class_id][0][0]}_{self.scene_definition[self.scene_class_id][0][1]}/vehicle_0.csv'
        leader_1_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{self.scene_length}s/{self.scene_definition[self.scene_class_id][1][0]}_{self.scene_definition[self.scene_class_id][1][1]}/vehicle_1.csv'
        leader_2_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{self.scene_length}s/{self.scene_definition[self.scene_class_id][2][0]}_{self.scene_definition[self.scene_class_id][2][1]}/vehicle_2.csv'
        leaders_path = [leader_0_path, leader_1_path, leader_2_path]

        self.vehs_start, self.current_speed, self.vehicle_names, self.leader_counts = generate_scenario_NGSIM(
            config_file=self.config_file,
            aggressive=self.aggressive, cautious=self.cautious, normal=self.normal,
            use_gui=self.use_gui, sce_name=self.name_scenario,
            CAV_num=self.num_CAVs, HDV_num=self.num_HDVs, CAV_penetration=self.CAV_penetration,
            distribution="uniform", scene_info=self.scene_info, leaders_path=leaders_path)

        # if 0 <= self.total_timesteps < 1000000:
        #     assert self.num_CAVs == 5
        #     assert self.CAV_penetration == 0.5
        #     # 生成车流
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")  # generate_scene_MTF.py - "random" or "uniform" distribution
        #
        # elif 1000000 <= self.total_timesteps < 2000000:
        #     self.num_CAVs = 5
        #     self.CAV_penetration = 0.3
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")
        # elif 2000000 <= self.total_timesteps <= 3000000:
        #     self.num_CAVs = 5
        #     self.CAV_penetration = 0.1
        #     generate_scenario(use_gui=self.use_gui, sce_name=self.name_scenario,
        #                       CAV_num=self.num_CAVs, CAV_penetration=self.CAV_penetration,
        #                       distribution="uniform")

        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 对于warmup step = 0也适用
        for _ in range(self.warmup_steps + 1):
            init_state, _, _, _, _ = super().step(self.actions)
            init_state = self.append_surrounding(init_state)

            # 检查是否有碰撞
            collisions, warnings, crash_infos = crash_evaluation(init_state['vehicle'], self.ego_ids, self.parameters)
            assert len(collisions) == 0, f'Collision with {collisions} at reset!!! Regenerate the flow'
            assert len(warnings) == 0, f'Warning with {warnings} at reset!!! Regenerate the flow'

            # collisions_vehs, warn_vehs, collision_infos = check_collisions(init_state['vehicle'],
            #                                                                self.ego_ids,
            #                                                                gap_threshold=GAP_THRESHOLD,
            #                                                                gap_warn_collision=WARN_GAP_THRESHOLD)
            # # reset 时不应该有碰撞
            # assert len(collisions_vehs) == 0, f'Collision with {collisions_vehs} at reset!!! Regenerate the flow'
            # assert len(warn_vehs) == 0, f'Warning with {warn_vehs} at reset!!! Regenerate the flow'

            # 对 state 进行处理
            now_time = 0.0
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, classified_scene_mark = self.state_wrapper(state=init_state, sim_time=now_time)
            # shared_feature_vectors = compute_centralized_vehicle_features(lane_statistics,
            #                                                               feature_vectors,
            #                                                               self.bottle_neck_positions)
            actor_features, actor_features_flatten, shared_features, shared_features_flatten = compute_centralized_vehicle_features_hierarchical_version(
                self.obs_size, self.shared_obs_size,
                lane_statistics,
                feature_vectors_current, feature_vectors_current_flatten,
                feature_vectors, feature_vectors_flatten, self.ego_ids)
            self.__init_actions(raw_state=init_state)

        return feature_vectors_flatten, shared_features_flatten, {'step_time': self.warmup_steps + 1}, classified_scene_mark

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        """
        self.total_timesteps += 1

        # 已经死了的车辆不控制 - 从 action 中删除
        for ego_id, ego_live in self.agent_mask.items():
            if not ego_live:
                del action[ego_id]

        # 更新 action
        action = self.__update_actions(raw_action=action).copy()

        # 在环境里走一步
        init_state, rewards, truncated, _dones, infos = super().step(action)
        init_state = self.append_surrounding(init_state)
        now_time = infos['step_time'] - 0.1
        self.current_speed = {key: init_state['vehicle'][key]['speed'] if key in init_state['vehicle'] else 0 for key in
                              self.ego_ids}
        self.current_lane = {key: init_state['vehicle'][key]['lane_id'] if key in init_state['vehicle'] else 0 for key
                             in self.ego_ids}

        ################# 碰撞检查 ###########################################
        self.check_collisions(init_state)
        ####################################################################

        # 对 state 进行处理 (feature_vectors的长度是没有行驶出CAV的数量)
        feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, classified_scene_mark = self.state_wrapper(
            state=init_state, sim_time=now_time)

        # 处理离开路网的车辆 agent_mask 和 out_of_road
        for _ego_id in self.ego_ids:
            if _ego_id not in feature_vectors:
                assert _ego_id not in cav_statistics, f'ego vehicle {_ego_id} should not be in ego_statistics'
                assert _ego_id not in reward_statistics, f'ego vehicle {_ego_id} should not be in reward_statistics'
                self.agent_mask[_ego_id] = False  # agent 离开了路网, mask 设置为 False
                if _ego_id not in self.out_of_road:
                    self.out_of_road.append(_ego_id)

        # 初始化车辆的速度
        self.__init_actions(raw_state=init_state)

        # 处理 dones 和 infos
        if len(self.coll_ego_ids) == 0 and len(feature_vectors) > 0:  # 还有车在路上 且还没有碰撞发生
            # 计算此时的reward （这里的reward只有还在路网上的车的reward）
            # rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics, self.rew_weights)

            # ###### 使用动态奖励函数计算分量奖励和总奖励`
            # reward_dict, evaluation_dict = dynamic_reward_computation(self, self.ego_ids, reward_infos, reward_statistics, flow_infos)

            # 使用分层奖励计算替换原有的dynamic_reward_computation
            reward_dict, evaluation_dict = hierarchical_reward_computation(
                self, self.ego_ids, reward_statistics, flow_statistics, evaluation)

            # 获取各奖励分量
            rewards = {}
            rew_safety = {}
            rew_efficiency = {}
            rew_stability = {}
            rew_comfort = {}

            for ego_id in self.ego_ids:
                rewards[ego_id] = reward_dict[ego_id]['sum_reward']
                rew_safety[ego_id] = reward_dict[ego_id]['global_safety']
                rew_efficiency[ego_id] = reward_dict[ego_id]['global_efficiency']
                rew_stability[ego_id] = reward_dict[ego_id]['global_stability']
                rew_comfort[ego_id] = reward_dict[ego_id]['global_comfort']

            # 提取平均速度和加速度
            mean_speeds = flow_statistics['mean_speed']
            mean_accelerations = flow_statistics['mean_acceleration']

            safety_SI = evaluation_dict['safety_SI']
            efficiency_ASR = evaluation_dict['efficiency_ASR']
            efficiency_TFR = evaluation_dict['efficiency_TFR']
            stability_SSI = 0.5 * evaluation_dict['stability_SSVD'] + 0.5 * evaluation_dict['stability_SSSD']
            comfort_AI = evaluation_dict['comfort_AI']
            comfort_JI = evaluation_dict['comfort_JI']
            control_efficiency = evaluation_dict['control_efficiency']
            control_reward = evaluation_dict['control_reward']
            comfort_cost = evaluation_dict['comfort_cost']
            comfort_reward = evaluation_dict['comfort_reward']

            if self.is_evaluation and now_time==49.9:
                evaluation_results = evaluate_method(
                    parameters=self.parameters,
                    eval_weights=self.eval_weights,
                    method_name=self.save_model_name,
                    output_dir=self.output_dir,
                    CAV_num=self.num_CAVs,
                    HDV_num=self.num_HDVs,
                    CAV_penetration=self.CAV_penetration,
                    vehicle_names=self.vehicle_names,
                    desired_velocity=self.parameters["max_v"],
                    ttc_record=self.ttc_record,
                    act_record=self.act_record,
                    crash_record=self.crash_record,
                )

            ###### dynamio reward definition is ended here

        elif len(self.coll_ego_ids) > 0 and len(feature_vectors) > 0:  # 还有车在路上 但有车辆碰撞
            # 计算此时的reward
            # rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics, self.rew_weights)  # 更新 veh info

            # ###### 使用动态奖励函数计算分量奖励和总奖励
            # reward_dict, evaluation_dict = dynamic_reward_computation(self, self.ego_ids, reward_infos, reward_statistics, flow_infos)

            # 使用分层奖励计算替换原有的dynamic_reward_computation
            reward_dict, evaluation_dict = hierarchical_reward_computation(
                self, self.ego_ids, reward_statistics, flow_statistics, evaluation)

            # 获取各奖励分量
            rewards = {}
            rew_safety = {}
            rew_efficiency = {}
            rew_stability = {}
            rew_comfort = {}

            for ego_id in self.ego_ids:
                rewards[ego_id] = reward_dict[ego_id]['sum_reward']
                rew_safety[ego_id] = reward_dict[ego_id]['global_safety']
                rew_efficiency[ego_id] = reward_dict[ego_id]['global_efficiency']
                rew_stability[ego_id] = reward_dict[ego_id]['global_stability']
                rew_comfort[ego_id] = reward_dict[ego_id]['global_comfort']

            # 提取平均速度和加速度
            mean_speeds = flow_statistics['mean_speed']
            mean_accelerations = flow_statistics['mean_acceleration']

            safety_SI = evaluation_dict['safety_SI']
            efficiency_ASR = evaluation_dict['efficiency_ASR']
            efficiency_TFR = evaluation_dict['efficiency_TFR']
            stability_SSI = 0.5 * evaluation_dict['stability_SSVD'] + 0.5 * evaluation_dict['stability_SSSD']
            comfort_AI = evaluation_dict['comfort_AI']
            comfort_JI = evaluation_dict['comfort_JI']
            control_efficiency = evaluation_dict['control_efficiency']
            control_reward = evaluation_dict['control_reward']
            comfort_cost = evaluation_dict['comfort_cost']
            comfort_reward = evaluation_dict['comfort_reward']

            ###### dynamio reward definition is ended here

            for collid_ego_id in self.coll_ego_ids:
                infos['collision'].append(collid_ego_id)
                self.agent_mask[collid_ego_id] = False
            infos['done_reason'] = 'collision'

            if self.is_evaluation:
                evaluation_results = evaluate_method(
                    parameters=self.parameters,
                    eval_weights=self.eval_weights,
                    method_name=self.save_model_name,
                    output_dir=self.output_dir,
                    CAV_num=self.num_CAVs,
                    HDV_num=self.num_HDVs,
                    CAV_penetration=self.CAV_penetration,
                    vehicle_names=self.vehicle_names,
                    desired_velocity=self.parameters["max_v"],
                    ttc_record=self.ttc_record,
                    act_record=self.act_record,
                    crash_record=self.crash_record,
                )

        else:  # 所有RL车离开的时候, 就结束
            assert len(feature_vectors) == 0, f'All RL vehicles should leave the environment'
            infos['done_reason'] = 'all CAV vehicles leave the environment'
            # 全局记录下来self.vehicles_info里面不应该包含已经离开的车辆
            init_state, rew, truncated, _d, _ = super().step(self.actions)
            init_state = self.append_surrounding(init_state)
            feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, classified_scene_mark = self.state_wrapper(
                state=init_state, sim_time=now_time)

            if self.is_evaluation:
                evaluation_results = evaluate_method(
                    parameters=self.parameters,
                    eval_weights=self.eval_weights,
                    method_name=self.save_model_name,
                    output_dir=self.output_dir,
                    CAV_num=self.num_CAVs,
                    HDV_num=self.num_HDVs,
                    CAV_penetration=self.CAV_penetration,
                    vehicle_names=self.vehicle_names,
                    desired_velocity=self.parameters["max_v"],
                    ttc_record=self.ttc_record,
                    act_record=self.act_record,
                    crash_record=self.crash_record,
                )

            self.__init_actions(raw_state=init_state)

            while len(reward_statistics) > 0:
                init_state, rew, truncated, _d, _ = super().step(self.actions)
                init_state = self.append_surrounding(init_state)
                feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten, cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, classified_scene_mark = self.state_wrapper(
                    state=init_state, sim_time=now_time)
                self.__init_actions(raw_state=init_state)
                # rewards = self.reward_wrapper(lane_statistics, ego_statistics, reward_statistics)  # 更新 veh info

            infos['out_of_road'] = self.ego_ids
            assert set(self.out_of_road) == set(self.ego_ids), f'All RL vehicles should leave the environment'
            rewards = {key: 20.0 for key in self.ego_ids}
            for out_of_road_ego_id in self.out_of_road:
                self.agent_mask[out_of_road_ego_id] = False
                feature_vectors_flatten[out_of_road_ego_id] = np.zeros(self.obs_size)

        # 处理以下reward
        if len(self.out_of_road) > 0 and len(feature_vectors) > 0:
            for out_of_road_ego_id in self.out_of_road:
                rewards.update({out_of_road_ego_id: 0.0})  # 离开路网之后 reward 也是 0  # TODO: 注意一下dead mask MARL
                if out_of_road_ego_id not in infos['out_of_road']:
                    infos['out_of_road'].append(out_of_road_ego_id)
                self.agent_mask[out_of_road_ego_id] = False
                feature_vectors_flatten[out_of_road_ego_id] = np.zeros(self.obs_size)

        # 获取shared_feature_vectors
        # shared_feature_vectors = compute_centralized_vehicle_features(lane_statistics,
        #                                                               feature_vectors,
        #                                                               self.bottle_neck_positions)
        actor_features, actor_features_flatten, shared_features, shared_features_flatten = compute_centralized_vehicle_features_hierarchical_version(
            self.obs_size,
            self.shared_obs_size,
            lane_statistics,
            feature_vectors_current,
            feature_vectors_current_flatten,
            feature_vectors,
            feature_vectors_flatten,
            self.ego_ids)
        # 处理以下 infos
        if len(self.warn_ego_ids) > 0:
            infos['warning'].append(self.warn_ego_ids)

        # 处理以下done
        dones = {}
        for _ego_id in self.ego_ids:
            dones[_ego_id] = not self.agent_mask[_ego_id]

        # 只要有一个车辆碰撞，就结束所有车辆的仿真
        if len(self.coll_ego_ids) > 0:
            for ego_id in self.ego_ids:
                dones[ego_id] = True

        # # 只要有一个车辆碰撞，不要结束所有车辆的仿真
        # if len(self.coll_ego_ids) > 0:
        #     for ego_id in self.coll_ego_ids:
        #         dones[ego_id] = True

        # 超出时间 结束仿真
        if infos['step_time'] >= self.max_num_seconds:
            for ego_id in self.ego_ids:
                dones[ego_id] = True
                infos['done_reason'] = 'time out'

        # # For DEBUG render
        # infos['done'] = dones.copy()
        # infos['reward'] = rewards.copy()
        # print(infos)
        # debug = []
        # for key, value in feature_vectors.items():
        #     debug.append([key, value[0] * 15])
        # print(debug)

        # TODO: 完成时间越短，reward越高 - [0, 5]

        # # check if all element in feature_vectors_flatten have same length
        # if len(shared_features_flatten) != 5:
        #     print('break')
        # for value in shared_features_flatten.values():
        #     if len(value) != 253 or (not isinstance(value, np.ndarray)):
        #         print('break')

        return feature_vectors_flatten, shared_features_flatten, rewards, mean_speeds, mean_accelerations, rew_safety, rew_stability, rew_efficiency, rew_comfort, \
            safety_SI, efficiency_ASR, efficiency_TFR, stability_SSI, comfort_AI, comfort_JI, control_efficiency, control_reward, comfort_cost, comfort_reward, dones.copy(), dones.copy(), infos, classified_scene_mark

    def close(self) -> None:
        return super().close()

