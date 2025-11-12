import yaml
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.check_folder import check_folder
from tshub.utils.init_log import set_logger
from tshub.tshub_env.tshub_env import TshubEnvironment
from harl.envs.a_multi_lane.env_utils.generate_scene_NGSIM import generate_scenario_NGSIM
import random
import time
import os
import csv
import traci
import pandas as pd
import numpy as np

from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, LateralDecision
)
from harl.envs.a_multi_lane.hierarchical_controller.get_traffic_state import get_traffic_state, get_surrounding_state
from harl.envs.a_multi_lane.hierarchical_controller.scene_recognition import SceneRecognitionModule
from harl.envs.a_multi_lane.hierarchical_controller.lateral_decision import LateralDecisionModule
from harl.envs.a_multi_lane.hierarchical_controller.conflict_assessment_controller import ConflictAssessmentController
from harl.envs.a_multi_lane.hierarchical_controller.mpc_cooperative_controller import (
    MPCCooperativeController, MPCParameters
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

def get_target_speed(pre_decision, surrounding_vehicles):
    if pre_decision == 'pre_change_left':
        return surrounding_vehicles['left_leader_1']['veh_vel']
    elif pre_decision == 'pre_change_right':
        return surrounding_vehicles['right_leader_1']['veh_vel']
    else:
        return surrounding_vehicles['ego_leader_1']['veh_vel']

DECISION_STR = {
    0: 'keep_lane',
    -1: 'pre_change_left',
    1: 'pre_change_right',
}

if __name__ == '__main__':
    path_convert = get_abs_path(__file__)
    set_logger(path_convert('./'))

    sumo_cfg = path_convert("env_utils/SUMO_files/scenario.sumocfg")
    net_file = path_convert("env_utils/SUMO_files/veh.net.xml")
    # 创建存储图像的文件夹
    image_save_folder = path_convert('./global/')
    check_folder(image_save_folder)
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    csv_save_folder = path_convert('./motion/' + now_time + '/')
    check_folder(csv_save_folder)

    yaml_file = "/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/a_multi_lane.yaml"
    with open(yaml_file, "r", encoding="utf-8") as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    # build a memory block for the surrounding_info
    observed_vehicles = []
    aggressive = args["aggressive"]
    cautious = args["cautious"]
    normal = args["normal"]
    CAV_num = args["num_CAVs"]
    HDV_num = args["num_HDVs"]
    lane_veh_num = int((CAV_num + HDV_num) / 3)
    CAV_penetration = args["penetration_CAV"]
    scene_length = args["scene_length"]
    use_gui = args["use_gui"]
    scene_definition = args["scene_definition"]
    scene_specify = args["scene_specify"]
    scene_id = args["scene_id"]
    lane_ids = args["calc_features_lane_ids"]

    # 初始化环境
    tshub_env = TshubEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        is_map_builder_initialized=True,
        is_vehicle_builder_initialized=True,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=False,
        # vehicle builder
        vehicle_action_type='vehicle_continuous_action',  # 'vehicle_continuous_action' 'lane_continuous_speed'
        use_gui=True, num_seconds=scene_length
    )
    # 初始化系统参数-协同控制策略使用
    system_params = SystemParameters()
    # 创建场景识别模块
    scene_recognition = SceneRecognitionModule(system_params)
    # 创建横向决策模块
    lateral_decision = LateralDecisionModule(system_params)
    # 创建冲突评估控制器
    conflict_controller = ConflictAssessmentController(system_params)
    # 初始化MPC协同控制器
    mpc_params = MPCParameters(
        prediction_horizon=20,  # 适当减少预测时域提高实时性
        control_horizon=10,  # 控制时域
        sampling_time=0.1,  # 与仿真步长一致
    )
    mpc_controller = MPCCooperativeController(mpc_params, system_params)

    # 开始仿真
    obs = tshub_env.reset()

    CAVs_id = []
    for i in range(CAV_num):
        CAVs_id.append(f'CAV_{i}')
    HDVs_id = []
    for i in range(HDV_num):
        HDVs_id.append(f'HDV_{i}')

    config_file = "/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/vehicle_config.json"

    # scene_class_id = random.randint(0, 6)
    if scene_specify:
        scene_class_id = scene_id
    else:
        scene_class_id = random.randint(0, 11)
    scene_info = scene_definition[scene_class_id]
    leader_0_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{scene_length}s/{scene_definition[scene_class_id][0][0]}_{scene_definition[scene_class_id][0][1]}/vehicle_0.csv'
    leader_1_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{scene_length}s/{scene_definition[scene_class_id][1][0]}_{scene_definition[scene_class_id][1][1]}/vehicle_1.csv'
    leader_2_path = f'/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/env_utils/NGSIM_trajectories/time_window_{scene_length}s/{scene_definition[scene_class_id][2][0]}_{scene_definition[scene_class_id][2][1]}/vehicle_2.csv'
    leaders_path = [leader_0_path, leader_1_path, leader_2_path]
    leader_0_df = pd.read_csv(leader_0_path)
    leader_1_df = pd.read_csv(leader_1_path)
    leader_2_df = pd.read_csv(leader_2_path)
    leaders_df = {0: leader_0_df, 1: leader_1_df, 2: leader_2_df}

    quene_start, start_speed, vehicle_names, leader_counts = generate_scenario_NGSIM(
        config_file=config_file,
        aggressive=aggressive, cautious=cautious, normal=normal,
        use_gui=use_gui, sce_name="Env_MultiLane",
        CAV_num=CAV_num, HDV_num=HDV_num,
        CAV_penetration=CAV_penetration, distribution="uniform",
        scene_info=scene_info,
        leaders_path=leaders_path,
    )

    # TODO: check HDV_ids and vehicle_names
    # TODO: Leader does not belong to HDV_ids

    # Initialize variables for simulation
    done = False
    last_acceleration = {}
    now_acceleration = {}
    n_communication = 3
    sampling_time = 1.0

    # For debug: track crashes and near-misses
    crash_count = 0
    near_miss_count = 0
    s_crash_count = 0
    ttc_crash_count = 0
    act_crash_count = 0

    # Safety metrics tracking
    safety_violations = {veh_id: 0 for veh_id in CAVs_id}
    ttc_history = {veh_id: [] for veh_id in CAVs_id}
    act_history = {veh_id: [] for veh_id in CAVs_id}
    min_ttc_values = {veh_id: float('inf') for veh_id in CAVs_id}
    min_gap_values = {veh_id: float('inf') for veh_id in CAVs_id}
    min_act_values = {veh_id: float('inf') for veh_id in CAVs_id}

    # Main simulation loop
    while not done:
        # Initialize actions dictionary
        actions = {'vehicle': {veh_id: (-1, -1, -1) for veh_id in obs['vehicle'].keys()}}

        if obs['vehicle']:
            # Get current simulation time
            step = info['step_time']
            traffic_state = get_traffic_state(obs['vehicle'], lane_ids)
            # 执行场景识别
            scene_result = scene_recognition.recognize_scenario(traffic_state)
            new_weight = scene_result['weight_config']

            for i in range(3):
                [mean_v, std_v] = scene_info[i]
                for j in range(leader_counts[i]):
                    vehicle_name = f'Leader_{i}_{j}'
                    leader = traci.vehicle.getLeader(vehicle_name, 100)
                    if leader and leader[0] in obs['vehicle']:
                        actions['vehicle'][vehicle_name] = (0, -1, -1)
                    else:
                        if j == 0:
                            next_speed = leaders_df[i].loc[step, 'Velocity']
                        else:
                            now_speed = obs['vehicle'][vehicle_name]['speed']
                            next_speed = now_speed + random.uniform(-1, 1)
                        actions['vehicle'][vehicle_name] = (0, -1, next_speed)

                for j in range(10):
                    vehicle_name = f'Follower_{i}_{j}'
                    actions['vehicle'][vehicle_name] = (0, -1, -1)

            # Control CAVs
            for i_CAV in range(CAV_num):
                crash_mark = 0
                veh_name = f'CAV_{i_CAV}'
                if veh_name in obs['vehicle']:
                    # Get ego vehicle state
                    veh_pos_x = obs['vehicle'][veh_name]['position'][0]
                    veh_pos_y = obs['vehicle'][veh_name]['position'][1]
                    veh_speed = obs['vehicle'][veh_name]['speed']
                    veh_heading = obs['vehicle'][veh_name]['heading']
                    veh_acc = obs['vehicle'][veh_name]['acceleration']
                    veh_yaw_rate = 0
                    lane_index = obs['vehicle'][veh_name]['lane_index']
                    veh_type = 1

                    ego_lane_v = traffic_state['lane_velocities'][lane_index]
                    ego_lane_density = traffic_state['lane_densities'][lane_index]
                    if lane_index > 0:
                        right_lane_v = traffic_state['lane_velocities'][lane_index-1]
                        right_lane_density = traffic_state['lane_densities'][lane_index-1]
                    else:
                        right_lane_v = None
                        right_lane_density = None
                    if lane_index < 2:
                        left_lane_v = traffic_state['lane_velocities'][lane_index+1]
                        left_lane_density = traffic_state['lane_densities'][lane_index+1]
                    else:
                        left_lane_v = None
                        left_lane_density = None

                    surrounding_vehicles, surrounding_state = get_surrounding_state(obs['vehicle'], veh_name, use_gui)
                    if len(surrounding_vehicles) != 8 and len(surrounding_vehicles) != 12:
                        print('surrounding_vehicles is not a multiple of 12')

                    front_id = surrounding_vehicles['ego_leader_1']['veh_id']
                    front_v = obs['vehicle'][front_id]['speed']
                    front_pos_x = obs['vehicle'][front_id]['position'][0]
                    front_spacing = front_pos_x - veh_pos_x - 5
                    ego_state = format_conversion(veh_pos_x, veh_pos_y, veh_speed, veh_heading, veh_acc, veh_yaw_rate, lane_index, veh_type, front_v, front_spacing)

                    observation_stats = {
                        'left_lane_speeds': [],
                        'right_lane_speeds': [],
                        'ego_lane_speeds': [],
                    }
                    for sur_id in surrounding_vehicles.keys():
                        veh_id = surrounding_vehicles[sur_id]['veh_id']
                        if sur_id[:4] == 'left':
                            observation_stats['left_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])
                        elif sur_id[:5] == 'right':
                            observation_stats['right_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])
                        else:
                            observation_stats['ego_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])
                    if observation_stats['left_lane_speeds'] != []:
                        obs_left_lane_v = np.mean(observation_stats['left_lane_speeds'])
                    else:
                        obs_left_lane_v = None
                    if observation_stats['right_lane_speeds'] != []:
                        obs_right_lane_v = np.mean(observation_stats['right_lane_speeds'])
                    else:
                        obs_right_lane_v = None
                    if observation_stats['ego_lane_speeds'] != []:
                        obs_ego_lane_v = np.mean(observation_stats['ego_lane_speeds'])
                    else:
                        obs_ego_lane_v = veh_speed

                    # now_traffic_state = {
                    #     'v_left': left_lane_v,  # 左侧车道平均速度--来自全局信息
                    #     'v_ego': ego_lane_v,  # 自身车道平均速度--来自全局信息
                    #     'v_right': right_lane_v,  # 右侧车道平均速度--来自全局信息
                    #     'density_left': left_lane_density,  # 左侧车道交通密度
                    #     'density_ego': ego_lane_density,  # 自身车道交通密度
                    #     'density_right': right_lane_density  # 右侧车道交通密度
                    # }

                    now_traffic_state = {
                        'v_left': obs_left_lane_v,  # 左侧车道邻接车辆平均速度--来自局部信息
                        'v_ego': obs_ego_lane_v,  # 自身车道邻接车辆平均速度--来自局部信息
                        'v_right': obs_right_lane_v,  # 右侧车道邻接车辆平均速度--来自局部信息
                        'density_left': left_lane_density,  # 左侧车道交通密度
                        'density_ego': ego_lane_density,  # 自身车道交通密度
                        'density_right': right_lane_density  # 右侧车道交通密度
                    }
                    necessity_results = lateral_decision.evaluate_lane_change_necessity(
                        now_traffic_state, new_weight, use_sigmoid=True
                    )
                    necessity_results['keep_lane'] = 1

                    # 冲突评估--变道可行性
                    # 1. 设置场景
                    scenario_config = conflict_controller.setup_13_vehicle_scenario(ego_state, surrounding_state)

                    # 获取周围车辆概率分布（用于MPC）
                    surrounding_probability_distributions = {}

                    # 定义车辆分组
                    vehicle_groups = {
                        'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
                        'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
                        'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
                    }

                    assessment_records = {
                        'pre_change_left': {'feasibility_prob': 0.0},
                        'pre_change_right': {'feasibility_prob': 0.0},
                        'keep_lane': {'feasibility_prob': 1.0},
                    }
                    feasibility_probs = {
                        'feasibility_left': 0.0,  # 左变道可行性
                        'feasibility_right': 0.0  # 右变道可行性
                    }
                    for pre_decision, prediction_result in assessment_records.items():
                        if necessity_results[pre_decision] == 1:
                            # 根据决策确定需要观测的车道
                            if pre_decision == 'keep_lane':
                                vehicles_to_analysis = vehicle_groups['current_lane']
                            elif pre_decision == 'pre_change_left':
                                vehicles_to_analysis = vehicle_groups['current_lane']
                                vehicles_to_analysis.extend(vehicle_groups['left_lane'])
                            elif pre_decision == 'pre_change_right':
                                vehicles_to_analysis = vehicle_groups['current_lane']
                                vehicles_to_analysis.extend(vehicle_groups['right_lane'])

                            # 2. 生成自车轨迹
                            target_speed = get_target_speed(pre_decision, surrounding_vehicles)
                            ego_prediction_trajectory = conflict_controller.generate_ego_trajectory(
                                scenario_config, pre_decision, target_speed
                            )
                            assessment_records[pre_decision]['ego_prediction'] = ego_prediction_trajectory
                            surrounding_prediction_trajectories = conflict_controller.generate_vehicle_trajectories(
                                scenario_config, pre_decision, vehicles_to_analysis,
                            )
                            assessment_records[pre_decision]['surrounding_prediction'] = surrounding_prediction_trajectories

                            surrounding_envelopes = conflict_controller.generate_vehicle_envelopes(
                                surrounding_prediction_trajectories, scenario_config
                            )
                            assessment_records[pre_decision]['surrounding_envelopes'] = surrounding_envelopes

                            # 车辆颜色配置
                            vehicle_colors = {
                                'EF1': '#ff6b6b', 'EF2': '#ff8e8e',  # 前方车辆 - 红色系
                                'ER1': '#4ecdc4', 'ER2': '#7ed6cc',  # 后方车辆 - 青色系
                                'LF1': '#45b7d1', 'LF2': '#6bc5d2',  # 左前方车辆 - 蓝色系
                                'LR1': '#96ceb4', 'LR2': '#a8d5ba',  # 左后方车辆 - 绿色系
                                'RF1': '#feca57', 'RF2': '#ffd369',  # 右前方车辆 - 黄色系
                                'RR1': '#ff9ff3', 'RR2': '#ffb3f7'  # 右后方车辆 - 粉色系
                            }

                            # envelope_save_path = os.path.join(csv_save_folder, f'{step}_{veh_name}_{pre_decision}_envelope.png')
                            # conflict_controller.visualize_multi_vehicle_envelopes(
                            #     surrounding_envelopes, surrounding_prediction_trajectories, scenario_config, ego_prediction_trajectory, pre_decision, vehicle_colors,
                            #     envelope_save_path
                            # )

                            # 4. 生成概率分布
                            probability_distributions = conflict_controller.generate_probability_distributions(surrounding_envelopes, scenario_config, ego_prediction_trajectory, vehicles_to_analysis)
                            assessment_records[pre_decision]['probability_distributions'] = probability_distributions

                            # distribution_save_path = os.path.join(csv_save_folder, f'{step}_{veh_name}_{pre_decision}_distribution.png')
                            # conflict_controller.visualize_multi_vehicle_probability_distributions(
                            #     probability_distributions, scenario_config, ego_prediction_trajectory, pre_decision,
                            #     distribution_save_path
                            # )

                            # 5. 评估协同能力
                            cooperation_results = conflict_controller.evaluate_cooperation_capabilities(probability_distributions)
                            assessment_records[pre_decision]['cooperation_results'] = cooperation_results

                            # 6. 计算碰撞概率
                            collision_probabilities = conflict_controller.compute_collision_probabilities(
                                probability_distributions, cooperation_results
                            )
                            assessment_records[pre_decision]['collision_probabilities'] = collision_probabilities
                            # 7. 计算变道可行性
                            feasibility_result = conflict_controller.compute_lane_change_feasibility(
                                collision_probabilities, pre_decision
                            )
                            # print(feasibility_result)
                            if pre_decision == 'pre_change_left':
                                feasibility_probs['feasibility_left'] = feasibility_result['feasibility_probability']
                            elif pre_decision == 'pre_change_right':
                                feasibility_probs['feasibility_right'] = feasibility_result['feasibility_probability']

                            assessment_records[pre_decision]['feasibility_prob'] = feasibility_result['feasibility_probability']

                        else:
                            assessment_records[pre_decision]['feasibility_prob'] = 0.0

                    # 8. 做出最终横向决策
                    decision_result = lateral_decision.make_final_lateral_decision(
                        necessity_results, feasibility_probs
                    )
                    # print(decision_result)
                    final_decision = DECISION_STR[decision_result['decision'].value]
                    surrounding_probability_distributions = assessment_records[final_decision]['probability_distributions']
                    surrounding_prediction_centers = assessment_records[final_decision]['surrounding_envelopes']

                    # 9. 基于MPC的协同控制器--输出规划轨迹
                    # 确定目标参数
                    if decision_result['decision'] == LateralDecision.LEFT:
                        target_y = ego_state.y + system_params.lane_width
                        target_speed = get_target_speed('pre_change_left', surrounding_vehicles)
                    elif decision_result['decision'] == LateralDecision.RIGHT:
                        target_y = ego_state.y - system_params.lane_width
                        target_speed = get_target_speed('pre_change_right', surrounding_vehicles)
                    else:
                        target_y = ego_state.y  # 保持当前车道
                        target_speed = get_target_speed('keep_lane', surrounding_vehicles)

                    # 速度合理性检查
                    target_speed = max(5.0, min(target_speed, 25.0))

                    try:
                        # 调用MPC控制器
                        mpc_result = mpc_controller.solve_mpc(
                            current_state=ego_state,
                            reference_speed=target_speed,
                            target_y=target_y,
                            weights=new_weight,
                            surrounding_prediction_centers=surrounding_prediction_centers
                        )

                        if mpc_result.feasible:
                            # MPC求解成功，提取控制信息
                            optimal_control = mpc_result.optimal_control
                            next_state = mpc_result.predicted_states[10]
                            next_speed = next_state[2]
                            print(f"Optimal control: {optimal_control} \n"
                                  f"Predictive State (1s): {next_state}")

                        else:
                            # MPC求解失败
                            print(f"T={step_time:.1f}s CAV_{i_CAV}: MPC求解失败")

                    except Exception as e:
                        print(f"T={step_time:.1f}s CAV_{i_CAV}: MPC异常 - {str(e)}")


                    # # 10. 预测性安全评估（如果提供了计划轨迹）
                    # safety_risks = {}
                    # if ego_planned_trajectory is not None:
                    #     safety_risks = self.assess_predictive_safety(
                    #         ego_planned_trajectory, probability_distributions
                    #     )

                    if decision_result['decision'] == LateralDecision.KEEP:
                        actions['vehicle'][veh_name] = (0, -1, next_speed)
                    elif decision_result['decision'] == LateralDecision.LEFT:
                        actions['vehicle'][veh_name] = (1, -1, next_speed)
                    else:
                        actions['vehicle'][veh_name] = (2, -1, next_speed)


            # 统计实验使用
            # for veh_order, veh_id in vehicle_names.items():
            #     # Create dictionary with all the data you want to save
            #     veh_info = {
            #         't': info["step_time"] - 0.1,
            #         'Position_x': obs['vehicle'][veh_id]['position'][0],
            #         'Position_y': obs['vehicle'][veh_id]['position'][1],  # Uncommented this line
            #         'v_Vel': obs['vehicle'][veh_id]['speed'],
            #         'v_Acc': obs['vehicle'][veh_id]['acceleration']
            #     }
            #
            #     csv_path = csv_save_folder + '/' + f'{veh_order}' + '_' + veh_id + '.csv'
            #
            #     # Check if file exists to write headers only once
            #     file_exists = os.path.isfile(csv_path)
            #
            #     with open(csv_path, 'a', newline='') as csvfile:
            #         # Use DictWriter instead of writer to handle dictionary data properly
            #         writer = csv.DictWriter(csvfile, fieldnames=veh_info.keys())
            #
            #         # Write header if file is being created for the first time
            #         if not file_exists:
            #             writer.writeheader()
            #
            #         # Write the row of data
            #         writer.writerow(veh_info)

        # Step simulation
        obs, reward, info, done = tshub_env.step(actions=actions)

        # Render simulation if GUI is enabled
        if use_gui:
            fig = tshub_env.render(
                mode='sumo_gui',
                save_folder=image_save_folder,
                focus_id='Leader', focus_type='vehicle', focus_distance=200,
            )

        step_time = info["step_time"]
        logger.info(f"SIM: {step_time}")
        time.sleep(0.1)

    tshub_env._close_simulation()