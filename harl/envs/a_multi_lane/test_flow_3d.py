import numpy as np

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.check_folder import check_folder
from tshub.utils.init_log import set_logger
from tshub.tshub_env.tshub_env import TshubEnvironment
from tshub.tshub_env3d.tshub_env3d import Tshub3DEnvironment
from tshub.tshub_env3d.show_sensor_images import show_sensor_images
from harl.envs.a_multi_lane.env_utils.generate_scene_NGSIM import generate_scenario_NGSIM
import random
import math
import time
import os
import csv
import traci
import pandas as pd

aircraft_inits = {
        'a1': {
            "aircraft_type": "drone",
            "action_type": "stationary",  # 静止不动
            "position": (500, 0, 30), "speed": 0, "heading": (0, 0, 0),  # 初始信息
            "communication_range": 100,
            "if_sumo_visualization": True, "img_file": None,
            "custom_update_cover_radius": None  # 使用自定义的计算
        },
    }

if __name__ == '__main__':
    path_convert = get_abs_path(__file__)
    set_logger(path_convert('./'))

    sumo_cfg = path_convert("env_utils/SUMO_files/scenario.sumocfg")
    net_file = path_convert("env_utils/SUMO_files/veh.net.xml")  # "bottleneck.net.xml"
    # 创建存储图像的文件夹
    image_save_folder = path_convert('./global/')
    check_folder(image_save_folder)
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    csv_save_folder = path_convert('./motion/' + now_time + '/')
    check_folder(csv_save_folder)

    # build a memory block for the surrounding_info
    observed_vehicles = []

    # # 初始化环境
    # tshub_env = TshubEnvironment(
    #     sumo_cfg=sumo_cfg,
    #     net_file=net_file,
    #     is_map_builder_initialized=True,
    #     is_vehicle_builder_initialized=True,
    #     is_aircraft_builder_initialized=False,
    #     is_traffic_light_builder_initialized=False,
    #     # vehicle builder
    #     vehicle_action_type='vehicle_continuous_action',  # 'vehicle_continuous_action' 'lane_continuous_speed'
    #     use_gui=True, num_seconds=120
    # )
    scenario_glb_dir = path_convert(f"./model/")
    tshub_env3d = Tshub3DEnvironment(
        sumo_cfg=sumo_cfg,
        scenario_glb_dir=scenario_glb_dir,
        is_map_builder_initialized=False,
        is_aircraft_builder_initialized=True,
        is_vehicle_builder_initialized=True,
        is_traffic_light_builder_initialized=False,
        is_person_builder_initialized=False,
        aircraft_inits=aircraft_inits,
        vehicle_action_type='vehicle_continuous_action',
        use_gui=True,
        num_seconds=120,
        collision_action="warn",
        # 下面是用于渲染的参数
        preset="480P",
        render_mode="offscreen",  # 如果设置了 use_render_pipeline, 此时只能是 onscreen
        debuger_print_node=False,
        debuger_spin_camera=False,
        sensor_config={
            'vehicle': {
                '0': {'sensor_types': ['bev_all']},
            },  # 只给一辆车挂载摄像头
            'aircraft': {
                "a1": {"sensor_types": ['aircraft_all']}
            },
        }
    )

    # 开始仿真
    obs = tshub_env3d.reset()
    aggressive = 0.2
    cautious = 0.2
    normal = 0.6
    CAV_num = 12
    HDV_num = 18
    lane_veh_num = int((CAV_num + HDV_num) / 3)
    CAV_penetration = 0.4
    CAVs_id = []
    for i in range(CAV_num):
        CAVs_id.append(f'CAV_{i}')
    HDVs_id = []
    for i in range(HDV_num):
        HDVs_id.append(f'HDV_{i}')

    # leader_id = 8
    use_gui = True
    config_file = "/harl/envs/a_multi_lane/env_utils/vehicle_config.json"

    scene_definition = {
        0: {0: [3, 1], 1: [3, 1], 2: [3, 1]},
        1: {0: [8, 1], 1: [8, 1], 2: [8, 1]},
        2: {0: [11, 1], 1: [11, 1], 2: [11, 1]},
        3: {0: [14, 1], 1: [14, 1], 2: [14, 1]},
        4: {0: [8, 2], 1: [6, 2], 2: [8, 2]},
        5: {0: [11, 1], 1: [9, 1], 2: [11, 1]},
        6: {0: [14, 1], 1: [12, 1], 2: [14, 1]},
    }

    # scene_class_id = random.randint(0, 6)
    scene_class_id = 6

    scene_info = scene_definition[scene_class_id]

    leaders_trajectories = {}
    leaders_type = {}

    for i in range(3):
        for j in range(5):
            vehicle_name = f'Leader_{i}_{j}'
            leaders_type[vehicle_name] = 0
            leaders_type[vehicle_name] = 1 if j == 0 else 0

    quene_start, start_speed, vehicle_names, leader_counts = generate_scenario_NGSIM(
        config_file=config_file,
        aggressive=aggressive, cautious=cautious, normal=normal,
        use_gui=use_gui, sce_name="Env_MultiLane",
        CAV_num=CAV_num, HDV_num=HDV_num,
        CAV_penetration=CAV_penetration, distribution="uniform",
        scene_info=scene_info,
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

            for i in range(3):
                [mean_v, std_v] = scene_info[i]
                for j in range(leader_counts[i]):
                    vehicle_name = f'Leader_{i}_{j}'
                    leader = traci.vehicle.getLeader(vehicle_name, 100)
                    if leader and leader[0] in obs['vehicle']:
                        actions['vehicle'][vehicle_name] = (0, -1, -1)
                    else:
                        if j == 0:
                            next_speed = mean_v + std_v * np.sin(step * np.pi * 0.4 + i * np.pi)
                        else:
                            now_speed = obs['vehicle'][vehicle_name]['speed']
                            next_speed = now_speed + random.uniform(-1, 1) * 0.2
                        actions['vehicle'][vehicle_name] = (0, -1, next_speed)

                for j in range(10):
                    vehicle_name = f'Follower_{i}_{j}'
                    actions['vehicle'][vehicle_name] = (0, -1, -1)

            for veh_order, veh_id in vehicle_names.items():
                # Create dictionary with all the data you want to save
                veh_info = {
                    't': info["step_time"] - 1.0,
                    'Position_x': obs['vehicle'][veh_id]['position'][0],
                    'Position_y': obs['vehicle'][veh_id]['position'][1],  # Uncommented this line
                    'v_Vel': obs['vehicle'][veh_id]['speed'],
                    'v_Acc': obs['vehicle'][veh_id]['acceleration']
                }

                csv_path = csv_save_folder + '/' + f'{veh_order}' + '_' + veh_id + '.csv'

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

        # Step simulation
        obs, reward, info, done, sensor_data = tshub_env3d.step(actions=actions)

        # # Render simulation if GUI is enabled
        # if use_gui:
        #     fig = tshub_env.render(
        #         mode='rgb',
        #         save_folder=image_save_folder,
        #         focus_id='Leader', focus_type='vehicle', focus_distance=200,
        #     )

        # 显示图像
        try:
            show_sensor_images(
                [
                    sensor_data.get('0', {}).get('bev_all', None),
                    sensor_data.get('a1', {}).get('aircraft_all', None),
                ],
                scale=1,
                images_per_row=2
            )  # 展示路口的摄像头
        except:
            pass

        step_time = info["step_time"]
        logger.info(f"SIM: {step_time}")
        time.sleep(0.1)

    # tshub_env._close_simulation()
    tshub_env3d.close()