import copy
import gym
import sys
import numpy as np
from .env_utils.veh_env import VehEnvironment
from .env_utils.veh_env_wrapper import VehEnvWrapper
from tshub.utils.get_abs_path import get_abs_path

# 获得全局路径
path_convert = get_abs_path(__file__)

def make_multilane_envs(args):

    # base env
    sumo_cfg = path_convert(args['sumo_cfg'])
    num_seconds = args['max_num_seconds']  # 秒
    vehicle_action_type = args['vehicle_action_type']
    use_gui = args['use_gui']
    trip_info = None

    # for veh wrapper
    scene_name = args['scene_name']
    max_num_CAVs = args['max_num_CAVs']
    max_num_HDVs = args['max_num_HDVs']
    num_CAVs = args['num_CAVs']
    num_HDVs = args['num_HDVs']
    penetration_CAV = args['penetration_CAV']
    lane_max_num_vehs = args['lane_max_num_vehs']
    warmup_steps = args['warmup_steps']
    edge_ids = args['edge_ids']
    edge_lane_num = args['edge_lane_num']
    node_positions = args['node_positions']
    calc_features_lane_ids = args['calc_features_lane_ids']
    log_path = path_convert(args['log_path'])
    delta_t = args['delta_t']
    aggressive = args['aggressive']
    cautious = args['cautious']
    normal = args['normal']
    strategy = args['strategy']
    use_hist_info = args['use_hist_info']
    hist_length = args['hist_length']
    rew_weights = args['reward_weights']
    scene_specify = args['scene_specify']
    scene_id = args['scene_id']
    use_V2V_info = args['use_V2V_info']
    parameters = args['parameters']
    is_evaluation = args['is_evaluation']
    save_model_name = args['save_model_name']
    eval_weights = args['eval_weights']
    scene_centers = args['scene_centers']

    hierarchical_weights = args['hierarchical_weights']
    anti_fake_params = args['anti_fake_params']
    scene_definition = args['scene_definition']

    veh_env = VehEnvironment(
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        vehicle_action_type=vehicle_action_type,
        delta_time=delta_t,
        use_gui=use_gui,
        trip_info=trip_info,
    )
    veh_env = VehEnvWrapper(
        env=veh_env,
        name_scenario=scene_name,
        max_num_CAVs=max_num_CAVs,
        max_num_HDVs=max_num_HDVs,
        num_CAVs=num_CAVs,
        num_HDVs=num_HDVs,
        CAV_penetration=penetration_CAV,
        lane_max_num_vehs=lane_max_num_vehs,
        warmup_steps=warmup_steps,
        edge_ids=edge_ids,
        edge_lane_num=edge_lane_num,
        node_positions=node_positions,
        calc_features_lane_ids=calc_features_lane_ids,
        filepath=log_path,
        use_gui=use_gui,
        delta_t=delta_t,
        aggressive=aggressive,
        cautious=cautious,
        normal=normal,
        strategy=strategy,
        use_hist_info=use_hist_info,
        hist_length=hist_length,
        num_seconds=num_seconds,
        rew_weights=rew_weights,
        scene_specify=scene_specify,
        scene_id=scene_id,
        use_V2V_info=use_V2V_info,
        parameters=parameters,
        is_evaluation=is_evaluation,
        save_model_name=save_model_name,
        eval_weights=eval_weights,
        scene_centers=scene_centers,
        hierarchical_weights=hierarchical_weights,
        anti_fake_params=anti_fake_params,
        scene_definition=scene_definition,
    )
    return veh_env


class MultiLaneEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = make_multilane_envs(self.args)
        self.n_agents = self.env.num_CAVs
        self.share_observation_space = list(self.env.share_observation_space.values())
        self.observation_space = list(self.env.observation_space.values())
        self.action_space = list(self.env.action_space.values())
        self.discrete = False

        # FOR DEBUGGING
        self.ego_ids = self.env.ego_ids
        self.total_timesteps = self.env.total_timesteps

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        # for train
        action_dict = {ego_id: action for ego_id, action in zip(self.env.ego_ids, actions)}
        # for check
        # action_dict = {ego_id: actions[ego_id] for ego_id in self.env.ego_ids}

        obs, s_obs, rew, mean_v, mean_acc, rew_safety, rew_stability, rew_efficiency, rew_comfort, \
        safety_SI, efficiency_ASR, efficiency_TFR, stability_SSI, stability_VSS, comfort_JI, control_efficiency, control_reward, comfort_cost, comfort_reward, \
        truncated, done, info, classified_scene_mark = self.env.step(action_dict)

        # s_obs = self.convert_shared_obs(obs)
        obs = list(obs.values())
        s_obs = list(s_obs.values())
        rew = np.array(list(rew.values())).reshape((-1, 1))
        rew_safety = np.array(list(rew_safety.values())).reshape((-1, 1))
        rew_stability = np.array(list(rew_stability.values())).reshape((-1, 1))
        rew_efficiency = np.array(list(rew_efficiency.values())).reshape((-1, 1))
        rew_comfort = np.array(list(rew_comfort.values())).reshape((-1, 1))
        # mean_v = mean_v.reshape(-1, 1)
        # mean_acc = mean_acc.reshape(1, -1)
        done = np.array(list(done.values()))

        self.action_command = self.env.actions  # action_command
        self.current_speed = self.env.current_speed
        self.current_lane = self.env.current_lane
        self.warn_ego_ids = self.env.warn_ego_ids
        self.coll_ego_ids = self.env.coll_ego_ids
        self.total_timesteps = self.env.total_timesteps

        return obs, s_obs, rew, mean_v, mean_acc, rew_safety, rew_stability, rew_efficiency, rew_comfort, \
            safety_SI, efficiency_ASR, efficiency_TFR, stability_SSI, stability_VSS, comfort_JI, control_efficiency, control_reward, comfort_cost, comfort_reward, \
            done, self.repeat(info), self.get_avail_actions(), classified_scene_mark

    def reset(self):
        """Returns initial observations and states"""
        obs, s_obs, _, classified_scene_mark = self.env.reset()
        # s_obs = self.convert_shared_obs(obs)
        obs = list(obs.values())
        s_obs = list(s_obs.values())
        # classified_scene_mark = list(classified_scene_mark.values())
        avail_actions = self.get_avail_actions()

        return obs, s_obs, avail_actions, classified_scene_mark

    def seed(self, seed):
        pass

    # def get_avail_actions(self):
    #
    #     avail_actions = [[1] * self.action_space[0].n]*self.n_agents
    #     return np.array(avail_actions)
    #     # TODO: 换道的动作被mask掉
    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def close(self):
        self.env.close()

    def convert_shared_obs(self, obs_dict):
        # Concatenate all observations into one list
        all_observations = []
        for cav, observations in obs_dict.items():
            all_observations.extend(observations)

        # Create shared_obs with the same keys but with the concatenated list for each
        shared_obs = {cav: all_observations for cav in obs_dict.keys()}

        return shared_obs

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

