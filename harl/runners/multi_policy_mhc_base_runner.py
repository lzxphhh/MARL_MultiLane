"""Base runner for on-policy algorithms with multi-head critic."""

import time
import os
import numpy as np
import torch
import setproctitle
from collections import defaultdict
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.multi_policy_actor_buffer import MultiPolicyActorBuffer
from harl.common.buffers.multi_policy_mhcritic_buffer_ep import MultiPolicyCriticBufferEP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.mp_mh_v_critic import MHVCritic
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY

class MultiPolicyBaseRunner:
    """Base runner for multi-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the MultiPolicyMHCBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # 读取算法相关config
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]  # MLP隐藏层神经元数量
        self.rnn_hidden_size = self.hidden_sizes[-1]  # RNN隐藏层神经元数量
        self.recurrent_n = algo_args["model"]["recurrent_n"]  # RNN的层数
        self.action_aggregation = algo_args["algo"]["action_aggregation"]  # 多维动作空间的聚合方式，如mean/prod
        self.share_param = algo_args["algo"]["share_param"]  # actor是否共享参数
        self.fixed_order = algo_args["algo"]["fixed_order"]  # 是否固定agent的策略更新顺序
        set_seed(algo_args["seed"])  # 设置随机种子
        self.device = init_device(algo_args["device"])  # 设置设备

        # 多头critic相关参数
        self.critic_head_num = algo_args["model"]["critic_head_num"]  # critic的头数量
        self.critic_head_names = algo_args["model"]["critic_head_names"]  # 每个头对应的奖励组件名称

        # 场景分类相关参数
        self.scene_centers = env_args.get("scene_centers", {})
        self.num_scene_classes = len(self.scene_centers)
        if self.num_scene_classes == 0:
            self.num_scene_classes = 5  # 默认5个场景类别
            print("Warning: No scene_centers found in env_args, using default 5 scene classes")

        print(f"Initializing multi-scene runner with {self.num_scene_classes} scene classes")
        print(f"Scene centers: {list(self.scene_centers.keys())}")

        # train, not render 说明在训练，不在eval
        if not self.algo_args["render"]["use_render"]:
            # 初始化运行路径，日志路径，保存路径，tensorboard路径
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                args["test_desc"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            # 保存algo，env args，algo args所有config
            save_config(args, algo_args, env_args, self.run_dir)

        # set the title of the process
        # 设置进程的标题
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # 使用env tools中的函数创建训练/测试/render环境 （调取环境+插入env config）
        if self.algo_args["render"]["use_render"]:
            # 创建单线程render环境
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:
            # 创建多线程训练环境
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            # 创建多线程测试环境
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        # 默认使用EP作为state_type
        # EP：EnvironmentProvided global state (EP)：环境提供的全局状态
        # FP：Featured-Pruned Agent-Specific Global State (FP)： 特征裁剪的特定智能体全局状态(不同agent的全局状态不同, 需要agent number)
        self.state_type = env_args.get("state_type", "EP")
        # TODO： EP or FP need to be added to customized env

        # 智能体数量
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)
        print("num_agents: ", self.num_agents)
        # 奖励维度
        self.num_reward_head = env_args["num_reward_head"]

        # 初始化多场景的actor和critic
        self._init_multi_scene_components()

        # 初始化训练相关组件
        if not self.algo_args["render"]["use_render"]:
            # ValueNorm
            if self.algo_args["train"]["use_valuenorm"] is True:
                # 为每个场景类别创建独立的value normalizer
                self.value_normalizers = []
                for scene_id in range(self.num_scene_classes):
                    self.value_normalizers.append(ValueNorm(1, device=self.device))
            else:
                self.value_normalizers = [None] * self.num_scene_classes

            # 环境logger
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )

        # 场景分类统计
        self.scene_stats = defaultdict(int)
        self.total_steps = 0

        self.scene_episode_stats = defaultdict(int)  # 每个场景的episode数量
        self.scene_step_stats = defaultdict(int)  # 每个场景的总步数
        self.scene_reward_stats = defaultdict(list)  # 每个场景的奖励历史

        # 当前episode的场景统计
        self.current_episode_scene_count = defaultdict(int)
        self.current_episode_total_steps = 0

        # 动态权重更新频率
        self.weight_update_frequency = algo_args.get("weight_update_frequency", 20)

        # 可以restore之前训练到一半的模型继续训练
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def _init_multi_scene_components(self):
        """Initialize multi-scene actors, critics, and buffers."""
        print("Initializing multi-scene components...")

        # 为每个场景类别创建独立的actor
        self.scene_actors = []
        self.scene_actor_buffers = []

        for scene_id in range(self.num_scene_classes):
            scene_actors = []
            scene_actor_buffers = []

            if self.share_param:
                # 参数共享模式：每个场景一个actor，所有agent共享
                agent = ALGO_REGISTRY[self.args["algo"]](
                    {**self.algo_args["model"], **self.algo_args["algo"], **self.algo_args["train"], **self.env_args},
                    self.envs.observation_space[0],
                    self.envs.action_space[0],
                    device=self.device,
                )

                # 验证观测和动作空间一致性
                for agent_id in range(1, self.num_agents):
                    assert (
                            self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                    ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                    assert (
                            self.envs.action_space[agent_id] == self.envs.action_space[0]
                    ), "Agents have heterogeneous action spaces, parameter sharing is not valid."

                # 所有agent共享同一个actor
                for agent_id in range(self.num_agents):
                    scene_actors.append(agent)
            else:
                # 非参数共享模式：每个场景每个agent一个actor
                for agent_id in range(self.num_agents):
                    agent = ALGO_REGISTRY[self.args["algo"]](
                        {**self.algo_args["model"], **self.algo_args["algo"], **self.algo_args["train"],
                         **self.env_args},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        device=self.device,
                    )
                    scene_actors.append(agent)

            self.scene_actors.append(scene_actors)

            # 为每个场景的每个agent创建actor buffer
            if not self.algo_args["render"]["use_render"]:
                for agent_id in range(self.num_agents):
                    ac_bu = MultiPolicyActorBuffer(
                        {**self.algo_args["train"], **self.algo_args["model"], **self.env_args},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                    )
                    scene_actor_buffers.append(ac_bu)

                self.scene_actor_buffers.append(scene_actor_buffers)

        # 为每个场景类别创建独立的critic
        self.scene_critics = []
        self.scene_critic_buffers = []

        if not self.algo_args["render"]["use_render"]:
            share_observation_space = self.envs.share_observation_space[0]

            for scene_id in range(self.num_scene_classes):
                # 创建场景特定的critic
                critic = MHVCritic(
                    {**self.algo_args["model"], **self.algo_args["algo"], **self.algo_args["train"], **self.env_args},
                    share_observation_space,
                    device=self.device,
                )
                self.scene_critics.append(critic)

                # 创建critic buffer
                if self.state_type == "EP":
                    critic_buffer = MultiPolicyCriticBufferEP(
                        {**self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"],
                         **self.env_args},
                        share_observation_space,
                    )
                else:
                    raise NotImplementedError("FP state type not implemented for multi-head critic yet")

                self.scene_critic_buffers.append(critic_buffer)

        print(f"Successfully initialized {self.num_scene_classes} scene classes:")
        print(f"  - Actors per scene: {len(self.scene_actors[0]) if self.scene_actors else 0}")
        print(f"  - Parameter sharing: {self.share_param}")
        print(f"  - Multi-head critics: {self.critic_head_num} heads")

    def _get_scene_distribution(self):
        """Get current scene distribution statistics."""
        if self.total_steps == 0:
            return {i: 0.0 for i in range(self.num_scene_classes)}

        distribution = {}
        for scene_id in range(self.num_scene_classes):
            distribution[scene_id] = self.scene_stats[scene_id] / self.total_steps

        return distribution

    def _update_learning_rates(self, episode, episodes):
        """Update learning rates for all scene actors and critics."""
        for scene_id in range(self.num_scene_classes):
            if self.share_param:
                self.scene_actors[scene_id][0].lr_decay(episode, episodes)
            else:
                for agent_id in range(self.num_agents):
                    self.scene_actors[scene_id][agent_id].lr_decay(episode, episodes)

            self.scene_critics[scene_id].lr_decay(episode, episodes)

    def _update_dynamic_weights(self):
        """Update dynamic weights for all scene critics."""
        for scene_id in range(self.num_scene_classes):
            self.scene_critics[scene_id].update_dynamic_weights()

            # 将critic的动态权重传递给对应的actors
            dynamic_weights = self.scene_critics[scene_id].get_dynamic_weights()

            if self.share_param:
                self.scene_actors[scene_id][0].set_dynamic_weights(dynamic_weights)
            else:
                for agent_id in range(self.num_agents):
                    self.scene_actors[scene_id][agent_id].set_dynamic_weights(dynamic_weights)

    def run(self):
        """Run the training (or rendering) pipeline."""

        # render,不是训练
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return

        # 开始训练
        print("start running with scene distribution tracking")

        # 在环境reset之后返回的obs，share_obs，available_actions存入每一个actor的replay buffer 以及 集中式critic的replay buffer
        self.warmup()

        # 计算总共需要跑多少个episode = 总训练时间步数 / 每个episode的时间步数 / 并行的环境数 (int)
        episodes = (
                # 训练总时间步数 / 每个episode的时间步数 / 并行的环境数
                int(self.algo_args["train"]["num_env_steps"])
                // self.algo_args["train"]["episode_length"]
                // self.algo_args["train"]["n_rollout_threads"]
        )

        # 初始化logger
        self.logger.init(episodes)  # logger callback at the beginning of training

        # 开始训练！！！！！！
        # 对于每一个episode
        for episode in range(1, episodes + 1):
            # start = time.time()
            # 重置episode级别的场景统计
            self._reset_episode_scene_stats()
            # 学习率是否随着episode线性递减
            if self.algo_args["train"]["use_linear_lr_decay"]:
                self._update_learning_rates(episode, episodes)

            # 动态权重更新
            if episode % self.weight_update_frequency == 0:
                self._update_dynamic_weights()

            # 每个episode开始的时候更新logger里面的episode index
            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            # 把actor和critic网络都切换到eval模式
            self.prep_rollout()  # change to eval mode

            # 对于所有并行环境一个episode的每一个时间步
            for step in range(self.algo_args["train"]["episode_length"]):
                """
                采样动作 - 进入actor network 
                values: (n_threads, critic_head_num, 1) - 所有并行环境在这一个timestep的多头critic网络的输出
                actions: (n_threads, n_agents, 1) 
                action_log_probs: (n_threads, n_agents, 1)
                rnn_states: (进程数量, n_agents, rnn层数, rnn_hidden_dim)
                rnn_states_critic: (n_threads, rnn层数, rnn_hidden_dim)
                """
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,  # rnn_states是actor的rnn的hidden state
                    rnn_states_critic,  # rnn_states_critic是critic的rnn的hidden state
                    action_losss,
                    reward_weights,
                ) = self.collect(step)

                """
                在得到动作后，执行动作 - 进入环境 ShareVecEnv | step
                与环境交互一个step，得到obs，share_obs，rewards，dones，infos，available_actions
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                """
                (
                    obs,
                    share_obs,
                    rewards,
                    mean_v,
                    mean_acc,
                    rewards_safety,
                    rewards_stability,
                    rewards_efficiency,
                    rewards_comfort,
                    safety_SI,
                    efficiency_ASR,
                    efficiency_TFR,
                    stability_SSI,
                    stability_VSS,
                    comfort_JI,
                    control_efficiency,
                    control_reward,
                    comfort_cost,
                    comfort_reward,
                    dones,
                    infos,
                    available_actions,
                    classified_scene_mark,
                ) = self.envs.step(actions)
                """每个step更新logger里面的per_step data"""
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    action_losss,
                    mean_v,
                    mean_acc,
                    rewards_safety,
                    rewards_stability,
                    rewards_efficiency,
                    rewards_comfort,
                    reward_weights,
                    safety_SI,
                    efficiency_ASR,
                    efficiency_TFR,
                    stability_SSI,
                    stability_VSS,
                    comfort_JI,
                    control_efficiency,
                    control_reward,
                    comfort_cost,
                    comfort_reward,
                    classified_scene_mark,
                )

                self.logger.per_step(data)  # logger callback at each step

                """把这一步的数据存入每一个actor的replay buffer 以及 集中式critic的replay buffer"""
                self.insert(data)  # insert data into buffer

            # 在episode结束时更新场景统计
            episode_rewards_by_scene = self._calculate_episode_rewards_by_scene()
            self._update_scene_episode_stats(episode_rewards_by_scene)

            # 收集完了一个episode的所有timestep data，开始计算return，更新网络
            # compute Q and V using GAE or not
            self.compute()

            # 结束这一个episode的交互数据收集
            # 把actor和critic网络都切换回train模式
            self.prep_training()

            # 开始训练，在子类中实现
            actor_train_infos, critic_train_info = self.train()

            # 添加详细的场景统计信息到critic_train_info
            scene_stats = self.get_comprehensive_scene_statistics()
            critic_train_info.update({
                'comprehensive_scene_stats': scene_stats,
                'current_episode_scene_distribution': self._get_current_episode_scene_distribution()
            })

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                save_model_signal, current_timestep = self.logger.episode_mp_log(
                    actor_train_infos,
                    critic_train_info,
                    self.scene_actor_buffers,  # 传递所有场景的buffers
                    self.scene_critic_buffers,
                    self.env_args["save_collision"],
                    self.env_args["save_episode_mean_speed"],
                )
                if save_model_signal:
                    self.save_good_model(current_timestep)
                else:
                    pass

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            # 把上一个episode产生的最后一个timestep的state放入buffer的新的episode的第一个timestep
            self.after_update()
            # end = time.time()
            # print(f"Episode {episode} takes {end - start} seconds")

    def _classify_scene(self, classified_scene_marks):
        """Classify and validate scene marks with enhanced statistics."""
        # 确保scene marks在有效范围内
        valid_scene_marks = np.clip(classified_scene_marks, 0, self.num_scene_classes - 1)

        # 统计场景分布
        for scene_mark in valid_scene_marks:
            self.scene_stats[scene_mark] += 1
            self.scene_step_stats[scene_mark] += 1
            self.current_episode_scene_count[scene_mark] += 1

        self.total_steps += len(valid_scene_marks)
        self.current_episode_total_steps += len(valid_scene_marks)

        return valid_scene_marks

    def warmup(self):
        """
        Warm up the replay buffer.
        在环境reset之后返回的obs，share_obs，available_actions存入每一个actor的replay buffer 以及 集中式critic的replay buffer
        """
        """
        reset所有的并行环境，返回
        obs: (n_threads, n_agents, obs_dim)
        share_obs: (n_threads, n_agents, share_obs_dim)
        available_actions: (n_threads, n_agents, action_dim)
        """
        obs, share_obs, available_actions, classified_scene_mark = self.envs.reset()

        # 验证场景分类
        classified_scene_mark = self._classify_scene(classified_scene_mark)

        # 为每个场景的actor buffer准备初始数据
        for scene_id in range(self.num_scene_classes):
            # 找到属于当前场景的环境
            scene_mask = (classified_scene_mark == scene_id)
            if not np.any(scene_mask):
                continue  # 如果没有环境属于这个场景，跳过

            scene_obs = obs[scene_mask]
            scene_available_actions = available_actions[scene_mask] if available_actions[0] is not None else None

            # 为当前场景的每个agent初始化buffer
            for agent_id in range(self.num_agents):
                if len(scene_obs) > 0:  # 确保有数据
                    # 将场景特定的观测数据放入对应场景的buffer
                    # 这里需要特殊处理，因为scene_obs的形状可能不匹配
                    expanded_obs = np.zeros((self.algo_args["train"]["n_rollout_threads"], *obs.shape[2:]))
                    env_indices = np.where(scene_mask)[0]
                    if len(env_indices) > 0:
                        for i, env_idx in enumerate(env_indices):
                            if i < expanded_obs.shape[0]:
                                expanded_obs[i] = obs[env_idx, agent_id]

                    self.scene_actor_buffers[scene_id][agent_id].obs[0] = expanded_obs

                    if self.scene_actor_buffers[scene_id][
                        agent_id].available_actions is not None and scene_available_actions is not None:
                        expanded_actions = np.zeros(
                            (self.algo_args["train"]["n_rollout_threads"], *available_actions.shape[2:]))
                        for i, env_idx in enumerate(env_indices):
                            if i < expanded_actions.shape[0]:
                                expanded_actions[i] = available_actions[env_idx, agent_id]
                        self.scene_actor_buffers[scene_id][agent_id].available_actions[0] = expanded_actions

        # 为每个场景的critic buffer准备初始数据
        if self.state_type == "EP":
            for scene_id in range(self.num_scene_classes):
                scene_mask = (classified_scene_mark == scene_id)
                if not np.any(scene_mask):
                    continue

                scene_share_obs = share_obs[scene_mask]
                if len(scene_share_obs) > 0:
                    expanded_share_obs = np.zeros((self.algo_args["train"]["n_rollout_threads"], *share_obs.shape[2:]))
                    env_indices = np.where(scene_mask)[0]
                    for i, env_idx in enumerate(env_indices):
                        if i < expanded_share_obs.shape[0]:
                            expanded_share_obs[i] = share_obs[env_idx, 0]

                    self.scene_critic_buffers[scene_id].share_obs[0] = expanded_share_obs

    @torch.no_grad()  # 前向，没有反向传播，不需要计算梯度
    def collect(self, step):
        """Collect actions and values from all scene actors and critics.
        Args:
            step: current step in the episode
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic, action_losss, reward_weights
        """
        # 为所有环境准备输出容器
        n_threads = self.algo_args["train"]["n_rollout_threads"]

        values = np.zeros((n_threads, self.critic_head_num))
        actions = np.zeros((n_threads, self.num_agents, 1))
        action_log_probs = np.zeros((n_threads, self.num_agents, 1))
        rnn_states = np.zeros((n_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size))
        rnn_states_critic = np.zeros((n_threads, self.recurrent_n, self.rnn_hidden_size))
        action_losss = np.zeros((n_threads, self.num_agents, 1))
        reward_weights = np.zeros((n_threads, self.num_agents, self.num_reward_head))

        # 获取当前的场景分类 - 这里需要从某个地方获取，暂时使用简单的方法
        # 在实际实现中，这个信息应该从环境或者之前的step中获得
        if step == 0:
            # 第一步，随机分配场景（在实际使用中应该有更好的方法）
            classified_scene_mark = np.random.randint(0, self.num_scene_classes, n_threads)
        else:
            # 从previous step获取，这里简化处理
            classified_scene_mark = np.random.randint(0, self.num_scene_classes, n_threads)

        classified_scene_mark = self._classify_scene(classified_scene_mark)

        # 为每个场景分别收集数据
        for scene_id in range(self.num_scene_classes):
            scene_mask = (classified_scene_mark == scene_id)
            scene_env_indices = np.where(scene_mask)[0]

            if len(scene_env_indices) == 0:
                continue  # 没有环境属于这个场景

            # 从scene actors收集actions
            scene_action_collector = []
            scene_action_log_prob_collector = []
            scene_rnn_state_collector = []
            scene_action_loss_collector = []
            scene_reward_weights_collector = []

            for agent_id in range(self.num_agents):
                # 获取当前agent的观测数据
                agent_obs = np.zeros(
                    (len(scene_env_indices), *self.scene_actor_buffers[scene_id][agent_id].obs.shape[2:]))
                agent_rnn_states = np.zeros(
                    (len(scene_env_indices), *self.scene_actor_buffers[scene_id][agent_id].rnn_states.shape[2:]))
                agent_masks = np.zeros((len(scene_env_indices), 1))
                agent_available_actions = None

                # 从buffer中获取数据
                for i, env_idx in enumerate(scene_env_indices):
                    if env_idx < self.scene_actor_buffers[scene_id][agent_id].obs.shape[1]:
                        agent_obs[i] = self.scene_actor_buffers[scene_id][agent_id].obs[step, env_idx]
                        agent_rnn_states[i] = self.scene_actor_buffers[scene_id][agent_id].rnn_states[step, env_idx]
                        agent_masks[i] = self.scene_actor_buffers[scene_id][agent_id].masks[step, env_idx]

                        if self.scene_actor_buffers[scene_id][agent_id].available_actions is not None:
                            if agent_available_actions is None:
                                agent_available_actions = np.zeros((len(scene_env_indices),
                                                                    *self.scene_actor_buffers[scene_id][
                                                                         agent_id].available_actions.shape[2:]))
                            agent_available_actions[i] = self.scene_actor_buffers[scene_id][agent_id].available_actions[
                                step, env_idx]

                # 获取reward weights
                if step == 0:
                    agent_reward_weights = np.zeros((len(scene_env_indices), self.num_reward_head))
                else:
                    agent_reward_weights = np.zeros((len(scene_env_indices), self.num_reward_head))
                    for i, env_idx in enumerate(scene_env_indices):
                        if env_idx < self.scene_actor_buffers[scene_id][agent_id].rewards_weights.shape[1]:
                            agent_reward_weights[i] = self.scene_actor_buffers[scene_id][agent_id].rewards_weights[
                                step - 1, env_idx]

                # 调用场景特定的actor
                action, action_log_prob, rnn_state, action_loss, reward_weight = self.scene_actors[scene_id][
                    agent_id].get_actions(
                    agent_obs,
                    agent_rnn_states,
                    agent_reward_weights,
                    agent_masks,
                    agent_available_actions,
                )

                scene_action_collector.append(_t2n(action))
                scene_action_log_prob_collector.append(_t2n(action_log_prob))
                scene_rnn_state_collector.append(_t2n(rnn_state))
                scene_action_loss_collector.append(_t2n(action_loss))
                scene_reward_weights_collector.append(_t2n(reward_weight))

            # 将场景结果放回全局数组
            if scene_action_collector:
                scene_actions = np.array(scene_action_collector).transpose(1, 0, 2)  # (n_scene_envs, n_agents, 1)
                scene_action_log_probs = np.array(scene_action_log_prob_collector).transpose(1, 0, 2)
                scene_rnn_states = np.array(scene_rnn_state_collector).transpose(1, 0, 2, 3)
                scene_action_losss = np.array(scene_action_loss_collector).transpose(1, 0, 2)
                scene_reward_weights = np.array(scene_reward_weights_collector).transpose(1, 0, 2)

                # 放入对应的环境位置
                for i, env_idx in enumerate(scene_env_indices):
                    if i < len(scene_actions):
                        actions[env_idx] = scene_actions[i]
                        action_log_probs[env_idx] = scene_action_log_probs[i]
                        rnn_states[env_idx] = scene_rnn_states[i]
                        action_losss[env_idx] = scene_action_losss[i]
                        reward_weights[env_idx] = scene_reward_weights[i]

            # 从scene critic收集values
            if len(scene_env_indices) > 0:
                # 获取共享观测
                scene_share_obs = np.zeros(
                    (len(scene_env_indices), *self.scene_critic_buffers[scene_id].share_obs.shape[2:]))
                scene_rnn_states_critic = np.zeros(
                    (len(scene_env_indices), *self.scene_critic_buffers[scene_id].rnn_states_critic.shape[2:]))
                scene_masks = np.zeros((len(scene_env_indices), 1))

                for i, env_idx in enumerate(scene_env_indices):
                    if env_idx < self.scene_critic_buffers[scene_id].share_obs.shape[1]:
                        scene_share_obs[i] = self.scene_critic_buffers[scene_id].share_obs[step, env_idx]
                        scene_rnn_states_critic[i] = self.scene_critic_buffers[scene_id].rnn_states_critic[
                            step, env_idx]
                        scene_masks[i] = self.scene_critic_buffers[scene_id].masks[step, env_idx]

                # 调用场景特定的critic
                scene_values, scene_rnn_states_critic_new = self.scene_critics[scene_id].get_values(
                    scene_share_obs,
                    scene_rnn_states_critic,
                    scene_masks,
                )

                scene_values = _t2n(scene_values)
                scene_rnn_states_critic_new = _t2n(scene_rnn_states_critic_new)

                # 放入对应的环境位置
                for i, env_idx in enumerate(scene_env_indices):
                    if i < len(scene_values):
                        values[env_idx] = scene_values[i]
                        rnn_states_critic[env_idx] = scene_rnn_states_critic_new[i]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_losss, reward_weights

    def insert(self, data):
        """Insert data into appropriate scene buffers based on classified_scene_mark."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            action_losss,
            mean_v,
            mean_acc,
            rewards_safety,
            rewards_stability,
            rewards_efficiency,
            rewards_comfort,
            reward_weights,
            safety_SI,
            efficiency_ASR,
            efficiency_TFR,
            stability_SSI,
            stability_VSS,
            comfort_JIs,
            control_efficiencys,
            control_reward,
            comfort_cost,
            comfort_reward,
            classified_scene_mark,
        ) = data

        # 验证场景分类
        classified_scene_mark = self._classify_scene(classified_scene_mark)

        # 重置RNN states
        dones_env = np.all(dones, axis=1)

        # 处理masks
        masks = np.ones((self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks处理
        if self.state_type == "EP":
            bad_masks = np.array([
                [0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0]
                for info in infos
            ])

        # 为多头critic准备奖励组件数据
        n_threads = rewards_safety.shape[0]
        n_agents = rewards_safety.shape[1]
        reward_components_array = np.zeros((n_threads, n_agents, self.critic_head_num), dtype=np.float32)

        # 填充各个通道
        for i in range(self.critic_head_num):
            head_name = self.critic_head_names.get(i, f"head_{i}")
            if head_name == "safety":
                reward_components_array[:, :, i] = rewards_safety[:, :, 0] if len(
                    rewards_safety.shape) == 3 else rewards_safety
            elif head_name == "efficiency":
                reward_components_array[:, :, i] = rewards_efficiency[:, :, 0] if len(
                    rewards_efficiency.shape) == 3 else rewards_efficiency
            elif head_name == "stability":
                reward_components_array[:, :, i] = rewards_stability[:, :, 0] if len(
                    rewards_stability.shape) == 3 else rewards_stability
            elif head_name == "comfort":
                reward_components_array[:, :, i] = rewards_comfort[:, :, 0] if len(
                    rewards_comfort.shape) == 3 else rewards_comfort

        # 根据场景分类插入数据到对应的buffers
        for scene_id in range(self.num_scene_classes):
            scene_mask = (classified_scene_mark == scene_id)
            scene_env_indices = np.where(scene_mask)[0]

            if len(scene_env_indices) == 0:
                continue  # 没有环境属于这个场景

            # 重置对应场景环境的RNN states
            scene_rnn_states = rnn_states[scene_mask]
            scene_rnn_states_critic = rnn_states_critic[scene_mask]

            # 如果环境done了，重置RNN states
            scene_dones_env = dones_env[scene_mask]
            if np.any(scene_dones_env):
                scene_rnn_states[scene_dones_env] = np.zeros(
                    (scene_dones_env.sum(), self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                    dtype=np.float32,
                )
                scene_rnn_states_critic[scene_dones_env] = np.zeros(
                    (scene_dones_env.sum(), self.recurrent_n, self.rnn_hidden_size),
                    dtype=np.float32,
                )

            # 插入actor buffer数据
            for agent_id in range(self.num_agents):
                # 准备agent专用数据
                agent_obs = obs[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_rnn_states = scene_rnn_states[:, agent_id] if len(scene_rnn_states) > 0 else np.array([])
                agent_actions = actions[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_action_log_probs = action_log_probs[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_action_losss = action_losss[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_reward_weights = reward_weights[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_masks = masks[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_active_masks = active_masks[scene_mask, agent_id] if np.any(scene_mask) else np.array([])

                # 奖励组件数据
                agent_rewards_safety = rewards_safety[scene_mask, agent_id] if np.any(scene_mask) else np.array([])
                agent_rewards_stability = rewards_stability[scene_mask, agent_id] if np.any(scene_mask) else np.array(
                    [])
                agent_rewards_efficiency = rewards_efficiency[scene_mask, agent_id] if np.any(scene_mask) else np.array(
                    [])
                agent_rewards_comfort = rewards_comfort[scene_mask, agent_id] if np.any(scene_mask) else np.array([])

                # available_actions处理
                agent_available_actions = None
                if available_actions[0] is not None:
                    agent_available_actions = available_actions[scene_mask, agent_id] if np.any(scene_mask) else None

                # 确保数据不为空才插入
                if len(agent_obs) > 0:
                    # 扩展数据到完整的rollout threads维度
                    expanded_obs = np.zeros((self.algo_args["train"]["n_rollout_threads"], *agent_obs.shape[1:]))
                    expanded_rnn_states = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_rnn_states.shape[1:]))
                    expanded_actions = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_actions.shape[1:]))
                    expanded_action_log_probs = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_action_log_probs.shape[1:]))
                    expanded_action_losss = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_action_losss.shape[1:]))
                    expanded_reward_weights = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_reward_weights.shape[1:]))
                    expanded_masks = np.zeros((self.algo_args["train"]["n_rollout_threads"], *agent_masks.shape[1:]))
                    expanded_active_masks = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_active_masks.shape[1:]))

                    # 奖励组件扩展
                    expanded_rewards_safety = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_rewards_safety.shape[1:]))
                    expanded_rewards_stability = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_rewards_stability.shape[1:]))
                    expanded_rewards_efficiency = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_rewards_efficiency.shape[1:]))
                    expanded_rewards_comfort = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *agent_rewards_comfort.shape[1:]))

                    # 其他指标扩展
                    expanded_mean_v = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_mean_acc = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_safety_SI = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_efficiency_ASR = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_efficiency_TFR = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_stability_SSI = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_stability_VSS = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_comfort_JIs = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_control_efficiencys = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_control_reward = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_comfort_cost = np.zeros((self.algo_args["train"]["n_rollout_threads"],))
                    expanded_comfort_reward = np.zeros((self.algo_args["train"]["n_rollout_threads"],))

                    # 填充实际数据
                    for i, env_idx in enumerate(scene_env_indices):
                        if i < len(agent_obs):
                            expanded_obs[i] = agent_obs[i]
                            expanded_rnn_states[i] = agent_rnn_states[i]
                            expanded_actions[i] = agent_actions[i]
                            expanded_action_log_probs[i] = agent_action_log_probs[i]
                            expanded_action_losss[i] = agent_action_losss[i]
                            expanded_reward_weights[i] = agent_reward_weights[i]
                            expanded_masks[i] = agent_masks[i]
                            expanded_active_masks[i] = agent_active_masks[i]

                            # 填充奖励组件
                            expanded_rewards_safety[i] = agent_rewards_safety[i]
                            expanded_rewards_stability[i] = agent_rewards_stability[i]
                            expanded_rewards_efficiency[i] = agent_rewards_efficiency[i]
                            expanded_rewards_comfort[i] = agent_rewards_comfort[i]

                            # 填充其他指标
                            expanded_mean_v[i] = mean_v[env_idx]
                            expanded_mean_acc[i] = mean_acc[env_idx]
                            expanded_safety_SI[i] = safety_SI[env_idx]
                            expanded_efficiency_ASR[i] = efficiency_ASR[env_idx]
                            expanded_efficiency_TFR[i] = efficiency_TFR[env_idx]
                            expanded_stability_SSI[i] = stability_SSI[env_idx]
                            expanded_stability_VSS[i] = stability_VSS[env_idx]
                            expanded_comfort_JIs[i] = comfort_JIs[env_idx]
                            expanded_control_efficiencys[i] = control_efficiencys[env_idx]
                            expanded_control_reward[i] = control_reward[env_idx]
                            expanded_comfort_cost[i] = comfort_cost[env_idx]
                            expanded_comfort_reward[i] = comfort_reward[env_idx]

                    # 插入到场景特定的actor buffer
                    self.scene_actor_buffers[scene_id][agent_id].insert(
                        expanded_obs,
                        expanded_rnn_states,
                        expanded_actions,
                        expanded_action_log_probs,
                        expanded_action_losss,
                        expanded_mean_v,
                        expanded_mean_acc,
                        expanded_reward_weights,
                        expanded_rewards_safety,
                        expanded_rewards_stability,
                        expanded_rewards_efficiency,
                        expanded_rewards_comfort,
                        expanded_safety_SI,
                        expanded_efficiency_ASR,
                        expanded_efficiency_TFR,
                        expanded_stability_SSI,
                        expanded_stability_VSS,
                        expanded_comfort_JIs,
                        expanded_control_efficiencys,
                        expanded_control_reward,
                        expanded_comfort_cost,
                        expanded_comfort_reward,
                        expanded_masks,
                        expanded_active_masks,
                        agent_available_actions,
                    )

            # 插入critic buffer数据
            if self.state_type == "EP" and len(scene_env_indices) > 0:
                # 准备场景特定的critic数据
                scene_share_obs = share_obs[scene_mask, 0] if np.any(scene_mask) else np.array([])
                scene_values = values[scene_mask] if np.any(scene_mask) else np.array([])
                scene_reward_components = reward_components_array[scene_mask] if np.any(scene_mask) else np.array([])
                scene_masks = masks[scene_mask, 0] if np.any(scene_mask) else np.array([])
                scene_bad_masks = bad_masks[scene_mask] if np.any(scene_mask) else np.array([])

                if len(scene_share_obs) > 0:
                    # 扩展数据
                    expanded_share_obs = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *scene_share_obs.shape[1:]))
                    expanded_rnn_states_critic = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], *scene_rnn_states_critic.shape[1:]))
                    expanded_values = np.zeros((self.algo_args["train"]["n_rollout_threads"], self.critic_head_num))
                    expanded_reward_components = np.zeros(
                        (self.algo_args["train"]["n_rollout_threads"], self.critic_head_num))
                    expanded_masks = np.zeros((self.algo_args["train"]["n_rollout_threads"], 1))
                    expanded_bad_masks = np.zeros((self.algo_args["train"]["n_rollout_threads"], 1))

                    # 填充数据
                    for i, env_idx in enumerate(scene_env_indices):
                        if i < len(scene_share_obs):
                            expanded_share_obs[i] = scene_share_obs[i]
                            expanded_rnn_states_critic[i] = scene_rnn_states_critic[i]
                            expanded_values[i] = scene_values[i]
                            expanded_reward_components[i] = np.mean(scene_reward_components[i], axis=0)  # 对agent维度求平均
                            expanded_masks[i] = scene_masks[i]
                            expanded_bad_masks[i] = scene_bad_masks[i]

                    # 插入到场景特定的critic buffer
                    self.scene_critic_buffers[scene_id].insert(
                        expanded_share_obs,
                        expanded_rnn_states_critic,
                        expanded_values,
                        expanded_reward_components,
                        expanded_masks,
                        expanded_bad_masks,
                    )

        # 更新场景特定的奖励统计用于动态权重计算
        for scene_id in range(self.num_scene_classes):
            scene_mask = (classified_scene_mark == scene_id)
            if np.any(scene_mask):
                scene_reward_components = reward_components_array[scene_mask]
                if len(scene_reward_components) > 0:
                    # 对agent维度求平均
                    mean_scene_rewards = np.mean(scene_reward_components, axis=1)  # [n_scene_envs, critic_head_num]
                    self.scene_critics[scene_id].update_reward_statistics(mean_scene_rewards)

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages for all scenes."""
        for scene_id in range(self.num_scene_classes):
            # 计算critic的最后一个state的值
            if self.state_type == "EP":
                next_values, _ = self.scene_critics[scene_id].get_values(
                    self.scene_critic_buffers[scene_id].share_obs[-1],
                    self.scene_critic_buffers[scene_id].rnn_states_critic[-1],
                    self.scene_critic_buffers[scene_id].masks[-1],
                )
                next_values = _t2n(next_values)

            # 计算returns
            value_normalizer = self.value_normalizers[scene_id] if hasattr(self, 'value_normalizers') else None
            self.scene_critic_buffers[scene_id].compute_returns(next_values, value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError("train method should be implemented in subclass")

    def after_update(self):
        """Do the necessary data operations after an update for all scenes."""
        for scene_id in range(self.num_scene_classes):
            for agent_id in range(self.num_agents):
                self.scene_actor_buffers[scene_id][agent_id].after_update()
            self.scene_critic_buffers[scene_id].after_update()

    def eval(self):
        """Evaluate all scene models."""
        # 这个方法在子类中实现
        raise NotImplementedError("eval method should be implemented in subclass")

    def render(self):
        """Render with multi-scene support."""
        # 这个方法在子类中实现
        raise NotImplementedError("render method should be implemented in subclass")

    def prep_rollout(self):
        """Prepare for rollout for all scenes."""
        for scene_id in range(self.num_scene_classes):
            for agent_id in range(self.num_agents):
                self.scene_actors[scene_id][agent_id].prep_rollout()
            self.scene_critics[scene_id].prep_rollout()

    def prep_training(self):
        """Prepare for training for all scenes."""
        for scene_id in range(self.num_scene_classes):
            for agent_id in range(self.num_agents):
                self.scene_actors[scene_id][agent_id].prep_training()
            self.scene_critics[scene_id].prep_training()

    def save(self):
        """Save model parameters for all scenes."""
        for scene_id in range(self.num_scene_classes):
            scene_save_dir = os.path.join(self.save_dir, f"scene_{scene_id}")
            os.makedirs(scene_save_dir, exist_ok=True)

            for agent_id in range(self.num_agents):
                policy_actor = self.scene_actors[scene_id][agent_id].actor
                torch.save(
                    policy_actor.state_dict(),
                    os.path.join(scene_save_dir, f"actor_agent{agent_id}.pt"),
                )

            policy_critic = self.scene_critics[scene_id].critic
            torch.save(
                policy_critic.state_dict(),
                os.path.join(scene_save_dir, "critic.pt")
            )

            if hasattr(self, 'value_normalizers') and self.value_normalizers[scene_id] is not None:
                torch.save(
                    self.value_normalizers[scene_id].state_dict(),
                    os.path.join(scene_save_dir, "value_normalizer.pt")
                )

    def save_good_model(self, current_timestep):
        """Save good model when performance is good."""
        for scene_id in range(self.num_scene_classes):
            scene_save_dir = os.path.join(self.save_dir, "good_model", f"scene_{scene_id}")
            os.makedirs(scene_save_dir, exist_ok=True)

            if self.share_param:
                policy_actor = self.scene_actors[scene_id][0].actor
                torch.save(
                    policy_actor.state_dict(),
                    os.path.join(scene_save_dir, f"actor_agent0_{current_timestep}.pt"),
                )
            else:
                for agent_id in range(self.num_agents):
                    policy_actor = self.scene_actors[scene_id][agent_id].actor
                    torch.save(
                        policy_actor.state_dict(),
                        os.path.join(scene_save_dir, f"actor_agent{agent_id}_{current_timestep}.pt"),
                    )

            policy_critic = self.scene_critics[scene_id].critic
            torch.save(
                policy_critic.state_dict(),
                os.path.join(scene_save_dir, f"critic_{current_timestep}.pt")
            )

            if hasattr(self, 'value_normalizers') and self.value_normalizers[scene_id] is not None:
                torch.save(
                    self.value_normalizers[scene_id].state_dict(),
                    os.path.join(scene_save_dir, f"value_normalizer_{current_timestep}.pt")
                )

    def restore(self):
        """Restore model parameters for all scenes."""
        model_dir = self.algo_args["train"]["model_dir"]

        for scene_id in range(self.num_scene_classes):
            scene_model_dir = os.path.join(model_dir, f"scene_{scene_id}")

            if not os.path.exists(scene_model_dir):
                print(f"Warning: Model directory for scene {scene_id} not found: {scene_model_dir}")
                continue

            # 恢复actors
            if self.share_param:
                actor_path = os.path.join(scene_model_dir, "actor_agent0.pt")
                if os.path.exists(actor_path):
                    policy_actor_state_dict = torch.load(actor_path, weights_only=True)
                    for agent_id in range(self.num_agents):
                        self.scene_actors[scene_id][agent_id].actor.load_state_dict(policy_actor_state_dict)
            else:
                for agent_id in range(self.num_agents):
                    actor_path = os.path.join(scene_model_dir, f"actor_agent{agent_id}.pt")
                    if os.path.exists(actor_path):
                        policy_actor_state_dict = torch.load(actor_path, weights_only=True)
                        self.scene_actors[scene_id][agent_id].actor.load_state_dict(policy_actor_state_dict)

            # 恢复critic
            if not self.algo_args["render"]["use_render"]:
                critic_path = os.path.join(scene_model_dir, "critic.pt")
                if os.path.exists(critic_path):
                    policy_critic_state_dict = torch.load(critic_path, weights_only=True)
                    self.scene_critics[scene_id].critic.load_state_dict(policy_critic_state_dict)

                # 恢复value normalizer
                if hasattr(self, 'value_normalizers') and self.value_normalizers[scene_id] is not None:
                    vn_path = os.path.join(scene_model_dir, "value_normalizer.pt")
                    if os.path.exists(vn_path):
                        value_normalizer_state_dict = torch.load(vn_path, weights_only=True)
                        self.value_normalizers[scene_id].load_state_dict(value_normalizer_state_dict)


    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()

    def get_scene_statistics(self):
        """Get comprehensive scene statistics for monitoring."""
        stats = {
            'scene_distribution': self._get_scene_distribution(),
            'total_steps': self.total_steps,
            'scene_counts': dict(self.scene_stats),
            'num_scene_classes': self.num_scene_classes,
            'scene_centers': self.scene_centers
        }

        # 添加每个场景的动态权重信息
        for scene_id in range(self.num_scene_classes):
            if hasattr(self, 'scene_critics') and scene_id < len(self.scene_critics):
                scene_info = self.scene_critics[scene_id].get_statistics_info()
                stats[f'scene_{scene_id}'] = scene_info

        return stats

    def _get_current_episode_scene_distribution(self):
        """Get current episode scene distribution."""
        if self.current_episode_total_steps == 0:
            return {i: 0.0 for i in range(self.num_scene_classes)}

        distribution = {}
        for scene_id in range(self.num_scene_classes):
            distribution[scene_id] = self.current_episode_scene_count[scene_id] / self.current_episode_total_steps

        return distribution

    def _reset_episode_scene_stats(self):
        """Reset episode-level scene statistics."""
        self.current_episode_scene_count = defaultdict(int)
        self.current_episode_total_steps = 0

    def _update_scene_episode_stats(self, episode_rewards_by_scene=None):
        """Update scene-level episode statistics."""
        for scene_id in range(self.num_scene_classes):
            if self.current_episode_scene_count[scene_id] > 0:
                self.scene_episode_stats[scene_id] += 1

                # 如果提供了奖励信息，也记录下来
                if episode_rewards_by_scene and scene_id in episode_rewards_by_scene:
                    self.scene_reward_stats[scene_id].append(episode_rewards_by_scene[scene_id])

    def get_comprehensive_scene_statistics(self):
        """Get comprehensive scene statistics for monitoring."""
        stats = {
            'scene_distribution': self._get_scene_distribution(),
            'current_episode_distribution': self._get_current_episode_scene_distribution(),
            'total_steps': self.total_steps,
            'scene_step_counts': dict(self.scene_step_stats),
            'scene_episode_counts': dict(self.scene_episode_stats),
            'num_scene_classes': self.num_scene_classes,
            'scene_centers': self.scene_centers
        }

        # 添加每个场景的平均奖励统计
        scene_avg_rewards = {}
        for scene_id in range(self.num_scene_classes):
            if scene_id in self.scene_reward_stats and len(self.scene_reward_stats[scene_id]) > 0:
                scene_avg_rewards[scene_id] = {
                    'mean': np.mean(self.scene_reward_stats[scene_id]),
                    'std': np.std(self.scene_reward_stats[scene_id]),
                    'count': len(self.scene_reward_stats[scene_id])
                }
            else:
                scene_avg_rewards[scene_id] = {'mean': 0.0, 'std': 0.0, 'count': 0}

        stats['scene_avg_rewards'] = scene_avg_rewards

        # 添加每个场景的动态权重信息
        scene_weights = {}
        for scene_id in range(self.num_scene_classes):
            if hasattr(self, 'scene_critics') and scene_id < len(self.scene_critics):
                scene_info = self.scene_critics[scene_id].get_statistics_info()
                scene_weights[scene_id] = scene_info.get('dynamic_weights', [])

        stats['scene_dynamic_weights'] = scene_weights

        return stats

    def _calculate_episode_rewards_by_scene(self):
        """Calculate total rewards achieved in each scene during current episode."""
        episode_rewards_by_scene = {}

        for scene_id in range(self.num_scene_classes):
            if self.current_episode_scene_count[scene_id] > 0:
                # 这里可以根据实际需要计算场景特定的奖励
                # 简化版本：使用场景参与比例作为权重
                participation_ratio = self.current_episode_scene_count[scene_id] / max(1,
                                                                                       self.current_episode_total_steps)
                episode_rewards_by_scene[scene_id] = participation_ratio * 100  # 示例奖励计算

        return episode_rewards_by_scene