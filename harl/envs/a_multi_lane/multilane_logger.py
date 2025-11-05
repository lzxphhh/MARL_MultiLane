from harl.common.base_logger import BaseLogger
import time
import numpy as np
import csv
import os
from collections import defaultdict


# TODO： 统计以下指标：
# 1. 平均速度
# 2. 每30个episode的done的原因
# 3. 每30个episode的平均长度

class MultiLaneLogger(BaseLogger):

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(MultiLaneLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None
        self.episode_lens = None
        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.train_episode_rewards = None
        self.train_episode_mean_speed = None
        self.train_episode_mean_acceleration = None
        # Add new variables for tracking evaluation metrics
        self.train_episode_safety_SI = None
        self.train_episode_efficiency_ASR = None
        self.train_episode_efficiency_TFR = None
        self.train_episode_stability_SSI = None
        self.train_episode_comfort_AI = None
        self.train_episode_comfort_JI = None
        self.train_episode_control_efficiency = None
        self.train_episode_control_reward = None
        self.train_episode_comfort_cost = None
        self.train_episode_comfort_reward = None

        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_rewards = None
        self.done_episodes_mean_speed = None
        self.done_episodes_mean_acceleration = None
        # Add variables for done episodes
        self.done_episodes_safety_SI = None
        self.done_episodes_efficiency_ASR = None
        self.done_episodes_efficiency_TFR = None
        self.done_episodes_stability_SSI = None
        self.done_episodes_comfort_AI = None
        self.done_episodes_comfort_JI = None
        self.done_episodes_control_efficiency = None
        self.done_episodes_control_reward = None
        self.done_episodes_comfort_cost = None
        self.done_episodes_comfort_reward = None

        self.done_episode_lens = None

        # 场景分布统计相关变量
        self.scene_distribution_history = []
        self.scene_episode_counts = defaultdict(int)
        self.scene_step_counts = defaultdict(int)
        self.scene_reward_history = defaultdict(list)

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['task']}"

    def init(self, episodes):
        # 初始化logger

        self.start = time.time()
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=int)
        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_dynamic_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_rewards_safety = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_rewards_stability = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_rewards_efficiency = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_rewards_comfort = np.zeros(self.algo_args["train"]["n_rollout_threads"])

        # Initialize new metrics
        self.train_episode_safety_SI = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_efficiency_ASR = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_efficiency_TFR = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_stability_SSI = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_comfort_AI = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_comfort_JI = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_control_efficiency = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_control_reward = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_comfort_cost = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.train_episode_comfort_reward = np.zeros(self.algo_args["train"]["n_rollout_threads"])

        self.done_episodes_rewards = np.zeros(self.n_rollout_threads)
        self.done_episode_dynamic_rewards = np.zeros(self.n_rollout_threads)
        self.done_episodes_rewards_safety = np.zeros(self.n_rollout_threads)
        self.done_episodes_rewards_stability = np.zeros(self.n_rollout_threads)
        self.done_episodes_rewards_efficiency = np.zeros(self.n_rollout_threads)
        self.done_episodes_rewards_comfort = np.zeros(self.n_rollout_threads)

        # Initialize done episodes metrics
        self.done_episodes_safety_SI = np.zeros(self.n_rollout_threads)
        self.done_episodes_efficiency_ASR = np.zeros(self.n_rollout_threads)
        self.done_episodes_efficiency_TFR = np.zeros(self.n_rollout_threads)
        self.done_episodes_stability_SSI = np.zeros(self.n_rollout_threads)
        self.done_episodes_comfort_AI = np.zeros(self.n_rollout_threads)
        self.done_episodes_comfort_JI = np.zeros(self.n_rollout_threads)
        self.done_episodes_control_efficiency = np.zeros(self.n_rollout_threads)
        self.done_episodes_control_reward = np.zeros(self.n_rollout_threads)
        self.done_episodes_comfort_cost = np.zeros(self.n_rollout_threads)
        self.done_episodes_comfort_reward = np.zeros(self.n_rollout_threads)

        self.train_episode_mean_speed = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_mean_speed = np.zeros(self.n_rollout_threads)
        self.train_episode_mean_acceleration = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_mean_acceleration = np.zeros(self.n_rollout_threads)
        self.done_episode_lens = np.zeros(self.n_rollout_threads)
        self.done_episode_infos = [{} for _ in range(self.n_rollout_threads)]
        save_csv_dir = self.run_dir + '/csv'
        self.csv_path = save_csv_dir + '/episode_info.csv'
        os.makedirs(save_csv_dir)

        # 初始化CSV文件，添加表头 - Add the new metrics to the CSV header
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timesteps',
                'collision_rate',
                'episode_length',
                'total_reward',
                'safety_reward',
                'stability_reward',
                'efficiency_reward',
                'comfort_reward',
                'mean_speed',
                'mean_acceleration',
                'safety_SI',
                'efficiency_ASR',
                'efficiency_TFR',
                'stability_SSI',
                'comfort_AI',
                'comfort_JI',
                'control_efficiency',
                'control_reward',
                'comfort_cost',
                'comfort_reward',
                'scene_0_distribution',
                'scene_1_distribution',
                'scene_2_distribution',
                'scene_3_distribution',
                'scene_4_distribution',
                'scene_0_episodes',
                'scene_1_episodes',
                'scene_2_episodes',
                'scene_3_episodes',
                'scene_4_episodes'
            ])

        pass

    def episode_init(self, episode):
        # 每个episode开始的时候更新logger里面的episode index

        """Initialize the logger for each episode."""
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""

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
            comfort_AI,
            comfort_JI,
            control_efficiency,
            control_reward,
            comfort_cost,
            comfort_reward,
            classified_scene_mark,
        ) = data

        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)
        # 并行环境中的每个环境的step reward （n_env_threads, ）
        reward_env = np.mean(rewards, axis=1).flatten()
        dynamic_rewards = np.zeros_like(rewards)
        all_rewards = np.concatenate([
            rewards_safety,  # 对应 reward_weights 的第 0 列
            rewards_efficiency,  # 对应 reward_weights 的第 1 列
            rewards_stability,  # 对应 reward_weights 的第 2 列
            rewards_comfort  # 对应 reward_weights 的第 3 列
        ], axis=2)
        for agent_id in range(self.num_agents):
            dynamic_rewards[:, agent_id, 0] = np.sum(
                reward_weights[:, agent_id, :] * all_rewards[:, agent_id, :],
                axis=1
            ).squeeze()
        dynamic_reward_env = np.mean(dynamic_rewards, axis=1).flatten()
        reward_safety = np.mean(rewards_safety, axis=1).flatten()
        reward_stability = np.mean(rewards_stability, axis=1).flatten()
        reward_efficiency = np.mean(rewards_efficiency, axis=1).flatten()
        reward_comfort = np.mean(rewards_comfort, axis=1).flatten()
        speeds_env = mean_v
        acceleration_env = mean_acc
        # 并行环境中的每个环境的episode reward （n_env_threads, ）累积
        self.train_episode_rewards += reward_env
        self.train_episode_dynamic_rewards += dynamic_reward_env
        self.train_episode_rewards_safety += reward_safety
        self.train_episode_rewards_stability += reward_stability
        self.train_episode_rewards_efficiency += reward_efficiency
        self.train_episode_rewards_comfort += reward_comfort
        self.train_episode_mean_speed += speeds_env
        self.train_episode_mean_acceleration += acceleration_env

        self.train_episode_safety_SI += safety_SI
        self.train_episode_efficiency_ASR += efficiency_ASR
        self.train_episode_efficiency_TFR += efficiency_TFR
        self.train_episode_stability_SSI += stability_SSI
        self.train_episode_comfort_AI += comfort_AI
        self.train_episode_comfort_JI += comfort_JI
        self.train_episode_control_efficiency += control_efficiency
        self.train_episode_control_reward += control_reward
        self.train_episode_comfort_cost += comfort_cost
        self.train_episode_comfort_reward += comfort_reward

        # 并行环境中的每个环境的episode len （n_env_threads, ）累积
        for t in range(self.n_rollout_threads):
            if infos[t][0]['step_time'] == 0.2:
                self.one_episode_len[t] = 0
            self.one_episode_len[t] += 1

        if classified_scene_mark is not None:
            # 统计当前step的场景分布
            for t in range(self.n_rollout_threads):
                scene_id = classified_scene_mark[t] if hasattr(classified_scene_mark,
                                                               '__len__') else classified_scene_mark
                scene_id = max(0, min(scene_id, 4))  # 确保在有效范围内
                self.scene_step_counts[scene_id] += 1

        for t in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_rewards[t] = self.train_episode_rewards[t]
                self.done_episode_dynamic_rewards[t] = self.train_episode_dynamic_rewards[t]
                self.done_episodes_rewards_safety[t] = self.train_episode_rewards_safety[t]
                self.done_episodes_rewards_stability[t] = self.train_episode_rewards_stability[t]
                self.done_episodes_rewards_efficiency[t] = self.train_episode_rewards_efficiency[t]
                self.done_episodes_rewards_comfort[t] = self.train_episode_rewards_comfort[t]

                # Add the evaluation metrics to the done episodes metrics
                self.done_episodes_safety_SI[t] = self.train_episode_safety_SI[t] / self.one_episode_len[t]
                self.done_episodes_efficiency_ASR[t] = self.train_episode_efficiency_ASR[t] / self.one_episode_len[t]
                self.done_episodes_efficiency_TFR[t] = self.train_episode_efficiency_TFR[t] / self.one_episode_len[t]
                self.done_episodes_stability_SSI[t] = self.train_episode_stability_SSI[t] / self.one_episode_len[t]
                self.done_episodes_comfort_AI[t] = self.train_episode_comfort_AI[t] / self.one_episode_len[t]
                self.done_episodes_comfort_JI[t] = self.train_episode_comfort_JI[t] / self.one_episode_len[t]
                self.done_episodes_control_efficiency[t] = self.train_episode_control_efficiency[t] / self.one_episode_len[t]
                self.done_episodes_control_reward[t] = self.train_episode_control_reward[t] / self.one_episode_len[t]
                self.done_episodes_comfort_cost[t] = self.train_episode_comfort_cost[t] / self.one_episode_len[t]
                self.done_episodes_comfort_reward[t] = self.train_episode_comfort_reward[t] / self.one_episode_len[t]

                self.train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward
                self.train_episode_dynamic_rewards[t] = 0
                self.train_episode_rewards_safety[t] = 0  # 归零这个以及done的episode的reward
                self.train_episode_rewards_stability[t] = 0  # 归零这个以及done的episode的reward
                self.train_episode_rewards_efficiency[t] = 0  # 归零这个以及done的episode的reward
                self.train_episode_rewards_comfort[t] = 0  # 归零这个以及done的episode的reward

                # Reset the evaluation metrics
                self.train_episode_safety_SI[t] = 0
                self.train_episode_efficiency_ASR[t] = 0
                self.train_episode_efficiency_TFR[t] = 0
                self.train_episode_stability_SSI[t] = 0
                self.train_episode_comfort_AI[t] = 0
                self.train_episode_comfort_JI[t] = 0
                self.train_episode_control_efficiency[t] = 0
                self.train_episode_control_reward[t] = 0
                self.train_episode_comfort_cost[t] = 0
                self.train_episode_comfort_reward[t] = 0

                self.done_episodes_mean_speed[t] = self.train_episode_mean_speed[t] / self.one_episode_len[t]
                self.train_episode_mean_speed[t] = 0  # 归零这个以及done的episode的reward
                self.done_episodes_mean_acceleration[t] = self.train_episode_mean_acceleration[t] / self.one_episode_len[t]
                self.train_episode_mean_acceleration[t] = 0  # 归零这个以及done的episode的reward

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos[t] = infos[t][0]

                # 存一下这个已经done的episode的episode长度
                self.done_episode_lens[t] = self.one_episode_len[t]
                self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度
                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                if self.done_episode_infos[t]['step_time'] != ((self.done_episode_lens[t] + 1) / 10):
                    print(f'episode len not match: {self.done_episode_infos[t]["step_time"]} vs {((self.done_episode_lens[t] + 1) / 10)}')
                # assert self.done_episode_infos[t]['step_time'] == ((self.done_episode_lens[t]+1) / 10), 'episode len not match'

    def episode_log(
            self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer,
            save_collision, save_episode_value
    ):
        """Log information for each episode."""

        # 记录训练结束时间
        self.end = time.time()

        # 当前跑了多少time steps
        self.total_num_steps = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()

        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # 记录每个episode的平均avergae reward 和 average step和collision rate
        collision_count = 0
        done_episode_lens = []
        for each_env_info in self.done_episode_infos:
            if each_env_info['done_reason'] == 'collision':
                collision_count += 1
            else:
                done_episode_lens.append(each_env_info['step_time'] - 1)
        average_collision_rate = collision_count / len(self.done_episode_infos)
        average_episode_return = np.mean(self.done_episodes_rewards)
        average_episode_real_return = np.mean(self.done_episode_dynamic_rewards)
        average_episode_return_safety = np.mean(self.done_episodes_rewards_safety)
        average_episode_return_stability = np.mean(self.done_episodes_rewards_stability)
        average_episode_return_efficiency = np.mean(self.done_episodes_rewards_efficiency)
        average_episode_return_comfort = np.mean(self.done_episodes_rewards_comfort)
        average_episode_speed = np.mean(self.done_episodes_mean_speed)
        average_episode_acceleration = np.mean(self.done_episodes_mean_acceleration)
        average_episode_step = np.mean(self.done_episode_lens)

        # Calculate averages for new metrics
        average_episode_safety_SI = np.mean(self.done_episodes_safety_SI)
        average_episode_efficiency_ASR = np.mean(self.done_episodes_efficiency_ASR)
        average_episode_efficiency_TFR = np.mean(self.done_episodes_efficiency_TFR)
        average_episode_stability_SSI = np.mean(self.done_episodes_stability_SSI)
        average_episode_comfort_AI = np.mean(self.done_episodes_comfort_AI)
        average_episode_comfort_JI = np.mean(self.done_episodes_comfort_JI)
        average_episode_control_efficiency = np.mean(self.done_episodes_control_efficiency)
        average_episode_control_reward = np.mean(self.done_episodes_control_reward)
        average_episode_comfort_cost = np.mean(self.done_episodes_comfort_cost)
        average_episode_comfort_reward = np.mean(self.done_episodes_comfort_reward)

        # average_episode_step = np.mean(done_episode_lens) if done_episode_lens else 0

        # self.writter.add_scalars(
        #     "average_collision_rate",
        #     {"average_collision_rate": average_collision_rate},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "success_rate/00_average_collision_rate", average_collision_rate, self.total_num_steps
        )
        print(
            "Some episodes done, average collision rate is {}.\n".format(
                average_collision_rate
            )
        )

        # self.writter.add_scalars(
        #     "average_episode_length",
        #     {"average_episode_length": average_episode_step},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "success_rate/01_average_episode_length", average_episode_step, self.total_num_steps
        )

        print(
            "Some episodes done, average episode length is {}.\n".format(
                average_episode_step
            )
        )
        #
        # print(
        #     "Some episodes done, average episode reward (fixed weights) is {}.\n".format(
        #         average_episode_return
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode reward (dynamic weights) is {}.\n".format(
        #         average_episode_real_return
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode safety reward is {}.\n".format(
        #         average_episode_return_safety
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode stability reward is {}.\n".format(
        #         average_episode_return_stability
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode efficiency reward is {}.\n".format(
        #         average_episode_return_efficiency
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode comfort reward is {}.\n".format(
        #         average_episode_return_comfort
        #     )
        # )

        print(
            "Some episodes done, average episode speed is {}.\n".format(
                average_episode_speed
            )
        )
        #
        # print(
        #     "Some episodes done, average episode acceleration is {}.\n".format(
        #         average_episode_acceleration
        #     )
        # )
        #
        # # Add print statements for the new metrics
        # print(
        #     "Some episodes done, average episode safety_SI is {}.\n".format(
        #         average_episode_safety_SI
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode efficiency_ASR is {}.\n".format(
        #         average_episode_efficiency_ASR
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode efficiency_TFR is {}.\n".format(
        #         average_episode_efficiency_TFR
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode stability_SSI is {}.\n".format(
        #         average_episode_stability_SSI
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode comfort_AI is {}.\n".format(
        #         average_episode_comfort_AI
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode comfort_JI is {}.\n".format(
        #         average_episode_comfort_JI
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode control_efficiency is {}.\n".format(
        #         average_episode_control_efficiency
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode control_reward is {}.\n".format(
        #         average_episode_control_reward
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode comfort_cost is {}.\n".format(
        #         average_episode_comfort_cost
        #     )
        # )
        #
        # print(
        #     "Some episodes done, average episode comfort_reward is {}.\n".format(
        #         average_episode_comfort_reward
        #     )
        # )

        # self.writter.add_scalars(
        #     "train_episode_rewards",
        #     {"aver_rewards": average_episode_return},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "train_episode_rewards/00_total_fixed_weight", average_episode_return, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/00_total_dynamic_weight", average_episode_real_return, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/01_safety", average_episode_return_safety, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/02_efficiency", average_episode_return_efficiency, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/03_stability", average_episode_return_stability, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/04_comfort", average_episode_return_comfort, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_motion/00_speed", average_episode_speed, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_motion/01_acceleration", average_episode_acceleration, self.total_num_steps
        )

        # Add tensorboard tracking for the new metrics
        synthesis_score = 0.3 * average_episode_safety_SI \
                        + 0.3 * (0.5 * average_episode_efficiency_ASR + 0.5 * average_episode_efficiency_TFR) \
                        + 0.3 * average_episode_stability_SSI  \
                        + 0.1 * average_episode_comfort_JI
        self.writter.add_scalar(
            "evaluation/00_synthesis_score", synthesis_score, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/01_safety_SI", average_episode_safety_SI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/02_efficiency_ASR", average_episode_efficiency_ASR, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/03_efficiency_TFR", average_episode_efficiency_TFR, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/04_stability_SSI", average_episode_stability_SSI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/05_comfort_AI", average_episode_comfort_AI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/06_comfort_JI", average_episode_comfort_JI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/07_control_efficiency", average_episode_control_efficiency, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/08_control_reward", average_episode_control_reward, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/09_comfort_cost", average_episode_comfort_cost, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/10_comfort_reward", average_episode_comfort_reward, self.total_num_steps
        )

        # 记录每个episode的平均 step reward
        critic_train_info["average_step_rewards"] = critic_buffer.get_mean_rewards()
        self.log_train(actor_train_infos, critic_train_info)
        print(
            "Average step reward is {}.\n".format(
                critic_train_info["average_step_rewards"]
            )
        )
        # self.writter.add_scalars(
        #     "average_step_rewards",
        #     {"average_step_rewards": critic_train_info["average_step_rewards"]},
        #     self.total_num_steps,
        # )
        self.writter.add_scalar(
            "average_step_rewards", critic_train_info["average_step_rewards"], self.total_num_steps
        )

        # # 记录各分量的权重比例
        # total_weight = (self.env_args["reward_weights"]["safety"] +
        #                 self.env_args["reward_weights"]["efficiency"] +
        #                 self.env_args["reward_weights"]["stability"] +
        #                 self.env_args["reward_weights"]["comfort"])
        #
        # self.writter.add_scalar(
        #     "reward_weights/safety",
        #     self.env_args["reward_weights"]["safety"] / total_weight,
        #     self.total_num_steps
        # )
        # self.writter.add_scalar(
        #     "reward_weights/efficiency",
        #     self.env_args["reward_weights"]["efficiency"] / total_weight,
        #     self.total_num_steps
        # )
        # self.writter.add_scalar(
        #     "reward_weights/stability",
        #     self.env_args["reward_weights"]["stability"] / total_weight,
        #     self.total_num_steps
        # )
        # self.writter.add_scalar(
        #     "reward_weights/comfort",
        #     self.env_args["reward_weights"]["comfort"] / total_weight,
        #     self.total_num_steps
        # )

        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                self.total_num_steps,
                average_collision_rate,
                average_episode_step,
                average_episode_return,
                average_episode_return_safety,
                average_episode_return_stability,
                average_episode_return_efficiency,
                average_episode_return_comfort,
                average_episode_speed,
                average_episode_acceleration,
                average_episode_safety_SI,
                average_episode_efficiency_ASR,
                average_episode_efficiency_TFR,
                average_episode_stability_SSI,
                average_episode_comfort_AI,
                average_episode_comfort_JI,
                average_episode_control_efficiency,
                average_episode_control_reward,
                average_episode_comfort_cost,
                average_episode_comfort_reward
            ])

        # Multi-objective monitoring summary
        if hasattr(self, '_runner_ref') and hasattr(self._runner_ref, 'multi_monitor'):
            try:
                from harl.utils.multi_objective_monitor import log_summary
                log_summary(self._runner_ref.multi_monitor, self.episode, self.total_num_steps)
            except Exception as e:
                print(f"⚠️ Multi-objective logging failed: {e}")

        if average_collision_rate <= save_collision and average_episode_speed >= save_episode_value:
            return True, self.total_num_steps
        else:
            return False, self.total_num_steps
        # if average_collision_rate <= save_collision and average_episode_step <= save_episode_step:
        #     return True, self.total_num_steps
        # else:
        #     return False, self.total_num_steps

    # 在原有的 episode_log 方法后添加以下新方法：

    def episode_mp_log(
            self, actor_train_infos, critic_train_info, scene_actor_buffers, scene_critic_buffers,
            save_collision, save_episode_value
    ):
        """Log information for each episode - multi-scene multi-policy version.

        Args:
            actor_train_infos: (list) actor training information
            critic_train_info: (dict) critic training information
            scene_actor_buffers: (list) list of actor buffers for each scene
            scene_critic_buffers: (list) list of critic buffers for each scene
            save_collision: (float) collision threshold for saving model
            save_episode_value: (float) episode value threshold for saving model
        """

        # 记录训练结束时间
        self.end = time.time()

        # 当前跑了多少time steps
        self.total_num_steps = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
        )

        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # 统计所有场景的collision rate和episode length
        collision_count = 0
        done_episode_lens = []
        for each_env_info in self.done_episode_infos:
            if each_env_info.get('done_reason') == 'collision':
                collision_count += 1
            else:
                done_episode_lens.append(each_env_info.get('step_time', 1) - 1)

        average_collision_rate = collision_count / len(self.done_episode_infos) if len(
            self.done_episode_infos) > 0 else 0
        average_episode_return = np.mean(self.done_episodes_rewards)
        average_episode_real_return = np.mean(self.done_episode_dynamic_rewards)
        average_episode_return_safety = np.mean(self.done_episodes_rewards_safety)
        average_episode_return_stability = np.mean(self.done_episodes_rewards_stability)
        average_episode_return_efficiency = np.mean(self.done_episodes_rewards_efficiency)
        average_episode_return_comfort = np.mean(self.done_episodes_rewards_comfort)
        average_episode_speed = np.mean(self.done_episodes_mean_speed)
        average_episode_acceleration = np.mean(self.done_episodes_mean_acceleration)
        average_episode_step = np.mean(self.done_episode_lens)

        # Calculate averages for new metrics
        average_episode_safety_SI = np.mean(self.done_episodes_safety_SI)
        average_episode_efficiency_ASR = np.mean(self.done_episodes_efficiency_ASR)
        average_episode_efficiency_TFR = np.mean(self.done_episodes_efficiency_TFR)
        average_episode_stability_SSI = np.mean(self.done_episodes_stability_SSI)
        average_episode_comfort_AI = np.mean(self.done_episodes_comfort_AI)
        average_episode_comfort_JI = np.mean(self.done_episodes_comfort_JI)
        average_episode_control_efficiency = np.mean(self.done_episodes_control_efficiency)
        average_episode_control_reward = np.mean(self.done_episodes_control_reward)
        average_episode_comfort_cost = np.mean(self.done_episodes_comfort_cost)
        average_episode_comfort_reward = np.mean(self.done_episodes_comfort_reward)

        # 打印基本信息
        self.writter.add_scalar(
            "success_rate/00_average_collision_rate", average_collision_rate, self.total_num_steps
        )
        print(
            "Some episodes done, average collision rate is {}.\n".format(
                average_collision_rate
            )
        )

        self.writter.add_scalar(
            "success_rate/01_average_episode_length", average_episode_step, self.total_num_steps
        )
        print(
            "Some episodes done, average episode length is {}.\n".format(
                average_episode_step
            )
        )

        print(
            "Some episodes done, average episode speed is {}.\n".format(
                average_episode_speed
            )
        )

        # 记录奖励信息
        self.writter.add_scalar(
            "train_episode_rewards/00_total_fixed_weight", average_episode_return, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/00_total_dynamic_weight", average_episode_real_return, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/01_safety", average_episode_return_safety, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/02_efficiency", average_episode_return_efficiency, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/03_stability", average_episode_return_stability, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_rewards/04_comfort", average_episode_return_comfort, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_motion/00_speed", average_episode_speed, self.total_num_steps
        )
        self.writter.add_scalar(
            "train_episode_motion/01_acceleration", average_episode_acceleration, self.total_num_steps
        )

        # 记录评估指标
        synthesis_score = 0.3 * average_episode_safety_SI \
                          + 0.3 * (0.5 * average_episode_efficiency_ASR + 0.5 * average_episode_efficiency_TFR) \
                          + 0.3 * average_episode_stability_SSI  \
                          + 0.1 * average_episode_comfort_JI
        self.writter.add_scalar(
            "evaluation/00_synthesis_score", synthesis_score, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/01_safety_SI", average_episode_safety_SI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/02_efficiency_ASR", average_episode_efficiency_ASR, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/03_efficiency_TFR", average_episode_efficiency_TFR, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/04_stability_SSI", average_episode_stability_SSI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/05_comfort_AI", average_episode_comfort_AI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/06_comfort_JI", average_episode_comfort_JI, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/07_control_efficiency", average_episode_control_efficiency, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/08_control_reward", average_episode_control_reward, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/09_comfort_cost", average_episode_comfort_cost, self.total_num_steps
        )
        self.writter.add_scalar(
            "evaluation/10_comfort_reward", average_episode_comfort_reward, self.total_num_steps
        )

        # 计算多场景的平均step rewards
        total_mean_rewards = np.zeros(len(scene_critic_buffers[0].critic_head_names))
        valid_scenes = 0

        for scene_id, critic_buffer in enumerate(scene_critic_buffers):
            try:
                scene_mean_rewards = critic_buffer.get_mean_rewards()
                total_mean_rewards += scene_mean_rewards
                valid_scenes += 1

                # 记录每个场景的step rewards
                self.writter.add_scalar(
                    f"scene_rewards/scene_{scene_id}_average_step_rewards",
                    np.mean(scene_mean_rewards),
                    self.total_num_steps
                )

                # 记录每个奖励组件的平均值
                for head_idx, (head_id, head_name) in enumerate(critic_buffer.critic_head_names.items()):
                    if head_idx < len(scene_mean_rewards):
                        self.writter.add_scalar(
                            f"scene_rewards/scene_{scene_id}_{head_name}",
                            scene_mean_rewards[head_idx],
                            self.total_num_steps
                        )

            except Exception as e:
                print(f"Warning: Failed to get mean rewards for scene {scene_id}: {e}")
                continue

        # 计算整体平均step rewards
        if valid_scenes > 0:
            average_step_rewards = total_mean_rewards / valid_scenes
            overall_average_step_rewards = np.mean(average_step_rewards)
        else:
            overall_average_step_rewards = 0.0
            average_step_rewards = total_mean_rewards

        # 添加到critic_train_info中供后续使用
        critic_train_info["average_step_rewards"] = overall_average_step_rewards
        critic_train_info["multihead_step_rewards"] = average_step_rewards
        critic_train_info["valid_scenes"] = valid_scenes

        # 获取场景分布统计信息
        if hasattr(self, '_runner_ref'):
            scene_stats = self._runner_ref.get_comprehensive_scene_statistics()
        else:
            # fallback: 从其他来源获取场景统计
            scene_stats = self._calculate_scene_stats_fallback()

        # 记录场景分布到TensorBoard
        self._log_scene_distribution_to_tensorboard(scene_stats)

        # 打印场景分布统计
        self._print_scene_distribution_stats(scene_stats)

        # 添加场景统计到训练信息
        critic_train_info.update({
            'scene_distribution': scene_stats.get('scene_distribution', {}),
            'scene_episode_counts': scene_stats.get('scene_episode_counts', {}),
            'scene_step_counts': scene_stats.get('scene_step_counts', {}),
        })

        # 预处理训练信息，确保所有值都是标量
        processed_actor_train_infos = self._process_actor_train_infos(actor_train_infos)
        processed_critic_train_info = self._process_critic_train_info(critic_train_info)

        # 记录训练信息
        self.log_train(processed_actor_train_infos, processed_critic_train_info)

        print(
            "Average step reward is {}.\n".format(
                overall_average_step_rewards
            )
        )
        print(
            "Valid scenes for training: {}.\n".format(
                valid_scenes
            )
        )

        # 记录整体step rewards
        self.writter.add_scalar(
            "average_step_rewards", overall_average_step_rewards, self.total_num_steps
        )

        # 记录多头奖励的详细信息
        for head_idx, reward_value in enumerate(average_step_rewards):
            self.writter.add_scalar(
                f"multihead_step_rewards/head_{head_idx}", reward_value, self.total_num_steps
            )

        # 记录场景分布信息（如果有的话）
        if hasattr(critic_train_info, 'scene_distribution'):
            for scene_id, proportion in critic_train_info['scene_distribution'].items():
                self.writter.add_scalar(
                    f"scene_distribution/scene_{scene_id}", proportion, self.total_num_steps
                )

        # 记录动态权重信息
        if 'dynamic_weights' in critic_train_info:
            dynamic_weights = critic_train_info['dynamic_weights']
            if isinstance(dynamic_weights, list) and len(dynamic_weights) > 0:
                for i, weight in enumerate(dynamic_weights):
                    self.writter.add_scalar(
                        f"dynamic_weights/weight_{i}", weight, self.total_num_steps
                    )

        # 记录各个场景的训练信息
        if isinstance(actor_train_infos, list) and len(actor_train_infos) > 0:
            for agent_id, actor_info in enumerate(actor_train_infos):
                if isinstance(actor_info, dict):
                    for key, value in actor_info.items():
                        if isinstance(value, (int, float)):
                            self.writter.add_scalar(
                                f"actor_training/agent_{agent_id}_{key}", value, self.total_num_steps
                            )

        # 保存到CSV文件时包含场景分布信息
        scene_distributions = scene_stats.get('scene_distribution', {})
        scene_episode_counts = scene_stats.get('scene_episode_counts', {})

        # 保存到CSV文件
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                self.total_num_steps,
                average_collision_rate,
                average_episode_step,
                average_episode_return,
                average_episode_return_safety,
                average_episode_return_stability,
                average_episode_return_efficiency,
                average_episode_return_comfort,
                average_episode_speed,
                average_episode_acceleration,
                average_episode_safety_SI,
                average_episode_efficiency_ASR,
                average_episode_efficiency_TFR,
                average_episode_stability_SSI,
                average_episode_comfort_AI,
                average_episode_comfort_JI,
                average_episode_control_efficiency,
                average_episode_control_reward,
                average_episode_comfort_cost,
                average_episode_comfort_reward,
                scene_distributions.get(0, 0.0),    # 场景分布信息
                scene_distributions.get(1, 0.0),
                scene_distributions.get(2, 0.0),
                scene_distributions.get(3, 0.0),
                scene_distributions.get(4, 0.0),
                scene_episode_counts.get(0, 0),
                scene_episode_counts.get(1, 0),
                scene_episode_counts.get(2, 0),
                scene_episode_counts.get(3, 0),
                scene_episode_counts.get(4, 0)
            ])

        # # Multi-objective monitoring summary
        # if hasattr(self, '_runner_ref') and hasattr(self._runner_ref, 'multi_monitor'):
        #     try:
        #         from harl.utils.multi_objective_monitor import log_summary
        #         log_summary(self._runner_ref.multi_monitor, self.episode, self.total_num_steps)
        #     except Exception as e:
        #         print(f"⚠️ Multi-objective logging failed: {e}")

        # 决定是否保存模型
        if average_collision_rate <= save_collision and average_episode_speed >= save_episode_value:
            return True, self.total_num_steps
        else:
            return False, self.total_num_steps

    def eval_init(self):
        """Initialize evaluation logging."""
        # 重置评估相关的变量
        self.eval_episode_rewards = []
        self.eval_episode_lengths = []
        self.eval_collision_count = 0
        self.eval_total_episodes = 0

    def eval_per_step(self, eval_data):
        """Log evaluation data per step."""
        # 可以根据需要添加评估期间的步级记录
        pass

    def eval_thread_done(self, eval_i):
        """Called when an evaluation thread is done."""
        # 可以根据需要添加评估线程完成时的处理
        pass

    def eval_log(self, eval_episode):
        """Log evaluation results."""
        if hasattr(self, 'eval_episode_rewards') and len(self.eval_episode_rewards) > 0:
            avg_eval_reward = np.mean(self.eval_episode_rewards)
            avg_eval_length = np.mean(self.eval_episode_lengths)
            eval_collision_rate = self.eval_collision_count / self.eval_total_episodes if self.eval_total_episodes > 0 else 0

            self.writter.add_scalar(
                "eval/average_episode_reward", avg_eval_reward, self.total_num_steps
            )
            self.writter.add_scalar(
                "eval/average_episode_length", avg_eval_length, self.total_num_steps
            )
            self.writter.add_scalar(
                "eval/collision_rate", eval_collision_rate, self.total_num_steps
            )

            print(f"Evaluation completed: {eval_episode} episodes")
            print(f"Average evaluation reward: {avg_eval_reward}")
            print(f"Average evaluation length: {avg_eval_length}")
            print(f"Evaluation collision rate: {eval_collision_rate}")

    def close(self):
        """Close the logger."""
        if hasattr(self, 'writter'):
            self.writter.close()
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

    def _process_actor_train_infos(self, actor_train_infos):
        """Process actor training info to ensure all values are scalars."""
        if not isinstance(actor_train_infos, list):
            return actor_train_infos

        processed_infos = []
        for info in actor_train_infos:
            if isinstance(info, dict):
                processed_info = {}
                for key, value in info.items():
                    if isinstance(value, (list, np.ndarray)):
                        # 如果是列表或数组，取平均值
                        processed_info[key] = float(np.mean(value))
                    elif isinstance(value, (int, float, np.number)):
                        processed_info[key] = float(value)
                    else:
                        # 跳过无法处理的类型
                        continue
                processed_infos.append(processed_info)
            else:
                processed_infos.append(info)

        return processed_infos

    def _process_critic_train_info(self, critic_train_info):
        """Process critic training info to ensure all values are scalars."""
        if not isinstance(critic_train_info, dict):
            return critic_train_info

        processed_info = {}
        for key, value in critic_train_info.items():
            if key == 'value_loss' and isinstance(value, list):
                # 对于value_loss列表，记录平均值和各头的损失
                processed_info['value_loss'] = float(np.mean(value))
                for i, head_loss in enumerate(value):
                    processed_info[f'value_loss_head_{i}'] = float(head_loss)
            elif key == 'dynamic_weights' and isinstance(value, (list, np.ndarray)):
                # 对于动态权重，记录各个权重
                for i, weight in enumerate(value):
                    processed_info[f'dynamic_weight_{i}'] = float(weight)
                # 也记录权重的平均值和标准差
                processed_info['dynamic_weights_mean'] = float(np.mean(value))
                processed_info['dynamic_weights_std'] = float(np.std(value))
            elif key == 'multihead_step_rewards' and isinstance(value, (list, np.ndarray)):
                # 对于多头step rewards，记录各个头的奖励
                for i, reward in enumerate(value):
                    processed_info[f'step_reward_head_{i}'] = float(reward)
            elif isinstance(value, (list, np.ndarray)):
                # 其他列表或数组类型，取平均值
                processed_info[key] = float(np.mean(value))
            elif isinstance(value, (int, float, np.number)):
                processed_info[key] = float(value)
            elif isinstance(value, str):
                # 字符串类型不记录到TensorBoard
                continue
            else:
                # 其他类型尝试转换为浮点数
                try:
                    processed_info[key] = float(value)
                except (ValueError, TypeError):
                    # 无法转换的类型跳过
                    continue

        return processed_info

    def _log_scene_distribution_to_tensorboard(self, scene_stats):
        """Log scene distribution statistics to TensorBoard."""
        scene_distribution = scene_stats.get('scene_distribution', {})
        scene_episode_counts = scene_stats.get('scene_episode_counts', {})
        scene_step_counts = scene_stats.get('scene_step_counts', {})
        scene_avg_rewards = scene_stats.get('scene_avg_rewards', {})
        scene_dynamic_weights = scene_stats.get('scene_dynamic_weights', {})

        # 记录场景分布比例
        for scene_id, proportion in scene_distribution.items():
            self.writter.add_scalar(
                f"scene_distribution/scene_{scene_id}_proportion",
                proportion,
                self.total_num_steps
            )

        # 记录场景episode计数
        for scene_id, count in scene_episode_counts.items():
            self.writter.add_scalar(
                f"scene_episodes/scene_{scene_id}_episodes",
                count,
                self.total_num_steps
            )

        # 记录场景step计数
        for scene_id, count in scene_step_counts.items():
            self.writter.add_scalar(
                f"scene_steps/scene_{scene_id}_steps",
                count,
                self.total_num_steps
            )

        # 记录场景平均奖励
        for scene_id, reward_info in scene_avg_rewards.items():
            if reward_info['count'] > 0:
                self.writter.add_scalar(
                    f"scene_rewards/scene_{scene_id}_mean_reward",
                    reward_info['mean'],
                    self.total_num_steps
                )
                self.writter.add_scalar(
                    f"scene_rewards/scene_{scene_id}_reward_std",
                    reward_info['std'],
                    self.total_num_steps
                )

        # 记录场景动态权重
        for scene_id, weights in scene_dynamic_weights.items():
            if isinstance(weights, list) and len(weights) > 0:
                for i, weight in enumerate(weights):
                    self.writter.add_scalar(
                        f"scene_dynamic_weights/scene_{scene_id}_weight_{i}",
                        weight,
                        self.total_num_steps
                    )

        # 记录场景分布的熵（多样性指标）
        if scene_distribution:
            proportions = list(scene_distribution.values())
            proportions = [p for p in proportions if p > 0]  # 过滤掉0值
            if proportions:
                entropy = -sum(p * np.log(p) for p in proportions)
                self.writter.add_scalar(
                    "scene_distribution/entropy",
                    entropy,
                    self.total_num_steps
                )

    def _print_scene_distribution_stats(self, scene_stats):
        """Print scene distribution statistics."""
        print("\n" + "=" * 60)
        print("SCENE DISTRIBUTION STATISTICS")
        print("=" * 60)

        scene_distribution = scene_stats.get('scene_distribution', {})
        scene_episode_counts = scene_stats.get('scene_episode_counts', {})
        scene_step_counts = scene_stats.get('scene_step_counts', {})
        scene_avg_rewards = scene_stats.get('scene_avg_rewards', {})

        print(f"Total Steps: {scene_stats.get('total_steps', 0)}")
        print(f"Total Episodes: {sum(scene_episode_counts.values())}")
        print()

        print("Scene Distribution (by steps):")
        for scene_id in sorted(scene_distribution.keys()):
            proportion = scene_distribution[scene_id]
            step_count = scene_step_counts.get(scene_id, 0)
            episode_count = scene_episode_counts.get(scene_id, 0)
            print(f"  Scene {scene_id}: {proportion:.3f} ({step_count} steps, {episode_count} episodes)")

        print()
        print("Scene Average Rewards:")
        for scene_id in sorted(scene_avg_rewards.keys()):
            reward_info = scene_avg_rewards[scene_id]
            if reward_info['count'] > 0:
                print(
                    f"  Scene {scene_id}: μ={reward_info['mean']:.3f}, σ={reward_info['std']:.3f} (n={reward_info['count']})")
            else:
                print(f"  Scene {scene_id}: No episodes completed")

        # 打印动态权重信息
        scene_dynamic_weights = scene_stats.get('scene_dynamic_weights', {})
        if scene_dynamic_weights:
            print()
            print("Scene Dynamic Weights:")
            for scene_id, weights in scene_dynamic_weights.items():
                if isinstance(weights, list) and len(weights) > 0:
                    weights_str = ", ".join([f"{w:.3f}" for w in weights])
                    print(f"  Scene {scene_id}: [{weights_str}]")

        print("=" * 60)

    def _calculate_scene_stats_fallback(self):
        """Fallback method to calculate scene stats if runner reference not available."""
        total_steps = sum(self.scene_step_counts.values()) if self.scene_step_counts else 1

        scene_distribution = {}
        for scene_id in range(5):  # 假设有5个场景
            scene_distribution[scene_id] = self.scene_step_counts.get(scene_id, 0) / total_steps

        return {
            'scene_distribution': scene_distribution,
            'scene_episode_counts': dict(self.scene_episode_counts),
            'scene_step_counts': dict(self.scene_step_counts),
            'total_steps': total_steps,
            'scene_avg_rewards': {},
            'scene_dynamic_weights': {}
        }

    def set_runner_reference(self, runner):
        """Set reference to the runner for accessing scene statistics."""
        self._runner_ref = runner