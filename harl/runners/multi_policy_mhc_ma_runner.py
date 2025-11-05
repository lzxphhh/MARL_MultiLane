"""Runner for multi-scene multi-policy MA algorithms with multi-head critic."""
import numpy as np
import torch
from collections import defaultdict
from harl.runners.multi_policy_mhc_base_runner import MultiPolicyBaseRunner
from harl.utils.trans_tools import _t2n


class MultiPolicyMARunner(MultiPolicyBaseRunner):
    """Runner for multi-scene multi-policy MA algorithms with multi-head critic.

    Implements training loops that handle multiple scene classes simultaneously,
    each with their own actor/critic instances and dynamic weights.
    """

    def __init__(self, args, algo_args, env_args):
        """Initialize MultiPolicyMARunner."""
        # 正确的super()调用方式
        super(MultiPolicyMARunner, self).__init__(args, algo_args, env_args)

        # 设置logger的runner引用
        if hasattr(self, 'logger'):
            self.logger.set_runner_reference(self)

    def train(self):
        """Training procedure for multi-scene MHCMAPPO."""
        # 收集所有场景的训练信息
        all_actor_train_infos = []
        all_critic_train_infos = []

        # 为每个场景独立进行训练
        for scene_id in range(self.num_scene_classes):
            scene_actor_train_infos, scene_critic_train_info = self._train_single_scene(scene_id)

            # 收集训练信息
            all_actor_train_infos.extend(scene_actor_train_infos)
            all_critic_train_infos.append(scene_critic_train_info)

        # 聚合训练信息
        aggregated_actor_train_infos = self._aggregate_actor_train_infos(all_actor_train_infos)
        aggregated_critic_train_info = self._aggregate_critic_train_infos(all_critic_train_infos)

        # 添加场景级别的统计信息
        aggregated_critic_train_info.update({
            'scene_distribution': self._get_scene_distribution(),
            'total_scenes_active': len([info for info in all_critic_train_infos if info is not None])
        })

        return aggregated_actor_train_infos, aggregated_critic_train_info

    def _train_single_scene(self, scene_id):
        """Train actors and critic for a single scene."""
        # 检查是否有数据需要训练
        if not self._scene_has_data(scene_id):
            return [], None

        # 1. 计算多头优势函数
        advantages_heads = self._compute_multihead_advantages(scene_id)

        # 2. 获取动态权重并计算加权优势函数
        dynamic_weights = self.scene_critics[scene_id].get_dynamic_weights()
        weighted_advantages = self._compute_weighted_advantages(advantages_heads, dynamic_weights)

        # 3. 训练actors
        actor_train_infos = self._train_scene_actors(scene_id, weighted_advantages, advantages_heads)

        # 4. 训练critic
        critic_train_info = self._train_scene_critic(scene_id)

        # 5. 更新动态权重
        self.scene_critics[scene_id].episode_done()

        return actor_train_infos, critic_train_info

    def _scene_has_data(self, scene_id):
        """Check if a scene has sufficient data for training."""
        for agent_id in range(self.num_agents):
            buffer = self.scene_actor_buffers[scene_id][agent_id]
            if not np.all(buffer.active_masks[:-1] == 0.0):
                return True
        return False

    def _compute_multihead_advantages(self, scene_id):
        """Compute advantages for each head separately for a specific scene."""
        advantages_heads = []

        # 对于每个critic头，计算其对应的优势函数
        for head_idx in range(self.critic_head_num):
            # 从critic buffer获取值函数预测和返回值
            head_values = self.scene_critic_buffers[scene_id].value_preds[:-1, :, head_idx]
            head_returns = self.scene_critic_buffers[scene_id].returns[:-1, :, head_idx]

            # 添加维度以符合后续处理需求
            head_values = head_values[:, :, np.newaxis]
            head_returns = head_returns[:, :, np.newaxis]

            # 计算优势函数
            value_normalizer = self.value_normalizers[scene_id] if hasattr(self, 'value_normalizers') else None
            if value_normalizer is not None:
                head_advantages = head_returns - value_normalizer.denormalize(head_values)
            else:
                head_advantages = head_returns - head_values

            advantages_heads.append(head_advantages)

        return advantages_heads

    def _compute_weighted_advantages(self, advantages_heads, dynamic_weights):
        """Compute dynamically weighted advantages."""
        # 计算加权优势函数
        weighted_advantages = np.zeros_like(advantages_heads[0])
        for i, adv in enumerate(advantages_heads):
            weight = dynamic_weights[i] if i < len(dynamic_weights) else 0.0
            weighted_advantages += weight * adv

        # 标准化处理
        if self.state_type == "EP":
            flat_advantages = weighted_advantages.reshape(-1)
            valid_indices = ~(np.isnan(flat_advantages) | np.isinf(flat_advantages))
            valid_advantages = flat_advantages[valid_indices]

            if len(valid_advantages) > 0:
                mean_advantages = np.mean(valid_advantages)
                std_advantages = np.std(valid_advantages)
                weighted_advantages = (weighted_advantages - mean_advantages) / (std_advantages + 1e-5)

        # 确保形状正确
        weighted_advantages = weighted_advantages.reshape(weighted_advantages.shape[0], weighted_advantages.shape[1], 1)

        return weighted_advantages

    def _train_scene_actors(self, scene_id, weighted_advantages, advantages_heads):
        """Train all actors for a specific scene."""
        actor_train_infos = []

        # 设置dynamic weights到actors
        dynamic_weights = self.scene_critics[scene_id].get_dynamic_weights()

        if self.share_param:
            # 参数共享模式
            self.scene_actors[scene_id][0].set_dynamic_weights(dynamic_weights)

            # 使用多头优势函数训练
            actor_train_info = self.scene_actors[scene_id][0].share_param_train_with_multihead_advantages(
                self.scene_actor_buffers[scene_id],
                advantages_heads,
                self.num_agents,
                self.state_type
            )

            # 为所有agent添加相同的训练信息
            for _ in range(self.num_agents):
                actor_train_infos.append(actor_train_info.copy())
        else:
            # 非参数共享模式
            for agent_id in range(self.num_agents):
                # 设置动态权重
                self.scene_actors[scene_id][agent_id].set_dynamic_weights(dynamic_weights)

                # 训练单个actor
                if self.state_type == "EP":
                    actor_train_info = self.scene_actors[scene_id][agent_id].train_with_multihead_advantages(
                        self.scene_actor_buffers[scene_id][agent_id],
                        advantages_heads,
                        self.state_type
                    )
                elif self.state_type == "FP":
                    # FP模式下需要特殊处理advantages
                    agent_advantages_heads = []
                    for head_adv in advantages_heads:
                        agent_advantages_heads.append(head_adv[:, :, agent_id:agent_id+1])

                    actor_train_info = self.scene_actors[scene_id][agent_id].train_with_multihead_advantages(
                        self.scene_actor_buffers[scene_id][agent_id],
                        agent_advantages_heads,
                        self.state_type
                    )

                actor_train_infos.append(actor_train_info)

        return actor_train_infos

    def _train_scene_critic(self, scene_id):
        """Train critic for a specific scene."""
        value_normalizer = self.value_normalizers[scene_id] if hasattr(self, 'value_normalizers') else None
        critic_train_info = self.scene_critics[scene_id].train(
            self.scene_critic_buffers[scene_id],
            value_normalizer
        )

        # 添加场景特定信息
        critic_train_info['scene_id'] = scene_id

        return critic_train_info

    def _aggregate_actor_train_infos(self, all_actor_train_infos):
        """Aggregate actor training information across all scenes."""
        if not all_actor_train_infos:
            return []

        # 按agent位置聚合
        aggregated_info = []

        # 按agent位置聚合
        for agent_idx in range(self.num_agents):
            agent_infos = []

            # 收集这个agent在所有场景中的训练信息
            for scene_id in range(self.num_scene_classes):
                start_idx = scene_id * self.num_agents
                if start_idx + agent_idx < len(all_actor_train_infos):
                    info = all_actor_train_infos[start_idx + agent_idx]
                    if info is not None:
                        agent_infos.append(info)

            # 聚合这个agent的信息
            if agent_infos:
                aggregated = self._average_dict_values(agent_infos)
                aggregated['num_scenes_trained'] = len(agent_infos)
                aggregated_info.append(aggregated)

        return aggregated_info

    def _aggregate_critic_train_infos(self, all_critic_train_infos):
        """Aggregate critic training information across all scenes."""
        valid_infos = [info for info in all_critic_train_infos if info is not None]

        if not valid_infos:
            return {}

        # 聚合基础指标
        aggregated_info = self._average_dict_values(valid_infos)
        aggregated_info['num_scenes_trained'] = len(valid_infos)

        # 处理多头损失信息
        if 'value_loss' in aggregated_info and isinstance(aggregated_info['value_loss'], list):
            # 计算每个头的平均损失
            for head_idx in range(self.critic_head_num):
                head_name = self.critic_head_names.get(head_idx, f"head_{head_idx}")
                aggregated_info[f'value_loss_{head_name}'] = aggregated_info['value_loss'][head_idx]

        return aggregated_info

    def _average_dict_values(self, dict_list):
        """Average values in a list of dictionaries."""
        if not dict_list:
            return {}

        averaged_dict = {}

        for key in dict_list[0].keys():
            values = []
            for d in dict_list:
                if key in d and d[key] is not None:
                    if isinstance(d[key], (int, float)):
                        values.append(d[key])
                    elif isinstance(d[key], list) and len(d[key]) > 0 and isinstance(d[key][0], (int, float)):
                        values.append(d[key])

            if values:
                if isinstance(values[0], list):
                    # 处理列表类型的值（如多头损失）
                    averaged_dict[key] = np.mean(values, axis=0).tolist()
                else:
                    averaged_dict[key] = np.mean(values)

        return averaged_dict

    """
    修复多策略评估环境reset方法的兼容性问题
    """

    @torch.no_grad()
    def eval(self):
        """Evaluate all scene models."""
        self.logger.eval_init()
        eval_episode = 0

        # 兼容性处理：支持不同数量的返回值
        try:
            reset_result = self.eval_envs.reset()

            if len(reset_result) == 4:
                # 完整版本：包含场景分类
                eval_obs, eval_share_obs, eval_available_actions, eval_classified_scene_mark = reset_result
            elif len(reset_result) == 3:
                # 简化版本：不包含场景分类
                eval_obs, eval_share_obs, eval_available_actions = reset_result
                # 为评估分配默认场景或随机场景
                eval_classified_scene_mark = np.random.randint(
                    0, self.num_scene_classes,
                    self.algo_args["eval"]["n_eval_rollout_threads"]
                )
                print(f"Warning: eval_envs.reset() returned only 3 values, using random scene classification")
            else:
                raise ValueError(f"eval_envs.reset() returned {len(reset_result)} values, expected 3 or 4")

        except Exception as e:
            print(f"Error in eval_envs.reset(): {e}")
            # 创建默认值
            n_eval_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
            eval_obs = np.zeros((n_eval_threads, self.num_agents, *self.envs.observation_space[0].shape))
            eval_share_obs = np.zeros((n_eval_threads, self.num_agents, *self.envs.share_observation_space[0].shape))
            eval_available_actions = None
            eval_classified_scene_mark = np.zeros(n_eval_threads, dtype=int)

        # 验证场景分类的范围
        eval_classified_scene_mark = self._classify_scene(eval_classified_scene_mark)

        # 初始化eval状态
        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_reward_weights = np.zeros(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, self.num_reward_head),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            # 确保使用有效的场景分类
            if not hasattr(self, '_last_eval_scene_marks'):
                self._last_eval_scene_marks = eval_classified_scene_mark

            eval_scene_marks = self._last_eval_scene_marks
            eval_scene_marks = self._classify_scene(eval_scene_marks)

            eval_actions_collector = []

            # 为每个agent收集动作
            for agent_id in range(self.num_agents):
                agent_actions = np.zeros((self.algo_args["eval"]["n_eval_rollout_threads"], 1))
                agent_rnn_states = np.zeros(
                    (self.algo_args["eval"]["n_eval_rollout_threads"], self.recurrent_n, self.rnn_hidden_size))

                # 根据场景分类选择对应的actor
                for env_idx in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                    scene_id = eval_scene_marks[env_idx]

                    if (eval_available_actions is not None and not all(x is None for x in eval_available_actions)):
                        eval_available_action = eval_available_actions[env_idx:env_idx + 1, agent_id]
                    else:
                        eval_available_action = None

                    # 使用场景特定的actor
                    eval_action, temp_rnn_state = self.scene_actors[scene_id][agent_id].act(
                        eval_obs[env_idx:env_idx + 1, agent_id],
                        eval_rnn_states[env_idx:env_idx + 1, agent_id],
                        eval_reward_weights[env_idx:env_idx + 1, agent_id],
                        eval_masks[env_idx:env_idx + 1, agent_id],
                        eval_available_action,
                        deterministic=True,
                    )

                    agent_actions[env_idx] = _t2n(eval_action)[0]
                    agent_rnn_states[env_idx] = _t2n(temp_rnn_state)[0]

                eval_rnn_states[:, agent_id] = agent_rnn_states
                eval_actions_collector.append(agent_actions)

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # 环境步进 - 兼容性处理
            try:
                step_result = self.eval_envs.step(eval_actions)

                if len(step_result) >= 12:  # 完整版本
                    (
                        eval_obs,
                        eval_share_obs,
                        eval_rewards,
                        eval_speed,
                        eval_acceleration,
                        eval_rewards_safety,
                        eval_rewards_stability,
                        eval_rewards_efficiency,
                        eval_rewards_comfort,
                        eval_safety_SI,
                        eval_efficiency_ASR,
                        eval_efficiency_TFR,
                        eval_stability_SSI,
                        eval_stability_VSS,
                        eval_comfort_JIs,
                        eval_control_efficiencys,
                        eval_control_reward,
                        eval_comfort_cost,
                        eval_comfort_reward,
                        eval_dones,
                        eval_infos,
                        eval_available_actions,
                    ) = step_result[:22]  # 取前22个元素
                else:  # 简化版本
                    # 处理基本的环境返回值
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = step_result[
                                                                                                             :6]
                    # 创建缺失的指标
                    eval_speed = np.zeros_like(eval_rewards)
                    eval_acceleration = np.zeros_like(eval_rewards)

            except Exception as e:
                print(f"Error in eval_envs.step(): {e}")
                break

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_speed,
                eval_acceleration,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(eval_data)

            # 更新RNN states和masks
            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            # 统计episode完成
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

    @torch.no_grad()
    def render(self):
        """Render with multi-scene support."""
        print("start rendering with multi-scene support")

        if self.manual_expand_dims:
            # 处理需要手动扩展维度的环境
            for episode_idx in range(self.algo_args["render"]["render_episodes"]):
                print(f"Rendering episode {episode_idx + 1}/{self.algo_args['render']['render_episodes']}")

                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )

                eval_rnn_states = np.zeros(
                    (self.env_num, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                    dtype=np.float32,
                )
                eval_reward_weights = np.zeros(
                    (self.env_num, self.num_agents, self.num_reward_head),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )

                # 场景奖励累积
                scene_rewards = defaultdict(float)
                scene_components = defaultdict(lambda: defaultdict(float))

                rewards = 0
                mean_v = 0
                mean_acc = 0
                current_scene = 0  # 默认场景

                while True:
                    eval_actions_collector = []

                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.scene_actors[current_scene][agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_reward_weights[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))

                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_mean_v,
                        eval_mean_acc,
                        eval_rewards_safety,
                        eval_rewards_stability,
                        eval_rewards_efficiency,
                        eval_rewards_comfort,
                        eval_safety_SI,
                        eval_efficiency_ASR,
                        eval_efficiency_TFR,
                        eval_stability_SSI,
                        eval_stability_VSS,
                        eval_comfort_JIs,
                        eval_control_efficiencys,
                        eval_control_reward,
                        eval_comfort_cost,
                        eval_comfort_reward,
                        eval_dones,
                        infos,
                        eval_available_actions,
                        eval_classified_scene_mark,
                    ) = self.envs.step(eval_actions[0])

                    # 更新当前场景
                    if hasattr(eval_classified_scene_mark, '__len__'):
                        current_scene = eval_classified_scene_mark[0]
                    else:
                        current_scene = eval_classified_scene_mark
                    current_scene = max(0, min(current_scene, self.num_scene_classes - 1))

                    rewards += eval_rewards[0][0]
                    mean_v += eval_mean_v
                    mean_acc += eval_mean_acc

                    # 累积场景特定奖励
                    scene_rewards[current_scene] += eval_rewards[0][0]
                    scene_components[current_scene]['safety'] += np.sum(eval_rewards_safety)
                    scene_components[current_scene]['efficiency'] += np.sum(eval_rewards_efficiency)
                    scene_components[current_scene]['stability'] += np.sum(eval_rewards_stability)
                    scene_components[current_scene]['comfort'] += np.sum(eval_rewards_comfort)

                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )

                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        import time
                        time.sleep(0.1)

                    if np.all(eval_dones):
                        print(f"Episode {episode_idx + 1} completed:")
                        print(f"  Total reward: {rewards}")
                        print(f"  Episode steps: {infos[0]['step_time']}")
                        print(f"  Mean speed: {mean_v / (infos[0]['step_time'] * 10)}")
                        print(f"  Mean acceleration: {mean_acc / (infos[0]['step_time'] * 10)}")
                        print(f"  Collision: {infos[0]['collision']}")
                        print(f"  Done reason: {infos[0]['done_reason']}")
                        print(f"  Final scene: {current_scene}")

                        # 打印场景特定奖励
                        print("  Scene-specific rewards:")
                        for scene_id, scene_reward in scene_rewards.items():
                            print(f"    Scene {scene_id}: {scene_reward}")
                            components = scene_components[scene_id]
                            print(f"      Safety: {components['safety']}")
                            print(f"      Efficiency: {components['efficiency']}")
                            print(f"      Stability: {components['stability']}")
                            print(f"      Comfort: {components['comfort']}")

                        # 打印动态权重信息
                        print("  Current dynamic weights by scene:")
                        for scene_id in range(self.num_scene_classes):
                            weights = self.scene_critics[scene_id].get_dynamic_weights()
                            print(f"    Scene {scene_id}: {weights}")

                        print('--------------------------------------')
                        break
        else:
            # 处理不需要手动扩展维度的环境
            for episode_idx in range(self.algo_args["render"]["render_episodes"]):
                print(f"Rendering episode {episode_idx + 1}/{self.algo_args['render']['render_episodes']}")

                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (self.env_num, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )

                rewards = 0
                current_scene = 0

                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.scene_actors[current_scene][agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))

                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions) = self.envs.step(eval_actions)

                    rewards += eval_rewards[0][0][0]

                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        import time
                        time.sleep(0.1)

                    if eval_dones[0][0]:
                        print(f"Episode {episode_idx + 1} total reward: {rewards}")
                        break

        if "smac" in self.args["env"]:
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()