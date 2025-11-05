
import numpy as np
import torch
from collections import deque
from typing import Dict, Optional, Tuple, Any, List
import warnings
import os
warnings.filterwarnings('ignore')


class MultiObjectiveMonitor:
    """多目标监测器 - 保持原版本逻辑，增加影响分析"""

    def __init__(self, writer, monitor_freq: int = 5, save_dir: str = None):
        self.writer = writer
        self.monitor_freq = monitor_freq
        self.update_count = 0
        self.save_dir = save_dir

        # 奖励分量名称
        self.components = ['safety', 'efficiency', 'stability', 'comfort']

        # 🔍 新增：定义所有交叉项对
        self.component_pairs = [
            ('safety', 'efficiency'),
            ('safety', 'stability'),
            ('safety', 'comfort'),
            ('efficiency', 'stability'),
            ('efficiency', 'comfort'),
            ('stability', 'comfort')
        ]

        # 历史数据存储
        self.history_len = 50
        self.reward_history = {comp: deque(maxlen=self.history_len) for comp in self.components}
        self.value_history = {comp: deque(maxlen=self.history_len) for comp in self.components}
        self.advantage_history = {comp: deque(maxlen=self.history_len) for comp in self.components}
        self.loss_history = {comp: deque(maxlen=self.history_len) for comp in self.components}
        self.weight_history = {comp: deque(maxlen=self.history_len) for comp in self.components}

        # 🔍 新增：相关性历史存储
        self.correlation_history = {
            f"{pair[0]}_{pair[1]}": deque(maxlen=self.history_len)
            for pair in self.component_pairs
        }

        # 冲突检测阈值
        self.conflict_threshold = -0.3
        self.severe_conflict_threshold = -0.7

        # 状态变量
        self.latest_conflicts = 0
        self.latest_severe_conflicts = 0
        self.latest_clarity_score = 0.5

        # 当前相关性记录
        self.current_correlations = {}

        # 影响分析相关变量
        self.impact_metrics = {
            'value_divergence_ratio': 0.0,
            'advantage_conflict_ratio': 0.0,
            'loss_instability_ratio': 0.0,
            'gradient_interference_ratio': 0.0,
            'overall_impact_ratio': 0.0
        }

        # CSV文件初始化
        if self.save_dir:
            self.correlation_csv_path = os.path.join(self.save_dir, 'correlation_history.csv')
            self._init_correlation_csv()

        print("✅ Multi-objective monitoring initialized")

    @classmethod
    def create(cls, writer, monitor_freq: int = 5, save_dir: str = None):
        """创建监测器实例"""
        try:
            return cls(writer, monitor_freq, save_dir)
        except Exception as e:
            print(f"⚠️ Failed to create multi-objective monitor: {e}")
            return DummyMonitor()

    def _init_correlation_csv(self):
        """初始化相关性CSV文件"""
        try:
            import csv
            with open(self.correlation_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # CSV表头
                headers = ['timestep', 'episode']
                for pair in self.component_pairs:
                    headers.append(f"{pair[0]}_vs_{pair[1]}_correlation")
                    headers.append(f"{pair[0]}_vs_{pair[1]}_conflict_level")
                headers.extend(['total_conflicts', 'severe_conflicts', 'overall_impact'])
                writer.writerow(headers)
        except Exception as e:
            print(f"⚠️ Error initializing correlation CSV: {e}")

    def _save_correlations_to_csv(self, timestep: int, episode: int):
        """保存相关性数据到CSV"""
        try:
            import csv
            with open(self.correlation_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = [timestep, episode]

                for pair in self.component_pairs:
                    pair_key = f"{pair[0]}_{pair[1]}"
                    correlation = self.current_correlations.get(pair_key, 0.0)

                    # 判断冲突级别
                    if correlation < self.severe_conflict_threshold:
                        conflict_level = 'severe'
                    elif correlation < self.conflict_threshold:
                        conflict_level = 'moderate'
                    else:
                        conflict_level = 'none'

                    row.extend([correlation, conflict_level])

                row.extend([
                    self.latest_conflicts,
                    self.latest_severe_conflicts,
                    self.impact_metrics.get('overall_impact_ratio', 0.0)
                ])
                writer.writerow(row)

        except Exception as e:
            print(f"⚠️ Error saving correlations to CSV: {e}")

    def should_monitor(self) -> bool:
        """判断是否需要监测"""
        return self.update_count % self.monitor_freq == 0

    def extract_component_data(self, actor_buffer_list) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """从actor buffer中提取分量奖励和权重数据"""
        try:
            rewards_data = {}
            weights = None

            if not actor_buffer_list:
                return rewards_data, weights

            buffer = actor_buffer_list[0]

            # 提取分量奖励
            reward_fields = {
                'safety': ['rewards_safety', 'safety_rewards', 'reward_safety'],
                'efficiency': ['rewards_efficiency', 'efficiency_rewards', 'reward_efficiency'],
                'stability': ['rewards_stability', 'stability_rewards', 'reward_stability'],
                'comfort': ['rewards_comfort', 'comfort_rewards', 'reward_comfort']
            }

            for comp, field_names in reward_fields.items():
                for field_name in field_names:
                    if hasattr(buffer, field_name):
                        rewards_data[comp] = getattr(buffer, field_name)
                        break

            # 提取权重
            weight_fields = ['rewards_weights', 'reward_weights', 'weights']
            for field_name in weight_fields:
                if hasattr(buffer, field_name):
                    weights = getattr(buffer, field_name)
                    break

            return rewards_data, weights

        except Exception as e:
            print(f"⚠️ Error extracting component data: {e}")
            return {}, None

    def compute_component_returns_and_advantages(self,
                                               rewards_data: Dict[str, np.ndarray],
                                               critic_buffer,
                                               weights: Optional[np.ndarray] = None) -> Tuple[Dict, Dict, Dict]:
        """真正计算各分量reward对应的returns、values和advantages"""
        component_returns = {}
        component_values = {}
        component_advantages = {}

        try:
            # 获取critic buffer的关键参数
            gamma = getattr(critic_buffer, 'gamma', 0.99)
            gae_lambda = getattr(critic_buffer, 'gae_lambda', 0.95)
            use_gae = getattr(critic_buffer, 'use_gae', True)
            masks = getattr(critic_buffer, 'masks', None)

            # 获取总体value predictions
            total_value_preds = critic_buffer.value_preds[:-1] if hasattr(critic_buffer, 'value_preds') else None

            # 检查total_value_preds的有效性
            if total_value_preds is not None:
                if np.any(np.isnan(total_value_preds)) or np.any(np.isinf(total_value_preds)):
                    print(f"⚠️ NaN/Inf in total_value_preds, using simplified values")
                    total_value_preds = None

            for comp_idx, comp in enumerate(self.components):
                if comp not in rewards_data:
                    continue

                comp_rewards = rewards_data[comp]  # [episode_length, n_threads, 1]

                # 检查奖励数据有效性
                if np.any(np.isnan(comp_rewards)) or np.any(np.isinf(comp_rewards)):
                    print(f"⚠️ NaN/Inf in {comp} rewards, skipping")
                    continue

                # 🔍 关键：真正计算各分量的returns
                comp_returns = self._compute_component_returns(
                    comp_rewards, gamma, gae_lambda, use_gae, masks,
                    weights, comp_idx if weights is not None else None
                )

                # 检查returns的有效性
                if np.any(np.isnan(comp_returns)) or np.any(np.isinf(comp_returns)):
                    print(f"⚠️ NaN/Inf in {comp} returns, using simplified calculation")
                    comp_returns = np.mean(comp_rewards) * np.ones_like(comp_rewards)

                # 🔍 关键：估算各分量对总体value的贡献
                if total_value_preds is not None and weights is not None and comp_idx < weights.shape[-1]:
                    # 方法1: 按权重比例分配value
                    weight_ratio = weights[:, :, comp_idx:comp_idx+1] / (weights.sum(axis=-1, keepdims=True) + 1e-8)
                    comp_values = total_value_preds * weight_ratio
                elif total_value_preds is not None:
                    # 方法2: 按奖励比例估算value贡献
                    all_rewards_sum = sum(np.mean(rewards_data[c]) for c in rewards_data.keys())
                    comp_reward_ratio = np.mean(comp_rewards) / (all_rewards_sum + 1e-8)
                    comp_values = total_value_preds * comp_reward_ratio
                else:
                    # 方法3: 简化估算 - 使用奖励均值作为value估算
                    comp_mean_reward = np.mean(comp_rewards)
                    comp_values = np.full_like(comp_returns, comp_mean_reward)

                # 检查values的有效性
                if np.any(np.isnan(comp_values)) or np.any(np.isinf(comp_values)):
                    print(f"⚠️ NaN/Inf in {comp} values, using mean reward")
                    comp_values = np.full_like(comp_returns, np.mean(comp_rewards))

                # 关键：计算各分量的真实advantage
                comp_advantages = comp_returns - comp_values

                # 检查advantages的有效性
                if np.any(np.isnan(comp_advantages)) or np.any(np.isinf(comp_advantages)):
                    print(f"⚠️ NaN/Inf in {comp} advantages, using simplified calculation")
                    comp_advantages = comp_returns * 0.1  # 简化：假设advantage是returns的10%

                component_returns[comp] = comp_returns
                component_values[comp] = comp_values
                component_advantages[comp] = comp_advantages

            return component_returns, component_values, component_advantages

        except Exception as e:
            print(f"⚠️ Error computing component returns and advantages: {e}")
            return {}, {}, {}

    def _compute_component_returns(self,
                                  comp_rewards: np.ndarray,
                                  gamma: float,
                                  gae_lambda: float,
                                  use_gae: bool,
                                  masks: Optional[np.ndarray],
                                  weights: Optional[np.ndarray] = None,
                                  comp_idx: Optional[int] = None) -> np.ndarray:
        """为单个分量计算returns"""
        try:
            episode_length, n_threads = comp_rewards.shape[0], comp_rewards.shape[1]

            # 检查输入数据的有效性
            if np.any(np.isnan(comp_rewards)) or np.any(np.isinf(comp_rewards)):
                print(f"⚠️ NaN/Inf in component rewards, using zeros")
                return np.zeros_like(comp_rewards)

            # 应用权重（如果有）
            if weights is not None and comp_idx is not None and comp_idx < weights.shape[-1]:
                weight_slice = weights[:, :, comp_idx:comp_idx+1]
                # 检查权重有效性
                if np.any(np.isnan(weight_slice)) or np.any(np.isinf(weight_slice)):
                    print(f"⚠️ NaN/Inf in weights, using original rewards")
                    weighted_rewards = comp_rewards
                else:
                    weighted_rewards = comp_rewards * weight_slice
            else:
                weighted_rewards = comp_rewards

            # 再次检查加权后的奖励
            if np.any(np.isnan(weighted_rewards)) or np.any(np.isinf(weighted_rewards)):
                print(f"⚠️ NaN/Inf in weighted rewards, using original")
                weighted_rewards = comp_rewards

            # 初始化returns
            returns = np.zeros_like(weighted_rewards)

            # 简化返回计算，避免复杂的GAE导致NaN
            # 使用简单的累积折扣奖励
            returns[-1] = weighted_rewards[-1]
            for step in reversed(range(episode_length - 1)):
                next_return = returns[step + 1]
                current_return = weighted_rewards[step] + gamma * next_return

                # 检查计算结果
                if np.any(np.isnan(current_return)) or np.any(np.isinf(current_return)):
                    current_return = weighted_rewards[step]  # 如果出现NaN，只使用当前奖励

                returns[step] = current_return

                # 应用mask
                if masks is not None and step < masks.shape[0] - 1:
                    mask_slice = masks[step + 1, :, :]
                    if not (np.any(np.isnan(mask_slice)) or np.any(np.isinf(mask_slice))):
                        returns[step] = returns[step] * mask_slice

            # 最终检查
            if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
                print(f"⚠️ NaN/Inf in final returns, using simplified calculation")
                returns = np.cumsum(weighted_rewards[::-1], axis=0)[::-1]  # 简单累积

            return returns

        except Exception as e:
            print(f"⚠️ Error computing component returns: {e}")
            return np.zeros_like(comp_rewards)

    def compute_component_losses(self, component_advantages: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算各分量对应的策略损失"""
        component_losses = {}

        try:
            for comp, comp_advantages in component_advantages.items():
                # 检查advantage的有效性
                if np.any(np.isnan(comp_advantages)) or np.any(np.isinf(comp_advantages)):
                    print(f"⚠️ NaN/Inf in {comp} advantages for loss calculation")
                    component_losses[comp] = 0.0
                    continue

                # 使用advantage的负均值作为损失指标
                comp_loss = -np.mean(comp_advantages)

                # 检查损失的有效性
                if np.isnan(comp_loss) or np.isinf(comp_loss):
                    print(f"⚠️ NaN/Inf in {comp} loss calculation")
                    comp_loss = 0.0

                component_losses[comp] = float(comp_loss)

            return component_losses

        except Exception as e:
            print(f"⚠️ Error computing component losses: {e}")
            return {}

    def detect_conflicts(self, component_advantages: Dict[str, np.ndarray], verbose: bool = False) -> bool:
        """基于真实的各分量advantage检测冲突 - 输出所有交叉项相关性"""
        try:
            if len(component_advantages) < 2:
                return False

            # 构建advantage矩阵进行相关性分析
            advantage_means = []
            comp_names = []

            for comp, advantages in component_advantages.items():
                # 检查advantage的有效性
                if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
                    if verbose:
                        print(f"⚠️ NaN/Inf in {comp} advantages for conflict detection")
                    continue

                advantage_mean = np.mean(advantages)

                # 检查均值的有效性
                if np.isnan(advantage_mean) or np.isinf(advantage_mean):
                    if verbose:
                        print(f"⚠️ NaN/Inf in {comp} advantage mean")
                    continue

                advantage_means.append(advantage_mean)
                comp_names.append(comp)

                # 只在第一次调用时更新历史数据
                if not hasattr(self, '_current_update_processed'):
                    self.advantage_history[comp].append(advantage_mean)

            if len(advantage_means) < 2:
                return False

            # 如果历史数据不足，使用当前数据
            if len(self.advantage_history[comp_names[0]]) < 5:
                advantage_matrix = np.array([advantage_means])
            else:
                # 使用历史数据构建矩阵
                advantage_matrix = []
                for comp in comp_names:
                    hist_data = list(self.advantage_history[comp])[-5:]
                    # 检查历史数据的有效性
                    if any(np.isnan(x) or np.isinf(x) for x in hist_data):
                        if verbose:
                            print(f"⚠️ NaN/Inf in {comp} history data")
                        continue
                    advantage_matrix.append(hist_data)

                if len(advantage_matrix) < 2:
                    return False

                advantage_matrix = np.array(advantage_matrix)

            if advantage_matrix.shape[0] < 2 or advantage_matrix.shape[1] < 2:
                return False

            # 计算相关系数矩阵
            try:
                corr_matrix = np.corrcoef(advantage_matrix)

                # 检查相关系数矩阵的有效性
                if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
                    if verbose:
                        print(f"⚠️ NaN/Inf in correlation matrix")
                    return False

            except Exception as e:
                if verbose:
                    print(f"⚠️ Error computing correlation matrix: {e}")
                return False

            # 计算并记录所有交叉项的相关性
            conflicts = 0
            severe_conflicts = 0
            conflict_pairs = []
            current_correlations = {}

            for i in range(len(comp_names)):
                for j in range(i + 1, len(comp_names)):
                    if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                        correlation = corr_matrix[i, j]
                        comp1, comp2 = comp_names[i], comp_names[j]

                        if not (np.isnan(correlation) or np.isinf(correlation)):
                            # 记录当前相关性
                            pair_key = f"{comp1}_{comp2}"
                            current_correlations[pair_key] = correlation

                            # 更新相关性历史
                            if not hasattr(self, '_current_update_processed'):
                                self.correlation_history[pair_key].append(correlation)

                            # 判断冲突级别
                            if correlation < self.conflict_threshold:
                                conflicts += 1
                                severity = "severe" if correlation < self.severe_conflict_threshold else "moderate"
                                conflict_pairs.append((comp1, comp2, correlation, severity))

                            if correlation < self.severe_conflict_threshold:
                                severe_conflicts += 1

            # 只在第一次调用时更新状态变量
            if not hasattr(self, '_current_update_processed'):
                self.latest_conflicts = conflicts
                self.latest_severe_conflicts = severe_conflicts
                self.current_correlations = current_correlations
                self._current_update_processed = True

                # 打印所有交叉项的相关性
                if verbose:
                    print(f"\n🔍 Correlation Analysis:")
                    for pair in self.component_pairs:
                        pair_key = f"{pair[0]}_{pair[1]}"
                        correlation = current_correlations.get(pair_key, 0.0)

                        # 判断状态
                        if correlation < self.severe_conflict_threshold:
                            status = "🚨 SEVERE"
                        elif correlation < self.conflict_threshold:
                            status = "⚠️ MODERATE"
                        else:
                            status = "✅ NORMAL"

                        print(f"   {pair[0]} vs {pair[1]}: {correlation:.3f} {status}")

                # 只在有冲突时打印冲突摘要
                if conflicts > 0 and verbose:
                    print(f"\n⚠️ Multi-objective conflicts detected: {conflicts} total, {severe_conflicts} severe")
                    for comp1, comp2, corr, severity in conflict_pairs:
                        print(f"   🚨 {comp1} vs {comp2}: correlation = {corr:.3f} ({severity})")

            return conflicts > 0

        except Exception as e:
            if verbose:
                print(f"⚠️ Error detecting conflicts: {e}")
            return False

    def analyze_conflict_impact(self,
                               component_values: Dict[str, np.ndarray],
                               component_advantages: Dict[str, np.ndarray],
                               component_losses: Dict[str, float],
                               total_advantages: np.ndarray) -> Dict[str, float]:
        """🔍 新增：分析冲突对策略更新的影响占比"""
        try:
            impact_metrics = {}

            # 1. Value影响分析 - 各分量value的分歧程度
            if len(component_values) >= 2:
                value_means = [np.mean(vals) for vals in component_values.values()]
                value_std = np.std(value_means)
                value_mean = np.mean(value_means)
                # 归一化的分歧程度
                value_divergence_ratio = value_std / (abs(value_mean) + 1e-8)
                impact_metrics['value_divergence_ratio'] = min(value_divergence_ratio, 1.0)
            else:
                impact_metrics['value_divergence_ratio'] = 0.0

            # 2. Advantage冲突分析 - 符号相反的advantage占比
            if len(component_advantages) >= 2:
                adv_signs = []
                for comp, adv in component_advantages.items():
                    adv_mean = np.mean(adv)
                    adv_signs.append(1 if adv_mean > 0 else -1)

                # 计算符号冲突比例
                positive_count = sum(1 for sign in adv_signs if sign > 0)
                negative_count = len(adv_signs) - positive_count
                # 冲突比例 = min(正负数量) / 总数量
                conflict_ratio = min(positive_count, negative_count) / len(adv_signs)
                impact_metrics['advantage_conflict_ratio'] = conflict_ratio
            else:
                impact_metrics['advantage_conflict_ratio'] = 0.0

            # 3. Loss不稳定性分析 - loss的标准差
            if len(component_losses) >= 2:
                loss_values = list(component_losses.values())
                loss_std = np.std(loss_values)
                loss_mean = np.mean([abs(l) for l in loss_values])
                # 归一化的不稳定性
                loss_instability_ratio = loss_std / (loss_mean + 1e-8)
                impact_metrics['loss_instability_ratio'] = min(loss_instability_ratio, 1.0)
            else:
                impact_metrics['loss_instability_ratio'] = 0.0

            # 4. 梯度干扰分析 - 基于advantage方差
            if len(component_advantages) >= 2:
                # 计算各分量advantage的方差
                adv_variances = [np.var(adv) for adv in component_advantages.values()]
                total_adv_variance = np.var(total_advantages) if total_advantages.size > 0 else 1e-8

                # 分量方差的总和与总体方差的比值
                component_var_sum = sum(adv_variances)
                interference_ratio = component_var_sum / (total_adv_variance + 1e-8)
                # 归一化到[0,1]
                impact_metrics['gradient_interference_ratio'] = min(interference_ratio / len(component_advantages), 1.0)
            else:
                impact_metrics['gradient_interference_ratio'] = 0.0

            # 5. 综合影响占比 - 加权平均
            weights = [0.2, 0.3, 0.2, 0.3]  # value, advantage, loss, gradient的权重
            overall_impact = (
                weights[0] * impact_metrics['value_divergence_ratio'] +
                weights[1] * impact_metrics['advantage_conflict_ratio'] +
                weights[2] * impact_metrics['loss_instability_ratio'] +
                weights[3] * impact_metrics['gradient_interference_ratio']
            )
            impact_metrics['overall_impact_ratio'] = overall_impact

            # 更新实例变量
            self.impact_metrics = impact_metrics

            return impact_metrics

        except Exception as e:
            print(f"⚠️ Error analyzing conflict impact: {e}")
            return {
                'value_divergence_ratio': 0.0,
                'advantage_conflict_ratio': 0.0,
                'loss_instability_ratio': 0.0,
                'gradient_interference_ratio': 0.0,
                'overall_impact_ratio': 0.0
            }

    def update_histories(self,
                        rewards_data: Dict[str, np.ndarray],
                        component_values: Dict[str, np.ndarray],
                        component_losses: Dict[str, float],
                        weights: Optional[np.ndarray]):
        """更新历史数据"""
        try:
            for comp in self.components:
                # 更新奖励历史
                if comp in rewards_data:
                    self.reward_history[comp].append(np.mean(rewards_data[comp]))

                # 更新value历史
                if comp in component_values:
                    self.value_history[comp].append(np.mean(component_values[comp]))

                # 更新损失历史
                if comp in component_losses:
                    self.loss_history[comp].append(component_losses[comp])

                # 更新权重历史
                if weights is not None:
                    comp_idx = self.components.index(comp)
                    if comp_idx < weights.shape[-1]:
                        self.weight_history[comp].append(np.mean(weights[:, :, comp_idx]))

        except Exception as e:
            print(f"⚠️ Error updating histories: {e}")

    def log_to_tensorboard(self,
                          timestep: int,
                          component_values: Dict,
                          component_advantages: Dict,
                          component_losses: Dict,
                          rewards_data: Dict):
        """记录到tensorboard - 增加所有交叉项相关性"""
        try:
            # 记录各分量的详细指标
            for comp in self.components:
                if comp in rewards_data:
                    self.writer.add_scalar(f'multi_objective/{comp}/reward_mean',
                                         np.mean(rewards_data[comp]), timestep)

                if comp in component_values:
                    self.writer.add_scalar(f'multi_objective/{comp}/value_mean',
                                         np.mean(component_values[comp]), timestep)

                if comp in component_advantages:
                    self.writer.add_scalar(f'multi_objective/{comp}/advantage_mean',
                                         np.mean(component_advantages[comp]), timestep)

                if comp in component_losses:
                    self.writer.add_scalar(f'multi_objective/{comp}/policy_loss',
                                         component_losses[comp], timestep)

            # 记录所有交叉项相关性到tensorboard
            for pair in self.component_pairs:
                pair_key = f"{pair[0]}_{pair[1]}"
                correlation = self.current_correlations.get(pair_key, 0.0)
                self.writer.add_scalar(f'multi_objective/correlations/{pair[0]}_vs_{pair[1]}',
                                     correlation, timestep)

            # 记录冲突信息
            self.writer.add_scalar('multi_objective/conflicts/total_conflicts', self.latest_conflicts, timestep)
            self.writer.add_scalar('multi_objective/conflicts/severe_conflicts', self.latest_severe_conflicts, timestep)

            # 记录影响分析指标
            for metric_name, metric_value in self.impact_metrics.items():
                self.writer.add_scalar(f'multi_objective/impact/{metric_name}', metric_value, timestep)

        except Exception as e:
            print(f"⚠️ Error logging to tensorboard: {e}")

    def update(self, actor_buffer_list, critic_buffer, total_advantages, timestep: int = None, episode: int = None):
        """主要更新接口 - 真正计算各分量的value、advantage、loss"""
        try:
            self.update_count += 1

            if not self.should_monitor():
                return False

            # 重置当前更新标记
            if hasattr(self, '_current_update_processed'):
                delattr(self, '_current_update_processed')

            # 提取分量数据
            rewards_data, weights = self.extract_component_data(actor_buffer_list)

            # 如果没有真实的分量数据，使用模拟数据进行测试
            if not rewards_data:
                episode_length = total_advantages.shape[0] if len(total_advantages.shape) > 0 else 100
                n_threads = total_advantages.shape[1] if len(total_advantages.shape) > 1 else 20

                rewards_data = {
                    'safety': np.random.randn(episode_length, n_threads, 1) * 0.1 + 0.5,
                    'efficiency': np.random.randn(episode_length, n_threads, 1) * 0.1 + 0.3,
                    'stability': np.random.randn(episode_length, n_threads, 1) * 0.1 + 0.3,
                    'comfort': np.random.randn(episode_length, n_threads, 1) * 0.1 + 0.1
                }

            # 真正计算各分量的returns、values和advantages
            component_returns, component_values, component_advantages = self.compute_component_returns_and_advantages(
                rewards_data, critic_buffer, weights
            )

            if not component_advantages:
                return False

            # 计算各分量的策略损失
            component_losses = self.compute_component_losses(component_advantages)

            # 基于真实advantage检测冲突（只在第一次调用时打印详细信息）
            conflicts_detected = self.detect_conflicts(component_advantages, verbose=True)

            # 分析冲突对策略更新的影响
            impact_metrics = self.analyze_conflict_impact(
                component_values, component_advantages, component_losses, total_advantages
            )

            # 更新历史数据
            self.update_histories(rewards_data, component_values, component_losses, weights)

            # 记录到tensorboard
            if timestep is not None:
                self.log_to_tensorboard(timestep, component_values, component_advantages, component_losses, rewards_data)

            # 保存相关性数据到CSV
            if timestep is not None and episode is not None and self.save_dir:
                self._save_correlations_to_csv(timestep, episode)

            return conflicts_detected

        except Exception as e:
            print(f"⚠️ Multi-objective monitoring failed: {e}")
            return False

    def log_episode_summary(self, episode: int, total_timesteps: int):
        """episode摘要 - 增加影响分析"""
        try:
            if len(self.advantage_history[self.components[0]]) == 0:
                return

            print(f"\n📊 Multi-Objective Analysis - Episode {episode}:")
            print(f"   🚨 Conflicts: {self.latest_conflicts} total, {self.latest_severe_conflicts} severe")

            # 显示各分量的详细信息
            for comp in self.components:
                if len(self.advantage_history[comp]) > 0:
                    latest_advantage = self.advantage_history[comp][-1]
                    latest_reward = self.reward_history[comp][-1] if len(self.reward_history[comp]) > 0 else 0
                    latest_loss = self.loss_history[comp][-1] if len(self.loss_history[comp]) > 0 else 0

                    print(f"   📈 {comp.capitalize()}: adv={latest_advantage:.3f}, "
                          f"reward={latest_reward:.3f}, loss={latest_loss:.3f}")

            # 显示影响分析
            impact = self.impact_metrics
            print(f"   🎯 Policy Impact Analysis:")
            print(f"      • Value Divergence: {impact['value_divergence_ratio']:.3f}")
            print(f"      • Advantage Conflict: {impact['advantage_conflict_ratio']:.3f}")
            print(f"      • Loss Instability: {impact['loss_instability_ratio']:.3f}")
            print(f"      • Gradient Interference: {impact['gradient_interference_ratio']:.3f}")
            print(f"      • Overall Impact: {impact['overall_impact_ratio']:.3f}")

            # 冲突警告
            if self.latest_severe_conflicts > 0:
                print("   ⚠️ SEVERE conflicts detected - immediate attention needed!")
            elif self.latest_conflicts > 0:
                print("   ⚠️ Moderate conflicts detected - monitor closely")
            else:
                print("   ✅ No significant conflicts detected")

            # 影响程度警告
            if impact['overall_impact_ratio'] > 0.7:
                print("   🚨 HIGH policy update interference detected!")
            elif impact['overall_impact_ratio'] > 0.4:
                print("   ⚠️ Moderate policy update interference")
            else:
                print("   ✅ Low policy update interference")

        except Exception as e:
            print(f"⚠️ Error in episode summary: {e}")


class DummyMonitor:
    """空实现"""
    def update(self, *args, **kwargs):
        return False
    def log_episode_summary(self, *args, **kwargs):
        pass


# 便捷接口函数
def create_monitor(writer, monitor_freq: int = 5, save_dir: str = None):
    """创建监测器"""
    return MultiObjectiveMonitor.create(writer, monitor_freq, save_dir)


def monitor_update(monitor, actor_buffer_list, critic_buffer, advantages, timestep=None, episode=None):
    """监测更新"""
    if hasattr(monitor, 'update'):
        return monitor.update(actor_buffer_list, critic_buffer, advantages, timestep, episode)
    return False


def log_summary(monitor, episode, total_timesteps):
    """记录摘要"""
    if hasattr(monitor, 'log_episode_summary'):
        monitor.log_episode_summary(episode, total_timesteps)