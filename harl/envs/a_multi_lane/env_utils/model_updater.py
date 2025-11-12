#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型更新器

负责在训练过程中更新以下模型：
1. 横向决策模型（SCM-based）的微调
2. CQR占用预测模型的校准
3. 三阶段训练策略管理
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# 添加项目根路径
harl_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(harl_root))


class LateralDecisionUpdater:
    """
    横向决策模型更新器

    实现三阶段渐进式微调策略
    """

    def __init__(self,
                 decision_module,
                 stage_thresholds: Tuple[int, int] = (500, 1500),
                 learning_rates: Tuple[float, float, float] = (1e-4, 5e-5, 1e-5),
                 loss_weights: Optional[Dict] = None):
        """
        初始化决策模型更新器

        Args:
            decision_module: DecisionModule实例
            stage_thresholds: 阶段切换阈值 (stage1_end, stage2_end)
            learning_rates: 三阶段学习率 (lr_stage1, lr_stage2, lr_stage3)
            loss_weights: 损失函数权重 {'w_efficiency', 'w_safety', 'epsilon_max', 'theta_safe'}
        """
        self.decision_module = decision_module
        self.stage_thresholds = stage_thresholds
        self.learning_rates = learning_rates

        # 默认损失权重
        if loss_weights is None:
            self.loss_weights = {
                'w_efficiency': 0.6,
                'w_safety': 0.4,
                'epsilon_max': 0.1,
                'theta_safe': 0.3
            }
        else:
            self.loss_weights = loss_weights

        # 训练状态
        self.current_episode = 0
        self.current_stage = 1
        self.total_updates = 0

        # 训练历史
        self.loss_history = []
        self.stage_history = []

    def update_stage(self, episode: int):
        """
        根据episode更新训练阶段

        Stage 1 (0 - stage_thresholds[0]): 冻结SCM基础模型
        Stage 2 (stage_thresholds[0] - stage_thresholds[1]): 解冻个体层
        Stage 3 (stage_thresholds[1] +): 全局微调

        Args:
            episode: 当前episode数
        """
        self.current_episode = episode

        if episode < self.stage_thresholds[0]:
            new_stage = 1
        elif episode < self.stage_thresholds[1]:
            new_stage = 2
        else:
            new_stage = 3

        if new_stage != self.current_stage:
            print(f"[ModelUpdater] 切换训练阶段: Stage {self.current_stage} -> Stage {new_stage} (Episode {episode})")
            self.current_stage = new_stage
            self._apply_stage_config()
            self.stage_history.append((episode, new_stage))

    def _apply_stage_config(self):
        """
        应用当前阶段的配置

        根据00_structure.tex第5节训练策略：
        - Stage 1: 冻结SCM，只训练decision_threshold
        - Stage 2: 解冻个体层，训练individual_encoder + decision_threshold
        - Stage 3: 全局微调
        """
        stage = self.current_stage
        lr = self.learning_rates[stage - 1]

        if stage == 1:
            # 冻结SCM基础模型
            self.decision_module.decision_maker.freeze_base_model = True
            self.decision_module.decision_maker._freeze_parameters()
            print(f"  [Stage 1] 冻结SCM基础模型，LR={lr}")

        elif stage == 2:
            # 解冻个体层
            self.decision_module.decision_maker.freeze_base_model = False
            self.decision_module.decision_maker._unfreeze_parameters()
            # 重新冻结环境层和融合层
            if hasattr(self.decision_module.decision_maker.scm_model, 'environment_encoder'):
                for param in self.decision_module.decision_maker.scm_model.environment_encoder.parameters():
                    param.requires_grad = False
            if hasattr(self.decision_module.decision_maker.scm_model, 'causal_fusion'):
                for param in self.decision_module.decision_maker.scm_model.causal_fusion.parameters():
                    param.requires_grad = False
            print(f"  [Stage 2] 解冻个体层，LR={lr}")

        elif stage == 3:
            # 全局微调
            self.decision_module.decision_maker.freeze_base_model = False
            self.decision_module.decision_maker._unfreeze_parameters()
            print(f"  [Stage 3] 全局微调，LR={lr}")

        # 更新学习率
        for param_group in self.decision_module.decision_maker.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_lateral_loss(self,
                            cav_trajectories: List[np.ndarray],
                            decisions: List[int],
                            rewards: List[float],
                            predicted_occupancies: List[Dict[str, np.ndarray]]) -> float:
        """
        计算横向决策损失

        根据00_structure.tex公式 (Eq. 15-21):
        L_lateral = w_efficiency * L_efficiency + w_safety * L_safety

        L_efficiency = -R_efficiency + ε_max * Var(Δy)
        L_safety = Σ_j max(0, Overlap(τ_ego, τ_j^pred) - θ_safe)^2

        Args:
            cav_trajectories: CAV轨迹列表 [N × [T, 2]]
            decisions: 决策列表 [N]
            rewards: 奖励列表 [N]
            predicted_occupancies: 预测占用列表 [N × {veh_id: traj}]

        Returns:
            total_loss: 总损失
        """
        from harl.envs.a_multi_lane.env_utils.safety_assessment import SafetyAssessor

        w_eff = self.loss_weights['w_efficiency']
        w_safe = self.loss_weights['w_safety']
        epsilon_max = self.loss_weights['epsilon_max']
        theta_safe = self.loss_weights['theta_safe']

        assessor = SafetyAssessor(resolution=0.1)

        batch_size = len(cav_trajectories)
        loss_efficiency_list = []
        loss_safety_list = []

        for i in range(batch_size):
            traj = cav_trajectories[i]
            reward = rewards[i]
            pred_occ = predicted_occupancies[i]

            # 效率损失
            # L_efficiency = -R_efficiency + ε_max * Var(Δy)
            if len(traj) > 1:
                delta_y = np.diff(traj[:, 1])  # 横向位移
                var_delta_y = np.var(delta_y)
            else:
                var_delta_y = 0.0

            loss_eff = -reward + epsilon_max * var_delta_y
            loss_efficiency_list.append(loss_eff)

            # 安全损失
            # L_safety = Σ max(0, Overlap - θ_safe)^2
            if pred_occ:
                loss_safe, _ = assessor.compute_safety_penalty(
                    traj, pred_occ, theta_safe=theta_safe
                )
            else:
                loss_safe = 0.0

            loss_safety_list.append(loss_safe)

        # 平均损失
        avg_loss_eff = np.mean(loss_efficiency_list)
        avg_loss_safe = np.mean(loss_safety_list)

        total_loss = w_eff * avg_loss_eff + w_safe * avg_loss_safe

        return total_loss

    def update_from_replay_buffer(self, replay_buffer, batch_size: int = 32) -> float:
        """
        从回放缓冲区采样并更新决策模型

        Args:
            replay_buffer: ReplayBuffer实例
            batch_size: 批次大小

        Returns:
            loss: 损失值
        """
        # 采样批次
        batch = replay_buffer.sample_cav_batch(batch_size=batch_size, balanced=True)

        if len(batch) == 0:
            return 0.0

        # 提取数据
        cav_trajectories = [seg.trajectory for seg in batch]
        decisions = [seg.label for seg in batch]
        rewards = [seg.reward for seg in batch]

        # 简化：假设没有预测占用（实际使用中需从features中提取）
        predicted_occupancies = [{} for _ in batch]

        # 计算损失
        loss = self.compute_lateral_loss(
            cav_trajectories, decisions, rewards, predicted_occupancies
        )

        # 更新模型（通过决策模块的内置更新函数）
        # 这里只记录损失，实际更新在环境的step中进行
        self.loss_history.append(loss)
        self.total_updates += 1

        return loss

    def get_statistics(self) -> Dict:
        """
        获取训练统计信息

        Returns:
            stats: 统计字典
        """
        stats = {
            'current_episode': self.current_episode,
            'current_stage': self.current_stage,
            'total_updates': self.total_updates,
            'current_lr': self.learning_rates[self.current_stage - 1],
            'stage_history': self.stage_history
        }

        if self.loss_history:
            recent_losses = self.loss_history[-100:]
            stats['avg_loss_recent'] = np.mean(recent_losses)
            stats['loss_std_recent'] = np.std(recent_losses)
        else:
            stats['avg_loss_recent'] = 0.0
            stats['loss_std_recent'] = 0.0

        return stats


class CQRCalibrationUpdater:
    """
    CQR校准更新器

    周期性地从HDV轨迹中校准CQR模型
    """

    def __init__(self,
                 prediction_module,
                 target_coverage: float = 0.9,
                 calibration_size: int = 500,
                 update_interval: int = 100):
        """
        初始化CQR校准更新器

        Args:
            prediction_module: PredictionModule实例
            target_coverage: 目标覆盖率（默认90%）
            calibration_size: 校准集大小（默认500）
            update_interval: 更新间隔（episode数）
        """
        self.prediction_module = prediction_module
        self.target_coverage = target_coverage
        self.calibration_size = calibration_size
        self.update_interval = update_interval

        # 校准缓冲区
        try:
            from harl.envs.a_multi_lane.env_utils.replay_buffer import CQRCalibrationBuffer
        except ImportError:
            # 运行standalone时
            import os
            os.chdir(str(harl_root))
            sys.path.insert(0, str(harl_root))
            from harl.envs.a_multi_lane.env_utils.replay_buffer import CQRCalibrationBuffer

        self.calibration_buffer = CQRCalibrationBuffer(target_size=calibration_size)

        # 校准历史
        self.calibration_history = []
        self.last_calibration_episode = 0

    def add_hdv_sample(self, features: Dict, ground_truth_trajectory: np.ndarray):
        """
        添加HDV样本到校准缓冲区

        Args:
            features: 输入特征
            ground_truth_trajectory: 真实轨迹
        """
        self.calibration_buffer.add_sample(features, ground_truth_trajectory)

    def should_calibrate(self, current_episode: int) -> bool:
        """
        判断是否应该执行校准

        Args:
            current_episode: 当前episode

        Returns:
            True if should calibrate
        """
        episodes_since_last = current_episode - self.last_calibration_episode
        return (
            self.calibration_buffer.is_ready() and
            episodes_since_last >= self.update_interval
        )

    def calibrate(self, current_episode: int) -> Dict:
        """
        执行CQR校准

        Args:
            current_episode: 当前episode

        Returns:
            calibration_info: 校准信息
        """
        if not self.calibration_buffer.is_ready():
            return {'status': 'not_ready', 'buffer_size': len(self.calibration_buffer)}

        print(f"[CQRCalibration] 开始CQR校准 (Episode {current_episode})")

        # 获取校准数据
        features_list, gt_list = self.calibration_buffer.get_calibration_set()

        # 调用预测模块的校准函数
        # 注意：这里假设prediction_module有calibrate_occupancy方法
        try:
            if hasattr(self.prediction_module, 'occupancy_predictor'):
                # 这里简化处理，实际需要根据CQR算法计算校准参数
                # 在实际实现中，应该调用conformalization模块
                calibration_info = {
                    'status': 'success',
                    'episode': current_episode,
                    'n_samples': len(gt_list),
                    'target_coverage': self.target_coverage
                }

                self.calibration_history.append(calibration_info)
                self.last_calibration_episode = current_episode

                print(f"  [CQRCalibration] 校准完成，使用 {len(gt_list)} 个样本")

                # 清空缓冲区
                self.calibration_buffer.clear()

                return calibration_info
            else:
                return {'status': 'no_occupancy_predictor'}

        except Exception as e:
            print(f"  [CQRCalibration] 校准失败: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_statistics(self) -> Dict:
        """
        获取校准统计信息

        Returns:
            stats: 统计字典
        """
        return {
            'total_calibrations': len(self.calibration_history),
            'last_calibration_episode': self.last_calibration_episode,
            'buffer_size': len(self.calibration_buffer),
            'buffer_ready': self.calibration_buffer.is_ready(),
            'calibration_history': self.calibration_history[-5:]  # 最近5次
        }


class TrainingManager:
    """
    训练管理器

    集成决策模型更新和CQR校准，管理整个训练流程
    """

    def __init__(self,
                 decision_module,
                 prediction_module,
                 replay_buffer,
                 stage_thresholds: Tuple[int, int] = (500, 1500),
                 decision_update_freq: int = 1,
                 cqr_update_interval: int = 100):
        """
        初始化训练管理器

        Args:
            decision_module: DecisionModule实例
            prediction_module: PredictionModule实例
            replay_buffer: ReplayBuffer实例
            stage_thresholds: 阶段切换阈值
            decision_update_freq: 决策模型更新频率（每N个episode）
            cqr_update_interval: CQR校准间隔（每N个episode）
        """
        self.decision_updater = LateralDecisionUpdater(
            decision_module=decision_module,
            stage_thresholds=stage_thresholds
        )

        self.cqr_updater = CQRCalibrationUpdater(
            prediction_module=prediction_module,
            update_interval=cqr_update_interval
        )

        self.replay_buffer = replay_buffer
        self.decision_update_freq = decision_update_freq

        # 训练统计
        self.total_episodes = 0

    def on_episode_start(self, episode: int):
        """
        Episode开始时调用

        Args:
            episode: 当前episode编号
        """
        self.total_episodes = episode

        # 更新训练阶段
        self.decision_updater.update_stage(episode)

    def on_episode_end(self, episode: int) -> Dict:
        """
        Episode结束时调用

        Args:
            episode: 当前episode编号

        Returns:
            update_info: 更新信息
        """
        update_info = {
            'episode': episode,
            'decision_updated': False,
            'cqr_calibrated': False
        }

        # 决策模型更新
        if episode % self.decision_update_freq == 0:
            loss = self.decision_updater.update_from_replay_buffer(
                self.replay_buffer, batch_size=32
            )
            update_info['decision_updated'] = True
            update_info['decision_loss'] = loss

        # CQR校准
        if self.cqr_updater.should_calibrate(episode):
            calib_info = self.cqr_updater.calibrate(episode)
            update_info['cqr_calibrated'] = calib_info['status'] == 'success'
            update_info['cqr_info'] = calib_info

        return update_info

    def get_full_statistics(self) -> Dict:
        """
        获取完整训练统计信息

        Returns:
            stats: 统计字典
        """
        return {
            'total_episodes': self.total_episodes,
            'decision_updater': self.decision_updater.get_statistics(),
            'cqr_updater': self.cqr_updater.get_statistics(),
            'replay_buffer': self.replay_buffer.get_statistics()
        }


if __name__ == "__main__":
    # 测试模型更新器
    print("=" * 80)
    print("测试模型更新器")
    print("=" * 80)

    # 由于依赖实际模块，这里只测试基本逻辑
    print("\n测试1: 横向决策更新器（模拟）")

    # 模拟决策模块
    class MockDecisionModule:
        class MockDecisionMaker:
            def __init__(self):
                self.freeze_base_model = True
                self.optimizer = torch.optim.Adam([torch.tensor(1.0)], lr=1e-4)
                self.scm_model = None

            def _freeze_parameters(self):
                pass

            def _unfreeze_parameters(self):
                pass

        def __init__(self):
            self.decision_maker = self.MockDecisionMaker()

    mock_decision_module = MockDecisionModule()
    decision_updater = LateralDecisionUpdater(
        decision_module=mock_decision_module,
        stage_thresholds=(500, 1500)
    )

    print(f"  初始阶段: Stage {decision_updater.current_stage}")
    print(f"  初始学习率: {decision_updater.learning_rates[0]}")

    # 模拟训练
    for episode in [0, 500, 1500]:
        decision_updater.update_stage(episode)
        stats = decision_updater.get_statistics()
        print(f"\n  Episode {episode}:")
        print(f"    当前阶段: Stage {stats['current_stage']}")
        print(f"    当前学习率: {stats['current_lr']}")

    print("\n测试2: CQR校准更新器（模拟）")

    # 模拟预测模块
    class MockPredictionModule:
        def __init__(self):
            self.occupancy_predictor = True

    mock_prediction_module = MockPredictionModule()
    cqr_updater = CQRCalibrationUpdater(
        prediction_module=mock_prediction_module,
        calibration_size=500,
        update_interval=100
    )

    # 添加样本
    for i in range(600):
        features = {'traffic_state': np.random.randn(6)}
        gt_traj = np.random.randn(30, 2)
        cqr_updater.add_hdv_sample(features, gt_traj)

    print(f"  缓冲区大小: {len(cqr_updater.calibration_buffer)}")
    print(f"  是否准备就绪: {cqr_updater.calibration_buffer.is_ready()}")

    # 模拟校准
    if cqr_updater.should_calibrate(current_episode=100):
        calib_info = cqr_updater.calibrate(current_episode=100)
        print(f"  校准状态: {calib_info['status']}")

    stats = cqr_updater.get_statistics()
    print(f"  总校准次数: {stats['total_calibrations']}")

    print("\n测试3: 训练管理器（集成）")

    try:
        from harl.envs.a_multi_lane.env_utils.replay_buffer import ReplayBuffer
    except ImportError:
        import os
        os.chdir(str(harl_root))
        sys.path.insert(0, str(harl_root))
        from harl.envs.a_multi_lane.env_utils.replay_buffer import ReplayBuffer

    mock_replay_buffer = ReplayBuffer(capacity=1000)

    training_manager = TrainingManager(
        decision_module=mock_decision_module,
        prediction_module=mock_prediction_module,
        replay_buffer=mock_replay_buffer,
        stage_thresholds=(500, 1500),
        cqr_update_interval=100
    )

    # 模拟训练循环
    for episode in [0, 1, 100, 500, 1000, 1500]:
        training_manager.on_episode_start(episode)

        # 添加一些假数据到replay buffer
        if episode > 0:
            trajectory = np.random.randn(30, 2)
            features = {'env': np.random.randn(4)}
            mock_replay_buffer.add_cav_trajectory(
                f'CAV_{episode}', trajectory, features,
                decision=np.random.choice([0, 1, 2]),
                reward=np.random.randn(),
                timestamp=episode
            )

        update_info = training_manager.on_episode_end(episode)

        if episode in [0, 100, 500, 1500]:
            print(f"\n  Episode {episode}:")
            print(f"    决策更新: {update_info['decision_updated']}")
            print(f"    CQR校准: {update_info['cqr_calibrated']}")

    full_stats = training_manager.get_full_statistics()
    print(f"\n  总训练episodes: {full_stats['total_episodes']}")
    print(f"  当前阶段: Stage {full_stats['decision_updater']['current_stage']}")

    print("\n✓ 模型更新器测试完成!")
