#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置

定义三阶段训练策略的所有参数配置
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Stage1Config:
    """
    阶段1配置：基础MARL训练（0-500 episodes）

    - 冻结所有预训练模型（SCM意图、SCM决策、CQR占用）
    - 仅训练MARL策略网络（Actor-Critic）
    - 重点：学习基本的多智能体协调策略
    """
    # 阶段范围
    episode_start: int = 0
    episode_end: int = 500

    # MARL参数
    marl_learning_rate: float = 5e-4
    marl_clip_param: float = 0.2
    marl_value_loss_coef: float = 1.0
    marl_entropy_coef: float = 0.01
    marl_max_grad_norm: float = 0.5

    # MARL奖励权重（根据00_structure.tex Eq. 22-27）
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.4,    # 效率权重
        'w_comfort': 0.3,       # 舒适性权重
        'w_safety': 0.3,        # 安全性权重
        'w_lc_penalty': 0.1,    # 换道惩罚权重
        'w_collision': 10.0     # 碰撞惩罚权重
    })

    # 预训练模型配置
    freeze_intention_model: bool = True
    freeze_occupancy_model: bool = True
    freeze_decision_model: bool = True

    # 决策模型参数（冻结状态）
    decision_lr: float = 0.0  # 冻结
    decision_update_freq: int = 0  # 不更新

    # CQR校准参数
    cqr_calibration_enabled: bool = False  # 不校准
    cqr_update_interval: int = 0

    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100


@dataclass
class Stage2Config:
    """
    阶段2配置：决策微调（500-1500 episodes）

    - 解冻横向决策模型的个体层
    - 开始CQR周期性校准
    - 继续训练MARL策略
    """
    # 阶段范围
    episode_start: int = 500
    episode_end: int = 1500

    # MARL参数（降低学习率）
    marl_learning_rate: float = 3e-4
    marl_clip_param: float = 0.2
    marl_value_loss_coef: float = 1.0
    marl_entropy_coef: float = 0.01
    marl_max_grad_norm: float = 0.5

    # MARL奖励权重（增加安全性权重）
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.35,
        'w_comfort': 0.3,
        'w_safety': 0.35,        # 增加
        'w_lc_penalty': 0.15,    # 增加
        'w_collision': 15.0      # 增加
    })

    # 预训练模型配置
    freeze_intention_model: bool = True
    freeze_occupancy_model: bool = True
    freeze_decision_model: bool = False  # 解冻个体层

    # 决策模型参数（根据00_structure.tex Eq. 15-21）
    decision_lr: float = 5e-5
    decision_update_freq: int = 1  # 每个episode更新

    # 决策损失权重
    decision_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.6,     # 效率损失权重
        'w_safety': 0.4,         # 安全损失权重
        'epsilon_max': 0.1,      # 横向位移方差惩罚系数
        'theta_safe': 0.3        # 安全阈值
    })

    # CQR校准参数
    cqr_calibration_enabled: bool = True
    cqr_update_interval: int = 100  # 每100个episode校准一次
    cqr_calibration_size: int = 500  # 校准集大小
    cqr_target_coverage: float = 0.9  # 目标覆盖率90%

    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100


@dataclass
class Stage3Config:
    """
    阶段3配置：全局优化（1500-3000 episodes）

    - 全局微调所有可训练模块
    - 增加CQR校准频率
    - 精细调整MARL策略
    """
    # 阶段范围
    episode_start: int = 1500
    episode_end: int = 3000

    # MARL参数（进一步降低学习率）
    marl_learning_rate: float = 1e-4
    marl_clip_param: float = 0.15
    marl_value_loss_coef: float = 1.0
    marl_entropy_coef: float = 0.005  # 降低探索
    marl_max_grad_norm: float = 0.5

    # MARL奖励权重（平衡三者）
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.35,
        'w_comfort': 0.3,
        'w_safety': 0.35,
        'w_lc_penalty': 0.2,
        'w_collision': 20.0
    })

    # 预训练模型配置（全局微调）
    freeze_intention_model: bool = True  # 意图模型保持冻结
    freeze_occupancy_model: bool = True  # 占用模型保持冻结
    freeze_decision_model: bool = False  # 全局微调

    # 决策模型参数
    decision_lr: float = 1e-5  # 降低学习率
    decision_update_freq: int = 1

    # 决策损失权重
    decision_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.5,     # 降低效率权重
        'w_safety': 0.5,         # 增加安全权重
        'epsilon_max': 0.15,     # 增加平滑性约束
        'theta_safe': 0.25       # 降低安全阈值（更严格）
    })

    # CQR校准参数（增加校准频率）
    cqr_calibration_enabled: bool = True
    cqr_update_interval: int = 50  # 每50个episode校准一次
    cqr_calibration_size: int = 500
    cqr_target_coverage: float = 0.9

    # 日志和保存
    log_interval: int = 10
    save_interval: int = 50  # 更频繁保存


@dataclass
class TrainingConfig:
    """
    完整训练配置

    集成三个阶段的所有参数
    """
    # 三阶段配置
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    # 全局参数
    total_episodes: int = 3000
    num_agents: int = 5  # CAV数量
    batch_size: int = 32
    buffer_capacity: int = 10000

    # 环境参数
    lane_width: float = 3.75  # 车道宽度（米）
    default_lane_change_time: float = 3.0  # 默认换道时间（秒）
    simulation_step_length: float = 0.1  # 仿真步长（秒）

    # 预测参数
    prediction_horizon: int = 30  # 预测时域（步）= 3秒
    intention_model_type: str = "shallow_hierarchical"
    occupancy_model_type: str = "CQR-GRU-uncertainty"
    decision_model_type: str = "shallow_hierarchical"

    # 安全评估参数
    occupancy_grid_resolution: float = 0.1  # 占用网格分辨率（米）
    safety_threshold: float = 0.3  # 安全重叠阈值
    ttc_threshold: float = 3.0  # TTC安全阈值（秒）

    # 设备配置
    device: str = "cpu"  # 或 "cuda" if available
    num_workers: int = 4
    seed: int = 42

    # 检查点和日志
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    tensorboard_dir: str = "./runs"

    def get_stage_config(self, episode: int):
        """
        根据episode获取对应阶段的配置

        Args:
            episode: 当前episode编号

        Returns:
            stage_config: 对应阶段的配置对象
        """
        if episode < self.stage1.episode_end:
            return self.stage1
        elif episode < self.stage2.episode_end:
            return self.stage2
        else:
            return self.stage3

    def get_stage_number(self, episode: int) -> int:
        """
        获取当前阶段编号

        Args:
            episode: 当前episode编号

        Returns:
            stage_number: 1, 2, 或 3
        """
        if episode < self.stage1.episode_end:
            return 1
        elif episode < self.stage2.episode_end:
            return 2
        else:
            return 3


def get_default_training_config() -> TrainingConfig:
    """
    获取默认训练配置

    Returns:
        config: TrainingConfig实例
    """
    return TrainingConfig()


def print_stage_summary(config: TrainingConfig):
    """
    打印训练配置摘要

    Args:
        config: TrainingConfig实例
    """
    print("=" * 80)
    print("训练配置摘要")
    print("=" * 80)

    print(f"\n总Episodes: {config.total_episodes}")
    print(f"智能体数量: {config.num_agents}")
    print(f"批次大小: {config.batch_size}")
    print(f"设备: {config.device}")

    print("\n" + "=" * 80)
    print("阶段1: 基础MARL训练 (Episodes 0-500)")
    print("=" * 80)
    print(f"  MARL学习率: {config.stage1.marl_learning_rate}")
    print(f"  奖励权重: 效率={config.stage1.reward_weights['w_efficiency']}, "
          f"舒适性={config.stage1.reward_weights['w_comfort']}, "
          f"安全={config.stage1.reward_weights['w_safety']}")
    print(f"  冻结状态: 意图={config.stage1.freeze_intention_model}, "
          f"占用={config.stage1.freeze_occupancy_model}, "
          f"决策={config.stage1.freeze_decision_model}")
    print(f"  CQR校准: {'关闭' if not config.stage1.cqr_calibration_enabled else '开启'}")

    print("\n" + "=" * 80)
    print("阶段2: 决策微调 (Episodes 500-1500)")
    print("=" * 80)
    print(f"  MARL学习率: {config.stage2.marl_learning_rate}")
    print(f"  决策学习率: {config.stage2.decision_lr}")
    print(f"  决策更新频率: 每{config.stage2.decision_update_freq}个episode")
    print(f"  决策损失权重: 效率={config.stage2.decision_loss_weights['w_efficiency']}, "
          f"安全={config.stage2.decision_loss_weights['w_safety']}")
    print(f"  CQR校准: 每{config.stage2.cqr_update_interval}个episode")
    print(f"  安全阈值θ_safe: {config.stage2.decision_loss_weights['theta_safe']}")

    print("\n" + "=" * 80)
    print("阶段3: 全局优化 (Episodes 1500-3000)")
    print("=" * 80)
    print(f"  MARL学习率: {config.stage3.marl_learning_rate}")
    print(f"  决策学习率: {config.stage3.decision_lr}")
    print(f"  熵系数: {config.stage3.marl_entropy_coef} (降低探索)")
    print(f"  决策损失权重: 效率={config.stage3.decision_loss_weights['w_efficiency']}, "
          f"安全={config.stage3.decision_loss_weights['w_safety']}")
    print(f"  CQR校准: 每{config.stage3.cqr_update_interval}个episode")
    print(f"  安全阈值θ_safe: {config.stage3.decision_loss_weights['theta_safe']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 测试训练配置
    print("=" * 80)
    print("测试训练配置")
    print("=" * 80)

    # 创建默认配置
    config = get_default_training_config()

    # 打印配置摘要
    print_stage_summary(config)

    # 测试阶段获取
    print("\n" + "=" * 80)
    print("测试阶段获取")
    print("=" * 80)

    test_episodes = [0, 250, 500, 1000, 1500, 2500]
    for ep in test_episodes:
        stage_num = config.get_stage_number(ep)
        stage_cfg = config.get_stage_config(ep)
        print(f"\nEpisode {ep}:")
        print(f"  阶段: Stage {stage_num}")
        print(f"  MARL学习率: {stage_cfg.marl_learning_rate}")
        print(f"  决策学习率: {stage_cfg.decision_lr}")
        print(f"  冻结决策模型: {stage_cfg.freeze_decision_model}")

    # 测试配置保存
    print("\n" + "=" * 80)
    print("测试配置序列化")
    print("=" * 80)

    import json
    from dataclasses import asdict

    config_dict = asdict(config)
    print(f"配置字典键: {list(config_dict.keys())}")
    print(f"Stage1配置键: {list(config_dict['stage1'].keys())[:5]}...")

    # 保存为JSON
    import tempfile
    temp_file = tempfile.mktemp(suffix='.json')

    with open(temp_file, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n配置已保存到: {temp_file}")

    # 加载验证
    with open(temp_file, 'r') as f:
        loaded_config = json.load(f)

    print(f"加载后Stage1开始Episode: {loaded_config['stage1']['episode_start']}")
    print(f"加载后Stage2结束Episode: {loaded_config['stage2']['episode_end']}")

    # 清理
    import os
    os.remove(temp_file)

    print("\n✓ 训练配置测试完成!")
