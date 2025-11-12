# MARL多车道系统训练执行指南

## 文档概述

本文档详细阐述整个MARL多车道系统的训练规则、训练过程、模型更新策略以及训练指标监控方法。

**版本**: v1.0
**日期**: 2025-01
**作者**: 交通流研究团队

---

## 目录

1. [系统架构概述](#系统架构概述)
2. [训练前准备](#训练前准备)
3. [训练启动方法](#训练启动方法)
4. [三阶段训练策略](#三阶段训练策略)
5. [模型更新与存储](#模型更新与存储)
6. [训练指标监控](#训练指标监控)
7. [训练日志系统](#训练日志系统)
8. [训练配置参数](#训练配置参数)
9. [故障排查指南](#故障排查指南)
10. [性能优化建议](#性能优化建议)

---

## 系统架构概述

### 整体框架

根据[harl/envs/00_structure.tex](harl/envs/00_structure.tex),整个系统包含以下核心模块:

```
┌─────────────────────────────────────────────────────────────┐
│                    训练入口 (examples/train.py)              │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                              │
  ┌───────▼────────┐           ┌────────▼────────┐
  │  MAPPO算法     │           │  环境交互模块    │
  │  (harl/algos)  │◄─────────►│ (a_multi_lane)  │
  └───────┬────────┘           └────────┬────────┘
          │                              │
  ┌───────┴────────┐           ┌────────┴────────┐
  │ Actor/Critic   │           │  特征提取器      │
  │   网络设计     │           │  预测模块        │
  │ (harl/models)  │           │  决策模块        │
  └────────────────┘           │  安全评估        │
                               │  模型更新器      │
                               └─────────────────┘
```

### 关键组件说明

1. **MAPPO算法**: 多智能体近端策略优化算法
   - Actor: 生成CAV纵向加速度动作
   - Critic: 评估全局状态价值

2. **预测模块**:
   - 意图预测 (SCM-based)
   - 占用预测 (CQR-based)

3. **决策模块**:
   - 横向决策 (SCM-based,支持渐进式微调)

4. **模型更新器**:
   - 三阶段训练策略管理
   - 横向决策模型微调
   - CQR周期性校准

---

## 训练前准备

### 1. 环境检查

确保以下依赖已安装:

```bash
# Python 3.8+
python --version

# PyTorch 1.13+
python -c "import torch; print(torch.__version__)"

# SUMO (交通仿真环境)
sumo --version

# 其他依赖
pip install -r requirements.txt
```

### 2. 预训练模型准备

训练需要以下预训练模型:

```bash
# 检查预训练模型是否存在
ls harl/prediction/intention/saved_models/
# 应包含: shallow_hierarchical_scm_v2.pth

ls harl/prediction/occupancy/saved_models/
# 应包含: gru_cqr_model.pth

ls harl/decisions/lateral_decisions/saved_models/
# 应包含: decision_scm_model.pth
```

如果缺少预训练模型,请参考各模块的README进行训练或下载。

### 3. 数据集准备(可选)

如果需要使用真实数据进行CQR校准:

```bash
# 准备NGSIM或HighD数据集
# 数据应放置在: data/trajectory_data/
```

### 4. 配置文件检查

检查YAML配置文件:

```bash
# MAPPO算法配置
cat harl/configs/algos_cfgs/mappo.yaml

# 环境配置
cat harl/configs/envs_cfgs/a_multi_lane.yaml
```

---

## 训练启动方法

### 基础训练命令

#### 1. 使用默认配置训练

```bash
cd /path/to/MARL_MultiLane

# 基础训练(MAPPO + 多车道环境)
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name my_first_training \
  --test_desc "baseline_test"
```

#### 2. 使用自定义配置

```bash
# 创建配置JSON文件
cat > configs/my_config.json << EOF
{
  "main_args": {
    "algo": "mappo",
    "env": "a_multi_lane"
  },
  "algo_args": {
    "hidden_sizes": [256, 256],
    "lr": 0.0005,
    "use_recurrent_policy": false
  },
  "env_args": {
    "num_agents": 5,
    "episode_length": 3600,
    "traffic_density": "medium"
  }
}
EOF

# 使用配置文件训练
python examples/train.py \
  --load_config configs/my_config.json \
  --exp_name my_custom_training
```

#### 3. 命令行覆盖参数

```bash
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name high_lr_test \
  --lr 0.001 \
  --hidden_sizes [512,512] \
  --num_agents 10 \
  --episode_length 7200
```

### 训练参数说明

#### 必选参数

- `--algo`: 算法名称 (`mappo`)
- `--env`: 环境名称 (`a_multi_lane`)
- `--exp_name`: 实验名称(用于日志和模型保存)

#### 常用可选参数

```bash
# MARL算法参数
--lr                    # MARL学习率(默认: 5e-4)
--hidden_sizes          # 网络隐藏层大小(默认: [256,256])
--gamma                 # 折扣因子(默认: 0.99)
--clip_param            # PPO裁剪参数(默认: 0.2)

# 环境参数
--num_agents            # CAV数量(默认: 5)
--episode_length        # Episode长度/秒(默认: 3600)
--traffic_density       # 交通密度(low/medium/high)
--lane_width            # 车道宽度/米(默认: 3.75)

# 训练参数
--n_rollout_threads     # 并行环境数(默认: 1)
--num_episodes          # 训练episode数(默认: 3000)
--seed                  # 随机种子(默认: 1)

# 预测与决策参数
--use_prediction        # 是否启用预测模块(默认: True)
--use_decision          # 是否启用决策模块(默认: True)
--freeze_decision_init  # 初始是否冻结决策模型(默认: True)

# 日志参数
--use_wandb             # 是否使用W&B(默认: False)
--save_interval         # 模型保存间隔/episode(默认: 100)
```

---

## 三阶段训练策略

根据[harl/envs/a_multi_lane/env_utils/training_config.py](harl/envs/a_multi_lane/env_utils/training_config.py)设计。

### 阶段1: 基础MARL训练 (Episodes 0-500)

**目标**: 学习基本的多智能体协调策略

#### 配置特点

```python
{
    "episode_range": (0, 500),
    "marl_lr": 5e-4,                  # MARL学习率
    "decision_lr": 0.0,                # 决策学习率(冻结)
    "freeze_decision": True,           # 冻结决策模型
    "cqr_update_interval": None,       # 不进行CQR校准
    "loss_weights": {
        "w_efficiency": 0.4,
        "w_comfort": 0.3,
        "w_safety": 0.3
    }
}
```

#### 训练要点

1. **冻结所有预训练模型**
   - SCM意图预测模型: 冻结
   - CQR占用预测模型: 冻结
   - SCM横向决策模型: 冻结

2. **仅训练MARL Actor/Critic**
   - Actor网络学习纵向控制策略
   - Critic网络学习状态价值函数

3. **使用预训练模型输出作为观测增强**
   - 意图预测结果 → 提供周车未来意图
   - 占用预测结果 → 提供安全评估依据
   - 横向决策结果 → 提供横向动作参考

#### 启动命令

```bash
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name stage1_training \
  --test_desc "Stage1: Frozen Pretrained Models" \
  --freeze_decision_init True \
  --num_episodes 500 \
  --lr 5e-4
```

#### 预期效果

- Episode 100: CAV开始学会跟车
- Episode 300: CAV学会简单的车道保持
- Episode 500: 基本的多车协调能力

---

### 阶段2: 决策微调 (Episodes 500-1500)

**目标**: 微调横向决策模型,开始CQR校准

#### 配置特点

```python
{
    "episode_range": (500, 1500),
    "marl_lr": 3e-4,                   # 降低MARL学习率
    "decision_lr": 5e-5,               # 启用决策学习
    "freeze_decision": False,          # 解冻决策模型
    "unfreeze_layers": ["individual_encoder", "decision_threshold"],
    "cqr_update_interval": 100,        # 每100 episode校准CQR
    "loss_weights": {
        "w_efficiency": 0.4,
        "w_comfort": 0.25,
        "w_safety": 0.35,              # 增加安全权重
        "theta_safe": 0.3,             # 安全阈值
        "epsilon_max": 0.1
    }
}
```

#### 训练要点

1. **解冻决策模型个体层**
   - 个体层因果机制: 解冻
   - 环境层因果机制: 保持冻结
   - 决策阈值: 可训练

2. **开始CQR周期性校准**
   - 每100 episode从HDV轨迹校准CQR
   - 更新CAV和HDV的共形化参数
   - 提高占用预测覆盖率

3. **横向决策损失函数**

根据00_structure.tex公式(Eq. 15-21):

```
L_lateral = w_efficiency * L_efficiency + w_safety * L_safety

L_efficiency = -R_efficiency + ε_max * Var(Δy)
L_safety = Σ_j max(0, Overlap(τ_ego, τ_j^pred) - θ_safe)²
```

#### 启动命令

```bash
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name stage2_training \
  --test_desc "Stage2: Decision Fine-tuning + CQR Calibration" \
  --load_pretrained path/to/stage1_model.pt \
  --start_episode 500 \
  --num_episodes 1500 \
  --lr 3e-4 \
  --decision_lr 5e-5 \
  --cqr_calibration_interval 100
```

#### 预期效果

- Episode 600: 决策模型开始适应MARL策略
- Episode 900: CQR校准后占用预测更准确
- Episode 1200: 决策与MARL策略协同优化
- Episode 1500: 安全性显著提升

---

### 阶段3: 全局优化 (Episodes 1500-3000)

**目标**: 全局微调所有模块,达到最优性能

#### 配置特点

```python
{
    "episode_range": (1500, 3000),
    "marl_lr": 1e-4,                   # 进一步降低学习率
    "decision_lr": 1e-5,               # 小学习率精细调优
    "freeze_decision": False,          # 全局微调
    "unfreeze_layers": "all",          # 所有层可训练
    "cqr_update_interval": 50,         # 更频繁的CQR校准
    "loss_weights": {
        "w_efficiency": 0.4,
        "w_comfort": 0.25,
        "w_safety": 0.35,
        "theta_safe": 0.25,            # 更严格的安全阈值
        "epsilon_max": 0.08
    }
}
```

#### 训练要点

1. **全局微调所有模块**
   - SCM决策模型: 环境层+个体层+融合层全部微调
   - MARL Actor/Critic: 继续优化
   - 所有模块联合训练

2. **更频繁的CQR校准**
   - 每50 episode校准一次
   - 保持预测模型与实际数据分布一致

3. **更严格的安全约束**
   - 降低安全阈值θ_safe至0.25
   - 减小epsilon_max至0.08
   - 强化安全惩罚

#### 启动命令

```bash
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name stage3_training \
  --test_desc "Stage3: Global Fine-tuning" \
  --load_pretrained path/to/stage2_model.pt \
  --start_episode 1500 \
  --num_episodes 3000 \
  --lr 1e-4 \
  --decision_lr 1e-5 \
  --cqr_calibration_interval 50 \
  --theta_safe 0.25
```

#### 预期效果

- Episode 1800: 所有模块协同工作
- Episode 2200: 接近最优性能
- Episode 2700: 策略稳定收敛
- Episode 3000: 训练完成,达到最优

---

## 模型更新与存储

### 自动模型保存

训练过程中会自动保存模型:

```bash
# 模型保存路径结构
results/
└── a_multi_lane/
    └── mappo/
        └── {exp_name}/
            ├── models/
            │   ├── actor_episode_100.pt
            │   ├── critic_episode_100.pt
            │   ├── decision_model_episode_600.pt    # 阶段2开始保存
            │   ├── cqr_calibration_episode_600.pkl
            │   ├── actor_episode_200.pt
            │   └── ...
            ├── logs/
            │   ├── training.log
            │   ├── multilane_metrics.log
            │   └── tensorboard/
            └── configs/
                └── config.json
```

### 模型保存策略

#### 1. MARL模型保存

```python
# 每save_interval个episode保存一次
save_interval = 100  # 默认值

# 保存内容
{
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'actor_optimizer': actor_optimizer.state_dict(),
    'critic_optimizer': critic_optimizer.state_dict(),
    'episode': current_episode,
    'total_steps': total_steps,
    'avg_reward': avg_reward
}
```

#### 2. 决策模型保存

```python
# 阶段2开始,每次更新后保存
# 保存触发条件: decision_loss下降或每100 episode

{
    'model_state_dict': decision_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'decision_threshold': threshold_value,
    'episode': current_episode,
    'loss_history': loss_list,
    'stage': current_stage
}
```

#### 3. CQR校准参数保存

```python
# 每次CQR校准后保存
{
    'hdv_calibration_params': {
        'conformal_scores': scores_array,
        'quantile_threshold': threshold,
        'coverage_achieved': coverage_rate
    },
    'cav_calibration_params': {...},
    'calibration_episode': episode,
    'buffer_size': buffer_size
}
```

### 手动保存检查点

```bash
# 在训练脚本中添加
python examples/train.py \
  ... \
  --save_checkpoint_at 500,1000,1500,2000,2500  # 指定episode保存
```

### 加载预训练模型继续训练

```bash
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name resume_training \
  --load_pretrained results/a_multi_lane/mappo/previous_exp/models/actor_episode_1000.pt \
  --start_episode 1000 \
  --num_episodes 3000
```

---

## 训练指标监控

### 核心指标体系

根据[harl/envs/a_multi_lane/multilane_logger.py](harl/envs/a_multi_lane/multilane_logger.py)设计。

#### 1. MARL训练指标

```python
# Actor性能指标
{
    'policy_loss': float,          # Actor损失
    'entropy': float,               # 策略熵(探索程度)
    'approx_kl': float,            # KL散度
    'clip_fraction': float          # 裁剪比例
}

# Critic性能指标
{
    'value_loss': float,           # Critic损失
    'explained_variance': float,   # 解释方差
    'value_pred': float,           # 价值预测均值
    'value_target': float          # 价值目标均值
}

# 整体性能
{
    'avg_episode_reward': float,   # 平均episode奖励
    'success_rate': float,         # 成功率
    'collision_rate': float        # 碰撞率
}
```

#### 2. 预测模型指标

```python
# 意图预测准确率
{
    'intention_accuracy_hdv': float,  # HDV意图预测准确率
    'intention_accuracy_cav': float,  # CAV意图预测准确率
    'intention_left_precision': float,
    'intention_right_precision': float,
    'intention_keep_precision': float
}

# 占用预测性能
{
    'occupancy_coverage_rate': float,  # 覆盖率(目标90%)
    'occupancy_interval_width': float, # 预测区间宽度
    'occupancy_ade': float,            # 平均位移误差
    'occupancy_fde': float             # 最终位移误差
}

# CQR校准指标
{
    'cqr_hdv_coverage': float,        # HDV CQR覆盖率
    'cqr_cav_coverage': float,        # CAV CQR覆盖率
    'cqr_conformal_score': float,     # 共形化分数
    'cqr_last_calibration_ep': int    # 上次校准episode
}
```

#### 3. 横向决策指标

```python
{
    'decision_loss': float,           # 决策损失
    'decision_efficiency_loss': float,
    'decision_safety_loss': float,
    'decision_update_count': int,
    'decision_lr': float,             # 当前学习率
    'decision_stage': int,            # 当前训练阶段(1/2/3)
    'lane_change_rate': float,        # 换道率
    'lane_change_success_rate': float # 换道成功率
}
```

#### 4. 安全评估指标

```python
{
    'safety_penalty_avg': float,      # 平均安全惩罚
    'ttc_violations': int,            # TTC违规次数
    'collision_count': int,           # 碰撞次数
    'near_miss_count': int,           # 险情次数
    'overlap_ratio_avg': float        # 平均重叠率
}
```

#### 5. 交通效率指标

```python
{
    'avg_speed_cav': float,           # CAV平均速度
    'avg_speed_all': float,           # 所有车辆平均速度
    'throughput': float,              # 通行量(veh/hour)
    'lane_utilization': [float, float, float],  # 各车道利用率
    'time_headway_avg': float,        # 平均车头时距
    'spacing_variance': float         # 车间距方差
}
```

### 实时监控方法

#### 1. 终端输出

训练过程中终端实时显示:

```
[Episode 500/3000] Stage 1
========================================
MARL Metrics:
  Avg Reward: 125.43
  Policy Loss: 0.0234
  Value Loss: 12.56
  Entropy: 1.23

Prediction Metrics:
  Intention Acc: 87.3%
  Occupancy Coverage: 89.1%

Decision Metrics:
  Decision Loss: N/A (Frozen)
  Lane Change Rate: 12.5%

Safety Metrics:
  Collision Rate: 0.8%
  TTC Violations: 3
  Safety Penalty: -2.34

Traffic Metrics:
  Avg Speed: 25.6 m/s
  Throughput: 1800 veh/h
========================================
Time: 00:15:32 | ETA: 01:31:08
```

#### 2. TensorBoard可视化

启动TensorBoard:

```bash
tensorboard --logdir results/a_multi_lane/mappo/{exp_name}/logs/tensorboard
```

可视化内容:

- **Scalars**: 所有数值指标的曲线
- **Distributions**: 奖励分布、动作分布
- **Histograms**: 网络权重和梯度分布
- **Custom**: 安全评估热力图、轨迹可视化

#### 3. Weights & Biases (可选)

```bash
# 启用W&B
python examples/train.py \
  ... \
  --use_wandb True \
  --wandb_entity your_entity \
  --wandb_project marl_multilane \
  --wandb_name {exp_name}
```

W&B提供:
- 实时指标同步
- 超参数对比
- 模型版本管理
- 团队协作

#### 4. 自定义日志查询

```python
# 读取日志文件
import pandas as pd

# 加载MARL训练日志
df_marl = pd.read_csv('results/.../logs/marl_training.csv')

# 加载多车道专用日志
df_multilane = pd.read_csv('results/.../logs/multilane_metrics.csv')

# 绘制指标
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df_marl['episode'], df_marl['avg_reward'])
plt.title('Average Reward')

plt.subplot(1, 3, 2)
plt.plot(df_multilane['episode'], df_multilane['collision_rate'])
plt.title('Collision Rate')

plt.subplot(1, 3, 3)
plt.plot(df_multilane['episode'], df_multilane['decision_loss'])
plt.title('Decision Loss')

plt.tight_layout()
plt.savefig('training_metrics.png')
```

---

## 训练日志系统

### 日志文件结构

```
results/a_multi_lane/mappo/{exp_name}/logs/
├── training.log                    # 主日志文件
├── marl_training.csv               # MARL指标CSV
├── multilane_metrics.csv           # 多车道专用指标CSV
├── decision_updates.log            # 决策模型更新日志
├── cqr_calibration.log             # CQR校准日志
├── safety_events.log               # 安全事件日志
└── tensorboard/
    ├── events.out.tfevents.xxx
    └── ...
```

### 日志级别与内容

#### 1. INFO级别

记录正常训练过程:

```
[2025-01-11 10:15:32] INFO: Episode 500 completed
[2025-01-11 10:15:32] INFO: Avg Reward: 125.43
[2025-01-11 10:15:33] INFO: Models saved to results/...
```

#### 2. WARNING级别

记录需要注意的情况:

```
[2025-01-11 10:20:15] WARNING: Collision detected in episode 523
[2025-01-11 10:25:42] WARNING: CQR coverage below target: 85.2% < 90%
[2025-01-11 10:30:18] WARNING: Decision loss spike: 0.456 > threshold 0.3
```

#### 3. ERROR级别

记录错误和异常:

```
[2025-01-11 10:35:20] ERROR: SUMO simulation crashed in episode 547
[2025-01-11 10:35:20] ERROR: Traceback: ...
[2025-01-11 10:40:12] ERROR: NaN detected in actor gradients
```

### 多车道专用日志

[harl/envs/a_multi_lane/multilane_logger.py](harl/envs/a_multi_lane/multilane_logger.py)提供专门的日志接口:

```python
from harl.envs.a_multi_lane.multilane_logger import MultiLaneLogger

# 初始化logger
logger = MultiLaneLogger(
    log_dir='results/a_multi_lane/mappo/exp_name/logs',
    log_to_console=True,
    log_to_file=True,
    log_to_tensorboard=True
)

# 记录MARL指标
logger.log_marl_metrics(
    episode=500,
    avg_reward=125.43,
    policy_loss=0.0234,
    value_loss=12.56,
    ...
)

# 记录预测指标
logger.log_prediction_metrics(
    episode=500,
    intention_acc=0.873,
    occupancy_coverage=0.891,
    ...
)

# 记录决策更新
logger.log_decision_update(
    episode=600,
    decision_loss=0.0145,
    efficiency_loss=-15.6,
    safety_loss=2.34,
    lr=5e-5,
    stage=2
)

# 记录CQR校准
logger.log_cqr_calibration(
    episode=700,
    hdv_coverage=0.902,
    cav_coverage=0.895,
    buffer_size=500
)

# 记录安全事件
logger.log_safety_event(
    episode=520,
    event_type='near_miss',
    vehicles=['CAV_1', 'HDV_5'],
    ttc=1.2,
    overlap_ratio=0.05
)
```

---

## 训练配置参数

### 完整参数列表

#### MARL算法参数 (harl/configs/algos_cfgs/mappo.yaml)

```yaml
# 基础参数
algo_name: "mappo"
gamma: 0.99                     # 折扣因子
use_gae: True                   # 是否使用GAE
gae_lambda: 0.95                # GAE参数
use_policy_active_masks: True
use_naive_recurrent_policy: False
use_recurrent_policy: False
recurrent_n: 1

# 学习参数
lr: 5.0e-4                      # 学习率
critic_lr: 5.0e-4               # Critic学习率
opti_eps: 1.0e-5                # Adam epsilon
weight_decay: 0

# PPO参数
ppo_epoch: 15                   # PPO更新轮数
use_clipped_value_loss: True
clip_param: 0.2                 # PPO裁剪参数
num_mini_batch: 1
entropy_coef: 0.01              # 熵系数
value_loss_coef: 1
use_max_grad_norm: True
max_grad_norm: 10.0

# 网络参数
hidden_sizes: [256, 256]
activation_func: "relu"
use_feature_normalization: True
use_orthogonal_init: True
gain: 0.01

# 其他
use_popart: True
use_valuenorm: True
use_policy_vhead: False
```

#### 环境参数 (harl/configs/envs_cfgs/a_multi_lane.yaml)

```yaml
# 环境基础配置
env_name: "a_multi_lane"
scenario_name: "highway_merge"
num_agents: 5                   # CAV数量
max_cycles: 3600                # Episode最大步数(秒)

# 车道配置
num_lanes: 3
lane_width: 3.75                # 米
road_length: 2000               # 米

# 交通流配置
traffic_density: "medium"       # low/medium/high
hdv_penetration_rate: 0.8       # HDV比例
initial_speed: 25.0             # 初始速度 m/s
speed_limit: 33.3               # 限速 120 km/h

# 车辆参数
vehicle_length: 5.0
vehicle_width: 2.0
max_acceleration: 3.0           # 最大加速度 m/s²
max_deceleration: -5.0          # 最大减速度 m/s²
comfortable_deceleration: -3.0

# 预测模块配置
use_prediction: True
intention_model_type: "shallow_hierarchical"
occupancy_model_type: "gru_cqr"
prediction_horizon: 30          # 预测时间步(3秒)

# 决策模块配置
use_decision: True
decision_model_type: "scm_decision"
freeze_decision_init: True
lane_change_duration: 3.0       # 换道时长(秒)

# 安全配置
use_safety_assessment: True
occupancy_resolution: 0.1       # 占用网格分辨率(米)
ttc_threshold: 2.5              # TTC阈值(秒)
safety_penalty_coef: 1.0

# 奖励权重
reward_efficiency_weight: 0.4
reward_comfort_weight: 0.3
reward_safety_weight: 0.3

# 训练配置
use_training_manager: True
stage_thresholds: [500, 1500]   # 阶段切换episode数
cqr_update_interval_stage2: 100
cqr_update_interval_stage3: 50
```

#### 三阶段配置 (harl/envs/a_multi_lane/env_utils/training_config.py)

```python
@dataclass
class Stage1Config:
    episode_start: int = 0
    episode_end: int = 500
    marl_lr: float = 5e-4
    decision_lr: float = 0.0
    freeze_decision: bool = True
    cqr_update_interval: Optional[int] = None
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.4,
        'w_comfort': 0.3,
        'w_safety': 0.3
    })

@dataclass
class Stage2Config:
    episode_start: int = 500
    episode_end: int = 1500
    marl_lr: float = 3e-4
    decision_lr: float = 5e-5
    freeze_decision: bool = False
    unfreeze_layers: List[str] = field(default_factory=lambda: [
        'individual_encoder', 'decision_threshold'
    ])
    cqr_update_interval: int = 100
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.4,
        'w_comfort': 0.25,
        'w_safety': 0.35,
        'theta_safe': 0.3,
        'epsilon_max': 0.1
    })

@dataclass
class Stage3Config:
    episode_start: int = 1500
    episode_end: int = 3000
    marl_lr: float = 1e-4
    decision_lr: float = 1e-5
    freeze_decision: bool = False
    unfreeze_layers: List[str] = field(default_factory=lambda: ['all'])
    cqr_update_interval: int = 50
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_efficiency': 0.4,
        'w_comfort': 0.25,
        'w_safety': 0.35,
        'theta_safe': 0.25,
        'epsilon_max': 0.08
    })
```

---

## 故障排查指南

### 常见问题与解决方案

#### 1. 训练无法启动

**症状**: 运行train.py后立即报错

**可能原因**:
- 预训练模型路径错误
- SUMO未安装或配置错误
- 依赖包缺失

**解决方案**:
```bash
# 检查预训练模型
ls harl/prediction/intention/saved_models/
ls harl/prediction/occupancy/saved_models/
ls harl/decisions/lateral_decisions/saved_models/

# 检查SUMO
sumo --version
export SUMO_HOME=/usr/share/sumo  # 根据实际路径

# 安装依赖
pip install -r requirements.txt
```

#### 2. CQR导入错误

**症状**: `ImportError: cannot import name 'FeatureEncoder'`

**原因**: [harl/prediction/occupancy/strategy/CQR_prediction/models.py:25](harl/prediction/occupancy/strategy/CQR_prediction/models.py:25) 导入错误

**解决方案**: 已修复,确保使用最新代码:
```python
# 正确的导入
from .feature_encoder import FeatureEncoder
```

#### 3. 训练过程中NaN

**症状**: Policy loss或Value loss变为NaN

**可能原因**:
- 学习率过大
- 梯度爆炸
- 奖励尺度问题

**解决方案**:
```bash
# 降低学习率
--lr 1e-4 --critic_lr 1e-4

# 调整梯度裁剪
--max_grad_norm 5.0

# 启用值归一化
--use_valuenorm True
```

#### 4. 决策模型参数冻结失败

**症状**: 阶段1中decision loss不为0

**原因**: `_freeze_parameters`未正确调用

**解决方案**: 已修复,检查[harl/decisions/lateral_decisions/SCM_decisions/execute_decision.py](harl/decisions/lateral_decisions/SCM_decisions/execute_decision.py) 中的`_freeze_parameters`方法

#### 5. CQR校准失败

**症状**: `CQRCalibrationBuffer not ready`

**原因**: HDV轨迹数据不足

**解决方案**:
```bash
# 确保有足够的HDV
--hdv_penetration_rate 0.8  # 增加HDV比例

# 延长Episode长度以收集更多数据
--episode_length 7200  # 2小时
```

#### 6. 内存溢出

**症状**: `CUDA out of memory`或系统内存不足

**解决方案**:
```bash
# 减小batch size
--num_mini_batch 2

# 减少并行环境数
--n_rollout_threads 1

# 使用CPU
--cuda False
```

#### 7. 训练速度慢

**症状**: 每个episode耗时过长

**优化方案**:
```bash
# 使用GPU
--cuda True

# 减少PPO更新轮数
--ppo_epoch 10

# 减少预测频率
--prediction_interval 5  # 每5步预测一次
```

### 调试模式

启用详细日志:

```bash
python examples/train.py \
  ... \
  --log_level DEBUG \
  --verbose True \
  --save_replay True  # 保存轨迹用于调试
```

---

## 性能优化建议

### 1. 计算性能优化

#### 使用GPU加速

```bash
python examples/train.py \
  ... \
  --cuda True \
  --cuda_deterministic False  # 允许非确定性以提速
```

#### 并行环境

```bash
python examples/train.py \
  ... \
  --n_rollout_threads 8  # 8个并行环境
```

#### 模型优化

```yaml
# 使用更小的网络
hidden_sizes: [128, 128]  # 替代 [256, 256]

# 使用浅层预测模型
intention_model_type: "shallow_hierarchical"
occupancy_model_type: "mlp_cqr"  # 替代 "gru_cqr"
```

### 2. 内存优化

#### 减少缓冲区大小

```python
# 修改 training_config.py
buffer_capacity: 5000  # 默认10000
cqr_calibration_size: 300  # 默认500
```

#### 清理历史数据

```python
# 仅保留最近N个episode的轨迹
trajectory_history_size: 100
```

### 3. 训练效率优化

#### 课程学习

```bash
# 逐步增加难度
# Stage 1: 低密度交通
--traffic_density low

# Stage 2: 中等密度
--traffic_density medium

# Stage 3: 高密度
--traffic_density high
```

#### 经验复用

```bash
# 使用更大的replay buffer
--buffer_size 100000

# 增加重用次数
--ppo_epoch 20
```

#### 学习率调度

```python
# 使用余弦退火
lr_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_episodes,
    eta_min=1e-6
)
```

### 4. 超参数调优建议

#### 关键超参数

```yaml
# Actor学习率: 影响策略更新速度
lr: [1e-5, 1e-4, 5e-4, 1e-3]

# Entropy系数: 影响探索程度
entropy_coef: [0.001, 0.01, 0.1]

# PPO裁剪参数: 影响更新幅度
clip_param: [0.1, 0.2, 0.3]

# 折扣因子: 影响长期奖励权重
gamma: [0.95, 0.99, 0.995]
```

#### 调优工具

```bash
# 使用Optuna自动调参
pip install optuna

python tune_hyperparams.py \
  --n_trials 50 \
  --study_name marl_multilane_tuning
```

---

## 总结与最佳实践

### 训练流程最佳实践

1. **阶段化训练**: 严格按照三阶段策略进行
2. **定期保存**: 每100 episode保存检查点
3. **监控指标**: 实时关注碰撞率、决策损失、CQR覆盖率
4. **提前停止**: 如果连续200 episode无改善,考虑调整超参数
5. **模型验证**: 每阶段结束后进行独立验证集测试

### 关键成功因素

1. ✅ 预训练模型质量: 确保SCM和CQR模型在标准数据集上表现良好
2. ✅ 三阶段策略执行: 不要跳过任何阶段
3. ✅ CQR定期校准: 保持预测准确性
4. ✅ 安全约束强化: 阶段3增加安全权重
5. ✅ 超参数调优: 根据实际环境调整学习率和奖励权重

### 下一步工作

训练完成后:

1. **模型评估**: 在测试集上评估性能
2. **对抗测试**: 测试极端交通场景
3. **实车部署**: 准备模型压缩和量化
4. **持续学习**: 建立在线学习机制

---

## 附录

### A. 配置文件模板

#### custom_training_config.yaml

```yaml
# 自定义训练配置模板
experiment:
  name: "my_custom_exp"
  description: "Description of this experiment"
  seed: 42

marl:
  algo: "mappo"
  lr: 5.0e-4
  hidden_sizes: [256, 256]
  gamma: 0.99
  clip_param: 0.2

environment:
  num_agents: 5
  episode_length: 3600
  traffic_density: "medium"
  reward_weights:
    efficiency: 0.4
    comfort: 0.3
    safety: 0.3

prediction:
  use_prediction: True
  intention_model: "shallow_hierarchical"
  occupancy_model: "gru_cqr"

decision:
  use_decision: True
  freeze_init: True
  lane_change_duration: 3.0

training_stages:
  stage1:
    episodes: [0, 500]
    marl_lr: 5e-4
    decision_lr: 0.0
  stage2:
    episodes: [500, 1500]
    marl_lr: 3e-4
    decision_lr: 5e-5
    cqr_interval: 100
  stage3:
    episodes: [1500, 3000]
    marl_lr: 1e-4
    decision_lr: 1e-5
    cqr_interval: 50
```

### B. 训练脚本示例

#### run_full_training.sh

```bash
#!/bin/bash

# 完整三阶段训练脚本

EXP_NAME="full_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXP_NAME}"

mkdir -p ${LOG_DIR}

echo "开始训练: ${EXP_NAME}"
echo "日志目录: ${LOG_DIR}"

# 阶段1: 基础MARL训练
echo "=== 阶段1: 基础MARL训练 (0-500) ==="
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name ${EXP_NAME} \
  --test_desc "Stage1" \
  --num_episodes 500 \
  --lr 5e-4 \
  --freeze_decision_init True \
  2>&1 | tee ${LOG_DIR}/stage1.log

# 阶段2: 决策微调
echo "=== 阶段2: 决策微调 (500-1500) ==="
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name ${EXP_NAME} \
  --test_desc "Stage2" \
  --load_pretrained results/a_multi_lane/mappo/${EXP_NAME}/models/actor_episode_500.pt \
  --start_episode 500 \
  --num_episodes 1500 \
  --lr 3e-4 \
  --decision_lr 5e-5 \
  --cqr_calibration_interval 100 \
  2>&1 | tee ${LOG_DIR}/stage2.log

# 阶段3: 全局优化
echo "=== 阶段3: 全局优化 (1500-3000) ==="
python examples/train.py \
  --algo mappo \
  --env a_multi_lane \
  --exp_name ${EXP_NAME} \
  --test_desc "Stage3" \
  --load_pretrained results/a_multi_lane/mappo/${EXP_NAME}/models/actor_episode_1500.pt \
  --start_episode 1500 \
  --num_episodes 3000 \
  --lr 1e-4 \
  --decision_lr 1e-5 \
  --cqr_calibration_interval 50 \
  --theta_safe 0.25 \
  2>&1 | tee ${LOG_DIR}/stage3.log

echo "训练完成!"
echo "结果保存在: results/a_multi_lane/mappo/${EXP_NAME}/"
```

### C. 快速启动命令

```bash
# 快速开始(使用默认配置)
python examples/train.py --algo mappo --env a_multi_lane --exp_name quickstart

# 调试模式(小规模测试)
python examples/train.py --algo mappo --env a_multi_lane --exp_name debug \
  --num_episodes 50 --episode_length 600 --num_agents 3

# 高性能训练(使用GPU和并行)
python examples/train.py --algo mappo --env a_multi_lane --exp_name high_perf \
  --cuda True --n_rollout_threads 8 --num_mini_batch 4

# 完整训练(三阶段)
bash scripts/run_full_training.sh
```

---

**文档版本**: 1.0
**最后更新**: 2025-01-11
**维护者**: 交通流研究团队

如有问题,请查阅:
- 项目README: [README.md](README.md)
- 核查报告: [PROJECT_AUDIT_REPORT.md](PROJECT_AUDIT_REPORT.md)
- 实现完成报告: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
