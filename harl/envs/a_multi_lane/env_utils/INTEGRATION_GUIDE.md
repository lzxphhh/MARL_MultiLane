# 新环境模块集成指南

## 概述

本指南说明如何将新的模块化环境工具集成到MARL训练流程中。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      MARL Training Loop                          │
│                     (veh_env_wrapper.py)                         │
└──────────────┬──────────────────────────────────┬────────────────┘
               │                                  │
               ▼                                  ▼
    ┌──────────────────────┐         ┌──────────────────────┐
    │  Prediction Module   │         │  Decision Module     │
    │  (prediction_module) │         │  (decision_module)   │
    └──────────┬───────────┘         └──────────┬───────────┘
               │                                  │
               │                                  │
               │         ┌──────────────────────┐ │
               └────────>│  Feature Extractor   │<┘
                         │ (feature_extractor)  │
                         └──────────────────────┘
```

## 模块功能

### 1. Feature Extractor (特征提取器)
**文件**: `feature_extractor.py`

**功能**:
- 将环境状态转换为预测/决策模型所需的标准化特征
- 支持三种特征提取模式：
  1. 意图预测特征（以被预测车辆为ego）
  2. 占用预测特征（以被预测车辆为ego）
  3. 决策特征（以本车CAV为ego）

**核心方法**:
```python
# 意图预测特征
env_feat, ind_feat, veh_type = extractor.extract_intention_features(
    target_vehicle_id, state, lane_statistics
)

# 占用预测特征
occ_feat_dict = extractor.extract_occupancy_features(
    target_vehicle_id, state, lane_statistics, intention=0.8
)

# 决策特征
env_feat, ind_feat, veh_type = extractor.extract_decision_features(
    ego_vehicle_id, state, lane_statistics
)
```

### 2. Prediction Module (预测集成模块)
**文件**: `prediction_module.py`

**功能**:
- 集成SCM意图预测和CQR占用预测
- 端到端预测流程：意图 → 占用

**工作流程**:
```
周边车辆状态 → 意图预测 → 换道概率 → 占用预测 → 未来轨迹区间
```

**核心方法**:
```python
# 创建预测模块
pred_module = PredictionModuleFactory.create_module(
    intention_model_type="shallow_hierarchical",
    occupancy_model_type="CQR-GRU-uncertainty",
    device="cpu"
)

# 批量预测意图
intentions = pred_module.predict_intentions(vehicle_ids, state, lane_stats)

# 预测占用
lower, median, upper = pred_module.predict_occupancy(veh_id, state, lane_stats)

# 端到端预测
predictions = pred_module.predict_intentions_and_occupancy(vehicle_ids, state, lane_stats)
```

### 3. Decision Module (决策集成模块)
**文件**: `decision_module.py`

**功能**:
- 集成SCM横向决策模型
- 支持MARL训练中的渐进式微调

**微调策略**:
- **阶段1 (0-1000 eps)**: 冻结基础SCM，仅训练决策阈值
- **阶段2 (1000-2000 eps)**: 解冻个体层，微调因果路径
- **阶段3 (2000+ eps)**: 全局微调

**核心方法**:
```python
# 创建决策模块
dec_module = DecisionModuleFactory.create_module(
    model_type="shallow_hierarchical",
    freeze_base_model=True,
    enable_training=True,
    device="cpu"
)

# 配置微调
dec_module.setup_fine_tuning(
    learning_rate=1e-4,
    stage_thresholds=(1000, 2000)
)

# 生成决策
decision, prob = dec_module.make_decision(ego_id, state, lane_stats, return_prob=True)

# MARL更新
loss = dec_module.update_decision_model(ego_ids, state, lane_stats, rewards)
dec_module.on_episode_end(save_dir="./checkpoints")
```

## 集成到veh_env_wrapper

### 步骤1: 初始化模块

```python
from harl.envs.a_multi_lane.env_utils.prediction_module import PredictionModuleFactory
from harl.envs.a_multi_lane.env_utils.decision_module import DecisionModuleFactory
from harl.envs.a_multi_lane.env_utils.env_config import get_default_config

class VehEnvWrapper:
    def __init__(self, args):
        # 加载配置
        self.config = get_default_config()

        # 创建预测模块
        self.pred_module = PredictionModuleFactory.create_module(
            intention_model_type=self.config.intention_model_type,
            occupancy_model_type=self.config.occupancy_model_type,
            use_conformal=self.config.use_conformal,
            device=args.device,
            use_cache=True  # 使用单例缓存
        )

        # 创建决策模块
        self.dec_module = DecisionModuleFactory.create_module(
            model_type=self.config.decision_model_type,
            freeze_base_model=self.config.freeze_base_model,
            enable_training=self.config.enable_decision_training,
            device=args.device,
            use_cache=True
        )

        # 配置微调策略
        if self.config.enable_decision_training:
            self.dec_module.setup_fine_tuning(
                learning_rate=self.config.fine_tune_lr,
                stage_thresholds=self.config.fine_tune_stage_thresholds
            )
```

### 步骤2: 在step()中使用

```python
def step(self, action):
    """环境步进"""

    # 1. 获取当前状态
    state, lane_statistics = self._get_current_state()

    # 2. 预测周边车辆意图和占用（用于感知和碰撞预测）
    hdv_ids = [veh_id for veh_id in state.keys() if 'HDV' in veh_id]
    hdv_predictions = self.pred_module.predict_intentions_and_occupancy(
        hdv_ids, state, lane_statistics
    )

    # 3. 基于预测生成CAV横向决策
    cav_ids = [veh_id for veh_id in state.keys() if 'CAV' in veh_id]
    cav_decisions = self.dec_module.make_batch_decisions(
        cav_ids, state, lane_statistics, return_prob=True
    )

    # 4. 整合RL动作和决策模型输出
    final_actions = self._merge_actions(action, cav_decisions)

    # 5. 执行动作（包含安全约束）
    next_state, rewards, done, info = self._execute_actions(final_actions)

    # 6. 更新决策模型（MARL微调）
    if self.config.enable_decision_training:
        loss = self.dec_module.update_decision_model(
            cav_ids, state, lane_statistics, rewards
        )
        info['decision_loss'] = loss

    # 7. 将预测结果添加到info（用于分析）
    info['hdv_predictions'] = hdv_predictions
    info['cav_decisions'] = cav_decisions

    return next_state, rewards, done, info
```

### 步骤3: 在reset()中重置

```python
def reset(self):
    """重置环境"""

    # 重置决策统计
    self.dec_module.on_episode_end(save_dir="./checkpoints")

    # 重置环境...
    state = super().reset()

    return state
```

## 关键集成点

### 1. 状态格式要求

环境状态必须包含以下字段：

```python
state = {
    'vehicle_id': {
        'lane_index': int,           # 车道索引 (0, 1, 2)
        'speed': float,              # 速度 (m/s)
        'acceleration': float,       # 加速度 (m/s^2)
        'position_x': float,         # 纵向位置 (m)
        'position_y': float,         # 横向位置 (m)
        'surrounding_vehicles': {
            'front_1': {             # 当前车道前车
                'id': str,
                'long_dist': float,  # 纵向距离
                'long_rel_v': float, # 相对速度
                'speed': float,
                ...
            },
            'adj_front': {...},      # 目标车道前车
            'adj_rear': {...},       # 目标车道后车
            ...
        }
    }
}

lane_statistics = {
    lane_id: {
        'mean_speed': float,  # 车道平均速度
        'density': float,     # 车道密度
        ...
    }
}
```

### 2. 动作空间整合

**方案A: 决策作为横向动作**
```python
def _merge_actions(self, rl_action, cav_decisions):
    """将RL纵向动作和决策横向动作整合"""
    merged_actions = {}
    for cav_id in cav_decisions.keys():
        lateral_decision, prob = cav_decisions[cav_id]
        longitudinal_action = rl_action[cav_id][1]  # 纵向动作

        # 将决策转换为横向动作
        lateral_action = self._decision_to_action(lateral_decision)

        merged_actions[cav_id] = [lateral_action, longitudinal_action]

    return merged_actions

def _decision_to_action(self, decision):
    """将二值决策转换为连续动作"""
    if decision == 1:
        return 1.0  # 换道
    else:
        return 0.0  # 保持
```

**方案B: 决策作为辅助信号**
```python
def _merge_actions(self, rl_action, cav_decisions):
    """将决策作为奖励调制因子"""
    # RL保留完全控制，决策用于奖励塑形
    for cav_id, (decision, prob) in cav_decisions.items():
        if decision != self._extract_decision_from_rl(rl_action[cav_id]):
            # RL动作与决策不一致，惩罚
            self.decision_consistency_penalty[cav_id] = -0.1

    return rl_action
```

### 3. 奖励设计

```python
def _compute_reward(self, state, action, cav_decisions):
    """计算奖励（结合决策一致性）"""

    # 基础奖励（安全、效率、舒适度）
    base_reward = self._compute_base_reward(state, action)

    # 决策一致性奖励
    consistency_reward = 0.0
    for cav_id, (decision, prob) in cav_decisions.items():
        rl_decision = self._extract_decision_from_rl(action[cav_id])
        if decision == rl_decision:
            # 奖励与人类驾驶一致的决策
            consistency_reward += 0.1 * prob

    # 总奖励
    total_reward = base_reward + consistency_reward

    return total_reward
```

## 微调训练流程

```python
# 训练循环
for episode in range(total_episodes):
    state = env.reset()
    episode_rewards = []

    for step in range(max_steps):
        # 1. RL agent生成动作
        rl_action = agent.get_action(state)

        # 2. 环境内部会调用decision_module生成横向决策
        next_state, reward, done, info = env.step(rl_action)

        # 3. 决策模型自动微调（在step内部完成）
        # dec_module.update_decision_model() 已在step中调用

        # 4. RL agent更新
        agent.update(state, rl_action, reward, next_state)

        episode_rewards.append(reward)
        state = next_state

        if done:
            break

    # 5. Episode结束时自动切换微调阶段（在reset中调用）
    # dec_module.on_episode_end() 已在reset中调用

    # 6. 记录训练统计
    if episode % 100 == 0:
        stats = env.dec_module.get_training_stats()
        print(f"Episode {episode}:")
        print(f"  - Fine-tune stage: {stats['fine_tune_stage']}")
        print(f"  - Decision loss: {stats['avg_loss']:.6f}")
        print(f"  - Lane change rate: {stats['decision_stats']['lane_change_rate']:.2%}")
```

## 配置参数

在`env_config.py`中调整：

```python
config = EnvironmentConfig(
    # 预测模型配置
    intention_model_type="shallow_hierarchical",  # 或 "medium_hierarchical"
    occupancy_model_type="CQR-GRU-uncertainty",   # 或其他4个模型
    use_conformal=True,                           # 是否使用CQR

    # 决策模型配置
    decision_model_type="shallow_hierarchical",
    freeze_base_model=True,                       # 初期冻结
    enable_decision_training=True,                # 启用微调

    # 微调策略
    fine_tune_lr=1e-4,                           # 学习率
    fine_tune_stage_thresholds=(1000, 2000),     # 阶段切换
)
```

## 性能优化建议

1. **使用单例缓存**: `use_cache=True` 避免重复加载模型
2. **批量处理**: 使用`batch_`方法而非循环调用
3. **GPU加速**: 设置`device="cuda"`（如果可用）
4. **异步预测**: 可将预测放在单独线程（如果性能瓶颈）

## 调试和监控

```python
# 获取模型信息
pred_info = env.pred_module.get_model_info()
dec_info = env.dec_module.get_model_info()

# 获取训练统计
training_stats = env.dec_module.get_training_stats()

# 打印参数状态
env.dec_module.decision_model.print_parameter_status()
```

## 常见问题

### Q1: 如何平衡RL探索和决策模型的利用？

A: 使用epsilon-greedy策略：
```python
if np.random.rand() < epsilon:
    action = rl_agent.get_action(state)  # RL探索
else:
    decision, _ = dec_module.make_decision(...)  # 决策模型
    action = decision_to_action(decision)
```

### Q2: 何时解冻决策模型？

A: 建议渐进式解冻：
- 前30%: 冻结（学习基本策略）
- 30%-70%: 解冻个体层（微调因果路径）
- 70%-100%: 全局微调（精细调整）

### Q3: 如何避免决策模型过拟合？

A:
- 使用小学习率 (1e-4 ~ 1e-5)
- 定期保存checkpoint
- 监控决策统计（换道率不应剧烈变化）
- 应用因果约束（自动在update中调用）

## 下一步

1. 实现完整的`veh_env_wrapper.py`
2. 测试MARL训练循环
3. 评估预测精度和决策一致性
4. 优化微调策略和超参数

## 联系方式

如有问题，请联系交通流研究团队。
