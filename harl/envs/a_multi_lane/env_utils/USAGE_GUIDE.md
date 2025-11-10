# 预测和决策模块使用指南

## 快速集成到现有环境

现有的环境代码已经移至`old_version/`，您可以通过以下方式快速集成新的预测和决策功能。

## 方案 1: 使用增强包装器（推荐）

在现有的`veh_env_wrapper.py`中最小化修改，使用`prediction_decision_wrapper.py`增强功能。

### 步骤1: 初始化模块

在`VehEnvWrapper.__init__()`中添加：

```python
from harl.envs.a_multi_lane.env_utils.prediction_module import PredictionModuleFactory
from harl.envs.a_multi_lane.env_utils.decision_module import DecisionModuleFactory
from harl.envs.a_multi_lane.env_utils.env_config import get_default_config

def __init__(self, args):
    # ... 原有代码 ...

    # 加载配置
    self.config = get_default_config()

    # 创建预测模块
    self.pred_module = PredictionModuleFactory.create_module(
        intention_model_type=self.config.intention_model_type,
        occupancy_model_type=self.config.occupancy_model_type,
        device=getattr(args, 'device', 'cpu'),
        use_cache=True
    )

    # 创建决策模块
    self.dec_module = DecisionModuleFactory.create_module(
        model_type=self.config.decision_model_type,
        freeze_base_model=self.config.freeze_base_model,
        enable_training=self.config.enable_decision_training,
        device=getattr(args, 'device', 'cpu'),
        use_cache=True
    )

    # 配置微调
    self.dec_module.setup_fine_tuning(
        learning_rate=self.config.fine_tune_lr,
        stage_thresholds=self.config.fine_tune_stage_thresholds
    )
```

### 步骤2: 在state_wrapper中增强统计

在`state_wrapper()`方法中，在调用原有`analyze_traffic()`后增加预测和决策：

```python
from harl.envs.a_multi_lane.env_utils.prediction_decision_wrapper import (
    enhance_traffic_analysis_with_predictions,
    integrate_predictions_into_observations,
    create_prediction_decision_info
)

def state_wrapper(self, state, sim_time):
    # 1. 调用原有的analyze_traffic
    cav_statistics, hdv_statistics, reward_statistics, lane_statistics, flow_statistics, evaluation, self.TTC_assessment = analyze_traffic(
        state=state,
        lane_ids=self.calc_features_lane_ids,
        max_veh_num=self.lane_max_num_vehs,
        parameters=self.parameters,
        vehicles_hist=self.vehicles_hist,
        hist_length=self.hist_length,
        TTC_assessment=self.TTC_assessment
    )

    # 2. 增强：添加预测和决策信息
    cav_statistics, hdv_statistics = enhance_traffic_analysis_with_predictions(
        cav_statistics=cav_statistics,
        hdv_statistics=hdv_statistics,
        state=state['vehicle'],
        lane_statistics=lane_statistics,
        pred_module=self.pred_module,
        dec_module=self.dec_module
    )

    # 3. 创建预测决策信息（用于日志和分析）
    pred_dec_info = create_prediction_decision_info(cav_statistics, hdv_statistics)

    # 4. 继续原有的特征选择逻辑...
    if self.strategy == 'base':
        feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = feature_selection_simple_version(...)
    elif self.strategy == 'improve':
        feature_vectors_current, feature_vectors_current_flatten, feature_vectors, feature_vectors_flatten = feature_selection_improved_version(...)

    return (feature_vectors_current, feature_vectors_current_flatten,
            feature_vectors, feature_vectors_flatten,
            cav_statistics, hdv_statistics, reward_statistics,
            lane_statistics, flow_statistics, evaluation, classified_scene_mark,
            pred_dec_info)  # 新增返回值
```

### 步骤3: 在step中使用决策并更新模型

在`step()`方法中：

```python
from harl.envs.a_multi_lane.env_utils.prediction_decision_wrapper import (
    update_rewards_with_decision_consistency
)

def step(self, action):
    # 1. 更新动作（可选：使用决策信息）
    action = self.__update_actions(raw_action=action)

    # 2. 执行SUMO step
    init_state, rewards, truncated, _dones, infos = super().step(action)

    # 3. 状态处理（包含预测和决策）
    (feature_vectors_current, ..., pred_dec_info) = self.state_wrapper(state=init_state, sim_time=now_time)

    # 4. 奖励计算
    reward_dict, evaluation_dict = hierarchical_reward_computation(...)

    # 5. 增强：基于决策一致性调整奖励
    reward_dict = update_rewards_with_decision_consistency(
        reward_dict=reward_dict,
        cav_statistics=cav_statistics,
        rl_actions=self.actual_actions,
        consistency_weight=0.1
    )

    # 6. 更新决策模型（MARL微调）
    if self.config.enable_decision_training:
        cav_ids = list(cav_statistics.keys())
        loss = self.dec_module.update_decision_model(
            cav_ids, init_state['vehicle'], lane_statistics, reward_dict
        )
        infos['decision_loss'] = loss

    # 7. 添加预测决策信息到infos
    infos['pred_dec_info'] = pred_dec_info

    return ...
```

### 步骤4: 在reset中重置

```python
def reset(self, seed=1):
    # ... 原有代码 ...

    # 重置决策模块
    self.dec_module.on_episode_end(save_dir="./checkpoints")

    return ...
```

## 方案 2: 直接在动作生成中使用决策

如果希望决策模块直接生成横向动作（而非仅作为辅助），可以在`__update_actions()`中：

```python
def __update_actions(self, raw_action):
    # ... 原有代码 ...

    for _veh_id in raw_action:
        if _veh_id in self.actions:
            # 获取决策信息
            if _veh_id in cav_statistics:
                lateral_decision = cav_statistics[_veh_id].get('lateral_decision', 0)
                decision_prob = cav_statistics[_veh_id].get('decision_probability', 0.0)

                # 方案A: 完全使用决策（高置信度时）
                if decision_prob > 0.8:
                    lateral_action = 1.0 if lateral_decision == 1 else 0.0
                else:
                    # 使用RL动作
                    lateral_action = raw_action[_veh_id][0]

                # 方案B: 混合决策和RL
                # lateral_action = decision_prob * (1.0 if lateral_decision == 1 else 0.0) + \
                #                 (1 - decision_prob) * raw_action[_veh_id][0]
            else:
                lateral_action = raw_action[_veh_id][0]

            # TTC安全约束（保留）
            if self.TTC_assessment[_veh_id]['right'] < 1.0:
                lateral_action = np.clip(lateral_action, -0.5, 1.5)
            if self.TTC_assessment[_veh_id]['left'] < 1.0:
                lateral_action = np.clip(lateral_action, -1.5, 0.5)

            actual_lanechange = map_action_to_lane(lateral_action)

            # 纵向控制（保持IDM + RL）
            # ...

            self.actions[_veh_id] = (actual_lanechange, -1, speed_command)

    return self.actions
```

## 方案 3: 在观测中添加预测信息

如果希望RL agent能够感知周边车辆的预测意图，可以在特征选择中添加：

```python
# 在 state_selection.py 中的特征选择函数中

def feature_selection_improved_version(...):
    # ... 原有代码 ...

    for ego_id, ego_info in cav_statistics.items():
        # ... 原有特征提取 ...

        # 新增：周边车辆预测意图特征
        surround_intentions = []
        surroundings = ego_info.get('surroundings', {})

        for sur_key in ['front_1', 'front_2', 'rear_1', 'rear_2', 'adj_front', 'adj_rear']:
            sur_content = surroundings.get(sur_key, {})
            if isinstance(sur_content, dict):
                sur_veh_id = sur_content.get('veh_id', None)
                if sur_veh_id and sur_veh_id in hdv_statistics:
                    # 获取预测意图
                    intention = hdv_statistics[sur_veh_id].get('predicted_intention', 0.0)
                    surround_intentions.append(intention)
                else:
                    surround_intentions.append(0.0)
            else:
                surround_intentions.append(0.0)

        # 将预测意图添加到特征向量
        feature_vectors[ego_id].extend(surround_intentions)  # 6维

    return ...
```

## 配置参数调整

在`env_config.py`或环境初始化时调整：

```python
# 预测模型选择
config.intention_model_type = "shallow_hierarchical"  # 快速
# config.intention_model_type = "medium_hierarchical"  # 精确

# 占用预测模型
config.occupancy_model_type = "CQR-GRU-uncertainty"  # 基础CQR
# config.occupancy_model_type = "CQR-GRU-uncertainty_pcgrad"  # PCGrad优化

# 决策模型
config.decision_model_type = "shallow_hierarchical"
config.freeze_base_model = True  # 初期冻结
config.enable_decision_training = True  # 启用微调

# 微调策略
config.fine_tune_lr = 1e-4
config.fine_tune_stage_thresholds = (1000, 2000)  # 阶段切换
```

## 监控和调试

### 查看训练统计

```python
# 在训练循环中
stats = env.dec_module.get_training_stats()
print(f"Episode {episode}:")
print(f"  Fine-tune stage: {stats['fine_tune_stage']}")
print(f"  Decision loss: {stats['avg_loss']:.6f}")
print(f"  Lane change rate: {stats['decision_stats']['lane_change_rate']:.2%}")
```

### 查看预测决策信息

```python
# 在step后
if 'pred_dec_info' in info:
    pred_info = info['pred_dec_info']
    print(f"CAV decisions: {len(pred_info['cav_decisions'])}")
    print(f"HDV predictions: {len(pred_info['hdv_predictions'])}")
    print(f"Avg decision prob: {pred_info['statistics']['avg_decision_prob']:.4f}")
```

### 保存模型

```python
# 定期保存
if episode % 100 == 0:
    env.dec_module.decision_maker.save_model(f"checkpoints/decision_ep{episode}.pth")
```

## 性能考虑

1. **使用单例缓存**: `use_cache=True` 避免重复加载模型
2. **批量处理**: 预测和决策自动批量处理多车辆
3. **异步预测**（可选）: 如果预测成为瓶颈，可考虑异步
4. **GPU加速**: 设置`device="cuda"`（如果可用）

## 注意事项

1. **状态字段**: 确保`state['vehicle']`包含所有必需字段（尤其是`surrounding_vehicles`）
2. **特征一致性**: 预测/决策特征提取与训练时保持一致
3. **梯度管理**: 微调时注意梯度累积
4. **阶段切换**: 微调阶段自动切换，无需手动干预

## 完整示例

参见`INTEGRATION_GUIDE.md`获取完整的集成示例代码。

## 问题排查

### Q: 预测/决策失败
A: 检查状态字段是否完整，尤其是`surrounding_vehicles`

### Q: 微调不稳定
A: 降低学习率至1e-5，或增加冻结阶段的episode数

### Q: 性能下降
A: 检查是否使用了`use_cache=True`，考虑使用GPU

## 联系方式

如有问题，请查阅`README.md`和`INTEGRATION_GUIDE.md`，或联系团队。
