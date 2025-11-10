# 多车道环境工具模块

重构后的模块化环境工具，集成了预测和决策功能。

## 目录结构

```
env_utils/
├── feature_extractor.py      # 特征提取模块
├── prediction_module.py       # 预测集成模块（意图+占用）
├── decision_module.py         # 决策集成模块（支持MARL微调）
├── env_config.py             # 环境配置
├── README.md                 # 本文件
└── old_version/              # 旧版本代码（已移至此处）
```

## 模块说明

### 1. 特征提取模块 (`feature_extractor.py`)

**功能**: 将环境状态转换为预测/决策模型所需的特征

**核心类**: `FeatureExtractor`

**主要方法**:
- `extract_intention_features()`: 提取意图预测特征（以被预测车辆为ego）
- `extract_occupancy_features()`: 提取占用预测特征（以被预测车辆为ego）
- `extract_decision_features()`: 提取决策特征（以本车CAV为ego）
- `batch_extract_intention_features()`: 批量提取意图特征

**使用示例**:
```python
from env_utils.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# 为意图预测提取特征
env_feat, ind_feat, veh_type = extractor.extract_intention_features(
    target_vehicle_id='HDV_0',
    state=state,
    lane_statistics=lane_stats
)

# 为占用预测提取特征
occ_feat = extractor.extract_occupancy_features(
    target_vehicle_id='HDV_0',
    state=state,
    lane_statistics=lane_stats,
    intention=0.8  # 换道概率
)

# 为决策提取特征
env_feat, ind_feat, veh_type = extractor.extract_decision_features(
    ego_vehicle_id='CAV_0',
    state=state,
    lane_statistics=lane_stats
)
```

### 2. 预测集成模块 (`prediction_module.py`)

**功能**: 集成SCM意图预测和CQR占用预测

**核心类**: `PredictionModule`, `PredictionModuleFactory`

**工作流程**:
1. 调用SCM模型预测周边车辆换道意图
2. 将意图作为输入特征的一部分
3. 调用CQR模型预测未来轨迹占用

**使用示例**:
```python
from env_utils.prediction_module import PredictionModuleFactory

# 创建预测模块
pred_module = PredictionModuleFactory.create_module(
    intention_model_type="shallow_hierarchical",
    occupancy_model_type="CQR-GRU-uncertainty",
    use_conformal=True,
    device="cpu"
)

# 批量预测意图
vehicle_ids = ['HDV_0', 'HDV_1', 'CAV_0']
intentions = pred_module.predict_intentions(vehicle_ids, state, lane_stats)
# 结果: {'HDV_0': 0.75, 'HDV_1': 0.12, 'CAV_0': 0.88}

# 预测占用（基于意图）
lower, median, upper = pred_module.predict_occupancy(
    'HDV_0', state, lane_stats, intention=0.75
)
# 结果: 下界/中位/上界轨迹 [30,] (3秒预测)

# 端到端预测（意图+占用）
all_predictions = pred_module.predict_intentions_and_occupancy(
    vehicle_ids, state, lane_stats
)
# 结果: {vehicle_id: {'intention': float, 'occupancy': {...}}}
```

### 3. 决策集成模块 (`decision_module.py`)

**功能**: 集成SCM决策模型，支持MARL训练中的在线微调

**核心类**: `DecisionModule`, `DecisionModuleFactory`

**微调策略**: 渐进式三阶段微调
- **阶段1 (0-1000 episodes)**: 冻结基础SCM模型，仅训练决策阈值
- **阶段2 (1000-2000 episodes)**: 解冻个体层，微调因果路径
- **阶段3 (2000+ episodes)**: 全局微调所有层

**使用示例**:
```python
from env_utils.decision_module import DecisionModuleFactory

# 创建决策模块（训练模式）
dec_module = DecisionModuleFactory.create_module(
    model_type="shallow_hierarchical",
    freeze_base_model=True,  # 初期冻结
    enable_training=True,    # 启用MARL微调
    device="cpu"
)

# 配置微调策略
dec_module.setup_fine_tuning(
    learning_rate=1e-4,
    stage_thresholds=(1000, 2000)  # 阶段切换阈值
)

# 在MARL环境中使用
# 1. 生成决策
decision, prob = dec_module.make_decision(
    ego_vehicle_id='CAV_0',
    state=state,
    lane_statistics=lane_stats,
    return_prob=True
)
# 结果: decision=1 (换道), prob=0.85

# 2. 执行动作，获得奖励
rewards = {'CAV_0': 0.8}

# 3. 更新决策模型
loss = dec_module.update_decision_model(
    ego_vehicle_ids=['CAV_0'],
    state=state,
    lane_statistics=lane_stats,
    rewards=rewards
)

# 4. Episode结束时
dec_module.on_episode_end(save_dir="./checkpoints")
```

### 4. 环境配置 (`env_config.py`)

**功能**: 集中管理环境参数

**核心类**: `EnvironmentConfig`

**配置内容**:
- 基础配置（CAV数量、时间步长等）
- 车道配置（车道数、宽度等）
- 车辆参数（速度、加速度范围）
- 安全参数（TTC阈值、间距阈值）
- 观测/动作空间配置
- 预测/决策模型配置
- 微调策略配置

**使用示例**:
```python
from env_utils.env_config import EnvironmentConfig, get_default_config

# 使用默认配置
config = get_default_config()

# 自定义配置
config = EnvironmentConfig(
    num_CAVs=10,
    CAV_penetration=0.7,
    intention_model_type="medium_hierarchical",
    fine_tune_lr=5e-5
)
```

## 集成到环境

在新的`veh_env_wrapper.py`中使用这些模块：

```python
from env_utils.prediction_module import PredictionModuleFactory
from env_utils.decision_module import DecisionModuleFactory
from env_utils.env_config import get_default_config

class VehEnvWrapper:
    def __init__(self, args):
        # 加载配置
        self.config = get_default_config()

        # 创建预测模块
        self.pred_module = PredictionModuleFactory.create_module(
            intention_model_type=self.config.intention_model_type,
            occupancy_model_type=self.config.occupancy_model_type,
            device=args.device
        )

        # 创建决策模块
        self.dec_module = DecisionModuleFactory.create_module(
            model_type=self.config.decision_model_type,
            freeze_base_model=self.config.freeze_base_model,
            enable_training=self.config.enable_decision_training,
            device=args.device
        )

        # 配置微调
        self.dec_module.setup_fine_tuning(
            learning_rate=self.config.fine_tune_lr,
            stage_thresholds=self.config.fine_tune_stage_thresholds
        )

    def step(self, action):
        # 1. 预测周边车辆意图和占用
        hdv_ids = [veh_id for veh_id in state.keys() if 'HDV' in veh_id]
        predictions = self.pred_module.predict_intentions_and_occupancy(
            hdv_ids, state, lane_stats
        )

        # 2. 基于预测生成CAV决策
        cav_ids = [veh_id for veh_id in state.keys() if 'CAV' in veh_id]
        decisions = self.dec_module.make_batch_decisions(
            cav_ids, state, lane_stats
        )

        # 3. 执行动作...

        # 4. 更新决策模型
        self.dec_module.update_decision_model(
            cav_ids, state, lane_stats, rewards
        )

        return obs, reward, done, info

    def reset(self):
        self.dec_module.on_episode_end()
        ...
```

## 测试

每个模块都包含独立的测试代码，可直接运行：

```bash
# 测试特征提取
python feature_extractor.py

# 测试预测模块
python prediction_module.py

# 测试决策模块
python decision_module.py
```

## 与旧版本的区别

### 旧版本（已移至old_version/）
- 单体式设计，代码耦合度高
- 预测和决策逻辑分散
- 难以维护和扩展

### 新版本（当前）
- **模块化设计**: 特征提取、预测、决策分离
- **易于集成**: 工厂模式，统一接口
- **支持微调**: 内置渐进式微调策略
- **配置集中**: 所有参数集中管理
- **代码复用**: 避免重复代码

## 注意事项

1. **特征一致性**: 确保环境状态包含所有必需字段（surrounding_vehicles等）
2. **设备一致性**: 预测/决策模块应使用相同的device
3. **梯度管理**: 微调时注意梯度累积和清零
4. **阶段切换**: 微调阶段自动切换，无需手动干预
5. **模型保存**: 定期保存微调后的模型checkpoint

## 性能优化

- 使用`use_cache=True`启用单例模式，避免重复加载模型
- 批量处理：使用`batch_extract_*`和`make_batch_decisions`
- GPU加速：设置`device="cuda"`（如果可用）

## 联系方式

如有问题，请联系交通流研究团队。
