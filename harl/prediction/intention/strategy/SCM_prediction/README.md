# SCM意图预测模块

基于结构因果模型(Structural Causal Model, SCM)的车辆换道意图预测模块，用于MARL多车道环境中的意图预测。

## 目录结构

```
SCM_prediction/
├── __init__.py              # 模块初始化文件
├── scm_model_v2.py         # SCM模型定义
├── execute_prediction.py   # 预测执行器
└── README.md               # 本文件
```

## 模型说明

### 可用模型

当前支持两个预训练模型（位于 `../../models/SCM_models/`）：

1. **shallow_hierarchical**: Shallow特征提取 + Hierarchical特征融合 + Pred only loss
   - 路径架构: SHALLOW (R^4 → R^8 → R)
   - 融合策略: HIERARCHICAL (安全门控 + 动机组合)
   - 推荐用于实时性要求较高的场景

2. **medium_hierarchical**: Medium特征提取 + Hierarchical特征融合 + Pred only loss
   - 路径架构: MEDIUM (R^4 → R^12 → R^4 → R)
   - 融合策略: HIERARCHICAL
   - 推荐用于精度要求较高的场景

### 模型输入

#### 环境特征 (4维)
- `v_avg_norm`: 平均速度（归一化）
- `kappa_avg_norm`: 平均密度（归一化）
- `delta_v_norm`: 车道速度差（归一化）
- `delta_kappa_norm`: 车道密度差（归一化）

#### 个体特征 (10维)
- `v_ego_norm`: 自车速度（归一化）
- `v_rel_norm`: 与当前车道前车相对速度
- `d_headway_norm`: 车头间距
- `T_headway_norm`: 时间间距
- `delta_v_adj_front_norm`: 与目标车道前车速度差
- `d_adj_front_norm`: 与目标车道前车距离
- `T_adj_front_norm`: 与目标车道前车时间间隙
- `delta_v_adj_rear_norm`: 与目标车道后车速度差
- `d_adj_rear_norm`: 与目标车道后车距离
- `T_adj_rear_norm`: 与目标车道后车时间间隙

#### 车辆类型 (1维, 可选)
- `0`: HDV (Human-Driven Vehicle)
- `1`: CAV (Connected Autonomous Vehicle)

### 模型输出

- **换道意图概率**: 0-1之间的浮点数，表示车辆换道的概率
- **中间结果** (可选): 包含各个因果路径的输出和门控值等

## 使用方法

### 基本使用

```python
from harl.prediction.intention.strategy.SCM_prediction import SCMPredictorFactory
import numpy as np

# 1. 创建预测器
predictor = SCMPredictorFactory.create_predictor(
    model_type="shallow_hierarchical",  # 或 "medium_hierarchical"
    device="cpu"  # 或 "cuda"
)

# 2. 准备输入数据（需要预先归一化）
env_features = np.array([[0.5, -0.2, 0.1, -0.05]])  # [1, 4]
ind_features = np.array([[0.6, 0.1, 50.0, 2.5,
                         -0.05, 60.0, 3.0,
                         0.15, 40.0, 2.0]])  # [1, 10]
vehicle_type = np.array([[0]])  # HDV

# 3. 执行预测
predictions = predictor.predict(env_features, ind_features, vehicle_type)
print(f"换道意图概率: {predictions[0, 0]:.4f}")
```

### 单样本预测

```python
# 单样本预测的便捷接口
probability = predictor.predict_single(
    env_feature=[0.5, -0.2, 0.1, -0.05],
    ind_feature=[0.6, 0.1, 50.0, 2.5, -0.05, 60.0, 3.0, 0.15, 40.0, 2.0],
    vehicle_type=0  # HDV
)
print(f"换道意图概率: {probability:.4f}")
```

### 批量预测

```python
# 批量预测多个样本
env_features = np.random.randn(10, 4).astype(np.float32)
ind_features = np.random.randn(10, 10).astype(np.float32)
vehicle_types = np.random.randint(0, 2, (10, 1)).astype(np.float32)

predictions = predictor.predict(env_features, ind_features, vehicle_types)
# predictions shape: [10, 1]
```

### 获取中间结果

```python
# 获取详细的中间结果
predictions, intermediates = predictor.predict(
    env_features, ind_features, vehicle_types,
    return_intermediates=True
)

# intermediates包含:
# - 'psi_env': 环境层输出
# - 'psi_ind': 个体层输出
# - 'path_N': 必要性路径输出
# - 'path_F': 可行性路径输出
# - 'path_S': 安全性路径输出
# - 'gate': 门控值
# - 'P_env': 环境层概率
# 等等...
```

### 从观测字典预测

```python
# 适用于MARL环境的观测格式
observations = [
    {
        'env_features': [0.5, -0.2, 0.1, -0.05],
        'ind_features': [0.6, 0.1, 50.0, 2.5, -0.05, 60.0, 3.0, 0.15, 40.0, 2.0],
        'vehicle_type': 0
    },
    # ... 更多观测
]

predictions = predictor.predict_batch_from_dict(observations)
```

### 查看可用模型

```python
# 获取所有可用的预训练模型
available_models = SCMPredictorFactory.get_available_models()
print("可用模型:", available_models)
# 输出: ['shallow_hierarchical', 'medium_hierarchical']
```

### 查看模型信息

```python
# 获取当前模型的详细信息
info = predictor.get_model_info()
print(info)
# 输出:
# {
#     'model_type': 'shallow_hierarchical',
#     'path_architecture': 'shallow',
#     'fusion_strategy': 'hierarchical',
#     'device': 'cpu',
#     'causal_parameters': {...}
# }
```

## 在MARL环境中使用

### 集成到多车道环境

```python
# 在multilane_env.py中使用
from harl.prediction.intention.strategy.SCM_prediction import SCMPredictorFactory

class MultiLaneEnv:
    def __init__(self, ...):
        # 初始化预测器
        self.intention_predictor = SCMPredictorFactory.create_predictor(
            model_type="shallow_hierarchical",
            device="cpu",
            use_cache=True  # 使用单例缓存
        )

    def predict_surrounding_intentions(self, vehicle_ids):
        """预测周边车辆的换道意图"""
        observations = []

        for veh_id in vehicle_ids:
            # 提取环境特征和个体特征
            env_feat = self._extract_env_features(veh_id)
            ind_feat = self._extract_ind_features(veh_id)
            veh_type = self._get_vehicle_type(veh_id)

            observations.append({
                'env_features': env_feat,
                'ind_features': ind_feat,
                'vehicle_type': veh_type
            })

        # 批量预测
        predictions = self.intention_predictor.predict_batch_from_dict(observations)

        # 返回字典映射
        return {veh_id: pred for veh_id, pred in zip(vehicle_ids, predictions)}
```

### 特征提取示例

```python
def _extract_env_features(self, vehicle_id):
    """提取环境特征"""
    # 计算平均速度和密度
    v_avg = self._get_average_speed()
    kappa_avg = self._get_average_density()

    # 计算车道差异
    current_lane = self._get_vehicle_lane(vehicle_id)
    target_lane = current_lane + 1  # 假设向右换道

    delta_v = self._get_lane_speed(target_lane) - self._get_lane_speed(current_lane)
    delta_kappa = self._get_lane_density(target_lane) - self._get_lane_density(current_lane)

    # 归一化
    env_features = np.array([
        self._normalize(v_avg, 'v_avg'),
        self._normalize(kappa_avg, 'kappa_avg'),
        self._normalize(delta_v, 'delta_v'),
        self._normalize(delta_kappa, 'delta_kappa')
    ])

    return env_features

def _extract_ind_features(self, vehicle_id):
    """提取个体特征"""
    # 获取车辆状态
    ego_speed = self._get_vehicle_speed(vehicle_id)

    # 当前车道前车信息
    front_veh = self._get_front_vehicle(vehicle_id)
    v_rel = front_veh['speed'] - ego_speed if front_veh else 0
    d_headway = front_veh['distance'] if front_veh else 100.0
    T_headway = d_headway / ego_speed if ego_speed > 0 else 10.0

    # 目标车道前车信息
    target_front = self._get_target_front_vehicle(vehicle_id)
    delta_v_adj_front = target_front['speed'] - ego_speed if target_front else 0
    d_adj_front = target_front['distance'] if target_front else 100.0
    T_adj_front = d_adj_front / ego_speed if ego_speed > 0 else 10.0

    # 目标车道后车信息
    target_rear = self._get_target_rear_vehicle(vehicle_id)
    delta_v_adj_rear = target_rear['speed'] - ego_speed if target_rear else 0
    d_adj_rear = target_rear['distance'] if target_rear else 100.0
    T_adj_rear = d_adj_rear / abs(delta_v_adj_rear) if delta_v_adj_rear != 0 else 10.0

    # 归一化并返回
    ind_features = np.array([
        self._normalize(ego_speed, 'v_ego'),
        self._normalize(v_rel, 'v_rel'),
        self._normalize(d_headway, 'd_headway'),
        self._normalize(T_headway, 'T_headway'),
        self._normalize(delta_v_adj_front, 'delta_v_adj_front'),
        self._normalize(d_adj_front, 'd_adj_front'),
        self._normalize(T_adj_front, 'T_adj_front'),
        self._normalize(delta_v_adj_rear, 'delta_v_adj_rear'),
        self._normalize(d_adj_rear, 'd_adj_rear'),
        self._normalize(T_adj_rear, 'T_adj_rear')
    ])

    return ind_features
```

## 性能优化建议

1. **使用缓存**: 通过 `use_cache=True` 避免重复加载模型
2. **批量预测**: 尽量使用批量预测接口，提高效率
3. **GPU加速**: 如果有GPU，设置 `device="cuda"`
4. **预先归一化**: 确保输入特征已经归一化，避免在推理时重复计算

## 注意事项

1. **特征归一化**: 输入特征必须使用与训练时相同的归一化方法
2. **特征顺序**: 特征的顺序必须与训练时一致
3. **缺失值处理**: 如果周边车辆不存在，使用合理的默认值（如大距离、零速度差）
4. **单位一致性**: 确保速度、距离、时间等单位与训练数据一致

## 运行演示

```bash
cd /Users/liuzhengxuan/00_Project_Code/01_MARL_MultiLane/MARL_MultiLane/harl/prediction/intention/strategy/SCM_prediction
python execute_prediction.py
```

## 模型架构详情

### Shallow架构
```
Path: R^4 → R^8 → R
├── Linear(4, 8)
├── Tanh()
└── Linear(8, 1)
```

### Medium架构
```
Path: R^4 → R^12 → R^4 → R
├── Linear(4, 12)
├── Tanh()
├── Linear(12, 4)
├── Tanh()
└── Linear(4, 1)
```

### Hierarchical融合策略
```
1. 安全门控: g_S = σ(β·(S - θ_S))
2. 动机组合: M = w·N + (1-w)·F
3. 层次输出: Y = g_S · M

其中:
- N: 必要性路径（与当前车道前车的交互）
- F: 可行性路径（与目标车道前车的关系）
- S: 安全性路径（与目标车道后车的关系）
```

## 联系方式

如有问题，请联系交通流研究团队。
