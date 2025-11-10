# CQR轨迹占用预测模块

基于Conformal Quantile Regression (CQR)的车辆未来轨迹占用预测模块，用于MARL多车道环境中的轨迹预测和不确定性量化。

## 目录结构

```
CQR_prediction/
├── __init__.py              # 模块初始化文件
├── models.py                # GRU-QR和MLP-QR模型定义
├── feature_encoder.py       # 特征编码器
├── conformalization.py      # CQR校准器
├── execute_prediction.py    # 预测执行器
└── README.md                # 本文件
```

## 模型说明

### 可用模型

当前支持四个预训练模型（位于 `../../models/`）：

#### 1. QR模型（仅分位数回归）

- **QR-GRU-uncertainty**: 基础GRU-QR模型
  - 不确定性加权损失
  - 预测3个分位数：5%, 50%, 95%
  - 推荐用于快速推理

- **QR-GRU-uncertainty_pcgrad**: GRU-QR + PCGrad
  - 不确定性加权损失 + 梯度投影
  - 更好的分位数平衡
  - 推荐用于高精度场景

#### 2. CQR模型（分位数回归 + 共形化校准）

- **CQR-GRU-uncertainty**: GRU-QR + CQR校准
  - 需要先读取`00-QR-GRU/best_model.pt`
  - 然后读取`calibrator_standard.npz`
  - 提供≥90%覆盖保证
  - 推荐用于安全关键应用

- **CQR-GRU-uncertainty_pcgrad**: GRU-QR + PCGrad + CQR校准
  - 最高精度 + 覆盖保证
  - 推荐用于高要求场景

### 模型输入

#### 特征字典格式

```python
features = {
    # 交通状态 [batch, 6]
    'traffic_state': [EL_v, EL_k, LL_v, LL_k, RL_v, RL_k],

    # 自车当前状态 [batch, 4]
    'ego_current': [v, a, lane_id, eta],

    # 自车历史轨迹 [batch, 20, 5]
    'ego_history': [[Δx, Δy, Δv, Δa, lane_id], ...],

    # 周边车辆当前状态 [batch, 6, 8]
    'sur_current': 6个车辆 × 8个特征,

    # 周边车辆历史轨迹 [batch, 6, 20, 5]
    'sur_history': 6个车辆 × 20步 × 5个特征,

    # 横向意图 [batch, 1]
    'intention': 0 或 1,

    # 初始状态 [batch, 3] (可选，会自动生成)
    'initial_state': [x(t0), v(t0), a(t0)]
}
```

#### 特征详细说明

| 特征组 | 特征名 | 维度 | 描述 |
|--------|--------|------|------|
| **交通状态** | EL_v | 1 | 紧急车道平均速度 |
| | EL_k | 1 | 紧急车道密度 |
| | LL_v | 1 | 左侧车道平均速度 |
| | LL_k | 1 | 左侧车道密度 |
| | RL_v | 1 | 右侧车道平均速度 |
| | RL_k | 1 | 右侧车道密度 |
| **自车当前** | v | 1 | 速度 |
| | a | 1 | 加速度 |
| | lane_id | 1 | 车道ID |
| | eta | 1 | 预计到达时间 |
| **自车历史** | Δx | 20 | 纵向位移序列 |
| | Δy | 20 | 横向位移序列 |
| | Δv | 20 | 速度变化序列 |
| | Δa | 20 | 加速度变化序列 |
| | lane_id | 20 | 车道ID序列 |
| **周边当前** | - | 6×8 | 6个邻车的8维特征 |
| **周边历史** | - | 6×20×5 | 6个邻车的20步轨迹 |
| **横向意图** | intention | 1 | 0=保持, 1=换道 |

### 模型输出

#### QR输出（3个分位数）

```python
q_lo:     [batch, 30]  # 第5百分位数（下界）
q_median: [batch, 30]  # 第50百分位数（中位数）
q_hi:     [batch, 30]  # 第95百分位数（上界）
```

#### CQR输出（校准后的预测区间）

```python
x_min:  [batch, 30]  # 校准后的下界 = q_lo - Q
median: [batch, 30]  # 中位数（可能经过偏差校正）
x_max:  [batch, 30]  # 校准后的上界 = q_hi + Q
```

其中：
- `Q`: 校准分位数，从校准集上计算得到
- **覆盖保证**: P(Y ∈ [x_min, x_max]) ≥ 90%

## 使用方法

### 基本使用

```python
from harl.prediction.occupancy.strategy.CQR_prediction import CQRPredictorFactory
import numpy as np

# 1. 创建预测器
predictor = CQRPredictorFactory.create_predictor(
    model_type="CQR-GRU-uncertainty",  # 或其他模型
    calibrator_type="standard",         # "standard" 或 "asymmetric"
    device="cpu"                        # 或 "cuda"
)

# 2. 准备输入数据
features = {
    'traffic_state': np.random.randn(batch_size, 6),
    'ego_current': np.random.randn(batch_size, 4),
    'ego_history': np.random.randn(batch_size, 20, 5),
    'sur_current': np.random.randn(batch_size, 6, 8),
    'sur_history': np.random.randn(batch_size, 6, 20, 5),
    'intention': np.array([[0], [1]])
}

# 3. 执行预测
x_min, median, x_max = predictor.predict(features, return_intervals=True)
print(f"预测的中位数轨迹: {median.shape}")  # [batch, 30]
print(f"90%置信区间: [{x_min[0, 0]:.2f}, {x_max[0, 0]:.2f}]")
```

### 仅使用QR模型

```python
# 创建QR预测器（不使用共形化）
qr_predictor = CQRPredictorFactory.create_predictor(
    model_type="QR-GRU-uncertainty",
    device="cpu"
)

# QR预测
q_lo, q_median, q_hi = qr_predictor.predict(features, return_intervals=True)
print(f"QR区间宽度: {(q_hi - q_lo).mean():.3f}")
```

### 单样本预测

```python
# 单样本预测的便捷接口
result = predictor.predict_single(
    traffic_state=[25.0, 0.05, 28.0, 0.04, 26.0, 0.045],
    ego_current=[27.0, 0.5, 1.0, 10.0],
    ego_history=np.random.randn(20, 5),
    sur_current=np.random.randn(6, 8),
    sur_history=np.random.randn(6, 20, 5),
    intention=0.0
)

print(f"中位数预测: {result['median']}")        # [30]
print(f"下界: {result['lower']}")               # [30]
print(f"上界: {result['upper']}")               # [30]
print(f"区间宽度: {result['interval_width']}")  # [30]
```

### 查看可用模型

```python
# 获取所有可用的预训练模型
available = CQRPredictorFactory.get_available_models()
print(f"QR模型: {available['QR']}")
print(f"CQR模型: {available['CQR']}")
```

### 查看模型信息

```python
# 获取当前模型的详细信息
info = predictor.get_model_info()
print(info)
# 输出:
# {
#     'model_type': 'CQR-GRU-uncertainty',
#     'use_conformal': True,
#     'device': 'cpu',
#     'model_config': {...},
#     'calibrator': {
#         'type': 'standard',
#         'alpha': 0.1,
#         'target_coverage': '>= 90%',
#         'n_calib': 1000
#     }
# }
```

## 在MARL环境中使用

### 集成到多车道环境

```python
class MultiLaneEnv:
    def __init__(self):
        # 初始化预测器（使用单例缓存）
        from harl.prediction.occupancy.strategy.CQR_prediction import CQRPredictorFactory

        self.occupancy_predictor = CQRPredictorFactory.create_predictor(
            model_type="CQR-GRU-uncertainty",
            calibrator_type="standard",
            device="cpu",
            use_cache=True  # 使用单例缓存
        )

    def predict_future_occupancy(self, vehicle_ids):
        """预测周边车辆的未来占用"""
        observations = []

        for veh_id in vehicle_ids:
            # 提取特征
            features = self._extract_trajectory_features(veh_id)
            observations.append(features)

        # 批量预测
        batch_features = self._batch_features(observations)
        x_min, median, x_max = self.occupancy_predictor.predict(
            batch_features, return_intervals=True
        )

        # 返回字典映射
        results = {}
        for i, veh_id in enumerate(vehicle_ids):
            results[veh_id] = {
                'median': median[i],
                'lower': x_min[i],
                'upper': x_max[i],
                'occupancy_region': self._compute_occupancy_region(
                    x_min[i], x_max[i]
                )
            }
        return results
```

### 特征提取示例

```python
def _extract_trajectory_features(self, vehicle_id):
    """提取轨迹预测特征"""
    # 1. 交通状态
    traffic_state = self._get_traffic_state()  # [6]

    # 2. 自车当前状态
    ego_state = self._get_vehicle_state(vehicle_id)
    ego_current = [
        ego_state['speed'],
        ego_state['acceleration'],
        ego_state['lane_id'],
        ego_state['eta']
    ]  # [4]

    # 3. 自车历史轨迹（最近20步）
    ego_history = self._get_vehicle_history(vehicle_id, steps=20)  # [20, 5]

    # 4. 周边车辆当前状态
    sur_current = self._get_surrounding_vehicles_state(vehicle_id, max_vehicles=6)  # [6, 8]

    # 5. 周边车辆历史轨迹
    sur_history = self._get_surrounding_vehicles_history(vehicle_id, max_vehicles=6, steps=20)  # [6, 20, 5]

    # 6. 横向意图（从意图预测模块获取）
    intention = self._get_lane_change_intention(vehicle_id)  # 0 或 1

    return {
        'traffic_state': traffic_state,
        'ego_current': ego_current,
        'ego_history': ego_history,
        'sur_current': sur_current,
        'sur_history': sur_history,
        'intention': intention
    }
```

## QR vs CQR 对比

| 方面 | QR (Quantile Regression) | CQR (Conformal QR) |
|------|-------------------------|-------------------|
| **输出** | 直接分位数预测 | 校准后的预测区间 |
| **覆盖保证** | 无理论保证 | 保证≥90% |
| **校准** | 不需要 | 需要校准集 |
| **区间宽度** | 较窄（可能不可靠） | 较宽（但可靠） |
| **推理速度** | 快 | 快（仅需额外一次加法） |
| **适用场景** | 普通预测任务 | 安全关键应用 |

### 何时使用

- **仅QR**:
  - 快速推理
  - 不需要覆盖保证
  - 计算资源受限

- **CQR**:
  - 安全关键应用（自动驾驶）
  - 需要可靠的不确定性量化
  - 风险敏感决策

## 模型架构详情

### GRU-QR架构

```
Input Features
    ├── Traffic State [6]
    ├── Ego Current [4]
    ├── Ego History [20, 5]
    ├── Sur Current [6, 8]
    └── Sur History [6, 20, 5]
         ↓
Feature Encoder (h_fused: [128])
         ↓
GRU Decoder (共享)
    - 输入: [s_t, time_encoding, h_fused]
    - s_t: [x, v, a] 当前状态
    - time_encoding: [32] 时间编码
    - GRU: 2层 × 128隐藏单元
         ↓
Multi-head Output Layer
    - 输出: [q_lo, q_median, q_hi]
         ↓
Softplus Constraints
    - q_lo = q_lo_raw
    - q_median = q_lo + softplus(q_median_raw - q_lo)
    - q_hi = q_median + softplus(q_hi_raw - q_median)
         ↓
Output: [q_lo, q_median, q_hi] each [batch, 30]
```

### CQR校准流程

```
1. QR预测
   ├── 校准集: N_calib samples
   └── 输出: (q_lo, q_median, q_hi)

2. 计算Conformity Scores
   ├── E = max(q_lo - y, y - q_hi)
   └── shape: [N_calib, 30]

3. 计算校准分位数
   ├── quantile_level = (1-α)(1+1/N_calib)
   └── Q = quantile(E, quantile_level)  # [30]

4. 应用校准
   ├── x_min = q_lo - Q
   ├── x_max = q_hi + Q
   └── median = q_median (不变)

5. 覆盖保证
   └── P(Y ∈ [x_min, x_max]) ≥ 90%
```

## 性能优化建议

1. **使用单例缓存**: `use_cache=True` 避免重复加载
2. **批量预测**: 使用 `predict` 而非多次调用 `predict_single`
3. **选择合适的模型**:
   - 实时性优先: `QR-GRU-uncertainty`
   - 安全性优先: `CQR-GRU-uncertainty`
   - 最高精度: `CQR-GRU-uncertainty_pcgrad`
4. **GPU加速**: 如果可用，设置 `device="cuda"`

## 注意事项

### 必须注意

1. **特征格式**: 必须严格按照上述格式提供输入特征
2. **特征顺序**: 特征的顺序必须与训练时一致
3. **缺失值处理**: 当周边车辆不存在时，使用零填充
4. **时间同步**: 历史轨迹必须与当前时刻对齐

### 推荐做法

1. **批量大小**: 推荐batch_size=32以获得最佳性能
2. **GPU使用**: 对于大批量推理，强烈推荐使用GPU
3. **模型选择**:
   - 开发测试: QR模型
   - 生产部署: CQR模型

## 运行演示

```bash
cd /Users/liuzhengxuan/00_Project_Code/01_MARL_MultiLane/MARL_MultiLane/harl/prediction/occupancy/strategy/CQR_prediction
python3 execute_prediction.py
```

## 源代码来源

- **算法设计**: `00_Occupancy_Prediction/OccupancyPred/00_trajectory_prediction/CQR_SCM_design/`
- **模型训练**: 使用CQR_SCM_design框架训练
- **模型文件**: 从训练输出复制到 `models/QR_models/` 和 `models/CQR_models/`

## 文件清单

### 新建文件

| 文件路径 | 描述 |
|---------|------|
| CQR_prediction/__init__.py | 模块初始化 |
| CQR_prediction/models.py | 模型定义（从CQR_SCM_design复制） |
| CQR_prediction/feature_encoder.py | 特征编码器（从CQR_SCM_design复制） |
| CQR_prediction/conformalization.py | CQR校准器（从CQR_SCM_design复制） |
| CQR_prediction/execute_prediction.py | 预测执行器（新设计） |
| CQR_prediction/README.md | 本文件 |

### 依赖的预训练模型

| 模型类型 | 文件路径 | 大小 |
|---------|---------|------|
| QR-GRU-uncertainty | models/QR_models/QR-GRU-uncertainty/best_model.pt | ~9MB |
| QR-GRU-uncertainty_pcgrad | models/QR_models/QR-GRU-uncertainty_pcgrad/best_model.pt | ~9MB |
| CQR-GRU-uncertainty (QR部分) | models/CQR_models/CQR-GRU-uncertainty/00-QR-GRU/best_model.pt | ~9MB |
| CQR-GRU-uncertainty (校准器) | models/CQR_models/CQR-GRU-uncertainty/calibrator_standard.npz | ~1.4KB |
| CQR-GRU-uncertainty_pcgrad (QR部分) | models/CQR_models/CQR-GRU-uncertainty_pcgrad/00-QR-GRU/best_model.pt | ~9MB |
| CQR-GRU-uncertainty_pcgrad (校准器) | models/CQR_models/CQR-GRU-uncertainty_pcgrad/calibrator_standard.npz | ~1.4KB |

## 联系方式

如有问题，请联系交通流研究团队。
