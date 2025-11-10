# SCM横向意图决策模块

基于预训练的SCM意图预测模型生成CAV的换道决策，支持MARL训练过程中的在线微调。

## 核心思想

该模块通过模拟人类驾驶意图的逻辑来生成CAV的横向换道决策：

1. **初始化**: 从预训练的SCM意图预测模型加载权重
2. **决策生成**: 使用相同的三路径因果结构生成决策概率
3. **MARL微调**: 在强化学习训练过程中对决策模型进行微调
4. **人机一致性**: 保证CAV决策与人类驾驶意图逻辑一致

## 目录结构

```
SCM_decisions/
├── __init__.py                 # 模块初始化
├── scm_decision_model.py      # SCM决策模型定义
├── execute_decision.py        # 决策执行器
├── README.md                  # 本文件
└── MARL_INTEGRATION_GUIDE.md  # MARL集成指南
```

## 模型架构

### 基于预训练SCM的决策模型

```
预训练SCM模型 (HierarchicalSCMModelV2)
    ├── 环境层 (env_layer)
    ├── 个体层 (ind_layer)
    │   ├── 必要性路径 (Necessity)
    │   ├── 可行性路径 (Feasibility)
    │   └── 安全性路径 (Safety)
    └── 类型调制 (vehicle_type)
         ↓
决策概率 P_final [0, 1]
         ↓
决策阈值 (decision_threshold)
         ↓
二值决策 {0, 1}
```

### 参数配置

| 配置 | 描述 | 推荐值 |
|------|------|--------|
| freeze_base_model | 冻结基础SCM模型 | True（MARL初期）<br>False（MARL后期） |
| decision_threshold | 决策阈值 | 0.5（可微调） |
| enable_training | 启用训练模式 | True（MARL训练）<br>False（部署） |

## 快速开始

### 1. 基础使用（推理模式）

```python
from harl.decisions.lateral_decisions.SCM_decisions import SCMDecisionMakerFactory

# 创建决策生成器
decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    model_type="shallow_hierarchical",  # 或 "medium_hierarchical"
    freeze_base_model=True,
    enable_training=False,
    device="cpu"
)

# 准备输入特征（与SCM预测模型相同）
env_features = np.array([[v_avg, kappa_avg, delta_v, delta_kappa]])  # [1, 4]
ind_features = np.array([[v_ego, v_rel, d_headway, T_headway, ...]])  # [1, 10]
vehicle_type = np.array([[1]])  # CAV

# 生成决策
decision = decision_maker.decide(env_features, ind_features, vehicle_type)
print(f"换道决策: {decision[0, 0]}")  # 0=保持, 1=换道
```

### 2. MARL训练模式

```python
# 创建用于MARL训练的决策生成器
decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    model_type="shallow_hierarchical",
    freeze_base_model=True,  # 初期冻结，后期可解冻
    enable_training=True,    # 启用训练模式
    device="cuda"
)

# 获取模型进行MARL训练
decision_model = decision_maker.get_model()

# 在MARL训练循环中
optimizer = torch.optim.Adam(decision_model.get_trainable_parameters(), lr=1e-4)

for episode in range(num_episodes):
    # 生成决策（保持梯度）
    decision_prob = decision_maker.decide(
        env_features, ind_features, vehicle_types,
        return_prob=True  # 返回概率用于计算损失
    )

    # 计算MARL奖励
    reward = env.step(decision_prob)

    # 反向传播
    loss = -reward * torch.log(decision_prob)  # 策略梯度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 保存微调后的模型
decision_maker.save_model("finetuned_scm_decision.pth")
```

### 3. 单样本决策

```python
# 便捷的单样本决策接口
result = decision_maker.decide_single(
    env_feature=[25.0, 0.05, 2.0, -0.01],
    ind_feature=[27.0, 1.5, 50.0, 2.0, -0.5, 60.0, 2.5, 1.0, 40.0, 1.5],
    vehicle_type=1  # CAV
)

print(f"决策: {result['decision']}")           # 二值决策
print(f"概率: {result['probability']:.4f}")    # 换道概率
print(f"置信度: {result['confidence']:.4f}")   # 决策置信度
print(f"阈值: {result['threshold']:.4f}")      # 当前阈值
```

## 输入特征规范

### 与SCM预测模型完全一致

#### 环境特征 (4维)

| 特征名 | 描述 |
|--------|------|
| v_avg_norm | 平均速度（归一化） |
| kappa_avg_norm | 平均密度（归一化） |
| delta_v_norm | 车道速度差（归一化） |
| delta_kappa_norm | 车道密度差（归一化） |

#### 个体特征 (10维)

| 特征名 | 描述 |
|--------|------|
| v_ego_norm | 自车速度（归一化） |
| v_rel_norm | 与前车相对速度 |
| d_headway_norm | 车头间距 |
| T_headway_norm | 时间间距 |
| delta_v_adj_front_norm | 与目标车道前车速度差 |
| d_adj_front_norm | 与目标车道前车距离 |
| T_adj_front_norm | 与目标车道前车时间间隙 |
| delta_v_adj_rear_norm | 与目标车道后车速度差 |
| d_adj_rear_norm | 与目标车道后车距离 |
| T_adj_rear_norm | 与目标车道后车时间间隙 |

#### 车辆类型

- `0`: HDV (Human-Driven Vehicle)
- `1`: CAV (Connected Autonomous Vehicle)

## 输出说明

### 决策输出

```python
# 二值决策
decision: {0, 1}
  - 0: 保持当前车道
  - 1: 执行换道

# 决策概率
probability: [0, 1]
  - 接近0: 强烈倾向保持车道
  - 接近1: 强烈倾向换道

# 决策置信度
confidence: [0, 0.5]
  - 概率距离阈值的远近
  - 越大表示决策越确定
```

### 中间结果

```python
intermediates = {
    'psi_env': 环境层输出,
    'psi_ind': 个体层输出,
    'path_N': 必要性路径输出,
    'path_F': 可行性路径输出,
    'path_S': 安全性路径输出,
    'gate': 门控值,
    'decision_threshold': 决策阈值,
    'decision': 二值决策
}
```

## 在MARL环境中使用

### 集成到MultiLaneEnv

```python
class MultiLaneEnv:
    def __init__(self, args):
        # 初始化SCM决策生成器
        from harl.decisions.lateral_decisions.SCM_decisions import SCMDecisionMakerFactory

        self.scm_decision_maker = SCMDecisionMakerFactory.create_decision_maker(
            model_type="shallow_hierarchical",
            freeze_base_model=True,
            enable_training=True,  # MARL训练时设为True
            device=args.device,
            use_cache=True
        )

    def get_cav_lateral_decision(self, cav_id):
        """为CAV生成横向决策"""
        # 提取特征
        env_feat = self._extract_env_features(cav_id)
        ind_feat = self._extract_ind_features(cav_id)

        # 生成决策
        decision = self.scm_decision_maker.decide_single(
            env_feature=env_feat,
            ind_feature=ind_feat,
            vehicle_type=1  # CAV
        )

        return decision['decision'], decision['probability']

    def update_scm_decision_model(self, rewards, optimizer):
        """在MARL训练中更新SCM决策模型"""
        # 获取模型
        model = self.scm_decision_maker.get_model()

        # 计算损失
        loss = self._compute_policy_loss(rewards)

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 应用因果约束
        model.apply_causal_constraints()
```

## 微调策略

### 1. 渐进式微调（推荐）

```python
# 阶段1: 冻结基础模型，仅训练决策阈值
decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    freeze_base_model=True,
    enable_training=True
)
# 训练 1000 episodes

# 阶段2: 解冻个体层，微调因果路径
model = decision_maker.get_model()
model.unfreeze_layer(model.scm_model.ind_layer)
# 训练 500 episodes

# 阶段3: 全局微调
model.unfreeze_layer(model.scm_model.env_layer)
# 训练 200 episodes
```

### 2. 端到端微调

```python
# 直接解冻所有层
decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    freeze_base_model=False,
    enable_training=True
)
```

## 决策统计

### 获取决策统计信息

```python
# 当前episode统计
stats = decision_maker.get_decision_stats()
print(f"换道率: {stats['lane_change_rate']:.2%}")
print(f"换道次数: {stats['lane_change_count']}/{stats['total_decisions']}")

# 所有episodes统计
all_stats = decision_maker.get_all_episode_stats()
for i, ep_stats in enumerate(all_stats):
    print(f"Episode {i}: 换道率 = {ep_stats['lane_change_rate']:.2%}")

# 重置统计（新episode开始时）
decision_maker.reset_episode_stats()
```

## 模型保存与加载

### 保存微调后的模型

```python
# 保存包含episode统计的完整模型
decision_maker.save_model("scm_decision_finetuned.pth")
```

### 加载微调后的模型

```python
# 先创建决策生成器
decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    model_type="shallow_hierarchical",
    enable_training=False
)

# 加载微调后的权重
decision_maker.load_finetuned_model("scm_decision_finetuned.pth")
```

## 参数状态查看

```python
# 查看参数冻结状态
model = decision_maker.get_model()
model.print_parameter_status()

# 输出示例:
# [SCMDecisionModel] 参数状态:
#   总参数量: 162
#   可训练参数: 5 (3.1%)
#   冻结参数: 157 (96.9%)
#
#   各层状态:
#     环境层: 冻结
#     个体层: 冻结
#     类型调制: 可训练
#     决策阈值: 0.5000 (可训练)
```

## 与SCM预测模块的关系

| 模块 | 用途 | 输入 | 输出 | 训练 |
|------|------|------|------|------|
| SCM预测 | 预测周边车辆意图 | 周边车辆特征 | 换道意图概率 | 预训练完成 |
| SCM决策 | 生成ego CAV决策 | ego车辆特征 | 换道决策 | MARL微调 |

**关键区别**:
- **预测模块**: 用于预测**其他车辆**的换道意图（固定权重）
- **决策模块**: 用于生成**ego CAV**的换道决策（可微调）

## 性能优化建议

1. **使用单例缓存**: `use_cache=True` 避免重复加载
2. **合理选择冻结策略**:
   - 初期：冻结基础模型
   - 中期：解冻个体层
   - 后期：全局微调
3. **GPU加速**: 设置 `device="cuda"`
4. **批量决策**: 使用 `decide` 而非多次调用 `decide_single`

## 注意事项

### 必须注意

1. **特征一致性**: 输入特征必须与SCM预测模型保持一致
2. **归一化**: 所有输入必须经过与训练时相同的归一化
3. **因果约束**: 微调时定期调用 `apply_causal_constraints()`
4. **梯度累积**: 训练模式下注意梯度管理

### 推荐做法

1. **渐进式微调**: 从冻结到解冻，逐步微调
2. **小学习率**: 使用1e-4到1e-5的学习率
3. **定期保存**: 每隔N个episodes保存checkpoint
4. **监控统计**: 跟踪换道率，避免过度激进或保守

## 运行演示

```bash
cd /Users/liuzhengxuan/00_Project_Code/01_MARL_MultiLane/MARL_MultiLane/harl/decisions/lateral_decisions/SCM_decisions
python3 execute_decision.py
```

## 文件清单

| 文件 | 大小 | 描述 |
|------|------|------|
| __init__.py | 1.2KB | 模块初始化 |
| scm_decision_model.py | 17.5KB | 决策模型定义 |
| execute_decision.py | 16.8KB | 决策执行器 |
| README.md | 本文件 | 使用文档 |
| MARL_INTEGRATION_GUIDE.md | - | MARL集成详细指南 |

## 常见问题

### Q1: 如何选择模型类型？

A: 与SCM预测模块保持一致：
- `shallow_hierarchical`: 快速推理
- `medium_hierarchical`: 高精度

### Q2: 何时解冻基础模型？

A:
- MARL训练初期（前30%）: 冻结
- MARL训练中期（30%-70%）: 解冻个体层
- MARL训练后期（70%-100%）: 全局微调

### Q3: 如何平衡探索与利用？

A:
- 使用epsilon-greedy策略
- 或者直接使用概率进行采样决策

### Q4: 如何保证人机一致性？

A:
- 初始化时加载预训练权重
- 微调时使用小学习率
- 定期评估与人类决策的相似度

## 联系方式

如有问题，请联系交通流研究团队。
