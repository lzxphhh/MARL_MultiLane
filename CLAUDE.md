# CLAUDE.md - AI Assistant Guide for MARL Multi-Lane Project

**Last Updated**: 2025-11-14
**Project Completion**: ~90% (3/3 high-priority issues fixed)
**Purpose**: Multi-Agent Reinforcement Learning (MARL) for autonomous multi-lane driving with integrated prediction and decision-making

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Workflows](#development-workflows)
4. [Key Modules & APIs](#key-modules--apis)
5. [Configuration System](#configuration-system)
6. [Code Conventions](#code-conventions)
7. [Common Tasks](#common-tasks)
8. [Known Issues & Fixes](#known-issues--fixes)
9. [Testing & Debugging](#testing--debugging)
10. [Important Files Reference](#important-files-reference)

---

## 🎯 Project Overview

### What is this project?

This is a **Multi-Agent Reinforcement Learning (MARL)** framework for multi-lane autonomous driving that combines:

- **MAPPO/MPMAPPO algorithms** for longitudinal acceleration control
- **SCM (Structural Causal Model)** for lane-change intention prediction
- **CQR (Conformalized Quantile Regression)** for trajectory occupancy prediction
- **SCM-based lateral decision-making** with online fine-tuning
- **SUMO traffic simulation** for realistic multi-lane scenarios

### Key Features

- **Centralized Training, Decentralized Execution (CTDE)**: Critic uses global state, actors use local observations
- **Progressive Fine-Tuning**: 3-stage training strategy for decision models
- **Multi-Objective Optimization**: Dynamic balancing of safety, efficiency, stability, comfort
- **Scene-Aware Learning**: Different policies for different traffic conditions
- **Causal Interpretability**: SCM-based decisions maintain causal structure

### Architecture

```
Environment (SUMO)
    ↓
Feature Extraction → Prediction (SCM + CQR) → Decision (SCM + MARL) → Control
    ↓                                                                      ↓
Observation                                                           Rewards
    ↓                                                                      ↓
Actor Network (MAPPO) ← PPO Updates ← Critic Network (V-function)
```

---

## 📁 Repository Structure

```
MARL_MultiLane/
├── train.py                              # Main training entry point
├── examples/                             # Training examples
│   ├── train.py                         # Alternative training script
│   └── batch_training/                  # Batch experiment utilities
│       ├── run_batch_training.py        # Orchestrate multiple experiments
│       ├── batch_train.py               # Individual experiment runner
│       ├── training_monitor.py          # Real-time monitoring
│       ├── results_analyzer.py          # Post-training analysis
│       └── batch_config.yaml            # Batch configuration
│
├── harl/                                # Core HARL framework
│   ├── algorithms/                      # MARL algorithms
│   │   ├── actors/
│   │   │   ├── mappo.py                # MAPPO actor (PPO updates)
│   │   │   ├── multi_policy_mappo.py   # Scene-aware multi-policy
│   │   │   └── on_policy_base.py       # Base actor class
│   │   └── critics/
│   │       ├── v_critic.py             # Value function critic
│   │       └── mp_mh_v_critic.py       # Multi-policy multi-head critic
│   │
│   ├── models/                          # Neural network architectures
│   │   ├── base/                        # Base components
│   │   │   ├── mlp.py                  # Multi-layer perceptron
│   │   │   ├── cnn.py                  # Convolutional layers
│   │   │   ├── rnn.py                  # GRU/LSTM layers
│   │   │   └── multi_lane/             # Multi-lane encoders
│   │   │       ├── improved_encoder.py  # Multi-lane state encoder
│   │   │       ├── interaction_encoder.py  # Vehicle interaction
│   │   │       └── motion_encoder.py    # Motion patterns
│   │   ├── policy_models/
│   │   │   └── stochastic_policy.py    # Actor network (Gaussian policy)
│   │   └── value_function_models/
│   │       ├── v_net.py                # Simple V-net (MLP-based)
│   │       ├── historical_v_net.py     # Full V-net (GAT + CrossAttn) ✓
│   │       └── multihead_v_net.py      # Multi-objective V-net
│   │
│   ├── envs/                            # Environment implementations
│   │   ├── a_multi_lane/               # Multi-lane environment
│   │   │   ├── multilane_env.py        # Gym wrapper (main env)
│   │   │   ├── multilane_logger.py     # Logging utilities
│   │   │   ├── env_utils/              # Environment utilities
│   │   │   │   ├── veh_env.py         # Base SUMO environment
│   │   │   │   ├── veh_env_wrapper.py # Feature-rich wrapper (2000+ lines)
│   │   │   │   ├── feature_extractor.py  # Feature extraction ✓
│   │   │   │   ├── prediction_module.py  # Prediction integration ✓
│   │   │   │   ├── decision_module.py    # Decision integration ✓
│   │   │   │   ├── lateral_controller.py # Bezier trajectory planning ✓
│   │   │   │   ├── safety_assessment.py  # Collision risk ✓
│   │   │   │   ├── replay_buffer.py      # Experience replay ✓
│   │   │   │   ├── model_updater.py      # Progressive fine-tuning ✓
│   │   │   │   └── training_config.py    # Training parameters ✓
│   │   │   └── hierarchical_controller/  # MPC controllers
│   │   └── env_wrappers.py             # General wrappers
│   │
│   ├── prediction/                      # Prediction modules
│   │   ├── intention/
│   │   │   └── strategy/SCM_prediction/ # SCM intention predictor
│   │   │       ├── scm_model_v2.py     # Hierarchical SCM model
│   │   │       ├── execute_prediction.py  # Factory interface
│   │   │       └── README.md           # Usage guide
│   │   └── occupancy/
│   │       └── strategy/CQR_prediction/ # CQR occupancy predictor
│   │           ├── models.py           # GRU-CQR models (FIXED ✓)
│   │           ├── feature_encoder.py  # Feature encoding
│   │           ├── execute_prediction.py  # Factory interface
│   │           └── README.md           # Usage guide
│   │
│   ├── decisions/                       # Decision modules
│   │   └── lateral_decisions/
│   │       └── SCM_decisions/          # SCM lateral decisions
│   │           ├── scm_decision_model.py  # Decision model
│   │           ├── execute_decision.py    # Factory + fine-tuning (FIXED ✓)
│   │           └── README.md           # Usage guide
│   │
│   ├── runners/                         # Training orchestration
│   │   ├── on_policy_base_runner.py    # Base runner (1279 lines)
│   │   ├── on_policy_ma_runner.py      # MAPPO runner
│   │   └── multi_policy_mhc_ma_runner.py  # Multi-policy runner
│   │
│   ├── common/                          # Common utilities
│   │   ├── buffers/                    # Experience buffers
│   │   │   ├── on_policy_actor_buffer.py
│   │   │   ├── on_policy_critic_buffer_ep.py
│   │   │   └── multi_policy_actor_buffer.py
│   │   ├── base_logger.py              # Training logger
│   │   └── valuenorm.py                # Value normalization
│   │
│   ├── functions/                       # Advanced techniques
│   │   ├── gradient_projection.py      # PCGrad (731 lines)
│   │   └── weight_computation.py       # Dynamic reward weights (373 lines)
│   │
│   ├── configs/                         # Configuration files
│   │   ├── algos_cfgs/
│   │   │   ├── mappo.yaml             # MAPPO configuration
│   │   │   └── mpmappo.yaml           # Multi-policy MAPPO
│   │   └── envs_cfgs/
│   │       └── a_multi_lane.yaml      # Environment configuration
│   │
│   └── utils/                           # Utility functions
│       └── configs_tools.py            # Config loading/merging
│
├── PROJECT_AUDIT_REPORT.md             # Comprehensive audit (510 lines)
├── AUDIT_FIX_VERIFICATION.md           # Fix verification (840 lines)
├── train_execution.md                  # Training execution guide
└── .gitignore                          # Git ignore patterns
```

---

## 🔄 Development Workflows

### 1. Training Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Parse Arguments & Load Configs                          │
│    - Command-line args → JSON config → YAML defaults       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Create Environment                                       │
│    - MultiLaneEnv wraps VehEnvWrapper                      │
│    - Initialize SUMO simulation                             │
│    - Setup prediction/decision modules                      │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Initialize Runner                                        │
│    - Create actor/critic networks                           │
│    - Setup buffers, optimizer, logger                       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Training Loop (runner.run())                            │
│    ┌────────────────────────────────────────────────────┐  │
│    │ 4a. Collect Rollouts (n_threads × episode_length)  │  │
│    │     - Get observations from env                    │  │
│    │     - Actor outputs actions (Gaussian sampling)    │  │
│    │     - Execute in SUMO, get rewards                 │  │
│    │     - Store in buffers                             │  │
│    └───────────────────┬────────────────────────────────┘  │
│                        ↓                                    │
│    ┌────────────────────────────────────────────────────┐  │
│    │ 4b. Compute Advantages (GAE)                       │  │
│    │     - Critic evaluates state values V(s)           │  │
│    │     - Compute TD-errors: δ = r + γV(s') - V(s)    │  │
│    │     - Compute GAE: A = Σ(γλ)^t δ_t                │  │
│    └───────────────────┬────────────────────────────────┘  │
│                        ↓                                    │
│    ┌────────────────────────────────────────────────────┐  │
│    │ 4c. Update Networks (PPO)                          │  │
│    │     - Actor: L_CLIP + L_entropy                    │  │
│    │     - Critic: L_value (Huber or MSE)               │  │
│    │     - Apply gradient clipping                      │  │
│    └───────────────────┬────────────────────────────────┘  │
│                        ↓                                    │
│    ┌────────────────────────────────────────────────────┐  │
│    │ 4d. Fine-tune Decision Model (every N episodes)    │  │
│    │     - Stage-aware parameter freezing               │  │
│    │     - Update SCM decision model                    │  │
│    └───────────────────┬────────────────────────────────┘  │
│                        ↓                                    │
│    └────────────────── ↑                                    │
│         Loop until max_steps                                │
└─────────────────────────────────────────────────────────────┘
```

### 2. Environment Step Workflow

```
env.step(actions):
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Prediction Phase                                         │
│    a. Extract intention features (feature_extractor)        │
│    b. Predict lane-change intentions (SCM)                  │
│    c. Extract occupancy features                            │
│    d. Predict trajectory occupancies (CQR)                  │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Decision Phase                                           │
│    a. Extract decision features                             │
│    b. Make lateral decisions (SCM + MARL)                   │
│    c. Plan Bezier trajectories (lateral_controller)         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Execution Phase                                          │
│    a. Combine longitudinal (MARL) + lateral (SCM) actions   │
│    b. Execute in SUMO via TraCI                             │
│    c. Step simulation Δt seconds                            │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Observation & Reward                                     │
│    a. Extract vehicle states from SUMO                      │
│    b. Compute observations (ego + neighbors)                │
│    c. Compute rewards (safety + efficiency + ...)          │
│    d. Compute safety penalties (collision risk)             │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
return (obs, rewards, dones, infos)
```

### 3. Progressive Fine-Tuning Strategy

```
Episode 0 ─────────────► Episode 1000 ─────────► Episode 2000 ─────────► Episode 3000+
    │                         │                        │                       │
    ▼                         ▼                        ▼                       ▼
┌─────────────┐      ┌─────────────────┐     ┌──────────────────┐    ┌────────────┐
│  Stage 1    │      │    Stage 2      │     │    Stage 3       │    │  Continue  │
│             │      │                 │     │                  │    │  Stage 3   │
│ Freeze SCM  │      │ Unfreeze Ind.   │     │ Global Fine-tune │    │            │
│ base model  │      │ layer, keep     │     │ All layers       │    │ All layers │
│             │      │ env/fusion      │     │ trainable        │    │ trainable  │
│ LR: 1e-4    │      │ frozen          │     │ LR: 5e-5         │    │ LR: 5e-5   │
│             │      │ LR: 5e-5        │     │                  │    │            │
└─────────────┘      └─────────────────┘     └──────────────────┘    └────────────┘
     │                       │                        │                       │
     └───────────────────────┴────────────────────────┴───────────────────────┘
                         model_updater.py (FIXED ✓)
                    execute_decision.py (FIXED ✓)
```

---

## 🔧 Key Modules & APIs

### 1. Feature Extraction

**File**: `harl/envs/a_multi_lane/env_utils/feature_extractor.py`

```python
from harl.envs.a_multi_lane.env_utils.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# For intention prediction (SCM input)
env_features, ind_features, vehicle_types = extractor.batch_extract_intention_features(
    state=current_state,                    # Vehicle positions, velocities
    lane_statistics=lane_stats,             # Average speed, density per lane
    vehicle_ids=["veh_0", "veh_1", ...]    # List of vehicle IDs
)
# Returns:
#   env_features: [N, 4]  - lane avg speed, density, lane diff, ...
#   ind_features: [N, 10] - relative velocities, headways, gaps, ...
#   vehicle_types: [N, 1] - CAV=1, HDV=0

# For occupancy prediction (CQR input)
occ_features = extractor.extract_occupancy_features(
    vehicle_id="veh_0",
    state=current_state,
    history_trajectories=hist_dict,
    intention_probs=intention_dict
)

# For decision making (SCM decision input)
decision_features = extractor.extract_decision_features(
    vehicle_id="veh_0",
    state=current_state,
    lane_statistics=lane_stats
)
```

### 2. Prediction Module

**File**: `harl/envs/a_multi_lane/env_utils/prediction_module.py`

```python
from harl.envs.a_multi_lane.env_utils.prediction_module import PredictionModule

pred_module = PredictionModule(
    scm_model_type="shallow_hierarchical",  # or "medium_hierarchical"
    cqr_model_type="gru_cqr",              # or "mlp_cqr"
    device="cuda"
)

# Predict intentions (SCM)
intention_probs = pred_module.predict_intentions(
    env_features=env_feats,      # [N, 4]
    ind_features=ind_feats,      # [N, 10]
    vehicle_types=veh_types,     # [N, 1]
    vehicle_ids=veh_ids          # List[str]
)
# Returns: {vehicle_id: probability} where probability ∈ [0, 1]

# Predict occupancy (CQR)
occupancy_pred = pred_module.predict_occupancy(
    features_dict=occ_features,
    prediction_horizon=30,       # 3.0 seconds (0.1s per step)
    alpha=0.1                    # 90% confidence interval
)
# Returns: {
#     'lower': array([N, horizon, 2]),   # Lower bound (x, y)
#     'median': array([N, horizon, 2]),  # Median prediction
#     'upper': array([N, horizon, 2])    # Upper bound
# }
```

### 3. Decision Module

**File**: `harl/envs/a_multi_lane/env_utils/decision_module.py`

```python
from harl.envs.a_multi_lane.env_utils.decision_module import DecisionModule

decision_module = DecisionModule(
    model_type="scm_decision",
    freeze_base_model=True,      # Stage 1: freeze SCM
    device="cuda"
)

# Make decisions
decisions = decision_module.make_batch_decisions(
    env_features=env_feats,      # [N, 4]
    ind_features=ind_feats,      # [N, 10]
    vehicle_types=veh_types,     # [N, 1]
    vehicle_ids=veh_ids,         # List[str]
    return_prob=True             # Return probabilities
)
# Returns: {
#     vehicle_id: {
#         'decision': 0 or 1,     # 0=keep lane, 1=lane change
#         'probability': float    # Decision confidence
#     }
# }

# Fine-tune decision model (called periodically)
loss = decision_module.update_model(
    env_features=env_feats,
    ind_features=ind_feats,
    vehicle_types=veh_types,
    rewards=reward_dict,         # {vehicle_id: reward}
    learning_rate=5e-5
)
```

### 4. Model Updater (Progressive Fine-Tuning)

**File**: `harl/envs/a_multi_lane/env_utils/model_updater.py`

```python
from harl.envs.a_multi_lane.env_utils.model_updater import LateralDecisionUpdater

updater = LateralDecisionUpdater(
    decision_module=decision_module,
    stage_boundaries=[1000, 2000],      # Stage transitions
    learning_rates=[1e-4, 5e-5, 5e-5],  # LR per stage
    update_frequency=10                  # Update every N episodes
)

# In training loop
for episode in range(max_episodes):
    # ... collect data ...

    # Update stage and model
    if episode % updater.update_frequency == 0:
        updater.update_stage(episode)    # Check and switch stages
        updater.update_model(
            trajectories=episode_data,
            rewards=episode_rewards
        )
```

### 5. Safety Assessment

**File**: `harl/envs/a_multi_lane/env_utils/safety_assessment.py`

```python
from harl.envs.a_multi_lane.env_utils.safety_assessment import SafetyAssessor

assessor = SafetyAssessor(
    resolution=0.1,              # 0.1m grid resolution
    horizon=30                   # 3.0 seconds
)

# Compute safety penalty
safety_penalty = assessor.compute_safety_penalty(
    ego_trajectory=planned_traj,         # [T, 2] (x, y)
    predicted_occupancies=occ_pred,      # From CQR
    lambda_penalty=1.0                   # Weight
)

# Check collision risk
is_safe, risk_metrics = assessor.check_collision_risk(
    ego_trajectory=planned_traj,
    other_trajectories=other_trajs,
    ttc_threshold=1.5,           # Time-to-collision threshold
    spacing_threshold=3.0        # Minimum spacing (m)
)
```

### 6. Actor Network (MAPPO)

**File**: `harl/models/policy_models/stochastic_policy.py`

```python
from harl.models.policy_models.stochastic_policy import StochasticPolicy

# Created automatically by runner
actor = StochasticPolicy(
    args=algo_args,
    obs_space=env.observation_space,
    act_space=env.action_space,
    device=device
)

# Forward pass
actions, action_log_probs = actor(
    obs=observations,            # [batch, obs_dim]
    rnn_states=actor_rnn_states, # [batch, hidden_dim]
    masks=masks,                 # [batch, 1]
    deterministic=False          # Stochastic during training
)
# Returns:
#   actions: [batch, act_dim]
#   action_log_probs: [batch, 1]

# Evaluate actions (for PPO update)
action_log_probs, dist_entropy = actor.evaluate_actions(
    obs=observations,
    rnn_states=actor_rnn_states,
    action=actions,
    masks=masks
)
```

### 7. Critic Network (V-function)

**File**: `harl/models/value_function_models/historical_v_net.py` (FIXED ✓)

```python
from harl.models.value_function_models.historical_v_net import HistoricalVNet

# Recommended: Use HistoricalVNet (full architecture with GAT + CrossAttention)
critic = HistoricalVNet(
    args=algo_args,
    cent_obs_space=env.share_observation_space,
    device=device
)

# Forward pass
values, critic_rnn_states = critic(
    cent_obs=centralized_observations,  # [batch, global_state_dim]
    rnn_states=critic_rnn_states,       # [batch, hidden_dim]
    masks=masks                          # [batch, 1]
)
# Returns:
#   values: [batch, 1] - State value V(s)
```

---

## ⚙️ Configuration System

### Configuration Priority

```
Command-line args  (highest)
    ↓
JSON config file  (--load_config)
    ↓
YAML defaults    (lowest)
```

### 1. Algorithm Configuration

**File**: `harl/configs/algos_cfgs/mappo.yaml`

**Key Parameters**:

```yaml
seed:
  seed_specify: True
  seed: 42                           # Random seed

device:
  cuda: True                         # Use GPU
  torch_threads: 4                   # CPU threads for PyTorch

train:
  n_rollout_threads: 1               # Parallel environments
  num_env_steps: 10000000            # Total training steps
  episode_length: 200                # Steps per episode
  save_interval: 100                 # Save model every N episodes
  log_interval: 5                    # Log every N episodes
  use_valuenorm: True                # Value normalization
  use_linear_lr_decay: True          # Linear LR decay

model:
  hidden_sizes: [128, 256, 64]       # Network architecture
  activation_func: leaky_relu        # relu, tanh, leaky_relu
  use_recurrent_policy: True         # Use GRU for temporal dependencies
  recurrent_n: 1                     # Number of recurrent layers
  data_chunk_length: 10              # Sequence length for RNN training
  lr: 0.0005                         # Actor learning rate
  critic_lr: 0.0005                  # Critic learning rate
  opti_eps: 1.0e-5                   # Optimizer epsilon

algo:
  ppo_epoch: 5                       # PPO update epochs per rollout
  num_mini_batch: 1                  # Mini-batches per update
  clip_param: 0.2                    # PPO clip parameter ε
  entropy_coef: 0.01                 # Entropy bonus coefficient
  value_loss_coef: 1.0               # Value loss coefficient
  use_gae: True                      # Generalized Advantage Estimation
  gamma: 0.99                        # Discount factor γ
  gae_lambda: 0.95                   # GAE λ
  use_clipped_value_loss: True       # Clip value loss
  use_huber_loss: True               # Use Huber loss for critic
  huber_delta: 10.0                  # Huber loss threshold
  use_max_grad_norm: True            # Gradient clipping
  max_grad_norm: 10.0                # Gradient clip threshold

eval:
  use_eval: False                    # Enable evaluation
  eval_interval: 25                  # Evaluate every N episodes
  n_eval_rollout_threads: 1          # Parallel eval environments
  eval_episodes: 10                  # Episodes per evaluation
```

### 2. Environment Configuration

**File**: `harl/configs/envs_cfgs/a_multi_lane.yaml`

**Key Parameters**:

```yaml
# SUMO Configuration
sumo_cfg: "env_utils/SUMO_files/scenario.sumocfg"
max_num_seconds: 20                  # Episode duration (seconds)
use_gui: False                       # Show SUMO GUI

# Scenario Configuration
scene_name: "multi_lane"
edge_ids: ["edge_0", "edge_1"]       # Road edges
edge_lane_num: 3                     # Lanes per edge
calc_features_lane_ids: [0, 1, 2]    # Lanes to monitor
delta_t: 0.1                         # Simulation timestep (seconds)

# Vehicle Configuration
max_num_CAVs: 12                     # Maximum CAVs
max_num_HDVs: 18                     # Maximum HDVs
num_CAVs: 12                         # Active CAVs
num_HDVs: 18                         # Active HDVs
penetration_CAV: 0.4                 # CAV penetration rate (40%)
lane_max_num_vehs: 10                # Max vehicles per lane
warmup_steps: 100                    # Warmup before training
vehicle_action_type: "continuous"    # Action space type
use_V2V_info: True                   # Vehicle-to-vehicle communication

# Observation Configuration
strategy: "improve"                  # Feature extraction strategy
use_hist_info: True                  # Use historical trajectories
hist_length: 10                      # Trajectory history length

# Reward Configuration
reward_weights:
  safety: 1.0                        # FIXED (always 1.0)
  efficiency: 0.4
  stability: 0.0
  comfort: 0.3

# Dynamic Reward Weighting (optional)
use_dynamic_weight: False            # Enable dynamic weight generation
weight_update_frequency: 50          # Update weights every N steps
weight_ema_decay: 0.85               # Exponential moving average
epsilon_weight: 0.01                 # Variance regularization

# Safety Parameters
parameters:
  max_v: 20                          # Max velocity (m/s)
  max_a: 8                           # Max acceleration (m/s²)
  max_decel: -8                      # Max deceleration (m/s²)
  max_lane_change_v: 15              # Max velocity for lane change
  safe_warn_threshold:
    TTC: 1.5                         # Time-to-collision warning (s)
    ACT: 2.0                         # Accepted crash time (s)
    spacing: 3.0                     # Spacing warning (m)
  crash_threshold:
    TTC: 0.5                         # Collision threshold (s)
    spacing: 1.5                     # Collision threshold (m)

# Multi-Scene Configuration
num_scene_classes: 5                 # Number of traffic scenarios
scene_specify: False                 # Force specific scene
scene_id: 0                          # Scene ID if specified
scene_centers:                       # K-means clustering centers
  class_0: {v_mean: 14.486, density: 44.600}  # Free flow
  class_1: {v_mean: 10.647, density: 56.291}  # Transition
  class_2: {v_mean: 8.669, density: 65.223}   # Moderate congestion
  class_3: {v_mean: 4.627, density: 86.506}   # Heavy congestion
  class_4: {v_mean: 2.134, density: 102.345}  # Gridlock

# Evaluation Configuration
is_evaluation: False                 # Evaluation mode
save_model_name: "seed-42"           # Model checkpoint name
eval_weights:
  w1: 0.3                            # Safety weight
  w2: 0.3                            # Efficiency weight
  w3: 0.3                            # Stability weight
  w4: 0.1                            # Comfort weight

# Gradient Monitoring (for debugging)
enable_gradient_monitoring: False
gradient_clip_threshold: 10.0
monitor_frequency: 100
```

### 3. Loading Configuration

```python
# Method 1: From YAML defaults
python train.py --algo mappo --env a_multi_lane --exp_name test01

# Method 2: From JSON config
python train.py --load_config results/experiment_01/config.json --exp_name resume

# Method 3: With command-line overrides
python train.py --algo mappo --env a_multi_lane \
    --exp_name test02 \
    --num_env_steps 5000000 \
    --episode_length 300 \
    --lr 0.0003 \
    --use_gui True
```

---

## 📝 Code Conventions

### 1. File & Naming Conventions

**File Names**:
- Python files: `snake_case.py` (e.g., `veh_env_wrapper.py`)
- Config files: `lowercase.yaml` (e.g., `mappo.yaml`)
- Documentation: `UPPERCASE.md` or `PascalCase.md`

**Class Names**:
- PascalCase: `MultiLaneEnv`, `StochasticPolicy`, `FeatureExtractor`

**Function Names**:
- snake_case: `extract_features()`, `make_decision()`, `compute_reward()`

**Variables**:
- snake_case: `num_vehicles`, `observation_space`, `reward_weights`
- Abbreviations allowed: `obs`, `rew`, `act`, `cent_obs`

**Constants**:
- UPPERCASE: `MAX_VEHICLES`, `DEFAULT_LR`, `HORIZON`

### 2. Type Annotations

**Recommended** (newer modules follow this):
```python
from typing import Dict, List, Tuple, Optional
import numpy as np

def extract_features(
    state: Dict[str, np.ndarray],
    vehicle_ids: List[str],
    lane_statistics: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for prediction.

    Args:
        state: Vehicle states {vehicle_id: [x, y, v, ...]}
        vehicle_ids: List of vehicle IDs to process
        lane_statistics: Lane-level statistics

    Returns:
        env_features: [N, 4] array
        ind_features: [N, 10] array
        vehicle_types: [N, 1] array
    """
    ...
```

**Current State**:
- ✓ New modules (`feature_extractor.py`, `prediction_module.py`): Full type hints
- ⚠️ Older modules (`veh_env_wrapper.py`): Minimal or no type hints
- ✓ Models/algorithms: Partial type hints

### 3. Documentation Style

**Docstrings**: Mix of English and Chinese (Chinese for implementation details)

```python
def predict_intentions(self, vehicle_ids, state, lane_statistics):
    """
    批量预测车辆换道意图 (Batch predict lane-change intentions)

    Args:
        vehicle_ids: 需要预测的车辆ID列表 (List of vehicle IDs)
        state: 环境状态字典 (Environment state dict)
        lane_statistics: 车道统计信息 (Lane statistics)

    Returns:
        intentions: {vehicle_id: probability} 字典 (dict)
    """
```

**Comments**:
- English for public APIs and high-level logic
- Chinese for implementation details and edge cases
- Code examples in module `__main__` blocks

### 4. Error Handling

**Graceful Degradation**:
```python
# Return default values for edge cases
if len(vehicle_ids) == 0:
    return {}

# Handle missing data
try:
    decision, prob = self.make_decision(features)
except (ValueError, KeyError) as e:
    logger.warning(f"Decision failed: {e}, using default")
    decision, prob = 0, None  # Keep lane
```

**Validation**:
```python
# Validate inputs
assert len(env_features) == len(vehicle_ids), "Feature-ID mismatch"
assert 0 <= alpha <= 1, "Alpha must be in [0, 1]"
```

### 5. Factory Pattern

**Used extensively for model creation**:

```python
# Prediction factory
from harl.prediction.intention.strategy.SCM_prediction.execute_prediction import SCMPredictorFactory

predictor = SCMPredictorFactory.create_predictor(
    model_type="shallow_hierarchical",
    device="cuda"
)

# Decision factory
from harl.decisions.lateral_decisions.SCM_decisions.execute_decision import SCMDecisionMakerFactory

decision_maker = SCMDecisionMakerFactory.create_decision_maker(
    model_type="scm_decision",
    freeze_base_model=True
)

# Runner registry
from harl.runners import RUNNER_REGISTRY

runner = RUNNER_REGISTRY[algo_name](args, algo_args, env_args)
```

### 6. Logging Conventions

```python
# Use loguru or print with prefixes
print(f"[FeatureExtractor] Extracted {len(features)} features")
print(f"[PredictionModule] Intention prediction: {np.mean(probs):.3f}")
print(f"[DecisionModule] Stage {stage}: LR={lr:.2e}")

# TensorBoard logging (in logger classes)
self.writer.add_scalar('Loss/Actor', actor_loss, episode)
self.writer.add_scalar('Reward/Mean', mean_reward, episode)
```

---

## 🛠️ Common Tasks

### 1. Run Training

**Basic Training**:
```bash
# Single experiment
python train.py --algo mappo --env a_multi_lane --exp_name experiment_01

# With custom parameters
python train.py --algo mappo --env a_multi_lane \
    --exp_name exp_02 \
    --num_env_steps 5000000 \
    --episode_length 300 \
    --lr 0.0003 \
    --num_CAVs 8 \
    --penetration_CAV 0.3
```

**Resume Training**:
```bash
# Load from saved config
python train.py --load_config results/exp_01/config.json --exp_name exp_01_resume
```

**Multi-Policy Training**:
```bash
# Scene-aware multi-policy MAPPO
python train.py --algo mpmappo --env a_multi_lane --exp_name multi_policy_01
```

**Batch Training**:
```bash
# Run multiple experiments
cd examples/batch_training
python run_batch_training.py --config batch_config.yaml
```

### 2. Modify Reward Function

**Edit**: `harl/configs/envs_cfgs/a_multi_lane.yaml`

```yaml
reward_weights:
  safety: 1.0        # Keep at 1.0 (fixed)
  efficiency: 0.5    # Increase efficiency priority
  stability: 0.2     # Add stability reward
  comfort: 0.3       # Comfort reward
```

Or **override via command-line**:
```bash
python train.py --algo mappo --env a_multi_lane \
    --reward_weights.efficiency 0.6 \
    --reward_weights.stability 0.1
```

**Enable Dynamic Weighting**:
```yaml
use_dynamic_weight: True
weight_update_frequency: 50
weight_ema_decay: 0.85
epsilon_weight: 0.01
```

### 3. Change Network Architecture

**Actor/Critic Networks** - Edit `harl/configs/algos_cfgs/mappo.yaml`:

```yaml
model:
  hidden_sizes: [256, 512, 128]      # Increase capacity
  activation_func: relu              # Change activation
  use_recurrent_policy: True         # Enable GRU
  recurrent_n: 2                     # 2-layer GRU
```

**Use HistoricalVNet** (recommended):

Edit `harl/configs/envs_cfgs/a_multi_lane.yaml`:
```yaml
critic_type: "historical_v_net"      # Use full GAT + CrossAttention
```

### 4. Adjust PPO Hyperparameters

**Edit**: `harl/configs/algos_cfgs/mappo.yaml`

```yaml
algo:
  ppo_epoch: 10              # More update epochs (default: 5)
  clip_param: 0.1            # Smaller clip for stability (default: 0.2)
  entropy_coef: 0.02         # More exploration (default: 0.01)
  lr: 0.0003                 # Lower learning rate (default: 0.0005)
  gamma: 0.995               # Longer horizon (default: 0.99)
  gae_lambda: 0.97           # More bias towards value (default: 0.95)
```

### 5. Enable/Disable Modules

**Prediction Module**:
```python
# In veh_env_wrapper.py or multilane_env.py
use_prediction = True           # Enable SCM + CQR prediction
prediction_horizon = 30         # 3.0 seconds
```

**Decision Module Fine-Tuning**:
```python
# In model_updater.py
enable_fine_tuning = True       # Enable progressive fine-tuning
update_frequency = 10           # Update every 10 episodes
stage_boundaries = [1000, 2000] # Stage transitions
```

**Safety Assessment**:
```python
# In safety_assessment.py
enable_safety_penalty = True    # Add collision risk penalty
lambda_penalty = 1.0            # Penalty weight
```

### 6. Add New Algorithm

**Step 1**: Implement algorithm in `harl/algorithms/actors/`

```python
# harl/algorithms/actors/my_algo.py
from .on_policy_base import OnPolicyBase

class MyAlgo(OnPolicyBase):
    def update(self, sample):
        # Implement your update logic
        ...
```

**Step 2**: Create runner in `harl/runners/`

```python
# harl/runners/my_algo_runner.py
from .on_policy_base_runner import OnPolicyBaseRunner

class MyAlgoRunner(OnPolicyBaseRunner):
    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        # Initialize your algorithm
```

**Step 3**: Register in `harl/runners/__init__.py`

```python
from .my_algo_runner import MyAlgoRunner

RUNNER_REGISTRY = {
    "mappo": OnPolicyMARunner,
    "mpmappo": MultiPolicyMARunner,
    "my_algo": MyAlgoRunner,  # Add here
}
```

**Step 4**: Create config in `harl/configs/algos_cfgs/my_algo.yaml`

**Step 5**: Run
```bash
python train.py --algo my_algo --env a_multi_lane --exp_name test
```

### 7. Debug Training

**Enable Logging**:
```yaml
# In mappo.yaml
train:
  log_interval: 1              # Log every episode
  enable_gradient_monitoring: True
  gradient_clip_threshold: 10.0
```

**Monitor Gradients**:
```python
# In runner or algorithm
for name, param in self.actor.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"[Gradient] {name}: {grad_norm:.4f}")
```

**Visualize with TensorBoard**:
```bash
tensorboard --logdir results/exp_name/logs
```

### 8. Evaluate Trained Model

**Set Evaluation Mode**:
```yaml
# In a_multi_lane.yaml
is_evaluation: True
save_model_name: "seed-42"      # Model to load

eval_weights:
  w1: 0.25    # Safety
  w2: 0.25    # Efficiency
  w3: 0.25    # Stability
  w4: 0.25    # Comfort
```

**Run Evaluation**:
```bash
python train.py --algo mappo --env a_multi_lane \
    --exp_name eval_01 \
    --is_evaluation True \
    --save_model_name "best_model"
```

### 9. Analyze Results

**Use Results Analyzer**:
```bash
cd examples/batch_training
python results_analyzer.py --results_dir ../../results/exp_name
```

**Manual Analysis**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training logs
logs = pd.read_csv('results/exp_name/logs/progress.csv')

# Plot rewards
plt.plot(logs['episode'], logs['mean_reward'])
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.show()
```

---

## ⚠️ Known Issues & Fixes

### 1. CQR Import Error ✅ FIXED

**Issue**: `ModuleNotFoundError` when loading CQR models

**Location**: `harl/prediction/occupancy/strategy/CQR_prediction/models.py:25`

**Original Code** (WRONG):
```python
from feature_encoder import FeatureEncoder
```

**Fixed Code**:
```python
from .feature_encoder import FeatureEncoder
```

**Status**: ✅ Fixed in latest version

---

### 2. VNet Architecture Incomplete ✅ FIXED

**Issue**: `v_net.py` uses simple MLP, missing GAT and CrossAttention

**Location**: `harl/models/value_function_models/v_net.py`

**Problem**: Design document requires:
- Historical trajectory encoder (MLP)
- Graph Attention Network (GAT) for vehicle interactions
- Traffic feature encoder (MLP)
- Cross-Attention fusion
- Value output head

**Solution**: Use `HistoricalVNet` instead

**File**: `harl/models/value_function_models/historical_v_net.py` ✅

**Configuration**:
```yaml
# In a_multi_lane.yaml
critic_type: "historical_v_net"
```

**Status**: ✅ Implemented and verified

---

### 3. Model Updater Parameter Freezing ✅ FIXED

**Issue**: `_apply_stage_config()` missing parameter freeze/unfreeze logic

**Location**: `harl/envs/a_multi_lane/env_utils/model_updater.py`

**Fixed Methods** in `harl/decisions/lateral_decisions/SCM_decisions/execute_decision.py`:

```python
def _freeze_parameters(self):
    """Freeze SCM base model, keep decision_threshold trainable."""
    if self.freeze_base_model:
        for param in self.model.scm_model.parameters():
            param.requires_grad = False
        self.model.decision_threshold.requires_grad = True

def _unfreeze_parameters(self):
    """Unfreeze all parameters."""
    for param in self.model.parameters():
        param.requires_grad = True
    self.freeze_base_model = False
```

**Status**: ✅ Implemented and verified

---

### 4. Actor Dynamic Weight Mechanism ⚠️ NEEDS VERIFICATION

**Issue**: `weight_head` defined but may not be called in `forward()`

**Location**: `harl/models/policy_models/stochastic_policy.py:60-75`

**Action Required**:
1. Check `forward()` method for `self.weight_head` usage
2. Verify `use_dynamic_weight=True` activates weight generation
3. Ensure weights are applied to multi-objective rewards

**Verification**:
```bash
grep -A 50 "def forward" harl/models/policy_models/stochastic_policy.py | grep "weight_head"
```

---

### 5. Feature Extraction Integration ⚠️ NEEDS VERIFICATION

**Issue**: Unclear if `feature_extractor` is called in environment `step()`

**Location**: `harl/envs/a_multi_lane/multilane_env.py` or `veh_env_wrapper.py`

**Expected Integration**:
```python
# In env.step()
env_features, ind_features, vehicle_types = self.feature_extractor.batch_extract_intention_features(...)
intention_probs = self.prediction_module.predict_intentions(...)
decisions = self.decision_module.make_batch_decisions(...)
```

**Action Required**: Verify call chain in environment step loop

---

### 6. Safety Assessment Integration ⚠️ NEEDS VERIFICATION

**Issue**: Unclear if `safety_assessment.py` penalties are added to rewards

**Location**: Reward computation in `multilane_env.py` or `veh_env_wrapper.py`

**Expected Integration**:
```python
# In reward computation
safety_penalty = self.safety_assessor.compute_safety_penalty(
    ego_trajectory=planned_traj,
    predicted_occupancies=occupancy_pred,
    lambda_penalty=1.0
)
rewards[vehicle_id] -= safety_penalty
```

**Action Required**: Verify safety penalty integration

---

### 7. Logging Metrics Incomplete ⚠️ PARTIAL

**Issue**: Decision model fine-tuning and CQR calibration metrics not logged

**Location**: `harl/envs/a_multi_lane/multilane_logger.py`

**Missing**:
- Decision model loss breakdown (efficiency, safety)
- CQR calibration coverage (HDV vs CAV)
- Stage transition logs

**Recommended Addition**:
```python
def log_decision_update(self, episode, decision_loss, lr, stage):
    self.decision_update_history.append({
        'episode': episode,
        'loss': decision_loss,
        'learning_rate': lr,
        'stage': stage
    })
    if self.tensorboard_writer:
        self.tensorboard_writer.add_scalar('Decision/Loss', decision_loss, episode)

def log_cqr_calibration(self, episode, hdv_coverage, cav_coverage):
    if self.tensorboard_writer:
        self.tensorboard_writer.add_scalar('CQR/HDV_Coverage', hdv_coverage, episode)
        self.tensorboard_writer.add_scalar('CQR/CAV_Coverage', cav_coverage, episode)
```

---

## 🧪 Testing & Debugging

### 1. Module Self-Tests

Most modules include test code in `__main__` blocks:

**Test Feature Extractor**:
```bash
python -m harl.envs.a_multi_lane.env_utils.feature_extractor
```

**Test Prediction Module**:
```bash
python -m harl.envs.a_multi_lane.env_utils.prediction_module
```

**Test Decision Module**:
```bash
python -m harl.envs.a_multi_lane.env_utils.decision_module
```

### 2. Import Verification

**Test CQR Import** (should pass now):
```bash
python -c "from harl.prediction.occupancy.strategy.CQR_prediction.models import MLPQR, GRUQR; print('✓ CQR import successful')"
```

**Test HistoricalVNet**:
```bash
python -c "
import torch
from harl.models.value_function_models.historical_v_net import HistoricalVNet
args = {'hidden_sizes': [256, 256], 'initialization_method': 'orthogonal', 'num_vehicles': 15, 'num_lanes': 3, 'hist_steps': 20, 'state_dim': 6, 'traffic_dim': 6}
model = HistoricalVNet(args, None, device=torch.device('cpu'))
print('✓ HistoricalVNet initialization successful')
"
```

### 3. Small-Scale Training Test

**Quick Validation** (100 episodes):
```bash
python train.py --algo mappo --env a_multi_lane \
    --exp_name quick_test \
    --num_env_steps 20000 \
    --episode_length 200 \
    --num_CAVs 3 \
    --num_HDVs 6 \
    --use_gui False
```

### 4. Debug Environment

**Enable SUMO GUI**:
```bash
python train.py --algo mappo --env a_multi_lane \
    --exp_name debug \
    --use_gui True
```

**Print State Information**:
```python
# In veh_env_wrapper.py step()
print(f"[DEBUG] Step {self.step_count}")
print(f"  CAVs: {list(self.CAV_dict.keys())}")
print(f"  HDVs: {list(self.HDV_dict.keys())}")
print(f"  Observations: {obs.shape}")
print(f"  Rewards: {rewards}")
```

### 5. Check Model Parameters

**Verify Parameter Freezing**:
```python
from harl.decisions.lateral_decisions.SCM_decisions.execute_decision import SCMDecisionMaker

decision_maker = SCMDecisionMaker(freeze_base_model=True)
decision_maker._freeze_parameters()

# Check frozen status
for name, param in decision_maker.model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Expected:
# - scm_model.* : requires_grad=False
# - decision_threshold: requires_grad=True
```

### 6. Monitor Training

**TensorBoard**:
```bash
# Start TensorBoard
tensorboard --logdir results/exp_name/logs

# Navigate to http://localhost:6006
```

**Real-Time Monitoring**:
```bash
# In separate terminal
watch -n 5 'tail -20 results/exp_name/logs/progress.txt'
```

### 7. Profile Performance

**Time Environment Steps**:
```python
import time

start = time.time()
for _ in range(100):
    obs, rew, done, info = env.step(actions)
elapsed = time.time() - start
print(f"Average step time: {elapsed/100*1000:.2f} ms")
```

**GPU Utilization**:
```bash
watch -n 1 nvidia-smi
```

---

## 📚 Important Files Reference

### Documentation

| File | Description |
|------|-------------|
| `PROJECT_AUDIT_REPORT.md` | Comprehensive audit (510 lines, identifies 6 issues) |
| `AUDIT_FIX_VERIFICATION.md` | Fix verification report (840 lines) |
| `train_execution.md` | Training execution guide |
| `harl/envs/a_multi_lane/README_GAME_THEORY.md` | Game theory concepts |
| `harl/envs/a_multi_lane/README_SAFETY_FIELD.md` | Safety field explanation |
| `harl/envs/a_multi_lane/env_utils/INTEGRATION_GUIDE.md` | Integration guide |
| `harl/envs/a_multi_lane/env_utils/USAGE_GUIDE.md` | Usage instructions |

### Entry Points

| File | Purpose |
|------|---------|
| `train.py` | Main training script (102 lines) |
| `examples/train.py` | Alternative training entry |
| `examples/batch_training/run_batch_training.py` | Batch experiments |

### Core Environment

| File | Lines | Purpose |
|------|-------|---------|
| `harl/envs/a_multi_lane/multilane_env.py` | ~300 | Gym wrapper |
| `harl/envs/a_multi_lane/env_utils/veh_env.py` | ~800 | SUMO interface |
| `harl/envs/a_multi_lane/env_utils/veh_env_wrapper.py` | 2000+ | Main environment logic |
| `harl/envs/a_multi_lane/env_utils/feature_extractor.py` | ~400 | Feature extraction ✓ |
| `harl/envs/a_multi_lane/env_utils/prediction_module.py` | ~300 | Prediction integration ✓ |
| `harl/envs/a_multi_lane/env_utils/decision_module.py` | ~250 | Decision integration ✓ |
| `harl/envs/a_multi_lane/env_utils/lateral_controller.py` | ~200 | Trajectory planning ✓ |
| `harl/envs/a_multi_lane/env_utils/safety_assessment.py` | ~300 | Collision risk ✓ |
| `harl/envs/a_multi_lane/env_utils/model_updater.py` | ~200 | Fine-tuning strategy ✓ |

### Algorithms

| File | Lines | Purpose |
|------|-------|---------|
| `harl/algorithms/actors/mappo.py` | ~200 | MAPPO actor updates |
| `harl/algorithms/critics/v_critic.py` | ~150 | Critic updates |
| `harl/runners/on_policy_base_runner.py` | 1279 | Base training loop |
| `harl/runners/on_policy_ma_runner.py` | ~400 | MAPPO runner |

### Neural Networks

| File | Lines | Purpose |
|------|-------|---------|
| `harl/models/policy_models/stochastic_policy.py` | ~300 | Actor network |
| `harl/models/value_function_models/v_net.py` | ~100 | Simple V-net |
| `harl/models/value_function_models/historical_v_net.py` | ~400 | Full V-net ✓ |
| `harl/models/base/multi_lane/improved_encoder.py` | ~200 | Multi-lane encoder |

### Prediction & Decision

| File | Lines | Purpose |
|------|-------|---------|
| `harl/prediction/intention/strategy/SCM_prediction/scm_model_v2.py` | ~300 | SCM model |
| `harl/prediction/occupancy/strategy/CQR_prediction/models.py` | ~400 | CQR models ✓ |
| `harl/decisions/lateral_decisions/SCM_decisions/scm_decision_model.py` | ~200 | Decision model |
| `harl/decisions/lateral_decisions/SCM_decisions/execute_decision.py` | ~300 | Decision interface ✓ |

### Configuration

| File | Purpose |
|------|---------|
| `harl/configs/algos_cfgs/mappo.yaml` | MAPPO hyperparameters |
| `harl/configs/algos_cfgs/mpmappo.yaml` | Multi-policy MAPPO |
| `harl/configs/envs_cfgs/a_multi_lane.yaml` | Environment configuration |
| `harl/utils/configs_tools.py` | Config loading/merging |

---

## 🚀 Quick Start Checklist

### For First-Time Users

- [ ] **Read Project Overview** (this document)
- [ ] **Check Known Issues** (#1, #2, #3 are fixed ✅)
- [ ] **Verify Dependencies**: PyTorch, SUMO, numpy, etc.
- [ ] **Test Imports**: Run import verification commands
- [ ] **Run Small Test**: 100-episode quick validation
- [ ] **Review Configurations**: Understand YAML structure
- [ ] **Read Audit Reports**: `PROJECT_AUDIT_REPORT.md`, `AUDIT_FIX_VERIFICATION.md`

### For Development

- [ ] **Follow Code Conventions**: Type hints, docstrings, factory pattern
- [ ] **Test Modules**: Use `__main__` blocks for self-tests
- [ ] **Update Configurations**: Modify YAML, not hardcoded values
- [ ] **Log Properly**: Use prefixes, TensorBoard integration
- [ ] **Document Changes**: Update relevant READMEs
- [ ] **Verify Integration**: Ensure modules connect properly

### For Debugging

- [ ] **Enable Logging**: Set `log_interval=1`
- [ ] **Use SUMO GUI**: Set `use_gui=True`
- [ ] **Monitor Gradients**: Enable gradient monitoring
- [ ] **Check TensorBoard**: Visualize training metrics
- [ ] **Profile Performance**: Time critical operations
- [ ] **Verify Data Flow**: Print intermediate values

---

## 📞 Additional Resources

### External Documentation

- **SUMO**: https://sumo.dlr.de/docs/
- **PyTorch**: https://pytorch.org/docs/
- **PPO Algorithm**: https://arxiv.org/abs/1707.06347
- **MAPPO**: https://arxiv.org/abs/2103.01955

### Repository-Specific Guides

- **SCM Prediction**: `harl/prediction/intention/strategy/SCM_prediction/README.md`
- **CQR Prediction**: `harl/prediction/occupancy/strategy/CQR_prediction/README.md`
- **SCM Decisions**: `harl/decisions/lateral_decisions/SCM_decisions/README.md`
- **Environment Utils**: `harl/envs/a_multi_lane/env_utils/README.md`

---

## ✅ Status Summary

**Project Completion**: ~90%

**High-Priority Issues**: 3/3 Fixed ✅
- ✅ CQR import error
- ✅ VNet architecture (HistoricalVNet implemented)
- ✅ Model updater parameter freezing

**Medium-Priority Issues**: 3/3 Need Verification ⚠️
- ⚠️ Actor dynamic weight mechanism
- ⚠️ Feature extraction integration
- ⚠️ Safety assessment integration

**Recommended Next Steps**:
1. Verify actor dynamic weight mechanism (2 hours)
2. Verify feature/prediction/decision integration in environment (4 hours)
3. Add missing logging methods (2 hours)
4. Run end-to-end integration test (1 day)

---

**Last Updated**: 2025-11-14
**Version**: 1.0
**Maintained By**: AI Assistant (Claude)
