# Game Theory-Based Safety Assessment Implementation

## Overview

This implementation integrates cooperative game theory-based safety assessment into the MPC controller framework, based on **Yang et al., IEEE Transactions on Intelligent Transportation Systems, 2023**: *"Multi-Lane Coordinated Control Strategy of Connected and Automated Vehicles for On-Ramp Merging Area Based on Cooperative Game"*.

## Files

- **`safety_game.py`**: Core implementation of cooperative game-based safety metrics
- **`evaluate_flow.py`**: Integration with MPC controller framework via `GameTheoryMethod` class

## Theoretical Foundation

### Cooperative Game Framework

The implementation uses a cooperative game approach where:

1. **Players**: Ego vehicle and each surrounding vehicle
2. **Strategy Space**: `{leader, follower}` for merging scenarios
3. **Objective**: Find Nash equilibrium that minimizes joint cost
4. **Cost Function**: Multi-objective combination of:
   - **Efficiency** (ω₁ = 1.0): Velocity tracking error
   - **Fuel Consumption** (ω₂ = 4.47): Acceleration minimization
   - **Comfort** (ω₃ = 5.0): Jerk minimization

### Key Equations

#### Longitudinal Cost Function (Equation 3 from paper)

```
J_lon = (1/2) ∫[ω₁(vₓ - v_des)² + ω₂aₓ² + ω₃uₓ²]dt
```

Where:
- `vₓ`: Vehicle velocity
- `v_des`: Desired velocity
- `aₓ`: Acceleration
- `uₓ`: Jerk (acceleration rate)

#### Lateral Cost Function (Equation 4 from paper)

```
J_lat = (1/2) ∫[u_y²]dt
```

Where:
- `u_y`: Lateral jerk during lane change

### Nash Equilibrium Solution

For each ego-vehicle pair, the algorithm:
1. Constructs payoff matrix for all strategy combinations
2. Evaluates costs for each (ego_strategy, other_strategy) pair
3. Computes collision probability for each strategy pair
4. Selects Nash equilibrium: strategy pair minimizing joint cost while avoiding high collision risk

## Implementation Details

### Class Structure

```python
# Core Classes
GameParameters          # Configuration parameters
GameStrategy            # Single strategy representation
CooperativeGameSafety   # Main game solver

# Utility Functions
compute_game_based_safety_metrics()      # Main interface
generate_prediction_centers_from_game()  # MPC integration
```

### Safety Metrics Output

The `GameTheoryMethod.assess_safety()` returns:

```python
{
    'surrounding_prediction_centers': {
        vehicle_id: {
            'reference': {
                'position': [...],  # Predicted positions
                'time': [...]       # Time steps
            },
            'nash_strategy': 'leader'/'follower',
            'game_cost': float
        }
    },
    'safety_metrics': {
        'max_collision_probability': float,     # Highest collision risk
        'min_game_safety_score': float,         # Lowest safety score
        'total_game_cost': float,               # Sum of all costs
        'overall_safety_risk': float,           # 1 - min_safety_score
        'num_interactions': int,                # Number of games solved
        'game_solutions': {
            vehicle_id: {
                'ego_strategy': 'leader'/'follower',
                'other_strategy': 'leader'/'follower',
                'ego_cost': float,
                'joint_cost': float,
                'collision_probability': float
            }
        },
        'directional_risks': {
            'ego_front': float,
            'ego_rear': float,
            'left_front': float,
            'left_rear': float,
            'right_front': float,
            'right_rear': float
        }
    },
    'computation_time': float
}
```

## Parameter Configuration

### Default Game Parameters

```python
GameParameters(
    # Cost weights (from paper recommendations)
    omega1=1.0,    # Efficiency weight
    omega2=4.47,   # Fuel consumption weight
    omega3=5.0,    # Comfort weight

    # Safety parameters
    safe_headway=2.0,          # Safe time headway [s]
    min_spacing=5.0,           # Minimum safe spacing [m]
    collision_threshold=2.0,   # Collision distance threshold [m]

    # Vehicle constraints
    v_max=35.0,    # Maximum velocity [m/s]
    a_max=3.0,     # Maximum acceleration [m/s²]
    a_min=-5.0,    # Maximum deceleration [m/s²]
)
```

### Tuning Guidelines

1. **Increase ω₁ (efficiency)**: Prioritize velocity tracking
2. **Increase ω₂ (fuel)**: Encourage smoother acceleration
3. **Increase ω₃ (comfort)**: Reduce jerk for passenger comfort
4. **Decrease safe_headway**: Allow tighter following (less conservative)
5. **Increase safe_headway**: Enforce larger safety margins (more conservative)

## Usage Example

```python
from harl.envs.a_multi_lane.project_structure import SystemParameters
from evaluate_flow import GameTheoryMethod

# Initialize system parameters
system_params = SystemParameters()

# Create game theory safety method
game_method = GameTheoryMethod(system_params)

# Assess safety
safety_result = game_method.assess_safety(
    ego_state=ego_state,
    surrounding_state=surrounding_state,
    surrounding_vehicles=surrounding_vehicles,
    lateral_decision='keep_lane'  # or 'pre_change_left', 'pre_change_right'
)

# Access results
max_collision_prob = safety_result['safety_metrics']['max_collision_probability']
game_solutions = safety_result['safety_metrics']['game_solutions']
prediction_centers = safety_result['surrounding_prediction_centers']
```

## Integration with MPC Controller

The game theory method integrates seamlessly with the MPC controller:

```python
# In evaluate_flow.py
mpc_wrapper = CooperativeMPCWrapper(
    mpc_params,
    system_params,
    safety_method=GameTheoryMethod(system_params)
)

# Controller uses game-based predictions
next_speed, control_info = mpc_wrapper.control(
    ego_state, surrounding_state, surrounding_vehicles,
    lateral_decision, traffic_state, weights
)
```

## Comparison with Other Methods

| Aspect | Game Theory | Conflict Assessment | Safety Field |
|--------|-------------|---------------------|--------------|
| **Foundation** | Nash equilibrium | Trajectory prediction | Potential field |
| **Paper** | Yang et al. 2023 | Custom implementation | Jing et al. 2022 |
| **Prediction** | Game-strategy based | Monte Carlo sampling | Constant velocity |
| **Cost Model** | Multi-objective | Probability-based | Energy-based |
| **Cooperation** | Explicit (game payoff) | Implicit (envelope) | None (field only) |
| **Computation** | Game solving | MC simulation | Field integration |
| **Strength** | Strategic reasoning | Uncertainty modeling | Real-time efficiency |
| **Use Case** | Merging scenarios | General interaction | Dense traffic |

## Key Advantages

1. **Strategic Reasoning**: Explicitly models vehicle cooperation through game theory
2. **Multi-Objective**: Balances efficiency, fuel, and comfort simultaneously
3. **Collision Avoidance**: Built into payoff matrix via high-risk penalties
4. **Nash Equilibrium**: Mathematically optimal solution for cooperative scenarios
5. **Interpretable**: Strategy outputs ('leader'/'follower') are human-understandable

## Limitations

1. **Computational Cost**: Solving multiple games per timestep
2. **Strategy Space**: Currently limited to {leader, follower}
3. **Prediction Simplification**: Uses strategy-adjusted constant velocity
4. **Two-Player Games**: Does not model multi-vehicle interactions simultaneously

## Performance Metrics

The evaluation framework tracks:
- **Computation Time**: Time to solve all games
- **Collision Probability**: Maximum risk across all interactions
- **Game Cost**: Total cost from all Nash equilibria
- **Safety Score**: Overall safety rating (0-1)

## Validation

The implementation has been validated against:
1. **Paper Examples**: Reproduces scenarios from Yang et al. 2023
2. **Nash Equilibrium Properties**: Verified stability of solutions
3. **Collision Avoidance**: Tested in high-risk scenarios
4. **MPC Integration**: Confirmed compatibility with cooperative controller

## Future Extensions

Possible improvements:
1. **N-Player Games**: Simultaneous multi-vehicle game solving
2. **Extended Strategy Space**: Add 'aggressive', 'defensive' strategies
3. **Dynamic Payoff Weights**: Adapt ω₁, ω₂, ω₃ based on traffic conditions
4. **Repeated Games**: Model long-term interaction effects
5. **Mixed Strategies**: Support probabilistic strategy selection

## References

**Primary Paper:**
Yang, L., et al. (2023). "Multi-Lane Coordinated Control Strategy of Connected and Automated Vehicles for On-Ramp Merging Area Based on Cooperative Game." *IEEE Transactions on Intelligent Transportation Systems*.

**Related Work:**
- Nash, J. (1950). "Equilibrium points in n-person games."
- Başar, T., & Olsder, G. J. (1998). "Dynamic Noncooperative Game Theory."

## Contact

For questions or issues related to this implementation, please refer to the main project documentation.

---

**Implementation Date**: 2025
**Last Updated**: 2025-10-08
