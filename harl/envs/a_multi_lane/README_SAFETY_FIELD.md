# Driving Safety Field Implementation

## Overview

This implementation provides a **Driving Safety Field** (DSF) based safety assessment method for cooperative MPC controllers in multi-lane autonomous vehicle scenarios. The implementation is based on the paper:

> **"Integrated Longitudinal and Lateral Hierarchical Control of Cooperative Merging of Connected and Automated Vehicles at On-Ramps"**
> Jing et al., IEEE Transactions on Intelligent Transportation Systems, Vol. 23, No. 12, December 2022

## Key Concepts

The Driving Safety Field integrates two components to comprehensively describe driving risk:

### 1. **Safety Potential Energy (SE)**
   - Represents the **spatial variability** of driving risk
   - Based on relative positions and velocities of surrounding vehicles
   - Computed using kinetic field strength

### 2. **Rate of Change of SE (SE_dot)**
   - Represents the **temporal variability** of driving risk
   - Captures dynamic changes in the safety situation
   - Based on relative velocity differences

### 3. **Driving Safety Index (DSI)**
   - **Combined metric**: `DSI = γ × SE + (1 - γ) × SE_dot`
   - Where `γ = 0.06` (weight between spatial and temporal components)
   - Higher DSI indicates higher collision risk

## Mathematical Formulation

### Kinetic Field Strength
```
E_V,i1(t) = (k1 × Ri × mi × k3) / ((k3 - |vi(t)| × cos(αi(t))) × |ri1(t)|^k2)
```

### Safety Potential Energy
```
SE_V,i1(t) = (k1 × Ri × R1 × mi × m1 × k3) / ((k2-1) × |r21(t)|^(k2-1)) ×
             [(k3 - |vi(t)| × cos(αi))^(1-k2) - k3^(1-k2)]
```

### Weighting Factor (Lateral Influence)
```
H_i(t) = min[(Lw / (2 × Di(t)))^k4, 1]
```

### Total Safety Metrics
```
SE_1(t) = Σ(Hi × SE_V,i1(t))
SE_dot_1(t) = Σ(Hi × SE_dot_i1(t))
DSI_1(t) = γ × SE_1(t) + (1 - γ) × SE_dot_1(t)
```

## Implementation Structure

### File Organization

```
hierarchical_controller/
├── safety_field.py                  # Core DSF implementation
├── mpc_cooperative_controller.py    # MPC controller (uses safety metrics)
└── ...

evaluate_flow.py                     # Evaluation framework with pluggable safety methods
```

### Core Classes

#### 1. `SafetyFieldParameters` (Dataclass)
Configuration parameters for DSF calculation:
- `R_i`: Road condition influence factor (default: 1.0)
- `m_i`: Virtual mass [kg] (default: 1723.0)
- `k1`, `k2`, `k3`, `k4`: Undetermined constants
- `lane_width`: Lane width [m] (default: 3.2)
- `gamma`: Temporal weight (default: 0.06)

#### 2. `DrivingSafetyField`
Main calculator class with methods:
- `compute_kinetic_field_strength()`: Calculate field strength from vehicle interactions
- `compute_safety_potential_energy()`: Calculate spatial risk component
- `compute_weighting_factor()`: Calculate lateral distance weighting
- `compute_safety_energy_rate()`: Calculate temporal risk component
- `compute_driving_safety_index()`: Calculate overall DSI
- `compute_directional_safety_index()`: Calculate DSI by direction group
- `compute_collision_risk_probability()`: Convert DSI to risk probability [0,1]

#### 3. `SafetyFieldMethod` (in evaluate_flow.py)
Integration class that:
- Implements the `SafetyAssessmentMethod` interface
- Computes DSI for surrounding vehicles
- Generates simplified prediction centers for MPC
- Packages safety metrics for evaluation

## Integration with Cooperative MPC

The safety field method integrates with the MPC controller through the evaluation framework:

```python
# 1. Safety Field assesses risk
safety_result = safety_field_method.assess_safety(
    ego_state, surrounding_state, surrounding_vehicles, lateral_decision
)

# 2. MPC uses prediction centers for collision avoidance
mpc_result = mpc_controller.solve_mpc(
    current_state=ego_state,
    reference_speed=target_speed,
    target_y=target_y,
    weights=weights,
    surrounding_prediction_centers=safety_result['surrounding_prediction_centers']
)
```

### Output Structure

```python
{
    'surrounding_prediction_centers': {
        'EF_1': {
            'reference': {'position': [...], 'time': [...]},
            'current_state': VehicleState(...),
            'dsi_contribution': 0.xx
        },
        ...
    },
    'safety_metrics': {
        'overall_dsi': float,                    # Overall DSI value
        'overall_risk_probability': float,       # Risk probability [0,1]
        'directional_dsi': {...},               # DSI by direction
        'directional_risks': {...},             # Risk probabilities by direction
        'relevant_dsi': float,                  # DSI for current maneuver
        'total_SE': float,                      # Spatial risk component
        'total_SE_dot': float,                  # Temporal risk component
        'vehicle_contributions': {...}           # Individual vehicle contributions
    },
    'computation_time': float
}
```

## Usage Example

### Basic Usage

```python
from harl.envs.a_multi_lane.hierarchical_controller.safety_field import (
    DrivingSafetyField, SafetyFieldParameters, compute_safety_field_metrics
)
from harl.envs.a_multi_lane.project_structure import SystemParameters, VehicleState, VehicleType

# Initialize
system_params = SystemParameters()

# Create ego and surrounding vehicle states
ego_state = VehicleState(
    x=100.0, y=0.0, v=20.0, theta=90.0, a=0.0, omega=0.0,
    lane_id=1, vehicle_type=VehicleType.CAV, front_v=20.0, front_spacing=30.0
)

surrounding_vehicles = {
    'EF_1': VehicleState(...),
    'ER_1': VehicleState(...),
    ...
}

# Compute safety metrics
safety_metrics = compute_safety_field_metrics(
    ego_state, surrounding_vehicles, system_params
)

# Access results
print(f"Overall DSI: {safety_metrics['overall_dsi']['DSI']:.4f}")
print(f"Risk Probability: {safety_metrics['overall_risk_probability']:.4f}")
```

### Integration with Evaluation Framework

```python
# Run evaluation with Safety Field method
python evaluate_flow.py

# Or programmatically:
from evaluate_flow import run_evaluation
import yaml

with open("a_multi_lane.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

run_evaluation(config, safety_method_name="SafetyField")
```

## Comparison with Other Methods

| Method | Approach | Spatial Risk | Temporal Risk | Trajectory Prediction |
|--------|----------|--------------|---------------|----------------------|
| **ConflictAssessment** | Envelope-based | ✓ (via envelopes) | ✓ (via probability) | ✓ (explicit) |
| **SafetyField** | Field-based | ✓ (SE) | ✓ (SE_dot) | ✗ (implicit via DSI) |
| **GameTheory** | Optimization | TBD | TBD | TBD |
| **APF** | Force-based | TBD | TBD | TBD |

## Advantages of Safety Field Method

1. **Computational Efficiency**: No explicit trajectory prediction required
2. **Unified Risk Metric**: Single DSI value integrates spatial and temporal risk
3. **Physically Interpretable**: Based on field theory concepts
4. **Directional Awareness**: Can compute risk by lane/direction
5. **Vehicle Size Integration**: Through weighting factors
6. **Motion State Consideration**: Velocity and acceleration included

## Parameters Tuning

Key parameters that affect DSI calculation:

| Parameter | Default | Effect | Tuning Guidance |
|-----------|---------|--------|-----------------|
| `k1` | 0.5 | Field strength scaling | ↑ increases overall risk sensitivity |
| `k2` | 1.0 | Distance exponent | ↑ makes risk decay faster with distance |
| `k3` | 45 m/s | Wave speed | Should match max vehicle speed |
| `k4` | 1.0 | Lateral weighting | ↑ increases lateral distance effect |
| `gamma` | 0.06 | Spatial vs temporal | ↑ emphasizes spatial risk over temporal |
| `m_i` | 1723 kg | Virtual mass | Vehicle-specific (car mass) |

## Performance Metrics

The evaluation framework tracks:
- `safety_computation_time`: Time to compute DSI (ms)
- `mpc_solve_time`: Time for MPC optimization (ms)
- `control_success_rate`: Percentage of feasible MPC solutions
- `overall_dsi`: Average DSI across all vehicles and timesteps
- `collision_events`: Number of collisions (should be 0)

Results are saved to:
- `evaluation_results/SafetyField_{timestamp}/results_summary.json`
- `evaluation_results/SafetyField_{timestamp}/performance_metrics.csv`

## Future Enhancements

1. **Adaptive Parameters**: Learn `k1`-`k4` from historical data
2. **Multi-Modal Risk**: Extend to handle different vehicle types/behaviors
3. **Uncertainty Quantification**: Add confidence intervals for DSI
4. **Real-Time Optimization**: Use DSI gradient for reactive control
5. **Machine Learning Integration**: Use DSI as reward/cost in RL

## References

1. Jing et al., "Integrated Longitudinal and Lateral Hierarchical Control of Cooperative Merging of Connected and Automated Vehicles at On-Ramps," IEEE TITS, 2022
2. Wang et al., "The driving safety field based on driver–vehicle–road interactions," IEEE TITS, 2015
3. Wang et al., "Driving safety field theory modeling and its application in pre-collision warning system," Transportation Research Part C, 2016

## Citation

If you use this implementation, please cite:

```bibtex
@article{jing2022integrated,
  title={Integrated Longitudinal and Lateral Hierarchical Control of Cooperative Merging of Connected and Automated Vehicles at On-Ramps},
  author={Jing, Shoucai and Hui, Fei and Zhao, Xiangmo and Rios-Torres, Jackeline and Khattak, Asad J},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={12},
  pages={24248--24262},
  year={2022},
  publisher={IEEE}
}
```

## Contact & Support

For questions or issues:
- Open an issue in the repository
- Refer to the paper for theoretical details
- Check `safety_field.py` for implementation specifics
