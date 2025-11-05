"""
safety_game.py
================================================================================
Game Theory-Based Safety Assessment for Cooperative Merging Control

Based on the paper:
"Multi-Lane Coordinated Control Strategy of Connected and Automated Vehicles
for On-Ramp Merging Area Based on Cooperative Game"
Yang et al., IEEE Transactions on Intelligent Transportation Systems, 2023

This module implements:
1. Cooperative game framework for merging sequence determination
2. Multi-objective cost function (efficiency, comfort, fuel consumption)
3. Safety metrics based on game payoffs and collision probabilities
4. Nash equilibrium computation for multi-vehicle interaction

Author: Research Team
Date: 2025
================================================================================
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from itertools import product

from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType
)


@dataclass
class GameParameters:
    """Parameters for cooperative game-based safety assessment."""
    # Cost function weights (Equation 3 in paper)
    omega1: float = 1.0  # Efficiency weight (velocity tracking)
    omega2: float = 4.47  # Fuel consumption weight (acceleration minimization)
    omega3: float = 5.0  # Comfort weight (jerk minimization)

    # Safety parameters
    safe_headway: float = 2.0  # Safe time headway [s]
    min_spacing: float = 5.0  # Minimum safe spacing [m]
    collision_threshold: float = 2.0  # Collision distance threshold [m]

    # Game parameters
    max_game_iterations: int = 100
    convergence_tolerance: float = 1e-4
    nash_epsilon: float = 0.1  # Nash equilibrium tolerance

    # Vehicle dynamics constraints
    v_max: float = 35.0  # Maximum velocity [m/s]
    v_min: float = 0.0  # Minimum velocity [m/s]
    a_max: float = 3.0  # Maximum acceleration [m/s²]
    a_min: float = -5.0  # Maximum deceleration [m/s²]
    u_max: float = 3.0  # Maximum jerk [m/s³]
    u_min: float = -3.0  # Minimum jerk [m/s³]


@dataclass
class GameStrategy:
    """Represents a strategy in the cooperative game."""
    action: str  # 'leader', 'follower', 'change_lane'
    merging_time: float  # Time to reach merging point
    cost: float  # Associated cost
    collision_probability: float  # Estimated collision risk


class CooperativeGameSafety:
    """
    Cooperative game-based safety assessment for vehicle merging.

    Based on Yang et al. 2023:
    - Formulates merging as a cooperative game
    - Computes Nash equilibrium for optimal strategies
    - Evaluates safety through game payoffs and collision risks
    """

    def __init__(self,
                 system_params: SystemParameters,
                 game_params: Optional[GameParameters] = None):
        """
        Initialize cooperative game safety assessor.

        Args:
            system_params: System parameters
            game_params: Game-specific parameters
        """
        self.system_params = system_params
        self.game_params = game_params if game_params else GameParameters()

    def compute_longitudinal_cost(self,
                                 vehicle_state: VehicleState,
                                 desired_velocity: float,
                                 travel_time: float) -> float:
        """
        Compute longitudinal cost function (Equation 3 in paper).

        J_lon = 1/2 ∫[ω₁(vₓ - vdes)² + ω₂aₓ² + ω₃uₓ²]dt

        Args:
            vehicle_state: Current vehicle state
            desired_velocity: Desired velocity [m/s]
            travel_time: Time duration [s]

        Returns:
            Longitudinal cost value
        """
        omega1 = self.game_params.omega1
        omega2 = self.game_params.omega2
        omega3 = self.game_params.omega3

        # Velocity tracking cost (efficiency)
        velocity_error = vehicle_state.v - desired_velocity
        efficiency_cost = omega1 * (velocity_error ** 2)

        # Acceleration cost (fuel consumption proxy)
        fuel_cost = omega2 * (vehicle_state.a ** 2)

        # Jerk cost (comfort) - approximate from acceleration change
        # In practice, jerk would be computed from acceleration history
        estimated_jerk = vehicle_state.a / self.system_params.dt
        comfort_cost = omega3 * (estimated_jerk ** 2)

        # Integrate over travel time (simplified as multiplication)
        total_cost = 0.5 * travel_time * (efficiency_cost + fuel_cost + comfort_cost)

        return total_cost

    def compute_lateral_cost(self,
                           lane_change_time: float = 5.0) -> float:
        """
        Compute lateral cost for lane changing (Equation 4 in paper).

        J_lat = 1/2 ∫[u_y²]dt

        Args:
            lane_change_time: Time to complete lane change [s]

        Returns:
            Lateral cost value
        """
        # Simplified lateral jerk estimation
        # Assuming lane change follows polynomial trajectory
        lane_width = self.system_params.lane_width

        # Lateral jerk estimation (derived from 5th order polynomial)
        # For smooth lane change, peak lateral jerk ≈ 60*d/T³
        lateral_jerk_peak = 60 * lane_width / (lane_change_time ** 3)

        # Integrate squared jerk (approximate)
        lateral_cost = 0.5 * lane_change_time * (lateral_jerk_peak ** 2) / 3.0

        return lateral_cost

    def compute_collision_probability(self,
                                    ego_state: VehicleState,
                                    other_state: VehicleState,
                                    time_horizon: float) -> float:
        """
        Compute collision probability based on relative states.

        Uses a probabilistic model considering:
        - Spatial proximity
        - Relative velocity
        - Time to collision

        Args:
            ego_state: Ego vehicle state
            other_state: Other vehicle state
            time_horizon: Prediction horizon [s]

        Returns:
            Collision probability [0, 1]
        """
        # Compute relative position and velocity
        rel_position = other_state.x - ego_state.x
        rel_velocity = other_state.v - ego_state.v

        # Lateral distance
        lateral_distance = abs(ego_state.y - other_state.y)

        # Time to collision (TTC)
        if rel_velocity < -0.1:  # Approaching
            ttc = rel_position / (-rel_velocity)
            ttc = max(ttc, 0.01)  # Avoid division by zero
        else:
            ttc = float('inf')

        # Spatial collision risk based on distance
        longitudinal_risk = np.exp(-rel_position / 10.0)  # Decay with distance

        # Lateral collision risk
        if lateral_distance < self.system_params.lane_width:
            lateral_risk = 1.0 - (lateral_distance / self.system_params.lane_width)
        else:
            lateral_risk = 0.0

        # TTC-based temporal risk
        if ttc < self.game_params.safe_headway:
            temporal_risk = 1.0 - (ttc / self.game_params.safe_headway)
        else:
            temporal_risk = 0.0

        # Combined collision probability (using probabilistic OR)
        spatial_risk = longitudinal_risk * lateral_risk
        combined_risk = 1.0 - (1.0 - spatial_risk) * (1.0 - temporal_risk)

        # Clip to [0, 1]
        collision_prob = np.clip(combined_risk, 0.0, 1.0)

        return collision_prob

    def evaluate_strategy_pair(self,
                              ego_state: VehicleState,
                              other_state: VehicleState,
                              ego_strategy: str,
                              other_strategy: str,
                              desired_velocity: float) -> Tuple[float, float, float]:
        """
        Evaluate cost for a strategy pair in two-player game.

        Args:
            ego_state: Ego vehicle state
            other_state: Other vehicle state
            ego_strategy: Ego strategy ('leader' or 'follower')
            other_strategy: Other strategy ('leader' or 'follower')
            desired_velocity: Desired velocity

        Returns:
            Tuple of (ego_cost, other_cost, collision_probability)
        """
        # Determine arrival times based on strategies
        safe_headway_time = self.game_params.safe_headway

        # Distance to merging point (simplified)
        ego_distance = 100.0  # Placeholder
        other_distance = 100.0  # Placeholder

        if ego_strategy == 'leader' and other_strategy == 'follower':
            # Ego arrives first
            ego_travel_time = ego_distance / max(ego_state.v, 1.0)
            other_travel_time = ego_travel_time + safe_headway_time
            collision_prob = 0.05  # Low risk

        elif ego_strategy == 'follower' and other_strategy == 'leader':
            # Other arrives first
            other_travel_time = other_distance / max(other_state.v, 1.0)
            ego_travel_time = other_travel_time + safe_headway_time
            collision_prob = 0.05  # Low risk

        else:
            # Conflict: both want to be leader or follower
            ego_travel_time = ego_distance / max(ego_state.v, 1.0)
            other_travel_time = other_distance / max(other_state.v, 1.0)
            collision_prob = 0.95  # High risk (near certain collision)

        # Compute costs
        ego_cost = self.compute_longitudinal_cost(
            ego_state, desired_velocity, ego_travel_time
        )
        other_cost = self.compute_longitudinal_cost(
            other_state, desired_velocity, other_travel_time
        )

        # Add collision penalty to costs
        collision_penalty = 1000.0 * collision_prob
        ego_cost += collision_penalty
        other_cost += collision_penalty

        return ego_cost, other_cost, collision_prob

    def solve_two_player_game(self,
                             ego_state: VehicleState,
                             other_state: VehicleState,
                             desired_velocity: float) -> Dict:
        """
        Solve two-player cooperative game to find Nash equilibrium.

        Game matrix (Table I in paper):
        - Strategies: {leader, follower}
        - Payoffs: Costs for each strategy combination

        Args:
            ego_state: Ego vehicle state
            other_state: Other vehicle state
            desired_velocity: Desired velocity

        Returns:
            Dictionary with game solution and safety metrics
        """
        # Define strategy space
        strategies = ['leader', 'follower']

        # Build payoff matrix
        payoff_matrix = np.zeros((2, 2, 3))  # (ego_strat, other_strat, [ego_cost, other_cost, collision_prob])

        for i, ego_strat in enumerate(strategies):
            for j, other_strat in enumerate(strategies):
                ego_cost, other_cost, collision_prob = self.evaluate_strategy_pair(
                    ego_state, other_state, ego_strat, other_strat, desired_velocity
                )
                payoff_matrix[i, j, :] = [ego_cost, other_cost, collision_prob]

        # Find Nash equilibrium (minimize joint cost in cooperative game)
        min_joint_cost = float('inf')
        nash_equilibrium = None

        for i, ego_strat in enumerate(strategies):
            for j, other_strat in enumerate(strategies):
                joint_cost = payoff_matrix[i, j, 0] + payoff_matrix[i, j, 1]
                collision_prob = payoff_matrix[i, j, 2]

                # Avoid strategies with high collision risk
                if collision_prob > 0.5:
                    continue

                if joint_cost < min_joint_cost:
                    min_joint_cost = joint_cost
                    nash_equilibrium = {
                        'ego_strategy': ego_strat,
                        'other_strategy': other_strat,
                        'ego_cost': payoff_matrix[i, j, 0],
                        'other_cost': payoff_matrix[i, j, 1],
                        'joint_cost': joint_cost,
                        'collision_probability': collision_prob
                    }

        return nash_equilibrium if nash_equilibrium else {
            'ego_strategy': 'follower',
            'other_strategy': 'leader',
            'ego_cost': float('inf'),
            'other_cost': float('inf'),
            'joint_cost': float('inf'),
            'collision_probability': 1.0
        }

    def compute_merging_sequence_cost(self,
                                     vehicles: List[VehicleState],
                                     sequence: List[int],
                                     desired_velocity: float) -> float:
        """
        Compute total cost for a given merging sequence.

        Args:
            vehicles: List of vehicle states
            sequence: Merging sequence (vehicle indices)
            desired_velocity: Desired velocity

        Returns:
            Total cost for the sequence
        """
        total_cost = 0.0
        safe_headway_time = self.game_params.safe_headway

        for idx, veh_idx in enumerate(sequence):
            vehicle = vehicles[veh_idx]

            # Compute arrival time
            travel_time = idx * safe_headway_time + 10.0  # Base time + spacing

            # Compute individual cost
            cost = self.compute_longitudinal_cost(
                vehicle, desired_velocity, travel_time
            )
            total_cost += cost

        return total_cost

    def compute_safety_metrics(self,
                              game_solution: Dict,
                              ego_state: VehicleState,
                              other_state: VehicleState) -> Dict:
        """
        Compute comprehensive safety metrics from game solution.

        Args:
            game_solution: Nash equilibrium solution
            ego_state: Ego vehicle state
            other_state: Other vehicle state

        Returns:
            Dictionary of safety metrics
        """
        # Extract collision probability from game solution
        collision_prob = game_solution['collision_probability']

        # Compute spatial safety margin
        rel_distance = other_state.x - ego_state.x
        spatial_safety_margin = max(0.0, rel_distance - self.game_params.min_spacing)

        # Compute temporal safety margin (TTC)
        rel_velocity = other_state.v - ego_state.v
        if rel_velocity < -0.1:
            ttc = rel_distance / (-rel_velocity)
        else:
            ttc = float('inf')
        temporal_safety_margin = max(0.0, ttc - self.game_params.safe_headway)

        # Game-theoretic safety score (based on cost)
        # Lower cost = safer/better strategy
        max_acceptable_cost = 100.0
        game_safety_score = 1.0 - min(game_solution['ego_cost'] / max_acceptable_cost, 1.0)

        # Cooperative benefit (difference between non-cooperative and cooperative)
        # In cooperative game, joint cost is minimized
        cooperative_benefit = game_solution['ego_cost'] - game_solution['joint_cost'] / 2.0

        return {
            'collision_probability': collision_prob,
            'spatial_safety_margin': spatial_safety_margin,
            'temporal_safety_margin': temporal_safety_margin,
            'game_safety_score': game_safety_score,
            'cooperative_benefit': cooperative_benefit,
            'ego_cost': game_solution['ego_cost'],
            'joint_cost': game_solution['joint_cost'],
            'nash_strategy': game_solution['ego_strategy']
        }


# ============================================================================
# Utility Functions for Integration with evaluate_flow.py
# ============================================================================

def compute_game_based_safety_metrics(
    ego_state: VehicleState,
    surrounding_state: Dict[str, VehicleState],
    system_params: SystemParameters,
    desired_velocity: float = 25.0) -> Dict:
    """
    Compute game theory-based safety metrics for ego vehicle.

    This function provides the main interface for game-based safety assessment.

    Args:
        ego_state: Ego vehicle state
        surrounding_state: Surrounding vehicles state dictionary
        system_params: System parameters
        desired_velocity: Desired velocity [m/s]

    Returns:
        Dictionary containing comprehensive safety metrics
    """
    # Initialize game safety assessor
    game_params = GameParameters()
    game_safety = CooperativeGameSafety(system_params, game_params)

    # Solve games with primary adjacent vehicles
    game_solutions = {}
    safety_metrics_by_vehicle = {}

    primary_vehicles = ['EF_1', 'ER_1', 'LF_1', 'LR_1', 'RF_1', 'RR_1']

    for vehicle_id in primary_vehicles:
        if vehicle_id in surrounding_state:
            other_state = surrounding_state[vehicle_id]

            # Solve two-player game
            game_solution = game_safety.solve_two_player_game(
                ego_state, other_state, desired_velocity
            )
            game_solutions[vehicle_id] = game_solution

            # Compute safety metrics
            safety_metrics = game_safety.compute_safety_metrics(
                game_solution, ego_state, other_state
            )
            safety_metrics_by_vehicle[vehicle_id] = safety_metrics

    # Aggregate metrics across all vehicles
    if safety_metrics_by_vehicle:
        all_collision_probs = [m['collision_probability'] for m in safety_metrics_by_vehicle.values()]
        all_game_scores = [m['game_safety_score'] for m in safety_metrics_by_vehicle.values()]
        all_costs = [m['ego_cost'] for m in safety_metrics_by_vehicle.values()]

        max_collision_prob = max(all_collision_probs)
        min_game_score = min(all_game_scores)
        total_cost = sum(all_costs)

        # Overall safety risk (1 - safety_score)
        overall_safety_risk = 1.0 - min_game_score
    else:
        max_collision_prob = 0.0
        min_game_score = 1.0
        total_cost = 0.0
        overall_safety_risk = 0.0

    return {
        'game_solutions': game_solutions,
        'safety_metrics_by_vehicle': safety_metrics_by_vehicle,
        'max_collision_probability': max_collision_prob,
        'min_game_safety_score': min_game_score,
        'total_game_cost': total_cost,
        'overall_safety_risk': overall_safety_risk,
        'num_interactions': len(safety_metrics_by_vehicle)
    }


def generate_prediction_centers_from_game(
    ego_state: VehicleState,
    surrounding_state: Dict[str, VehicleState],
    game_solutions: Dict,
    prediction_horizon: float = 3.0,
    dt: float = 0.1) -> Dict:
    """
    Generate simplified prediction centers based on game solutions.

    Args:
        ego_state: Ego vehicle state
        surrounding_state: Surrounding vehicles
        game_solutions: Game solutions for each vehicle
        prediction_horizon: Prediction time horizon [s]
        dt: Time step [s]

    Returns:
        Dictionary of prediction centers for MPC
    """
    prediction_centers = {}
    time_steps = int(prediction_horizon / dt)

    for vehicle_id, vehicle_state in surrounding_state.items():
        if vehicle_id.endswith('_1'):  # Only primary adjacent vehicles
            # Get game solution if available
            if vehicle_id in game_solutions:
                nash_strategy = game_solutions[vehicle_id]['ego_strategy']

                # Adjust prediction based on strategy
                if nash_strategy == 'leader':
                    # Vehicle will maintain or increase speed
                    speed_factor = 1.0
                else:  # follower
                    # Vehicle will maintain or decrease speed
                    speed_factor = 0.95
            else:
                speed_factor = 1.0

            # Simple constant velocity prediction with strategy adjustment
            predicted_positions = []
            for i in range(time_steps):
                t = i * dt
                pred_velocity = vehicle_state.v * speed_factor
                pred_x = vehicle_state.x + pred_velocity * np.cos(vehicle_state.theta * np.pi / 180) * t
                predicted_positions.append(pred_x)

            prediction_centers[vehicle_id] = {
                'reference': {
                    'position': np.array(predicted_positions),
                    'time': np.arange(time_steps) * dt
                },
                'current_state': vehicle_state,
                'nash_strategy': nash_strategy if vehicle_id in game_solutions else 'unknown',
                'game_cost': game_solutions[vehicle_id]['ego_cost'] if vehicle_id in game_solutions else 0.0
            }

    return prediction_centers


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the cooperative game-based safety assessment.
    """
    # Initialize system parameters
    system_params = SystemParameters()

    # Create example ego vehicle state
    ego_state = VehicleState(
        x=100.0, y=0.0, v=20.0, theta=90.0, a=0.0, omega=0.0,
        lane_id=1, vehicle_type=VehicleType.CAV, front_v=20.0, front_spacing=30.0
    )

    # Create example surrounding vehicles
    surrounding_vehicles = {
        'EF_1': VehicleState(
            x=130.0, y=0.0, v=18.0, theta=90.0, a=0.0, omega=0.0,
            lane_id=1, vehicle_type=VehicleType.HDV, front_v=18.0, front_spacing=50.0
        ),
        'ER_1': VehicleState(
            x=70.0, y=0.0, v=22.0, theta=90.0, a=0.0, omega=0.0,
            lane_id=1, vehicle_type=VehicleType.HDV, front_v=22.0, front_spacing=40.0
        ),
        'LF_1': VehicleState(
            x=120.0, y=3.2, v=19.0, theta=90.0, a=0.0, omega=0.0,
            lane_id=2, vehicle_type=VehicleType.HDV, front_v=19.0, front_spacing=45.0
        )
    }

    # Compute game-based safety metrics
    safety_metrics = compute_game_based_safety_metrics(
        ego_state, surrounding_vehicles, system_params
    )

    # Print results
    print("="*80)
    print("Game Theory-Based Safety Assessment")
    print("="*80)
    print(f"\nOverall Safety Risk: {safety_metrics['overall_safety_risk']:.4f}")
    print(f"Max Collision Probability: {safety_metrics['max_collision_probability']:.4f}")
    print(f"Min Game Safety Score: {safety_metrics['min_game_safety_score']:.4f}")
    print(f"Total Game Cost: {safety_metrics['total_game_cost']:.2f}")
    print(f"Number of Interactions: {safety_metrics['num_interactions']}")

    print("\nGame Solutions by Vehicle:")
    for vehicle_id, solution in safety_metrics['game_solutions'].items():
        print(f"\n  {vehicle_id}:")
        print(f"    Nash Strategy: {solution['ego_strategy']}")
        print(f"    Ego Cost: {solution['ego_cost']:.2f}")
        print(f"    Joint Cost: {solution['joint_cost']:.2f}")
        print(f"    Collision Probability: {solution['collision_probability']:.4f}")

    print("\nDetailed Safety Metrics by Vehicle:")
    for vehicle_id, metrics in safety_metrics['safety_metrics_by_vehicle'].items():
        print(f"\n  {vehicle_id}:")
        print(f"    Collision Probability: {metrics['collision_probability']:.4f}")
        print(f"    Spatial Safety Margin: {metrics['spatial_safety_margin']:.2f} m")
        print(f"    Temporal Safety Margin: {metrics['temporal_safety_margin']:.2f} s")
        print(f"    Game Safety Score: {metrics['game_safety_score']:.4f}")
        print(f"    Cooperative Benefit: {metrics['cooperative_benefit']:.2f}")
