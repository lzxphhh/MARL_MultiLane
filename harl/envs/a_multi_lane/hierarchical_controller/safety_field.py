"""
safety_field.py
================================================================================
Driving Safety Field (DSF) Implementation for Safety Assessment

Based on the paper:
"Integrated Longitudinal and Lateral Hierarchical Control of Cooperative
Merging of Connected and Automated Vehicles at On-Ramps"
Jing et al., IEEE Transactions on Intelligent Transportation Systems, 2022

The driving safety field integrates:
1. Safety Potential Energy (SE) - spatial variability of driving risk
2. Rate of Change of SE (SE_dot) - temporal variability of driving risk

Author: Research Team
Date: 2025
================================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, Direction
)


@dataclass
class SafetyFieldParameters:
    """Parameters for driving safety field calculation."""
    # Virtual mass and influence factors
    R_i: float = 1.0  # Road condition influence factor
    m_i: float = 1723.0  # Virtual mass [kg]

    # Safety field parameters
    k1: float = 0.5  # Undetermined constant for field strength
    k2: float = 1.0  # Power exponent for distance
    k3: float = 45.0  # Wave speed [m/s]
    k4: float = 1.0  # Weighting factor exponent

    # Lane parameters
    lane_width: float = 3.2  # Lane width [m]

    # Temporal weight
    gamma: float = 0.06  # Weight between SE and SE_dot


class DrivingSafetyField:
    """
    Driving Safety Field calculator for vehicle safety assessment.

    The driving safety field provides a mathematical method for:
    - Computing virtual mass calculations
    - Comprehensively describing driving risk
    - Integrating spatial and temporal risk variability
    """

    def __init__(self,
                 system_params: SystemParameters,
                 safety_field_params: Optional[SafetyFieldParameters] = None):
        """
        Initialize Driving Safety Field calculator.

        Args:
            system_params: System parameters
            safety_field_params: Safety field specific parameters
        """
        self.system_params = system_params
        self.sf_params = safety_field_params if safety_field_params else SafetyFieldParameters()

    def compute_kinetic_field_strength(self,
                                      ego_state: VehicleState,
                                      vehicle_state: VehicleState) -> float:
        """
        Compute kinetic field strength from vehicle i to ego vehicle.

        Based on Equation (30) from the paper:
        E_V,i1(t) = k1*Ri*mi*k3 / ((k3 - |vi(t)|*cos(αi(t))) * |ri1(t)|^k2)

        Args:
            ego_state: Ego vehicle state
            vehicle_state: Surrounding vehicle state

        Returns:
            Kinetic field strength
        """
        # Calculate relative position vector
        dx = ego_state.x - vehicle_state.x
        dy = ego_state.y - vehicle_state.y
        distance = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        if distance < 1e-3:
            distance = 1e-3

        # Calculate angle between velocity direction and relative position
        # vi is the velocity vector of vehicle i
        vi_x = vehicle_state.v * np.cos(vehicle_state.theta * np.pi / 180)
        vi_y = vehicle_state.v * np.sin(vehicle_state.theta * np.pi / 180)

        # ri1 is the relative position vector from vehicle i to ego
        ri1_x = dx
        ri1_y = dy

        # cos(alpha_i) = (vi · ri1) / (|vi| * |ri1|)
        dot_product = vi_x * ri1_x + vi_y * ri1_y
        vi_magnitude = np.sqrt(vi_x**2 + vi_y**2)

        if vi_magnitude < 1e-3:
            cos_alpha = 0.0
        else:
            cos_alpha = dot_product / (vi_magnitude * distance)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        # Calculate field strength
        numerator = self.sf_params.k1 * self.sf_params.R_i * self.sf_params.m_i * self.sf_params.k3
        denominator = (self.sf_params.k3 - vehicle_state.v * cos_alpha) * (distance ** self.sf_params.k2)

        # Avoid division by zero
        if abs(denominator) < 1e-6:
            field_strength = 0.0
        else:
            field_strength = numerator / denominator

        return field_strength

    def compute_safety_potential_energy(self,
                                       ego_state: VehicleState,
                                       vehicle_state: VehicleState) -> float:
        """
        Compute safety potential energy from vehicle i to ego vehicle.

        Based on Equation (31) from the paper:
        SE_V,i1(t) = k1*Ri*R1*mi*m1*k3 / ((k2-1)*|r21(t)|^(k2-1)) *
                     [(k3 - |vi(t)|*cos(αi))^(1-k2) - k3^(1-k2)]

        Args:
            ego_state: Ego vehicle state
            vehicle_state: Surrounding vehicle state

        Returns:
            Safety potential energy
        """
        # Calculate relative position
        dx = ego_state.x - vehicle_state.x
        dy = ego_state.y - vehicle_state.y
        distance = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero
        if distance < 1e-3:
            distance = 1e-3

        # Calculate cos(alpha)
        vi_x = vehicle_state.v * np.cos(vehicle_state.theta * np.pi / 180)
        vi_y = vehicle_state.v * np.sin(vehicle_state.theta * np.pi / 180)
        ri1_x = dx
        ri1_y = dy

        dot_product = vi_x * ri1_x + vi_y * ri1_y
        vi_magnitude = np.sqrt(vi_x**2 + vi_y**2)

        if vi_magnitude < 1e-3:
            cos_alpha = 0.0
        else:
            cos_alpha = dot_product / (vi_magnitude * distance)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

        # Compute SE_V,i1(t) using simplified formula from paper
        k1 = self.sf_params.k1
        k2 = self.sf_params.k2
        k3 = self.sf_params.k3
        Ri = self.sf_params.R_i
        R1 = self.sf_params.R_i
        mi = self.sf_params.m_i
        m1 = self.sf_params.m_i

        # Avoid division by zero in exponent
        if abs(k2 - 1) < 1e-6:
            k2 = 1.01

        coefficient = (k1 * Ri * R1 * mi * m1 * k3) / ((k2 - 1) * (distance ** (k2 - 1)))

        # Calculate the bracket term
        v_term = k3 - vehicle_state.v * cos_alpha
        if v_term <= 0:
            v_term = 0.01

        bracket_term = (v_term ** (1 - k2)) - (k3 ** (1 - k2))

        SE = coefficient * bracket_term

        return SE

    def compute_weighting_factor(self,
                                ego_state: VehicleState,
                                vehicle_state: VehicleState) -> float:
        """
        Compute weighting factor based on lateral distance.

        Based on Equation (31) from the paper:
        H_i(t) = min[(Lw / (2*Di(t)))^k4, 1]

        Args:
            ego_state: Ego vehicle state
            vehicle_state: Surrounding vehicle state

        Returns:
            Weighting factor [0, 1]
        """
        # Calculate lateral distance (distance to center line of right lane)
        lateral_distance = abs(ego_state.y - vehicle_state.y)

        # Avoid division by zero
        if lateral_distance < 1e-3:
            lateral_distance = 1e-3

        # Calculate weighting factor
        weight = (self.sf_params.lane_width / (2 * lateral_distance)) ** self.sf_params.k4
        weight = min(weight, 1.0)

        return weight

    def compute_safety_energy_rate(self,
                                  ego_state: VehicleState,
                                  vehicle_state: VehicleState,
                                  field_strength: float) -> float:
        """
        Compute rate of change of safety potential energy.

        Based on Equation (31) from the paper:
        SE_dot_i1(t) = M1*R1*E_V,i1 * (vi(t) - v1(t))

        Args:
            ego_state: Ego vehicle state
            vehicle_state: Surrounding vehicle state
            field_strength: Kinetic field strength E_V,i1

        Returns:
            Rate of change of safety potential energy
        """
        M1 = self.sf_params.m_i
        R1 = self.sf_params.R_i

        SE_dot = M1 * R1 * field_strength * (vehicle_state.v - ego_state.v)

        return SE_dot

    def compute_driving_safety_index(self,
                                    ego_state: VehicleState,
                                    surrounding_vehicles: Dict[str, VehicleState]) -> Dict:
        """
        Compute Driving Safety Index (DSI) for ego vehicle.

        Based on Equation (31) from the paper:
        DSI_1(t) = γ * SE_1(t) + (1 - γ) * SE_dot_1(t)

        Args:
            ego_state: Ego vehicle state
            surrounding_vehicles: Dictionary of surrounding vehicle states

        Returns:
            Dictionary containing DSI and detailed safety metrics
        """
        # Initialize accumulators
        total_SE = 0.0
        total_SE_dot = 0.0

        vehicle_contributions = {}

        # Compute contributions from each surrounding vehicle
        for vehicle_id, vehicle_state in surrounding_vehicles.items():
            # Compute kinetic field strength
            E_V = self.compute_kinetic_field_strength(ego_state, vehicle_state)

            # Compute safety potential energy
            SE_V = self.compute_safety_potential_energy(ego_state, vehicle_state)

            # Compute weighting factor
            H_i = self.compute_weighting_factor(ego_state, vehicle_state)

            # Compute safety energy rate
            SE_dot_i = self.compute_safety_energy_rate(ego_state, vehicle_state, E_V)

            # Weighted contributions
            weighted_SE = H_i * SE_V
            weighted_SE_dot = H_i * SE_dot_i

            # Accumulate
            total_SE += weighted_SE
            total_SE_dot += weighted_SE_dot

            # Store individual contributions
            vehicle_contributions[vehicle_id] = {
                'field_strength': E_V,
                'safety_potential_energy': SE_V,
                'weighting_factor': H_i,
                'safety_energy_rate': SE_dot_i,
                'weighted_SE': weighted_SE,
                'weighted_SE_dot': weighted_SE_dot
            }

        # Compute DSI
        DSI = self.sf_params.gamma * total_SE + (1 - self.sf_params.gamma) * total_SE_dot

        return {
            'DSI': DSI,
            'total_SE': total_SE,
            'total_SE_dot': total_SE_dot,
            'vehicle_contributions': vehicle_contributions
        }

    def compute_directional_safety_index(self,
                                        ego_state: VehicleState,
                                        surrounding_vehicles: Dict[str, VehicleState],
                                        direction_groups: Dict[str, List[str]]) -> Dict:
        """
        Compute directional safety indices for different vehicle groups.

        Args:
            ego_state: Ego vehicle state
            surrounding_vehicles: Dictionary of surrounding vehicle states
            direction_groups: Grouping of vehicles by direction
                             e.g., {'front': ['EF_1', 'EF_2'], 'rear': ['ER_1', 'ER_2']}

        Returns:
            Dictionary containing DSI for each direction group
        """
        directional_dsi = {}

        for direction_name, vehicle_ids in direction_groups.items():
            # Filter surrounding vehicles for this direction
            direction_vehicles = {
                vid: surrounding_vehicles[vid]
                for vid in vehicle_ids
                if vid in surrounding_vehicles
            }

            if direction_vehicles:
                # Compute DSI for this direction
                direction_result = self.compute_driving_safety_index(
                    ego_state, direction_vehicles
                )
                directional_dsi[direction_name] = direction_result
            else:
                directional_dsi[direction_name] = {
                    'DSI': 0.0,
                    'total_SE': 0.0,
                    'total_SE_dot': 0.0,
                    'vehicle_contributions': {}
                }

        return directional_dsi

    def compute_collision_risk_probability(self, DSI: float) -> float:
        """
        Convert DSI to collision risk probability.

        Higher DSI indicates higher risk.
        Uses sigmoid transformation to map DSI to [0, 1] probability.

        Args:
            DSI: Driving Safety Index

        Returns:
            Collision risk probability [0, 1]
        """
        # Sigmoid transformation: P_risk = 1 / (1 + exp(-k*DSI))
        # where k controls the steepness
        k = 0.1  # Tunable parameter

        risk_probability = 1.0 / (1.0 + np.exp(-k * DSI))

        return risk_probability


# ============================================================================
# Utility Functions for Integration with evaluate_flow.py
# ============================================================================

def format_surrounding_vehicles_for_safety_field(
    surrounding_state: Dict[str, VehicleState]) -> Dict[str, VehicleState]:
    """
    Format surrounding vehicle states for safety field calculation.

    Args:
        surrounding_state: Dictionary with vehicle IDs as keys

    Returns:
        Formatted dictionary compatible with safety field methods
    """
    return surrounding_state


def compute_safety_field_metrics(
    ego_state: VehicleState,
    surrounding_state: Dict[str, VehicleState],
    system_params: SystemParameters) -> Dict:
    """
    Compute comprehensive safety field metrics for a vehicle.

    This function provides the main interface for safety field calculations.

    Args:
        ego_state: Ego vehicle state
        surrounding_state: Surrounding vehicles state dictionary
        system_params: System parameters

    Returns:
        Dictionary containing comprehensive safety metrics
    """
    # Initialize safety field calculator
    safety_field = DrivingSafetyField(system_params)

    # Compute overall DSI
    overall_dsi_result = safety_field.compute_driving_safety_index(
        ego_state, surrounding_state
    )

    # Define direction groups for detailed analysis
    direction_groups = {
        'current_lane_front': ['EF_1', 'EF_2'],
        'current_lane_rear': ['ER_1', 'ER_2'],
        'left_lane_front': ['LF_1', 'LF_2'],
        'left_lane_rear': ['LR_1', 'LR_2'],
        'right_lane_front': ['RF_1', 'RF_2'],
        'right_lane_rear': ['RR_1', 'RR_2']
    }

    # Compute directional DSI
    directional_dsi = safety_field.compute_directional_safety_index(
        ego_state, surrounding_state, direction_groups
    )

    # Compute collision risk probability
    overall_risk = safety_field.compute_collision_risk_probability(
        overall_dsi_result['DSI']
    )

    # Aggregate direction risks
    directional_risks = {}
    for direction, dsi_result in directional_dsi.items():
        directional_risks[direction] = safety_field.compute_collision_risk_probability(
            dsi_result['DSI']
        )

    return {
        'overall_dsi': overall_dsi_result,
        'directional_dsi': directional_dsi,
        'overall_risk_probability': overall_risk,
        'directional_risk_probabilities': directional_risks,
        'safety_field_calculator': safety_field
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the Driving Safety Field calculator.
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

    # Compute safety metrics
    safety_metrics = compute_safety_field_metrics(
        ego_state, surrounding_vehicles, system_params
    )

    # Print results
    print("="*80)
    print("Driving Safety Field Analysis")
    print("="*80)
    print(f"\nOverall DSI: {safety_metrics['overall_dsi']['DSI']:.4f}")
    print(f"Overall Risk Probability: {safety_metrics['overall_risk_probability']:.4f}")
    print(f"Total Safety Potential Energy: {safety_metrics['overall_dsi']['total_SE']:.4f}")
    print(f"Total Safety Energy Rate: {safety_metrics['overall_dsi']['total_SE_dot']:.4f}")

    print("\nDirectional Risk Probabilities:")
    for direction, risk in safety_metrics['directional_risk_probabilities'].items():
        print(f"  {direction}: {risk:.4f}")

    print("\nVehicle Contributions:")
    for vehicle_id, contrib in safety_metrics['overall_dsi']['vehicle_contributions'].items():
        print(f"  {vehicle_id}:")
        print(f"    Field Strength: {contrib['field_strength']:.4f}")
        print(f"    Weighting Factor: {contrib['weighting_factor']:.4f}")
        print(f"    Weighted SE: {contrib['weighted_SE']:.4f}")
