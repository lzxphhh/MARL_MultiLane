"""
evaluate_flow.py
================================================================================
Evaluation Framework for Cooperative MPC Controller with Different Safety Assessment Methods

This script provides a clear structure for comparing different safety assessment approaches:
1. Conflict Assessment Controller (trajectory prediction + envelope generation)
2. Game Theory-based Safety Metrics (Yang et al., IEEE TITS 2023)
3. Safety Field Method (Jing et al., IEEE TITS 2022)
4. Artificial Potential Field-based Safety Metrics (to be implemented)

Author: Research Team
Date: 2025
================================================================================
"""

import yaml
import random
import time
import os
import numpy as np
import pandas as pd
import traci
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from loguru import logger

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.check_folder import check_folder
from tshub.utils.init_log import set_logger
from tshub.tshub_env.tshub_env import TshubEnvironment
from harl.envs.a_multi_lane.env_utils.generate_scene_NGSIM import generate_scenario_NGSIM

from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, LateralDecision
)
from harl.envs.a_multi_lane.hierarchical_controller.get_traffic_state import get_traffic_state, get_surrounding_state
from harl.envs.a_multi_lane.hierarchical_controller.scene_recognition import SceneRecognitionModule
from harl.envs.a_multi_lane.hierarchical_controller.lateral_decision import LateralDecisionModule
from harl.envs.a_multi_lane.hierarchical_controller.conflict_assessment_controller import ConflictAssessmentController
from harl.envs.a_multi_lane.hierarchical_controller.mpc_cooperative_controller import (
    MPCCooperativeController, MPCParameters
)


# ============================================================================
# Abstract Base Class for Safety Assessment Methods
# ============================================================================

class SafetyAssessmentMethod(ABC):
    """
    Abstract base class for different safety assessment methods.
    All safety assessment methods should inherit from this class.
    """

    def __init__(self, system_params: SystemParameters):
        """
        Initialize safety assessment method.

        Args:
            system_params: System parameters configuration
        """
        self.params = system_params
        self.method_name = "BaseSafetyMethod"

    @abstractmethod
    def assess_safety(self,
                     ego_state: VehicleState,
                     surrounding_state: Dict,
                     surrounding_vehicles: Dict,
                     lateral_decision: str) -> Dict:
        """
        Assess safety and generate prediction data for MPC controller.

        Args:
            ego_state: Ego vehicle state
            surrounding_state: Surrounding vehicles state dictionary
            surrounding_vehicles: Surrounding vehicles information
            lateral_decision: Lateral decision ('keep_lane', 'pre_change_left', 'pre_change_right')

        Returns:
            Dictionary containing:
                - 'surrounding_prediction_centers': Prediction centers for MPC
                - 'safety_metrics': Additional safety metrics
                - 'computation_time': Time cost for assessment
        """
        pass

    def get_method_name(self) -> str:
        """Return the name of the safety assessment method."""
        return self.method_name


# ============================================================================
# Safety Assessment Method 1: Conflict Assessment (Current Implementation)
# ============================================================================

class ConflictAssessmentMethod(SafetyAssessmentMethod):
    """
    Safety assessment using conflict assessment controller.
    Based on trajectory prediction and envelope generation.
    """

    def __init__(self, system_params: SystemParameters):
        super().__init__(system_params)
        self.method_name = "ConflictAssessment"
        self.conflict_controller = ConflictAssessmentController(system_params)
        self.vehicle_groups = {
            'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
            'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
            'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
        }

    def assess_safety(self,
                     ego_state: VehicleState,
                     surrounding_state: Dict,
                     surrounding_vehicles: Dict,
                     lateral_decision: str) -> Dict:
        """
        Assess safety using conflict assessment approach.
        """
        start_time = time.time()

        # 1. Setup scenario configuration
        scenario_config = self.conflict_controller.setup_13_vehicle_scenario(
            ego_state, surrounding_state
        )

        # 2. Determine vehicles to analyze based on lateral decision
        if lateral_decision == 'keep_lane':
            vehicles_to_analysis = self.vehicle_groups['current_lane']
        elif lateral_decision == 'pre_change_left':
            vehicles_to_analysis = self.vehicle_groups['current_lane'] + self.vehicle_groups['left_lane']
        elif lateral_decision == 'pre_change_right':
            vehicles_to_analysis = self.vehicle_groups['current_lane'] + self.vehicle_groups['right_lane']
        else:
            vehicles_to_analysis = self.vehicle_groups['current_lane']

        # 3. Generate ego vehicle trajectory
        target_speed = self._get_target_speed(lateral_decision, surrounding_vehicles)
        ego_prediction_trajectory = self.conflict_controller.generate_ego_trajectory(
            scenario_config, lateral_decision, target_speed
        )

        # 4. Generate surrounding vehicle trajectories
        surrounding_prediction_trajectories = self.conflict_controller.generate_vehicle_trajectories(
            scenario_config, lateral_decision, vehicles_to_analysis
        )

        # 5. Generate vehicle envelopes
        surrounding_envelopes = self.conflict_controller.generate_vehicle_envelopes(
            surrounding_prediction_trajectories, scenario_config
        )

        # 6. Generate probability distributions
        probability_distributions = self.conflict_controller.generate_probability_distributions(
            surrounding_envelopes, scenario_config, ego_prediction_trajectory, vehicles_to_analysis
        )

        # 7. Evaluate cooperation capabilities
        cooperation_results = self.conflict_controller.evaluate_cooperation_capabilities(
            probability_distributions
        )

        # 8. Compute collision probabilities
        collision_probabilities = self.conflict_controller.compute_collision_probabilities(
            probability_distributions, cooperation_results
        )

        computation_time = time.time() - start_time

        return {
            'surrounding_prediction_centers': surrounding_envelopes,
            'safety_metrics': {
                'collision_probabilities': collision_probabilities,
                'cooperation_results': cooperation_results,
                'probability_distributions': probability_distributions
            },
            'computation_time': computation_time
        }

    def _get_target_speed(self, lateral_decision: str, surrounding_vehicles: Dict) -> float:
        """Get target speed based on lateral decision."""
        if lateral_decision == 'pre_change_left':
            return surrounding_vehicles['left_leader_1']['veh_vel']
        elif lateral_decision == 'pre_change_right':
            return surrounding_vehicles['right_leader_1']['veh_vel']
        else:
            return surrounding_vehicles['ego_leader_1']['veh_vel']


# ============================================================================
# Safety Assessment Method 2: Game Theory (Implemented)
# ============================================================================

class GameTheoryMethod(SafetyAssessmentMethod):
    """
    Safety assessment using cooperative game theory approach.
    Based on Yang et al., IEEE TITS 2023.

    The cooperative game framework:
    - Two-player games between ego and neighboring vehicles
    - Strategy space: {leader, follower}
    - Nash equilibrium minimizes joint cost
    - Cost function: efficiency + fuel consumption + comfort
    """

    def __init__(self, system_params: SystemParameters):
        super().__init__(system_params)
        self.method_name = "GameTheory"

        # Import game theory components
        from harl.envs.a_multi_lane.hierarchical_controller.safety_game import (
            CooperativeGameSafety, GameParameters, compute_game_based_safety_metrics,
            generate_prediction_centers_from_game
        )

        # Initialize game parameters
        self.game_params = GameParameters(
            omega1=1.0,  # Efficiency weight
            omega2=4.47,  # Fuel consumption weight
            omega3=5.0,  # Comfort weight
            safe_headway=2.0,
            min_spacing=5.0,
            collision_threshold=2.0
        )

        self.game_safety = CooperativeGameSafety(system_params, self.game_params)
        self.compute_game_metrics = compute_game_based_safety_metrics
        self.generate_predictions = generate_prediction_centers_from_game

        self.vehicle_groups = {
            'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
            'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
            'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
        }

    def assess_safety(self,
                     ego_state: VehicleState,
                     surrounding_state: Dict,
                     surrounding_vehicles: Dict,
                     lateral_decision: str) -> Dict:
        """
        Assess safety using cooperative game theory approach.

        The method:
        1. Solves two-player games with primary adjacent vehicles
        2. Finds Nash equilibrium for each game
        3. Evaluates collision probabilities and costs
        4. Generates prediction centers for MPC
        """
        start_time = time.time()

        # 1. Determine target speed based on lateral decision
        if lateral_decision == 'pre_change_left':
            desired_velocity = surrounding_vehicles['left_leader_1']['veh_vel']
        elif lateral_decision == 'pre_change_right':
            desired_velocity = surrounding_vehicles['right_leader_1']['veh_vel']
        else:
            desired_velocity = surrounding_vehicles['ego_leader_1']['veh_vel']

        # 2. Compute game-based safety metrics
        safety_metrics = self.compute_game_metrics(
            ego_state, surrounding_state, self.params, desired_velocity
        )

        # 3. Generate prediction centers for MPC from game solutions
        game_solutions = safety_metrics['game_solutions']
        surrounding_prediction_centers = self.generate_predictions(
            ego_state, surrounding_state, game_solutions,
            prediction_horizon=self.params.prediction_horizon,
            dt=self.params.dt
        )

        # 4. Determine relevant vehicles based on lateral decision
        if lateral_decision == 'keep_lane':
            relevant_vehicles = self.vehicle_groups['current_lane']
        elif lateral_decision == 'pre_change_left':
            relevant_vehicles = self.vehicle_groups['current_lane'] + self.vehicle_groups['left_lane']
        elif lateral_decision == 'pre_change_right':
            relevant_vehicles = self.vehicle_groups['current_lane'] + self.vehicle_groups['right_lane']
        else:
            relevant_vehicles = self.vehicle_groups['current_lane']

        # 5. Filter metrics for relevant vehicles
        relevant_game_solutions = {
            vid: game_solutions[vid]
            for vid in relevant_vehicles
            if vid in game_solutions
        }

        # 6. Compute directional collision risks
        directional_risks = self._compute_directional_risks(relevant_game_solutions, lateral_decision)

        computation_time = time.time() - start_time

        # 7. Package results
        return {
            'surrounding_prediction_centers': surrounding_prediction_centers,
            'safety_metrics': {
                'max_collision_probability': safety_metrics['max_collision_probability'],
                'min_game_safety_score': safety_metrics['min_game_safety_score'],
                'total_game_cost': safety_metrics['total_game_cost'],
                'overall_safety_risk': safety_metrics['overall_safety_risk'],
                'num_interactions': safety_metrics['num_interactions'],
                'game_solutions': relevant_game_solutions,
                'directional_risks': directional_risks
            },
            'computation_time': computation_time
        }

    def _compute_directional_risks(self, game_solutions: Dict, lateral_decision: str) -> Dict:
        """Compute collision risks by direction."""
        directional_risks = {
            'ego_front': 0.0,
            'ego_rear': 0.0,
            'left_front': 0.0,
            'left_rear': 0.0,
            'right_front': 0.0,
            'right_rear': 0.0
        }

        direction_map = {
            'EF_1': 'ego_front', 'EF_2': 'ego_front',
            'ER_1': 'ego_rear', 'ER_2': 'ego_rear',
            'LF_1': 'left_front', 'LF_2': 'left_front',
            'LR_1': 'left_rear', 'LR_2': 'left_rear',
            'RF_1': 'right_front', 'RF_2': 'right_front',
            'RR_1': 'right_rear', 'RR_2': 'right_rear'
        }

        for vehicle_id, solution in game_solutions.items():
            direction = direction_map.get(vehicle_id, None)
            if direction:
                collision_prob = solution['collision_probability']
                # Take maximum risk for each direction
                directional_risks[direction] = max(
                    directional_risks[direction],
                    collision_prob
                )

        return directional_risks


# ============================================================================
# Safety Assessment Method 3: Safety Field Method (Implemented)
# ============================================================================

class SafetyFieldMethod(SafetyAssessmentMethod):
    """
    Safety assessment using Driving Safety Field approach.
    Based on Jing et al., IEEE TITS 2022.

    The driving safety field integrates:
    - Safety Potential Energy (SE) - spatial variability of risk
    - Rate of Change of SE (SE_dot) - temporal variability of risk
    """

    def __init__(self, system_params: SystemParameters):
        super().__init__(system_params)
        self.method_name = "SafetyField"

        # Import safety field calculator
        from harl.envs.a_multi_lane.hierarchical_controller.safety_field import (
            DrivingSafetyField, SafetyFieldParameters, compute_safety_field_metrics
        )

        # Initialize safety field calculator with custom parameters
        self.safety_field_params = SafetyFieldParameters(
            R_i=1.0,
            m_i=1723.0,
            k1=0.5,
            k2=1.0,
            k3=45.0,
            k4=1.0,
            lane_width=3.2,
            gamma=0.06
        )
        self.safety_field = DrivingSafetyField(system_params, self.safety_field_params)
        self.compute_safety_field_metrics = compute_safety_field_metrics

        self.vehicle_groups = {
            'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
            'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
            'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
        }

    def assess_safety(self,
                     ego_state: VehicleState,
                     surrounding_state: Dict,
                     surrounding_vehicles: Dict,
                     lateral_decision: str) -> Dict:
        """
        Assess safety using driving safety field approach.

        Computes DSI (Driving Safety Index) which integrates:
        - Spatial driving risk (Safety Potential Energy)
        - Temporal driving risk (Rate of Change)
        """
        start_time = time.time()

        # 1. Compute overall safety field metrics
        safety_metrics = self.compute_safety_field_metrics(
            ego_state, surrounding_state, self.params
        )

        # 2. Generate simplified prediction centers for MPC
        # The safety field provides risk assessment without explicit trajectory prediction
        # We create a simplified representation based on current positions
        surrounding_prediction_centers = {}

        for vehicle_id, vehicle_state in surrounding_state.items():
            # Extract only primary adjacent vehicles (_1 suffix)
            if vehicle_id.endswith('_1'):
                # Create a simple position-based "envelope" centered at current position
                # This is used by MPC for collision avoidance
                time_steps = 30  # 3 seconds at 0.1s intervals

                # Simple constant velocity prediction
                predicted_positions = []
                for i in range(time_steps):
                    t = i * 0.1
                    # Predict position assuming constant velocity
                    pred_x = vehicle_state.x + vehicle_state.v * np.cos(vehicle_state.theta * np.pi / 180) * t
                    predicted_positions.append(pred_x)

                surrounding_prediction_centers[vehicle_id] = {
                    'reference': {
                        'position': np.array(predicted_positions),
                        'time': np.arange(time_steps) * 0.1
                    },
                    'current_state': vehicle_state,
                    'dsi_contribution': safety_metrics['overall_dsi']['vehicle_contributions'].get(
                        vehicle_id, {'weighted_SE': 0.0}
                    )['weighted_SE']
                }

        # 3. Compute directional risk assessments
        if lateral_decision == 'keep_lane':
            relevant_vehicles = self.vehicle_groups['current_lane']
        elif lateral_decision == 'pre_change_left':
            relevant_vehicles = self.vehicle_groups['current_lane'] + self.vehicle_groups['left_lane']
        elif lateral_decision == 'pre_change_right':
            relevant_vehicles = self.vehicle_groups['current_lane'] + self.vehicle_groups['right_lane']
        else:
            relevant_vehicles = self.vehicle_groups['current_lane']

        # Filter surrounding vehicles for relevant direction
        relevant_surrounding = {
            vid: surrounding_state[vid]
            for vid in relevant_vehicles
            if vid in surrounding_state
        }

        # Compute DSI for relevant vehicles only
        relevant_dsi_result = self.safety_field.compute_driving_safety_index(
            ego_state, relevant_surrounding
        )

        computation_time = time.time() - start_time

        # 4. Package results
        return {
            'surrounding_prediction_centers': surrounding_prediction_centers,
            'safety_metrics': {
                'overall_dsi': safety_metrics['overall_dsi']['DSI'],
                'overall_risk_probability': safety_metrics['overall_risk_probability'],
                'directional_dsi': safety_metrics['directional_dsi'],
                'directional_risks': safety_metrics['directional_risk_probabilities'],
                'relevant_dsi': relevant_dsi_result['DSI'],
                'total_SE': safety_metrics['overall_dsi']['total_SE'],
                'total_SE_dot': safety_metrics['overall_dsi']['total_SE_dot'],
                'vehicle_contributions': safety_metrics['overall_dsi']['vehicle_contributions']
            },
            'computation_time': computation_time
        }


# ============================================================================
# Safety Assessment Method 4: Artificial Potential Field (To be implemented)
# ============================================================================

class ArtificialPotentialFieldMethod(SafetyAssessmentMethod):
    """
    Safety assessment using artificial potential field approach.
    TODO: Implement APF-based safety metrics.
    """

    def __init__(self, system_params: SystemParameters):
        super().__init__(system_params)
        self.method_name = "ArtificialPotentialField"
        # TODO: Initialize APF parameters

    def assess_safety(self,
                     ego_state: VehicleState,
                     surrounding_state: Dict,
                     surrounding_vehicles: Dict,
                     lateral_decision: str) -> Dict:
        """
        Assess safety using artificial potential field approach.
        TODO: Implement APF-based safety assessment.
        """
        start_time = time.time()

        # TODO: Implement APF-based safety assessment
        # Placeholder implementation
        surrounding_prediction_centers = {}
        safety_metrics = {
            'potential_field': None,
            'repulsive_force': None,
            'attractive_force': None
        }

        computation_time = time.time() - start_time

        return {
            'surrounding_prediction_centers': surrounding_prediction_centers,
            'safety_metrics': safety_metrics,
            'computation_time': computation_time
        }


# ============================================================================
# Cooperative MPC Controller Wrapper
# ============================================================================

class CooperativeMPCWrapper:
    """
    Wrapper for cooperative MPC controller with different safety assessment methods.
    """

    def __init__(self,
                 mpc_params: MPCParameters,
                 system_params: SystemParameters,
                 safety_method: SafetyAssessmentMethod):
        """
        Initialize MPC wrapper.

        Args:
            mpc_params: MPC parameters
            system_params: System parameters
            safety_method: Safety assessment method instance
        """
        self.mpc_controller = MPCCooperativeController(mpc_params, system_params)
        self.safety_method = safety_method
        self.system_params = system_params

        logger.info(f"Initialized MPC Controller with {safety_method.get_method_name()} method")

    def control(self,
                ego_state: VehicleState,
                surrounding_state: Dict,
                surrounding_vehicles: Dict,
                lateral_decision: str,
                traffic_state: Dict,
                weights) -> Tuple[float, Dict]:
        """
        Execute control with safety assessment.

        Args:
            ego_state: Ego vehicle state
            surrounding_state: Surrounding vehicles state
            surrounding_vehicles: Surrounding vehicles information
            lateral_decision: Lateral decision
            traffic_state: Traffic state
            weights: Control weights

        Returns:
            Tuple of (next_speed, control_info)
        """
        # 1. Perform safety assessment
        safety_result = self.safety_method.assess_safety(
            ego_state, surrounding_state, surrounding_vehicles, lateral_decision
        )

        # 2. Determine target parameters for MPC
        target_speed = self._get_target_speed(lateral_decision, surrounding_vehicles)
        target_y = self._get_target_y(ego_state, lateral_decision)

        # 3. Solve MPC with safety constraints
        mpc_result = self.mpc_controller.solve_mpc(
            current_state=ego_state,
            reference_speed=target_speed,
            target_y=target_y,
            weights=weights,
            surrounding_prediction_centers=safety_result['surrounding_prediction_centers']
        )

        # 4. Extract control command
        if mpc_result.feasible:
            next_state = mpc_result.predicted_states[9]  # 1s prediction (10th step)
            next_speed = next_state[2]
        else:
            logger.warning(f"MPC infeasible, using fallback control")
            next_speed = target_speed

        # 5. Compile control information
        control_info = {
            'mpc_result': mpc_result,
            'safety_result': safety_result,
            'target_speed': target_speed,
            'target_y': target_y,
            'lateral_decision': lateral_decision
        }

        return next_speed, control_info

    def _get_target_speed(self, lateral_decision: str, surrounding_vehicles: Dict) -> float:
        """Get target speed based on lateral decision."""
        if lateral_decision == 'pre_change_left':
            return surrounding_vehicles['left_leader_1']['veh_vel']
        elif lateral_decision == 'pre_change_right':
            return surrounding_vehicles['right_leader_1']['veh_vel']
        else:
            return surrounding_vehicles['ego_leader_1']['veh_vel']

    def _get_target_y(self, ego_state: VehicleState, lateral_decision: str) -> float:
        """Get target lateral position based on lateral decision."""
        if lateral_decision == 'pre_change_left':
            return ego_state.y + self.system_params.lane_width
        elif lateral_decision == 'pre_change_right':
            return ego_state.y - self.system_params.lane_width
        else:
            return ego_state.y


# ============================================================================
# Utility Functions
# ============================================================================

def format_conversion(pos_x, pos_y, speed, heading, acc, yaw_rate, lane_id, veh_type, front_v, front_spacing):
    """Convert observation data to VehicleState format."""
    if veh_type == 0:
        type = VehicleType.HDV
    else:
        type = VehicleType.CAV
    return VehicleState(
        x=pos_x, y=pos_y, v=speed, theta=heading, a=acc, omega=yaw_rate,
        lane_id=lane_id, vehicle_type=type, front_v=front_v, front_spacing=front_spacing
    )


DECISION_STR = {
    0: 'keep_lane',
    -1: 'pre_change_left',
    1: 'pre_change_right',
}


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_evaluation(config: Dict, safety_method_name: str = "ConflictAssessment"):
    """
    Run evaluation with specified safety assessment method.

    Args:
        config: Configuration dictionary
        safety_method_name: Name of safety method
                          ('ConflictAssessment', 'SafetyField', 'GameTheory', 'ArtificialPotentialField')
    """
    # ========================================================================
    # 1. Environment Setup
    # ========================================================================
    path_convert = get_abs_path(__file__)
    set_logger(path_convert('./'))

    sumo_cfg = path_convert("env_utils/SUMO_files/scenario.sumocfg")
    net_file = path_convert("env_utils/SUMO_files/veh.net.xml")

    # Create save folders
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    result_folder = path_convert(f'./evaluation_results/{safety_method_name}_{now_time}/')
    check_folder(result_folder)

    logger.info(f"Starting evaluation with {safety_method_name} method")
    logger.info(f"Results will be saved to: {result_folder}")

    # ========================================================================
    # 2. Load Configuration
    # ========================================================================
    args = config
    CAV_num = args["num_CAVs"]
    HDV_num = args["num_HDVs"]
    scene_length = args["scene_length"]
    use_gui = args["use_gui"]
    scene_definition = args["scene_definition"]
    scene_specify = args["scene_specify"]
    scene_id = args["scene_id"]
    lane_ids = args["calc_features_lane_ids"]

    # ========================================================================
    # 3. Initialize SUMO Environment
    # ========================================================================
    tshub_env = TshubEnvironment(
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        is_map_builder_initialized=True,
        is_vehicle_builder_initialized=True,
        is_aircraft_builder_initialized=False,
        is_traffic_light_builder_initialized=False,
        vehicle_action_type='vehicle_continuous_action',
        use_gui=use_gui,
        num_seconds=scene_length
    )

    # ========================================================================
    # 4. Initialize Control Components
    # ========================================================================
    system_params = SystemParameters()
    scene_recognition = SceneRecognitionModule(system_params)
    lateral_decision = LateralDecisionModule(system_params)

    # Initialize MPC parameters
    mpc_params = MPCParameters(
        prediction_horizon=20,
        control_horizon=10,
        sampling_time=0.1,
    )

    # ========================================================================
    # 5. Initialize Safety Assessment Method
    # ========================================================================
    if safety_method_name == "ConflictAssessment":
        safety_method = ConflictAssessmentMethod(system_params)
    elif safety_method_name == "GameTheory":
        safety_method = GameTheoryMethod(system_params)
    elif safety_method_name == "SafetyField":
        safety_method = SafetyFieldMethod(system_params)
    elif safety_method_name == "ArtificialPotentialField":
        safety_method = ArtificialPotentialFieldMethod(system_params)
    else:
        raise ValueError(f"Unknown safety method: {safety_method_name}")

    # ========================================================================
    # 6. Initialize MPC Controller Wrapper
    # ========================================================================
    mpc_wrapper = CooperativeMPCWrapper(mpc_params, system_params, safety_method)

    # ========================================================================
    # 7. Generate Scenario
    # ========================================================================
    obs = tshub_env.reset()

    CAVs_id = [f'CAV_{i}' for i in range(CAV_num)]
    HDVs_id = [f'HDV_{i}' for i in range(HDV_num)]

    config_file = path_convert("env_utils/vehicle_config.json")

    # Select or specify scenario
    if scene_specify:
        scene_class_id = scene_id
    else:
        scene_class_id = random.randint(0, len(scene_definition) - 1)

    scene_info = scene_definition[scene_class_id]
    logger.info(f"Using scenario class {scene_class_id}: {scene_info}")

    # Load leader trajectories
    leaders_df = {}
    leaders_path = []
    for i in range(3):
        leader_path = path_convert(
            f'env_utils/NGSIM_trajectories/time_window_{scene_length}s/'
            f'{scene_definition[scene_class_id][i][0]}_{scene_definition[scene_class_id][i][1]}/vehicle_{i}.csv'
        )
        leaders_path.append(leader_path)
        leaders_df[i] = pd.read_csv(leader_path)

    # Generate scenario
    queue_start, start_speed, vehicle_names, leader_counts = generate_scenario_NGSIM(
        config_file=config_file,
        aggressive=args["aggressive"],
        cautious=args["cautious"],
        normal=args["normal"],
        use_gui=use_gui,
        sce_name="Env_MultiLane",
        CAV_num=CAV_num,
        HDV_num=HDV_num,
        CAV_penetration=args["penetration_CAV"],
        distribution="uniform",
        scene_info=scene_info,
        leaders_path=leaders_path,
    )

    # ========================================================================
    # 8. Main Simulation Loop
    # ========================================================================
    done = False
    step_count = 0
    performance_metrics = {
        'safety_computation_time': [],
        'mpc_solve_time': [],
        'control_success_rate': [],
        'lateral_decisions': [],
    }

    logger.info("Starting simulation loop...")

    while not done:
        step_count += 1

        # Initialize actions
        actions = {'vehicle': {veh_id: (-1, -1, -1) for veh_id in obs['vehicle'].keys()}}

        if not obs['vehicle']:
            break

        # Get current simulation time
        step = info['step_time'] if 'info' in locals() else 0
        traffic_state = get_traffic_state(obs['vehicle'], lane_ids)

        # Scene recognition
        scene_result = scene_recognition.recognize_scenario(traffic_state)
        new_weight = scene_result['weight_config']

        # ====================================================================
        # Control Leader Vehicles
        # ====================================================================
        for i in range(3):
            [mean_v, std_v] = scene_info[i]
            for j in range(leader_counts[i]):
                vehicle_name = f'Leader_{i}_{j}'
                if vehicle_name in obs['vehicle']:
                    leader = traci.vehicle.getLeader(vehicle_name, 100)
                    if leader and leader[0] in obs['vehicle']:
                        actions['vehicle'][vehicle_name] = (0, -1, -1)
                    else:
                        if j == 0:
                            next_speed = leaders_df[i].loc[step, 'Velocity']
                        else:
                            now_speed = obs['vehicle'][vehicle_name]['speed']
                            next_speed = now_speed + random.uniform(-1, 1)
                        actions['vehicle'][vehicle_name] = (0, -1, next_speed)

            for j in range(10):
                vehicle_name = f'Follower_{i}_{j}'
                if vehicle_name in obs['vehicle']:
                    actions['vehicle'][vehicle_name] = (0, -1, -1)

        # ====================================================================
        # Control CAVs with MPC + Safety Assessment
        # ====================================================================
        for i_CAV in range(CAV_num):
            veh_name = f'CAV_{i_CAV}'

            if veh_name not in obs['vehicle']:
                continue

            # Get ego vehicle state
            veh_pos_x = obs['vehicle'][veh_name]['position'][0]
            veh_pos_y = obs['vehicle'][veh_name]['position'][1]
            veh_speed = obs['vehicle'][veh_name]['speed']
            veh_heading = obs['vehicle'][veh_name]['heading']
            veh_acc = obs['vehicle'][veh_name]['acceleration']
            veh_yaw_rate = 0
            lane_index = obs['vehicle'][veh_name]['lane_index']
            veh_type = 1

            # Get surrounding vehicles
            surrounding_vehicles, surrounding_state = get_surrounding_state(
                obs['vehicle'], veh_name, use_gui
            )

            if len(surrounding_vehicles) not in [8, 12]:
                logger.warning(f'{veh_name}: Invalid surrounding vehicles count: {len(surrounding_vehicles)}')
                continue

            # Format ego state
            front_id = surrounding_vehicles['ego_leader_1']['veh_id']
            front_v = obs['vehicle'][front_id]['speed']
            front_pos_x = obs['vehicle'][front_id]['position'][0]
            front_spacing = front_pos_x - veh_pos_x - 5

            ego_state = format_conversion(
                veh_pos_x, veh_pos_y, veh_speed, veh_heading, veh_acc,
                veh_yaw_rate, lane_index, veh_type, front_v, front_spacing
            )

            # Get traffic state for lateral decision
            ego_lane_v = traffic_state['lane_velocities'][lane_index]
            ego_lane_density = traffic_state['lane_densities'][lane_index]

            left_lane_v = traffic_state['lane_velocities'][lane_index + 1] if lane_index < 2 else None
            left_lane_density = traffic_state['lane_densities'][lane_index + 1] if lane_index < 2 else None

            right_lane_v = traffic_state['lane_velocities'][lane_index - 1] if lane_index > 0 else None
            right_lane_density = traffic_state['lane_densities'][lane_index - 1] if lane_index > 0 else None

            # Compute observation-based lane velocities
            observation_stats = {
                'left_lane_speeds': [],
                'right_lane_speeds': [],
                'ego_lane_speeds': [],
            }
            for sur_id in surrounding_vehicles.keys():
                veh_id = surrounding_vehicles[sur_id]['veh_id']
                if sur_id[:4] == 'left':
                    observation_stats['left_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])
                elif sur_id[:5] == 'right':
                    observation_stats['right_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])
                else:
                    observation_stats['ego_lane_speeds'].append(obs['vehicle'][veh_id]['speed'])

            obs_left_lane_v = np.mean(observation_stats['left_lane_speeds']) if observation_stats['left_lane_speeds'] else None
            obs_right_lane_v = np.mean(observation_stats['right_lane_speeds']) if observation_stats['right_lane_speeds'] else None
            obs_ego_lane_v = np.mean(observation_stats['ego_lane_speeds']) if observation_stats['ego_lane_speeds'] else veh_speed

            now_traffic_state = {
                'v_left': obs_left_lane_v,
                'v_ego': obs_ego_lane_v,
                'v_right': obs_right_lane_v,
                'density_left': left_lane_density,
                'density_ego': ego_lane_density,
                'density_right': right_lane_density
            }

            # Evaluate lane change necessity
            necessity_results = lateral_decision.evaluate_lane_change_necessity(
                now_traffic_state, new_weight, use_sigmoid=True
            )
            necessity_results['keep_lane'] = 1

            # Make lateral decision (simplified for evaluation)
            # For now, assume keep_lane for stable evaluation
            pre_decision = 'keep_lane'

            # Execute MPC control with safety assessment
            try:
                next_speed, control_info = mpc_wrapper.control(
                    ego_state=ego_state,
                    surrounding_state=surrounding_state,
                    surrounding_vehicles=surrounding_vehicles,
                    lateral_decision=pre_decision,
                    traffic_state=now_traffic_state,
                    weights=new_weight
                )

                # Record performance metrics
                performance_metrics['safety_computation_time'].append(
                    control_info['safety_result']['computation_time']
                )
                performance_metrics['mpc_solve_time'].append(
                    control_info['mpc_result'].solve_time
                )
                performance_metrics['control_success_rate'].append(
                    1.0 if control_info['mpc_result'].feasible else 0.0
                )
                performance_metrics['lateral_decisions'].append(pre_decision)

                # Set action
                actions['vehicle'][veh_name] = (0, -1, next_speed)

            except Exception as e:
                logger.error(f"Control failed for {veh_name}: {str(e)}")
                actions['vehicle'][veh_name] = (0, -1, veh_speed)

        # Step simulation
        obs, reward, info, done = tshub_env.step(actions=actions)

        # Render if GUI enabled
        if use_gui and step_count % 10 == 0:
            tshub_env.render(mode='sumo_gui')

        step_time = info["step_time"]
        logger.info(f"Step {step_count}: T={step_time:.1f}s")

    # ========================================================================
    # 9. Save Results and Statistics
    # ========================================================================
    logger.info("Simulation completed. Saving results...")

    results_summary = {
        'safety_method': safety_method_name,
        'total_steps': step_count,
        'avg_safety_computation_time': np.mean(performance_metrics['safety_computation_time']),
        'avg_mpc_solve_time': np.mean(performance_metrics['mpc_solve_time']),
        'control_success_rate': np.mean(performance_metrics['control_success_rate']),
        'scenario_id': scene_class_id,
    }

    # Save results
    import json
    with open(os.path.join(result_folder, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)

    # Save detailed metrics
    metrics_df = pd.DataFrame(performance_metrics)
    metrics_df.to_csv(os.path.join(result_folder, 'performance_metrics.csv'), index=False)

    logger.info(f"Results saved to {result_folder}")
    logger.info(f"Summary: {results_summary}")

    tshub_env._close_simulation()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Load configuration
    yaml_file = "/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/a_multi_lane.yaml"
    with open(yaml_file, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Run evaluation with different safety methods
    safety_methods = [
        "ConflictAssessment",
        "SafetyField",  # Based on Jing et al. IEEE TITS 2022
        "GameTheory",  # Based on Yang et al. IEEE TITS 2023
        # "ArtificialPotentialField",  # Uncomment when implemented
    ]

    for method in safety_methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running evaluation with {method} method")
        logger.info(f"{'='*80}\n")

        try:
            run_evaluation(config, safety_method_name=method)
        except Exception as e:
            logger.error(f"Evaluation failed for {method}: {str(e)}")
            import traceback
            traceback.print_exc()

        logger.info(f"\nCompleted evaluation with {method} method\n")
