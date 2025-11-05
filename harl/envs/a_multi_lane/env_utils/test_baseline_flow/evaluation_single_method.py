import numpy as np
import pandas as pd
import os


class FlowEvaluator:
    """
    Evaluation framework for mixed platoon control strategies as defined in evaluation_1.tex
    Calculates various metrics related to safety, efficiency, stability, and comfort.
    """

    def __init__(self, parameters, eval_weights, method_name,
                 CAV_num, HDV_num, CAV_penetration,
                 vehicle_names, vehicles_data,
                 ttc_data, act_data, crash_data,
                 vehicle_lengths=None, desired_velocity=20.0, output_dir=None):
        """
        Initialize the evaluator with platoon data.

        Parameters:
        -----------
        method_name : str
            Name of the control strategy being evaluated (e.g., 'MPC', 'IDM', 'MARL')
        vehicles_data : dict
            Dictionary where keys are vehicle IDs and values are DataFrames with columns:
            - t: time (s)
            - Position: position (m)
            - v_Vel: velocity (m/s)
            - v_Acc: acceleration (m/s²)
        vehicle_lengths : dict, optional
            Dictionary of vehicle lengths (m) with vehicle IDs as keys
        desired_velocity : float, optional
            Desired velocity for the platoon (m/s)
        """
        self.method_name = method_name
        self.CAV_num = CAV_num
        self.HDV_num = HDV_num
        self.CAV_penetration = CAV_penetration
        self.N = self.CAV_num + self.HDV_num  # Number of vehicles
        self.n = self.CAV_num
        self.vehicle_names = vehicle_names
        self.vehicle_ids = vehicle_names.values()
        self.vehicles_data = vehicles_data
        self.ttc_data = ttc_data
        self.act_data = act_data
        self.crash_data = crash_data
        self.output_dir = output_dir

        # Set default vehicle lengths if not provided
        if vehicle_lengths is None:
            self.vehicle_lengths = {v_id: 5.0 for v_id in self.vehicle_ids}
        else:
            self.vehicle_lengths = vehicle_lengths

        # Parameters
        self.v_des = parameters["max_v"]  # Desired velocity (m/s)
        self.v_max = parameters["max_v"]
        self.s_0 = parameters["minGap"]  # Minimum spacing (m)
        self.tau = parameters["tau"]  # Time gap parameter (s)
        self.TTC_thresh = parameters["safe_warn_threshold"]["TTC"]  # TTC threshold (s)
        self.TTC_max = parameters["max_TTC"]  # Maximum considered TTC (s)
        self.ACT_thresh = parameters["safe_warn_threshold"]["ACT"]
        self.ACT_max = parameters["max_ACT"]
        self.a_max = parameters["max_a"]
        self.j_max = parameters["max_jerk"]  # Maximum comfortable jerk (m/s³)
        self.TFR_max = 3600  # Normalization factor for traffic flow rate (veh/h)

        # Weights for synthesis score
        self.weights = eval_weights

    def calculate_safety_metric(self):
        """
        Calculate TTC Safety Index (TSI) as defined in evaluation_1.tex

        TSI = 1 - (1/n) * sum_{i=1}^{n} (1/T) * sum_{t=0}^{T} max(0, 1 - min(TTC_i(t), TTC_max)/TTC_thresh)^2
        """
        if self.n == 0:
            return 1.0, 1.0  # No CAVs

        total_penalty = 0.0

        for veh_id, ttc_values in self.ttc_data.items():
            # Calculate per-timestep safety penalties
            penalties = np.zeros_like(ttc_values)
            crash_detected = False

            for t in range(len(ttc_values)):
                penalties[t] = max(0, 1 - min(ttc_values[t], self.TTC_max)/self.TTC_thresh) ** 2
                if ttc_values[t] <= 0:
                    crash_detected = True
            avg_penalty = np.mean(penalties)

            # If a crash was detected, ensure the penalty is significant
            if crash_detected:
                avg_penalty = max(avg_penalty, 0.8)  # Minimum 0.8 penalty for any crash

            total_penalty += avg_penalty

            # Debug: Print safety metrics for this vehicle
            # Convert the list to a NumPy array before comparison
            ttc_array = np.array(ttc_values)
            critical_count = np.sum(ttc_array < self.TTC_thresh)
            crash_count = np.sum(ttc_array <= 0.1)
            print(f"Vehicle {veh_id} safety metrics (using TTC evaluation):")
            print(f"  Crashes detected: {crash_count}/{len(ttc_values)} ({crash_count / len(ttc_values) * 100:.1f}%)")
            print(
                f"  Critical TTC events (<{self.TTC_thresh}s): {critical_count}/{len(ttc_values)} ({critical_count / len(ttc_values) * 100:.1f}%)")
            print(f"  Min TTC: {np.min(ttc_array):.2f}s")
            print(f"  Avg TTC: {np.mean(ttc_array):.2f}s")
            print(f"  Avg safety penalty: {avg_penalty:.4f}")

            # Average across all follower vehicles
        avg_vehicle_penalty = total_penalty / self.n

        # Convert to safety index (higher is better)
        tsi = 1 - avg_vehicle_penalty

        # # If any vehicle has a crash, TSI should be low
        # if avg_vehicle_penalty > 0.8:
        #     tsi = max(0, min(tsi, 0.2))  # Cap TSI at 0.2 when severe safety issues exist

        # Debug: Print overall safety index
        print(f"Overall TSI: {tsi:.4f}")

        total_penalty = 0.0

        for veh_id, act_values in self.act_data.items():
            # Calculate per-timestep safety penalties
            penalties = np.zeros_like(act_values)
            crash_detected = False

            for t in range(len(act_values)):
                penalties[t] = max(0, 1 - min(act_values[t], self.ACT_max) / self.ACT_thresh) ** 2
                if act_values[t] <= 0:
                    crash_detected = True
            avg_penalty = np.mean(penalties)

            # If a crash was detected, ensure the penalty is significant
            if crash_detected:
                avg_penalty = max(avg_penalty, 0.8)  # Minimum 0.8 penalty for any crash

            total_penalty += avg_penalty

            # Debug: Print safety metrics for this vehicle
            act_array = np.array(act_values)
            critical_count = np.sum(act_array < self.ACT_thresh)
            crash_count = np.sum(act_array <= 0.1)
            print(f"Vehicle {veh_id} safety metrics (using ACT evaluation):")
            print(f"  Crashes detected: {crash_count}/{len(act_values)} ({crash_count / len(act_values) * 100:.1f}%)")
            print(
                f"  Critical ACT events (<{self.ACT_thresh}s): {critical_count}/{len(act_values)} ({critical_count / len(act_values) * 100:.1f}%)")
            print(f"  Min ACT: {np.min(act_array):.2f}s")
            print(f"  Avg ACT: {np.mean(act_array):.2f}s")
            print(f"  Avg safety penalty: {avg_penalty:.4f}")

            # Average across all follower vehicles
        avg_vehicle_penalty = total_penalty / self.n

        # Convert to safety index (higher is better)
        asi = 1 - avg_vehicle_penalty

        # # If any vehicle has a crash, TSI should be low
        # if avg_vehicle_penalty > 0.8:
        #     asi = max(0, min(asi, 0.2))  # Cap TSI at 0.2 when severe safety issues exist

        # Debug: Print overall safety index
        print(f"Overall ASI: {asi:.4f}")

        return tsi, asi

    def calculate_efficiency_metrics(self):
        """
        Calculate efficiency metrics:
        1. Average Speed Ratio (ASR)
        2. Traffic Flow Rate (TFR)
        """
        # Calculate ASR: average of (v_i / v_des) across all vehicles and time steps
        asr_sum = 0.0
        platoon_vehicles = [v for v in self.vehicle_ids if v != 'Leader']  # Fixed list subtraction issue
        for v_id in platoon_vehicles:
            df = self.vehicles_data[v_id]
            asr_sum += np.mean(df['v_Vel'] / self.v_max)

        asr = asr_sum / self.N

        print("Overall Efficiency metrics:")
        print(f"Average Speed Ratio (ASR): {asr:.4f}")

        # Calculate TFR: 1/T * {sum_{t=0}^{T} [(N * avg_speed(t)) / platoon_length]}, normalized by TFR_max
        tfr_values = []
        leader_id = self.vehicle_names[0]  # First vehicle is the leader
        last_id = self.vehicle_names[len(self.vehicle_names)-1]  # Last vehicle in the platoon
        leader_trajectory = self.vehicles_data['Leader']
        sum_timesteps = len(leader_trajectory)
        time_sampling = leader_trajectory['t'][1] - leader_trajectory['t'][0]
        for t_idx in range(sum_timesteps):
            # Calculate average speed of all vehicles at time t_idx
            t = t_idx * time_sampling  # Assuming self.dt is the time step

            # Collect speeds of all vehicles at this timestamp
            speeds_at_t = []
            for v_id in platoon_vehicles:
                # Find the closest timestamp in the vehicle's data
                df = self.vehicles_data[v_id]
                closest_row = df.iloc[(df['t'] - t).abs().argsort()[0]]
                speeds_at_t.append(closest_row['v_Vel'])

            avg_speed = np.mean(speeds_at_t)

            # Calculate platoon length (distance between leader and last vehicle)
            leader_df = self.vehicles_data[leader_id]
            last_df = self.vehicles_data[last_id]

            # Find positions at the closest timestamp
            leader_pos = leader_df.iloc[(leader_df['t'] - t).abs().argsort()[0]]['Position_x']
            last_pos = last_df.iloc[(last_df['t'] - t).abs().argsort()[0]]['Position_x']

            platoon_length = abs(leader_pos - last_pos)

            # Calculate TFR at this timestamp
            if platoon_length > 0:  # Avoid division by zero
                tfr_t = (self.N * avg_speed) / platoon_length * 3600  # -> veh/h
                tfr_values.append(tfr_t)
            else:
                tfr_values.append(0)

            # Calculate the overall TFR (average over all time steps)
        tfr = np.sum(tfr_values) / sum_timesteps

        # Normalize by TFR_max
        # Assuming TFR_max is provided or calculated elsewhere
        tfr_normalized = tfr / self.TFR_max

        print(f"Average TFR: {tfr_normalized:.4f}")

        return asr, tfr_normalized

    def calculate_stability_metrics(self):
        """
        Calculate stability metrics:
        1. String Stability - Velocity Deviation (SSVD)
        2. String Stability - Spacing Deviation (SSSD)
        3. String Stability Index (SSI) - weighted combination of SSVD and SSSD
        4. Velocity & Spacing Smoothness (VSS)
        """
        # Initialize metrics
        veh_vel_deviation = []
        veh_spacing_deviation = []
        total_smoothness_penalty = 0.0
        stability_Q = np.array([[1, 0], [0, 0.5]])
        all_vehicle_ess = []
        all_ego_vehicle_ess = []
        all_vehicle_stability_r = []
        all_ego_vehicle_stability_r = []

        # Process each vehicle except the leader
        for i, v_id in enumerate(self.vehicle_names.items()):
            # Skip the leader vehicle
            if v_id[1] == 'Leader':
                continue

            # Get ego vehicle data
            ego_df = self.vehicles_data[v_id[1]]

            # Get front vehicle data
            front_id = self.vehicle_names[i - 1]
            front_df = self.vehicles_data[front_id]

            # Add new columns to ego_df for spacing data
            ego_df['Spacing'] = np.zeros(len(ego_df))
            ego_df['Desired_Spacing'] = np.zeros(len(ego_df))
            ego_df['Norm_Spacing'] = np.zeros(len(ego_df))

            norm_vel_diffs = []
            norm_spacing_diffs = []
            time_steps = len(ego_df)

            # Process each time step
            for t_idx in range(time_steps):
                # Velocity calculations
                v_front = front_df.iloc[t_idx]['v_Vel']
                v_ego = ego_df.iloc[t_idx]['v_Vel']
                delta_v = v_ego - v_front
                norm_vel_diff = np.abs(v_ego - v_front) / v_front if v_front > 0 else 1.0
                norm_vel_diffs.append(norm_vel_diff)

                # Spacing calculations
                s_front = front_df.iloc[t_idx]['Position_x']
                s_ego = ego_df.iloc[t_idx]['Position_x']
                spacing = s_front - s_ego - self.vehicle_lengths[v_id[1]]  # Use vehicle-specific length

                # Store spacing values
                ego_df.at[ego_df.index[t_idx], 'Spacing'] = spacing

                # Calculate desired spacing
                spacing_des = self.s_0 + v_ego * self.tau  # Usually based on ego velocity, not front
                ego_df.at[ego_df.index[t_idx], 'Desired_Spacing'] = spacing_des

                # Calculate normalized spacing
                ego_df.at[ego_df.index[t_idx], 'Norm_Spacing'] = spacing / spacing_des

                # Calculate spacing deviation
                delta_s = spacing - spacing_des
                norm_spacing_diff = np.abs(spacing - spacing_des) / spacing_des
                norm_spacing_diffs.append(norm_spacing_diff)

                stability_matric = np.array([delta_s, delta_v])
                e_ss = stability_matric @ stability_Q @ stability_matric.T
                stability_r = np.exp(-e_ss)
                all_vehicle_ess.append(e_ss)
                all_vehicle_stability_r.append(stability_r)
                if v_id[1][:3] == 'CAV':
                    all_ego_vehicle_ess.append(e_ss)
                    all_ego_vehicle_stability_r.append(stability_r)

            mean_vel_deviation = np.mean(norm_vel_diffs)
            veh_vel_deviation.append(mean_vel_deviation)
            mean_spacing_deviation = np.mean(norm_spacing_diffs)
            veh_spacing_deviation.append(mean_spacing_deviation)
            # Store vehicle-specific metrics
            vehicle_idx = i - 1  # Adjust index (assuming leader is index 0)
            if 0 <= vehicle_idx < self.N:  # Ensure index is valid
                # Calculate smoothness metrics
                vel_std = np.std(ego_df['v_Vel']) / self.v_max
                spacing_std = np.std(ego_df['Norm_Spacing'])
                smoothness_penalty = (self.weights['w_v'] * vel_std +
                                      self.weights['w_s'] * spacing_std) / (
                                             self.weights['w_v'] + self.weights['w_s'])

                total_smoothness_penalty += smoothness_penalty

        # Calculate final metrics (higher is better)
        follower_count = sum(1 for v in self.vehicle_names.values() if v != 'Leader')

        print("Overall Stability Metrics:")
        # 1. String Stability - Velocity Deviation (higher is better)
        total_vel_deviation = min(0.8, np.mean(veh_vel_deviation))
        ssvd = 1 - total_vel_deviation

        # 2. String Stability - Spacing Deviation (higher is better)
        total_spacing_deviation = min(0.8, np.mean(veh_spacing_deviation))
        sssd = 1 - total_spacing_deviation

        # 3. String Stability Index - weighted combination
        ssi = self.weights['w_VD'] * ssvd + self.weights['w_SD'] * sssd
        print(f"Overall Stability: {ssi:.4f}")

        # 4. Velocity & Spacing Smoothness (higher is better)
        vss = 1 - total_smoothness_penalty / follower_count if follower_count > 0 else 1.0
        print(f"Overall Stability Smoothness: {vss:.4f}")

        all_control_efficiency = np.mean(all_vehicle_ess)
        cav_control_efficiency = np.mean(all_ego_vehicle_ess)

        all_stability_reward = np.mean(all_vehicle_stability_r)
        cav_stability_reward = np.mean(all_ego_vehicle_stability_r)

        return ssvd, sssd, ssi, vss, all_control_efficiency, cav_control_efficiency, all_stability_reward, cav_stability_reward

    def calculate_comfort_metric(self):
        """
        Calculate Jerk Index (JI) as a measure of comfort:
        JI = 1 - (1/N) * sum_{i=1}^{N} (1/T) * sum_{t=0}^{T} min(1, |j_i(t)|/j_max)
        """
        total_ai_penalty = 0.0
        total_ji_penalty = 0.0
        comfort_R = 0.5
        all_vehicle_comfort_cost = []
        all_ego_vehicle_comfort_cost = []
        all_vehicle_comfort_r = []
        all_ego_vehicle_comfort_r = []
        for i, v_id in enumerate(self.vehicle_names.items()):
            # Skip the leader vehicle
            if v_id[1] == 'Leader':
                continue

            df = self.vehicles_data[v_id[1]]

            df['Jerk'] = np.zeros(len(df))
            time_steps = len(df)

            # Process each time step
            for t_idx in range(time_steps):
                if t_idx == 0:
                    df.at[df.index[t_idx], 'Jerk'] = 0
                    acc_now = df.iloc[t_idx]['v_Acc']
                else:
                    acc_last = df.iloc[t_idx-1]['v_Acc']
                    acc_now = df.iloc[t_idx]['v_Acc']
                    jerk = (acc_now - acc_last) / 0.1
                    df.at[df.index[t_idx], 'Jerk'] = jerk
                acc_cost = comfort_R*((acc_now) ** 2)
                comfort_r = np.exp(-acc_cost)
                all_vehicle_comfort_cost.append(acc_cost)
                all_vehicle_comfort_r.append(comfort_r)
                if v_id[1][:3] == 'CAV':
                    all_ego_vehicle_comfort_cost.append(acc_cost)
                    all_ego_vehicle_comfort_r.append(comfort_r)

            # Calculate normalized jerks, capped at 1
            norm_jerks = np.minimum(self.j_max, np.abs(df['Jerk']) / self.j_max)
            # Average over time
            avg_norm_jerk = np.mean(norm_jerks)
            total_ji_penalty += avg_norm_jerk

            nrom_accs = np.minimum(self.a_max, np.abs(df['v_Acc']) / self.a_max)
            avg_nrom_acc = np.mean(nrom_accs)
            total_ai_penalty += avg_nrom_acc

        # Average across all vehicles and convert to JI (higher is better)
        ji = 1 - total_ji_penalty / self.N
        ai = 1 - total_ai_penalty / self.N

        print("Overall Comfort Metrics:")
        print("Overall Jerk Index (JI): {:.4f}".format(ji))
        print("Overall AI Penalty: {:.4f}".format(ai))

        all_comfort_cost = np.mean(all_vehicle_comfort_cost)
        cav_comfort_cost = np.mean(all_ego_vehicle_comfort_cost)
        all_comfort_reward = np.mean(all_vehicle_comfort_r)
        cav_comfort_reward = np.mean(all_ego_vehicle_comfort_r)

        return ji, ai, all_comfort_cost, cav_comfort_cost, all_comfort_reward, cav_comfort_reward

    def calculate_synthesis_score(self, tsi, asi, asr, tfr, ssi, vss, comfort):
        """
        Calculate the overall synthesis score using the weighted formula from evaluation_1.tex:

        S_synthesis = w1*TSI + w2*(w21*ASR + w22*TFR) + w3*(w31*SSI + w32*VSS) + w4*JI
        """
        # Efficiency combined score
        efficiency = self.weights['w21'] * asr + self.weights['w22'] * tfr

        # Stability combined score
        stability = self.weights['w31'] * ssi + self.weights['w32'] * vss

        # Overall synthesis score
        synthesis_ttc = (self.weights['w1'] * tsi +
                     self.weights['w2'] * efficiency +
                     self.weights['w3'] * stability +
                     self.weights['w4'] * comfort)

        synthesis_act = (self.weights['w1'] * asi +
                     self.weights['w2'] * efficiency +
                     self.weights['w3'] * stability +
                     self.weights['w4'] * comfort)

        return synthesis_ttc, synthesis_act, efficiency, stability

    def evaluate(self):
        """
        Perform a complete evaluation and return results as a dictionary.
        Also saves results to a CSV file.
        """
        # Calculate all metrics
        tsi, asi = self.calculate_safety_metric()
        asr, tfr = self.calculate_efficiency_metrics()
        ssvd, sssd, ssi, vss, all_control_efficiency, cav_control_efficiency, all_stability_reward, cav_stability_reward = self.calculate_stability_metrics()
        ji, ai, all_comfort_cost, cav_comfort_cost, all_comfort_reward, cav_comfort_reward = self.calculate_comfort_metric()

        # Calculate synthesis score
        synthesis_ttc, synthesis_act, efficiency, stability = self.calculate_synthesis_score(tsi, asi, asr, tfr, ssi, vss, ji)

        # Compile results
        results = {
            'Method': self.method_name,
            'Safety_TSI': tsi,
            'Safety_ASI': asi,
            'Efficiency_ASR': asr,
            'Efficiency_TFR': tfr,
            'Efficiency_Combined': efficiency,
            'Stability_SSVD': ssvd,
            'Stability_SSSD': sssd,
            'Stability_SSI': ssi,
            'Stability_VSS': vss,
            'Stability_Combined': stability,
            'Comfort_JI': ji,
            'Synthesis_Score_TTC': synthesis_ttc,
            'Synthesis_Score_ACT': synthesis_act,
            'Control_Efficiency_All': all_control_efficiency,
            'Control_Efficiency_Cav': cav_control_efficiency,
            'Comfort_Cost_All': all_comfort_cost,
            'Comfort_Cost_Cav': cav_comfort_cost,
            'Stability_Reward_All': all_stability_reward,
            'Stability_Reward_Cav': cav_stability_reward,
            'Comfort_Reward_All': all_comfort_reward,
            'Comfort_Reward_Cav': cav_comfort_reward,
        }

        print(f"Evaluation Results for {self.method_name}:")
        print(f"Safety TSI: {tsi:.4f}")
        print(f"Safety ASI: {asi:.4f}")
        print(f"Efficiency ASR: {asr:.4f}")
        print(f"Efficiency TFR: {tfr:.4f}")
        print(f"Efficiency Combined: {efficiency:.4f}")
        print(f"Stability SSSI: {ssi:.4f}")
        print(f"Stability VSS: {vss:.4f}")
        print(f"Stability Combined: {stability:.4f}")
        print(f"Comfort JI: {ji:.4f}")
        print(f"Synthesis Score (Using TTC): {synthesis_ttc:.4f}")
        print(f"Synthesis Score (Using ACT): {synthesis_act:.4f}")
        print(f"Control Efficiency (All): {all_control_efficiency:.4f}")
        print(f"Control Efficiency (CAV): {cav_control_efficiency:.4f}")
        print(f"Comfort Cost (All): {all_comfort_cost:.4f}")
        print(f"Comfort Cost (CAV): {cav_comfort_cost:.4f}")
        print(f"Stability Reward (All): {all_stability_reward:.4f}")
        print(f"Stability Reward (CAV): {cav_stability_reward:.4f}")
        print(f"Comfort Reward (All): {all_comfort_reward:.4f}")
        print(f"Comfort Reward (CAV): {cav_comfort_reward:.4f}")

        # Save to CSV
        df = pd.DataFrame([results])
        csv_filename = os.path.join(self.output_dir, f"evaluation_{self.method_name}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Evaluation results saved to {csv_filename}")

        return results

def load_vehicle_data(output_dir, vehicle_names):
    """
    Load vehicle trajectory data from CSV files.

    Parameters:
    -----------
    output_dir : str
        Directory containing vehicle trajectory CSV files
    vehicle_names : dict
        Dictionary mapping vehicle IDs to vehicle names

    Returns:
    --------
    dict
        Dictionary with vehicle IDs as keys and DataFrames as values
    """
    vehicles_data = {}
    for vehicle_id, vehicle_name in vehicle_names.items():
        csv_file = os.path.join(output_dir, f"trajectory_csv/{vehicle_id}_{vehicle_name}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            vehicles_data[vehicle_name] = df

    return vehicles_data

def evaluate_method(parameters, eval_weights, method_name, output_dir,
                    CAV_num, HDV_num, CAV_penetration,
                    vehicle_names, desired_velocity,
                    ttc_record, act_record, crash_record):
    """
    Evaluate a control method using the PlatoonEvaluator.

    Parameters:
    -----------
    method_name : str
        Name of the control method
    output_dir : str
        Directory containing vehicle trajectory CSV files
    vehicle_names : dict
        Dictionary mapping vehicle IDs to vehicle names
    desired_velocity : float, optional
        Desired velocity for the platoon

    Returns:
    --------
    dict
        Evaluation results
    """
    # Load vehicle data

    # Load vehicle data
    vehicles_data = load_vehicle_data(output_dir, vehicle_names)

    if not vehicles_data:
        print(f"No vehicle data found in {output_dir}")
        return None

    # Create evaluator
    evaluator = PlatoonEvaluator(parameters, eval_weights, method_name,
                                 CAV_num, HDV_num, CAV_penetration,
                                 vehicle_names, vehicles_data,
                                 ttc_record, act_record, crash_record,
                                 desired_velocity=desired_velocity,
                                 output_dir=output_dir)

    # Perform evaluation
    results = evaluator.evaluate()

    return results