# conflict_assessment_controller.py
"""
冲突评估主控制器
Conflict Assessment Main Controller
整合轨迹预测与安全评估模块的各个子模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, Direction, LateralDecision
)
from harl.envs.a_multi_lane.hierarchical_controller.trajectory_generator import TrajectoryGenerator
from harl.envs.a_multi_lane.hierarchical_controller.envelope_generator import EnvelopeGenerator
from harl.envs.a_multi_lane.hierarchical_controller.probability_generator import ProbabilityDistributionGenerator
from harl.envs.a_multi_lane.hierarchical_controller.cooperation_evaluator import CooperationCapabilityEvaluator
from typing import Tuple, List, Dict, Optional

class ConflictAssessmentController:
    """冲突评估主控制器"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化冲突评估控制器

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params

        # 初始化子模块
        self.trajectory_generator = TrajectoryGenerator(system_params)
        self.envelope_generator = EnvelopeGenerator(system_params)
        self.probability_generator = ProbabilityDistributionGenerator(system_params)
        self.cooperation_evaluator = CooperationCapabilityEvaluator(system_params)

    def setup_13_vehicle_scenario(self,
                                  ego_state: VehicleState,
                                  surrounding_vehicles: Dict[str, VehicleState]) -> Dict:
        """
        设置13车局部交通场景

        Args:
            ego_state: ego车辆状态
            surrounding_vehicles: 周围车辆状态字典
                键格式: "{direction}_{adjacency}"
                例如: "EF_1", "LR_2" 等

        Returns:
            场景配置字典
        """
        scenario_config = {
            'ego': {
                'state': ego_state,
                'vehicle_type': VehicleType.CAV,
                'cooperation_intent': 1.0
            }
        }

        # 解析周围车辆信息
        for vehicle_id, vehicle_state in surrounding_vehicles.items():
            # 解析方向和邻接类型
            try:
                direction_str, adjacency_str = vehicle_id.split('_')
                direction = Direction(direction_str)
                adjacency = int(adjacency_str)  # 1=主邻接, 2=次邻接
            except (ValueError, KeyError):
                print(f"警告: 无法解析车辆ID {vehicle_id}, 跳过")
                continue

            # 计算协同意图
            cooperation_intent = self.probability_generator.compute_vehicle_cooperation_intent(
                vehicle_state.vehicle_type, direction
            )

            scenario_config[vehicle_id] = {
                'state': vehicle_state,
                'direction': direction,
                'adjacency': adjacency,
                'vehicle_type': vehicle_state.vehicle_type,
                'cooperation_intent': cooperation_intent
            }

        return scenario_config

    def generate_ego_trajectory(self,
                                scenario_config: Dict,
                                pre_decision: str,
                                target_speed: float,):
        # 生成ego车辆轨迹
        ego_state = scenario_config['ego']['state']
        ego_target_speed = target_speed

        if pre_decision == 'keep_lane':
            # 车道保持轨迹
            time_array, coeffs = self.trajectory_generator.generate_lane_keeping_trajectory(
                ego_state, ego_target_speed, self.params.prediction_horizon
            )
            x_pos, x_vel, x_acc = self.trajectory_generator.evaluate_polynomial_trajectory(
                coeffs, time_array
            )

            ego_trajectory = {
                'time': time_array,
                'longitudinal': {
                    'position': x_pos,
                    'velocity': x_vel,
                    'acceleration': x_acc,
                    'coefficients': coeffs
                },
                'lateral': {
                    'position': np.full_like(time_array, ego_state.y),
                    'velocity': np.zeros_like(time_array),
                    'acceleration': np.zeros_like(time_array)
                },
                'trajectory_type': 'lane_keeping'
            }
        else:
            # 变道轨迹
            lane_change_time = min(3.0, self.params.prediction_horizon)

            # 确定目标横向位置
            if pre_decision == 'pre_change_left':
                target_y = ego_state.y + self.params.lane_width
            else:  # RIGHT
                target_y = ego_state.y - self.params.lane_width

            trajectory = self.trajectory_generator.generate_complete_lane_change_trajectory(
                ego_state, target_y, ego_target_speed, lane_change_time
            )
            ego_trajectory = {**trajectory, 'trajectory_type': 'lane_change'}

        return ego_trajectory

    def generate_vehicle_trajectories(self,
                                      scenario_config: Dict,
                                      lateral_decision: str,
                                      vehicles_to_analysis: List,
                                      target_speeds: Optional[Dict[str, float]] = None) -> Dict:
        """
        生成所有车辆的轨迹

        Args:
            scenario_config: 场景配置
            lateral_decision: ego车辆的横向决策
            target_speeds: 各车辆目标速度字典

        Returns:
            轨迹字典
        """
        trajectories = {}

        if target_speeds is None:
            target_speeds = {}

        # 生成周围车辆轨迹（假设都为车道保持）
        for vehicle_id, vehicle_config in scenario_config.items():
            if vehicle_id not in vehicles_to_analysis:
                continue

            vehicle_state = vehicle_config['state']
            target_speed = target_speeds.get(vehicle_id, vehicle_state.front_v)

            # 周围车辆默认车道保持
            time_array, coeffs = self.trajectory_generator.generate_lane_keeping_trajectory(
                vehicle_state, target_speed, self.params.prediction_horizon
            )
            x_pos, x_vel, x_acc = self.trajectory_generator.evaluate_polynomial_trajectory(
                coeffs, time_array
            )

            trajectories[vehicle_id] = {
                'time': time_array,
                'longitudinal': {
                    'position': x_pos,
                    'velocity': x_vel,
                    'acceleration': x_acc,
                    'coefficients': coeffs
                },
                'lateral': {
                    'position': np.full_like(time_array, vehicle_state.y),
                    'velocity': np.zeros_like(time_array),
                    'acceleration': np.zeros_like(time_array)
                },
                'trajectory_type': 'lane_keeping'
            }

        return trajectories

    def generate_vehicle_envelopes(self,
                                   trajectories: Dict,
                                   scenario_config: Dict) -> Dict:
        """
        生成所有车辆的轨迹包络线

        Args:
            trajectories: 轨迹字典
            scenario_config: 场景配置

        Returns:
            包络线字典
        """
        envelopes = {}

        for vehicle_id, trajectory in trajectories.items():
            if vehicle_id not in scenario_config:
                continue

            vehicle_type = scenario_config[vehicle_id]['vehicle_type']

            # 提取轨迹数据
            time_array = trajectory['time']
            x_ref = trajectory['longitudinal']['position']
            v_ref = trajectory['longitudinal']['velocity']
            a_ref = trajectory['longitudinal']['acceleration']

            # 初始状态
            initial_x = x_ref[0]
            initial_v = v_ref[0]
            initial_a = a_ref[0]

            # 生成包络线
            envelope = self.envelope_generator.generate_comprehensive_envelope(
                initial_x, initial_v, initial_a, x_ref, v_ref, a_ref,
                time_array, vehicle_type
            )

            envelopes[vehicle_id] = envelope

        return envelopes

    def visualize_multi_vehicle_envelopes(self,
                                          envelopes: Dict,
                                          trajectories: Dict,
                                          scenario_config: Dict,
                                          ego_trajectory: Dict,
                                          pre_decision: str,
                                          vehicle_colors: Dict,
                                          save_path: str = None):
        """
        根据横向决策可视化相关车辆的包络线

        Args:
            envelopes: 所有车辆包络线字典
            trajectories: 所有车辆轨迹字典
            scenario_config: 场景配置
            ego_trajectory: 主车轨迹
            pre_decision: 横向预决策
            save_path: 保存路径前缀
        """
        # 定义车辆位置映射
        vehicle_groups = {
            'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
            'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
            'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
        }

        ego_color = '#2c2c54'  # 主车轨迹 - 深蓝色

        # 根据决策确定需要显示的车道
        if pre_decision == 'keep_lane':
            # 保持车道：只显示当前车道
            lanes_to_show = [('Current Lane', vehicle_groups['current_lane'])]
            fig_size = (14, 8)
            subplot_layout = (1, 1)

        elif pre_decision == 'pre_change_left':
            # 向左变道：显示当前车道和左侧车道
            lanes_to_show = [
                ('Current Lane', vehicle_groups['current_lane']),
                ('Target Left Lane', vehicle_groups['left_lane'])
            ]
            fig_size = (16, 12)
            subplot_layout = (2, 1)

        elif pre_decision == 'pre_change_right':
            # 向右变道：显示当前车道和右侧车道
            lanes_to_show = [
                ('Current Lane', vehicle_groups['current_lane']),
                ('Target Right Lane', vehicle_groups['right_lane'])
            ]
            fig_size = (16, 12)
            subplot_layout = (2, 1)

        # 创建图形
        if len(lanes_to_show) == 1:
            fig, ax = plt.subplots(figsize=fig_size)
            axes = [ax]
        else:
            fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=fig_size)
            if not isinstance(axes, np.ndarray):
                axes = [axes]

        # 为每个车道创建子图
        for idx, (lane_name, vehicle_ids) in enumerate(lanes_to_show):
            ax = axes[idx]

            # 绘制主车轨迹（每个子图都包含）
            ego_time = ego_trajectory['time']
            ego_x = ego_trajectory['longitudinal']['position']

            ax.plot(ego_time, ego_x, '-', color=ego_color, linewidth=4,
                    label='Ego Vehicle Trajectory', alpha=0.9, zorder=10)

            # 标记主车起始点
            ax.plot(ego_time[0], ego_x[0], 'o', color='darkgreen',
                    markersize=12, markeredgecolor='black', markeredgewidth=2,
                    label='Ego Start Point', zorder=11)

            # 绘制该车道内每个车辆的包络线
            vehicles_plotted = 0
            for vehicle_id in vehicle_ids:
                if vehicle_id in envelopes:
                    envelope = envelopes[vehicle_id]
                    color = vehicle_colors.get(vehicle_id, '#888888')

                    time_array = envelope['time']

                    # # 绘制蒙特卡洛包络（主要关注的不确定性）
                    # ax.fill_between(time_array,
                    #                 envelope['monte_carlo']['min'],
                    #                 envelope['monte_carlo']['max'],
                    #                 alpha=0.6, color=color,
                    #                 label=f'{vehicle_id} MC Envelope',
                    #                 edgecolor='black', linewidth=1.5)

                    # 绘制舒适性包络边界（虚线）
                    ax.plot(time_array, envelope['comfort']['min'], '-.',
                            color=color, linewidth=1.5, alpha=0.8)
                    ax.plot(time_array, envelope['comfort']['max'], '-.',
                            color=color, linewidth=1.5, alpha=0.8)

                    # 绘制物理可达包络边界（虚线）
                    ax.plot(time_array, envelope['physical']['min'], '--',
                            color=color, linewidth=1.0, alpha=0.6)
                    ax.plot(time_array, envelope['physical']['max'], '--',
                            color=color, linewidth=1.0, alpha=0.6)

                    # 绘制参考轨迹
                    if vehicle_id in trajectories:
                        ref_x = trajectories[vehicle_id]['longitudinal']['position']
                        ax.plot(time_array, ref_x, '-', color=color,
                                linewidth=2, alpha=0.9,
                                label=f'{vehicle_id} Reference')

                    vehicles_plotted += 1

            # 设置子图属性
            ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Longitudinal Position [m]', fontsize=12, fontweight='bold')
            ax.set_title(f'{lane_name} - Vehicle Envelopes\n'
                         f'Decision: {pre_decision} ({vehicles_plotted} vehicles)',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            # 添加车道信息文本框
            lane_info = self.get_lane_info_text(vehicle_ids, envelopes, trajectories)
            ax.text(0.02, 0.98, lane_info, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

        # 设置整体标题
        decision_text = {
            'keep_lane': "Lane Keeping",
            'pre_change_left': "Left Lane Change",
            'pre_change_right': "Right Lane Change"
        }

        fig.suptitle(f'Multi-Vehicle Envelope Analysis - {decision_text[pre_decision]}\n'
                     f'13-Vehicle Scenario (1 Ego + 6 Primary + 6 Secondary Neighbors)',
                     fontsize=16, fontweight='bold')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_path:
            filename = f"{save_path}_{pre_decision}_envelopes.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Multi-vehicle envelopes saved to: {filename}")

        plt.show()

    def get_lane_info_text(self,
                           vehicle_ids: List[str],
                           envelopes: Dict,
                           trajectories: Dict) -> str:
        """
        获取车道信息文本

        Args:
            vehicle_ids: 车辆ID列表
            envelopes: 包络线数据
            trajectories: 轨迹数据

        Returns:
            格式化的信息文本
        """
        active_vehicles = []
        total_uncertainty = 0

        for vehicle_id in vehicle_ids:
            if vehicle_id in envelopes:
                active_vehicles.append(vehicle_id)
                # 计算平均不确定性宽度
                envelope = envelopes[vehicle_id]
                # mc_width = np.mean(envelope['monte_carlo']['max'] -
                #                    envelope['monte_carlo']['min'])
                # total_uncertainty += mc_width

        # avg_uncertainty = total_uncertainty / len(active_vehicles) if active_vehicles else 0

        info_text = f"Lane Statistics:\n"
        info_text += f"• Active Vehicles: {len(active_vehicles)}/{len(vehicle_ids)}\n"
        info_text += f"• Vehicle IDs: {', '.join(active_vehicles)}\n"
        # info_text += f"• Avg Uncertainty: {avg_uncertainty:.2f} m\n"

        return info_text

    def discretize_space_time(self,
                              prediction_horizon: float,
                              x_min: float,
                              x_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        离散化时空网格

        Args:
            prediction_horizon: 预测周期 [s]
            x_min: 最小位置 [m]
            x_max: 最大位置 [m]

        Returns:
            (时间网格, 空间网格)
        """
        # 时间离散化
        time_steps = int(prediction_horizon / self.params.dt) + 1
        time_grid = np.linspace(0, prediction_horizon, time_steps)

        # 空间离散化
        dx = 2.5  # 空间分辨率 [m]
        x_steps = int((x_max - x_min) / dx) + 1
        space_grid = np.linspace(x_min, x_max, x_steps)

        return time_grid, space_grid

    def generate_probability_distributions(self,
                                           envelopes: Dict,
                                           scenario_config: Dict,
                                           ego_trajectory: Dict,
                                           vehicles_to_analysis: List) -> Dict:
        """
        生成所有车辆的概率分布

        Args:
            envelopes: 包络线字典
            scenario_config: 场景配置

        Returns:
            概率分布字典
        """
        # 提取车辆类型和方向信息
        vehicle_types = {}
        vehicle_directions = {}
        vehicle_envelopes = {}
        space_range = {
            'min': [],
            'max': []
        }

        for vehicle_id, config in scenario_config.items():
            if vehicle_id not in vehicles_to_analysis:
                continue

            vehicle_types[vehicle_id] = config['vehicle_type']
            vehicle_directions[vehicle_id] = config['direction']
            if vehicle_id in envelopes:
                vehicle_envelopes[vehicle_id] = envelopes[vehicle_id]
                min_x = min(np.min(vehicle_envelopes[vehicle_id]['physical']['min']),
                            np.min(vehicle_envelopes[vehicle_id]['comfort']['min']))
                max_x = max(np.max(vehicle_envelopes[vehicle_id]['physical']['max']),
                            np.max(vehicle_envelopes[vehicle_id]['comfort']['max']))
                space_range['min'].append(min_x)
                space_range['max'].append(max_x)
                time_array = vehicle_envelopes[vehicle_id]['time']

        # 确定空间范围
        x_min = np.min(space_range['min'])
        x_max = np.max(space_range['max'])

        # 离散化时空网格
        time_grid, space_grid = self.discretize_space_time(
            time_array[-1], x_min - 10, x_max + 10
        )

        # 生成概率分布
        probability_distributions = self.probability_generator.generate_multi_vehicle_probability_distributions(
            vehicle_envelopes, vehicle_types, vehicle_directions, ego_trajectory, time_grid, space_grid
        )

        return probability_distributions

    def visualize_multi_vehicle_probability_distributions(self,
                                                          probability_distributions: Dict,
                                                          scenario_config: Dict,
                                                          ego_trajectory: Dict,
                                                          pre_decision: str,
                                                          save_path: str = None):
        """
        根据横向决策可视化相关车辆的概率分布 - 离散化网格填充方式

        Args:
            probability_distributions: 所有车辆概率分布字典
            scenario_config: 场景配置
            ego_trajectory: 主车轨迹
            pre_decision: 横向预决策
            save_path: 保存路径前缀
        """
        # 定义车辆位置映射
        vehicle_groups = {
            'current_lane': ['EF_1', 'EF_2', 'ER_1', 'ER_2'],
            'left_lane': ['LF_1', 'LF_2', 'LR_1', 'LR_2'],
            'right_lane': ['RF_1', 'RF_2', 'RR_1', 'RR_2']
        }

        # 定义车辆颜色映射 - 每个车辆独特颜色
        vehicle_colors = {
            'EF_1': '#1f77b4',  # 蓝色
            'EF_2': '#ff7f0e',  # 橙色
            'ER_1': '#2ca02c',  # 绿色
            'ER_2': '#d62728',  # 红色
            'LF_1': '#9467bd',  # 紫色
            'LF_2': '#8c564b',  # 棕色
            'LR_1': '#e377c2',  # 粉色
            'LR_2': '#7f7f7f',  # 灰色
            'RF_1': '#bcbd22',  # 橄榄绿
            'RF_2': '#17becf',  # 青色
            'RR_1': '#ff9896',  # 浅红色
            'RR_2': '#98df8a'  # 浅绿色
        }

        # 根据决策确定需要显示的车道
        if pre_decision == 'keep_lane':
            lanes_to_show = [('Current Lane', vehicle_groups['current_lane'])]
            fig_size = (14, 8)
            subplot_layout = (1, 1)
        elif pre_decision == 'pre_change_left':
            lanes_to_show = [
                ('Current Lane', vehicle_groups['current_lane']),
                ('Target Left Lane', vehicle_groups['left_lane'])
            ]
            fig_size = (16, 12)
            subplot_layout = (2, 1)
        elif pre_decision == 'pre_change_right':
            lanes_to_show = [
                ('Current Lane', vehicle_groups['current_lane']),
                ('Target Right Lane', vehicle_groups['right_lane'])
            ]
            fig_size = (16, 12)
            subplot_layout = (2, 1)

        # 创建图形
        if len(lanes_to_show) == 1:
            fig, ax = plt.subplots(figsize=fig_size)
            axes = [ax]
        else:
            fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=fig_size)
            if not isinstance(axes, np.ndarray):
                axes = [axes]

        # 为每个车道创建子图
        for idx, (lane_name, vehicle_ids) in enumerate(lanes_to_show):
            ax = axes[idx]

            # 获取该车道所有车辆的概率分布数据，确定统一的离散化范围
            time_range, space_range, dt, dx = self.get_unified_discretization_range(
                probability_distributions, vehicle_ids
            )

            # 绘制该车道内每个车辆的概率分布
            vehicles_plotted = 0
            legend_elements = []

            for vehicle_id in vehicle_ids:
                if vehicle_id in probability_distributions:
                    prob_data = probability_distributions[vehicle_id]['probability_distribution']

                    # 使用离散化网格填充方式绘制
                    legend_element = self.plot_vehicle_probability_grid_fill(
                        ax, prob_data, vehicle_id, vehicle_colors,
                        time_range, space_range, dt, dx
                    )
                    if legend_element:
                        legend_elements.append(legend_element)
                    vehicles_plotted += 1

            # 绘制主车轨迹（在概率分布之后，确保在最上层）
            ego_time = ego_trajectory['time']
            ego_x = ego_trajectory['longitudinal']['position']

            ax.plot(ego_time, ego_x, '-', color='black', linewidth=4,
                    label='Ego Vehicle Trajectory', alpha=1.0, zorder=100)

            # 标记主车起始点
            ax.plot(ego_time[0], ego_x[0], 'o', color='darkgreen',
                    markersize=12, markeredgecolor='white', markeredgewidth=2,
                    label='Ego Start Point', zorder=101)

            # 设置子图属性
            ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Longitudinal Position [m]', fontsize=12, fontweight='bold')
            ax.set_title(f'{lane_name} - Vehicle Probability Distributions\n'
                         f'Decision: {pre_decision} ({vehicles_plotted} vehicles)',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

            # 设置坐标轴范围
            ax.set_xlim(time_range)
            ax.set_ylim(space_range)

            # 合并图例
            ego_handles, ego_labels = ax.get_legend_handles_labels()
            all_handles = ego_handles + legend_elements
            all_labels = ego_labels + [elem.get_label() for elem in legend_elements]

            ax.legend(handles=all_handles, labels=all_labels,
                      bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            # 添加车道信息文本框
            lane_info = self.get_lane_probability_info_text_grid(
                vehicle_ids, probability_distributions, vehicle_colors
            )
            ax.text(1.02, 0.58, lane_info, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))

        # 设置整体标题
        decision_text = {
            'keep_lane': "Lane Keeping",
            'pre_change_left': "Left Lane Change",
            'pre_change_right': "Right Lane Change"
        }

        fig.suptitle(f'Multi-Vehicle Probability Distribution Analysis - {decision_text[pre_decision]}\n'
                     f'Grid-Based Discrete Probability Visualization',
                     fontsize=16, fontweight='bold')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_path:
            filename = f"{save_path}_{pre_decision}_probability_distributions_grid.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Grid-based probability distributions saved to: {filename}")

        plt.show()

    def get_unified_discretization_range(self, probability_distributions: Dict,
                                         vehicle_ids: List[str]) -> tuple:
        """
        获取统一的离散化范围

        Args:
            probability_distributions: 概率分布字典
            vehicle_ids: 车辆ID列表

        Returns:
            (time_range, space_range, dt, dx): 时间范围、空间范围和步长
        """
        all_time_grids = []
        all_space_grids = []

        # 收集所有车辆的时空网格
        for vehicle_id in vehicle_ids:
            if vehicle_id in probability_distributions:
                prob_data = probability_distributions[vehicle_id]['probability_distribution']
                all_time_grids.extend(prob_data['time_grid'])
                all_space_grids.extend(prob_data['space_grid'])

        if not all_time_grids or not all_space_grids:
            # 默认范围
            return (0, 10), (0, 100), 0.1, 0.5

        # 计算统一范围
        time_min, time_max = min(all_time_grids), max(all_time_grids)
        space_min, space_max = min(all_space_grids), max(all_space_grids)

        # 计算网格步长（基于第一个有效车辆的数据）
        first_valid_vehicle = None
        for vehicle_id in vehicle_ids:
            if vehicle_id in probability_distributions:
                first_valid_vehicle = vehicle_id
                break

        if first_valid_vehicle:
            prob_data = probability_distributions[first_valid_vehicle]['probability_distribution']
            time_grid = prob_data['time_grid']
            space_grid = prob_data['space_grid']
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.1
            dx = space_grid[1] - space_grid[0] if len(space_grid) > 1 else 0.5
        else:
            dt, dx = 0.1, 0.5

        return (time_min, time_max), (space_min, space_max), dt, dx

    def plot_vehicle_probability_grid_fill(self, ax, prob_data: Dict, vehicle_id: str,
                                           vehicle_colors: Dict, time_range: tuple,
                                           space_range: tuple, dt: float, dx: float):
        """
        使用网格填充方式绘制车辆概率分布

        Args:
            ax: matplotlib轴对象
            prob_data: 概率分布数据
            vehicle_id: 车辆ID
            vehicle_colors: 车辆颜色字典
            time_range: 时间范围
            space_range: 空间范围
            dt: 时间步长
            dx: 空间步长

        Returns:
            legend_element: 图例元素
        """
        time_grid = prob_data['time_grid']
        position_grid = prob_data['space_grid']
        prob_matrix = prob_data['total_probability']

        # 获取车辆颜色
        base_color = vehicle_colors.get(vehicle_id, '#808080')
        base_rgba = to_rgba(base_color)

        # 概率阈值，小于此值的概率视为0（透明）
        prob_threshold = 1e-6

        # 记录绘制的概率块数量和最大概率
        drawn_blocks = 0
        max_probability = 0
        max_prob_location = None

        # 遍历每个离散块
        for i, t in enumerate(time_grid):
            for j, x in enumerate(position_grid):
                prob_value = prob_matrix[i, j]

                # 只绘制概率大于阈值的块
                if prob_value > prob_threshold:
                    # 计算网格块的边界
                    t_left = t - dt / 2
                    t_right = t + dt / 2
                    x_bottom = x - dx / 2
                    x_top = x + dx / 2

                    # 创建矩形块，alpha值等于概率值
                    alpha = min(prob_value * 2, 1.0)  # 可以调整这个系数来控制透明度映射

                    rect = patches.Rectangle(
                        (t_left, x_bottom),
                        dt, dx,
                        facecolor=base_color,
                        alpha=alpha,
                        edgecolor='none',
                        zorder=10
                    )
                    ax.add_patch(rect)
                    drawn_blocks += 1

                    # 记录最大概率位置
                    if prob_value > max_probability:
                        max_probability = prob_value
                        max_prob_location = (t, x)

        # # 标记最大概率点
        # if max_prob_location:
        #     ax.plot(max_prob_location[0], max_prob_location[1],
        #             marker='x', color='white', markersize=12,
        #             markeredgewidth=3, markeredgecolor=base_color,
        #             zorder=50)

        # 创建图例元素
        if drawn_blocks > 0:
            from matplotlib.lines import Line2D
            legend_element = Line2D([0], [0], marker='s', color='w',
                                    markerfacecolor=base_color, markersize=10,
                                    label=f'{vehicle_id} (Max: {max_probability:.3f})',
                                    alpha=0.8)
            return legend_element

        return None

    def get_lane_probability_info_text_grid(self, vehicle_ids: List[str],
                                            probability_distributions: Dict,
                                            vehicle_colors: Dict) -> str:
        """
        获取网格填充方式的车道概率分布信息

        Args:
            vehicle_ids: 车辆ID列表
            probability_distributions: 概率分布数据
            vehicle_colors: 车辆颜色字典

        Returns:
            格式化的信息文本
        """
        active_vehicles = []
        vehicle_stats = {}
        total_blocks = 0

        for vehicle_id in vehicle_ids:
            if vehicle_id in probability_distributions:
                active_vehicles.append(vehicle_id)
                prob_data = probability_distributions[vehicle_id]['probability_distribution']
                prob_matrix = prob_data['total_probability']

                # 统计有效概率块数量
                prob_threshold = 1e-6
                valid_blocks = np.sum(prob_matrix > prob_threshold)
                total_blocks += valid_blocks

                # 计算统计量
                max_prob = np.max(prob_matrix)
                mean_prob = np.mean(prob_matrix[prob_matrix > prob_threshold]) if valid_blocks > 0 else 0

                # 计算概率分布的熵
                prob_matrix_safe = prob_matrix + 1e-12
                entropy = -np.sum(prob_matrix * np.log(prob_matrix_safe))

                vehicle_stats[vehicle_id] = {
                    'blocks': valid_blocks,
                    'max_prob': max_prob,
                    'mean_prob': mean_prob,
                    'entropy': entropy,
                    'color': vehicle_colors.get(vehicle_id, '#808080')
                }

        # 构建信息文本
        info_text = f"Grid-Based Probability Stats:\n"
        info_text += f"• Active Vehicles: {len(active_vehicles)}/{len(vehicle_ids)}\n"
        info_text += f"• Total Probability Blocks: {total_blocks}\n\n"

        # 显示每个车辆的统计信息
        info_text += "Vehicle Details:\n"
        for vehicle_id in active_vehicles[:6]:  # 限制显示数量
            stats = vehicle_stats[vehicle_id]
            info_text += f"• {vehicle_id}: {stats['blocks']} blocks\n"
            info_text += f"  Max={stats['max_prob']:.3f}, Avg={stats['mean_prob']:.3f}\n"

        if len(active_vehicles) > 6:
            info_text += f"... +{len(active_vehicles) - 6} more vehicles\n"

        return info_text

    def create_probability_grid_colorbar(fig, ax, vehicle_colors: Dict,
                                         active_vehicles: List[str]):
        """
        为网格概率分布创建颜色条说明

        Args:
            fig: 图形对象
            ax: 轴对象
            vehicle_colors: 车辆颜色字典
            active_vehicles: 活跃车辆列表
        """
        # 创建颜色条说明
        from matplotlib.patches import Rectangle

        # 在图的右侧创建颜色说明
        colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        colorbar_ax.set_xlim(0, 1)
        colorbar_ax.set_ylim(0, len(active_vehicles))

        for i, vehicle_id in enumerate(active_vehicles):
            color = vehicle_colors.get(vehicle_id, '#808080')

            # 绘制不同透明度的示例
            for alpha_level in [0.2, 0.5, 0.8]:
                rect = Rectangle((alpha_level / 2, i), 0.3, 0.8,
                                 facecolor=color, alpha=alpha_level)
                colorbar_ax.add_patch(rect)

            colorbar_ax.text(1.1, i + 0.4, vehicle_id,
                             verticalalignment='center', fontsize=8)

        colorbar_ax.set_xticks([0.1, 0.25, 0.4])
        colorbar_ax.set_xticklabels(['Low', 'Med', 'High'], fontsize=8)
        colorbar_ax.set_yticks([])
        colorbar_ax.set_title('Probability\nLevel', fontsize=9, fontweight='bold')

        return colorbar_ax

    def evaluate_cooperation_capabilities(self,
                                          probability_distributions: Dict) -> Dict:
        """
        评估协同能力

        Args:
            probability_distributions: 概率分布字典

        Returns:
            协同能力评估结果
        """
        cooperation_results = self.cooperation_evaluator.evaluate_multi_vehicle_cooperation(
            probability_distributions
        )

        return cooperation_results

    def compute_collision_probabilities(self,
                                        probability_distributions: Dict,
                                        cooperation_results: Dict) -> Dict:
        """
        计算碰撞概率

        Args:
            probability_distributions: 概率分布字典
            cooperation_results: 协同能力评估结果

        Returns:
            碰撞概率字典
        """
        collision_probabilities = {}

        ego_data = probability_distributions.get('ego', {})
        ego_prob_dist = ego_data.get('probability_distribution', {}).get('total_probability')

        if ego_prob_dist is None:
            return collision_probabilities

        time_grid = ego_data['probability_distribution']['time_grid']
        space_grid = ego_data['probability_distribution']['space_grid']

        # 计算各方向碰撞概率
        for direction, cooperation_result in cooperation_results.items():
            if 'cooperation_evaluation' not in cooperation_result:
                continue

            primary_vehicle_id = cooperation_result['primary_vehicle']
            if primary_vehicle_id not in probability_distributions:
                continue

            primary_data = probability_distributions[primary_vehicle_id]
            primary_prob_dist = primary_data['probability_distribution']['total_probability']

            # 计算位置重叠概率
            overlap_probs = {}
            for t_idx, t in enumerate(time_grid):
                overlap_prob = 0.0

                # 横向位置影响系数（暂设为1.0）
                lateral_coeff = 1.0

                # 时间影响系数
                time_coeff = self.cooperation_evaluator.compute_temporal_influence_coefficient(
                    t, time_grid[-1]
                )

                # 计算重叠概率
                for x_idx in range(len(space_grid)):
                    ego_prob = ego_prob_dist[t_idx, x_idx]
                    neighbor_prob = primary_prob_dist[t_idx, x_idx]
                    local_overlap = lateral_coeff * ego_prob * neighbor_prob * time_coeff
                    overlap_prob += local_overlap

                overlap_probs[t_idx] = {
                    'time': t,
                    'overlap_probability': overlap_prob,
                    'lateral_coefficient': lateral_coeff,
                    'time_coefficient': time_coeff
                }

            # 单方向冲突概率（取最大值）
            max_overlap_prob = max([data['overlap_probability'] for data in overlap_probs.values()] or [0.0])

            collision_probabilities[direction] = {
                'direction_collision_probability': max_overlap_prob,
                'timeline_overlap_probabilities': overlap_probs,
                'primary_vehicle': primary_vehicle_id
            }

        return collision_probabilities

    def compute_lane_change_feasibility(self,
                                        collision_probabilities: Dict,
                                        lateral_decision: str) -> Dict:
        """
        计算变道可行性

        Args:
            collision_probabilities: 碰撞概率字典
            lateral_decision: 横向决策

        Returns:
            变道可行性字典
        """
        if lateral_decision == 'keep_lane':
            return {'feasibility_probability': 1.0, 'collision_risk': 0.0}

        # 确定相关车道的碰撞概率
        ego_lane_collision = 0.0
        target_lane_collision = 0.0

        # ego车道（当前车道）碰撞风险
        ego_front_prob = collision_probabilities.get(Direction.EF, {}).get('direction_collision_probability', 0.0)
        ego_rear_prob = collision_probabilities.get(Direction.ER, {}).get('direction_collision_probability', 0.0)
        ego_lane_collision = ego_front_prob + ego_rear_prob * (1 - ego_front_prob)

        # 目标车道碰撞风险
        if lateral_decision == 'pre_change_left':
            target_front_prob = collision_probabilities.get(Direction.LF, {}).get('direction_collision_probability',
                                                                                  0.0)
            target_rear_prob = collision_probabilities.get(Direction.LR, {}).get('direction_collision_probability', 0.0)
        else:  # RIGHT
            target_front_prob = collision_probabilities.get(Direction.RF, {}).get('direction_collision_probability',
                                                                                  0.0)
            target_rear_prob = collision_probabilities.get(Direction.RR, {}).get('direction_collision_probability', 0.0)

        target_lane_collision = target_front_prob + target_rear_prob * (1 - target_front_prob)

        # 变道过程总碰撞风险
        total_collision_risk = ego_lane_collision + target_lane_collision * (1 - ego_lane_collision)

        # 变道可行性概率
        feasibility_probability = 1.0 - total_collision_risk

        return {
            'feasibility_probability': max(0.0, feasibility_probability),
            'total_collision_risk': total_collision_risk,
            'ego_lane_collision': ego_lane_collision,
            'target_lane_collision': target_lane_collision,
            'lateral_decision': lateral_decision
        }

    def assess_predictive_safety(self,
                                 ego_trajectory_planned: np.ndarray,
                                 probability_distributions: Dict,
                                 prediction_steps: int = 10) -> Dict:
        """
        预测性安全评估

        Args:
            ego_trajectory_planned: ego车辆计划轨迹 [N x 1] 位置数组
            probability_distributions: 概率分布字典
            prediction_steps: 预测步数

        Returns:
            安全风险评估结果
        """
        safety_risks = {}

        # 确保ego轨迹长度
        trajectory_length = min(len(ego_trajectory_planned), prediction_steps)

        # 车辆占用区域参数
        vehicle_length = self.params.vehicle_length
        safe_distance = self.params.safe_distance

        for vehicle_id, vehicle_data in probability_distributions.items():
            if vehicle_id == 'ego':
                continue

            prob_dist = vehicle_data['probability_distribution']['total_probability']
            time_grid = vehicle_data['probability_distribution']['time_grid']
            space_grid = vehicle_data['probability_distribution']['space_grid']

            step_risks = {}

            for step in range(trajectory_length):
                if step >= len(time_grid):
                    break

                # ego车辆在该步的占用区域
                ego_position = ego_trajectory_planned[step]
                ego_occupied_min = ego_position - vehicle_length - safe_distance
                ego_occupied_max = ego_position + vehicle_length + safe_distance

                # 计算冲突概率
                conflict_probability = 0.0

                for x_idx, x in enumerate(space_grid):
                    if ego_occupied_min <= x <= ego_occupied_max:
                        conflict_probability += prob_dist[step, x_idx]

                step_risks[step] = {
                    'time': time_grid[step] if step < len(time_grid) else step * self.params.dt,
                    'ego_position': ego_position,
                    'ego_occupied_region': (ego_occupied_min, ego_occupied_max),
                    'conflict_probability': conflict_probability
                }

            safety_risks[vehicle_id] = {
                'step_risks': step_risks,
                'max_risk': max([data['conflict_probability'] for data in step_risks.values()] or [0.0]),
                'vehicle_type': vehicle_data['vehicle_type'],
                'direction': vehicle_data['direction']
            }

        return safety_risks