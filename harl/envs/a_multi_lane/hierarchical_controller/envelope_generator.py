# envelope_generator.py
"""
包络线生成器模块
Envelope Generator Module
负责生成车辆轨迹的可达性包络线
"""

import numpy as np
from typing import Tuple, Dict, Optional
from harl.envs.a_multi_lane.project_structure import SystemParameters, VehicleType
from harl.envs.a_multi_lane.hierarchical_controller.trajectory_generator import TrajectoryGenerator


import matplotlib.pyplot as plt
from enum import Enum
import matplotlib.patches as patches


class EnvelopeGenerator:
    """包络线生成器"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化包络线生成器

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params
        self.traj_gen = TrajectoryGenerator(system_params)

    def compute_physical_envelope(self,
                                  initial_x: float,
                                  initial_v: float,
                                  initial_a: float,
                                  prediction_horizon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算绝对物理包络线

        Args:
            initial_x: 初始位置 [m]
            initial_v: 初始速度 [m/s]
            initial_a: 初始加速度 [m/s²]
            prediction_horizon: 预测周期 [s]

        Returns:
            (时间数组, 最大位置数组, 最小位置数组)
        """
        time_steps = int(prediction_horizon / self.params.dt) + 1
        time_array = np.linspace(0, prediction_horizon, time_steps)

        x_max = np.zeros_like(time_array)
        x_min = np.zeros_like(time_array)

        for i, t in enumerate(time_array):
            x_max[i] = self._compute_max_reachable_position(initial_x, initial_v, initial_a, t)
            x_min[i] = self._compute_min_reachable_position(initial_x, initial_v, initial_a, t)

        return time_array, x_max, x_min

    def _compute_max_reachable_position(self,
                                        x0: float,
                                        v0: float,
                                        a0: float,
                                        t: float) -> float:
        """
        计算最大可达位置（最大加速情况）

        Args:
            x0: 初始位置
            v0: 初始速度
            a0: 初始加速度
            t: 时间

        Returns:
            最大可达位置
        """
        a_max = self.params.a_max
        a_dot_max = self.params.a_dot_max
        v_max = self.params.v_max

        # 计算达到最大加速度的时间
        t1 = (a_max - a0) / a_dot_max if a0 < a_max else 0.0
        t1 = max(0.0, t1)

        if t <= t1:
            # 阶段1：加速度变化率限制
            x = x0 + v0 * t + 0.5 * a0 * t ** 2 + (1 / 6) * a_dot_max * t ** 3
            v = v0 + a0 * t + 0.5 * a_dot_max * t ** 2
        else:
            # 阶段1结束时的状态
            x1 = x0 + v0 * t1 + 0.5 * a0 * t1 ** 2 + (1 / 6) * a_dot_max * t1 ** 3
            v1 = v0 + a0 * t1 + 0.5 * a_dot_max * t1 ** 2

            # 计算达到限速的时间
            t2 = t1 + (v_max - v1) / a_max if v1 < v_max and a_max > 0 else t1
            t2 = max(t1, t2)

            if t <= t2:
                # 阶段2：最大加速度
                dt = t - t1
                x = x1 + v1 * dt + 0.5 * a_max * dt ** 2
                v = v1 + a_max * dt
            else:
                # 阶段2结束时的状态
                dt2 = t2 - t1
                x2 = x1 + v1 * dt2 + 0.5 * a_max * dt2 ** 2
                v2 = min(v1 + a_max * dt2, v_max)

                # 阶段3：恒速行驶
                dt3 = t - t2
                x = x2 + v2 * dt3
                v = v2

        return x

    def _compute_min_reachable_position(self,
                                        x0: float,
                                        v0: float,
                                        a0: float,
                                        t: float) -> float:
        """
        计算最小可达位置（最大制动情况）

        Args:
            x0: 初始位置
            v0: 初始速度
            a0: 初始加速度
            t: 时间

        Returns:
            最小可达位置
        """
        a_min = self.params.a_min
        a_dot_max = self.params.a_dot_max

        # 计算达到最大制动加速度的时间
        t3 = (a0 - a_min) / a_dot_max if a0 > a_min else 0.0
        t3 = max(0.0, t3)

        if t <= t3:
            # 阶段1：加速度变化率限制
            x = x0 + v0 * t + 0.5 * a0 * t ** 2 - (1 / 6) * a_dot_max * t ** 3
            v = v0 + a0 * t - 0.5 * a_dot_max * t ** 2
        else:
            # 阶段1结束时的状态
            x3 = x0 + v0 * t3 + 0.5 * a0 * t3 ** 2 - (1 / 6) * a_dot_max * t3 ** 3
            v3 = v0 + a0 * t3 - 0.5 * a_dot_max * t3 ** 2

            # 计算完全停止的时间
            t4 = t3 + v3 / abs(a_min) if v3 > 0 and a_min < 0 else t3
            t4 = max(t3, t4)

            if t <= t4:
                # 阶段2：最大制动
                dt = t - t3
                x = x3 + v3 * dt + 0.5 * a_min * dt ** 2
                v = max(v3 + a_min * dt, 0.0)
            else:
                # 阶段2结束时的状态（完全停止）
                dt4 = t4 - t3
                x4 = x3 + v3 * dt4 + 0.5 * a_min * dt4 ** 2

                # 阶段3：保持停止
                x = x4
                v = 0.0

        return x

    def compute_comfort_envelope(self,
                                 initial_x: float,
                                 initial_v: float,
                                 initial_a: float,
                                 prediction_horizon: float,
                                 alpha_c: Optional[float] = None,
                                 beta_c: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算舒适性包络线

        Args:
            initial_x: 初始位置 [m]
            initial_v: 初始速度 [m/s]
            initial_a: 初始加速度 [m/s²]
            prediction_horizon: 预测周期 [s]
            alpha_c: 加速度舒适性系数
            beta_c: 加速度变化率舒适性系数

        Returns:
            (时间数组, 最大位置数组, 最小位置数组)
        """
        # 使用默认舒适性参数
        if alpha_c is None:
            alpha_c = self.params.alpha_c
        if beta_c is None:
            beta_c = self.params.beta_c

        # 舒适性约束
        a_c_max = alpha_c * self.params.a_max
        a_c_min = alpha_c * self.params.a_min
        a_dot_c_max = beta_c * self.params.a_dot_max

        time_steps = int(prediction_horizon / self.params.dt) + 1
        time_array = np.linspace(0, prediction_horizon, time_steps)

        x_c_max = np.zeros_like(time_array)
        x_c_min = np.zeros_like(time_array)

        for i, t in enumerate(time_array):
            x_c_max[i] = self._compute_comfort_max_position(
                initial_x, initial_v, initial_a, t, a_c_max, a_dot_c_max
            )
            x_c_min[i] = self._compute_comfort_min_position(
                initial_x, initial_v, initial_a, t, a_c_min, a_dot_c_max
            )

        return time_array, x_c_max, x_c_min

    def _compute_comfort_max_position(self,
                                      x0: float,
                                      v0: float,
                                      a0: float,
                                      t: float,
                                      a_c_max: float,
                                      a_dot_c_max: float) -> float:
        """计算舒适性最大位置"""
        v_max = self.params.v_max

        t1 = (a_c_max - a0) / a_dot_c_max if a0 < a_c_max else 0.0
        t1 = max(0.0, t1)

        if t <= t1:
            x = x0 + v0 * t + 0.5 * a0 * t ** 2 + (1 / 6) * a_dot_c_max * t ** 3
        else:
            x1 = x0 + v0 * t1 + 0.5 * a0 * t1 ** 2 + (1 / 6) * a_dot_c_max * t1 ** 3
            v1 = v0 + a0 * t1 + 0.5 * a_dot_c_max * t1 ** 2

            t2 = t1 + (v_max - v1) / a_c_max if v1 < v_max and a_c_max > 0 else t1
            t2 = max(t1, t2)

            if t <= t2:
                dt = t - t1
                x = x1 + v1 * dt + 0.5 * a_c_max * dt ** 2
            else:
                dt2 = t2 - t1
                x2 = x1 + v1 * dt2 + 0.5 * a_c_max * dt2 ** 2
                v2 = min(v1 + a_c_max * dt2, v_max)

                dt3 = t - t2
                x = x2 + v2 * dt3

        return x

    def _compute_comfort_min_position(self,
                                      x0: float,
                                      v0: float,
                                      a0: float,
                                      t: float,
                                      a_c_min: float,
                                      a_dot_c_max: float) -> float:
        """计算舒适性最小位置"""
        t3 = (a0 - a_c_min) / a_dot_c_max if a0 > a_c_min else 0.0
        t3 = max(0.0, t3)

        if t <= t3:
            x = x0 + v0 * t + 0.5 * a0 * t ** 2 - (1 / 6) * a_dot_c_max * t ** 3
        else:
            x3 = x0 + v0 * t3 + 0.5 * a0 * t3 ** 2 - (1 / 6) * a_dot_c_max * t3 ** 3
            v3 = v0 + a0 * t3 - 0.5 * a_dot_c_max * t3 ** 2

            t4 = t3 + v3 / abs(a_c_min) if v3 > 0 and a_c_min < 0 else t3
            t4 = max(t3, t4)

            if t <= t4:
                dt = t - t3
                x = x3 + v3 * dt + 0.5 * a_c_min * dt ** 2
            else:
                dt4 = t4 - t3
                x4 = x3 + v3 * dt4 + 0.5 * a_c_min * dt4 ** 2
                x = x4

        return x

    def generate_monte_carlo_envelope(self,
                                      x_ref: np.ndarray,
                                      v_ref: np.ndarray,
                                      a_ref: np.ndarray,
                                      time_array: np.ndarray,
                                      sigma_a: float,
                                      vehicle_type: VehicleType = VehicleType.HDV,
                                      num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于蒙特卡洛方法生成位置包络

        Args:
            x_ref: 参考位置轨迹
            v_ref: 参考速度轨迹
            a_ref: 参考加速度轨迹
            time_array: 时间数组
            sigma_a: 加速度标准偏差
            vehicle_type: 车辆类型
            num_samples: 采样数量

        Returns:
            (最小位置包络, 最大位置包络)
        """
        dt = self.params.dt
        a_min = self.params.a_min
        a_max = self.params.a_max
        a_dot_max = self.params.a_dot_max

        # 存储所有采样轨迹
        samples = {}
        results = np.zeros((num_samples, len(time_array)))

        for m in range(num_samples):
            samples[m] = np.zeros((3, len(time_array)))
            # 初始化状态
            x_sample = x_ref[0]
            v_sample = v_ref[0]
            a_sample = a_ref[0]
            samples[m][0, 0] = x_sample
            samples[m][1, 0] = v_sample
            samples[m][2, 0] = a_sample
            results[m, 0] = x_sample

            for k in range(1, len(time_array)):
                # 采样因果扰动
                epsilon_a = np.random.normal(0, sigma_a)

                # 计算实际加速度
                a_actual = a_ref[k - 1] + epsilon_a

                # 应用物理约束
                a_actual = np.clip(a_actual, a_min, a_max)

                # 应用加速度变化率约束
                if k > 1:
                    a_prev = samples[m][2, k - 1]
                    a_min_dot = a_prev - a_dot_max * dt
                    a_max_dot = a_prev + a_dot_max * dt
                    a_actual = np.clip(a_actual, a_min_dot, a_max_dot)

                # 数值积分更新状态
                v_sample = v_sample + a_actual * dt
                v_sample = max(0, min(v_sample, self.params.v_max))

                x_sample = x_sample + v_sample * dt + 0.5 * a_actual * dt**2

                samples[m][0, k] = x_sample
                samples[m][1, k] = v_sample
                samples[m][2, k] = a_actual

                results[m, k] = x_sample

        # 统计包络边界
        x_min_envelope = np.percentile(results, 5, axis=0)  # 5%分位数
        x_max_envelope = np.percentile(results, 95, axis=0)  # 95%分位数

        return x_min_envelope, x_max_envelope

    def generate_comprehensive_envelope(self,
                                        initial_x: float,
                                        initial_v: float,
                                        initial_a: float,
                                        x_ref: np.ndarray,
                                        v_ref: np.ndarray,
                                        a_ref: np.ndarray,
                                        time_array: np.ndarray,
                                        vehicle_type: VehicleType = VehicleType.HDV) -> Dict:
        """
        生成综合包络线（物理包络 + 舒适性包络 + 蒙特卡洛包络）

        Args:
            initial_x: 初始位置
            initial_v: 初始速度
            initial_a: 初始加速度
            x_ref: 参考位置轨迹
            v_ref: 参考速度轨迹
            a_ref: 参考加速度轨迹
            time_array: 时间数组
            vehicle_type: 车辆类型

        Returns:
            包络线字典
        """
        prediction_horizon = time_array[-1]

        # 1. 物理包络
        _, x_phys_max, x_phys_min = self.compute_physical_envelope(
            initial_x, initial_v, initial_a, prediction_horizon
        )

        # 2. 舒适性包络
        _, x_comfort_max, x_comfort_min = self.compute_comfort_envelope(
            initial_x, initial_v, initial_a, prediction_horizon
        )

        # 3. 蒙特卡洛包络
        # sigma_a = self.params.sigma_a_cav if vehicle_type == VehicleType.CAV else self.params.sigma_a_hdv
        # x_mc_min, x_mc_max = self.generate_monte_carlo_envelope(
        #     x_ref, v_ref, a_ref, time_array, sigma_a, vehicle_type
        # )

        # 构建结果字典
        envelope_result = {
            'time': time_array,
            'physical': {
                'min': x_phys_min,
                'max': x_phys_max
            },
            'comfort': {
                'min': x_comfort_min,
                'max': x_comfort_max
            },
            # 'monte_carlo': {
            #     'min': x_mc_min,
            #     'max': x_mc_max
            # },
            'reference': {
                'position': x_ref,
                'velocity': v_ref,
                'acceleration': a_ref
            }
        }

        # # 调用可视化函数
        # print(f"Generating comprehensive envelope visualization for {vehicle_type.name}...")
        # self.visualize_comprehensive_envelope(envelope_result, vehicle_type, initial_x, initial_v, initial_a)

        return envelope_result

    def visualize_comprehensive_envelope(self,
                                         envelope_data: Dict,
                                         vehicle_type: VehicleType,
                                         initial_x: float,
                                         initial_v: float,
                                         initial_a: float,
                                         save_path: str = None):
        """
        可视化综合包络线

        Args:
            envelope_data: 包络线数据字典
            vehicle_type: 车辆类型
            initial_x: 初始位置
            initial_v: 初始速度
            initial_a: 初始加速度
            save_path: 保存路径（可选）
        """
        # 设置包络线颜色方案
        envelope_colors = {
            'physical': '#ff9999',  # 浅红色 - 物理包络
            'comfort': '#99ccff',  # 浅蓝色 - 舒适性包络
            'monte_carlo': '#99ff99'  # 浅绿色 - 蒙特卡洛包络
        }

        vehicle_name = "CAV" if vehicle_type == VehicleType.CAV else "HDV"

        # 创建单个图形
        plt.figure(figsize=(14, 8))

        time_array = envelope_data['time']
        x_ref = envelope_data['reference']['position']

        # 绘制物理包络（最外层）
        plt.fill_between(time_array,
                         envelope_data['physical']['min'],
                         envelope_data['physical']['max'],
                         alpha=0.3, color=envelope_colors['physical'],
                         label='Physical Envelope',
                         edgecolor='red', linewidth=1.5)

        # 绘制舒适性包络（中间层）
        plt.fill_between(time_array,
                         envelope_data['comfort']['min'],
                         envelope_data['comfort']['max'],
                         alpha=0.4, color=envelope_colors['comfort'],
                         label='Comfort Envelope',
                         edgecolor='blue', linewidth=1.5)

        # 绘制蒙特卡洛包络（最内层）
        plt.fill_between(time_array,
                         envelope_data['monte_carlo']['min'],
                         envelope_data['monte_carlo']['max'],
                         alpha=0.5, color=envelope_colors['monte_carlo'],
                         label='Monte Carlo Envelope',
                         edgecolor='green', linewidth=1.5)

        # 绘制参考轨迹
        plt.plot(time_array, x_ref, '--', color='black', linewidth=3,
                 label='Reference Trajectory', alpha=0.8)

        # 标记初始点和终点
        plt.plot(time_array[0], x_ref[0], 'o', color='darkgreen',
                 markersize=12, markeredgecolor='black', markeredgewidth=2,
                 label='Initial Point')
        plt.plot(time_array[-1], x_ref[-1], 's', color='darkred',
                 markersize=12, markeredgecolor='black', markeredgewidth=2,
                 label='Final Point')

        # 设置图形属性
        plt.xlabel('Time [s]', fontsize=14, fontweight='bold')
        plt.ylabel('Longitudinal Position [m]', fontsize=14, fontweight='bold')
        plt.title(f'{vehicle_name} Comprehensive Trajectory Envelopes\n'
                  f'Initial State: x={initial_x:.1f}m, v={initial_v:.1f}m/s, a={initial_a:.1f}m/s²',
                  fontsize=16, fontweight='bold')

        plt.grid(True, alpha=0.3, linestyle=':')
        plt.legend(loc='upper left', fontsize=14, framealpha=0.9)

        # 计算并显示统计信息
        physical_width = np.mean(envelope_data['physical']['max'] - envelope_data['physical']['min'])
        comfort_width = np.mean(envelope_data['comfort']['max'] - envelope_data['comfort']['min'])
        mc_width = np.mean(envelope_data['monte_carlo']['max'] - envelope_data['monte_carlo']['min'])

        # 添加统计信息文本框
        stats_text = (f'Envelope Statistics:\n'
                      f'• Physical Width: {physical_width:.2f} m\n'
                      f'• Comfort Width: {comfort_width:.2f} m\n'
                      f'• Monte Carlo Width: {mc_width:.2f} m'
                      # f'• Prediction Horizon: {time_array[-1]:.1f} s\n'
                      # f'• Trajectory Distance: {x_ref[-1] - x_ref[0]:.1f} m'
                      )

        plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comprehensive envelope visualization saved to: {save_path}")

        plt.show()

    def visualize_envelope_comparison(self,
                                      envelope_data_list: list,
                                      vehicle_types: list,
                                      labels: list = None,
                                      save_path: str = None):
        """
        比较多个包络线

        Args:
            envelope_data_list: 包络线数据列表
            vehicle_types: 车辆类型列表
            labels: 自定义标签列表
            save_path: 保存路径
        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for i, (envelope_data, vehicle_type) in enumerate(zip(envelope_data_list, vehicle_types)):
            color = colors[i % len(colors)]
            vehicle_name = "CAV" if vehicle_type == VehicleType.CAV else "HDV"
            label = labels[i] if labels else f"{vehicle_name}-{i + 1}"

            time_array = envelope_data['time']

            # 位置包络线对比
            ax1.fill_between(time_array,
                             envelope_data['monte_carlo']['min'],
                             envelope_data['monte_carlo']['max'],
                             alpha=0.3, color=color, label=f'{label} MC Envelope')

            ax1.plot(time_array, envelope_data['reference']['position'],
                     '--', color=color, linewidth=2, label=f'{label} Reference')

            # 不确定性宽度对比
            mc_width = (envelope_data['monte_carlo']['max'] -
                        envelope_data['monte_carlo']['min'])
            ax2.plot(time_array, mc_width, '-', color=color,
                     linewidth=2, label=f'{label} Uncertainty')

        ax1.set_xlabel('Time [s]', fontweight='bold')
        ax1.set_ylabel('Longitudinal Position [m]', fontweight='bold')
        ax1.set_title('Multi-Vehicle Envelope Comparison', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_xlabel('Time [s]', fontweight='bold')
        ax2.set_ylabel('Envelope Width [m]', fontweight='bold')
        ax2.set_title('Uncertainty Width Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Envelope comparison saved to: {save_path}")
