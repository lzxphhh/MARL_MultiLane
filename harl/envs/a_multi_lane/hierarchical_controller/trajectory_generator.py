# trajectory_generator.py
"""
轨迹生成器模块
Trajectory Generator Module
负责生成车辆的参考轨迹和轨迹包络线
"""

import numpy as np
from typing import Tuple, Dict
from scipy.special import comb
from harl.envs.a_multi_lane.project_structure import SystemParameters, VehicleState, VehicleType
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class TrajectoryGenerator:
    """轨迹生成器"""

    def __init__(self, system_params: SystemParameters):
        """
        初始化轨迹生成器

        Args:
            system_params: 系统参数配置
        """
        self.params = system_params

    def generate_lane_change_longitudinal_trajectory(self,
                                                     initial_state: VehicleState,
                                                     target_velocity: float,
                                                     lane_change_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成变道纵向轨迹（三次多项式）

        Args:
            initial_state: 初始车辆状态
            target_velocity: 目标速度 [m/s]
            lane_change_time: 变道时间 [s]

        Returns:
            (时间数组, 轨迹系数[a0, a1, a2, a3])
        """
        x0 = initial_state.x
        v0 = initial_state.v
        a0 = initial_state.a

        # 构建约束方程组
        # x(0) = x0, dx/dt(0) = v0, d²x/dt²(0) = a0, dx/dt(T) = v_target
        A = np.array([
            [1, 0, 0, 0],  # x(0) = x0
            [0, 1, 0, 0],  # dx/dt(0) = v0
            [0, 0, 2, 0],  # d²x/dt²(0) = a0
            [0, 1, 2 * lane_change_time, 3 * lane_change_time ** 2]  # dx/dt(T) = v_target
        ])

        b = np.array([x0, v0, a0, target_velocity])

        # 解方程得到系数
        coeffs = np.linalg.solve(A, b)

        # 生成时间数组
        time_steps = int(lane_change_time / self.params.dt) + 1
        time_array = np.linspace(0, lane_change_time, time_steps)

        return time_array, coeffs

    def generate_lane_change_lateral_trajectory(self,
                                                initial_y: float,
                                                target_y: float,
                                                lane_change_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成变道横向轨迹（贝塞尔曲线）

        Args:
            initial_y: 初始横向位置 [m]
            target_y: 目标横向位置 [m]
            lane_change_time: 变道时间 [s]

        Returns:
            (时间数组, 控制点数组)
        """
        delta_y = target_y - initial_y

        # 定义贝塞尔曲线控制点
        control_points = np.array([
            initial_y,  # P0
            initial_y, # + delta_y / 4,  # P1
            (initial_y + target_y) / 2,  # P2
            target_y, # - delta_y / 4,  # P3
            target_y  # P4
        ])

        # 生成时间数组
        time_steps = int(lane_change_time / self.params.dt) + 1
        time_array = np.linspace(0, lane_change_time, time_steps)

        return time_array, control_points

    def compute_bezier_curve(self, control_points: np.ndarray, t_normalized: float) -> float:
        """
        计算贝塞尔曲线上的点

        Args:
            control_points: 控制点数组
            t_normalized: 归一化时间 [0, 1]

        Returns:
            贝塞尔曲线上对应点的值
        """
        n = len(control_points) - 1
        result = 0.0

        for i in range(n + 1):
            # 计算伯恩斯坦基函数
            bernstein = comb(n, i) * ((1 - t_normalized) ** (n - i)) * (t_normalized ** i)
            result += control_points[i] * bernstein

        return result

    def compute_bezier_derivative(self, control_points: np.ndarray, t_normalized: float, order: int = 1) -> float:
        """
        计算贝塞尔曲线的导数

        Args:
            control_points: 控制点数组
            t_normalized: 归一化时间 [0, 1]
            order: 导数阶数

        Returns:
            导数值
        """
        if order == 1:
            # 一阶导数
            n = len(control_points) - 1
            if n == 0:
                return 0.0

            derivative_points = n * np.diff(control_points)
            result = 0.0

            for i in range(n):
                bernstein = comb(n - 1, i) * ((1 - t_normalized) ** (n - 1 - i)) * (t_normalized ** i)
                result += derivative_points[i] * bernstein

            return result

        elif order == 2:
            # 二阶导数
            n = len(control_points) - 1
            if n <= 1:
                return 0.0

            first_diff = n * np.diff(control_points)
            second_diff = (n - 1) * np.diff(first_diff)
            result = 0.0

            for i in range(n - 1):
                bernstein = comb(n - 2, i) * ((1 - t_normalized) ** (n - 2 - i)) * (t_normalized ** i)
                result += second_diff[i] * bernstein

            return result

        else:
            raise ValueError("Only first and second order derivatives are supported")

    def generate_lane_keeping_trajectory(self,
                                         initial_state: VehicleState,
                                         target_velocity: float,
                                         prediction_horizon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成车道保持轨迹

        Args:
            initial_state: 初始车辆状态
            target_velocity: 目标速度 [m/s]
            prediction_horizon: 预测周期 [s]

        Returns:
            (时间数组, 轨迹系数[a0, a1, a2, a3])
        """
        # 纵向轨迹使用三次多项式
        time_array, coeffs = self.generate_lane_change_longitudinal_trajectory(
            initial_state, target_velocity, prediction_horizon
        )

        return time_array, coeffs

    def evaluate_polynomial_trajectory(self,
                                       coeffs: np.ndarray,
                                       time_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算多项式轨迹的位置、速度和加速度

        Args:
            coeffs: 多项式系数 [a0, a1, a2, a3]
            time_array: 时间数组

        Returns:
            (位置数组, 速度数组, 加速度数组)
        """
        a0, a1, a2, a3 = coeffs

        # 位置: x(t) = a0 + a1*t + a2*t^2 + a3*t^3
        position = a0 + a1 * time_array + a2 * time_array ** 2 + a3 * time_array ** 3

        # 速度: v(t) = a1 + 2*a2*t + 3*a3*t^2
        velocity = a1 + 2 * a2 * time_array + 3 * a3 * time_array ** 2

        # 加速度: a(t) = 2*a2 + 6*a3*t
        acceleration = 2 * a2 + 6 * a3 * time_array

        return position, velocity, acceleration

    def evaluate_bezier_trajectory(self,
                                   control_points: np.ndarray,
                                   time_array: np.ndarray,
                                   lane_change_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算贝塞尔轨迹的位置、速度和加速度

        Args:
            control_points: 控制点数组
            time_array: 时间数组
            lane_change_time: 变道时间

        Returns:
            (位置数组, 速度数组, 加速度数组)
        """
        position = np.zeros_like(time_array)
        velocity = np.zeros_like(time_array)
        acceleration = np.zeros_like(time_array)

        for i, t in enumerate(time_array):
            t_normalized = t / lane_change_time
            # t_normalized = np.clip(t_normalized, 0.0, 1.0)

            # 位置
            position[i] = self.compute_bezier_curve(control_points, t_normalized)

            # 速度 (需要考虑时间尺度变换)
            velocity[i] = self.compute_bezier_derivative(control_points, t_normalized, 1) / lane_change_time

            # 加速度
            acceleration[i] = self.compute_bezier_derivative(control_points, t_normalized, 2) / (lane_change_time ** 2)

        return position, velocity, acceleration

    def generate_complete_lane_change_trajectory(self,
                                                 initial_state: VehicleState,
                                                 target_y: float,
                                                 target_velocity: float,
                                                 lane_change_time: float) -> Dict:
        """
        生成完整的变道轨迹（纵向+横向）

        Args:
            initial_state: 初始车辆状态
            target_y: 目标横向位置 [m]
            target_velocity: 目标速度 [m/s]
            lane_change_time: 变道时间 [s]

        Returns:
            轨迹字典，包含位置、速度、加速度时间序列
        """
        # 生成纵向轨迹
        time_array, longitudinal_coeffs = self.generate_lane_change_longitudinal_trajectory(
            initial_state, target_velocity, lane_change_time
        )

        x_pos, x_vel, x_acc = self.evaluate_polynomial_trajectory(longitudinal_coeffs, time_array)

        # 生成横向轨迹
        _, lateral_control_points = self.generate_lane_change_lateral_trajectory(
            initial_state.y, target_y, lane_change_time
        )

        y_pos, y_vel, y_acc = self.evaluate_bezier_trajectory(
            lateral_control_points, time_array, lane_change_time
        )

        trajectory_result = {
            'time': time_array,
            'longitudinal': {
                'position': x_pos,
                'velocity': x_vel,
                'acceleration': x_acc,
                'coefficients': longitudinal_coeffs
            },
            'lateral': {
                'position': y_pos,
                'velocity': y_vel,
                'acceleration': y_acc,
                'control_points': lateral_control_points
            }
        }
        save_path = os.path.join('/home/lzx/00_Projects/00_platoon_MARL/HARL/harl/envs/a_multi_lane/motion/2025-09-16-21-52-27', 'lane-change-trajectory.png')
        self.visualize_trajectory_2d(trajectory_result, save_path)

        return trajectory_result

    def visualize_trajectory_2d(self,
                                trajectory_data: Dict,
                                save_path: str = None):
        """
        Independent 2D trajectory visualization function

        Args:
            trajectory_data: Trajectory data dictionary
            initial_state: Initial vehicle state
            target_y: Target lateral position
            lane_change_time: Lane change duration
            save_path: Save path (optional)
        """
        x_pos = trajectory_data['longitudinal']['position']
        y_pos = trajectory_data['lateral']['position']
        x_vel = trajectory_data['longitudinal']['velocity']
        time_array = trajectory_data['time']

        # 创建更加专业的可视化效果
        fig, ax = plt.subplots(figsize=(14, 4.5))

        # 绘制主轨迹
        line = ax.plot(x_pos, y_pos, 'b-', linewidth=4, label='Lane Change Trajectory', alpha=0.8)[0]

        # 添加方向箭头
        arrow_indices = np.linspace(10, len(x_pos) - 10, 2, dtype=int)
        for i in arrow_indices:
            dx = x_pos[i + 5] - x_pos[i - 5]
            dy = y_pos[i + 5] - y_pos[i - 5]
            ax.arrow(x_pos[i], y_pos[i], dx * 0.3, dy * 0.3,
                     head_width=0.3, head_length=0.5, fc='blue', ec='blue', alpha=0.7)

        # 车辆轮廓可视化（起始和结束位置）
        vehicle_length, vehicle_width = 4.5, 1.8

        # 起始位置车辆轮廓
        start_vehicle = patches.Rectangle((x_pos[0] - vehicle_length / 2, y_pos[0] - vehicle_width / 2),
                                          vehicle_length, vehicle_width,
                                          angle=0, facecolor='blue', alpha=0.6,
                                          edgecolor='darkblue', linewidth=2)
        ax.add_patch(start_vehicle)

        # 结束位置车辆轮廓
        end_vehicle = patches.Rectangle((x_pos[-1] - vehicle_length / 2, y_pos[-1] - vehicle_width / 2),
                                        vehicle_length, vehicle_width,
                                        angle=0, facecolor='skyblue', alpha=0.6,
                                        edgecolor='blue', linewidth=2)
        ax.add_patch(end_vehicle)

        # 设置专业的图形样式
        ax.set_xlabel('Longitudinal Position [m]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Lateral Position [m]', fontsize=14, fontweight='bold')
        ax.set_title('Connected Autonomous Vehicle Lane Change Trajectory Analysis', fontsize=16, fontweight='bold',
                     pad=20)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Trajectory plot saved to: {save_path}")

        plt.show()



# 使用示例
def example_usage():
    """轨迹生成器使用示例"""

    # 初始化系统参数
    system_params = SystemParameters()

    # 创建轨迹生成器
    traj_gen = TrajectoryGenerator(system_params)

    # 初始车辆状态
    initial_state = VehicleState(
        x=0.0, y=0.0, v=15.0, theta=0.0, a=0.0, omega=0.0,
        lane_id=1, vehicle_type=VehicleType.CAV
    )

    # 生成变道轨迹
    trajectory = traj_gen.generate_complete_lane_change_trajectory(
        initial_state=initial_state,
        target_y=3.7,  # 向左变道一个车道宽度
        target_velocity=16.0,
        lane_change_time=3.0
    )

    print("变道轨迹生成完成:")
    print(f"时间点数量: {len(trajectory['time'])}")
    print(f"最终纵向位置: {trajectory['longitudinal']['position'][-1]:.2f} m")
    print(f"最终横向位置: {trajectory['lateral']['position'][-1]:.2f} m")
    print(f"最终速度: {trajectory['longitudinal']['velocity'][-1]:.2f} m/s")


if __name__ == "__main__":
    example_usage()