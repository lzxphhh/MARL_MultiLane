# mpc_cooperative_controller.py
"""
基于MPC的协同控制器模块
严格按照PDF文档实现，使用CasADi求解
"""

import casadi as ca
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, LateralDecision, WeightConfig
)


@dataclass
class MPCParameters:
    """MPC控制器参数配置"""
    # 预测参数（PDF 2.2节）
    prediction_horizon: int = 20  # Np
    control_horizon: int = 10  # Nc
    sampling_time: float = 0.1  # Δt

    # 车辆参数（PDF 1.1.2节）
    tau: float = 0.5  # 加速度一阶滞后时间常数 [s]

    # 约束参数（PDF 4节）
    v_min: float = 0.0  # 最小速度 [m/s]
    v_max: float = 20.0  # 最大速度 [m/s]
    a_min: float = -5.0  # 最小加速度 [m/s²]
    a_max: float = 3.0  # 最大加速度 [m/s²]
    omega_min: float = -0.5  # 最小横摆角速度 [rad/s]
    omega_max: float = 0.5  # 最大横摆角速度 [rad/s]
    y_road_min: float = -3.2  # 道路左边界 [m]
    y_road_max: float = 3.2  # 道路右边界 [m]
    a_dot_max: float = 3.0  # 加速度变化率 [m/s³]
    omega_dot_max: float = 1.0  # 角速度变化率 [rad/s²]

    # 舒适性权重（PDF公式22）
    beta1: float = 0.1  # 纵向加速度权重
    beta2: float = 0.1  # 角速度权重

    # 车辆安全参数
    veh_length: float = 5.0
    dist_min: float = 2.0
    safe_dist_min: float = 5.0

@dataclass
class MPCResult:
    """MPC求解结果"""
    optimal_control: Tuple[float, float]  # (a_u, omega)
    predicted_states: List[List[float]]  # 预测状态序列
    control_sequence: List[Tuple[float, float]]  # 控制序列
    optimal_cost: float
    solve_time: float
    feasible: bool
    status: str


class MPCCooperativeController:
    """基于MPC的协同控制器"""

    def __init__(self, mpc_params: MPCParameters, system_params: SystemParameters):
        """
        初始化MPC协同控制器

        Args:
            mpc_params: MPC参数配置
            system_params: 系统参数配置
        """
        self.mpc_params = mpc_params
        self.system_params = system_params
        self.surrounding_prediction_centers = None

    def _build_mpc_problem(self):
        """构建CasADi MPC优化问题 - 严格按照PDF实现"""
        Np = self.mpc_params.prediction_horizon
        Nc = self.mpc_params.control_horizon
        dt = self.mpc_params.sampling_time
        tau = self.mpc_params.tau

        # 创建优化器
        self.opti = ca.Opti()

        # 定义决策变量（PDF 2节）
        self.X = self.opti.variable(5, Np + 1)  # 状态序列 [x, y, v, ψ, a]
        self.U = self.opti.variable(2, Nc)  # 控制序列 [a_u, ω]

        # 定义参数（每次求解时设置）
        self.p_x0 = self.opti.parameter(5)  # 初始状态
        self.p_v_ref = self.opti.parameter(1)  # 参考速度
        self.p_y_target = self.opti.parameter(1)  # 目标横向位置
        self.p_we = self.opti.parameter(1)  # 效率权重
        self.p_wc = self.opti.parameter(1)  # 舒适权重
        self.p_wt = self.opti.parameter(1)  # 任务权重
        self.p_ws = self.opti.parameter(1)  # 安全权重

        # 1. 初始状态约束
        self.opti.subject_to(self.X[:, 0] == self.p_x0)

        # 2. 车辆运动学模型约束
        for k in range(Np):
            # 确定控制输入索引
            if k < Nc:
                u_k = self.U[:, k]
            else:
                u_k = self.U[:, Nc - 1]  # 控制时域外保持最后一个控制

            x_k = self.X[:, k]

            # 状态方程
            x_next = x_k[0] + x_k[2] * ca.cos(x_k[3]) * dt
            y_next = x_k[1] + x_k[2] * ca.sin(x_k[3]) * dt
            v_next = x_k[2] + x_k[4] * dt
            psi_next = x_k[3] + u_k[1] * dt
            a_next = x_k[4] + (-x_k[4] / tau + u_k[0] / tau) * dt

            self.opti.subject_to(self.X[:, k + 1] == ca.vertcat(x_next, y_next, v_next, psi_next, a_next))

        # 3. 状态约束
        for k in range(Np + 1):
            # 速度约束
            self.opti.subject_to(self.opti.bounded(self.mpc_params.v_min, self.X[2, k], self.mpc_params.v_max))
            # 道路边界约束
            self.opti.subject_to(
                self.opti.bounded(self.mpc_params.y_road_min, self.X[1, k], self.mpc_params.y_road_max))

        # 4. 控制输入约束
        for k in range(Nc):
            # 加速度约束
            self.opti.subject_to(self.opti.bounded(self.mpc_params.a_min, self.U[0, k], self.mpc_params.a_max))
            # 横摆角速度约束
            self.opti.subject_to(self.opti.bounded(self.mpc_params.omega_min, self.U[1, k], self.mpc_params.omega_max))

        # 5. 控制变化率约束
        for k in range(Nc - 1):
            # 加速度变化率约束
            delta_a = self.U[0, k + 1] - self.U[0, k]
            self.opti.subject_to(
                self.opti.bounded(-self.mpc_params.a_dot_max * dt, delta_a, self.mpc_params.a_dot_max * dt))
            # 角速度变化率约束
            delta_omega = self.U[1, k + 1] - self.U[1, k]
            self.opti.subject_to(
                self.opti.bounded(-self.mpc_params.omega_dot_max * dt, delta_omega, self.mpc_params.omega_dot_max * dt))

        # 6. 多目标代价函数
        cost = 0

        # 效率代价
        for j in range(1, Np + 1):
            v_pred = self.X[2, j]
            cost += self.p_we * ((self.p_v_ref - v_pred) / self.mpc_params.v_max) ** 2

        # 舒适性代价
        for j in range(Nc):
            a_u = self.U[0, j]
            omega = self.U[1, j]
            cost += self.p_wc * (self.mpc_params.beta1 * (a_u / self.mpc_params.a_max) ** 2 +
                                 self.mpc_params.beta2 * (omega / self.mpc_params.omega_max) ** 2)

        # 横向任务执行代价
        for j in range(1, Np + 1):
            y_pred = self.X[1, j]
            cost += self.p_wt * ((y_pred - self.p_y_target) / self.mpc_params.y_road_max) ** 2

        # 预测性安全代价（含协同性安全）
        for j in range(1, Np + 1):
            x_pred = self.X[0, j]
            if self.surrounding_prediction_centers != None:
                safety_cost = 0
                for sur_id, sur_envelope in self.surrounding_prediction_centers.items():
                    if sur_id[3:] == '1':
                        sur_center = sur_envelope['reference']['position'][j]
                        relate_dist = ca.fabs(x_pred - sur_center)
                        safe_threshold = self.mpc_params.veh_length + self.mpc_params.dist_min + self.mpc_params.safe_dist_min
                        ref_dist = - 5 * (relate_dist - safe_threshold)
                        # # 理想选择：sigmoid形式--会出现梯度计算问题
                        # safety_cost += 1 / (1 + ca.exp(-ref_dist))
                        # 近似选择--tanh函数，转换为[0,1]
                        safety_cost += 0.5 * (1 + ca.tanh(ref_dist))
                        # # 近似选择--softplus函数的变体：log(1 + exp(x))
                        # safety_cost += ca.log(1 + ca.exp(ca.fmin(ref_dist, 10))) / ca.log(1 + ca.exp(10))
                    else:
                        safety_cost += 0
                cost += self.p_ws * safety_cost
            else:
                cost += 0

        # # 预测性安全代价（含协同性安全）--简单测试形式
        # for j in range(1, Np + 1):
        #     x_pred = self.X[0, j]
        #     if self.surrounding_prediction_centers != None:
        #         safety_cost = 0
        #         for sur_id, sur_envelope in self.surrounding_prediction_centers.items():
        #             if sur_id[3:] == '1':
        #                 sur_center = sur_envelope['reference']['position'][j]
        #
        #                 dist_diff = x_pred - sur_center
        #                 safe_threshold = self.mpc_params.veh_length + self.mpc_params.dist_min + self.mpc_params.safe_dist_min
        #
        #                 # 使用倒数形式：1/(distance² + threshold²)
        #                 # 距离越小，代价越高；距离很大时代价趋于0
        #                 eps = 1e-4
        #                 distance_sq = dist_diff ** 2 + eps
        #                 safety_cost += safe_threshold ** 2 / (distance_sq + safe_threshold ** 2)
        #
        #         cost += self.p_ws * safety_cost

        # 设置目标函数
        self.opti.minimize(cost)

        # 求解器设置
        solver_opts = {
            'ipopt': {
                'print_level': 2,
                'max_iter': 100,
                'tol': 1e-6
            }
        }
        self.opti.solver('ipopt', solver_opts)

    def solve_mpc(self,
                  current_state: VehicleState,
                  reference_speed: float,
                  target_y: float,
                  weights: WeightConfig,
                  surrounding_prediction_centers: Dict) -> MPCResult:
        """
        求解MPC优化问题

        Args:
            current_state: 当前车辆状态
            reference_speed: 参考速度
            target_y: 目标横向位置
            weights: 权重配置
            **kwargs: 其他参数（为兼容性保留）

        Returns:
            MPC求解结果
        """
        start_time = time.time()
        self.surrounding_prediction_centers = surrounding_prediction_centers

        # 准备初始状态
        x0 = [current_state.x, current_state.y, current_state.v,
              (current_state.theta - 90) * np.pi / 180, current_state.a]
        # 构建MPC优化问题
        self._build_mpc_problem()

        # 设置参数值
        self.opti.set_value(self.p_x0, x0)
        self.opti.set_value(self.p_v_ref, reference_speed)
        self.opti.set_value(self.p_y_target, target_y)
        self.opti.set_value(self.p_we, weights.we)
        self.opti.set_value(self.p_wc, weights.wc)
        self.opti.set_value(self.p_wt, weights.wt)
        self.opti.set_value(self.p_ws, weights.ws)

        # 求解优化问题
        try:
            sol = self.opti.solve()
            solve_time = time.time() - start_time

            # 提取解
            X_opt = sol.value(self.X)
            U_opt = sol.value(self.U)

            # 构建结果
            optimal_control = (float(U_opt[0, 0]), float(U_opt[1, 0]))

            # 预测状态序列
            predicted_states = []
            for k in range(1, self.mpc_params.prediction_horizon + 1):
                state = [
                    float(X_opt[0, k]),  # x
                    float(X_opt[1, k]),  # y
                    float(X_opt[2, k]),  # v
                    float(X_opt[3, k]),  # psi
                    float(X_opt[4, k])  # a
                ]
                predicted_states.append(state)

            # 控制序列
            control_sequence = []
            for k in range(self.mpc_params.control_horizon):
                control = (float(U_opt[0, k]), float(U_opt[1, k]))
                control_sequence.append(control)

            return MPCResult(
                optimal_control=optimal_control,
                predicted_states=predicted_states,
                control_sequence=control_sequence,
                optimal_cost=float(sol.value(self.opti.f)),
                solve_time=solve_time,
                feasible=True,
                status=self.opti.stats()['return_status']
            )

        except RuntimeError as e:
            solve_time = time.time() - start_time
            print(f"MPC求解失败: {e}")

            # 返回备选控制策略
            return self._get_fallback_result(current_state, reference_speed, solve_time, str(e))

    def _get_fallback_result(self, current_state: VehicleState, reference_speed: float,
                             solve_time: float, error_msg: str) -> MPCResult:
        """获取备选控制结果"""
        # 简单的比例控制
        speed_error = reference_speed - current_state.v
        a_u = np.clip(speed_error * 0.5, self.mpc_params.a_min, self.mpc_params.a_max)

        fallback_control = (a_u, 0.0)

        # 生成简单的预测轨迹
        predicted_states = []
        control_sequence = [(a_u, 0.0)] * self.mpc_params.control_horizon

        # 简单的前向预测
        state = [current_state.x, current_state.y, current_state.v, current_state.theta, current_state.a]
        for k in range(self.mpc_params.prediction_horizon):
            dt = self.mpc_params.sampling_time
            tau = self.mpc_params.tau

            state[0] += state[2] * np.cos(state[3]) * dt
            state[1] += state[2] * np.sin(state[3]) * dt
            state[2] += state[4] * dt
            state[3] += 0.0 * dt  # omega = 0
            state[4] += (-state[4] / tau + a_u / tau) * dt

            predicted_states.append(state.copy())

        return MPCResult(
            optimal_control=fallback_control,
            predicted_states=predicted_states,
            control_sequence=control_sequence,
            optimal_cost=float('inf'),
            solve_time=solve_time,
            feasible=False,
            status=f"FAILED: {error_msg}"
        )