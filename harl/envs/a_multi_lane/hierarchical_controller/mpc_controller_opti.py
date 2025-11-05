import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from harl.envs.a_multi_lane.project_structure import (
    SystemParameters, VehicleState, VehicleType, LateralDecision, WeightConfig
)

class MPCController:
    """
    基于PDF文档的MPC控制器实现
    状态向量: [x, y, v, ψ, a]
    控制向量: [a_u, ω]
    """

    def __init__(self):
        # 基本参数
        self.dt = 0.1  # 采样时间 [s]
        self.tau = 0.5  # 加速度一阶滞后时间常数 [s]

        # MPC参数
        self.Np = 20  # 预测时域长度
        self.Nc = 10  # 控制时域长度

        # 约束参数
        self.v_min = 0.0  # 最小速度 [m/s]
        self.v_max = 20.0  # 最大速度 [m/s]
        self.a_min = -5.0  # 最小加速度 [m/s²]
        self.a_max = 3.0  # 最大加速度 [m/s²]
        self.omega_min = -0.5  # 最小横摆角速度 [rad/s]
        self.omega_max = 0.5  # 最大横摆角速度 [rad/s]
        self.y_road_min = -3.2  # 道路左边界 [m]
        self.y_road_max = 3.2  # 道路右边界 [m]

        # 控制变化率约束
        self.a_dot_max = 3.0  # 加速度变化率 [m/s³]
        self.omega_dot_max = 1.0  # 角速度变化率 [rad/s²]

        # 代价函数权重
        self.weights = {
            'efficiency': 1.0,  # 效率权重
            'comfort': 0.5,  # 舒适性权重
            'task': 2.0  # 任务执行权重
        }

        # 子目标权重
        self.beta1 = 0.1  # 纵向加速度权重
        self.beta2 = 0.1  # 角速度权重

    def vehicle_dynamics(self, state, control):
        """
        车辆运动学模型
        state: [x, y, v, ψ, a]
        control: [a_u, ω]
        """
        x, y, v, psi, a = state[0], state[1], state[2], state[3], state[4]
        a_u, omega = control[0], control[1]

        # 状态方程 (PDF公式6-10)
        x_next = x + v * ca.cos(psi) * self.dt
        y_next = y + v * ca.sin(psi) * self.dt
        v_next = v + a * self.dt
        psi_next = psi + omega * self.dt
        a_next = a + (-a / self.tau + a_u / self.tau) * self.dt

        return ca.vertcat(x_next, y_next, v_next, psi_next, a_next)

    def efficiency_cost(self, states, v_ref):
        """
        效率代价函数 (PDF公式21)
        """
        cost = 0
        for j in range(self.Np):
            v_pred = states[2, j + 1]  # 预测速度
            cost += ((v_ref - v_pred) / self.v_max) ** 2   # 归一化
        return cost

    def comfort_cost(self, controls):
        """
        舒适性代价函数 (PDF公式22)
        """
        cost = 0
        for j in range(self.Nc):
            a_u = controls[0, j]
            omega = controls[1, j]
            cost += self.beta1 * (a_u/self.a_max) ** 2 + self.beta2 * (omega/self.omega_max) ** 2
        return cost

    def task_cost(self, states, y_target):
        """
        横向任务执行代价 (PDF公式23)
        """
        cost = 0
        for j in range(self.Np):
            y_pred = states[1, j + 1]  # 预测横向位置
            cost += ((y_pred - y_target) / self.y_road_max) ** 2
        return cost

    def solve_mpc(self, current_state, reference_params):
        """
        求解MPC优化问题

        Parameters:
        -----------
        current_state : np.ndarray
            当前状态 [x, y, v, ψ, a]
        reference_params : dict
            参考参数 {'v_ref': 速度参考, 'y_target': 目标车道}

        Returns:
        --------
        dict: MPC求解结果
        """
        # 创建优化问题
        opti = ca.Opti()

        # 决策变量
        X = opti.variable(5, self.Np + 1)  # 状态序列 [x, y, v, ψ, a]
        U = opti.variable(2, self.Nc)  # 控制序列 [a_u, ω]

        # 初始状态约束
        opti.subject_to(X[:, 0] == current_state)

        # 动力学约束
        for k in range(self.Np):
            if k < self.Nc:
                u_k = U[:, k]
            else:
                # 控制时域之后，控制输入保持最后一个值
                u_k = U[:, -1]

            x_next = self.vehicle_dynamics(X[:, k], u_k)
            opti.subject_to(X[:, k + 1] == x_next)

        # 状态约束
        for k in range(1, self.Np + 1):
            # 速度约束 (PDF公式24)
            opti.subject_to(opti.bounded(self.v_min, X[2, k], self.v_max))

            # 道路边界约束 (PDF公式25)
            opti.subject_to(opti.bounded(self.y_road_min, X[1, k], self.y_road_max))

        # 控制输入约束
        for k in range(self.Nc):
            # 加速度约束 (PDF公式26)
            opti.subject_to(opti.bounded(self.a_min, U[0, k], self.a_max))

            # 横摆角速度约束 (PDF公式27)
            opti.subject_to(opti.bounded(self.omega_min, U[1, k], self.omega_max))

        # 控制变化率约束
        for k in range(self.Nc - 1):
            # 加速度变化率约束 (PDF公式28)
            delta_a = U[0, k + 1] - U[0, k]
            opti.subject_to(opti.bounded(-self.a_dot_max * self.dt,
                                         delta_a,
                                         self.a_dot_max * self.dt))

            # 角速度变化率约束 (PDF公式29)
            delta_omega = U[1, k + 1] - U[1, k]
            opti.subject_to(opti.bounded(-self.omega_dot_max * self.dt,
                                         delta_omega,
                                         self.omega_dot_max * self.dt))

        # 多目标代价函数
        v_ref = reference_params.get('v_ref', 15.0)
        y_target = reference_params.get('y_target', 0.0)

        J_efficiency = self.efficiency_cost(X, v_ref)
        J_comfort = self.comfort_cost(U)
        J_task = self.task_cost(X, y_target)

        # 总代价函数
        total_cost = (self.weights['efficiency'] * J_efficiency +
                      self.weights['comfort'] * J_comfort +
                      self.weights['task'] * J_task)

        opti.minimize(total_cost)

        # 求解器设置
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-6
            }
        }
        opti.solver('ipopt', opts)

        # 求解
        try:
            sol = opti.solve()

            return {
                'states': sol.value(X),
                'controls': sol.value(U),
                'cost': sol.value(total_cost),
                'status': 'success'
            }
        except RuntimeError as e:
            print(f"MPC求解失败: {e}")
            return {
                'states': None,
                'controls': None,
                'cost': np.inf,
                'status': 'failed'
            }

    def get_control_command(self, current_state, reference_params):
        """
        获取当前时刻的控制指令
        """
        result = self.solve_mpc(current_state, reference_params)

        if result['status'] == 'success':
            # 返回第一个控制指令
            a_u_cmd = float(result['controls'][0, 0])
            omega_cmd = float(result['controls'][1, 0])
            return a_u_cmd, omega_cmd, result
        else:
            # 失败时返回安全控制指令
            return 0.0, 0.0, result


# 仿真测试函数
def simulate_mpc_tracking():
    """
    MPC轨迹跟踪仿真
    """
    # 初始化MPC控制器
    mpc = MPCController()

    # 仿真参数
    sim_time = 30.0  # 仿真时间 [s]
    steps = int(sim_time / mpc.dt)

    # 初始状态 [x, y, v, ψ, a]
    state = np.array([0.0, 1.0, 10.0, 0.0, 0.0])

    # 参考参数
    reference = {
        'v_ref': 15.0,  # 参考速度
        'y_target': 0.0  # 目标车道中心
    }

    # 存储仿真结果
    states_history = [state.copy()]
    controls_history = []
    costs_history = []

    print("开始MPC仿真...")

    for step in range(steps):
        # 获取控制指令
        a_u, omega, mpc_result = mpc.get_control_command(state, reference)

        if mpc_result['status'] == 'success':
            controls_history.append([a_u, omega])
            costs_history.append(mpc_result['cost'])

            # 状态更新（使用实际车辆动力学）
            x, y, v, psi, a = state

            # 状态更新方程
            state[0] = x + v * np.cos(psi) * mpc.dt
            state[1] = y + v * np.sin(psi) * mpc.dt
            state[2] = v + a * mpc.dt
            state[3] = psi + omega * mpc.dt
            state[4] = a + (-a / mpc.tau + a_u / mpc.tau) * mpc.dt

            states_history.append(state.copy())

            if step % 50 == 0:
                print(f"Step {step}: x={state[0]:.2f}, y={state[1]:.2f}, "
                      f"v={state[2]:.2f}, ψ={state[3]:.3f}, a={state[4]:.2f}")
        else:
            print(f"Step {step}: MPC求解失败")
            break

    # 转换为numpy数组便于分析
    states_history = np.array(states_history)
    controls_history = np.array(controls_history)

    return states_history, controls_history, costs_history

if __name__ == "__main__":
    simulate_mpc_tracking()