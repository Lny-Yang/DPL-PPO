"""
可微分物理模拟器（简化版点质量模型）
基于论文《Learning vision-based agile flight via differentiable physics》

用于生成未来状态预测，支持梯度反向传播
核心思想：将物理方程作为可微分的神经网络层

作者：基于论文改编
日期：2025年
"""
import torch
import torch.nn as nn

class DifferentiablePointMassSimulator(nn.Module):
    """可微分点质量无人机模拟器
    
    物理模型：
        v_{t+1} = v_t + a_t * dt
        p_{t+1} = p_t + v_t * dt + 0.5 * a_t * dt^2
    
    其中加速度a_t由动作分解：
        a_forward = thrust * vel_direction
        a_lateral = torque * vel_direction_perpendicular
        a_t = a_forward + a_lateral
    """
    
    def __init__(self, dt=1/30, enforce_planar=True, air_resistance=0.0):
        """
        Args:
            dt: 时间步长（默认1/30秒，对应30Hz控制频率）
            enforce_planar: 是否强制平面运动（2D）
            air_resistance: 空气阻力系数（0表示无阻力）
        """
        super().__init__()
        self.dt = dt
        self.enforce_planar = enforce_planar
        self.air_resistance = air_resistance
        
    def forward(self, initial_state, actions):
        """
        前向仿真（完全可微分）
        
        Args:
            initial_state: (batch, state_dim) 初始状态
                格式：[pos_x, pos_y, vel_x, vel_y, ...]
            actions: (batch, horizon, action_dim) 动作序列
                格式：[thrust, torque] 或 [thrust, torque, ...]
            
        Returns:
            trajectory: (batch, horizon, state_dim) 预测的状态轨迹
        """
        batch_size = initial_state.shape[0]
        horizon = actions.shape[1]
        device = initial_state.device
        
        # 初始化轨迹存储
        trajectory = torch.zeros(batch_size, horizon, initial_state.shape[1], device=device)
        
        # 当前状态（可微分的叶子节点）
        pos = initial_state[:, :2].clone()  # [batch, 2]
        vel = initial_state[:, 2:4].clone()  # [batch, 2]
        
        # 逐步积分（每一步都保持梯度流）
        for t in range(horizon):
            # 提取动作
            thrust = actions[:, t, 0:1]  # [batch, 1]
            torque = actions[:, t, 1:2]  # [batch, 1]
            
            # 🔥 点质量动力学（可微分）
            # 计算速度方向（归一化）
            vel_norm = torch.norm(vel, dim=1, keepdim=True).clamp(min=1e-6)
            vel_dir = vel / vel_norm  # [batch, 2]
            
            # 加速度分解
            # 前进加速度：沿速度方向
            accel_forward = thrust * vel_dir  # [batch, 2]
            
            # 侧向加速度：垂直于速度方向
            # vel_dir = [vx, vy] → perpendicular = [-vy, vx]
            vel_perpendicular = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
            accel_lateral = torque * vel_perpendicular  # [batch, 2]
            
            # 总加速度
            accel = accel_forward + accel_lateral  # [batch, 2]
            
            # 🔥 可选：添加空气阻力（与速度成正比）
            if self.air_resistance > 0:
                drag = -self.air_resistance * vel
                accel = accel + drag
            
            # 🔥 欧拉积分（一阶近似，可改用RK4提高精度）
            vel_new = vel + accel * self.dt
            pos_new = pos + vel * self.dt + 0.5 * accel * self.dt**2
            
            # 保存到轨迹（保持梯度）
            trajectory[:, t, :2] = pos_new
            trajectory[:, t, 2:4] = vel_new
            
            # 其他状态维度保持不变（如姿态、角速度等）
            if initial_state.shape[1] > 4:
                trajectory[:, t, 4:] = initial_state[:, 4:]
            
            # 更新状态（下一步迭代）
            pos = pos_new
            vel = vel_new
        
        return trajectory
    
    def predict_single_step(self, state, action):
        """
        预测单步状态转移（用于快速验证）
        
        Args:
            state: (batch, state_dim) 当前状态
            action: (batch, action_dim) 单步动作
            
        Returns:
            next_state: (batch, state_dim) 下一状态
        """
        # 将单步动作扩展为horizon=1
        action_expanded = action.unsqueeze(1)  # [batch, 1, action_dim]
        trajectory = self.forward(state, action_expanded)
        return trajectory[:, 0, :]  # 返回第一步


class RK4PointMassSimulator(nn.Module):
    """使用Runge-Kutta 4阶方法的高精度模拟器
    
    相比欧拉法，RK4具有更高的数值精度，但计算量增加4倍
    适用于需要高精度仿真的场景
    """
    
    def __init__(self, dt=1/30, enforce_planar=True):
        super().__init__()
        self.dt = dt
        self.enforce_planar = enforce_planar
        
    def _dynamics(self, pos, vel, thrust, torque):
        """计算动力学导数 dx/dt
        
        Returns:
            dpos/dt, dvel/dt
        """
        # 速度方向
        vel_norm = torch.norm(vel, dim=1, keepdim=True).clamp(min=1e-6)
        vel_dir = vel / vel_norm
        
        # 加速度
        accel_forward = thrust * vel_dir
        vel_perpendicular = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
        accel_lateral = torque * vel_perpendicular
        accel = accel_forward + accel_lateral
        
        return vel, accel  # dpos/dt = vel, dvel/dt = accel
    
    def forward(self, initial_state, actions):
        """RK4积分"""
        batch_size = initial_state.shape[0]
        horizon = actions.shape[1]
        device = initial_state.device
        
        trajectory = torch.zeros(batch_size, horizon, initial_state.shape[1], device=device)
        
        pos = initial_state[:, :2].clone()
        vel = initial_state[:, 2:4].clone()
        
        for t in range(horizon):
            thrust = actions[:, t, 0:1]
            torque = actions[:, t, 1:2]
            
            # RK4 四个阶段
            # k1
            dp1, dv1 = self._dynamics(pos, vel, thrust, torque)
            
            # k2
            pos_k2 = pos + 0.5 * self.dt * dp1
            vel_k2 = vel + 0.5 * self.dt * dv1
            dp2, dv2 = self._dynamics(pos_k2, vel_k2, thrust, torque)
            
            # k3
            pos_k3 = pos + 0.5 * self.dt * dp2
            vel_k3 = vel + 0.5 * self.dt * dv2
            dp3, dv3 = self._dynamics(pos_k3, vel_k3, thrust, torque)
            
            # k4
            pos_k4 = pos + self.dt * dp3
            vel_k4 = vel + self.dt * dv3
            dp4, dv4 = self._dynamics(pos_k4, vel_k4, thrust, torque)
            
            # 组合更新
            pos_new = pos + (self.dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
            vel_new = vel + (self.dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
            
            trajectory[:, t, :2] = pos_new
            trajectory[:, t, 2:4] = vel_new
            if initial_state.shape[1] > 4:
                trajectory[:, t, 4:] = initial_state[:, 4:]
            
            pos = pos_new
            vel = vel_new
        
        return trajectory
