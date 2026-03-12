"""
物理一致性损失计算模块
基于论文《Learning vision-based agile flight via differentiable physics》

包含五个物理损失分量：
1. 速度跟踪损失 - 鼓励向目标方向飞行
2. 障碍物规避损失 - 基于论文公式的软约束
3. 控制平滑性损失 - 惩罚加加速度
4. 能量效率损失 - 降低能耗
5. 动力学可行性损失 - 确保动作在物理约束内

作者：基于Stable-Baselines3和论文改编
日期：2025年
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsLossCalculator(nn.Module):
    """物理一致性损失计算器"""
    
    def __init__(self, config):
        """
        Args:
            config: dict包含物理参数
                - dt: 时间步长 (默认1/30)
                - horizon: 仿真展开步数 (默认5)
                - safe_distance: 安全距离阈值 (默认1.0米)
                - max_acceleration: 最大加速度 (默认5.0 m/s²)
                - drone_radius: 无人机半径 (默认0.2米)
                - weights: 各损失项权重字典
                - decay_alpha: 时间梯度衰减系数 (默认0.5)
        """
        super().__init__()
        self.dt = config.get('dt', 1/30)
        self.horizon = config.get('horizon', 5)
        self.safe_distance = config.get('safe_distance', 1.0)
        self.max_acceleration = config.get('max_acceleration', 5.0)
        self.drone_radius = config.get('drone_radius', 0.2)
        
        # 损失权重
        weights = config.get('weights', {})
        self.w_velocity = weights.get('velocity', 1.0)
        self.w_obstacle = weights.get('obstacle', 2.0)
        self.w_smooth = weights.get('smooth', 0.1)
        self.w_energy = weights.get('energy', 0.01)
        self.w_feasibility = weights.get('feasibility', 0.5)
        
        # 时间衰减参数（论文中的α）
        self.decay_alpha = config.get('decay_alpha', 0.5)
        
    def forward(self, states, actions, next_states_pred, env_info):
        """
        计算总物理损失
        
        Args:
            states: (batch, state_dim) 当前状态 [pos_x, pos_y, vel_x, vel_y]
            actions: (batch, horizon, action_dim) 动作序列
            next_states_pred: (batch, horizon, state_dim) 预测的未来状态
            env_info: dict包含环境信息
                - target_velocity: (batch, 2) 目标速度
                - depth_maps: (batch, H, W) 深度图（可选）
                - obstacles: list of dicts（可选）
                
        Returns:
            loss_dict: 各损失分量的字典
        """
        batch_size = states.shape[0]
        device = states.device
        
        # 初始化损失累积
        total_velocity_loss = torch.zeros(batch_size, device=device)
        total_obstacle_loss = torch.zeros(batch_size, device=device)
        total_smooth_loss = torch.zeros(batch_size, device=device)
        total_energy_loss = torch.zeros(batch_size, device=device)
        total_feasibility_loss = torch.zeros(batch_size, device=device)

        # 🆕 约束违约统计（用于对偶更新/可解释日志）
        obstacle_violation_rate_sum = torch.zeros(batch_size, device=device)
        feasibility_violation_rate_sum = torch.zeros(batch_size, device=device)
        
        # 逐步计算物理损失（带时间衰减）
        for t in range(self.horizon):
            # 时间衰减因子（论文公式：e^(-α·t·Δt)）
            decay = torch.exp(torch.tensor(-self.decay_alpha * (t+1) * self.dt, device=device))
            
            # 当前步的动作和预测状态
            action_t = actions[:, t]
            next_state_t = next_states_pred[:, t]
            
            # 1️⃣ 速度跟踪损失
            if 'target_velocity' in env_info:
                vel_loss = self._velocity_tracking_loss(
                    next_state_t, 
                    env_info['target_velocity']
                )
                total_velocity_loss += decay * vel_loss
            
            # 2️⃣ 障碍物规避损失
            if 'depth_maps' in env_info or 'obstacles' in env_info:
                obs_loss, obs_stats = self._obstacle_avoidance_loss(
                    next_state_t,
                    env_info
                )
                total_obstacle_loss += decay * obs_loss

                # 违约率（不加时间衰减，便于解释为“未来horizon内违约比例”）
                if obs_stats is not None and 'violation_rate' in obs_stats:
                    obstacle_violation_rate_sum += obs_stats['violation_rate']
            
            # 3️⃣ 控制平滑性损失
            if t > 0:
                smooth_loss = self._control_smoothness_loss(
                    actions[:, t],
                    actions[:, t-1]
                )
                total_smooth_loss += decay * smooth_loss
            
            # 4️⃣ 能量效率损失
            energy_loss = self._energy_efficiency_loss(action_t)
            total_energy_loss += decay * energy_loss
            
            # 5️⃣ 动力学可行性损失
            if t > 0:
                feas_loss, feas_stats = self._dynamic_feasibility_loss(
                    next_states_pred[:, t-1],
                    action_t,
                    next_state_t
                )
                total_feasibility_loss += decay * feas_loss

                if feas_stats is not None and 'violation_rate' in feas_stats:
                    feasibility_violation_rate_sum += feas_stats['violation_rate']
        
        # raw（不加权/不截断）统计：更适合做“约束指标/对偶更新”
        obstacle_raw_mean = total_obstacle_loss.mean()
        energy_raw_mean = total_energy_loss.mean()
        feasibility_raw_mean = total_feasibility_loss.mean()

        obstacle_violation_rate = (obstacle_violation_rate_sum / float(self.horizon)).mean()
        # feasibility 在 t>0 才计算，分母用 (horizon-1)
        feasibility_den = max(self.horizon - 1, 1)
        feasibility_violation_rate = (feasibility_violation_rate_sum / float(feasibility_den)).mean()

        # 计算加权总损失（对 obstacle 进行上限截断，防止极端尖峰导致训练不稳定）
        loss_components = {
            'velocity': self.w_velocity * total_velocity_loss.mean(),
            'obstacle': self.w_obstacle * total_obstacle_loss.mean(),  
            'smooth': self.w_smooth * total_smooth_loss.mean(),
            'energy': self.w_energy * total_energy_loss.mean(),
            'feasibility': self.w_feasibility * total_feasibility_loss.mean()
        }
        
        loss_components['total'] = sum(loss_components.values())

        # 🆕 额外输出：raw分量与违约率（不影响现有训练逻辑，只提供更多可控信号）
        loss_components['obstacle_raw'] = obstacle_raw_mean
        loss_components['energy_raw'] = energy_raw_mean
        loss_components['feasibility_raw'] = feasibility_raw_mean
        loss_components['obstacle_violation_rate'] = obstacle_violation_rate
        loss_components['feasibility_violation_rate'] = feasibility_violation_rate
        
        return loss_components
    
    def _velocity_tracking_loss(self, next_state, target_velocity):
        """
        速度跟踪损失（论文中的L_v）
        
        鼓励无人机速度接近目标方向和期望速度
        使用Smooth L1损失（比MSE更鲁棒）
        """
        # 提取预测速度（假设state格式：[pos_x, pos_y, vel_x, vel_y, ...]）
        pred_velocity = next_state[:, 2:4]  # 平面速度
        target_vel = target_velocity[:, :2] if target_velocity.shape[1] > 2 else target_velocity
        
        # Smooth L1损失（比MSE更鲁棒，对异常值不敏感）
        return F.smooth_l1_loss(pred_velocity, target_vel, reduction='none').sum(dim=1)
    
    def _obstacle_avoidance_loss(self, next_state, env_info):
        """
        障碍物规避损失（论文中的L_c）
        
        使用论文公式：
        L_obs = v^c * [(1 - (d - r_d))^2 + 2.5 * log(1 + exp(32*(r_d - d)))]
        
        其中：
        - v^c: 接近速度（在障碍物方向上的速度分量）
        - d: 到障碍物的距离
        - r_d: 无人机半径
        """
        batch_size = next_state.shape[0]
        device = next_state.device
        
        # 提取位置和速度
        position = next_state[:, :2]  # [batch, 2]
        velocity = next_state[:, 2:4]  # [batch, 2]
        
        loss = torch.zeros(batch_size, device=device)

        violation_rate = torch.zeros(batch_size, device=device)
        
        # 方式1：基于深度图
        if 'depth_maps' in env_info:
            depth_maps = env_info['depth_maps']  # [batch, H, W]
            
            # 深度图是前视相机，不是俯视图
            # 使用中心区域的最小深度作为前方障碍物距离
            map_h, map_w = depth_maps.shape[1], depth_maps.shape[2]
            
            # 提取中心区域（前方主要障碍物）
            center_h_start = map_h // 4
            center_h_end = 3 * map_h // 4
            center_w_start = map_w // 4
            center_w_end = 3 * map_w // 4
            
            center_region = depth_maps[:, center_h_start:center_h_end, center_w_start:center_w_end]
            
            # 计算前方最小距离(障碍物距离)
            distances = torch.min(center_region.reshape(batch_size, -1), dim=1)[0]  # [batch]
            
            # 计算接近速度（前进速度，简化为速度模长）
            approach_velocity = torch.norm(velocity, dim=1).clamp(min=0)
            
            # 论文公式（只对距离<安全距离的情况计算）
            safe_mask = distances < self.safe_distance
            if safe_mask.any():
                d_k = distances[safe_mask]
                v_c = approach_velocity[safe_mask]
                r_d = self.drone_radius
                
                # 截断二次项
                truncation = torch.relu(1 - (d_k - r_d)) ** 2
                
                # 软障碍函数（论文公式）
                softplus = torch.log(1 + torch.exp(32 * (r_d - d_k)))
                
                # 组合损失
                obs_loss = v_c * (truncation + 2.5 * softplus)
                loss[safe_mask] = obs_loss

            violation_rate = safe_mask.float()
        
        # 方式2：基于显式障碍物列表
        elif 'obstacles' in env_info:
            for obs in env_info['obstacles']:
                obs_pos = obs['position']  # [2] or [batch, 2]
                obs_radius = obs.get('radius', 0.5)
                
                # 计算距离
                if obs_pos.dim() == 1:
                    obs_pos = obs_pos.unsqueeze(0).expand(batch_size, -1)
                dist = torch.norm(position - obs_pos, dim=1) - obs_radius
                
                # 计算接近速度
                to_obs = obs_pos - position
                to_obs_norm = F.normalize(to_obs, dim=1)
                approach_vel = torch.sum(velocity * to_obs_norm, dim=1).clamp(min=0)
                
                # 应用论文公式
                safe_mask = dist < self.safe_distance
                if safe_mask.any():
                    loss[safe_mask] += approach_vel[safe_mask] * torch.relu(1 - dist[safe_mask])**2

                violation_rate = torch.maximum(violation_rate, safe_mask.float())

        stats = {
            'violation_rate': violation_rate
        }
        return loss, stats
    
    def _control_smoothness_loss(self, action_t, action_prev):
        """
        控制平滑性损失（惩罚加加速度）
        
        鼓励动作序列平滑，避免剧烈变化
        L_smooth = ||u_t - u_{t-1}||^2
        """
        # L2范数的平方（加速度变化率）
        jerk = action_t - action_prev
        return torch.sum(jerk**2, dim=1)
    
    def _energy_efficiency_loss(self, action):
        """
        能量效率损失
        
        惩罚过大的控制输入（降低能耗）
        假设action = [thrust, torque]
        """
        # 推力和扭矩的平方和
        return torch.sum(action**2, dim=1)
    
    def _dynamic_feasibility_loss(self, state_prev, action, state_next):
        """
        动力学可行性损失
        
        确保状态转移符合物理约束
        检查加速度是否超出最大限制
        """
        # 计算实际加速度（从状态变化推导）
        vel_prev = state_prev[:, 2:4]
        vel_next = state_next[:, 2:4]
        accel_actual = (vel_next - vel_prev) / self.dt
        accel_norm = torch.norm(accel_actual, dim=1)
        
        # 超出最大加速度的惩罚
        excess = torch.relu(accel_norm - self.max_acceleration)
        violation_rate = (excess > 0).float()

        stats = {
            'violation_rate': violation_rate,
            'excess_mean': excess.mean(),
        }

        return excess ** 2, stats
