from __future__ import annotations
import math, random
from typing import List, Tuple, Dict, Any
import gym, numpy as np, pybullet as p
import pybullet_data
from ..resources.drone import Drone
from ..utils.depth_obstacle_processor import DepthObstacleProcessor
from ..utils.reward_calculator import RewardCalculator, create_default_reward_config
from ..utils.state_processor import StateProcessor, create_default_state_config
from ..utils.environment_manager import EnvironmentManager, create_default_environment_config
from ..utils.camera_manager import CameraManager, create_default_camera_config
from ..utils.observation_manager import ObservationSpaceManager, create_default_observation_config
from ..config import multi_drone_env as config
import time
__all__ = ["DroneNavigationMulti"]

class DroneNavigationMulti(gym.Env):
    
    """简化版多无人机单目标编队导航环境 """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self,
                 num_drones: int = 5,
                 environment_type: str = "indoor",  # 恢复：默认使用室内环境，与用户原始环境一致
                 enforce_planar: bool | None = None,
                 use_depth_camera: bool = True,
                 depth_camera_range: float = 10.0,
                 camera_pixel: int = 64,
                 depth_resolution: int = 16,
                 formation_distance: float = config.get("formation_distance", 0.05),
                 max_steps: int = 3000,  # 修复：从1500增加到3000，给予充足时间到达目标
                 success_radius_xy: float = 2.5,  # 修复：从1.5增加到2.5米，放宽成功条件
                 success_height_tol: float = 0.5,  # 修复：从1.0降低到0.5米，提高精度要求
                 catchup_gain_pos: float = 1.5,
                 catchup_gain_pos_z: float = 2.0,
                 catchup_gain_speed: float = 2.0,
                 catchup_speed_target: float = 5.0,
                 catchup_max_force_xy: float = 8.0,
                 catchup_max_force_z: float = 6.0,
                 dt: float = 1/30,
                 use_leader_camera: bool = True,
                 enable_formation_force: bool = False,
                 training_stage: int = 1,
                 # 新增：相机配置参数
                 enable_fixed_overhead_camera: bool = False,  # 是否启用固定俯视摄像头
                 fixed_camera_height: float = 3.0,  # 固定摄像头高度
                 fixed_camera_pitch: float = 1.5):  # 固定摄像头俯仰角
        # 基本参数
        self.num_drones = num_drones
        self.environment_type = environment_type
        self.enforce_planar = config.get("enforce_planar", True)  # 从配置文件读取平面模式设置
        self.use_leader_camera = use_leader_camera and use_depth_camera
        self.use_depth_camera = use_depth_camera
        self.depth_camera_range = depth_camera_range
        self.camera_pixel = camera_pixel
        self.depth_resolution = depth_resolution
        self.depth_feature_dim = config.get("depth_feature_dim", 130)  # 从配置文件获取总深度特征维度
        self.formation_distance = formation_distance
        self.max_steps = max_steps
        self.success_radius_xy = success_radius_xy
        self.success_height_tol = success_height_tol
        self.catchup_gain_pos = catchup_gain_pos
        self.catchup_gain_pos_z = catchup_gain_pos_z
        self.catchup_gain_speed = catchup_gain_speed
        self.catchup_speed_target = catchup_speed_target
        self.catchup_max_force_xy = catchup_max_force_xy
        self.catchup_max_force_z = catchup_max_force_z
        self.dt = dt
        self.leader_index = 0
        self.enable_formation_force = enable_formation_force
        
        # 训练阶段控制: 1=leader导航训练, 2=完整编队训练
        self.training_stage = training_stage
        
        # 新增：相机配置参数
        self.enable_fixed_overhead_camera = enable_fixed_overhead_camera
        self.fixed_camera_height = fixed_camera_height
        self.fixed_camera_pitch = fixed_camera_pitch

        # 物理世界
        self.client = p.connect(config['display'])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 配置物理引擎，减少旋转效应
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=50,  # 增加求解迭代次数提高稳定性
            numSubSteps=8,  # 增加子步骤数量，提高模拟精度
            enableConeFriction=1,  # 启用圆锥摩擦
            restitutionVelocityThreshold=0.05,  # 调低反弹速度阈值
            contactERP=0.8,  # 增大接触错误减少参数，减小震荡
            frictionERP=0.8,  # 增大摩擦错误减少参数，减小震荡
            physicsClientId=self.client
        )
        
        # 设置重力很小，减少旋转积累影响
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        # 初始化管理模块
        self.reward_calculator = RewardCalculator(create_default_reward_config())
        
        # 创建与相机像素一致的状态配置
        state_config = create_default_state_config()
        state_config['depth_height'] = self.camera_pixel
        state_config['depth_width'] = self.camera_pixel
        state_config['cnn_feature_dim'] = 128  # CNN特征维度
        self.state_processor = StateProcessor(state_config)
        
        self.environment_manager = EnvironmentManager(self.client, create_default_environment_config())
        
        # 创建与相机像素一致的相机配置
        camera_config = create_default_camera_config()
        camera_config['depth_width'] = self.camera_pixel
        camera_config['depth_height'] = self.camera_pixel
        # 根据构造函数参数配置固定俯视摄像头
        camera_config['fixed_overhead_camera'] = self.enable_fixed_overhead_camera
        camera_config['fixed_camera_height'] = self.fixed_camera_height
        camera_config['fixed_camera_pitch'] = self.fixed_camera_pitch
        self.camera_manager = CameraManager(self.client, camera_config)
        
        # 🔥 修复：CNN特征维度 + 2个额外特征（obstacle_detected, min_depth）
        # get_navigation_features() 返回: 128 CNN + 2 增强 = 130维
        depth_features_dim = 128 + 2
        
        self.observation_manager = ObservationSpaceManager(create_default_observation_config(
            num_agents=self.num_drones, 
            depth_features_dim=depth_features_dim,
            use_cnn_features=True,
            cnn_feature_dim=128
        ))
        
        # 深度障碍处理器和动作映射器
        self.depth_obstacle_processor = DepthObstacleProcessor(
            depth_image_size=(self.camera_pixel, self.camera_pixel),
            collision_threshold=config.get('collision_distance', 0.8) / config.get('depth_scale', 4.0),  # 使用config中的collision_distance
            depth_scale=config.get('depth_scale', 4.0),     # 从config读取
            max_depth=config.get('max_depth', 2.0),       # 从config读取
            cnn_feature_dim=config.get('cnn_feature_dim', 128)
        )
        
        # 观测和动作空间
        # 根据训练阶段设置观测空间
        if self.training_stage == 1:
            # 第一阶段：只训练领航者，使用单个智能体的观测空间
            single_agent_config = create_default_observation_config(
                num_agents=1,  # 只为领航者创建观测空间
                depth_features_dim=depth_features_dim,
                use_cnn_features=True,
                cnn_feature_dim=128,
                enforce_planar=self.enforce_planar  # 传递平面模式参数
            )
            single_agent_manager = ObservationSpaceManager(single_agent_config)
            self.observation_space = single_agent_manager.get_observation_space()
            
            # 第一阶段：只控制领航者，动作空间为2维 [thrust, torque] - 前进/后退力和转向扭矩
            # 使用配置文件中的推力限制，提高飞行速度
            act_high = np.array([config['thrust_upper_bound'], config['torque_upper_bound']], dtype=np.float32)
            act_low = np.array([config['thrust_lower_bound'], config['torque_lower_bound']], dtype=np.float32)
            self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        else:
            # 第二阶段：完整编队训练，使用所有智能体的观测空间
            self.observation_space = self.observation_manager.get_observation_space()
            
            # 第二阶段：控制所有无人机，每架无人机3维动作 [thrust_forward, thrust_lateral, thrust_z]
            # thrust_forward: 前进推力（机头坐标系），thrust_lateral: 侧向推力，thrust_z: 垂直推力
            # 从config读取动作空间限制
            act_high_single = np.array([config['thrust_x_upper_bound'], 
                                       config['thrust_y_upper_bound'], 
                                       config['thrust_z_upper_bound']], dtype=np.float32)
            act_high = np.tile(act_high_single, self.num_drones)
            act_low_single = np.array([config['thrust_x_lower_bound'], 
                                      config['thrust_y_lower_bound'], 
                                      config['thrust_z_lower_bound']], dtype=np.float32)
            act_low = np.tile(act_low_single, self.num_drones)
            self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # 设置物理世界
        self.environment_manager.setup_physics_world(self.dt, self.enforce_planar)

        # 状态变量
        self.goal = None
        self.goal_id = None
        self.drones = []
        self.current_step = 0
        self.success = False
        self.leader_rgb_image = None
        self.leader_depth_image = None

        # 初始化 (调用 reset 创建场景/无人机/目标)
        self.reset()




    def _get_formation_positions(self) -> List[np.ndarray]:
        """获取编队期望位置（考虑目标朝向）"""
        if self.goal is None:
            return [np.zeros(3) for _ in range(self.num_drones)]

        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        leader_pos = np.array(leader_pos)
        
        # 计算领航机朝向目标的方向
        vec = np.array(self.goal) - leader_pos
        yaw = math.atan2(vec[1], vec[0]) if np.linalg.norm(vec[:2]) > 1e-6 else 0.0
        cy, sy = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        
        # 方形编队模式
        body_offsets = [
            np.array([0, 0, 0]),  # 领航者
            np.array([self.formation_distance, self.formation_distance, 0]),
            np.array([self.formation_distance, -self.formation_distance, 0]),
            np.array([-self.formation_distance, self.formation_distance, 0]),
            np.array([-self.formation_distance, -self.formation_distance, 0])
        ]
        
        return [leader_pos + rot @ offset for offset in body_offsets[:self.num_drones]]

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        p.resetSimulation(self.client)
        
        # 设置物理世界
        self.environment_manager.setup_physics_world(self.dt, self.enforce_planar)

        # 生成环境
        env_info = self.environment_manager.generate_environment()

        # 采样目标
        self.goal = self.environment_manager.sample_goal()
        self.goal_id = self.environment_manager.create_goal_object(self.goal)

        # 创建无人机
        self.drones = [Drone(self.client) for _ in range(self.num_drones)]

        # 设置无人机起始位置
        self.environment_manager.set_drone_start_positions(self.drones, self.num_drones)
        
        # 记录起始位置，用于计算距起点的距离
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        self.start_position = np.array(leader_pos)
        self.max_traveled_distance = 0.0  # 记录最远移动距离

        # 设置领航者相机
        if self.use_leader_camera:
            # 启用合成相机视图显示
            self.camera_manager.enable_synthetic_camera_views()

        # 重置状态
        self.current_step = 0
        self.success = False
        
        # 重置处理模块
        self.reward_calculator.reset_state()
        self.camera_manager.cleanup()

        # 渲染环境
        self.render()

        obs = self._build_observation()
        return obs, {}

    def _leader_goal_distance(self) -> float:
        """计算领航者到目标距离"""
        if self.goal is None:
            return 0.0
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        return float(np.linalg.norm(np.array(leader_pos) - np.array(self.goal)))

    def _build_observation(self) -> np.ndarray:
        """构建观测 - 使用封装的状态处理器"""
        observations = []
        
        # 获取领航者深度图像（如果启用）
        leader_depth_image = None
        if self.use_leader_camera:
            leader_depth_image = self.get_leader_depth_image()
        
        # 为每个无人机构建观测
        for i in range(self.num_drones):
            pos, quat = p.getBasePositionAndOrientation(self.drones[i].drone, self.client)
            vel, _ = p.getBaseVelocity(self.drones[i].drone, self.client)
            pos = np.array(pos)
            quat = np.array(quat)
            vel = np.array(vel)
            
            # 使用状态处理器构建状态
            if i == self.leader_index:
                # 领航者使用深度信息
                obs = self.state_processor.build_state(
                    drone_id=i,
                    position=pos,
                    velocity=vel,
                    orientation=quat,
                    target_position=np.array(self.goal),
                    depth_image=leader_depth_image,
                    enforce_planar=self.enforce_planar  # 传递平面模式参数
                )
            else:
                # 跟随者不使用深度信息
                obs = self.state_processor.build_state(
                    drone_id=i,
                    position=pos,
                    velocity=vel,
                    orientation=quat,
                    target_position=np.array(self.goal),
                    depth_image=None,
                    enforce_planar=self.enforce_planar  # 传递平面模式参数
                )
            
            observations.append(obs)
        
        # 组合所有观测
        # 第一阶段训练：只返回领航者观测
        if self.training_stage == 1:
            final_obs = observations[self.leader_index].astype(np.float32)
        else:
            final_obs = np.concatenate(observations).astype(np.float32)
        
        # 🔥 数值稳定性检查：确保观测值不包含NaN或无穷值
        if np.any(np.isnan(final_obs)) or np.any(np.isinf(final_obs)):
            print(f"⚠️  警告: 观测值包含NaN或无穷值，正在修复...")
            # 替换NaN为0，替换无穷值为有限值
            final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            print(f"✅ 已修复观测值，范围: [{final_obs.min():.3f}, {final_obs.max():.3f}]")
        
        return final_obs


    def get_leader_depth_image(self):
        """获取领航者相机深度图像（用于避障）"""
        if hasattr(self, 'leader_depth_image'):
            return self.leader_depth_image
        else:
            cam_pos, cam_orn = self.drones[self.leader_index].get_camera_pose()
            _, depth_image = self.camera_manager.get_leader_camera_image_by_pose(cam_pos, cam_orn)
            return depth_image

    def _apply_formation_forces(self):
        """应用编队辅助力控制（仅跟随者）"""
        if not self.enable_formation_force or self.goal is None:
            return
            
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        leader_vel, _ = p.getBaseVelocity(self.drones[self.leader_index].drone, self.client)
        leader_pos = np.array(leader_pos)
        leader_speed = np.linalg.norm(np.array(leader_vel)[:2])

        formation_positions = self._get_formation_positions()

        # 只对跟随者施加编队辅助力（不控制领航者）
        for i in range(1, self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.drones[i].drone, self.client)
            vel, _ = p.getBaseVelocity(self.drones[i].drone, self.client)
            pos = np.array(pos)
            vel = np.array(vel)
            slot = formation_positions[i]

            # 位置误差力
            err = slot - pos
            lateral = err.copy()
            lateral[2] = 0
            force = self.catchup_gain_pos * lateral

            # 垂直误差力
            if not self.enforce_planar:
                force[2] = self.catchup_gain_pos_z * err[2]
            else:
                force[2] = 0.0

            # 速度同步力
            leader_dir = np.array(leader_vel)
            leader_dir[2] = 0
            if np.linalg.norm(leader_dir) > 1e-3:
                ld = leader_dir / np.linalg.norm(leader_dir)
                speed_i = np.linalg.norm(vel[:2])
                target = min(self.catchup_speed_target, leader_speed + 0.5)
                deficit = target - speed_i
                if deficit > 0:
                    force += self.catchup_gain_speed * deficit * ld

            # 限制力的大小
            force[:2] = np.clip(force[:2], -self.catchup_max_force_xy, self.catchup_max_force_xy)
            force[2] = np.clip(force[2], -self.catchup_max_force_z, self.catchup_max_force_z)

            p.applyExternalForce(self.drones[i].drone, -1, force.tolist(), [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)

    def _apply_drone_action(self, drone_idx, action):
        # 如果禁用编队力且是跟随者，则保持静止
        if hasattr(self, 'enable_formation_force') and not self.enable_formation_force and drone_idx > 0:
            # 重置速度保持静止
            p.resetBaseVelocity(self.drones[drone_idx].drone, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        else:
            # 【简化】只负责调用无人机的基本动作（推力和扭矩），不处理重力补偿
            self.drones[drone_idx].apply_action(action, apply_gravity_compensation=False)
            
            # 【统一】重力补偿统一在此处处理
            if not self.enforce_planar:
                # 3D模式：提供完整重力补偿
                drone_mass = p.getDynamicsInfo(self.drones[drone_idx].drone, -1, physicsClientId=self.client)[0]
                gravity_compensation = drone_mass * 9.8
                p.applyExternalForce(self.drones[drone_idx].drone, -1, [0, 0, gravity_compensation], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self.client)
            # 平面模式：不提供重力补偿（因为PyBullet重力设为0）
            
            # 注意：平面模式速度重置已移至step方法末尾，在物理仿真之后进行

    def _compute_reward(self) -> Tuple[float, Dict[str, float], bool]:
        """计算奖励 - 使用封装的奖励计算器"""
        done = False
        
        # 获取领航者状态
        leader_pos, leader_quat = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        leader_pos = np.array(leader_pos)
        leader_vel, leader_ang_vel = p.getBaseVelocity(self.drones[self.leader_index].drone, self.client)
        leader_vel = np.array(leader_vel)
        leader_ang_vel = np.array(leader_ang_vel)
        
        # 检查成功条件
        success = self._check_success()
        self.success = success  # 设置实例变量，确保step方法能返回正确的成功状态
        
        # 获取深度信息（包含碰撞信息）
        depth_info = self._get_depth_info()
        
        # 为避障奖励计算添加速度信息
        depth_info['velocity'] = np.linalg.norm(leader_vel[:2])  # 平面速度
        depth_info['angular_velocity'] = leader_ang_vel[2]  # 偏航角速度
        
        # 检查终止条件：成功、超时、或碰撞
        collision_occurred = depth_info.get('collision', False)
        done = success or self.current_step >= self.max_steps or collision_occurred
        
        # 🎯 使用极简奖励计算器（4组件无冗余设计）
        # 组件: success(+2000) + crash(-1500) + navigation(~1.5/step) + safe_navigation(~0.5/step)
        total_reward, reward_details = self.reward_calculator.compute_total_reward(
            drone_id="leader",
            position=leader_pos,
            target_position=np.array(self.goal),
            velocity=leader_vel,
            depth_info=depth_info,
            orientation=leader_quat,  # 朝向信息（用于navigation的对齐奖励）
            formation_info=None,  # 可以后续添加编队信息
            done=done,
            success=success,
            current_step=0  # 极简系统不使用步数惩罚，传0即可
        )
        
        return total_reward, reward_details, done
    
    def _get_depth_info(self) -> Dict[str, float]:
        """获取深度信息和碰撞信息"""
        depth_info = {'min_depth': float('inf'), 'mean_depth': 3.0, 'forward_min': 3.0, 'left_min': 3.0, 'right_min': 3.0}
        
        # 添加碰撞检测信息
        collision_info = self._check_collision()
        depth_info['collision'] = collision_info['collision']
        depth_info['collision_type'] = collision_info['collision_type']
        depth_info['contact_points'] = collision_info['contact_points']
        
        # 添加位置信息和到起点的距离
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        distance_from_start = np.linalg.norm(np.array(leader_pos) - self.start_position)
        depth_info['distance_to_start'] = float(distance_from_start)  # 当前距起点距离
        depth_info['max_traveled_distance'] = float(self.max_traveled_distance)  # 最远移动距离
        depth_info['current_step'] = self.current_step  # 添加当前步数信息
        
        if self.use_leader_camera:
            # 使用屏蔽后的深度图像进行奖励计算，避免自遮挡影响
            depth_image = self._get_masked_leader_depth()
            if depth_image is not None and depth_image.size > 0:
                # 预处理深度图像
                raw_depth = depth_image if len(depth_image.shape) == 2 else depth_image[:, :, 0]
                depth_map = self.state_processor.depth_processor.preprocess_depth_image(raw_depth)
                
                # 保存深度图像用于避障奖励计算
                depth_info['depth_map'] = depth_map
                
                h, w = depth_map.shape
                
                # 计算各区域深度
                regions = {
                    'center': depth_map[h//4:3*h//4, w//4:3*w//4],
                    'forward': depth_map[h//3:2*h//3, w//3:2*w//3],
                    'left': depth_map[h//4:3*h//4, :w//3],
                    'right': depth_map[h//4:3*h//4, 2*w//3:]
                }
                
                # 计算中心区域深度
                valid_depths = regions['center'][regions['center'] > 0.1]
                if len(valid_depths) > 0:
                    depth_info['min_depth'] = float(np.min(valid_depths))
                    depth_info['mean_depth'] = float(np.mean(valid_depths))
                
                # 计算各方向最小深度
                for name, region in regions.items():
                    if name == 'center':
                        continue
                    valid = region[region > 0.1]
                    if len(valid) > 0:
                        depth_info[f"{name}_min"] = float(np.min(valid))
        
        return depth_info
    
    def _check_success(self) -> bool:
        """检查是否成功到达目标 - 平面模式只检查水平距离"""
        if self.goal is None:
            return False
            
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        leader_pos = np.array(leader_pos)
        
        horiz_dist = np.linalg.norm(np.array(self.goal)[:2] - leader_pos[:2])
        
        # 平面模式：只检查水平距离，不检查高度
        if self.enforce_planar:
            return horiz_dist < self.success_radius_xy
        else:
            # 3D模式：检查水平距离和高度差
            height_diff = abs(self.goal[2] - leader_pos[2])
            return horiz_dist < self.success_radius_xy and height_diff < self.success_height_tol

    def _check_collision(self, drone_idx: int = None) -> Dict[str, any]:
        """
        检查碰撞 - 使用PyBullet接触点检测（参考单无人机方法）
        
        Args:
            drone_idx: 检查的无人机索引，None表示检查领航者
            
        Returns:
            Dict包含: collision (bool), contact_points (int), collision_type (str), position (list)
        """
        if drone_idx is None:
            drone_idx = self.leader_index
            
        collision_info = {
            'collision': False,
            'contact_points': 0,
            'collision_type': 'none',
            'position': [0, 0, 0]
        }
        
        # 获取无人机位置
        drone_pos, _ = p.getBasePositionAndOrientation(self.drones[drone_idx].drone, self.client)
        collision_info['position'] = list(drone_pos)
        
        # 1. PyBullet接触点检测（最准确的物理碰撞）
        contact_points = p.getContactPoints(self.drones[drone_idx].drone, physicsClientId=self.client)
        if contact_points:
            # 过滤与目标对象的碰撞 - 接触目标不算碰撞
            valid_contacts = []
            for contact in contact_points:
                contact_object_id = contact[2]  # bodyUniqueIdB
                # 如果接触的是目标对象，不算碰撞
                if hasattr(self, 'goal_id') and self.goal_id is not None and contact_object_id == self.goal_id:
                    continue  # 跳过与目标的接触
                valid_contacts.append(contact)
            
            if valid_contacts:  # 只有非目标的碰撞才算真正的碰撞
                collision_info['collision'] = True
                collision_info['contact_points'] = len(valid_contacts)
                collision_info['collision_type'] = 'physical_contact'
                if config.get('debug_collision', False):
                    print(f"调试: 检测到物理碰撞，接触点数: {len(valid_contacts)}")
                return collision_info
        
        # 2. 边界检测 - 平面模式下不检查高度边界
        if self.enforce_planar:
            # 平面模式：只检查x-y边界
            if abs(drone_pos[0]) > 18.0 or abs(drone_pos[1]) > 18.0:  # 从14.5增加到18.0米
                collision_info['collision'] = True
                collision_info['collision_type'] = 'boundary'
                return collision_info
        else:
            # 3D模式：检查所有边界
            if (abs(drone_pos[0]) > 18.0 or abs(drone_pos[1]) > 18.0 or  # 从14.5增加到18.0米
                drone_pos[2] < 0.3 or drone_pos[2] > 2.4):
                collision_info['collision'] = True
                collision_info['collision_type'] = 'boundary'
                return collision_info
        
        return collision_info

    def step(self, action: np.ndarray):
        """执行一步"""
        action = np.asarray(action, dtype=np.float32)
        
        if self.training_stage == 1:
            # 第一阶段：动作是领航者的2维动作 [thrust, torque] - 前进/后退力和转向扭矩
            if action.shape[0] != 2:
                raise ValueError(f"第一阶段动作维度错误：期望2维动作(领航者)，实际收到 {action.shape[0]}维。请检查动作生成逻辑！")
            
            # 应用领航者动作
            self._apply_drone_action(0, action)
            
            # 跟随者保持静止（重力补偿）
            for i in range(1, self.num_drones):
                self._apply_drone_action(i, np.zeros(2, dtype=np.float32))
                
        else:
            # 第二阶段：解析动作，所有无人机都是3维 [fx, fy, fz] - 世界坐标系直接力控制
            expected_dim = 3 * self.num_drones
            if action.shape[0] != expected_dim:
                raise ValueError(f"第二阶段动作维度错误：期望 {expected_dim}，实际收到 {action.shape[0]}。请检查动作生成逻辑！")
            
            # 应用动作到所有无人机（世界坐标系直接力控制）
            for i in range(self.num_drones):
                drone_action = action[i*3:(i+1)*3]  # [fx, fy, fz]
                
                # 平面模式：强制z轴动作为0，只在x-y平面内移动
                if self.enforce_planar:
                    drone_action = np.array([drone_action[0], drone_action[1], 0.0])
                
                self._apply_drone_action(i, drone_action)

        # 编队辅助力控制（仅跟随者，且仅当无人机数量>1时）
        if self.num_drones > 1:
            self._apply_formation_forces()

        # 物理仿真
        p.stepSimulation(self.client)
        
        self.render()
        
        # 更新最大移动距离
        leader_pos, _ = p.getBasePositionAndOrientation(self.drones[self.leader_index].drone, self.client)
        current_distance_from_start = np.linalg.norm(np.array(leader_pos) - self.start_position)
        self.max_traveled_distance = max(self.max_traveled_distance, current_distance_from_start)
        
        self.current_step += 1

        # 在平面模式下，强制约束无人机姿态和运动
        if self.enforce_planar:
            for i in range(self.num_drones):
                # 获取物理仿真后的速度和姿态
                current_vel, current_ang_vel = p.getBaseVelocity(self.drones[i].drone, self.client)
                _, current_orn = p.getBasePositionAndOrientation(self.drones[i].drone, self.client)
                current_euler = p.getEulerFromQuaternion(current_orn)

                # 【关键修复】不完全重置速度，只修正约束部分
                # 保留推力产生的xy速度，只强制z轴速度为0
                constrained_vel = [current_vel[0], current_vel[1], 0.0]  # 保留xy速度，z=0

                # 强制姿态为水平（只允许yaw旋转）
                constrained_euler = [0.0, 0.0, current_euler[2]]  # [roll=0, pitch=0, yaw保持]
                constrained_orn = p.getQuaternionFromEuler(constrained_euler)

                # 获取当前位置
                current_pos, _ = p.getBasePositionAndOrientation(self.drones[i].drone, self.client)

                # 重置姿态和部分速度
                p.resetBasePositionAndOrientation(self.drones[i].drone, current_pos, constrained_orn, physicsClientId=self.client)

                # 只重置角速度：强制roll和pitch角速度为0，保留yaw角速度
                constrained_ang_vel = [0.0, 0.0, current_ang_vel[2]]  # [roll_rate=0, pitch_rate=0, yaw_rate保持]
                p.resetBaseVelocity(self.drones[i].drone, constrained_vel, constrained_ang_vel, physicsClientId=self.client)

        # 获取观测和奖励
        obs = self._build_observation()
        reward, rinfo, done = self._compute_reward()

        truncated = self.current_step >= self.max_steps
        terminated = done

        return obs, reward, terminated, truncated, {"reward_info": rinfo, "success": self.success}

    def render(self, mode="human"):
        """渲染环境"""
        try:
            # 设置观察相机视角
            if self.enable_fixed_overhead_camera:
                self.camera_manager.setup_fixed_overhead_camera(self.drones[self.leader_index])
            else:
                # 使用配置的相机设置
                camera_config = {
                    'camera_follow': config.get('camera_follow', True),
                    'camera_target': config.get('camera_target', 'leader'),
                    'camera_distance': config.get('camera_distance', 3.0),
                    'camera_yaw': config.get('camera_yaw', 30.0),
                    'camera_pitch': config.get('camera_pitch', -20.0)
                }
                self.camera_manager.update_observer_camera(self.drones, self.leader_index, camera_config)
            
            # 更新侧边栏调试相机
            if self.use_leader_camera and len(self.drones) > self.leader_index:
                # 更新调试相机显示
                self.camera_manager.update_debug_camera_for_sidebar(self.drones[self.leader_index])
                
                # 更新合成相机面板，处理掩码深度
                rgb, depth, seg, self_mask = self._get_leader_images_with_mask()
                
                # 保存图像用于其他功能
                if rgb is not None and depth is not None:
                    self.leader_rgb_image, self.leader_depth_image = rgb, depth
                
                # 更新合成相机面板显示
                self.camera_manager.update_synthetic_camera_panel(self.drones[self.leader_index])
            
            # 渲染目标提示
            if self.goal is not None and config.get('render_goal_hint', True):
                self.camera_manager.render_goal_hint(self.drones, self.goal, self.leader_index)
                
        except Exception as e:
            # 渲染失败时跳过（DIRECT模式下可能不支持）
            pass
        
        return

    def close(self):
        """关闭环境"""
        try:
            p.disconnect(self.client)
        except Exception:
            pass

    def _get_masked_leader_depth(self) -> np.ndarray | None:
        """获取屏蔽了无人机自身的深度图像，用于避障算法"""
        if not self.use_leader_camera:
            return None
        try:
            # 获取图像和掩码
            rgb, depth, seg, self_mask = self._get_leader_images_with_mask()
            
            if depth is None or self_mask is None:
                return depth
                
            # 将自身像素的深度设置为远平面，避免自遮挡
            far_val = float(self.camera_manager.depth_camera_config.get('far_plane', 10.0))
            masked_depth = depth.copy()
            masked_depth[self_mask] = far_val
            return masked_depth
        except Exception as e:
            print(f"获取掩码深度图像失败，使用默认深度: {e}")
            # 返回默认深度
            width = self.camera_manager.depth_camera_config['width']
            height = self.camera_manager.depth_camera_config['height']
            return np.full((height, width), 5.0, dtype=np.float32)
            
    def _get_leader_images_with_mask(self):
        """获取领航者相机图像和掩码信息，集中处理自身掩码逻辑"""
        try:
            if not self.use_leader_camera or self.leader_index >= len(self.drones):
                return None, None, None, None
                
            # 从相机管理器获取图像和掩码
            cam_pos, cam_orn = self.drones[self.leader_index].get_camera_pose()
            rgb, depth, seg = self.camera_manager.get_leader_camera_frame_by_pose(cam_pos, cam_orn)
            
            if seg is None:
                return rgb, depth, None, None
                
            # 创建自身掩码
            leader_body_unique_id = int(self.drones[self.leader_index].drone)
            obj_ids = (seg >> 24).astype(np.int32)
            self_mask = (obj_ids == leader_body_unique_id)
            
            # 屏蔽目标对象（如果有）
            if hasattr(self, 'goal_id') and self.goal_id is not None:
                goal_mask = (obj_ids == self.goal_id)
                self_mask = self_mask | goal_mask
                
            return rgb, depth, seg, self_mask
            
        except Exception as e:
            print(f"获取领航者图像失败: {e}")
            return None, None, None, None
