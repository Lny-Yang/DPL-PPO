"""
🎯 极简版奖励计算模块 - 无冗余设计（2D 固定翼优化版）
"""
import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple, Optional
from .depth_obstacle_processor import DepthObstacleProcessor

class RewardCalculator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化奖励计算器
        
        Args:
            config: 奖励配置参数
        """
        # 稀疏奖励配置
        self.success_base_bonus = config.get('success_base_bonus', 30.0)
        self.bonus_per_step_saved = config.get('bonus_per_step_saved', 0.3)
        self.reference_max_steps = config.get('reference_max_steps', 250)
        self.crash_penalty = config.get('crash_penalty', -12.0)
        self.step_penalty = config.get('step_penalty', -0.01)
        
        # 成功判定阈值（2D 距离）
        self.success_distance_threshold = config.get('success_distance_threshold', 0.2)
        self.collision_distance = config.get('collision_distance', 0.6)
        
        # 状态记录
        self.previous_distances = {}
        
        # 深度处理器
        self.depth_processor = DepthObstacleProcessor(
            depth_image_size=(128, 160),
            collision_threshold=self.collision_distance,
            depth_scale=config.get('depth_scale', 4.0),
            max_depth=config.get('max_depth', 2.0),
            cnn_feature_dim=config.get('cnn_feature_dim', 128)
        )

        # 密集奖励缩放系数
        self.dense_scale = config.get('dense_scale', 0.15)
        
    def compute_total_reward(self,
                           drone_id: str,
                           position: np.ndarray,
                           target_position: np.ndarray,
                           velocity: np.ndarray,
                           depth_info: Dict[str, float],
                           orientation: Optional[np.ndarray] = None,
                           formation_info: Optional[Dict[str, Any]] = None,
                           done: bool = False,
                           success: bool = False,
                           current_step: int = 0) -> Tuple[float, Dict[str, float]]:
        reward_details = {}

        # 在计算奖励前，先获取距离
        distance_to_target = np.linalg.norm(position[:2] - target_position[:2])

        if distance_to_target < 3.0 and not success:
            reward_details['approaching'] = 5.0 * (3.0 - distance_to_target)  # 最多 +5.0

        # 1. 成功奖励 - 效率敏感型
        if success:
            steps_saved = max(0, self.reference_max_steps - current_step)
            reward_details['success'] = self.success_base_bonus + steps_saved * self.bonus_per_step_saved
        else:
            reward_details['success'] = 0.0

        # 2. 碰撞惩罚
        collision_occurred = depth_info.get('collision', False)
        reward_details['crash'] = self.crash_penalty if collision_occurred else 0.0
        
        # 3. 步数惩罚
        reward_details['step_penalty'] = self.step_penalty
        
        # 4. 超时奖励 - 动态：基于最终距离
        is_timeout = done and not success and not collision_occurred
        if is_timeout:
            final_distance = np.linalg.norm(position[:2] - target_position[:2])
            # 距离越近奖励越高，上限不超过 success_base_bonus 的 1/3
            reward_details['timeout'] = max(0.0, (10.0 - final_distance) * 0.5)
        else:
            reward_details['timeout'] = 0.0

        # 5. 导航奖励 - 绕路友好型（2D）
        navigation_reward = self._compute_navigation_reward_merged(
            drone_id, position, target_position, velocity, orientation
        )
        # 6. 安全导航奖励
        safe_nav_reward = self._compute_safe_navigation_reward(
            depth_info, velocity, orientation, 
            np.linalg.norm(position[:2] - target_position[:2])
        )
        reward_details['navigation'] = navigation_reward * self.dense_scale
        reward_details['safe_navigation'] = safe_nav_reward * self.dense_scale

        # 总奖励
        total_reward = sum(reward_details.values())
        return total_reward, reward_details
    
    def _compute_navigation_reward_merged(self, drone_id: str, position: np.ndarray, 
                                         target_position: np.ndarray,
                                         velocity: np.ndarray,
                                         orientation: Optional[np.ndarray]) -> float:
        """🎯 2D 固定翼导航奖励 - 绕路友好 + 平面特化"""
        pos_2d = position[:2]
        target_2d = target_position[:2]
        to_target = target_2d - pos_2d
        distance_2d = np.linalg.norm(to_target)

        # 接近目标：稳定高奖励
        if distance_2d < self.success_distance_threshold:
            return 2.5

        vel_2d = velocity[:2]
        speed_2d = np.linalg.norm(vel_2d)

        if speed_2d < 0.05:
            reward_progress = 0.0
        else:
            target_dir = to_target / (distance_2d + 1e-6)
            vel_along_target = np.dot(vel_2d, target_dir)

            if vel_along_target > 0.1:
                reward_progress = min(vel_along_target * 8.0, 3.0)
            elif vel_along_target < -0.1:
                reward_progress = max(vel_along_target * 4.0, -0.8)
            else:
                reward_progress = 0.0

        reward_alignment = 0.0
        if orientation is not None and speed_2d > 0.1:
            euler = p.getEulerFromQuaternion(orientation)
            yaw = euler[2]
            heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
            alignment = np.dot(heading_vec, target_dir)

            if alignment > 0.7:
                reward_alignment = 0.6 * (alignment - 0.7) / 0.3
            elif alignment < 0.0:
                reward_alignment = -0.3 * abs(alignment)

        return reward_progress + reward_alignment
    
    def _compute_safe_navigation_reward(self, depth_info: Dict[str, float],
                                       velocity: np.ndarray,
                                       orientation: Optional[np.ndarray],
                                       distance_to_target: float) -> float:
        """🎯 融合版安全导航奖励 - 固定翼特化设计（保持不变）"""
        depth_map = depth_info.get('depth_map', None)
        if depth_map is None:
            return 0.0

        obstacle_analysis = self.depth_processor.get_obstacle_analysis(depth_map)
        danger_level = obstacle_analysis['danger_level']
        forward_openness = obstacle_analysis['forward_openness']
        physical_min_depth = obstacle_analysis['physical_min_depth']
        
        left_depth = depth_info.get('left_min', 0.5)
        right_depth = depth_info.get('right_min', 0.5)
        speed_2d = np.linalg.norm(velocity[:2])
        angular_vel = depth_info.get('angular_velocity', 0.0)

        if danger_level < 0.2:
            if speed_2d > 1.5:
                return +2.0
            elif speed_2d > 1.0:
                return +1.2
            else:
                return +0.3
        elif danger_level < 0.4:
            left_right_diff = abs(left_depth - right_depth) * self.depth_processor.depth_scale
            if left_right_diff > 0.5:
                should_turn_left = left_depth > right_depth
                is_turning = (should_turn_left and angular_vel < -0.02) or \
                            (not should_turn_left and angular_vel > 0.02)
                if is_turning:
                    return +1.5 if speed_2d > 1.0 else +1.0
                else:
                    return +0.8 if speed_2d > 1.0 else +0.3
            else:
                return +1.2 if speed_2d > 1.0 else +0.5
        elif danger_level < 0.7:
            left_right_diff = abs(left_depth - right_depth) * self.depth_processor.depth_scale
            if left_right_diff > 0.3:
                should_turn_left = left_depth > right_depth
                is_turning_correctly = (should_turn_left and angular_vel < -0.03) or \
                                      (not should_turn_left and angular_vel > 0.03)
                if is_turning_correctly:
                    if abs(angular_vel) > 0.08:
                        return +1.5
                    elif abs(angular_vel) > 0.04:
                        return +1.0
                    else:
                        return +0.6
                elif abs(angular_vel) > 0.03:
                    return -0.8
                else:
                    return -1.2
            else:
                return +0.8 if abs(angular_vel) > 0.05 else -1.0
        else:
            if abs(angular_vel) > 0.1:
                return +1.2
            elif abs(angular_vel) > 0.05:
                return +0.6
            else:
                return -2.0
        return 0.0
    
    def reset_state(self):
        """重置状态（用于新回合）"""
        self.previous_distances.clear()


def create_default_reward_config() -> Dict[str, Any]:
    return {
        # 稀疏奖励（效率敏感）
        'success_base_bonus': 25.0,
        'bonus_per_step_saved': 0.3,
        'reference_max_steps': 250,
        'crash_penalty': -12.0,
        'timeout_reward': 0.0,  # 实际由动态逻辑覆盖
        'step_penalty': -0.01,
        
        # 成功与避障参数
        'success_distance_threshold': 0.2,
        'collision_distance': 0.6,
        
        # 密集奖励缩放
        'dense_scale': 0.15,
        
        # 深度处理器参数
        'depth_scale': 4.0,
        'max_depth': 2.0,
        'cnn_feature_dim': 128,
    }