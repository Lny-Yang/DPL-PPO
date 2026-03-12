"""
领航者单机导航训练

训练目标:
- 训练领航者无人机进行单机导航
- 学习避障和目标到达能力
- 使用DPL-PPO算法进行强化学习训练
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
from datetime import datetime
from collections import deque
import time
import pybullet as p

# 导入stable-baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import random
import gym
from drone_envs.envs.drone_env_multi import DroneNavigationMulti
from drone_envs.utils.normalization import StateNormalization, RewardScaling

# 导入自定义DPL-PPO和物理引导模块
from agent.PPOagent import PPO
from drone_envs.utils.physics_loss import PhysicsLossCalculator
from drone_envs.utils.differentiable_simulator import DifferentiablePointMassSimulator

# 导入进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️  tqdm 未安装，无法显示进度条: pip install tqdm")
    TQDM_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class PathConfig:
    """路径配置类 - 集中管理所有保存路径 (SB3专用)"""
    
    # 基础目录 - 放在agent目录下
    BASE_DIR = Path(__file__).parent
    AGENT_DIR = BASE_DIR / "agent"
    LOG_SB3_DIR = AGENT_DIR / "log_SB3"
    MODEL_SB3_DIR = AGENT_DIR / "model_SB3"
    
    # 简化引用
    LOG_DIR = LOG_SB3_DIR
    MODEL_DIR = MODEL_SB3_DIR
    
    # 训练进度相关路径
    TRAINING_PROGRESS_PLOT = LOG_DIR / "training_progress.png"
    TRAINING_DATA_JSON = LOG_DIR / "training_data.json"
    TRAJECTORIES_JSON = LOG_DIR / "trajectories.json"
    
    # 最终结果路径
    FINAL_MODEL = MODEL_DIR / "leader_phase1_final"
    FINAL_PROGRESS_PLOT = LOG_DIR / "leader_phase1_final_progress.png"
    FINAL_DATA_JSON = LOG_DIR / "leader_phase1_final_data.json"
    FINAL_TRAJECTORIES_JSON = LOG_DIR / "leader_phase1_final_trajectories.json"
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        cls.AGENT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_SB3_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_SB3_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_episode_model_path(cls, episode_num):
        """获取指定回合的模型保存路径"""
        return cls.MODEL_DIR / f"leader_phase1_episode_{episode_num}"
    
    @classmethod
    def get_timestamped_path(cls, base_name, extension="json"):
        """获取带时间戳的路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls.LOG_DIR / f"{base_name}_{timestamp}.{extension}"


class StateNormalizationWrapper(gym.Wrapper):
    """状态归一化包装器 - 用于gym环境
    
    功能：
    - 动态统计状态的均值和标准差（RunningMeanStd）
    - 将所有观测标准化到均值0、方差1
    - 训练时更新统计量，评估时不更新
    - 支持保存/加载归一化参数
    - 支持Reward Scaling（DPL-PPO Trick 3 & 4）
    """
    
    def __init__(self, env, clip_range: float = 10.0, use_reward_scaling: bool = True, gamma: float = 0.99):
        super().__init__(env)
        # 获取状态维度（140维）
        obs_shape = env.observation_space.shape
        self.state_normalizer = StateNormalization(shape=obs_shape, clip_range=clip_range)
        self.is_training = True  # 训练模式标志
        
        # 🔥 Reward Scaling
        self.use_reward_scaling = use_reward_scaling
        if use_reward_scaling:
            self.reward_scaler = RewardScaling(shape=1, gamma=gamma)
        else:
            self.reward_scaler = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 归一化观测（训练时更新统计量）
        normalized_obs = self.state_normalizer(obs, update=self.is_training)
        
        # 🔥 每个回合结束时重置Reward Scaling的累积折扣奖励
        if self.use_reward_scaling and self.reward_scaler is not None:
            self.reward_scaler.reset()
        
        return normalized_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 归一化观测（训练时更新统计量）
        normalized_obs = self.state_normalizer(obs, update=self.is_training)
        
        # 🔥 Reward Scaling（训练时更新）
        if self.use_reward_scaling and self.reward_scaler is not None:
            scaled_reward = self.reward_scaler(reward, update=self.is_training)
            # 📊 在info中保存原始奖励，用于监控和日志
            info['original_reward'] = reward
            info['scaled_reward'] = scaled_reward
            reward = scaled_reward
        
        return normalized_obs, reward, terminated, truncated, info
    
    def set_training_mode(self, mode: bool):
        """设置训练/评估模式
        
        Args:
            mode: True=训练模式（更新统计量），False=评估模式（不更新）
        """
        self.is_training = mode
    
    def save_normalization_params(self, path):
        """保存归一化参数"""
        self.state_normalizer.save(path)
        print(f"✅ 状态归一化参数已保存: {path}")
        
        # 🔥 保存Reward Scaling参数
        if self.use_reward_scaling and self.reward_scaler is not None:
            reward_scaling_path = str(path).replace('state_norm', 'reward_scaling')
            self.reward_scaler.save(reward_scaling_path)
            print(f"✅ 奖励缩放参数已保存: {reward_scaling_path}")
    
    def load_normalization_params(self, path):
        """加载归一化参数"""
        self.state_normalizer.load(path)
        print(f"✅ 状态归一化参数已加载: {path}")
        
        # 🔥 加载Reward Scaling参数
        if self.use_reward_scaling and self.reward_scaler is not None:
            reward_scaling_path = str(path).replace('state_norm', 'reward_scaling')
            if Path(reward_scaling_path).exists():
                self.reward_scaler.load(reward_scaling_path)
                print(f"✅ 奖励缩放参数已加载: {reward_scaling_path}")
            else:
                print(f"⚠️  未找到奖励缩放参数文件: {reward_scaling_path}")


class RewardTracker:
    """奖励跟踪和可视化类 - 内存优化版
    
    设计说明：
    - 使用deque(maxlen)限制内存占用（仅保留最近数据用于实时监控）
    - 使用额外列表存储稀疏采样数据（用于最终完整曲线绘制）
    - 内存占用：~30KB（实时数据）+ ~500KB（采样数据，每50回合采样1次）
    """
    def __init__(self, window_size=100, enable_plotting=False, save_full_history=False, initial_episode=0):
        # 🔥 实时监控数据：只保留最近window_size*10的数据，节省内存
        self.episode_rewards = deque(maxlen=window_size * 10)  # 最多保留500条
        self.episode_lengths = deque(maxlen=window_size * 10)
        self.moving_avg_rewards = deque(maxlen=window_size * 10)
        self.success_rate = deque(maxlen=window_size * 10)
        self.collision_rate = deque(maxlen=window_size * 10)
        self.moving_avg_collision = deque(maxlen=window_size * 10)
        self.success_flags = deque(maxlen=window_size * 10)
        self.window_size = window_size
        self.enable_plotting = enable_plotting  # 🔥 控制是否绘图
        self.initial_episode = initial_episode  # 🔥 记录起始回合数（用于绘图X轴）
        
        # 🔥 可选：保存完整历史（用于最终分析，采样保存以节省内存）
        self.save_full_history = save_full_history
        if save_full_history:
            # 🎨 采样策略选择（根据需求调整）：
            # - sample_interval=10:  200000回合→20000点（细腻曲线，~2MB内存）
            # - sample_interval=50:  200000回合→4000点（流畅曲线，~500KB内存）✅ 推荐
            # - sample_interval=100: 200000回合→2000点（略粗糙，~250KB内存）
            self.full_episode_rewards = []
            self.full_episode_lengths = []
            self.full_moving_avg_rewards = []
            self.full_success_rate = []
            self.full_collision_rate = []
            self.sample_interval = 50  # 🔥 推荐：50（平衡细腻度和内存）
            self.episode_counter = 0   # 计数器
        
        # 🔥 只在需要绘图时才设置matplotlib
        if enable_plotting:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.max_open_warning'] = 50
        
    def add_episode(self, episode_reward, episode_length, success, collision):
        """添加新回合数据 - 内存优化版（支持可选的完整历史采样）"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.success_flags.append(success)
        self.collision_rate.append(1.0 if collision else 0.0)
        
        # 🔥 使用deque自动限制长度，只计算最近数据的统计
        recent_rewards = list(self.episode_rewards)[-self.window_size:]
        recent_flags = list(self.success_flags)[-self.window_size:]
        recent_collisions = list(self.collision_rate)[-self.window_size:]
        
        current_avg_reward = np.mean(recent_rewards)
        current_success_rate = sum(recent_flags) / len(recent_flags)
        current_collision_rate = np.mean(recent_collisions)
        
        self.moving_avg_rewards.append(current_avg_reward)
        self.success_rate.append(current_success_rate)
        self.moving_avg_collision.append(current_collision_rate)
        
        # 🔥 可选：采样保存完整历史（每50回合保存一次，200000回合只需4000条）
        if self.save_full_history:
            self.episode_counter += 1
            if self.episode_counter % self.sample_interval == 0:
                self.full_episode_rewards.append(episode_reward)
                self.full_episode_lengths.append(episode_length)
                self.full_moving_avg_rewards.append(current_avg_reward)
                self.full_success_rate.append(current_success_rate)
                self.full_collision_rate.append(current_collision_rate)
        
    def plot_training_progress(self, save_path=None):
        """绘制训练进度图 - 仅在enable_plotting=True时执行
        
        优先使用完整历史采样数据（如果有），否则使用最近500条deque数据
        """
        if not self.enable_plotting:
            return None  # 🔥 禁用绘图时直接返回
        
        if save_path is None:
            save_path = PathConfig.TRAINING_PROGRESS_PLOT
        
        try:
            # 关闭之前的图表，避免内存泄漏
            plt.close('all')
            
            # 🔥 智能选择数据源：优先使用完整历史采样数据
            if self.save_full_history and len(self.full_episode_rewards) > 0:
                # 使用采样的完整历史数据（每50回合1个点）
                plot_rewards = self.full_episode_rewards
                plot_lengths = self.full_episode_lengths
                plot_avg_rewards = self.full_moving_avg_rewards
                plot_success_rate = self.full_success_rate
                plot_collision_rate = self.full_collision_rate
                # X轴：实际的回合数（考虑采样间隔 + 起始回合数）
                episodes = range(self.initial_episode + self.sample_interval, 
                               self.initial_episode + len(plot_rewards) * self.sample_interval + 1, 
                               self.sample_interval)
                data_source = f"采样数据（每{self.sample_interval}回合，共{len(plot_rewards)}点）"
            else:
                # 使用deque的最近数据（最多500条）
                plot_rewards = list(self.episode_rewards)
                plot_lengths = list(self.episode_lengths)
                plot_avg_rewards = list(self.moving_avg_rewards)
                plot_success_rate = list(self.success_rate)
                plot_collision_rate = list(self.moving_avg_collision)
                # 🔥 从initial_episode开始计数，考虑deque可能不足500条的情况
                current_episode_count = self.episode_counter if self.save_full_history else len(plot_rewards)
                start_episode = max(self.initial_episode, current_episode_count - len(plot_rewards))
                episodes = range(start_episode + 1, start_episode + len(plot_rewards) + 1)
                data_source = f"最近{len(plot_rewards)}回合"
            
            # 🔥 降低DPI和图像尺寸，减少内存占用
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. 奖励曲线
            if self.save_full_history and len(self.full_episode_rewards) > 0:
                # 采样数据：只画移动平均（采样点本身就是平均值）
                ax1.plot(episodes, plot_avg_rewards, color='red', linewidth=2, 
                        label=f'{self.window_size}回合移动平均')
            else:
                # deque数据：画原始+移动平均
                ax1.plot(episodes, plot_rewards, alpha=0.3, color='blue', label='原始奖励')
                ax1.plot(episodes, plot_avg_rewards, color='red', linewidth=2, 
                        label=f'{self.window_size}回合移动平均')
            ax1.set_xlabel('回合数')
            ax1.set_ylabel('奖励')
            ax1.set_title(f'训练奖励曲线 ({data_source})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 回合长度
            ax2.plot(episodes, plot_lengths, color='green', alpha=0.7)
            ax2.set_xlabel('回合数')
            ax2.set_ylabel('回合长度')
            ax2.set_title('回合长度变化')
            ax2.grid(True, alpha=0.3)
            
            # 3. 碰撞率
            ax3.plot(episodes, plot_collision_rate, color='orange', linewidth=2)
            ax3.set_xlabel('回合数')
            ax3.set_ylabel('碰撞率')
            ax3.set_title('碰撞率')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # 4. 成功率
            ax4.plot(episodes, plot_success_rate, color='purple', alpha=0.7)
            ax4.set_xlabel('回合数')
            ax4.set_ylabel('成功率')
            ax4.set_title('成功率')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            # 🔥 降低DPI，减少文件大小和内存占用
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 训练进度图已保存: {save_path} ({data_source})")
            return fig
        except Exception as e:
            print(f"⚠️ 绘图失败（已跳过）: {e}")
            return None
    
    def save_data(self, save_path=None, minimal=True):
        """保存训练数据 - 内存优化版
        
        说明：
        - minimal=True: 只保存最终统计（6个字段，<1KB）
        - minimal=False且save_full_history=True: 保存采样历史（每50回合1条，4000条记录）
        - minimal=False且save_full_history=False: 保存最近500条完整记录
        """
        if save_path is None:
            save_path = PathConfig.TRAINING_DATA_JSON
        
        try:
            if minimal:
                # 🔥 最小化保存：只保存关键统计信息，不保存完整历史
                data = {
                    'total_episodes': len(self.episode_rewards) if not self.save_full_history else self.episode_counter,
                    'final_avg_reward': float(list(self.moving_avg_rewards)[-1]) if self.moving_avg_rewards else 0,
                    'final_success_rate': float(list(self.success_rate)[-1]) if self.success_rate else 0,
                    'final_collision_rate': float(list(self.moving_avg_collision)[-1]) if self.moving_avg_collision else 0,
                    'algorithm': 'SB3 DPL-PPO with SDE',
                    'timestamp': str(datetime.now())
                }
            else:
                # 完整保存：根据save_full_history决定保存哪些数据
                if self.save_full_history:
                    # 保存采样数据（每50回合1条，内存友好）
                    data = {
                        'episode_rewards': [float(x) for x in self.full_episode_rewards],
                        'episode_lengths': [int(x) for x in self.full_episode_lengths],
                        'moving_avg_rewards': [float(x) for x in self.full_moving_avg_rewards],
                        'success_rate': [float(x) for x in self.full_success_rate],
                        'collision_rate': [float(x) for x in self.full_collision_rate],
                        'total_episodes': self.episode_counter,
                        'sample_interval': self.sample_interval,  # 记录采样间隔
                        'final_avg_reward': float(self.full_moving_avg_rewards[-1]) if self.full_moving_avg_rewards else 0,
                        'final_success_rate': float(self.full_success_rate[-1]) if self.full_success_rate else 0,
                        'algorithm': 'SB3 DPL-PPO with SDE (sampled history)',
                        'timestamp': str(datetime.now())
                    }
                else:
                    # 保存最近500条完整记录（deque数据）
                    data = {
                        'episode_rewards': [float(x) for x in self.episode_rewards],
                        'episode_lengths': [int(x) for x in self.episode_lengths],
                        'moving_avg_rewards': [float(x) for x in self.moving_avg_rewards],
                        'success_rate': [float(x) for x in self.success_rate],
                        'success_flags': [bool(x) for x in self.success_flags],
                        'collision_rate': [float(x) for x in self.collision_rate],
                        'moving_avg_collision': [float(x) for x in self.moving_avg_collision],
                        'total_episodes': len(self.episode_rewards),
                        'note': 'Only last 500 episodes (deque maxlen)',
                        'final_avg_reward': float(list(self.moving_avg_rewards)[-1]) if self.moving_avg_rewards else 0,
                        'final_success_rate': float(list(self.success_rate)[-1]) if self.success_rate else 0,
                        'algorithm': 'SB3 DPL-PPO with SDE',
                        'timestamp': str(datetime.now())
                    }
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 数据保存失败（已跳过）: {e}")


class TrainingCallback(BaseCallback):
            
    """自定义回调函数 - 用于跟踪训练进度和定期保存"""
    
    def __init__(self, reward_tracker, max_episodes, plot_interval=500, save_interval=500, verbose=1, initial_episode=0):
        super(TrainingCallback, self).__init__(verbose)
        self.reward_tracker = reward_tracker
        self.max_episodes = max_episodes  # 🔥 添加最大回合数限制
        self.plot_interval = plot_interval
        self.save_interval = save_interval
        self.episode_count = initial_episode  # 🔥 支持从指定回合数开始
        self.initial_episode = initial_episode  # 🔥 记录起始回合数
        self.total_target_episodes = initial_episode + max_episodes  # 🔥 总目标回合数
        self.episode_reward = 0
        self.episode_length = 0
        self.start_time = time.time()
        
        # 初始化进度条
        if TQDM_AVAILABLE:
            # 🔥 优化进度条格式：显示已训练时间、剩余时间、速度等
            self.pbar = tqdm(
                total=self.total_target_episodes,  # 🔥 总目标：起始+新增
                initial=initial_episode,  # 🔥 起始位置 
                desc="🚀训练中", 
                unit="回合",
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️已训练:{elapsed} ⏳剩余:{remaining} 🔥{rate_fmt}] {postfix}',
                ncols=150  # 设置进度条宽度，避免换行
            )
        else:
            self.pbar = None
        
        
    def _on_step(self) -> bool:
        """每步调用"""
        try:
            # 🔥 检测NaN值（兼容numpy和tensor类型）
            reward = self.locals['rewards'][0]
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            if np.isnan(reward) or np.isinf(reward):
                print(f"⚠️  警告: 在步骤 {self.num_timesteps} 检测到异常奖励值!")
                return False
            
            # 🔥🔥 检测观测值中的NaN/Inf
            obs = self.locals.get('new_obs', None)
            if obs is not None:
                if isinstance(obs, np.ndarray):
                    if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                        print(f"⚠️  警告: 在步骤 {self.num_timesteps} 检测到异常观测值!")
                        print(f"     NaN数量: {np.sum(np.isnan(obs))}, Inf数量: {np.sum(np.isinf(obs))}")
                        return False
            
            # 累积奖励
            self.episode_reward += self.locals['rewards'][0]
            self.episode_length += 1
            
            # 检查是否回合结束
            if self.locals['dones'][0]:
                self.episode_count += 1
                # 🔥 使用total_target_episodes作为总回合数（包含initial_episode）
                total_episodes = self.total_target_episodes
                # 获取info信息
                info = self.locals['infos'][0]
                success = info.get('success', False)
                # 从reward_info中判断是否碰撞
                reward_info = info.get('reward_info', {})
                crash_reward = reward_info.get('crash', 0)
                collision = crash_reward < 0  # 如果有碰撞惩罚，说明发生了碰撞
                # 记录到tracker
                self.reward_tracker.add_episode(
                    self.episode_reward,
                    self.episode_length,
                    success,
                    collision
                )
                # 直接使用RewardTracker已计算好的滑动平均/窗口统计
                current_avg_reward = self.reward_tracker.moving_avg_rewards[-1] if self.reward_tracker.moving_avg_rewards else self.episode_reward
                current_success_rate = self.reward_tracker.success_rate[-1] if self.reward_tracker.success_rate else 0
                current_collision_rate = self.reward_tracker.moving_avg_collision[-1] if self.reward_tracker.moving_avg_collision else (1.0 if collision else 0.0)
                # 计算ETA
                elapsed_time = time.time() - self.start_time

                # 🔥 优化：减少TensorBoard写入频率（每10回合写入一次）
                if self.episode_count % 10 == 0:
                    try:
                        if hasattr(self.model, 'logger') and self.model.logger:
                            self.model.logger.record('custom/episode_reward', self.episode_reward)
                            self.model.logger.record('custom/episode_length', self.episode_length)
                            self.model.logger.record('custom/success_rate', current_success_rate)
                            self.model.logger.record('custom/avg_collision_rate', current_collision_rate)
                            self.model.logger.record('custom/avg_reward', current_avg_reward)
                            self.model.logger.dump(step=self.num_timesteps)
                    except Exception as e:
                        pass  # 🔥 静默失败，不打印错误信息

                eta = (elapsed_time / self.episode_count) * (total_episodes - self.episode_count) if self.episode_count > 0 else 0
                # 获取终止类型
                if success:
                    termination_type = "成功"
                elif collision:
                    # 尝试从reward_info获取更详细的碰撞类型
                    contact_points = reward_info.get('contact_points', 0)
                    if contact_points > 0:
                        termination_type = f"物理碰撞({contact_points}点)"
                    else:
                        termination_type = "碰撞"
                else:
                    termination_type = "超时"
                
                # 更新进度条（显示关键指标）
                if self.pbar:
                    self.pbar.update(1)
                    # 🔥 计算已训练时长（格式化为时分秒）
                    elapsed_hours = int(elapsed_time // 3600)
                    elapsed_mins = int((elapsed_time % 3600) // 60)
                    elapsed_secs = int(elapsed_time % 60)
                    
                    self.pbar.set_postfix({
                        '当前奖励': f"{self.episode_reward:6.1f}",
                        '平均奖励': f"{current_avg_reward:6.1f}",
                        '成功率': f"{current_success_rate:5.1%}",
                        '碰撞率': f"{current_collision_rate:5.1%}",
                        '已训练': f"{elapsed_hours}h{elapsed_mins}m"
                    })
                
                # 每100回合打印一次详细进度信息（包含训练时长）
                if self.episode_count % 100 == 0:
                    # 格式化已训练时长
                    elapsed_hours = int(elapsed_time // 3600)
                    elapsed_mins = int((elapsed_time % 3600) // 60)
                    
                    print(f"\n📊 回合 {self.episode_count:5d}/{total_episodes} | "
                          f"奖励:{self.episode_reward:7.1f} | "
                          f"平均:{current_avg_reward:6.1f} | "
                          f"成功率:{current_success_rate:5.1%} | "
                          f"碰撞率:{current_collision_rate:5.1%} | "
                          f"⏱️已训练:{elapsed_hours}h{elapsed_mins}m | "
                          f"⏳预计剩余:{eta/60:.0f}分钟")
                
                
                # 🔥 优化：减少I/O频率
                # 只在关键节点保存模型（同时保存JSON和图表）
                if self.episode_count % self.save_interval == 0:
                    model_path = PathConfig.get_episode_model_path(self.episode_count)
                    self.model.save(model_path)
                    
                    # 🔥 保存归一化参数（带回合数版本 + 最新版本）
                    env = self.model.get_env()
                    if env is not None and hasattr(env, 'envs') and env.envs:
                        base_env = env.envs[0]
                        if hasattr(base_env, 'save_normalization_params'):
                            # 带回合数的版本（永久保存）
                            norm_params_versioned = PathConfig.MODEL_DIR / f"state_norm_params_ep{self.episode_count}.npz"
                            base_env.save_normalization_params(str(norm_params_versioned))
                            # 最新版本（每次覆盖，测试时默认使用这个）
                            norm_params_latest = PathConfig.MODEL_DIR / "state_norm_params.npz"
                            base_env.save_normalization_params(str(norm_params_latest))
                    
                    # 💾 同时保存当前进度图和数据（支持中断恢复）
                    # 保存到同一个training_data.json文件中
                    self.reward_tracker.save_data(PathConfig.TRAINING_DATA_JSON, minimal=False)
                    self.reward_tracker.plot_training_progress()
                    print(f"💾 检查点 {self.episode_count}: 模型、归一化参数、完整数据、进度图已保存")
                    
                # 🔥 不再需要额外的绘图频率控制（已在上面统一处理）
                
                # 重置回合统计
                self.episode_reward = 0
                self.episode_length = 0
                
                # 🔥 检查是否达到总目标回合数，如果是则停止训练
                if self.episode_count >= self.total_target_episodes:
                    print("="*80)
                    print(f"✅ 已完成目标训练（从{self.initial_episode} 到{self.total_target_episodes}，共 {self.max_episodes} 回合），停止训练")
                    print("="*80)
                    return False  # 返回False停止训练
            
            return True
        except Exception as e:
            print(f"_on_step 异常: {e}")
            import traceback
            traceback.print_exc()
            return False

def linear_schedule(initial_value: float, final_value: float = 1e-5):
    """线性衰减学习率调度器（DPL-PPO Trick 6）
    
    学习率从 initial_value 线性衰减到 0
    
    Args:
        initial_value: 初始学习率（例如 3e-4）
    
    Returns:
        一个函数，接受 progress_remaining (1.0 -> 0.0) 并返回当前学习率
    
    原理：
        progress_remaining = 1 - (当前步数 / 总步数)
        learning_rate = initial_value * progress_remaining
        
    效果：
        训练初期：lr = 3e-4 （充分探索和学习）
        训练中期：lr = 1.5e-4（稳定学习）
        训练后期：lr → 0    （微调，避免震荡）
    """
    def schedule(progress_remaining: float) -> float:
        """
        Args:
            progress_remaining: 剩余训练进度 (1.0 在开始时, 0.0 在结束时)
        
        Returns:
            当前学习率
        """
        return final_value + (initial_value - final_value) * progress_remaining
    
    return schedule


def make_env(max_steps=400, use_state_norm=True, use_reward_scaling=True, gamma=0.99):
    """创建环境的工厂函数
    
    Args:
        max_steps: 每个回合的最大步数 (默认400)
        use_state_norm: 是否启用状态归一化 (推荐True)
        use_reward_scaling: 是否启用奖励缩放 (推荐True, DPL-PPO Trick 3 & 4)
        gamma: 折扣因子，与DPL-PPO算法保持一致
    """

    def _init():
        env = DroneNavigationMulti(
            num_drones=1,
            use_depth_camera=True,
            depth_camera_range=10.0,
            depth_resolution=16,
            enable_formation_force=False,
            training_stage=1,
            max_steps=max_steps
        )
        
        # 🔥 包装状态归一化层 + Reward Scaling
        if use_state_norm or use_reward_scaling:
            env = StateNormalizationWrapper(
                env, 
                clip_range=10.0,
                use_reward_scaling=use_reward_scaling,
                gamma=gamma
            )
        
        return env
    return _init


def train_leader_phase1_sb3(max_episodes=20000, total_timesteps=None, plot_interval=500, 
                           continue_training=False, load_model_path=None, use_state_norm=True,
                           use_reward_scaling=True, use_physics_guidance=False, lambda_phys=0.1):
    """领航者单机导航训练（使用DPL-PPO算法）
    
    Args:
        max_episodes: 最大训练回合数 (默认20000)
        total_timesteps: 总训练步数（如果为None，则根据max_episodes估算）
        plot_interval: 绘图和保存间隔
        continue_training: 是否继续训练（加载之前的模型）
        load_model_path: 要加载的模型路径（如果为None且continue_training=True，则加载最新模型）
        use_state_norm: 是否启用状态归一化 (推荐True，DPL-PPO Trick 2)
        use_reward_scaling: 是否启用奖励缩放 (推荐True，DPL-PPO Trick 3 & 4)
        use_physics_guidance: 🆕 是否启用物理引导学习 (推荐True)
        lambda_phys: 🆕 物理损失权重系数 (默认0.1)
    """
    print("="*80)
    if continue_training:
        print("领航者导航训练（继续训练）")
    else:
        print("领航者导航训练（从头开始）")
    print("="*80)
    
    # 确保所有目录存在
    PathConfig.ensure_directories()
    
    # 创建单环境（启用状态归一化 + Reward Scaling）
    print("正在创建环境...")
    env = DummyVecEnv([make_env(
        max_steps=400, 
        use_state_norm=use_state_norm,
        use_reward_scaling=use_reward_scaling,
        gamma=0.99  # 与DPL-PPO的gamma保持一致
    )])  # 🔥 启用状态归一化 + Reward Scaling
    # 获取环境配置信息
    test_env = env.envs[0]
    print(f"环境配置:")
    print(f"  - 无人机数量: {test_env.num_drones}")
    print(f"  - 观测空间: {test_env.observation_space.shape}")
    print(f"  - 动作空间: {test_env.action_space.shape}")
    print(f"  - 深度特征维度: {test_env.depth_feature_dim}")
    print(f"  - 训练阶段: {test_env.training_stage}")
    print(f"  - 编队力状态: {'禁用' if not test_env.enable_formation_force else '启用'}")
    print(f"  - 平面模式: {'启用' if test_env.enforce_planar else '禁用'}")
    print(f"  - 最大步数: {test_env.max_steps}")
    print(f"  - 状态归一化: {'启用 ✅' if use_state_norm else '禁用'}")
    print(f"  - 奖励缩放: {'启用 ✅' if use_reward_scaling else '禁用'} (DPL-PPO Trick 3 & 4)")
    print(f"  - 物理引导: {'启用 ✅' if use_physics_guidance else '禁用'}")
    if use_physics_guidance:
        print(f"    └─ λ_phys: {lambda_phys}")
    # 设置环境的总回合数（用于ETA计算）
    test_env.max_episodes = max_episodes
    
    # 估算总步数（如果未指定）
    if total_timesteps is None:
        # 🔥 使用更保守的估计，确保不会因为步数限制而过早停止
        # 假设平均每回合150步（500步上限，考虑提前终止）
        avg_steps_per_episode = 150
        total_timesteps = max_episodes * avg_steps_per_episode
        print(f"  - 估算总步数: {total_timesteps:,} ({max_episodes}回合 × {avg_steps_per_episode}步)")
        print(f"  ⚠️  注意: 实际训练将在达到 {max_episodes} 回合时停止（由回调函数控制）")
    
    print("="*80)
    
    # 🔥 检测并强制使用GPU
    if torch.cuda.is_available():
        device_name = 'cuda'
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device_name = 'cpu'
        print("⚠️  GPU不可用，使用CPU训练（速度较慢）")
    
    # 🔄 检查是否继续训练
    if continue_training:
        # 确定要加载的模型路径
        if load_model_path is None:
            # 自动查找最新的模型
            if PathConfig.FINAL_MODEL.with_suffix('.zip').exists():
                load_model_path = PathConfig.FINAL_MODEL
                print(f"🔄 找到最终模型，加载: {load_model_path}")
            else:
                # 查找最新的episode模型
                model_files = list(PathConfig.MODEL_DIR.glob("leader_phase1_episode_*.zip"))
                if model_files:
                    # 按episode数排序，取最大的
                    model_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    load_model_path = model_files[-1].with_suffix('')
                    print(f"� 找到最新episode模型，加载: {load_model_path}")
                else:
                    print("⚠️  未找到任何已保存模型，将从头开始训练")
                    continue_training = False
        else:
            print(f"🔄 加载指定模型: {load_model_path}")
            # 如果是字符串，转换为完整的路径
            if isinstance(load_model_path, str):
                load_model_path = PathConfig.get_episode_model_path(int(load_model_path.split('_')[-1]))
        
        if continue_training:
            try:
                print("正在加载模型...")
                model = PPO.load(load_model_path, env=env, device=device_name)
                print("✅ 模型加载成功！")
                
                # 尝试加载训练历史数据
                if PathConfig.TRAINING_DATA_JSON.exists():
                    print("\n正在加载训练历史...")
                    with open(PathConfig.TRAINING_DATA_JSON, 'r', encoding='utf-8') as f:
                        history_data = json.load(f)
                    print(f"✅ 已加载 {history_data['total_episodes']} 回合的历史数据")
                else:
                    print("⚠️  未找到训练历史数据，将重新统计")
                    history_data = None
                
                # 🔥 加载归一化参数（如果启用了状态归一化）
                if use_state_norm:
                    norm_params_path = PathConfig.MODEL_DIR / "state_norm_params.npz"
                    if norm_params_path.exists():
                        print("\n正在加载状态归一化参数...")
                        if hasattr(env.envs[0], 'load_normalization_params'):
                            env.envs[0].load_normalization_params(str(norm_params_path))
                        else:
                            print("⚠️  环境未启用归一化包装器，跳过归一化参数加载")
                    else:
                        print("⚠️  未找到归一化参数文件，将从头统计")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("将从头开始训练")
                continue_training = False
                history_data = None
    
    # 🆕 创建或已加载模型
    if not continue_training:
        print("创建新的DPL-PPO模型...")
        
        # 🆕 创建物理模块（如果启用）
        physics_loss_fn = None
        physics_simulator = None
        
        if use_physics_guidance:
            print("\n🔬 初始化物理引导模块...")
            
            # 🔥 从YAML配置文件加载物理参数
            config_path = PathConfig.BASE_DIR / "drone_envs" / "physics_config.yaml"
            if config_path.exists():
                print(f"  📄 加载配置文件: {config_path}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                physics_config = full_config['physics']
                print(f"  ✅ 配置已从YAML文件加载")
                
                # 🔥 从配置中提取lambda_phys参数(优先使用配置文件)
                lambda_phys = physics_config.get('lambda_phys_init', 1.0)
                lambda_phys_min = physics_config.get('lambda_phys_min', 0.1)
                lambda_phys_max = physics_config.get('lambda_phys_max', 0.25)
                print(f"  📌 使用配置文件中的 lambda_phys: {lambda_phys}")
                print(f"  📌 课程学习范围: {lambda_phys_min} → {lambda_phys_max}")
                
                # 🆕 读取拉格朗日约束配置
                lag_cfg = physics_config.get('lagrangian', {})
                use_lagrangian = lag_cfg.get('enabled', False)
                lagrangian_lr = lag_cfg.get('lr', 1e-3)
                lagrangian_clip = lag_cfg.get('clip', 10.0)
                lagrangian_warmup = lag_cfg.get('warmup_updates', 0)
                
                c_cfg = physics_config.get('constraints', {})
                c_obstacle_target = c_cfg.get('obstacle_violation_rate_target', 0.0)
                c_feasibility_target = c_cfg.get('feasibility_violation_rate_target', 0.0)
                c_energy_target = c_cfg.get('energy_raw_target', None)
                
                cn_cfg = physics_config.get('constraint_normalization', {})
                cn_momentum = cn_cfg.get('momentum', 0.995)
                cn_warmup = cn_cfg.get('warmup_updates', 1000)
                cn_std_floor = cn_cfg.get('std_floor', 1e-3)
                cn_clip = cn_cfg.get('clip', 2.0)
                
                if use_lagrangian:
                    print(f"  🆕 拉格朗日约束: 启用 ✅")
                    print(f"     - 对偶lr: {lagrangian_lr}, clip: {lagrangian_clip}, warmup: {lagrangian_warmup}")
                    print(f"     - 目标: obstacle={c_obstacle_target}, feasibility={c_feasibility_target}, energy={c_energy_target}")
                else:
                    print(f"  🆕 拉格朗日约束: 禁用")
            else:
                print(f"  ⚠️  未找到配置文件，使用默认参数")
                # 默认物理参数配置
                physics_config = {
                    'dt': 1/30,
                    'horizon': 5,
                    'safe_distance': 1.0,
                    'max_acceleration': 5.0,
                    'drone_radius': 0.2,
                    'weights': {
                        'velocity': 1.0,
                        'obstacle': 2.0,
                        'smooth': 0.1,
                        'energy': 0.01,
                        'feasibility': 0.5
                    },
                    'decay_alpha': 0.5,
                    'lambda_phys_init': 1.0
                }
                lambda_phys = 1.0
                lambda_phys_min = 0.1
                lambda_phys_max = 0.25
                print(f"  📌 使用默认 lambda_phys: {lambda_phys}")
                print(f"  📌 课程学习范围: {lambda_phys_min} → {lambda_phys_max}")
                
                # 🆕 默认拉格朗日配置（禁用）
                use_lagrangian = False
                lagrangian_lr = 1e-3
                lagrangian_clip = 10.0
                lagrangian_warmup = 0
                c_obstacle_target = 0.0
                c_feasibility_target = 0.0
                c_energy_target = None
                cn_momentum = 0.995
                cn_warmup = 1000
                cn_std_floor = 1e-3
                cn_clip = 2.0
                print(f"  🆕 拉格朗日约束: 禁用（使用默认配置）")
            
            # 创建物理损失计算器
            physics_loss_fn = PhysicsLossCalculator(physics_config)
            print(f"  ✅ 物理损失计算器已创建")
            print(f"     - Horizon: {physics_config['horizon']}步")
            print(f"     - 安全距离: {physics_config['safe_distance']}m")
            print(f"     - 🔥 lambda_phys: {lambda_phys} (总物理权重)")
            
            # 🔥 计算预期的加权贡献(基于实际数据校准)
            print(f"     - 📊 预期加权贡献:")
            print(f"       velocity:    ~{10 * physics_config['weights']['velocity']:.2f} (实际raw≈10)")
            print(f"       obstacle:    ~{0.2 * physics_config['weights']['obstacle']:.2f} (假设raw≈0.2)")
            print(f"       energy:      ~{0.01 * physics_config['weights']['energy']:.4f}")
            print(f"       feasibility: ~{1.0 * physics_config['weights']['feasibility']:.2f} (假设raw≈1.0)")
            print(f"       total×λ:     ~{(10*physics_config['weights']['velocity'] + 0.2*physics_config['weights']['obstacle'] + 0.01*physics_config['weights']['energy'] + 1.0*physics_config['weights']['feasibility'])*lambda_phys:.2f}")

            
            # 创建可微分物理模拟器
            physics_simulator = DifferentiablePointMassSimulator(
                dt=1/30, 
                enforce_planar=True,
                air_resistance=0.0  # 暂不考虑空气阻力
            )
            print(f"  ✅ 可微分物理模拟器已创建")
            print(f"     - 模型类型: 点质量模型（Point Mass）")
            print(f"     - 积分方法: 欧拉法（Euler）")
            print(f"     - 平面模式: 启用")
            print()
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=linear_schedule(3e-4, 1e-5),  # 🔥 线性衰减：3e-4 → 1e-5 (DPL-PPO Trick 6)
            n_steps=2048,              # 🔥 增大到2048（从1024），收集更多经验再更新
            batch_size=64,            # 🔥 从64提升到128（稳定物理梯度，配合物理引导）
            n_epochs=10,               # 每次更新的训练轮数
            gamma=0.99,                # 折扣因子
            gae_lambda=0.95,           # GAE参数
            clip_range=0.15,            # DPL-PPO裁剪范围
            clip_range_vf=0.2,         # Value function裁剪，稳定价值估计
            ent_coef=0.001,             # 🔥 从0.001提升到0.01（增强探索，防止过早收敛）
            vf_coef=0.8,               # Value function损失系数
            max_grad_norm=0.5,         # 🔥 梯度裁剪放宽（0.5→1.0），加快学习速度
            use_sde=True,              # ✅✅ 启用状态依赖探索（自适应噪声）
            sde_sample_freq=4,         # SDE采样频率（每4步重新采样噪声）
            target_kl=0.05,            # 不使用KL散度early stopping，依赖clip机制
            tensorboard_log=str(PathConfig.LOG_DIR / "tensorboard"),  # ✅ 开启TensorBoard监控
            policy_kwargs=dict(
                # 🎯 网络结构设计（针对140维观测空间 → 2维动作）:
                # 输入140维 → Actor[256→128] → 动作2维
                # 输入140维 → Critic[256→128] → 价值1维
                net_arch=[dict(pi=[256, 128], vf=[256, 128])],
                activation_fn=torch.nn.Tanh,
                ortho_init=True,
                # 🔥 gSDE 专用参数（仅在 use_sde=True 时生效）
                log_std_init=-2.0,      # 初始对数标准差，控制初始探索强度
                full_std=True,          # 使用完整协方差矩阵，提供更丰富的探索模式
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5),  # ✅ 保留：提高数值稳定性（与SDE无关）
            ),
            verbose=1,
            seed=SEED,                 # 🔥 设置随机种子，保证可复现
            device=device_name,
            # 🆕 物理引导参数
            use_physics_guidance=use_physics_guidance,
            physics_loss_fn=physics_loss_fn,
            physics_simulator=physics_simulator,
            lambda_phys=lambda_phys,
            lambda_phys_min=lambda_phys_min,
            lambda_phys_max=lambda_phys_max,
            physics_update_freq=1,  # 每次更新都使用物理引导
            # 🆕 拉格朗日约束参数
            use_lagrangian_constraints=use_lagrangian if use_physics_guidance else False,
            lagrangian_lr=lagrangian_lr if use_physics_guidance else 1e-3,
            lagrangian_clip=lagrangian_clip if use_physics_guidance else 10.0,
            lagrangian_warmup_updates=lagrangian_warmup if use_physics_guidance else 0,
            constraint_obstacle_violation_rate_target=c_obstacle_target if use_physics_guidance else 0.0,
            constraint_feasibility_violation_rate_target=c_feasibility_target if use_physics_guidance else 0.0,
            constraint_energy_raw_target=c_energy_target if use_physics_guidance else None,
            constraint_norm_momentum=cn_momentum if use_physics_guidance else 0.995,
            constraint_norm_warmup_updates=cn_warmup if use_physics_guidance else 1000,
            constraint_norm_std_floor=cn_std_floor if use_physics_guidance else 1e-3,
            constraint_norm_clip=cn_clip if use_physics_guidance else 2.0,
        )
    
    print("模型配置:")
    
    # 🆕 显示物理引导配置
    if use_physics_guidance:
        print("🔬 物理引导配置:")
        print(f"  - 状态: 启用 ✅")
        print(f"  - λ_phys: {lambda_phys}")
        print(f"  - Horizon: {physics_config['horizon']}步")
        print(f"  - 安全距离: {physics_config['safe_distance']}m")
        print()
    
    # 🔥 处理schedule函数显示
    lr_val = model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate
    clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
    clip_vf_val = model.clip_range_vf(1.0) if callable(model.clip_range_vf) else model.clip_range_vf
    
    # 🔥 显示学习率衰减信息
    if callable(model.learning_rate):
        lr_initial = model.learning_rate(1.0)  # 训练开始时
        lr_final = model.learning_rate(0.0)     # 训练结束时
        print(f"  - 学习率: {lr_initial:.2e} → {lr_final:.2e} (线性衰减 ✅)")
    else:
        print(f"  - 学习率: {lr_val} (固定)")
    print(f"  - Batch大小: {model.batch_size}")
    print(f"  - 训练轮数: {model.n_epochs}")
    print(f"  - N_steps: {model.n_steps}")
    print(f"  - Gamma: {model.gamma}")
    print(f"  - GAE Lambda: {model.gae_lambda}")
    print(f"  - Clip范围: {clip_val} (DPL-PPO核心机制)")
    print(f"  - Clip范围(VF): {clip_vf_val} (稳定价值估计)")
    print(f"  - 熵系数: {model.ent_coef}")
    print(f"  - VF系数: {model.vf_coef}")
    print(f"  - 梯度裁剪: {model.max_grad_norm} (防止梯度爆炸)")
    print(f"  - Target KL: {model.target_kl} (标准DPL-PPO)")
    print(f"  - SDE状态: {'启用' if model.use_sde else '禁用'}")
    if model.use_sde:
        print(f"  - SDE采样频率: {model.sde_sample_freq}")
    
    # 🆕 物理引导信息
    if use_physics_guidance:
        print(f"  - 物理引导: 启用 ✅")
        print(f"  - λ_phys: {model.lambda_phys} (物理损失权重)")
        print(f"  - 更新频率: 每{model.physics_update_freq}步")
        print(f"  - 预测horizon: {physics_config['horizon']}步")
    
    # 🔥 只在非SDE模式下显示log_std
    if not model.use_sde and hasattr(model.policy, 'log_std'):
        current_log_std = model.policy.log_std.data.cpu().numpy()
        print(f"  - Log_std: 均值={current_log_std.mean():.4f}, "
              f"范围=[{current_log_std.min():.4f}, {current_log_std.max():.4f}], "
              f"对应std≈{np.exp(current_log_std.mean()):.4f}")
    
    print(f"  - 设备: {model.device}")
    print("="*80)
    
    # 🔄 如果继续训练，先确定要恢复到哪个episode（优先使用模型文件名的episode数）
    target_episode = 0
    if continue_training and 'load_model_path' in locals() and load_model_path:
        if isinstance(load_model_path, (str, Path)):
            model_name = Path(load_model_path).name
            if 'episode_' in model_name:
                try:
                    target_episode = int(model_name.split('episode_')[-1])
                    print(f"🎯 目标恢复到episode: {target_episode}")
                except:
                    pass
    
    # 🔥 创建奖励跟踪器（使用正确的initial_episode）
    # ✅ 开启绘图功能（训练结束后会自动画图）
    # 🔥 save_full_history=True: 采样保存完整历史（每50回合1条，内存占用~500KB）
    reward_tracker = RewardTracker(
        window_size=50, 
        enable_plotting=True, 
        save_full_history=True, 
        initial_episode=target_episode  # 🔥 使用真实的起始回合数
    )
    
    if continue_training and 'history_data' in locals() and history_data:
        print("正在恢复训练历史...")
        
        # 🔥 加载所有历史数据
        all_rewards = history_data.get('episode_rewards', [])
        all_lengths = history_data.get('episode_lengths', [])
        all_moving_avg = history_data.get('moving_avg_rewards', [])
        all_success_rate = history_data.get('success_rate', [])
        all_success_flags = history_data.get('success_flags', [])
        all_collision_rate = history_data.get('collision_rate', [])
        all_moving_avg_collision = history_data.get('moving_avg_collision', [])
        
        # 🔥 如果目标episode < 历史数据总数，截断到目标episode
        if target_episode > 0 and target_episode < len(all_rewards):
            print(f"  ⚠️  历史数据有 {len(all_rewards)} 回合，但模型是episode_{target_episode}")
            print(f"  🔪 截断历史数据到前 {target_episode} 回合（丢弃后续数据）")
            
            reward_tracker.episode_rewards = all_rewards[:target_episode]
            reward_tracker.episode_lengths = all_lengths[:target_episode]
            reward_tracker.moving_avg_rewards = all_moving_avg[:target_episode]
            reward_tracker.success_rate = all_success_rate[:target_episode]
            reward_tracker.success_flags = all_success_flags[:target_episode]
            reward_tracker.collision_rate = all_collision_rate[:target_episode]
            
            # 对于moving_avg_collision，如果长度不匹配则重新计算
            if len(all_moving_avg_collision) >= target_episode:
                reward_tracker.moving_avg_collision = all_moving_avg_collision[:target_episode]
            else:
                print("  ⚠️  moving_avg_collision长度不足，重新计算...")
                reward_tracker.moving_avg_collision = []
                for i in range(target_episode):
                    if i >= reward_tracker.window_size:
                        avg_collision = np.mean(reward_tracker.collision_rate[i-reward_tracker.window_size+1:i+1])
                    else:
                        avg_collision = np.mean(reward_tracker.collision_rate[:i+1])
                    reward_tracker.moving_avg_collision.append(avg_collision)
        else:
            # 正常恢复所有数据
            reward_tracker.episode_rewards = all_rewards
            reward_tracker.episode_lengths = all_lengths
            reward_tracker.moving_avg_rewards = all_moving_avg
            reward_tracker.success_rate = all_success_rate
            reward_tracker.success_flags = all_success_flags
            reward_tracker.collision_rate = all_collision_rate
            
            # 重新计算moving_avg_collision（如果历史数据中没有）
            if all_moving_avg_collision:
                reward_tracker.moving_avg_collision = all_moving_avg_collision
            else:
                print("  ⚠️  历史数据缺少moving_avg_collision，重新计算...")
                reward_tracker.moving_avg_collision = []
                for i in range(len(reward_tracker.collision_rate)):
                    if i >= reward_tracker.window_size:
                        avg_collision = np.mean(reward_tracker.collision_rate[i-reward_tracker.window_size+1:i+1])
                    else:
                        avg_collision = np.mean(reward_tracker.collision_rate[:i+1])
                    reward_tracker.moving_avg_collision.append(avg_collision)
        
        # 🔥 确保initial_episode已正确设置（已在创建时设置，这里无需再改动）
        print(f"✅ 已恢复 {len(reward_tracker.episode_rewards)} 回合的训练历史（起始回合：{reward_tracker.initial_episode}）")
        if reward_tracker.success_rate:
            print(f"   恢复点成功率: {reward_tracker.success_rate[-1]:.1%}")
        if reward_tracker.moving_avg_rewards:
            print(f"   恢复点平均奖励: {reward_tracker.moving_avg_rewards[-1]:.2f}")
        if reward_tracker.moving_avg_collision:
            print(f"   恢复点碰撞率: {reward_tracker.moving_avg_collision[-1]:.1%}")
    
    # 创建回调函数
    initial_episode = 0
    if continue_training and 'history_data' in locals() and history_data:
        # 🔥 从模型文件名提取episode数，而不是从历史数据
        if isinstance(load_model_path, (str, Path)):
            model_name = Path(load_model_path).name
            if 'episode_' in model_name:
                try:
                    # 提取episode数字（例如：leader_phase1_episode_99000 → 99000）
                    episode_num = int(model_name.split('episode_')[-1])
                    initial_episode = episode_num
                    print(f"🔄 从模型文件episode数继续训练: {initial_episode}")
                except:
                    # 如果提取失败，使用历史数据
                    initial_episode = history_data.get('total_episodes', 0)
                    print(f"🔄 从历史数据继续训练: {initial_episode}")
            else:
                initial_episode = history_data.get('total_episodes', 0)
                print(f"🔄 从历史数据继续训练: {initial_episode}")
        else:
            initial_episode = history_data.get('total_episodes', 0)
            print(f"🔄 从历史数据继续训练: {initial_episode}")
    
    callback = TrainingCallback(
        reward_tracker=reward_tracker,
        max_episodes=max_episodes,  # 🔥 传入最大回合数
        plot_interval=plot_interval,
        save_interval=plot_interval,
        initial_episode=initial_episode  # 🔥 传入初始回合数
    )
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10000,     # 🔥 大幅降低日志频率（10000步打印一次）
            progress_bar=False      # 使用自定义进度条
        )
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️ 训练被用户中断（Ctrl+C）！")
        print("="*80)
        
        # 🔥 中断时保存当前进度
        training_time = time.time() - start_time
        current_episode = callback.episode_count
        
        print(f"\n📊 中断时统计:")
        print(f"  - 已完成回合: {current_episode}/{max_episodes}")
        print(f"  - 已训练时长: {training_time/3600:.1f}小时")
        
        # 保存中断点模型
        interrupt_model_path = PathConfig.MODEL_DIR / f"leader_phase1_interrupt_{current_episode}"
        model.save(interrupt_model_path)
        print(f"💾 中断点模型已保存: {interrupt_model_path}")
        
        # 🔥 保存归一化参数（中断点 - 带回合数版本 + 最新版本）
        if use_state_norm and hasattr(env.envs[0], 'save_normalization_params'):
            # 带回合数的版本（永久保存）
            norm_params_versioned = PathConfig.MODEL_DIR / f"state_norm_params_interrupt_{current_episode}.npz"
            env.envs[0].save_normalization_params(str(norm_params_versioned))
            # 最新版本（每次覆盖）
            norm_params_latest = PathConfig.MODEL_DIR / "state_norm_params.npz"
            env.envs[0].save_normalization_params(str(norm_params_latest))
        
        # 绘制中断点的训练曲线
        print("\n🎨 正在生成中断点训练曲线图...")
        interrupt_plot_path = PathConfig.LOG_DIR / f"training_progress_interrupt_{current_episode}.png"
        reward_tracker.plot_training_progress(interrupt_plot_path)
        
        # 保存中断点的训练数据
        print("💾 正在保存中断点训练数据...")
        interrupt_data_path = PathConfig.LOG_DIR / f"training_data_interrupt_{current_episode}.json"
        reward_tracker.save_data(interrupt_data_path, minimal=False)
        
        print("="*80)
        print(f"✅ 中断点数据已保存，可使用以下参数继续训练:")
        print(f"   continue_training=True")
        print(f"   load_model_path='{interrupt_model_path}'")
        print("="*80)
        
        # 关闭进度条和环境
        if hasattr(callback, 'pbar') and callback.pbar:
            callback.pbar.close()
        env.close()
        
        # 返回中断点的路径
        return interrupt_model_path, reward_tracker, model
    
    # 训练完成
    training_time = time.time() - start_time
    print("="*80)
    print("✅ 训练完成！")
    
    # 最终统计
    final_avg_reward = list(reward_tracker.moving_avg_rewards)[-1] if reward_tracker.moving_avg_rewards else 0
    final_success_rate = list(reward_tracker.success_rate)[-1] if reward_tracker.success_rate else 0
    
    print(f"最终统计:")
    print(f"  - 总回合数: {len(reward_tracker.episode_rewards)}")
    print(f"  - 总步数: {callback.num_timesteps}")
    print(f"  - 最终平均奖励: {final_avg_reward:.2f}")
    print(f"  - 最终成功率: {final_success_rate:.2%}")
    print(f"  - 训练时长: {training_time/60:.1f}分钟")
    
    # 保存最终模型和数据
    final_model_path = PathConfig.FINAL_MODEL
    model.save(final_model_path)
    print(f"💾 最终模型已保存: {final_model_path}")
    
    # 🔥 保存归一化参数（最终版本 + 最新版本）
    if use_state_norm and hasattr(env.envs[0], 'save_normalization_params'):
        # 最终版本（永久保存）
        norm_params_final = PathConfig.MODEL_DIR / "state_norm_params_final.npz"
        env.envs[0].save_normalization_params(str(norm_params_final))
        # 最新版本（每次覆盖）
        norm_params_latest = PathConfig.MODEL_DIR / "state_norm_params.npz"
        env.envs[0].save_normalization_params(str(norm_params_latest))
    
    # ✅ 训练结束后自动绘制完整训练曲线图
    print("\n🎨 正在生成训练曲线图...")
    reward_tracker.plot_training_progress(PathConfig.FINAL_PROGRESS_PLOT)
    
    # ✅ 保存完整训练数据
    print("💾 正在保存训练数据...")
    reward_tracker.save_data(PathConfig.FINAL_DATA_JSON, minimal=False)  # 完整保存
    
    
    print("="*80)
    print(f"所有结果已保存:")
    print(f"  📊 日志和图表: agent/log_SB3/")
    print(f"  📁 模型文件: agent/model_SB3/")
    print("="*80)
    
    # 关闭进度条
    if hasattr(callback, 'pbar') and callback.pbar:
        callback.pbar.close()
    
    env.close()
    return final_model_path, reward_tracker, model


if __name__ == '__main__':
    model_path, reward_tracker, model = train_leader_phase1_sb3(
        max_episodes=100000,       # 训练回合数
        total_timesteps=None,      # 自动根据回合数估算
        plot_interval=1000,        # 🔥 每1000回合保存一次检查点（降低频率）
        continue_training=False,   # 从头开始训练
        load_model_path=None,      # 不加载任何模型
        use_state_norm=True,       # 🔥 启用状态归一化（DPL-PPO Trick 2）
        use_reward_scaling=False,  # 🔥 启用奖励缩放（DPL-PPO Trick 3 & 4）
        # 🆕 物理引导参数（基于实际loss量级修正）
        use_physics_guidance=True,  # 🔬 启用物理引导学习
        # lambda_phys将从physics_config.yaml自动读取
    )
    
    print(f"\n🎉 训练完成！检查 {PathConfig.MODEL_DIR} 查看模型文件。")

