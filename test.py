"""
模型测试脚本（Stable-Baselines3版本）
测试领航者避障和导航性能
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
import random
import torch
from collections import defaultdict

# 导入stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # 🔥 添加：与训练一致
import gym  # 🔥 添加：用于Wrapper

from drone_envs.envs.drone_env_multi import DroneNavigationMulti
from drone_envs.utils.normalization import StateNormalization, RewardScaling  # 🔥 添加归一化工具
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class StateNormalizationWrapper(gym.Wrapper):
    """状态归一化包装器 - 用于gym环境（与训练脚本完全一致）
    
    功能：
    - 动态统计状态的均值和标准差（RunningMeanStd）
    - 将所有观测标准化到均值0、方差1
    - 训练时更新统计量，评估时不更新
    - 支持保存/加载归一化参数
    - 支持Reward Scaling（PPO Trick 3 & 4）
    """
    
    def __init__(self, env, clip_range: float = 10.0, use_reward_scaling: bool = True, gamma: float = 0.99):
        super().__init__(env)
        # 获取状态维度（第一阶段：140维）
        obs_shape = env.observation_space.shape
        self.state_normalizer = StateNormalization(shape=obs_shape, clip_range=clip_range)
        self.is_training = False  # 🔥 测试时默认为评估模式
        
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


def make_env(num_drones=1, max_steps=1000, use_state_norm=True, use_reward_scaling=False, gamma=0.99):
    """创建环境的工厂函数 - 与训练脚本完全一致
    
    Args:
        num_drones: 无人机数量
        max_steps: 每个回合的最大步数
        use_state_norm: 是否启用状态归一化（必须与训练时一致！）
        use_reward_scaling: 是否启用奖励缩放（测试时通常为False）
        gamma: 折扣因子
    """

    def _init():
        env = DroneNavigationMulti(
            num_drones=num_drones,
            use_depth_camera=True,
            depth_camera_range=10.0,
            depth_resolution=16,
            enable_formation_force=False,
            training_stage=1,
            max_steps=max_steps
        )
        
        # 🔥 包装状态归一化层（与训练一致）
        if use_state_norm:
            env = StateNormalizationWrapper(
                env, 
                clip_range=10.0,
                use_reward_scaling=use_reward_scaling,
                gamma=gamma
            )
        
        return env
    return _init

def test_model(model_path, num_episodes=20, max_steps=1000, render=True, 
               use_state_norm=True, norm_params_path=None):
    """测试第一阶段训练的SB3 PPO模型
    
    Args:
        model_path: 模型文件路径
        num_episodes: 测试回合数
        max_steps: 每回合最大步数
        render: 是否渲染
        use_state_norm: 是否使用状态归一化（必须与训练时一致！）
        norm_params_path: 归一化参数文件路径（如果为None则自动查找）
    """
    print("="*80)
    print("第一阶段模型测试（Stable-Baselines3 PPO）：领航者避障和导航")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"测试回合数: {num_episodes}")
    print(f"最大步数: {max_steps}")
    print(f"状态归一化: {'启用 ✅' if use_state_norm else '禁用'}")
    
    # 🔥 创建环境 - 与训练环境完全一致（使用DummyVecEnv包装 + 状态归一化）
    env = DummyVecEnv([make_env(
        num_drones=1, 
        max_steps=max_steps,
        use_state_norm=use_state_norm,
        use_reward_scaling=False,  # 测试时不需要奖励缩放
        gamma=0.99
    )])
    
    # 获取底层环境用于检查配置
    wrapper_env = env.envs[0]
    # 🔥 如果有StateNormalizationWrapper，需要访问被包装的环境
    if hasattr(wrapper_env, 'env'):
        base_env = wrapper_env.env
        print(f"  - 环境包装: DummyVecEnv + StateNormalizationWrapper ✅")
    else:
        base_env = wrapper_env
        print(f"  - 环境包装: DummyVecEnv")
    
    print(f"环境配置:")
    print(f"  - 无人机数量: {base_env.num_drones}")
    print(f"  - 观测空间: {base_env.observation_space.shape}")
    print(f"  - 动作空间: {base_env.action_space.shape}")
    print(f"  - 深度特征维度: {base_env.depth_feature_dim}")
    print(f"  - 训练阶段: {base_env.training_stage}")
    print(f"  - 编队力状态: {'禁用' if not base_env.enable_formation_force else '启用'}")
    print(f"  - 平面模式: {'启用' if base_env.enforce_planar else '禁用'}")
    print(f"  - 最大步数: {base_env.max_steps}")
    
    # 加载SB3 PPO模型
    if os.path.exists(model_path + '.zip') or os.path.exists(model_path):
        try:
            # SB3会自动添加.zip后缀
            model = PPO.load(model_path, env=env)
            print(f"✓ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return None
    else:
        print(f"✗ 模型文件不存在: {model_path}")
        return None
    
    # 🔥 加载归一化参数（如果启用了状态归一化）
    if use_state_norm:
        # 自动查找归一化参数文件
        if norm_params_path is None:
            # 尝试查找对应的归一化参数文件
            model_dir = Path(model_path).parent
            model_name = Path(model_path).stem
            
            # 优先级1: 对应episode的归一化参数
            if 'episode_' in model_name:
                episode_num = model_name.split('episode_')[-1]
                norm_params_path = model_dir / f"state_norm_params_ep{episode_num}.npz"
            
            # 优先级2: 最终归一化参数
            if norm_params_path is None or not norm_params_path.exists():
                norm_params_path = model_dir / "state_norm_params_final.npz"
            
            # 优先级3: 最新归一化参数
            if not norm_params_path.exists():
                norm_params_path = model_dir / "state_norm_params.npz"
        
        # 加载归一化参数
        if Path(norm_params_path).exists():
            # 获取底层环境的归一化包装器
            base_env = env.envs[0]
            if hasattr(base_env, 'load_normalization_params'):
                base_env.load_normalization_params(str(norm_params_path))
            else:
                print(f"⚠️  环境没有归一化包装器，无法加载参数")
        else:
            print(f"⚠️  未找到归一化参数文件: {norm_params_path}")
            print(f"⚠️  警告：测试将使用未归一化的统计量，结果可能不准确！")
    
    print("="*80)
    
    # 测试统计
    test_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'collision_count': 0,
        'boundary_collision_count': 0,
        'physical_collision_count': 0,
        'timeout_count': 0,
        'min_depths': [],
        'goal_distances': [],
        'reward_components': defaultdict(list)  # 记录各个奖励分量
    }
    
    # 开始测试
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()  # 🔥 DummyVecEnv返回的是(1, obs_dim)
        episode_reward = 0
        obstacle_detections = 0
        collision_occurred = False
        collision_type = ""
        min_depths = []
        episode_reward_components = defaultdict(float)
        
        for step in range(max_steps):
            # 🔥 获取底层环境进行监控（跳过包装器）
            wrapper_env = env.envs[0]
            real_env = wrapper_env.env if hasattr(wrapper_env, 'env') else wrapper_env
            
            # 监控避障信息
            if hasattr(real_env, 'depth_obstacle_processor') and real_env.use_leader_camera:
                try:
                    # 使用屏蔽后的深度图像进行避障检测，避免无人机自身被误认为障碍物
                    depth_image = real_env._get_masked_leader_depth()
                    if depth_image is not None and depth_image.size > 0:
                        raw_depth = depth_image if len(depth_image.shape) == 2 else depth_image[:, :, 0]
                        processed_depth = real_env.depth_obstacle_processor.preprocess_depth_image(raw_depth)
                        obstacle_detected, min_depth = real_env.depth_obstacle_processor.detect_obstacles(processed_depth)
                        
                        min_depths.append(min_depth)
                        if obstacle_detected:
                            obstacle_detections += 1
                except Exception:
                    pass
            
            # 使用SB3模型预测动作（确定性策略，无探索噪声）
            action, _states = model.predict(state, deterministic=True)
            
            # 🔥 环境步进 - DummyVecEnv返回的都是数组形式
            next_state, reward, done, info = env.step(action)
            episode_reward += reward[0]  # 🔥 reward是数组，取第一个元素
            
            # 🔥 info也是列表形式
            info = info[0]
            
            # 记录奖励分量
            reward_info = info.get('reward_info', {})
            for key, value in reward_info.items():
                episode_reward_components[key] += value
            
            # 检查碰撞类型
            crash_reward = reward_info.get('crash', 0)
            if crash_reward < 0:
                collision_occurred = True
                if hasattr(real_env, '_get_depth_info'):
                    try:
                        depth_info = real_env._get_depth_info()
                        collision_type = depth_info.get('collision_type', 'unknown')
                    except:
                        collision_type = "碰撞"
            
            # 渲染（如果启用）
            if render:
                env.render()
                time.sleep(0.01)  # 控制渲染速度
            
            state = next_state
            
            # 🔥 DummyVecEnv的done是数组
            if done[0]:
                break
        
        # 统计结果
        success = info.get('success', False)
        
        test_results['episode_rewards'].append(episode_reward)
        test_results['episode_lengths'].append(step + 1)
        if min_depths:
            test_results['min_depths'].append(np.mean(min_depths))
        
        # 记录奖励分量
        for key, value in episode_reward_components.items():
            test_results['reward_components'][key].append(value)
        
        # 记录结果类型
        if success:
            test_results['success_count'] += 1
            result_str = "✓ 成功"
        elif collision_occurred:
            test_results['collision_count'] += 1
            if collision_type == 'boundary':
                test_results['boundary_collision_count'] += 1
                result_str = "✗ 边界碰撞"
            else:
                test_results['physical_collision_count'] += 1
                result_str = f"✗ {collision_type}"
        elif step + 1 >= max_steps:
            test_results['timeout_count'] += 1
            result_str = "⏱ 超时"
        else:
            result_str = "? 其他"
        
        # 计算到目标距离
        if hasattr(real_env, 'goal') and real_env.goal is not None:
            leader_pos, _ = real_env.drones[0].get_position_and_orientation() if hasattr(real_env.drones[0], 'get_position_and_orientation') else ([0,0,0], [0,0,0,1])
            goal_distance = np.linalg.norm(np.array(leader_pos) - np.array(real_env.goal))
            test_results['goal_distances'].append(goal_distance)
        
        print(f"回合 {episode + 1:2d}/{num_episodes} | "
              f"奖励: {episode_reward:7.2f} | "
              f"步数: {step + 1:4d} | "
              f"障碍物检测: {obstacle_detections:3d} | "
              f"结果: {result_str}")
    
    # 计算最终统计
    total_time = time.time() - start_time
    success_rate = test_results['success_count'] / num_episodes
    collision_rate = test_results['collision_count'] / num_episodes
    avg_reward = np.mean(test_results['episode_rewards'])
    avg_length = np.mean(test_results['episode_lengths'])
    
    print("="*80)
    print("测试结果统计:")
    print("="*80)
    print(f"总回合数: {num_episodes}")
    print(f"成功回合: {test_results['success_count']} ({success_rate:.1%})")
    print(f"碰撞回合: {test_results['collision_count']} ({collision_rate:.1%})")
    print(f"  - 边界碰撞: {test_results['boundary_collision_count']}")
    print(f"  - 物理碰撞: {test_results['physical_collision_count']}")
    print(f"超时回合: {test_results['timeout_count']} ({test_results['timeout_count']/num_episodes:.1%})")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    if test_results['min_depths']:
        print(f"平均最小深度: {np.mean(test_results['min_depths']):.2f}m")
    if test_results['goal_distances']:
        print(f"平均目标距离: {np.mean(test_results['goal_distances']):.2f}m")
    
    # 打印奖励分量统计
    if test_results['reward_components']:
        print(f"\n奖励分量平均值:")
        for key in sorted(test_results['reward_components'].keys()):
            values = test_results['reward_components'][key]
            avg_value = np.mean(values)
            print(f"  - {key}: {avg_value:.2f}")
    
    print(f"\n测试时长: {total_time:.1f}秒")
    print("="*80)
    
    env.close()
    
    # 转换defaultdict为普通dict以便JSON序列化
    test_results['reward_components'] = {k: list(v) for k, v in test_results['reward_components'].items()}
    
    return test_results

def plot_test_results(results, model_name, save_dir, show=True):
    """绘制测试结果图表
    Args:
        show: 是否显示图窗；批量跑评估时关闭以避免阻塞"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'第一阶段模型测试结果（SB3 PPO）- {model_name}', fontsize=16)
    
    episodes = range(1, len(results['episode_rewards']) + 1)
    
    # 1. 奖励曲线
    ax1.plot(episodes, results['episode_rewards'], 'b-', alpha=0.7, marker='o', markersize=3)
    ax1.axhline(y=np.mean(results['episode_rewards']), color='r', linestyle='--', 
                label=f'平均值: {np.mean(results["episode_rewards"]):.1f}')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.set_title('回合奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回合长度
    ax2.plot(episodes, results['episode_lengths'], 'g-', alpha=0.7, marker='s', markersize=3)
    ax2.axhline(y=np.mean(results['episode_lengths']), color='r', linestyle='--', 
                label=f'平均值: {np.mean(results["episode_lengths"]):.1f}')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('步数')
    ax2.set_title('回合长度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 结果类型统计
    result_types = ['成功', '边界碰撞', '物理碰撞', '超时']
    result_counts = [results['success_count'], 
                    results['boundary_collision_count'],
                    results['physical_collision_count'],
                    results['timeout_count']]
    
    colors = ['lightgreen', 'orange', 'red', 'gray']
    bars = ax3.bar(result_types, result_counts, color=colors)
    ax3.set_ylabel('回合数')
    ax3.set_title('测试结果类型统计')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    # 4. 结果统计饼图
    total = len(results['episode_rewards'])
    labels = [f'{label}\n({count}/{total})' 
              for label, count in zip(result_types, result_counts)]
    
    # 只显示非零项
    non_zero_sizes = [s for s in result_counts if s > 0]
    non_zero_labels = [l for l, s in zip(labels, result_counts) if s > 0]
    non_zero_colors = [c for c, s in zip(colors, result_counts) if s > 0]
    
    if non_zero_sizes:
        ax4.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('测试结果分布')
    else:
        ax4.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = save_dir / f"phase1_test_results_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 测试结果图表已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)  # 避免阻塞并释放资源
    return fig

def to_python_types(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    parser = argparse.ArgumentParser(description='测试训练的SB3 PPO模型')
    parser.add_argument('--model', type=str, 
                       default='agent/model_SB3/leader_phase1_final',
                       help='模型文件路径（不含.zip后缀）')
    parser.add_argument('--episodes', type=int, default=200,
                       help='测试回合数')
    parser.add_argument('--max_steps', type=int, default=400,  # 与训练一致
                       help='每回合最大步数')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用渲染（加快测试速度）')
    parser.add_argument('--save_dir', type=str, default='agent/log_SB3',
                       help='结果保存目录')
    parser.add_argument('--use_state_norm', action='store_true', default=True,
                       help='使用状态归一化（默认启用，必须与训练一致）')
    parser.add_argument('--norm_params', type=str, default=None,
                       help='归一化参数文件路径（如果为None则自动查找）')
    parser.add_argument('--no_show_plot', action='store_true',
                       help='只保存图，不弹出窗口（适合批量测试）')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试模型
    results = test_model(
        model_path=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        use_state_norm=args.use_state_norm,
        norm_params_path=args.norm_params
    )

    results = to_python_types(results)
    
    if results is not None:
        # 保存测试结果
        model_name = Path(args.model).stem
        results_path = save_dir / f"phase1_test_results_{model_name}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 测试结果数据已保存: {results_path}")
        
        # 绘制结果图表
        plot_test_results(results, model_name, save_dir, show=not args.no_show_plot)

if __name__ == '__main__':
    main()
