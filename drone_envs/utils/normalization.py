"""
状态归一化模块 - 动态计算均值和标准差
基于 Welford 在线算法实现 RunningMeanStd
用于 PPO Trick 2: State Normalization
"""
import numpy as np
from typing import Optional
from pathlib import Path


class RunningMeanStd:
    """动态计算均值和标准差（Welford在线算法）
    
    核心思想：已知n个数据的均值和方差，高效计算n+1个数据的均值和方差
    无需存储所有历史数据，内存占用恒定
    """
    
    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        """
        Args:
            shape: 数据维度，例如 (140,) 对应140维状态
            epsilon: 数值稳定性参数，避免除零
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # 避免除零
        
    def update(self, x: np.ndarray):
        """
        更新统计量（Welford在线算法）
        
        原理：
        - mean_n+1 = mean_n + (x - mean_n) / (n+1)
        - var_n+1 = var_n + (x - mean_n) * (x - mean_n+1)
        
        Args:
            x: 新的观测样本，shape必须与初始化时一致
        """
        x = np.array(x, dtype=np.float64)
        batch_mean = x
        batch_count = 1
        
        # 增量更新均值和方差
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = np.zeros_like(self.mean)  # 单样本方差为0
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        """返回标准差"""
        return np.sqrt(self.var)


class StateNormalization:
    """状态归一化包装器
    
    功能：
    1. 动态统计状态的均值和标准差
    2. 将状态标准化到均值0、方差1
    3. 裁剪极端值，防止数值不稳定
    4. 支持保存/加载归一化参数
    """
    
    def __init__(self, shape: tuple, clip_range: float = 10.0):
        """
        Args:
            shape: 状态维度，例如 (140,)
            clip_range: 归一化后的裁剪范围 [-clip_range, clip_range]
                       推荐10.0，可以覆盖99.9%的正态分布数据
        """
        self.running_ms = RunningMeanStd(shape=shape)
        self.clip_range = clip_range
        
    def __call__(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        """
        归一化状态
        
        Args:
            x: 原始状态
            update: 是否更新统计量（训练时True，测试/评估时False）
        
        Returns:
            归一化后的状态 (均值0, 方差1, 裁剪到[-clip_range, clip_range])
        """
        if update:
            self.running_ms.update(x)
        
        # 标准化：(x - mean) / (std + 1e-8)
        normalized = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        
        # 裁剪到合理范围，防止极端值破坏训练
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return normalized.astype(np.float32)
    
    def save(self, path: str):
        """保存归一化参数到文件
        
        Args:
            path: 保存路径，例如 'state_norm_params.npz'
        """
        np.savez(path, 
                 mean=self.running_ms.mean,
                 var=self.running_ms.var,
                 count=self.running_ms.count)
    
    def load(self, path: str):
        """加载归一化参数
        
        Args:
            path: 参数文件路径
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"归一化参数文件不存在: {path}")
        
        data = np.load(path)
        self.running_ms.mean = data['mean']
        self.running_ms.var = data['var']
        self.running_ms.count = data['count']


class RewardScaling:
    """奖励缩放类 - PPO Trick 3 & 4
    
    原理：
    动态计算 discounted sum of rewards 的标准差，然后用当前奖励除以该标准差。
    与 Reward Normalization 的区别：
    - Reward Normalization: reward = (reward - mean) / std
    - Reward Scaling: reward = reward / std  (只除以std，不减去mean)
    
    优势：
    - 保留奖励的符号和相对大小关系
    - 只调整尺度，不改变奖励的相对值
    - 在连续控制任务中表现优于 Reward Normalization
    
    参考：《Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO》
    """
    
    def __init__(self, shape: int = 1, gamma: float = 0.99):
        """
        Args:
            shape: 奖励维度（通常为1，标量奖励）
            gamma: 折扣因子，与PPO算法中的gamma保持一致
        """
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape, dtype=np.float64)  # 累积折扣奖励
        
    def __call__(self, reward: float, update: bool = True) -> float:
        """
        缩放奖励
        
        Args:
            reward: 原始奖励
            update: 是否更新统计量（训练时True，测试时False）
        
        Returns:
            缩放后的奖励
        """
        reward_array = np.array([reward], dtype=np.float64)
        
        if update:
            # 更新累积折扣奖励: R_t = gamma * R_{t-1} + r_t
            self.R = self.gamma * self.R + reward_array
            # 更新统计量
            self.running_ms.update(self.R)
        
        # 只除以标准差，不减去均值
        scaled_reward = reward_array / (self.running_ms.std + 1e-8)
        
        return float(scaled_reward[0])
    
    def reset(self):
        """每个回合结束时重置累积折扣奖励"""
        self.R = np.zeros(self.shape, dtype=np.float64)
    
    def save(self, path: str):
        """保存缩放参数"""
        np.savez(path,
                 mean=self.running_ms.mean,
                 var=self.running_ms.var,
                 count=self.running_ms.count,
                 R=self.R,
                 gamma=self.gamma)
    
    def load(self, path: str):
        """加载缩放参数"""
        if not Path(path).exists():
            raise FileNotFoundError(f"奖励缩放参数文件不存在: {path}")
        
        data = np.load(path)
        self.running_ms.mean = data['mean']
        self.running_ms.var = data['var']
        self.running_ms.count = data['count']
        self.R = data['R']
        self.gamma = float(data['gamma'])
