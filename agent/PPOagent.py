import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # 🆕 Physics-guided parameters
        physics_loss_fn: Optional[Any] = None,
        physics_simulator: Optional[Any] = None,
        lambda_phys: float = 0.1,
        lambda_phys_min: float = 0.08,  # 🆕 课程调度最小值（从config读取）
        lambda_phys_max: float = 0.15,  # 🆕 课程调度最大值（从config读取）
        lambda_phys_curriculum_break: float = 0.6,  # 🆕 课程分段断点（默认 64 开：前 60% 更慢上升，后 40% 更平滑）
        lambda_safe: float = 0.0,
        use_physics_guidance: bool = False,
        physics_update_freq: int = 1,

        # 🆕 Lagrangian constraints (dual update)
        use_lagrangian_constraints: bool = False,
        lagrangian_lr: float = 1e-3,
        lagrangian_clip: float = 10.0,
        lagrangian_warmup_updates: int = 0,

        # Targets (more interpretable than raw loss): violation rates
        constraint_obstacle_violation_rate_target: float = 0.0,
        constraint_feasibility_violation_rate_target: float = 0.0,
        # Optional energy target (raw mean energy); set None to disable energy constraint
        constraint_energy_raw_target: Optional[float] = None,

        # Constraint normalization (separate from total physics normalization)
        constraint_norm_momentum: float = 0.995,
        constraint_norm_warmup_updates: int = 1000,
        constraint_norm_std_floor: float = 1e-3,
        constraint_norm_clip: float = 2.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        
        # 🆕 Physics-guided learning setup
        self.use_physics_guidance = use_physics_guidance
        self.physics_loss_fn = physics_loss_fn
        self.physics_simulator = physics_simulator
        self.lambda_phys = lambda_phys
        self.lambda_phys_min = lambda_phys_min  # 🆕 课程调度最小值
        self.lambda_phys_max = lambda_phys_max  # 🆕 课程调度最大值
        self.lambda_phys_curriculum_break = float(lambda_phys_curriculum_break)
        self.lambda_safe = lambda_safe
        self.physics_update_freq = physics_update_freq
        self._physics_update_counter = 0
        
        # Statistics tracking
        self.physics_loss_history = []
        self.lambda_phys_history = []
        
        # 🆕 Physics loss normalization (running statistics)
        self.physics_loss_mean = 0.0
        self.physics_loss_var = 1.0
        self.physics_norm_momentum = 0.995  # EMA momentum，降低统计抖动
        self.physics_norm_warmup_steps = 10000  # 延长warmup，先让统计稳定
        self.physics_norm_std_floor = 1.0  # 最小std，防止权重爆炸
        self.physics_norm_clip = 2.0  # 物理项幅度裁剪上限

        # 🆕 Lagrangian constraints setup
        self.use_lagrangian_constraints = use_lagrangian_constraints
        self.lagrangian_lr = lagrangian_lr
        self.lagrangian_clip = lagrangian_clip
        self.lagrangian_warmup_updates = lagrangian_warmup_updates

        self.constraint_obstacle_violation_rate_target = constraint_obstacle_violation_rate_target
        self.constraint_feasibility_violation_rate_target = constraint_feasibility_violation_rate_target
        self.constraint_energy_raw_target = constraint_energy_raw_target

        # Dual variables (non-negative)
        self.lambda_c_obstacle = 0.0
        self.lambda_c_feasibility = 0.0
        self.lambda_c_energy = 0.0

        # Per-constraint running stats for normalization
        self.constraint_norm_momentum = constraint_norm_momentum
        self.constraint_norm_warmup_updates = constraint_norm_warmup_updates
        self.constraint_norm_std_floor = constraint_norm_std_floor
        self.constraint_norm_clip = constraint_norm_clip

        self._constraint_mean = {
            'obstacle': 0.0,
            'feasibility': 0.0,
            'energy': 0.0,
        }
        self._constraint_var = {
            'obstacle': 1.0,
            'feasibility': 1.0,
            'energy': 1.0,
        }

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

                # 🎓 Phase 3: 课程学习式 lambda_phys 调度
        if self.use_physics_guidance:
            # 计算当前训练进度 (1.0 → 0.0 需要反转)
            training_progress = 1.0 - self._current_progress_remaining
            scheduled_lambda = self._lambda_phys_curriculum_schedule(training_progress)
            self.lambda_phys = scheduled_lambda

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        # 🆕 Physics loss tracking
        physics_losses = [] if self.use_physics_guidance else None
        physics_velocity_losses = [] if self.use_physics_guidance else None
        physics_obstacle_losses = [] if self.use_physics_guidance else None
        physics_smooth_losses = [] if self.use_physics_guidance else None
        physics_energy_losses = [] if self.use_physics_guidance else None
        physics_feasibility_losses = [] if self.use_physics_guidance else None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # 🔥🔥🔥 核心修改：添加物理损失 🔥🔥🔥
                physics_loss_total = th.tensor(0.0, device=self.device)
                physics_loss_normalized = th.tensor(0.0, device=self.device)

                # 🆕 Lagrangian constraint terms
                constraint_terms = th.tensor(0.0, device=self.device)
                obstacle_violation_rate = None
                feasibility_violation_rate = None
                energy_raw_mean = None
                
                if self.use_physics_guidance and self._physics_update_counter % self.physics_update_freq == 0:
                    physics_loss_components = self._compute_physics_loss(rollout_data)
                    
                    if physics_loss_components is not None:
                        # 提取各物理损失分量
                        physics_loss_total = physics_loss_components['total']
                        
                        # 记录详细分量（全部5个）
                        physics_losses.append(physics_loss_total.item())
                        physics_velocity_losses.append(physics_loss_components.get('velocity', th.tensor(0.0)).item())
                        physics_obstacle_losses.append(physics_loss_components.get('obstacle', th.tensor(0.0)).item())
                        physics_smooth_losses.append(physics_loss_components.get('smooth', th.tensor(0.0)).item())
                        physics_energy_losses.append(physics_loss_components.get('energy', th.tensor(0.0)).item())
                        physics_feasibility_losses.append(physics_loss_components.get('feasibility', th.tensor(0.0)).item())
                        
                        # 🔥🔥 归一化物理损失（解决后期LR衰减导致的梯度消失）🔥🔥
                        current_loss_value = physics_loss_total.item()
                        
                        # 更新运行统计量（EMA）
                        self.physics_loss_mean = (self.physics_norm_momentum * self.physics_loss_mean + 
                                                  (1 - self.physics_norm_momentum) * current_loss_value)
                        self.physics_loss_var = (self.physics_norm_momentum * self.physics_loss_var + 
                                                (1 - self.physics_norm_momentum) * (current_loss_value - self.physics_loss_mean)**2)
                        
                        # 仅按std缩放（不去均值），并加下限/裁剪保证平稳
                        if self._n_updates >= self.physics_norm_warmup_steps:
                            physics_std = max(np.sqrt(self.physics_loss_var) + 1e-8, self.physics_norm_std_floor)
                            physics_loss_normalized = physics_loss_total / physics_std
                            physics_loss_normalized = th.clamp(physics_loss_normalized, 0.0, self.physics_norm_clip)
                        else:
                            # warm-up期间不归一化，直接使用原值
                            physics_loss_normalized = physics_loss_total

                        # 🆕 Extract constraint signals (prefer raw + violation rate from PhysicsLossCalculator)
                        obstacle_raw = physics_loss_components.get('obstacle_raw', physics_loss_components.get('obstacle', th.tensor(0.0, device=self.device)))
                        feasibility_raw = physics_loss_components.get('feasibility_raw', physics_loss_components.get('feasibility', th.tensor(0.0, device=self.device)))
                        energy_raw = physics_loss_components.get('energy_raw', physics_loss_components.get('energy', th.tensor(0.0, device=self.device)))

                        obstacle_violation_rate = physics_loss_components.get('obstacle_violation_rate', None)
                        feasibility_violation_rate = physics_loss_components.get('feasibility_violation_rate', None)
                        energy_raw_mean = energy_raw

                        # ---- Dual update (projected gradient ascent on multipliers) ----
                        if self.use_lagrangian_constraints and self._n_updates >= self.lagrangian_warmup_updates:
                            # Ensure lagrangian_lr is a scalar even if loaded as list/array/string
                            try:
                                lag_lr = float(np.array(self.lagrangian_lr).item())
                            except Exception:
                                lag_lr = float(self.lagrangian_lr)

                            # Use violation rate when available, fallback to raw loss
                            obs_metric = obstacle_violation_rate if obstacle_violation_rate is not None else obstacle_raw
                            feas_metric = feasibility_violation_rate if feasibility_violation_rate is not None else feasibility_raw

                            obs_violation = (obs_metric - self.constraint_obstacle_violation_rate_target)
                            feas_violation = (feas_metric - self.constraint_feasibility_violation_rate_target)

                            # Detach to avoid leaking gradients into dual variables
                            obs_violation_val = float(obs_violation.detach().cpu().item())
                            feas_violation_val = float(feas_violation.detach().cpu().item())

                            self.lambda_c_obstacle = float(np.clip(max(0.0, self.lambda_c_obstacle + lag_lr * obs_violation_val), 0.0, self.lagrangian_clip))
                            self.lambda_c_feasibility = float(np.clip(max(0.0, self.lambda_c_feasibility + lag_lr * feas_violation_val), 0.0, self.lagrangian_clip))

                            if self.constraint_energy_raw_target is not None:
                                energy_violation = (energy_raw - self.constraint_energy_raw_target)
                                energy_violation_val = float(energy_violation.detach().cpu().item())
                                self.lambda_c_energy = float(np.clip(max(0.0, self.lambda_c_energy + lag_lr * energy_violation_val), 0.0, self.lagrangian_clip))

                        # ---- Constraint term normalization (separate running stats) ----
                        def _normalize_constraint(name: str, value: th.Tensor) -> th.Tensor:
                            v = float(value.detach().cpu().item())
                            m = self._constraint_mean[name]
                            mom = self.constraint_norm_momentum
                            m_new = mom * m + (1.0 - mom) * v
                            var = self._constraint_var[name]
                            var_new = mom * var + (1.0 - mom) * (v - m_new) ** 2
                            self._constraint_mean[name] = float(m_new)
                            self._constraint_var[name] = float(var_new)

                            if self._n_updates >= self.constraint_norm_warmup_updates:
                                std = max(np.sqrt(self._constraint_var[name]) + 1e-8, self.constraint_norm_std_floor)
                                out = value / std
                                return th.clamp(out, 0.0, self.constraint_norm_clip)

                            return value

                        obstacle_term = _normalize_constraint('obstacle', obstacle_raw)
                        feasibility_term = _normalize_constraint('feasibility', feasibility_raw)
                        energy_term = _normalize_constraint('energy', energy_raw)

                        if self.use_lagrangian_constraints:
                            constraint_terms = (
                                th.tensor(self.lambda_c_obstacle, device=self.device) * obstacle_term
                                + th.tensor(self.lambda_c_feasibility, device=self.device) * feasibility_term
                            )
                            if self.constraint_energy_raw_target is not None:
                                constraint_terms = constraint_terms + th.tensor(self.lambda_c_energy, device=self.device) * energy_term

                  # 🔥 总损失 = PPO损失 + 归一化后的物理损失 + 拉格朗日约束项
                loss = (policy_loss + 
                       self.ent_coef * entropy_loss + 
                       self.vf_coef * value_loss +
                      self.lambda_phys * physics_loss_normalized +
                      constraint_terms)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                self._physics_update_counter += 1

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        # 🆕 Physics-specific logs (记录全部5个物理损失分量)
        if self.use_physics_guidance and physics_losses:
            self.logger.record("physics/total_loss_raw", np.mean(physics_losses))  # 原始损失
            self.logger.record("physics/velocity_loss", np.mean(physics_velocity_losses))
            self.logger.record("physics/obstacle_loss", np.mean(physics_obstacle_losses))
            self.logger.record("physics/smooth_loss", np.mean(physics_smooth_losses))
            self.logger.record("physics/energy_loss", np.mean(physics_energy_losses))
            self.logger.record("physics/feasibility_loss", np.mean(physics_feasibility_losses))
            self.logger.record("physics/lambda_phys", self.lambda_phys)
            
            # 🆕 归一化统计信息
            self.logger.record("physics/loss_mean", self.physics_loss_mean)
            self.logger.record("physics/loss_std", np.sqrt(self.physics_loss_var))
            if self._n_updates >= self.physics_norm_warmup_steps:
                self.logger.record("physics/normalization_active", 1.0)
            else:
                self.logger.record("physics/normalization_active", 0.0)

        # 🆕 Constraint logs (dual variables + violation metrics)
        if self.use_physics_guidance and self.use_lagrangian_constraints:
            self.logger.record("constraints/lambda_obstacle", self.lambda_c_obstacle)
            self.logger.record("constraints/lambda_feasibility", self.lambda_c_feasibility)
            self.logger.record("constraints/obstacle_target", self.constraint_obstacle_violation_rate_target)
            self.logger.record("constraints/feasibility_target", self.constraint_feasibility_violation_rate_target)
            if self.constraint_energy_raw_target is not None:
                self.logger.record("constraints/lambda_energy", self.lambda_c_energy)
                self.logger.record("constraints/energy_target", self.constraint_energy_raw_target)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _compute_physics_loss(self, rollout_data) -> Optional[dict]:
        """
        计算物理一致性损失
        
        Args:
            rollout_data: RolloutBufferSamples对象
            
        Returns:
            包含各物理损失分量的字典，如果计算失败则返回None
        """
        if self.physics_loss_fn is None or self.physics_simulator is None:
            return None
        
        try:
            # 1️⃣ 提取状态信息
            observations = rollout_data.observations
            actions = rollout_data.actions
            
            # 从观测中分离状态和深度图
            # 🔥 观测格式：[pos(2), vel(2), quat(4), goal(2), depth_features(130)]
            # 总维度：140 = 2+2+4+2 + 130(128 CNN + 2 增强)
            state_dim = 4  # [pos_x, pos_y, vel_x, vel_y]
            depth_feature_dim = 128  # CNN特征维度
            
            states = observations[:, :state_dim]  # 提取位置和速度
            
            # 🔥 关键：CNN特征不能直接用作深度图！
            # 需要从环境获取原始深度图（64×64）
            batch_size = observations.shape[0]
            
            # 方案：尝试从环境获取原始深度图
            try:
                if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                    leader_env = self.env.envs[0]
                    if hasattr(leader_env, 'get_leader_depth_image'):
                        # 获取原始64×64深度图
                        raw_depth = leader_env.get_leader_depth_image()
                        if raw_depth is not None and raw_depth.size > 0:
                            # 转换为tensor并扩展到batch
                            depth_maps = th.from_numpy(raw_depth).float().to(self.device)
                            if depth_maps.dim() == 2:
                                # 🔥 修复：先unsqueeze，然后repeat到batch_size
                                depth_maps = depth_maps.unsqueeze(0).repeat(batch_size, 1, 1)
                        else:
                            # 回退：创建默认深度图
                            depth_maps = th.ones(batch_size, 64, 64, device=self.device) * 5.0
                    else:
                        # 环境没有depth image接口，使用默认
                        depth_maps = th.ones(batch_size, 64, 64, device=self.device) * 5.0
                else:
                    depth_maps = th.ones(batch_size, 64, 64, device=self.device) * 5.0
            except Exception as e:
                depth_maps = th.ones(batch_size, 64, 64, device=self.device) * 5.0
            
            # 2️⃣ 使用策略网络预测未来动作序列
            horizon = getattr(self.physics_loss_fn, 'horizon', 5)
            future_actions_list = []
            
            # 🔥🔥🔥 关键修复：不使用no_grad，让梯度可以反向传播 🔥🔥🔥
            current_obs = observations
            for _ in range(horizon):
                # 使用当前策略预测动作（保持梯度图连接）
                action_dist = self.policy.get_distribution(current_obs)
                predicted_action = action_dist.mode()  # 确定性预测
                future_actions_list.append(predicted_action)
            
            future_actions = th.stack(future_actions_list, dim=1)  # [batch, horizon, action_dim]
            # future_actions已经带有梯度，无需重新设置
            
            # 3️⃣ 使用可微分模拟器预测未来状态
            future_states = self.physics_simulator(states, future_actions)
            
            # 4️⃣ 准备环境信息
            env_info = self._prepare_env_info(states, depth_maps)
            
            # 5️⃣ 计算物理损失
            physics_losses = self.physics_loss_fn(
                states,
                future_actions,
                future_states,
                env_info
            )
            
            return physics_losses
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Physics loss computation failed: {e}")
            return None
    
    def _prepare_env_info(self, states, depth_maps) -> dict:
        """
        准备物理损失计算所需的环境信息
        
        Args:
            states: 当前状态 [batch, state_dim]
            depth_maps: 深度图 [batch, H, W]
            
        Returns:
            环境信息字典
        """
        # 🔥 计算目标速度（指向目标）
        # 假设环境有goal属性，这里简化为固定目标
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            env = self.env.envs[0]
            if hasattr(env, 'goal'):
                goal = th.tensor(env.goal[:2], device=self.device, dtype=th.float32)
            else:
                goal = th.tensor([5.0, 5.0], device=self.device, dtype=th.float32)
        else:
            goal = th.tensor([5.0, 5.0], device=self.device, dtype=th.float32)
        
        # 计算目标方向和期望速度
        batch_size = states.shape[0]
        current_positions = states[:, :2]  # [batch, 2]
        
        # 扩展goal到batch维度
        goal_batch = goal.unsqueeze(0).expand(batch_size, -1)
        
        # 计算指向目标的向量
        to_goal = goal_batch - current_positions
        distance_to_goal = th.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
        
        # 归一化方向 * 期望速度（2.0 m/s）
        desired_speed = 2.0
        target_velocity = (to_goal / distance_to_goal) * desired_speed
        
        env_info = {
            'target_velocity': target_velocity,
            'depth_maps': depth_maps,
            'goal': goal_batch,
            'distance_to_goal': distance_to_goal
        }
        
        return env_info
    
    def _lambda_phys_curriculum_schedule(self, progress: float) -> float:
        """
        🎓 课程学习式 lambda_phys 调度器（配合物理损失归一化）
        
        策略：前期低权重（PPO主导，快速学习基本技能）
              后期温和增强（物理约束打磨策略）
        
        ⚠️ 注意：归一化后，λ=0.15 的实际约束强度 ≈ 之前不归一化时的 0.25~0.3
        
        Args:
            progress: 训练进度 (0.0 → 1.0)
            
        Returns:
            当前lambda_phys值
            
        示例（使用config配置的范围）：
            progress=0.0 → λ=lambda_phys_min (从config读取)
            progress=0.6 → λ≈lambda_phys_min + (lambda_phys_max-lambda_phys_min) * 2/3
            progress=1.0 → λ=lambda_phys_max (从config读取)
        """
        # 🔥 分段线性（前快后慢）：
        # - 前 60%：从 lambda_min → lambda_mid（更慢更稳，避免你说的“太快了”）
        # - 后 40%：从 lambda_mid → lambda_max（更平滑收敛）
        lambda_min = float(self.lambda_phys_min)
        lambda_max = float(self.lambda_phys_max)

        # 防御：保证 progress 在 [0, 1]
        p = float(np.clip(progress, 0.0, 1.0))

        # 默认把“中点”设为区间的 2/3（对应你想要的 0.1→0.2→0.25 节奏）
        mid_ratio = 2.0 / 3.0
        lambda_mid = lambda_min + (lambda_max - lambda_min) * mid_ratio

        # 前后分段断点（默认 64 开：0.6；可通过构造参数 lambda_phys_curriculum_break 调整）
        # 说明：这里的“前快后慢”指斜率对比，斜率约为：
        #   slope_front = mid_ratio / p_break
        #   slope_back  = (1-mid_ratio) / (1-p_break)
        # 若希望前段更快（斜率更大），需要 p_break < mid_ratio。
        p_break = float(np.clip(getattr(self, 'lambda_phys_curriculum_break', 0.6), 1e-6, 1.0 - 1e-6))

        if p <= p_break:
            local_progress = p / max(p_break, 1e-8)
            return lambda_min + (lambda_mid - lambda_min) * local_progress

        local_progress = (p - p_break) / max(1.0 - p_break, 1e-8)
        return lambda_mid + (lambda_max - lambda_mid) * local_progress

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )