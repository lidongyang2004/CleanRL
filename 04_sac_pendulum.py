"""
================================================================================
CleanRL 实战教程 04：SAC (Soft Actor-Critic) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：SAC (柔性行动者-评论家 / 软演员-评论家 —— CleanRL 单文件 PyTorch 白盒实现)
- 策略类型：Off-Policy（异策略）—— 样本利用率极高，特别适合仿真耗时或真实机器人场景
- 学习范式：最大熵强化学习 (Maximum Entropy RL) + 随机策略 Actor-Critic
- 支持动作空间：**仅支持连续动作空间 (Box)**（连续控制任务的首选基准算法！）

【什么是“最大熵强化学习”？】
传统强化学习只追求最大化累积奖励：
    max Σ E[r(s_t, a_t)]
而 SAC 追求的是“既要拿高分，又要动作足够随机/多样化”：
    max Σ E[r(s_t, a_t) + α * H(π(·|s_t))]
其中：
- H(π(·|s_t)) = E[-log π(a|s_t)] 是策略分布的信息熵（熵越大代表分布越分散、探索越充分）；
- α（在 SB3 中对应 `ent_coef`）是温度系数，控制“探索（熵）”与“利用（奖励）”的相对权重。

【为什么 SAC 不需要像 TD3 那样手动加 ActionNoise？】
- TD3 的 Actor 是确定性网络，必须外部手动添加 NormalActionNoise 才能探索；
- SAC 的 Actor 是一个**随机策略网络**，它同时输出高斯分布的均值 μ(s) 和状态依赖标准差 σ(s)，
  并通过重参数化技巧 (Reparameterization Trick, `normal.rsample()`) + tanh 压缩采样得到有界连续动作。
  因此 SAC **天生自带自适应探索能力**！不仅如此，当开启自动温度调节（`ent_coef="auto"`）时，
  算法还会通过梯度下降自动优化可学习参数 `log_ent_coef` (log α)：前期探索不足时调大 α，后期策略成熟时调小 α。

【CleanRL 白盒实现 vs Stable-Baselines3 (SB3) 一一对应关系】
本脚本的 `main()` 主函数与 `stable-baseline3/04_sac_pendulum.py` **逐行 100% 一致**，
并在文件上半部分将 SB3 封装的高层类用纯 PyTorch 进行了透明白盒实现：
- `SACPolicy`                     <--> 显式定义 5 网络架构 (Squashed Gaussian `actor` + `critic(qf0+qf1)` + `critic_target`)
- `SAC` (`learn`/`save`/`load`)   <--> 显式展开四阶段梯度更新 (自动温度 α -> Soft 双 Critic -> Actor -> Polyak 软更新)
- `EvalCallback` & `evaluate_policy` <--> 定期评估并自动保存 `best_model.zip` 与 `evaluations.npz`

【实验环境：Pendulum-v1（倒立摆旋转控制）】
- 观测空间: Box(3,)
- 动作空间: Box(-2.0, 2.0, (1,), float32)
- 奖励机制: 每回合 200 步，奖励恒为负数，越接近 0 越好（优秀策略约 -150 ~ -250）。

运行方式：
    python 04_sac_pendulum.py                  # 使用默认步数 (20000 步) 完整训练
    python 04_sac_pendulum.py --timesteps 5000 # 快速测试全流程
================================================================================
"""

import argparse
import random
import time
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# SAC 标准对数标准差截断上下界，防止方差爆炸或数值下溢为 0（与 SB3 / CleanRL 源码一致）
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def set_seed(seed):
    """固定随机种子，保证实验结果可复现。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def Monitor(env):
    """CleanRL 白盒实现的 Monitor 包装器（对应 SB3 `Monitor`）。"""
    return gym.wrappers.RecordEpisodeStatistics(env)


# ==============================================================================
# CleanRL 白盒组件 1：连续动作经验回放池 (Replay Buffer)
# ==============================================================================
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminated):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(terminated)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return ReplayBufferSamples(
            observations=torch.as_tensor(self.observations[batch_inds], device=self.device),
            actions=torch.as_tensor(self.actions[batch_inds], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[batch_inds], device=self.device),
            dones=torch.as_tensor(self.dones[batch_inds], device=self.device),
            rewards=torch.as_tensor(self.rewards[batch_inds], device=self.device),
        )


# ==============================================================================
# CleanRL 白盒组件 2：SAC 五网络架构 (Squashed Gaussian Actor + Twin Critic + Target Critic)
# 严格对应 SB3 `SACPolicy(net_arch=dict(pi=[256, 256], qf=[256, 256]))`
# ==============================================================================
def _build_mlp(in_dim, hidden_dims, out_dim = None):
    """构建带 ReLU 隐藏层的多层感知机（当 out_dim=None 时仅返回隐藏层特征主干）。"""
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    if out_dim is not None:
        layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


def soft_update(source_net, target_net, tau):
    """Polyak 目标网络软更新：θ_target ← τ * θ + (1 - τ) * θ_target。"""
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class Actor(nn.Module):
    """
    SAC 随机策略网络（Squashed Gaussian Actor）：
    - 共享隐藏层 `latent_pi`：(batch, obs_dim) -> [256, 256] (ReLU)
    - 双头分叉：`mu` 输出高斯均值 μ(s)，`log_std` 输出状态依赖对数标准差 ln σ(s)
    - 重参数化采样 (Reparameterization Trick) + Tanh 压缩 + 雅可比行列式对数概率修正
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        pi_arch,
        action_low,
        action_high,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.latent_pi = _build_mlp(obs_dim, pi_arch, out_dim=None)

        last_dim = pi_arch[-1]
        self.mu = nn.Linear(last_dim, action_dim)       # 均值输出头 μ(s)
        self.log_std = nn.Linear(last_dim, action_dim)  # 对数标准差输出头 ln σ(s)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs):
        latent = self.latent_pi(self.flatten(obs))
        mean = self.mu(latent)
        log_std = torch.clamp(self.log_std(latent), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action_and_log_prob(self, obs):
        """
        核心数学步骤：
        1. 从无界高斯分布 N(μ(s), σ(s)^2) 中使用 rsample() 重参数化采样 x_t = μ(s) + σ(s) * ε；
        2. 经 y_t = tanh(x_t) 压缩到 (-1, 1)，再线性变换到真实力矩范围 [low, high]；
        3. 计算带雅可比行列式修正的对数概率：
           log π(a|s) = log N(x_t; μ, σ^2) - Σ log(action_scale * (1 - tanh(x_t)^2) + 1e-6)
        """
        mean, log_std = self(obs)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()  # rsample() 保持梯度可沿采样路径反向传播至 mean 和 log_std
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)  # (batch, 1)
        return action, log_prob

    def get_deterministic_action(self, obs):
        """评估/推理时直接输出 tanh(μ(s)) 映射后的确定性均值动作，消除随机采样抖动。"""
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class ContinuousCritic(nn.Module):
    """截断双 Q 价值网络组 (qf0 与 qf1)，输入维度为 obs_dim + action_dim。"""

    def __init__(self, obs_dim, action_dim, qf_arch):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        in_dim = obs_dim + action_dim
        self.qf0 = _build_mlp(in_dim, qf_arch, out_dim=1)
        self.qf1 = _build_mlp(in_dim, qf_arch, out_dim=1)

    def forward(self, obs, action):
        q_input = torch.cat([self.flatten(obs), action], dim=1)
        return self.qf0(q_input), self.qf1(q_input)


class SACPolicy(nn.Module):
    """
    SAC 完整五网络容器（无 actor_target！）：
    包含随机策略网络 actor、双主价值网络 critic (qf0+qf1)、双目标价值网络 critic_target (qf0+qf1)。
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        net_arch = None,
    ):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = list(net_arch.get("pi", [256, 256]))
            qf_arch = list(net_arch.get("qf", [256, 256]))
        elif isinstance(net_arch, list):
            pi_arch = list(net_arch)
            qf_arch = list(net_arch)
        else:
            pi_arch, qf_arch = [256, 256], [256, 256]

        self.actor = Actor(obs_dim, action_dim, pi_arch, action_low, action_high)
        self.critic = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def predict(self, obs, deterministic = True):
        """与 SB3 完全一致的推理接口：deterministic=True 时返回 tanh(μ(s)) 均值动作。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = self.actor.get_deterministic_action(obs_tensor)
            else:
                action, _ = self.actor.get_action_and_log_prob(obs_tensor)
            return action.squeeze(0).cpu().numpy(), None


# ==============================================================================
# CleanRL 白盒组件 3：策略评估 (evaluate_policy) 与评估回调 (EvalCallback)
# ==============================================================================
def evaluate_policy(
    model,
    eval_env,
    n_eval_episodes = 10,
    deterministic = True,
    return_episode_rewards = False,
):
    """在独立 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差。"""
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


class EvalCallback:
    """定期在独立 eval_env 上评估当前策略，记录 evaluations.npz 并在创新高时保存 best_model.zip。"""

    def __init__(
        self,
        eval_env,
        best_model_save_path = None,
        log_path = None,
        eval_freq = 10_000,
        n_eval_episodes = 5,
        deterministic = True,
        render = False,
    ):
        self.eval_env = eval_env
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self.log_path = Path(log_path) if log_path else None
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render

        self.n_calls = 0
        self.best_mean_reward = -float("inf")
        self.eval_timesteps = []
        self.eval_results = []
        self.eval_ep_lengths = []

    def on_step(self, model, num_timesteps, writer = None):
        self.n_calls += 1
        if self.n_calls % self.eval_freq != 0:
            return

        ep_rewards, ep_lengths = evaluate_policy(
            model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            return_episode_rewards=True,
        )
        eval_mean = float(np.mean(ep_rewards))
        eval_std = float(np.std(ep_rewards))

        self.eval_timesteps.append(num_timesteps)
        self.eval_results.append(ep_rewards)
        self.eval_ep_lengths.append(ep_lengths)

        if self.log_path is not None:
            self.log_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                self.log_path / "evaluations.npz",
                timesteps=np.array(self.eval_timesteps),
                results=np.array(self.eval_results),
                ep_lengths=np.array(self.eval_ep_lengths),
            )

        if writer is not None:
            writer.add_scalar("eval/mean_reward", eval_mean, num_timesteps)
            writer.add_scalar("eval/mean_ep_length", float(np.mean(ep_lengths)), num_timesteps)

        if getattr(model, "verbose", 1) >= 1:
            print(f"  [Eval @ {num_timesteps:>6d} steps] episode_reward={eval_mean:.2f} +/- {eval_std:.2f}")

        if eval_mean > self.best_mean_reward:
            self.best_mean_reward = eval_mean
            if self.best_model_save_path is not None:
                self.best_model_save_path.mkdir(parents=True, exist_ok=True)
                best_path = self.best_model_save_path / "best_model.zip"
                model.save(best_path)
                if getattr(model, "verbose", 1) >= 1:
                    print(f"    --> 刷新最高评估记录！已保存最优模型至: {best_path}")


# ==============================================================================
# CleanRL 白盒组件 4：SAC 算法主类（与 SB3 `SAC` 接口 1:1 完全对齐）
# ==============================================================================
class SAC:
    """
    CleanRL 白盒实现的 SAC 算法类（接口与 `stable_baselines3.SAC` 完全一致）：
    封装四阶段梯度更新：自动调节最大熵温度系数 α -> 双 Critic Soft Bellman 更新 -> Actor 更新 -> 目标网络软更新。
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate = 3e-4,
        gamma = 0.99,
        buffer_size = 100_000,
        learning_starts = 1_000,
        batch_size = 256,
        train_freq = 1,
        gradient_steps = 1,
        tau = 0.005,
        ent_coef = "auto",
        target_entropy = "auto",
        policy_kwargs = None,
        tensorboard_log = None,
        seed = None,
        verbose = 1,
        _obs_shape = None,
        _action_dim = None,
        _action_low = None,
        _action_high = None,
    ):
        set_seed(seed)
        self.policy_name = policy
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.tau = tau
        self.ent_coef_spec = ent_coef
        self.target_entropy_spec = target_entropy
        self.policy_kwargs = policy_kwargs or {}
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if env is not None:
            self.obs_shape = tuple(env.observation_space.shape)
            self.action_dim = int(env.action_space.shape[-1])
            self.action_low = np.array(env.action_space.low, dtype=np.float32)
            self.action_high = np.array(env.action_space.high, dtype=np.float32)
        else:
            assert _obs_shape is not None and _action_dim is not None
            assert _action_low is not None and _action_high is not None
            self.obs_shape = tuple(_obs_shape)
            self.action_dim = int(_action_dim)
            self.action_low = np.array(_action_low, dtype=np.float32)
            self.action_high = np.array(_action_high, dtype=np.float32)

        self.obs_dim = int(np.prod(self.obs_shape))
        self.net_arch = self.policy_kwargs.get("net_arch", dict(pi=[256, 256], qf=[256, 256]))

        self.policy = SACPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_low=self.action_low,
            action_high=self.action_high,
            net_arch=self.net_arch,
        ).to(self.device)

        # --- SAC 核心：最大熵温度系数自动调节 (ent_coef="auto", target_entropy="auto") ---
        # 目标熵下界默认为 -dim(A)（此处为 -1.0）；优化对数形式 log_ent_coef 保证 α = exp(log_ent_coef) > 0
        self.target_entropy = (
            -float(self.action_dim) if target_entropy == "auto" else float(target_entropy)
        )
        self.auto_entropy = ent_coef == "auto"
        init_ent = 1.0 if self.auto_entropy else float(ent_coef)
        self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_ent).requires_grad_(self.auto_entropy)

        # 三个独立 Adam 优化器：分别优化 Actor、Twin Critic 和对数熵温度参数 log_ent_coef
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=self.learning_rate)
        self.ent_coef_optimizer = (
            optim.Adam([self.log_ent_coef], lr=self.learning_rate) if self.auto_entropy else None
        )

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

    def predict(self, obs, deterministic = True):
        return self.policy.predict(obs, deterministic=deterministic)

    def _update_sac_step(self):
        """
        SAC 单次四阶段梯度更新：
        (a) 自动调节最大熵温度系数 α (ent_coef)
        (b) 用 Soft 状态价值 min(Q1', Q2') - α * log π(a'|s') 更新双 Critic 网络
        (c) 最小化 α * log π(a|s) - min(Q1, Q2) 更新随机策略 Actor
        (d) Polyak 软更新目标 Critic 网络
        """
        batch = self.replay_buffer.sample(self.batch_size)
        sampled_actions, log_prob = self.policy.actor.get_action_and_log_prob(batch.observations)

        # --- (a) 自动调节最大熵温度系数 α ---
        if self.auto_entropy and self.ent_coef_optimizer is not None:
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef_loss_val = ent_coef_loss.item()
        else:
            ent_coef_loss_val = 0.0
        ent_coef = self.log_ent_coef.exp().detach()

        # --- (b) 更新截断双 Q 价值网络 (ContinuousCritic) ---
        with torch.no_grad():
            next_actions, next_log_prob = self.policy.actor.get_action_and_log_prob(batch.next_observations)
            target_q1, target_q2 = self.policy.critic_target(batch.next_observations, next_actions)
            min_qf_next_target = torch.min(target_q1, target_q2) - ent_coef * next_log_prob
            next_q_value = batch.rewards + (1.0 - batch.dones) * self.gamma * min_qf_next_target

        current_q1, current_q2 = self.policy.critic(batch.observations, batch.actions)
        critic_loss = 0.5 * (F.mse_loss(current_q1, next_q_value) + F.mse_loss(current_q2, next_q_value))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- (c) 更新随机策略网络 (Actor) ---
        q1_pi, q2_pi = self.policy.critic(batch.observations, sampled_actions)
        min_qf_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- (d) Polyak 软更新目标 Critic 网络 ---
        soft_update(self.policy.critic, self.policy.critic_target, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "ent_coef": ent_coef.item(),
            "ent_coef_loss": ent_coef_loss_val,
        }

    def learn(
        self,
        total_timesteps,
        callback = None,
        tb_log_name = "SAC",
        progress_bar = False,
    ):
        assert self.env is not None, "训练前必须绑定环境 env"
        writer = None
        if self.tensorboard_log is not None:
            tb_run_dir = Path(self.tensorboard_log) / tb_log_name
            tb_run_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_run_dir))

        recent_ep_rewards = []
        recent_ep_lengths = []

        obs, _ = self.env.reset(seed=self.seed)
        start_time = time.time()

        for global_step in range(1, total_timesteps + 1):
            # 1. 动作采样：预热期纯随机探索；正式期从 Squashed Gaussian 随机策略中采样
            if global_step <= self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.policy.predict(obs, deterministic=False)

            # 2. 与环境交互一步并存入 ReplayBuffer
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.add(obs, next_obs, action, float(reward), terminated)
            obs = next_obs

            if terminated or truncated:
                if "episode" in info:
                    recent_ep_rewards.append(float(info["episode"]["r"]))
                    recent_ep_lengths.append(int(info["episode"]["l"]))
                    if writer is not None:
                        writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                        writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
                obs, _ = self.env.reset()

            # 3. 达到预热步数后，执行 SAC 四阶段梯度更新 (温度 α -> 双 Critic -> Actor -> 软更新)
            if global_step > self.learning_starts and global_step % self.train_freq == 0:
                metrics = None
                for _ in range(self.gradient_steps):
                    metrics = self._update_sac_step()
                if writer is not None and metrics is not None and global_step % 100 == 0:
                    writer.add_scalar("train/critic_loss", metrics["critic_loss"], global_step)
                    writer.add_scalar("train/actor_loss", metrics["actor_loss"], global_step)
                    writer.add_scalar("train/ent_coef", metrics["ent_coef"], global_step)
                    writer.add_scalar("train/ent_coef_loss", metrics["ent_coef_loss"], global_step)
                    writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

            # 4. 定期评估当前策略并保存最优模型
            if callback is not None:
                callback.on_step(self, global_step, writer=writer)

        if writer is not None:
            writer.close()
        return self

    def save(self, path):
        checkpoint = {
            "policy_name": self.policy_name,
            "obs_shape": self.obs_shape,
            "action_dim": self.action_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "tau": self.tau,
            "ent_coef": self.ent_coef_spec,
            "target_entropy": self.target_entropy_spec,
            "log_ent_coef": self.log_ent_coef.detach().cpu(),
            "policy_kwargs": self.policy_kwargs,
            "state_dict": self.policy.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, env = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            policy=checkpoint.get("policy_name", "MlpPolicy"),
            env=env,
            learning_rate=checkpoint.get("learning_rate", 3e-4),
            gamma=checkpoint.get("gamma", 0.99),
            buffer_size=checkpoint.get("buffer_size", 100_000),
            learning_starts=checkpoint.get("learning_starts", 1_000),
            batch_size=checkpoint.get("batch_size", 256),
            train_freq=checkpoint.get("train_freq", 1),
            gradient_steps=checkpoint.get("gradient_steps", 1),
            tau=checkpoint.get("tau", 0.005),
            ent_coef=checkpoint.get("ent_coef", "auto"),
            target_entropy=checkpoint.get("target_entropy", "auto"),
            policy_kwargs=checkpoint.get("policy_kwargs", dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))),
            verbose=0,
            _obs_shape=checkpoint.get("obs_shape"),
            _action_dim=checkpoint.get("action_dim"),
            _action_low=checkpoint.get("action_low"),
            _action_high=checkpoint.get("action_high"),
        )
        model.policy.load_state_dict(checkpoint["state_dict"])
        if "log_ent_coef" in checkpoint:
            model.log_ent_coef.data.copy_(checkpoint["log_ent_coef"].to(device))
        model.policy.eval()
        return model


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 SAC 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（SAC 样本效率极高，20000 步内通常可达到非常优秀的控制效果）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "Pendulum-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "sac_pendulum"
    log_dir = base_dir / "logs" / "sac_pendulum"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：环境初始化与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
    train_env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  动作范围:                     [{train_env.action_space.low[0]}, {train_env.action_space.high[0]}]")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置
    # ==========================================================================
    print("\n[第 2 步] 初始化 SAC 模型与核心超参数...")

    # 同样支持分别为 Actor (pi) 和 Twin Critic (qf) 指定隐藏层结构
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = SAC(
        policy="MlpPolicy",
        env=train_env,

        # --- 优化器与折扣因子 ---
        learning_rate=3e-4,       # SAC 最经典的默认学习率（对大多数连续控制任务都很稳健）
        gamma=0.99,               # 折扣因子 γ

        # --- 经验回放池 (Replay Buffer) ---
        buffer_size=100_000,      # 经验池容量
        learning_starts=1_000,    # 预热步数：先随机探索 1000 步积累基础样本再开始更新
        batch_size=256,           # SAC 推荐稍大的 batch_size（如 256）以降低梯度方差

        # --- 训练节奏与双 Q 目标网络软更新 ---
        train_freq=1,             # 每与环境交互 1 步，就做一次训练（高频更新带来极高的样本效率）
        gradient_steps=1,         # 每次训练执行 1 步梯度下降
        tau=0.005,                # 软更新系数：目标 Critic 网络以 0.5% 的步幅缓慢追踪主 Critic 网络

        # --- SAC 核心：最大熵温度系数自动调节 ---
        # ent_coef="auto" 是 SAC 最省心的特性！算法会自动优化温度系数 α，
        # 使策略熵维持在目标熵 target_entropy="auto"（默认为 -dim(A)，此处为 -1.0）附近。
        ent_coef="auto",
        target_entropy="auto",

        # 注意：SAC 无需设置 action_noise，因为策略本身就是随机高斯分布（重参数化采样）！
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # ==========================================================================
    # 第 3 步：训练前基线评估
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估...")
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：配置 EvalCallback 并启动训练
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 SAC（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, args.timesteps // 5)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name="SAC_Pendulum",
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "sac_pendulum_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    del model
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = SAC.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
    demo_env = gym.make(env_id)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # 注意 deterministic 参数的含义：
        # - 训练采样时内部为 deterministic=False：从高斯分布中采样随机动作，保证探索；
        # - 评估/上线推理时设 deterministic=True：直接输出 tanh(μ(s)) 均值动作，输出稳定无抖动！
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        if terminated or truncated:
            reason = "达到 200 步最大步数截断 (truncated)" if truncated else "任务终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
