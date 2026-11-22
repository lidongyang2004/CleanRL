"""
================================================================================
CleanRL 实战教程 03：TD3 (Twin Delayed DDPG) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：TD3 (双延迟深度确定性策略梯度 —— CleanRL 单文件 PyTorch 白盒实现)
- 策略类型：Off-Policy（异策略）—— 使用 ReplayBuffer 存储历史连续控制经验
- 学习范式：确定性策略 Actor-Critic（Actor 直接输出一个确定的连续动作值 a = μ(s)，而非概率分布）
- 支持动作空间：**仅支持连续动作空间 (Box)**，如机械臂关节力矩、油门开度、舵机角度等

【从 DDPG 到 TD3 的三大核心改进】
经典的 DDPG 算法在连续控制中经常出现“Q 值高估 (Overestimation Bias)”，导致策略学歪。
TD3 提出了三项极具工程威力的改进（也是名字 Twin Delayed 的由来）：
1. 截断双 Q 网络 (Clipped Double Q-Learning —— "Twin")：
   同时维护两个独立的 Critic 网络 Q1 和 Q2 及其目标网络 Q1' 和 Q2'。
   计算目标价值时取二者的较小值：y = r + γ * min(Q1'(s', a'), Q2'(s', a'))，
   用保守估计有效遏制 Q 值虚高！
2. 延迟策略更新 (Delayed Policy Updates —— "Delayed", policy_delay=2)：
   Critic 每次都会更新，但 Actor 和目标网络每隔 `policy_delay` 次（默认 2 次）才更新 1 次。
   先让 Critic 把 Q 值评估准确了，再去指导 Actor 更新，避免 Actor 被充满噪声的 Q 值带偏。
3. 目标策略平滑 (Target Policy Smoothing)：
   在计算目标动作 a' 时，给目标 Actor 输出的动作加上截断的高斯噪声：
   a' = clip(μ'(s') + clip(ε, -c, c), a_low, a_high)。
   这相当于对 Q 函数在动作维度做局部平滑，防止 Critic 在个别动作点产生尖峰误差被 Actor 钻空子。

【确定性策略如何做探索？】
由于 TD3 的 Actor 是确定性的（同一状态永远输出相同动作），如果不额外加噪声就永远不会去探索新动作！
因此在训练阶段，必须手动构造一个 ActionNoise 对象（例如 NormalActionNoise 高斯噪声）
叠加到动作上：a_exec = μ(s) + noise。评估/推理时自动关闭此噪声。

【CleanRL 白盒实现 vs Stable-Baselines3 (SB3) 一一对应关系】
本脚本的 `main()` 主函数与 `stable-baseline3/03_td3_pendulum.py` **逐行 100% 一致**，
并在文件上半部分将 SB3 封装的高层类用纯 PyTorch 进行了透明白盒实现：
- `NormalActionNoise`             <--> 独立同分布高斯动作探索噪声
- `TD3Policy`                     <--> 显式定义 6 网络架构 (`actor`, `actor_target`, `critic(qf0+qf1)`, `critic_target`)
- `TD3` (`learn`/`save`/`load`)   <--> 显式展开目标策略平滑、截断双 Q 损失、延迟 Actor 更新与 Polyak 软更新
- `EvalCallback` & `evaluate_policy` <--> 定期评估并自动保存 `best_model.zip` 与 `evaluations.npz`

【实验环境：Pendulum-v1（倒立摆旋转控制）】
- 观测空间: Box(3,) —— [cos(θ), sin(θ), 角速度 θ_dot]
- 动作空间: Box(-2.0, 2.0, (1,), float32) —— 施加在摆杆上的力矩，连续实数 [-2, 2]
- 奖励机制: r = -(θ^2 + 0.1 * θ_dot^2 + 0.001 * torque^2)，每回合固定 200 步。
  奖励恒为负数，理想最优策略是把摆杆甩上去并稳稳倒立在正上方（θ=0），每回合总分通常在 -150 ~ -250 左右；
  随机策略总分通常在 -1200 ~ -1600 左右。

运行方式：
    python 03_td3_pendulum.py                  # 使用默认步数 (20000 步) 完整训练
    python 03_td3_pendulum.py --timesteps 5000 # 快速测试全流程
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
from torch.utils.tensorboard import SummaryWriter


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
# CleanRL 白盒组件 1：高斯动作探索噪声与连续动作经验回放池
# 对应 SB3 `NormalActionNoise` 与 `ReplayBuffer`
# ==============================================================================
class NormalActionNoise:
    """独立同分布的高斯动作噪声（对应 SB3 `NormalActionNoise`）。"""

    def __init__(self, mean, sigma):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)

    def __call__(self):
        return np.random.normal(self.mean, self.sigma).astype(np.float32)


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """连续动作经验回放池，存储 (s, a, r, s', terminated)。"""

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
# CleanRL 白盒组件 2：TD3 六网络架构 (Actor + ActorTarget + Twin Critic + Twin CriticTarget)
# 严格对应 SB3 `TD3Policy(net_arch=dict(pi=[256, 256], qf=[256, 256]))`
# ==============================================================================
def _build_mlp(
    in_dim,
    hidden_dims,
    out_dim,
    out_activation = None,
):
    """构建带 ReLU 隐藏层的多层感知机（可选输出层激活函数，如 Actor 的 Tanh）。"""
    layers = []
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


def soft_update(source_net, target_net, tau):
    """Polyak 目标网络软更新：θ_target ← τ * θ + (1 - τ) * θ_target。"""
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class Actor(nn.Module):
    """
    确定性策略网络 μ(s, θ)：
    (batch, obs_dim) -> [256, 256] (ReLU) -> Tanh 压缩至 [-1, 1] -> 线性映射回真实力矩范围 [low, high]。
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
        self.mu = _build_mlp(obs_dim, pi_arch, action_dim, out_activation=nn.Tanh())

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs):
        scaled_tanh = self.mu(self.flatten(obs))
        return scaled_tanh * self.action_scale + self.action_bias


class ContinuousCritic(nn.Module):
    """
    截断双 Q 价值网络组 (qf0 与 qf1)：
    输入为状态 s 与动作 a 的拼接 torch.cat([obs, action], dim=1)，维度为 obs_dim + action_dim。
    """

    def __init__(self, obs_dim, action_dim, qf_arch):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        in_dim = obs_dim + action_dim
        self.qf0 = _build_mlp(in_dim, qf_arch, out_dim=1)
        self.qf1 = _build_mlp(in_dim, qf_arch, out_dim=1)

    def forward(self, obs, action):
        q_input = torch.cat([self.flatten(obs), action], dim=1)
        return self.qf0(q_input), self.qf1(q_input)

    def q1_forward(self, obs, action):
        """仅计算第一个 Q 网络 qf0(s, a)，用于计算确定性策略梯度更新 Actor。"""
        q_input = torch.cat([self.flatten(obs), action], dim=1)
        return self.qf0(q_input)


class TD3Policy(nn.Module):
    """
    TD3 完整六网络容器：包含 actor, actor_target, critic (qf0+qf1), critic_target (qf0+qf1)。
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
        self.actor_target = Actor(obs_dim, action_dim, pi_arch, action_low, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def predict(self, obs, deterministic = True):
        """推理接口：直接使用 Actor 输出确定性连续动作 μ(s)。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        return action, None


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
# CleanRL 白盒组件 4：TD3 算法主类（与 SB3 `TD3` 接口 1:1 完全对齐）
# ==============================================================================
class TD3:
    """
    CleanRL 白盒实现的 TD3 算法类（接口与 `stable_baselines3.TD3` 完全一致）：
    包含目标策略平滑 (Target Policy Smoothing)、截断双 Q 更新 (Clipped Double Q-Learning)
    与延迟策略软更新 (Delayed Policy Updates)。
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate = 1e-3,
        gamma = 0.99,
        buffer_size = 100_000,
        learning_starts = 1_000,
        batch_size = 100,
        train_freq = (1, "episode"),
        gradient_steps = -1,
        tau = 0.005,
        action_noise = None,
        policy_delay = 2,
        target_policy_noise = 0.2,
        target_noise_clip = 0.5,
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
        self.action_noise = action_noise
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
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

        self.policy = TD3Policy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_low=self.action_low,
            action_high=self.action_high,
            net_arch=self.net_arch,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

    def predict(self, obs, deterministic = True):
        return self.policy.predict(obs, deterministic=deterministic)

    def _update_td3(
        self,
        num_grad_steps,
        total_grad_updates,
        low_tensor,
        high_tensor,
    ):
        """
        TD3 批量梯度更新：
        1. 目标策略平滑 (Target Policy Smoothing)
        2. 截断双 Q 学习 (Clipped Double Q-Learning) 更新双 Critic
        3. 延迟策略更新 (Delayed Policy Updates)：每 policy_delay 步更新一次 Actor 并软更新目标网络
        """
        actor_losses = []
        critic_losses = []

        for _ in range(num_grad_steps):
            total_grad_updates += 1
            batch = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # 【TD3 改进 3：目标策略平滑】给目标动作加截断高斯噪声，平滑 Q 函数在动作维度的局部尖峰
                clipped_noise = (torch.randn_like(batch.actions) * self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = (self.policy.actor_target(batch.next_observations) + clipped_noise).clamp(
                    low_tensor, high_tensor
                )

                # 【TD3 改进 1：截断双 Q 学习】取两个目标 Critic 预测值的较小者，遏制 Q 值高估
                target_q1, target_q2 = self.policy.critic_target(batch.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2)
                td_target = batch.rewards + (1.0 - batch.dones) * self.gamma * target_q

            # 更新两个主 Critic 网络 (qf0 与 qf1)
            current_q1, current_q2 = self.policy.critic(batch.observations, batch.actions)
            critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

            # 【TD3 改进 2：延迟策略更新】每隔 policy_delay=2 步才更新一次 Actor 与 Target 网络
            if total_grad_updates % self.policy_delay == 0:
                actor_loss = -self.policy.critic.q1_forward(
                    batch.observations, self.policy.actor(batch.observations)
                ).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                soft_update(self.policy.critic, self.policy.critic_target, self.tau)
                soft_update(self.policy.actor, self.policy.actor_target, self.tau)

        return total_grad_updates, actor_losses, critic_losses

    def learn(
        self,
        total_timesteps,
        callback = None,
        tb_log_name = "TD3",
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
        episode_steps = 0
        total_grad_updates = 0
        start_time = time.time()
        low_tensor = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
        high_tensor = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)

        is_episode_freq = isinstance(self.train_freq, tuple) and self.train_freq[1] == "episode"
        step_freq = self.train_freq[0] if isinstance(self.train_freq, tuple) else int(self.train_freq)

        for global_step in range(1, total_timesteps + 1):
            episode_steps += 1

            # 1. 选择连续动作：预热期纯随机采样；正式期使用 Actor 输出并叠加高斯探索噪声
            if global_step <= self.learning_starts:
                action = self.env.action_space.sample()
            else:
                det_action, _ = self.policy.predict(obs, deterministic=True)
                noise = self.action_noise() if self.action_noise is not None else 0.0
                action = np.clip(det_action + noise, self.action_low, self.action_high)

            # 2. 环境交互并存入 ReplayBuffer（仅将真正终止 terminated 记为 done）
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.add(obs, next_obs, action, float(reward), terminated)
            obs = next_obs

            # 按步更新模式（若 train_freq 为整数步数）
            if not is_episode_freq and global_step > self.learning_starts and global_step % step_freq == 0:
                n_grads = step_freq if self.gradient_steps == -1 else self.gradient_steps
                total_grad_updates, actor_losses, critic_losses = self._update_td3(
                    num_grad_steps=n_grads,
                    total_grad_updates=total_grad_updates,
                    low_tensor=low_tensor,
                    high_tensor=high_tensor,
                )

            # 3. 回合结束时记录日志，并在 train_freq=(1, "episode") 模式下触发批量梯度更新
            if terminated or truncated:
                if "episode" in info:
                    recent_ep_rewards.append(float(info["episode"]["r"]))
                    recent_ep_lengths.append(int(info["episode"]["l"]))
                    if writer is not None:
                        writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                        writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)

                if is_episode_freq and global_step > self.learning_starts:
                    n_grads = episode_steps if self.gradient_steps == -1 else self.gradient_steps
                    total_grad_updates, actor_losses, critic_losses = self._update_td3(
                        num_grad_steps=n_grads,
                        total_grad_updates=total_grad_updates,
                        low_tensor=low_tensor,
                        high_tensor=high_tensor,
                    )
                    if writer is not None:
                        if critic_losses:
                            writer.add_scalar("train/critic_loss", float(np.mean(critic_losses)), global_step)
                        if actor_losses:
                            writer.add_scalar("train/actor_loss", float(np.mean(actor_losses)), global_step)
                        writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

                obs, _ = self.env.reset()
                episode_steps = 0

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
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
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
            learning_rate=checkpoint.get("learning_rate", 1e-3),
            gamma=checkpoint.get("gamma", 0.99),
            buffer_size=checkpoint.get("buffer_size", 100_000),
            learning_starts=checkpoint.get("learning_starts", 1_000),
            batch_size=checkpoint.get("batch_size", 100),
            train_freq=checkpoint.get("train_freq", (1, "episode")),
            gradient_steps=checkpoint.get("gradient_steps", -1),
            tau=checkpoint.get("tau", 0.005),
            policy_delay=checkpoint.get("policy_delay", 2),
            target_policy_noise=checkpoint.get("target_policy_noise", 0.2),
            target_noise_clip=checkpoint.get("target_noise_clip", 0.5),
            policy_kwargs=checkpoint.get("policy_kwargs", dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))),
            verbose=0,
            _obs_shape=checkpoint.get("obs_shape"),
            _action_dim=checkpoint.get("action_dim"),
            _action_low=checkpoint.get("action_low"),
            _action_high=checkpoint.get("action_high"),
        )
        model.policy.load_state_dict(checkpoint["state_dict"])
        model.policy.eval()
        return model


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 TD3 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（Pendulum-v1 在 20000 步左右即可取得优异表现）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "Pendulum-v1"

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "td3_pendulum"
    log_dir = base_dir / "logs" / "td3_pendulum"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：环境初始化与连续动作空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
    train_env = Monitor(gym.make(env_id))
    eval_env = Monitor(gym.make(env_id))

    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  动作维度 (n_actions):         {train_env.action_space.shape[-1]}")
    print(f"  动作下界 (Low):               {train_env.action_space.low}")
    print(f"  动作上界 (High):              {train_env.action_space.high}")

    # ==========================================================================
    # 第 2 步：配置探索噪声、网络结构与 TD3 模型
    # ==========================================================================
    print("\n[第 2 步] 配置动作探索噪声与 TD3 核心超参数...")

    # 获取连续动作的维度（Pendulum-v1 为 1 维）
    n_actions = train_env.action_space.shape[-1]
    # 构造独立同分布的高斯动作噪声：均值为 0，标准差为 0.1（相对于动作范围 [-2, 2] 来说探索幅度适中）
    # SB3 在 model.learn() 期间会自动把该噪声加到 Actor 输出的动作上（并裁剪到合法动作范围）
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    # 对 Off-Policy Actor-Critic 算法（TD3 / SAC），net_arch 字典中：
    # - pi 代表 Actor（策略网络）的隐藏层结构
    # - qf 代表 Critic（Q 价值网络）的隐藏层结构（注意不是 vf，因为 TD3 学的是 Q(s,a)）
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = TD3(
        policy="MlpPolicy",
        env=train_env,

        # --- 学习率、折扣与经验回放池 ---
        learning_rate=1e-3,          # 学习率（Pendulum 常用 1e-3，复杂 MuJoCo 环境常用 3e-4）
        gamma=0.99,                  # 折扣因子 γ
        buffer_size=100_000,         # 经验回放池大小
        learning_starts=1_000,       # 预热步数：前 1000 步纯粹随机采样填充经验池，之后开始梯度更新
        batch_size=100,              # 每次更新采样的批大小（TD3 论文默认 100 或 256）

        # --- 训练频率与软更新 ---
        train_freq=(1, "episode"),   # 每结束 1 个完整回合（或设为 1 表示每 1 步）执行一次更新
        gradient_steps=-1,           # -1 表示本次收集了多少步环境数据，就做多少次梯度更新（与 train_freq=(1,"episode") 配合极佳）
        tau=0.005,                   # 目标网络软更新系数（Polyak Averaging）：θ_target = τ*θ + (1-τ)*θ_target

        # --- TD3 专属核心超参数 ---
        action_noise=action_noise,   # 探索噪声对象（极其重要！不传则确定性策略几乎不探索）
        policy_delay=2,              # 延迟策略更新：Critic 更新 2 次，Actor 与目标网络才更新 1 次
        target_policy_noise=0.2,     # 目标策略平滑噪声标准差：计算目标 Q 值时给目标动作加的高斯噪声
        target_noise_clip=0.5,       # 目标策略平滑噪声裁剪边界 [-0.5, 0.5]

        # --- 网络结构与日志 ---
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
    print(f"\n[第 4 步] 开始训练 TD3（总步数 total_timesteps={args.timesteps}）...")
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
        tb_log_name="TD3_Pendulum",
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
    final_model_path = model_dir / "td3_pendulum_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    del model
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = TD3.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合...")
    demo_env = gym.make(env_id)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # 推理时 deterministic=True，直接使用 Actor 输出的确定性连续动作（不加探索噪声）
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        if terminated or truncated:
            # Pendulum-v1 没有失败终止状态，总是在 200 步时由时间上限触发 truncated=True
            reason = "达到 200 步最大步数截断 (truncated)" if truncated else "任务终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
