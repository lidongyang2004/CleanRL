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
import shutil
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


def parse_args():
    """解析命令行参数（与 SB3 04_sac_pendulum.py 完全一致）。"""
    parser = argparse.ArgumentParser(description="CleanRL SAC 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（SAC 样本效率极高，20000 步内通常可达到非常优秀的控制效果）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed):
    """固定随机种子，保证实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_dir(dir_path):
    """清空并重建目录，确保每次运行只保留最新一次记录或模型。"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


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
def _build_mlp(in_dim, hidden_dims, out_dim=None):
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

    def __init__(self, obs_dim, action_dim, pi_arch, action_low, action_high):
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

    def __init__(self, obs_dim, action_dim, action_low, action_high, net_arch):
        super().__init__()
        pi_arch = net_arch.get("pi", [256, 256])
        qf_arch = net_arch.get("qf", [256, 256])

        self.actor = Actor(obs_dim, action_dim, pi_arch, action_low, action_high)
        self.critic = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def predict(self, obs, deterministic=True):
        """与 SB3 完全一致的推理接口：deterministic=True 时返回 tanh(μ(s)) 均值动作。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if deterministic:
                action = self.actor.get_deterministic_action(obs_tensor)
            else:
                action, _ = self.actor.get_action_and_log_prob(obs_tensor)
            return action.squeeze(0).cpu().numpy(), None


def evaluate_policy(
    policy,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    return_episode_rewards=False,
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
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def maybe_evaluate_and_save(policy, eval_env, global_step, eval_state, log_dir, writer):
    """定期在独立 eval_env 上评估当前策略，记录 evaluations.npz 并在创新高时保存 best_model.pt。"""
    ep_rewards, ep_lengths = evaluate_policy(
        policy, eval_env, n_eval_episodes=5, deterministic=True, return_episode_rewards=True
    )
    eval_mean = float(np.mean(ep_rewards))
    eval_std = float(np.std(ep_rewards))

    eval_state["timesteps"].append(global_step)
    eval_state["results"].append(ep_rewards)
    eval_state["ep_lengths"].append(ep_lengths)
    np.savez(
        log_dir / "evaluations.npz",
        timesteps=np.array(eval_state["timesteps"]),
        results=np.array(eval_state["results"]),
        ep_lengths=np.array(eval_state["ep_lengths"]),
    )
    writer.add_scalar("eval/mean_reward", eval_mean, global_step)
    writer.add_scalar("eval/mean_ep_length", float(np.mean(ep_lengths)), global_step)
    print(f"  [Eval @ {global_step:>6d} steps] episode_reward={eval_mean:.2f} +/- {eval_std:.2f}")

    if eval_mean > eval_state["best_mean_reward"]:
        eval_state["best_mean_reward"] = eval_mean
        torch.save(policy.state_dict(), eval_state["best_model_path"])
        print(f"    --> 刷新最高评估记录！已保存最优模型至: {eval_state['best_model_path']}")


def update_sac_step(
    policy,
    actor_optimizer,
    critic_optimizer,
    ent_coef_optimizer,
    log_ent_coef,
    target_entropy,
    replay_buffer,
    batch_size,
    gamma,
    tau,
):
    """
    SAC 单次四阶段梯度更新：
    (a) 自动调节最大熵温度系数 α (ent_coef)
    (b) 用 Soft 状态价值 min(Q1', Q2') - α * log π(a'|s') 更新双 Critic 网络
    (c) 最小化 α * log π(a|s) - min(Q1, Q2) 更新随机策略 Actor
    (d) Polyak 软更新目标 Critic 网络
    """
    batch = replay_buffer.sample(batch_size)
    sampled_actions, log_prob = policy.actor.get_action_and_log_prob(batch.observations)

    # --- (a) 自动调节最大熵温度系数 α ---
    ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
    ent_coef_optimizer.zero_grad()
    ent_coef_loss.backward()
    ent_coef_optimizer.step()
    ent_coef = log_ent_coef.exp().detach()

    # --- (b) 更新截断双 Q 价值网络 (ContinuousCritic) ---
    with torch.no_grad():
        next_actions, next_log_prob = policy.actor.get_action_and_log_prob(batch.next_observations)
        target_q1, target_q2 = policy.critic_target(batch.next_observations, next_actions)
        min_qf_next_target = torch.min(target_q1, target_q2) - ent_coef * next_log_prob
        next_q_value = batch.rewards + (1.0 - batch.dones) * gamma * min_qf_next_target

    current_q1, current_q2 = policy.critic(batch.observations, batch.actions)
    critic_loss = 0.5 * (F.mse_loss(current_q1, next_q_value) + F.mse_loss(current_q2, next_q_value))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # --- (c) 更新随机策略网络 (Actor) ---
    q1_pi, q2_pi = policy.critic(batch.observations, sampled_actions)
    min_qf_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # --- (d) Polyak 软更新目标 Critic 网络 ---
    soft_update(policy.critic, policy.critic_target, tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "ent_coef": ent_coef.item(),
        "ent_coef_loss": ent_coef_loss.item(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    env_id = "Pendulum-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "sac_pendulum"
    log_dir = base_dir / "logs" / "sac_pendulum"
    reset_dir(model_dir)
    reset_dir(log_dir)

    # ==========================================================================
    # 第 1 步：环境初始化与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
    eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  动作范围:                     [{train_env.action_space.low[0]}, {train_env.action_space.high[0]}]")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置（CleanRL 白盒实例化）
    # ==========================================================================
    print("\n[第 2 步] 初始化 SAC 模型与核心超参数...")

    # 同样支持分别为 Actor (pi) 和 Twin Critic (qf) 指定隐藏层结构
    net_arch = dict(pi=[256, 256], qf=[256, 256])

    # --- 优化器与折扣因子（与 SB3 04_sac_pendulum.py 严格 1:1 对齐）---
    learning_rate = 3e-4       # SAC 最经典的默认学习率（对大多数连续控制任务都很稳健）
    gamma = 0.99               # 折扣因子 γ

    # --- 经验回放池 (Replay Buffer) ---
    buffer_size = 100_000      # 经验池容量
    learning_starts = 1_000    # 预热步数：先随机探索 1000 步积累基础样本再开始更新
    batch_size = 256           # SAC 推荐稍大的 batch_size（如 256）以降低梯度方差

    # --- 训练节奏与双 Q 目标网络软更新 ---
    train_freq = 1             # 每与环境交互 1 步，就做一次训练（高频更新带来极高的样本效率）
    gradient_steps = 1         # 每次训练执行 1 步梯度下降
    tau = 0.005                # 软更新系数：目标 Critic 网络以 0.5% 的步幅缓慢追踪主 Critic 网络

    obs_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(train_env.action_space.shape[-1])
    action_low = train_env.action_space.low
    action_high = train_env.action_space.high

    policy = SACPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        net_arch=net_arch,
    ).to(device)

    # --- SAC 核心：最大熵温度系数自动调节 (ent_coef="auto", target_entropy="auto") ---
    # 目标熵下界默认为 -dim(A)（此处为 -1.0）；优化对数形式 log_ent_coef 保证 α = exp(log_ent_coef) > 0
    target_entropy = -float(action_dim)
    log_ent_coef = torch.zeros(1, device=device, requires_grad=True)

    # 三个独立 Adam 优化器：分别优化 Actor、Twin Critic 和对数熵温度参数 log_ent_coef
    actor_optimizer = optim.Adam(policy.actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(policy.critic.parameters(), lr=learning_rate)
    ent_coef_optimizer = optim.Adam([log_ent_coef], lr=learning_rate)

    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        obs_shape=train_env.observation_space.shape,
        action_dim=action_dim,
        device=device,
    )

    # ==========================================================================
    # 第 3 步：训练前基线评估
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估...")
    mean_reward_before, std_reward_before = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：启动 CleanRL SAC 白盒训练主循环与定期评估
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 SAC（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, args.timesteps // 5)
    eval_state = {
        "best_mean_reward": -float("inf"),
        "best_model_path": model_dir / "best_model.pt",
        "timesteps": [],
        "results": [],
        "ep_lengths": [],
    }

    tb_run_dir = log_dir / "SAC_Pendulum"
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    recent_ep_rewards = []
    recent_ep_lengths = []

    obs, _ = train_env.reset(seed=args.seed)
    start_time = time.time()

    for global_step in range(1, args.timesteps + 1):
        # 1. 动作采样：预热期纯随机探索；正式期从 Squashed Gaussian 随机策略中采样
        if global_step <= learning_starts:
            action = train_env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=False)

        # 2. 与环境交互一步并存入 ReplayBuffer
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        replay_buffer.add(obs, next_obs, action, float(reward), terminated)
        obs = next_obs

        if terminated or truncated:
            if "episode" in info:
                recent_ep_rewards.append(float(info["episode"]["r"]))
                recent_ep_lengths.append(int(info["episode"]["l"]))
                writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
            obs, _ = train_env.reset()

        # 3. 达到预热步数后，执行 SAC 四阶段梯度更新 (温度 α -> 双 Critic -> Actor -> 软更新)
        if global_step > learning_starts and global_step % train_freq == 0:
            metrics = None
            for _ in range(gradient_steps):
                metrics = update_sac_step(
                    policy=policy,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    ent_coef_optimizer=ent_coef_optimizer,
                    log_ent_coef=log_ent_coef,
                    target_entropy=target_entropy,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    tau=tau,
                )
            if metrics is not None and global_step % 100 == 0:
                writer.add_scalar("train/critic_loss", metrics["critic_loss"], global_step)
                writer.add_scalar("train/actor_loss", metrics["actor_loss"], global_step)
                writer.add_scalar("train/ent_coef", metrics["ent_coef"], global_step)
                writer.add_scalar("train/ent_coef_loss", metrics["ent_coef_loss"], global_step)
                writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

        # 4. 定期评估当前策略并保存最优模型
        if global_step % eval_freq == 0:
            maybe_evaluate_and_save(policy, eval_env, global_step, eval_state, log_dir, writer)

    writer.close()
    best_model_path = eval_state["best_model_path"]

    # ==========================================================================
    # 第 5 步：训练后定量评估
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "sac_pendulum_final.pt"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    torch.save(policy.state_dict(), final_model_path)

    del policy
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = SACPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        net_arch=net_arch,
    ).to(device)
    loaded_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    loaded_model.eval()

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
