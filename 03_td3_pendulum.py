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
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """解析命令行参数（与 SB3 03_td3_pendulum.py 完全一致）。"""
    parser = argparse.ArgumentParser(description="CleanRL TD3 求解 Pendulum-v1 连续控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="总训练步数（Pendulum-v1 在 20000 步左右即可取得优异表现）",
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
# CleanRL 白盒组件 1：高斯动作探索噪声与连续动作经验回放池
# 对应 SB3 `NormalActionNoise` 与 `ReplayBuffer`
# ==============================================================================
class NormalActionNoise:
    """独立同分布的高斯动作噪声（对应 SB3 `NormalActionNoise`）。"""

    def __init__(self, mean, sigma):
        self.mean = mean.astype(np.float32)
        self.sigma = sigma.astype(np.float32)

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
class Actor(nn.Module):
    """
    确定性策略网络 μ(s, θ)：
    最后一层接 Tanh() 将输出压缩至 [-1, 1]，再通过仿射变换还原至物理环境真实量程 [low, high]。
    """

    def __init__(self, obs_dim, action_dim, pi_arch, action_low, action_high):
        super().__init__()
        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)
        layers = []
        in_dim = obs_dim
        for h_dim in pi_arch:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.mu = nn.Sequential(*layers)

        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs):
        features = self.features_extractor(obs)
        scaled_tanh = self.mu(features)
        return scaled_tanh * self.action_scale + self.action_bias


class ContinuousCritic(nn.Module):
    """
    截断双 Q 价值网络组 (qf0 与 qf1)：
    输入为状态向量 s 与动作向量 a 的拼接 torch.cat([obs, action], dim=1)，维度为 dim(S) + dim(A)。
    """

    def __init__(self, obs_dim, action_dim, qf_arch):
        super().__init__()
        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)

        def build_q_net():
            layers = []
            in_dim = obs_dim + action_dim
            for h_dim in qf_arch:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            return nn.Sequential(*layers)

        self.qf0 = build_q_net()
        self.qf1 = build_q_net()

    def forward(self, obs, action):
        features = self.features_extractor(obs)
        q_input = torch.cat([features, action], dim=1)
        return self.qf0(q_input), self.qf1(q_input)

    def q1_forward(self, obs, action):
        """仅计算第一个 Q 网络 qf0(s, a)，用于计算确定性策略梯度更新 Actor。"""
        features = self.features_extractor(obs)
        q_input = torch.cat([features, action], dim=1)
        return self.qf0(q_input)


class TD3Policy(nn.Module):
    """
    TD3 完整六网络容器：包含 actor, actor_target, critic (qf0+qf1), critic_target (qf0+qf1)。
    """

    def __init__(self, obs_dim, action_dim, action_low, action_high, net_arch):
        super().__init__()
        pi_arch = net_arch.get("pi", [256, 256])
        qf_arch = net_arch.get("qf", [256, 256])

        self.actor = Actor(obs_dim, action_dim, pi_arch, action_low, action_high)
        self.actor_target = Actor(obs_dim, action_dim, pi_arch, action_low, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target = ContinuousCritic(obs_dim, action_dim, qf_arch)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def predict(self, obs, deterministic=True):
        """推理接口：直接使用 Actor 输出确定性连续动作 μ(s)。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        return action, None


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


def main():
    args = parse_args()
    set_seed(args.seed)
    env_id = "Pendulum-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "td3_pendulum"
    log_dir = base_dir / "logs" / "td3_pendulum"
    reset_dir(model_dir)
    reset_dir(log_dir)

    # ==========================================================================
    # 第 1 步：环境初始化与连续动作空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 连续控制环境: {env_id}")
    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
    eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  动作维度 (n_actions):         {train_env.action_space.shape[-1]}")
    print(f"  动作下界 (Low):               {train_env.action_space.low}")
    print(f"  动作上界 (High):              {train_env.action_space.high}")

    # ==========================================================================
    # 第 2 步：配置探索噪声、网络结构与 TD3 模型（CleanRL 白盒实例化）
    # ==========================================================================
    print("\n[第 2 步] 配置动作探索噪声与 TD3 核心超参数...")

    obs_dim = int(np.prod(train_env.observation_space.shape))
    n_actions = int(train_env.action_space.shape[-1])
    action_low = train_env.action_space.low
    action_high = train_env.action_space.high
    action_scale = float((action_high[0] - action_low[0]) / 2.0)

    # 构造独立同分布的高斯动作噪声：均值为 0，标准差为 0.1 * action_scale（对应 SB3 NormalActionNoise）
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * action_scale * np.ones(n_actions),
    )

    # 对 Off-Policy Actor-Critic 算法（TD3 / SAC），net_arch 字典中：
    # - pi 代表 Actor（策略网络）的隐藏层结构
    # - qf 代表 Critic（Q 价值网络）的隐藏层结构（注意不是 vf，因为 TD3 学的是 Q(s,a)）
    net_arch = dict(pi=[256, 256], qf=[256, 256])

    # --- 学习率、折扣与经验回放池（与 SB3 03_td3_pendulum.py 严格 1:1 对齐）---
    learning_rate = 1e-3          # 学习率（Pendulum 常用 1e-3，复杂 MuJoCo 环境常用 3e-4）
    gamma = 0.99                  # 折扣因子 γ
    buffer_size = 100_000         # 经验回放池大小
    learning_starts = 1_000       # 预热步数：前 1000 步纯粹随机采样填充经验池，之后开始梯度更新
    batch_size = 100              # 每次更新采样的批大小（TD3 论文默认 100 或 256）

    # --- 训练频率与软更新 ---
    # 对应 SB3 train_freq=(1, "episode"), gradient_steps=-1：每结束 1 个回合，收集了多少步就做多少次梯度更新
    tau = 0.005                   # 目标网络软更新系数（Polyak Averaging）：θ_target = τ*θ + (1-τ)*θ_target

    # --- TD3 专属核心超参数 ---
    policy_delay = 2              # 延迟策略更新：Critic 更新 2 次，Actor 与目标网络才更新 1 次
    target_policy_noise = 0.2 * action_scale   # 目标策略平滑噪声标准差
    target_noise_clip = 0.5 * action_scale     # 目标策略平滑噪声裁剪边界

    policy = TD3Policy(
        obs_dim=obs_dim,
        action_dim=n_actions,
        action_low=action_low,
        action_high=action_high,
        net_arch=net_arch,
    ).to(device)
    actor_optimizer = optim.Adam(policy.actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(policy.critic.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        obs_shape=train_env.observation_space.shape,
        action_dim=n_actions,
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
    # 第 4 步：启动 CleanRL TD3 白盒训练主循环与定期评估
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 TD3（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, args.timesteps // 5)
    best_mean_reward = -float("inf")
    best_model_path = model_dir / "best_model.pt"

    tb_run_dir = log_dir / "TD3_Pendulum"
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    eval_timesteps = []
    eval_results = []
    eval_ep_lengths = []
    recent_ep_rewards = []
    recent_ep_lengths = []

    obs, _ = train_env.reset(seed=args.seed)
    episode_steps = 0
    total_grad_updates = 0
    start_time = time.time()
    low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=device)

    for global_step in range(1, args.timesteps + 1):
        episode_steps += 1

        # 1. 选择连续动作：预热期纯随机采样；正式期使用 Actor 输出并叠加高斯探索噪声
        if global_step <= learning_starts:
            action = train_env.action_space.sample()
        else:
            det_action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(det_action + action_noise(), action_low, action_high)

        # 2. 环境交互并存入 ReplayBuffer（注意仅将真正终止 terminated 记为 done）
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        replay_buffer.add(obs, next_obs, action, float(reward), terminated)
        obs = next_obs

        # 3. 当回合结束时（对应 SB3 train_freq=(1, "episode"), gradient_steps=-1）触发批量梯度更新
        if terminated or truncated:
            if "episode" in info:
                recent_ep_rewards.append(float(info["episode"]["r"]))
                recent_ep_lengths.append(int(info["episode"]["l"]))
                writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)

            if global_step > learning_starts:
                actor_losses = []
                critic_losses = []
                for _ in range(episode_steps):
                    total_grad_updates += 1
                    batch = replay_buffer.sample(batch_size)

                    with torch.no_grad():
                        # 【TD3 改进 3：目标策略平滑 (Target Policy Smoothing)】
                        # 给目标 Actor 输出的下一动作加上截断高斯噪声，平滑 Q 函数在动作维度的局部尖峰
                        clipped_noise = (
                            torch.randn_like(batch.actions) * target_policy_noise
                        ).clamp(-target_noise_clip, target_noise_clip)
                        next_actions = (policy.actor_target(batch.next_observations) + clipped_noise).clamp(
                            low_tensor, high_tensor
                        )

                        # 【TD3 改进 1：截断双 Q 学习 (Clipped Double Q-Learning)】
                        # 取两个目标 Critic 网络预测值的较小者，遏制 Q 值高估
                        target_q1, target_q2 = policy.critic_target(batch.next_observations, next_actions)
                        target_q = torch.min(target_q1, target_q2)
                        td_target = batch.rewards + (1.0 - batch.dones) * gamma * target_q

                    # 更新两个主 Critic 网络 (qf0 与 qf1)
                    current_q1, current_q2 = policy.critic(batch.observations, batch.actions)
                    critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    critic_losses.append(critic_loss.item())

                    # 【TD3 改进 2：延迟策略更新 (Delayed Policy Updates)】
                    # 每隔 policy_delay=2 步才更新一次 Actor 与所有 Target 网络
                    if total_grad_updates % policy_delay == 0:
                        actor_loss = -policy.critic.q1_forward(
                            batch.observations, policy.actor(batch.observations)
                        ).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        actor_losses.append(actor_loss.item())

                        # Polyak 软更新目标网络：θ_target ← τ * θ + (1 - τ) * θ_target
                        for param, target_param in zip(
                            policy.critic.parameters(), policy.critic_target.parameters()
                        ):
                            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                        for param, target_param in zip(
                            policy.actor.parameters(), policy.actor_target.parameters()
                        ):
                            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                if critic_losses:
                    writer.add_scalar("train/critic_loss", float(np.mean(critic_losses)), global_step)
                if actor_losses:
                    writer.add_scalar("train/actor_loss", float(np.mean(actor_losses)), global_step)
                writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

            obs, _ = train_env.reset()
            episode_steps = 0

        # 4. 定期评估当前策略并保存最优模型（对应 SB3 EvalCallback）
        if global_step % eval_freq == 0:
            ep_rewards, ep_lengths = evaluate_policy(
                policy, eval_env, n_eval_episodes=5, deterministic=True, return_episode_rewards=True
            )
            eval_mean = float(np.mean(ep_rewards))
            eval_std = float(np.std(ep_rewards))
            eval_timesteps.append(global_step)
            eval_results.append(ep_rewards)
            eval_ep_lengths.append(ep_lengths)
            np.savez(
                log_dir / "evaluations.npz",
                timesteps=np.array(eval_timesteps),
                results=np.array(eval_results),
                ep_lengths=np.array(eval_ep_lengths),
            )
            writer.add_scalar("eval/mean_reward", eval_mean, global_step)
            writer.add_scalar("eval/mean_ep_length", float(np.mean(ep_lengths)), global_step)
            print(f"  [Eval @ {global_step:>6d} steps] episode_reward={eval_mean:.2f} +/- {eval_std:.2f}")
            if eval_mean > best_mean_reward:
                best_mean_reward = eval_mean
                torch.save(policy.state_dict(), best_model_path)
                print(f"    --> 刷新最高评估记录！已保存最优模型至: {best_model_path}")

    writer.close()

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
    final_model_path = model_dir / "td3_pendulum_final.pt"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    torch.save(policy.state_dict(), final_model_path)

    del policy
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = TD3Policy(
        obs_dim=obs_dim,
        action_dim=n_actions,
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
