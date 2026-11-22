"""
================================================================================
CleanRL 实战教程 02b：PPO (Proximal Policy Optimization) 求解 Pendulum-v1
================================================================================

【算法定位】
- 算法名称：PPO (近端策略优化 —— 连续动作空间 CleanRL 单文件 PyTorch 白盒实现)
- 策略类型：On-Policy（同策略）—— 每次使用当前策略并行采样一批轨迹，更新后即清空丢弃
- 学习范式：随机策略 Actor-Critic（Actor 输出连续动作的高斯概率分布，Critic 估计状态价值 V(s)）
- 支持动作空间：本脚本聚焦 **连续动作空间 (Box)**，与 02_ppo_cartpole.py（离散动作空间 Discrete）形成对照

【核心揭秘：同一套 PPO 如何从离散动作切换到连续动作？】
很多初学者好奇：为什么不用改 PPO 的核心损失函数（GAE 优势估计 + Clip 裁剪），
仅仅换个环境，PPO 就能从“左右推车 (Discrete)”变成“连续力矩控制 (Box)”？
秘密全部在于 Actor 策略网络最后一层的 **动作概率分布 (Action Distribution)**：
1. 离散空间 (02_ppo_cartpole.py, Discrete(2))：
   - Actor 输出每个离散动作的 Logits，经 Softmax 构成分类分布 `CategoricalDistribution`。
   - 采样得到整数动作 a ∈ {0, 1}，计算其离散对数概率 log π(a|s)。
2. 连续空间 (02b_ppo_pendulum.py, Box(-2.0, 2.0, (1,)))：
   - 连续区间 [-2.0, 2.0] 内有无穷多个实数，无法穷举概率。
   - CleanRL / SB3 将 Actor 切换为 **对角高斯分布 (`DiagGaussianDistribution`)**：
     * 均值网络 μ_θ(s)：由 Actor 神经网络根据当前状态 s 动态输出每个动作维度的均值；
     * 对数标准差 `log_std` (log σ)：默认是一个与状态无关的可学习参数向量（初始值为 0，即标准差 σ = exp(0) = 1.0）。
       训练初期 σ 较大以充分探索，随着训练推进，梯度下降会自动调小 `log_std`，使策略收敛到精准控制！
   - 采样与更新：训练时从高斯分布 N(μ_θ(s), σ^2) 中采样连续动作 a，并用高斯概率密度函数计算对数似然 log π_θ(a|s)。
     只要能算出 log π_θ(a|s) 和策略熵 Entropy，后续的新旧策略概率比 r_t(θ) = exp(log π_new - log π_old)
     以及 Clipped Surrogate Loss 就与离散版本 **完全一模一样**！

【进阶特性：gSDE（广义状态依赖探索，use_sde=True）】
- 标准高斯策略在每个时间步独立采样高斯白噪声，在高频连续控制（如真实机械臂电机控制）中容易导致动作高频抖动。
- 开启 `--use-sde` 后，本脚本白盒实现了 `StateDependentNoiseDistribution` (gSDE)：
  噪声标准差直接依赖于当前状态隐藏特征 h_π(s)，且在若干步（`sde_sample_freq=4`）内固定探索噪声矩阵，使探索轨迹更加平滑高效。

【On-Policy (PPO) vs Off-Policy (TD3 / SAC) 连续控制对比】
- TD3 (03) 和 SAC (04) 拥有经验回放池 (ReplayBuffer)，历史数据可反复利用，在 Pendulum-v1 上 20,000 步即可收敛；
- PPO 作为 On-Policy 算法，每次更新后必须丢弃旧数据，样本效率相对较低，因此通常需要更多交互步数（如 100,000 步），
  但得益于并行环境采样与极小的单步计算开销，实际训练墙钟时间（Wall-clock time）依然非常快！

运行方式：
    python 02b_ppo_pendulum.py                           # 默认标准对角高斯策略 (100000 步)
    python 02b_ppo_pendulum.py --use-sde                 # 开启 gSDE 广义状态依赖探索
    python 02b_ppo_pendulum.py --timesteps 20000         # 快速测试全流程
================================================================================
"""

import argparse
import random
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """解析命令行参数（与 SB3 02b_ppo_pendulum.py 完全一致）。"""
    parser = argparse.ArgumentParser(description="CleanRL PPO 求解 Pendulum-v1 连续动作控制示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="总训练步数（On-Policy 样本效率低于 SAC/TD3，默认 100000 步，4 并行环境下约需 25000 次并行 step）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量（默认 4）")
    parser.add_argument(
        "--use-sde",
        action="store_true",
        help="是否开启 gSDE（广义状态依赖探索 State-Dependent Exploration）",
    )
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


def make_vec_env(env_id, n_envs, seed):
    """创建 n_envs 个并行的 Gymnasium 连续控制环境（SAME_STEP 自动重置模式）。"""

    def make_single_env(rank):
        def _init():
            env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
            env.reset(seed=seed + rank)
            return env

        return _init

    vec_kwargs = {}
    if hasattr(gym.vector, "AutoresetMode"):
        vec_kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    return gym.vector.SyncVectorEnv([make_single_env(i) for i in range(n_envs)], **vec_kwargs)


# ==============================================================================
# CleanRL 白盒组件 1：连续动作概率分布 (DiagGaussian vs gSDE) 与 ActorCriticPolicy
# 严格对应 SB3 `DiagGaussianDistribution` 与 `StateDependentNoiseDistribution`
# ==============================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化：隐藏层 gain=sqrt(2)，均值输出层 action_net gain=0.01，价值输出层 value_net gain=1.0。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(in_dim, hidden_dims, out_dim=None, out_std=1.0):
    """构建带 Tanh 激活与正交初始化的多层感知机（当 out_dim=None 时仅返回隐藏层特征主干）。"""
    layers = []
    for h_dim in hidden_dims:
        layers.append(layer_init(nn.Linear(in_dim, h_dim), std=np.sqrt(2)))  # 隐藏层用 sqrt(2) 补偿 Tanh 压缩
        layers.append(nn.Tanh())
        in_dim = h_dim
    if out_dim is not None:
        layers.append(layer_init(nn.Linear(in_dim, out_dim), std=out_std))
    return nn.Sequential(*layers)


class DiagGaussianDistribution:
    """标准对角高斯分布 (use_sde=False)：标准差 σ = exp(log_std) 独立于状态。"""

    def get_std(self, log_std):
        return torch.exp(log_std)


class StateDependentNoiseDistribution:
    """
    广义状态依赖探索分布 (gSDE, use_sde=True)：
    噪声权重矩阵 ε_θ ~ N(0, σ^2) 在若干步内保持固定，探索扰动为 h_π(s) @ ε_θ，
    对应条件高斯分布的方差为 Σ(s) = (h_π(s)^2) @ (σ^2)。
    """

    def __init__(self, latent_sde_dim, action_dim, epsilon=1e-6):
        self.latent_sde_dim = latent_sde_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.exploration_mat = None
        self.exploration_matrices = None

    def get_std(self, log_std):
        return torch.exp(log_std)

    def sample_weights(self, log_std, batch_size=1):
        std = self.get_std(log_std)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde, log_std):
        latent_sde = latent_sde.detach()
        if self.exploration_mat is None or self.exploration_matrices is None:
            self.sample_weights(log_std, batch_size=len(latent_sde))
        assert self.exploration_mat is not None and self.exploration_matrices is not None
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices).squeeze(1)


class ActorCriticPolicy(nn.Module):
    """
    连续动作空间 Actor-Critic 策略网络（同时支持标准对角高斯与 gSDE 状态依赖探索）：
    - self.actor_backbone + self.actor_mean: (batch, obs_dim) -> (batch, 64) -> (batch, action_dim) 输出高斯均值 μ(s)
    - self.critic: (batch, obs_dim) -> (batch, 1) 输出状态价值估计 v(s, w)
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        net_arch,
        use_sde=False,
    ):
        super().__init__()
        pi_arch = net_arch.get("pi", [64, 64])
        vf_arch = net_arch.get("vf", [64, 64])
        self.use_sde = use_sde
        self.action_dim = action_dim
        self.latent_dim_pi = pi_arch[-1]

        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.actor_backbone = _build_mlp(obs_dim, pi_arch, out_dim=None)
        self.actor_mean = layer_init(nn.Linear(self.latent_dim_pi, action_dim), std=0.01)  # 初始均值接近 0
        self.critic = _build_mlp(obs_dim, vf_arch, out_dim=1, out_std=1.0)

        if self.use_sde:
            # gSDE 模式下：log_std 是形状为 (64, action_dim) 的状态特征噪声矩阵参数（初始为 0.0）
            self.action_dist = StateDependentNoiseDistribution(self.latent_dim_pi, action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.latent_dim_pi, action_dim), requires_grad=True)
            self.action_dist.sample_weights(self.log_std, batch_size=1)
        else:
            # 标准对角高斯模式下：log_std 是形状为 (action_dim,) 的独立可学习参数向量（初始为 0.0）
            self.action_dist = DiagGaussianDistribution()
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def reset_noise(self, batch_size=1):
        """在 gSDE 模式下重新采样状态依赖噪声矩阵 ε_θ。"""
        if self.use_sde and isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_value(self, obs):
        """对应伪代码价值函数 v(s, w_k)：仅计算状态价值，形状 (batch_size, 1)。"""
        return self.critic(self.flatten(obs))

    def get_action_and_value(self, obs, action=None):
        """
        连续动作四合一前向接口（同时覆盖伪代码第 1 步采样与第 3 步重评估）：
        返回 (未裁剪原始高斯动作 action, 对数似然 log_prob, 分布熵 entropy, 状态价值 value)。
        """
        features = self.flatten(obs)
        latent_pi = self.actor_backbone(features)
        mean_actions = self.actor_mean(latent_pi)  # 高斯分布均值 μ(s)

        if self.use_sde and isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde = latent_pi.detach()
            std_mat = self.action_dist.get_std(self.log_std)
            variance = torch.mm(latent_sde**2, std_mat**2)
            dist = Normal(mean_actions, torch.sqrt(variance + 1e-6))
            if action is None:
                noise = self.action_dist.get_noise(latent_sde, self.log_std)
                action = mean_actions + noise
        else:
            action_std = torch.exp(self.log_std).expand_as(mean_actions)  # 标准差 σ = exp(log_std)
            dist = Normal(mean_actions, action_std)
            if action is None:
                action = dist.sample()

        # 多维独立高斯分布的联合对数概率等于各维度 log_prob 之和
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.critic(features)

    def predict(self, obs, deterministic=True):
        """
        连续动作评估/演示推理接口：
        - deterministic=True 时直接输出高斯均值 μ(s) 并裁剪至环境动作边界 [low, high]；
        - deterministic=False 时从高斯分布采样并裁剪。
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            single_env = obs_tensor.ndim == 1
            if single_env:
                obs_tensor = obs_tensor.unsqueeze(0)

            if deterministic:
                raw_actions = self.actor_mean(self.actor_backbone(self.flatten(obs_tensor)))
            else:
                raw_actions, _, _, _ = self.get_action_and_value(obs_tensor)

            clipped_actions = torch.clamp(raw_actions, self.action_low, self.action_high)
            if single_env:
                return clipped_actions.squeeze(0).cpu().numpy(), None
            return clipped_actions.cpu().numpy(), None


def format_policy_std(policy):
    """提取并格式化当前策略网络的高斯动作标准差 σ = exp(log_std)，方便观察探索幅度变化。"""
    with torch.no_grad():
        if policy.use_sde:
            # gSDE 模式下 log_std 是状态特征维度的矩阵，使用 action_dist.get_std() 获取标准差矩阵并统计均值与范围
            std_tensor = policy.action_dist.get_std(policy.log_std)
            return (
                f"均值={std_tensor.mean().item():.4f} "
                f"(范围 [{std_tensor.min().item():.4f}, {std_tensor.max().item():.4f}], shape={tuple(std_tensor.shape)})"
            )
        else:
            # 标准对角高斯模式下 log_std 是形状为 (action_dim,) 的一维可学习参数向量
            log_std = [round(float(x), 4) for x in policy.log_std.detach().cpu()]
            std = [round(float(x), 4) for x in torch.exp(policy.log_std).detach().cpu()]
            return f"σ = {std} (对应可学习参数 log_std = {log_std})"


# ==============================================================================
# CleanRL 白盒组件 2：连续动作同策略轨迹缓冲区 (RolloutBuffer)
# ==============================================================================
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, action_dim, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1

    def compute_returns_and_advantage(self, last_values, gamma, gae_lambda):
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


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


def bootstrap_truncated_rewards(rewards, terminations, truncations, infos, policy, gamma, device):
    """
    超时截断 (truncated=True 且 terminated=False) 的自举价值修正：
    Pendulum-v1 固定在 200 步触发 truncated=True，必须将终态价值 γ * V(s_final) 加回本步奖励。
    """
    step_rewards = rewards.astype(np.float32).copy()
    for idx in range(len(step_rewards)):
        if not (truncations[idx] and not terminations[idx]):
            continue
        final_obs = None
        if "final_obs" in infos and infos["final_obs"][idx] is not None:
            final_obs = infos["final_obs"][idx]
        elif "final_observation" in infos and infos["final_observation"][idx] is not None:
            final_obs = infos["final_observation"][idx]
        if final_obs is not None:
            with torch.no_grad():
                final_obs_t = torch.as_tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                terminal_v = float(policy.get_value(final_obs_t).item())
            step_rewards[idx] += gamma * terminal_v
    return step_rewards


def record_episode_stats(infos, n_envs, recent_ep_rewards, recent_ep_lengths):
    """从并行环境的 infos 中提取已完成回合的总奖励与步数。"""
    ep_source = infos["final_info"] if ("final_info" in infos and "episode" in infos["final_info"]) else infos
    if "episode" not in ep_source:
        return
    ep_info = ep_source["episode"]
    mask = ep_source.get("_episode", np.ones(n_envs, dtype=bool))
    for idx in range(n_envs):
        if mask[idx]:
            recent_ep_rewards.append(float(ep_info["r"][idx]))
            recent_ep_lengths.append(int(ep_info["l"][idx]))


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


def collect_rollouts(
    train_env,
    eval_env,
    policy,
    rollout_buffer,
    obs,
    global_step,
    vec_step_count,
    n_steps,
    n_envs,
    act_space,
    use_sde,
    sde_sample_freq,
    gamma,
    gae_lambda,
    device,
    eval_freq_vec_steps,
    eval_state,
    log_dir,
    writer,
    recent_ep_rewards,
    recent_ep_lengths,
):
    """
    对应 PPO 伪代码【第 1 步 + 第 2 步】：
    1. 用当前连续高斯策略 π(·|s, θ_k) 并行采样 n_steps 步（原始动作存入 Buffer，裁剪动作送入环境）；
    2. 采样结束后用 v(s_T, w_k) 引导，通过 GAE 计算优势 A_t 与回报目标 R_t。
    """
    rollout_buffer.reset()
    if use_sde:
        policy.reset_noise(n_envs)

    for step_idx in range(n_steps):
        vec_step_count += 1
        global_step += n_envs

        if use_sde and sde_sample_freq > 0 and step_idx % sde_sample_freq == 0:
            policy.reset_noise(n_envs)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            raw_actions, log_probs, _, values = policy.get_action_and_value(obs_tensor)

        # 存入 Buffer 的是未裁剪的原始高斯动作 raw_actions（与 log_prob 精确对应），
        # 传给物理环境 step() 的是裁剪到 [-2.0, 2.0] 的合法力矩 clipped_actions
        raw_actions_np = raw_actions.cpu().numpy()
        clipped_actions_np = np.clip(raw_actions_np, act_space.low, act_space.high)

        next_obs, rewards, terminations, truncations, infos = train_env.step(clipped_actions_np)
        dones = np.logical_or(terminations, truncations)

        step_rewards = bootstrap_truncated_rewards(
            rewards, terminations, truncations, infos, policy, gamma, device
        )
        record_episode_stats(infos, n_envs, recent_ep_rewards, recent_ep_lengths)

        rollout_buffer.add(
            obs=obs,
            action=raw_actions_np,
            reward=step_rewards,
            done=dones.astype(np.float32),
            value=values.flatten().cpu().numpy(),
            log_prob=log_probs.cpu().numpy(),
        )
        obs = next_obs

        if vec_step_count % eval_freq_vec_steps == 0:
            maybe_evaluate_and_save(policy, eval_env, global_step, eval_state, log_dir, writer)

    with torch.no_grad():
        next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        last_values = policy.get_value(next_obs_tensor).flatten().cpu().numpy()
    rollout_buffer.compute_returns_and_advantage(
        last_values=last_values, gamma=gamma, gae_lambda=gae_lambda
    )
    return obs, global_step, vec_step_count


def update_policy(
    policy,
    optimizer,
    rollout_buffer,
    action_dim,
    n_epochs,
    batch_size,
    use_sde,
    clip_range,
    vf_coef,
    ent_coef,
    max_grad_norm,
    device,
):
    """对应 PPO 伪代码【第 3 步】：展平轨迹缓冲区并在 n_epochs 轮内执行小批次 PPO-Clip 更新。"""
    buffer_size = rollout_buffer.n_steps * rollout_buffer.n_envs
    b_obs = torch.as_tensor(rollout_buffer.observations.reshape((-1, *rollout_buffer.obs_shape)), device=device)
    b_actions = torch.as_tensor(rollout_buffer.actions.reshape((-1, action_dim)), device=device)
    b_logprobs = torch.as_tensor(rollout_buffer.log_probs.reshape(-1), device=device)
    b_advantages = torch.as_tensor(rollout_buffer.advantages.reshape(-1), device=device)
    b_returns = torch.as_tensor(rollout_buffer.returns.reshape(-1), device=device)

    b_inds = np.arange(buffer_size)
    pg_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

    for _ in range(n_epochs):
        if use_sde:
            policy.reset_noise(batch_size)
        np.random.shuffle(b_inds)
        for start in range(0, buffer_size, batch_size):
            mb_inds = b_inds[start : start + batch_size]

            _, new_logprob, entropy, new_value = policy.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            ratio = (new_logprob - b_logprobs[mb_inds]).exp()

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = F.mse_loss(new_value.flatten(), b_returns[mb_inds])
            entropy_loss = -entropy.mean()
            loss = pg_loss + vf_coef * v_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            value_losses.append(v_loss.item())
            entropy_losses.append(entropy_loss.item())
            clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_range).float().mean().item()))

    return {
        "pg_loss": float(np.mean(pg_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy_loss": float(np.mean(entropy_losses)),
        "clip_fraction": float(np.mean(clip_fractions)),
    }


def log_update_metrics(writer, policy, metrics, recent_ep_rewards, recent_ep_lengths, global_step, start_time):
    """将每轮连续 PPO 更新后的训练统计与标准差写入 TensorBoard。"""
    if recent_ep_rewards:
        writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
        writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
    writer.add_scalar("train/policy_gradient_loss", metrics["pg_loss"], global_step)
    writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
    writer.add_scalar("train/entropy_loss", metrics["entropy_loss"], global_step)
    writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
    writer.add_scalar("train/std", float(torch.exp(policy.log_std).mean().item()), global_step)
    writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)


def main():
    args = parse_args()
    set_seed(args.seed)
    env_id = "Pendulum-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "ppo_pendulum"
    log_dir = base_dir / "logs" / "ppo_pendulum"
    reset_dir(model_dir)
    reset_dir(log_dir)

    # ==========================================================================
    # 第 1 步：创建矢量化并行连续控制环境与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 {args.n_envs} 个并行的 Gymnasium 连续控制环境: {env_id}")
    train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)
    eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

    obs_space = train_env.single_observation_space
    act_space = train_env.single_action_space
    print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
    print(f"  观测空间 (Observation Space): {obs_space}")
    print(f"  动作空间 (Action Space):      {act_space}")
    print(f"  连续动作维度 (n_actions):     {act_space.shape[-1]}")
    print(f"  连续力矩范围:                 [{act_space.low[0]}, {act_space.high[0]}]")

    # ==========================================================================
    # 第 2 步：模型创建、连续动作概率分布剖析与超参数配置
    # ==========================================================================
    print("\n[第 2 步] 初始化连续动作 PPO 模型与核心超参数...")

    # Actor (pi) 负责输出连续动作高斯分布的均值 μ(s)，Critic (vf) 负责估计状态价值 V(s)
    net_arch = dict(pi=[64, 64], vf=[64, 64])

    # --- 针对 Pendulum-v1 调优的学习率与采样批次参数（与 SB3 02b_ppo_pendulum.py 严格 1:1 对齐）---
    learning_rate = 1e-3       # 在 Pendulum 简单连续控制上，1e-3 比默认的 3e-4 收敛快得多
    n_steps = 1024             # 每个并行环境采集 1024 步（4 个环境共 4096 步）后触发一次更新
    batch_size = 64            # 小批次大小 64
    n_epochs = 10              # 每次采集的 4096 条数据重复训练 10 轮

    # --- 折扣因子与广义优势估计 (GAE) ---
    gamma = 0.9                # Pendulum 关注近期力矩平衡，较小的折扣因子 γ=0.9 能显著降低价值估计方差、加速收敛
    gae_lambda = 0.95          # GAE 平滑系数 λ

    # --- PPO 核心裁剪与联合 Loss 系数 ---
    clip_range = 0.2           # 新旧策略概率比裁剪区间 [0.8, 1.2]
    ent_coef = 0.0             # 连续高斯分布初期自带较大标准差 (σ=1.0)，无需额外加熵奖励即可充分探索
    vf_coef = 0.5              # Critic 价值函数 Loss 权重
    max_grad_norm = 0.5        # 梯度范数裁剪上限

    # --- 连续动作空间专属：探索机制选择（标准对角高斯 vs gSDE）---
    use_sde = args.use_sde
    sde_sample_freq = 4 if use_sde else -1

    rollout_buffer_size = n_steps * args.n_envs
    print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(act_space.shape[-1])

    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=act_space.low,
        action_high=act_space.high,
        net_arch=net_arch,
        use_sde=use_sde,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    rollout_buffer = RolloutBuffer(
        n_steps=n_steps,
        n_envs=args.n_envs,
        obs_shape=obs_space.shape,
        action_dim=action_dim,
        device=device,
    )

    # 直观打印底层动作分布类型与初始高斯标准差，对比离散版 CategoricalDistribution
    dist_cls_name = type(policy.action_dist).__name__
    print(f"  底层动作分布类型 (action_dist): {dist_cls_name} (use_sde={policy.use_sde})")
    print(f"  训练前初始探索标准差:           {format_policy_std(policy)}")

    # ==========================================================================
    # 第 3 步：训练前基线评估
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估...")
    mean_reward_before, std_reward_before = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：启动 CleanRL 连续动作 PPO 训练主循环与定期评估
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练连续动作 PPO（总步数 total_timesteps={args.timesteps}）...")
    eval_freq_vec_steps = max(1_000, (args.timesteps // 5) // args.n_envs)
    eval_state = {
        "best_mean_reward": -float("inf"),
        "best_model_path": model_dir / "best_model.pt",
        "timesteps": [],
        "results": [],
        "ep_lengths": [],
    }

    tb_log_name = "PPO_Pendulum_gSDE" if use_sde else "PPO_Pendulum"
    tb_run_dir = log_dir / tb_log_name
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    recent_ep_rewards = []
    recent_ep_lengths = []

    obs, _ = train_env.reset(seed=args.seed)
    global_step = 0
    vec_step_count = 0
    num_updates = max(1, args.timesteps // rollout_buffer_size)
    start_time = time.time()

    for _ in range(1, num_updates + 1):
        # 阶段 A（伪代码第 1-2 步）：并行连续环境采样 + 计算 GAE 优势与回报目标
        obs, global_step, vec_step_count = collect_rollouts(
            train_env=train_env,
            eval_env=eval_env,
            policy=policy,
            rollout_buffer=rollout_buffer,
            obs=obs,
            global_step=global_step,
            vec_step_count=vec_step_count,
            n_steps=n_steps,
            n_envs=args.n_envs,
            act_space=act_space,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            eval_freq_vec_steps=eval_freq_vec_steps,
            eval_state=eval_state,
            log_dir=log_dir,
            writer=writer,
            recent_ep_rewards=recent_ep_rewards,
            recent_ep_lengths=recent_ep_lengths,
        )

        # 阶段 B（伪代码第 3 步）：小批次 PPO-Clip 联合更新 Actor 与 Critic
        metrics = update_policy(
            policy=policy,
            optimizer=optimizer,
            rollout_buffer=rollout_buffer,
            action_dim=action_dim,
            n_epochs=n_epochs,
            batch_size=batch_size,
            use_sde=use_sde,
            clip_range=clip_range,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        log_update_metrics(writer, policy, metrics, recent_ep_rewards, recent_ep_lengths, global_step, start_time)

    writer.close()
    best_model_path = eval_state["best_model_path"]

    # ==========================================================================
    # 第 5 步：训练后定量评估与高斯标准差收敛观察
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励:       {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励:       {mean_reward_after:.2f} +/- {std_reward_after:.2f}")
    print(f"  训练后收敛探索标准差: {format_policy_std(policy)}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "ppo_pendulum_final.pt"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    torch.save(policy.state_dict(), final_model_path)

    del policy
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = ActorCriticPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=act_space.low,
        action_high=act_space.high,
        net_arch=net_arch,
        use_sde=use_sde,
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
        # 在连续动作 PPO 中：
        # - 训练采样时 deterministic=False：从高斯分布 N(μ(s), σ^2) 中随机采样以保证探索；
        # - 评估/部署时 deterministic=True：直接输出高斯均值 μ(s) 并裁剪到 [-2.0, 2.0]，消除随机抖动！
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
