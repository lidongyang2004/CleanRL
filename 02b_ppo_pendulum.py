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
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
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


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_run_dir(log_dir: Path, tb_log_name: str) -> Path:
    """模仿 SB3 自动递增生成 TensorBoard 子目录（如 PPO_Pendulum_1 或 PPO_Pendulum_gSDE_1）。"""
    idx = 1
    while (log_dir / f"{tb_log_name}_{idx}").exists():
        idx += 1
    run_dir = log_dir / f"{tb_log_name}_{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_vec_env(env_id: str, n_envs: int, seed: int) -> gym.vector.SyncVectorEnv:
    """创建 n_envs 个并行的 Gymnasium 连续控制环境（SAME_STEP 自动重置模式）。"""

    def make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
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
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """正交初始化：隐藏层 gain=sqrt(2)，均值输出层 action_net gain=0.01，价值输出层 value_net gain=1.0。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MlpExtractor(nn.Module):
    """双分支解耦特征提取主干：policy_net (pi) 负责动作均值，value_net (vf) 负责状态价值。"""

    def __init__(self, obs_dim: int, pi_arch: list[int], vf_arch: list[int]) -> None:
        super().__init__()
        pi_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in pi_arch:
            pi_layers.append(layer_init(nn.Linear(in_dim, h_dim), std=np.sqrt(2)))
            pi_layers.append(nn.Tanh())
            in_dim = h_dim
        self.policy_net = nn.Sequential(*pi_layers)

        vf_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in vf_arch:
            vf_layers.append(layer_init(nn.Linear(in_dim, h_dim), std=np.sqrt(2)))
            vf_layers.append(nn.Tanh())
            in_dim = h_dim
        self.value_net = nn.Sequential(*vf_layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)


class DiagGaussianDistribution:
    """标准对角高斯分布 (use_sde=False)：标准差 σ = exp(log_std) 独立于状态。"""

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_std)


class StateDependentNoiseDistribution:
    """
    广义状态依赖探索分布 (gSDE, use_sde=True)：
    噪声权重矩阵 ε_θ ~ N(0, σ^2) 在若干步内保持固定，探索扰动为 h_π(s) @ ε_θ，
    对应条件高斯分布的方差为 Σ(s) = (h_π(s)^2) @ (σ^2)。
    """

    def __init__(self, latent_sde_dim: int, action_dim: int, epsilon: float = 1e-6) -> None:
        self.latent_sde_dim = latent_sde_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.exploration_mat: torch.Tensor | None = None
        self.exploration_matrices: torch.Tensor | None = None

    def get_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return torch.exp(log_std)

    def sample_weights(self, log_std: torch.Tensor, batch_size: int = 1) -> None:
        std = self.get_std(log_std)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        latent_sde = latent_sde.detach()
        if self.exploration_mat is None or self.exploration_matrices is None:
            self.sample_weights(log_std, batch_size=len(latent_sde))
        assert self.exploration_mat is not None and self.exploration_matrices is not None
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices).squeeze(1)


class ActorCriticPolicy(nn.Module):
    """
    连续动作空间 Actor-Critic 策略网络（同时支持标准对角高斯与 gSDE 状态依赖探索）。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        net_arch: dict[str, list[int]],
        use_sde: bool = False,
    ) -> None:
        super().__init__()
        pi_arch = net_arch.get("pi", [64, 64])
        vf_arch = net_arch.get("vf", [64, 64])
        self.use_sde = use_sde
        self.action_dim = action_dim
        self.latent_dim_pi = pi_arch[-1]

        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)
        self.mlp_extractor = MlpExtractor(obs_dim, pi_arch=pi_arch, vf_arch=vf_arch)
        self.action_net = layer_init(nn.Linear(self.latent_dim_pi, action_dim), std=0.01)
        self.value_net = layer_init(nn.Linear(vf_arch[-1], 1), std=1.0)

        if self.use_sde:
            # gSDE 模式下：log_std 是形状为 (64, action_dim) 的状态特征噪声矩阵参数（初始为 0.0）
            self.action_dist = StateDependentNoiseDistribution(self.latent_dim_pi, action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.latent_dim_pi, action_dim), requires_grad=True)
            self.action_dist.sample_weights(self.log_std, batch_size=1)
        else:
            # 标准对角高斯模式下：log_std 是形状为 (action_dim,) 的独立可学习参数向量（初始为 0.0）
            self.action_dist = DiagGaussianDistribution()
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def reset_noise(self, batch_size: int = 1) -> None:
        """在 gSDE 模式下重新采样状态依赖噪声矩阵 ε_θ。"""
        if self.use_sde and isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (未裁剪原始高斯动作, 对数似然 log_prob, 分布熵 entropy, 状态价值 value)。"""
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        mean_actions = self.action_net(latent_pi)

        if self.use_sde and isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde = latent_pi.detach()
            std_mat = self.action_dist.get_std(self.log_std)
            variance = torch.mm(latent_sde**2, std_mat**2)
            dist = Normal(mean_actions, torch.sqrt(variance + 1e-6))
            if action is None:
                noise = self.action_dist.get_noise(latent_sde, self.log_std)
                action = mean_actions + noise
        else:
            action_std = torch.exp(self.log_std).expand_as(mean_actions)
            dist = Normal(mean_actions, action_std)
            if action is None:
                action = dist.sample()

        # 多维独立高斯分布的联合对数概率等于各维度 log_prob 之和
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.value_net(latent_vf)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """
        与 SB3 完全一致的连续动作推理接口：
        - deterministic=True 时直接输出高斯均值 μ(s) 并裁剪至环境动作边界 [low, high]；
        - deterministic=False 时从高斯分布采样并裁剪。
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            single_env = obs_tensor.ndim == 1
            if single_env:
                obs_tensor = obs_tensor.unsqueeze(0)
            features = self.features_extractor(obs_tensor)
            latent_pi, _ = self.mlp_extractor(features)
            mean_actions = self.action_net(latent_pi)
            if deterministic:
                raw_actions = mean_actions
            else:
                raw_actions, _, _, _ = self.get_action_and_value(obs_tensor)
            clipped_actions = torch.clamp(raw_actions, self.action_low, self.action_high)
            if single_env:
                return clipped_actions.squeeze(0).cpu().numpy(), None
            return clipped_actions.cpu().numpy(), None


def format_policy_std(policy: ActorCriticPolicy) -> str:
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
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.n_steps, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1

    def compute_returns_and_advantage(
        self, last_values: np.ndarray, gamma: float, gae_lambda: float
    ) -> None:
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
    policy: ActorCriticPolicy,
    eval_env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    return_episode_rewards: bool = False,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """在独立 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差。"""
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    env_id = "Pendulum-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "ppo_pendulum"
    log_dir = base_dir / "logs" / "ppo_pendulum"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

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
    best_mean_reward = -float("inf")
    best_model_path = model_dir / "best_model.pt"

    tb_log_name = "PPO_Pendulum_gSDE" if use_sde else "PPO_Pendulum"
    tb_run_dir = get_next_run_dir(log_dir, tb_log_name)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    eval_timesteps: list[int] = []
    eval_results: list[list[float]] = []
    eval_ep_lengths: list[list[int]] = []
    recent_ep_rewards: list[float] = []
    recent_ep_lengths: list[int] = []

    obs, _ = train_env.reset(seed=args.seed)
    global_step = 0
    vec_step_count = 0
    num_updates = max(1, args.timesteps // rollout_buffer_size)
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # ----------------------------
        # 阶段 A：并行连续环境采样 (Collect Rollouts)
        # ----------------------------
        rollout_buffer.reset()
        if use_sde:
            policy.reset_noise(args.n_envs)

        for step_idx in range(n_steps):
            vec_step_count += 1
            global_step += args.n_envs

            if use_sde and sde_sample_freq > 0 and step_idx % sde_sample_freq == 0:
                policy.reset_noise(args.n_envs)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                raw_actions, log_probs, _, values = policy.get_action_and_value(obs_tensor)

            # 注意：存入 RolloutBuffer 的是未裁剪的原始高斯采样动作 raw_actions（保证 log_prob 精确对应），
            # 而传给物理环境 step() 的是裁剪到 [-2.0, 2.0] 的合法力矩 clipped_actions（与 SB3 完全一致）
            raw_actions_np = raw_actions.cpu().numpy()
            clipped_actions_np = np.clip(raw_actions_np, act_space.low, act_space.high)

            next_obs, rewards, terminations, truncations, infos = train_env.step(clipped_actions_np)
            dones = np.logical_or(terminations, truncations)

            # 关键细节：Pendulum-v1 总是在 200 步时触发 truncated=True，必须用 final_obs 做自举价值修正！
            step_rewards = rewards.astype(np.float32).copy()
            for idx in range(args.n_envs):
                if truncations[idx] and not terminations[idx]:
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

            if "final_info" in infos and "episode" in infos["final_info"]:
                ep_info = infos["final_info"]["episode"]
                mask = infos["final_info"].get("_episode", np.ones(args.n_envs, dtype=bool))
                for idx in range(args.n_envs):
                    if mask[idx]:
                        recent_ep_rewards.append(float(ep_info["r"][idx]))
                        recent_ep_lengths.append(int(ep_info["l"][idx]))

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

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            last_values = policy.get_value(next_obs_tensor).flatten().cpu().numpy()
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, gamma=gamma, gae_lambda=gae_lambda
        )

        # ----------------------------
        # 阶段 B：展平 RolloutBuffer 并执行 n_epochs 轮小批次 PPO-Clip 更新
        # ----------------------------
        b_obs = torch.as_tensor(rollout_buffer.observations.reshape((-1, *rollout_buffer.obs_shape)), device=device)
        b_actions = torch.as_tensor(rollout_buffer.actions.reshape((-1, action_dim)), device=device)
        b_logprobs = torch.as_tensor(rollout_buffer.log_probs.reshape(-1), device=device)
        b_advantages = torch.as_tensor(rollout_buffer.advantages.reshape(-1), device=device)
        b_returns = torch.as_tensor(rollout_buffer.returns.reshape(-1), device=device)

        b_inds = np.arange(rollout_buffer_size)
        pg_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

        for _ in range(n_epochs):
            if use_sde:
                policy.reset_noise(batch_size)
            np.random.shuffle(b_inds)
            for start in range(0, rollout_buffer_size, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

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

        if recent_ep_rewards:
            writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
            writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
        writer.add_scalar("train/policy_gradient_loss", float(np.mean(pg_losses)), global_step)
        writer.add_scalar("train/value_loss", float(np.mean(value_losses)), global_step)
        writer.add_scalar("train/entropy_loss", float(np.mean(entropy_losses)), global_step)
        writer.add_scalar("train/clip_fraction", float(np.mean(clip_fractions)), global_step)
        writer.add_scalar("train/std", float(torch.exp(policy.log_std).mean().item()), global_step)
        writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

    writer.close()

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
