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

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
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


class VecEnvWrapper:
    """
    轻量级矢量化环境包装器：
    将 Gymnasium `SyncVectorEnv` 的 `observation_space` 和 `action_space` 属性对齐为单个子环境的空间
    （即 `single_observation_space` 与 `single_action_space`），与 SB3 的 `VecEnv` 属性行为保持完全一致。
    """

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.single_observation_space
        self.action_space = venv.single_action_space
        self.single_observation_space = venv.single_observation_space
        self.single_action_space = venv.single_action_space

    def reset(self, seed = None):
        return self.venv.reset(seed=seed)

    def step(self, actions):
        return self.venv.step(actions)

    def close(self):
        self.venv.close()


def make_vec_env(env_id, n_envs = 1, seed = None):
    """创建 n_envs 个并行的 Gymnasium 连续控制环境（SAME_STEP 自动重置模式）。"""
    base_seed = seed if seed is not None else 0

    def make_single_env(rank):
        def _init():
            env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
            env.reset(seed=base_seed + rank)
            return env

        return _init

    vec_kwargs = {}
    if hasattr(gym.vector, "AutoresetMode"):
        vec_kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    envs = gym.vector.SyncVectorEnv([make_single_env(i) for i in range(n_envs)], **vec_kwargs)
    return VecEnvWrapper(envs)


# ==============================================================================
# CleanRL 白盒组件 1：连续动作概率分布 (DiagGaussian vs gSDE) 与 ActorCriticPolicy
# 严格对应 SB3 `DiagGaussianDistribution` 与 `StateDependentNoiseDistribution`
# ==============================================================================
def layer_init(layer, std = float(np.sqrt(2)), bias_const = 0.0):
    """正交初始化：隐藏层 gain=sqrt(2)，均值输出层 action_net gain=0.01，价值输出层 value_net gain=1.0。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(
    in_dim,
    hidden_dims,
    out_dim = None,
    out_std = 1.0,
):
    """构建带 Tanh 激活与正交初始化的多层感知机（当 out_dim=None 时仅返回隐藏层特征主干）。"""
    layers = []
    for h_dim in hidden_dims:
        layers.append(layer_init(nn.Linear(in_dim, h_dim), std=float(np.sqrt(2))))  # 隐藏层用 sqrt(2) 补偿 Tanh 压缩
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

    def __init__(self, latent_sde_dim, action_dim, epsilon = 1e-6):
        self.latent_sde_dim = latent_sde_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.exploration_mat = None
        self.exploration_matrices = None

    def get_std(self, log_std):
        return torch.exp(log_std)

    def sample_weights(self, log_std, batch_size = 1):
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
        net_arch = None,
        use_sde = False,
    ):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = list(net_arch.get("pi", [64, 64]))
            vf_arch = list(net_arch.get("vf", [64, 64]))
        elif isinstance(net_arch, list):
            pi_arch = list(net_arch)
            vf_arch = list(net_arch)
        else:
            pi_arch, vf_arch = [64, 64], [64, 64]

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
            self.action_dist = (
                StateDependentNoiseDistribution(self.latent_dim_pi, action_dim)
            )
            self.log_std = nn.Parameter(torch.zeros(self.latent_dim_pi, action_dim), requires_grad=True)
            self.action_dist.sample_weights(self.log_std, batch_size=1)
        else:
            # 标准对角高斯模式下：log_std 是形状为 (action_dim,) 的独立可学习参数向量（初始为 0.0）
            self.action_dist = DiagGaussianDistribution()
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def reset_noise(self, batch_size = 1):
        """在 gSDE 模式下重新采样状态依赖噪声矩阵 ε_θ。"""
        if self.use_sde and isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_value(self, obs):
        """对应伪代码价值函数 v(s, w_k)：仅计算状态价值，形状 (batch_size, 1)。"""
        return self.critic(self.flatten(obs))

    def get_action_and_value(self, obs, action = None):
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

    def predict(self, obs, deterministic = True):
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
# CleanRL 白盒组件 4：连续动作 PPO 算法主类（与 SB3 `PPO` 接口 1:1 完全对齐）
# ==============================================================================
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


class PPO:
    """
    CleanRL 白盒实现的连续动作 PPO 算法类（接口与 `stable_baselines3.PPO` 完全一致）：
    支持对角高斯分布 (`use_sde=False`) 与广义状态依赖探索 (`use_sde=True`)。
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate = 3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        ent_coef = 0.0,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        use_sde = False,
        sde_sample_freq = -1,
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
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.policy_kwargs = policy_kwargs or {}
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if env is not None:
            self.n_envs = getattr(env, "num_envs", 1)
            obs_space = getattr(env, "single_observation_space", env.observation_space)
            act_space = getattr(env, "single_action_space", env.action_space)
            self.obs_shape = tuple(obs_space.shape)
            self.action_dim = int(act_space.shape[-1])
            self.action_low = np.array(act_space.low, dtype=np.float32)
            self.action_high = np.array(act_space.high, dtype=np.float32)
        else:
            assert _obs_shape is not None and _action_dim is not None
            assert _action_low is not None and _action_high is not None
            self.n_envs = 1
            self.obs_shape = tuple(_obs_shape)
            self.action_dim = int(_action_dim)
            self.action_low = np.array(_action_low, dtype=np.float32)
            self.action_high = np.array(_action_high, dtype=np.float32)

        self.obs_dim = int(np.prod(self.obs_shape))
        self.net_arch = self.policy_kwargs.get("net_arch", dict(pi=[64, 64], vf=[64, 64]))

        self.policy = ActorCriticPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_low=self.action_low,
            action_high=self.action_high,
            net_arch=self.net_arch,
            use_sde=self.use_sde,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
        )

    def predict(self, obs, deterministic = True):
        return self.policy.predict(obs, deterministic=deterministic)

    def _collect_rollouts(
        self,
        obs,
        global_step,
        callback,
        writer,
        recent_ep_rewards,
        recent_ep_lengths,
    ):
        """
        对应 PPO 伪代码【第 1 步 + 第 2 步】：
        1. 用当前连续高斯策略 π(·|s, θ_k) 并行采样 n_steps 步（原始动作存入 Buffer，裁剪动作送入环境）；
        2. 采样结束后用 v(s_T, w_k) 引导，通过 GAE 计算优势 A_t 与回报目标 R_t。
        """
        assert self.env is not None
        self.rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(self.n_envs)

        for step_idx in range(self.n_steps):
            global_step += self.n_envs

            if self.use_sde and self.sde_sample_freq > 0 and step_idx % self.sde_sample_freq == 0:
                self.policy.reset_noise(self.n_envs)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                raw_actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)

            # 存入 Buffer 的是未裁剪的原始高斯动作 raw_actions（与 log_prob 精确对应），
            # 传给物理环境 step() 的是裁剪到 [-2.0, 2.0] 的合法力矩 clipped_actions
            raw_actions_np = raw_actions.cpu().numpy()
            clipped_actions_np = np.clip(raw_actions_np, self.action_low, self.action_high)

            next_obs, rewards, terminations, truncations, infos = self.env.step(clipped_actions_np)
            dones = np.logical_or(terminations, truncations)

            step_rewards = bootstrap_truncated_rewards(
                rewards, terminations, truncations, infos, self.policy, self.gamma, self.device
            )
            record_episode_stats(infos, self.n_envs, recent_ep_rewards, recent_ep_lengths)

            self.rollout_buffer.add(
                obs=obs,
                action=raw_actions_np,
                reward=step_rewards,
                done=dones.astype(np.float32),
                value=values.flatten().cpu().numpy(),
                log_prob=log_probs.cpu().numpy(),
            )
            obs = next_obs

            if callback is not None:
                callback.on_step(self, global_step, writer=writer)

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_values = self.policy.get_value(next_obs_tensor).flatten().cpu().numpy()
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        return obs, global_step

    def _update_policy(self):
        """对应 PPO 伪代码【第 3 步】：展平轨迹缓冲区并在 n_epochs 轮内执行小批次 PPO-Clip 更新。"""
        buffer_size = self.rollout_buffer.n_steps * self.rollout_buffer.n_envs
        b_obs = torch.as_tensor(
            self.rollout_buffer.observations.reshape((-1, *self.rollout_buffer.obs_shape)), device=self.device
        )
        b_actions = torch.as_tensor(
            self.rollout_buffer.actions.reshape((-1, self.action_dim)), device=self.device
        )
        b_logprobs = torch.as_tensor(self.rollout_buffer.log_probs.reshape(-1), device=self.device)
        b_advantages = torch.as_tensor(self.rollout_buffer.advantages.reshape(-1), device=self.device)
        b_returns = torch.as_tensor(self.rollout_buffer.returns.reshape(-1), device=self.device)

        b_inds = np.arange(buffer_size)
        pg_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

        for _ in range(self.n_epochs):
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, self.batch_size):
                mb_inds = b_inds[start : start + self.batch_size]

                _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (new_logprob - b_logprobs[mb_inds]).exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(new_value.flatten(), b_returns[mb_inds])
                entropy_loss = -entropy.mean()
                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(float((torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()))

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }

    def learn(
        self,
        total_timesteps,
        callback = None,
        tb_log_name = "PPO",
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
        global_step = 0
        rollout_buffer_size = self.n_steps * self.n_envs
        num_updates = max(1, total_timesteps // rollout_buffer_size)
        start_time = time.time()

        for _ in range(1, num_updates + 1):
            # 阶段 A（伪代码第 1-2 步）：并行连续环境采样 + 计算 GAE 优势与回报目标
            obs, global_step = self._collect_rollouts(
                obs=obs,
                global_step=global_step,
                callback=callback,
                writer=writer,
                recent_ep_rewards=recent_ep_rewards,
                recent_ep_lengths=recent_ep_lengths,
            )

            # 阶段 B（伪代码第 3 步）：小批次 PPO-Clip 联合更新 Actor 与 Critic
            metrics = self._update_policy()

            if writer is not None:
                if recent_ep_rewards:
                    writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                    writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
                writer.add_scalar("train/policy_gradient_loss", metrics["pg_loss"], global_step)
                writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
                writer.add_scalar("train/entropy_loss", metrics["entropy_loss"], global_step)
                writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], global_step)
                writer.add_scalar("train/std", float(torch.exp(self.policy.log_std).mean().item()), global_step)
                writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

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
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
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
            n_steps=checkpoint.get("n_steps", 1024),
            batch_size=checkpoint.get("batch_size", 64),
            n_epochs=checkpoint.get("n_epochs", 10),
            gamma=checkpoint.get("gamma", 0.9),
            gae_lambda=checkpoint.get("gae_lambda", 0.95),
            clip_range=checkpoint.get("clip_range", 0.2),
            ent_coef=checkpoint.get("ent_coef", 0.0),
            vf_coef=checkpoint.get("vf_coef", 0.5),
            max_grad_norm=checkpoint.get("max_grad_norm", 0.5),
            use_sde=checkpoint.get("use_sde", False),
            sde_sample_freq=checkpoint.get("sde_sample_freq", -1),
            policy_kwargs=checkpoint.get("policy_kwargs", dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))),
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
    parser = argparse.ArgumentParser(description="SB3 PPO 求解 Pendulum-v1 连续动作控制示例")
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


def format_policy_std(model):
    """提取并格式化当前策略网络的高斯动作标准差 σ = exp(log_std)，方便观察探索幅度变化。"""
    with torch.no_grad():
        if model.use_sde:
            # gSDE 模式下 log_std 是状态特征维度的矩阵，使用 action_dist.get_std() 获取标准差矩阵并统计均值与范围
            std_tensor = model.policy.action_dist.get_std(model.policy.log_std)
            return (
                f"均值={std_tensor.mean().item():.4f} "
                f"(范围 [{std_tensor.min().item():.4f}, {std_tensor.max().item():.4f}], shape={tuple(std_tensor.shape)})"
            )
        else:
            # 标准对角高斯模式下 log_std 是形状为 (action_dim,) 的一维可学习参数向量
            log_std = [round(float(x), 4) for x in model.policy.log_std.detach().cpu()]
            std = [round(float(x), 4) for x in torch.exp(model.policy.log_std).detach().cpu()]
            return f"σ = {std} (对应可学习参数 log_std = {log_std})"


def main():
    args = parse_args()
    env_id = "Pendulum-v1"

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
    eval_env = Monitor(gym.make(env_id))

    print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")
    print(f"  连续动作维度 (n_actions):     {train_env.action_space.shape[-1]}")
    print(f"  连续力矩范围:                 [{train_env.action_space.low[0]}, {train_env.action_space.high[0]}]")

    # ==========================================================================
    # 第 2 步：模型创建、连续动作概率分布剖析与超参数配置
    # ==========================================================================
    print("\n[第 2 步] 初始化连续动作 PPO 模型与核心超参数...")

    # Actor (pi) 负责输出连续动作高斯分布的均值 μ(s)，Critic (vf) 负责估计状态价值 V(s)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    n_steps = 1024
    batch_size = 64
    rollout_buffer_size = n_steps * args.n_envs
    print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        # --- 针对 Pendulum-v1 调优的学习率与采样批次参数（参考 RL Baselines3 Zoo）---
        learning_rate=1e-3,       # 在 Pendulum 简单连续控制上，1e-3 比默认的 3e-4 收敛快得多
        n_steps=n_steps,          # 每个并行环境采集 1024 步（4 个环境共 4096 步）后触发一次更新
        batch_size=batch_size,    # 小批次大小 64
        n_epochs=10,              # 每次采集的 4096 条数据重复训练 10 轮

        # --- 折扣因子与广义优势估计 (GAE) ---
        gamma=0.9,                # Pendulum 关注近期力矩平衡，较小的折扣因子 γ=0.9 能显著降低价值估计方差、加速收敛
        gae_lambda=0.95,          # GAE 平滑系数 λ

        # --- PPO 核心裁剪与联合 Loss 系数 ---
        clip_range=0.2,           # 新旧策略概率比裁剪区间 [0.8, 1.2]
        ent_coef=0.0,             # 连续高斯分布初期自带较大标准差 (σ=1.0)，无需额外加熵奖励即可充分探索
        vf_coef=0.5,              # Critic 价值函数 Loss 权重
        max_grad_norm=0.5,        # 梯度范数裁剪上限

        # --- 连续动作空间专属：探索机制选择（标准对角高斯 vs gSDE）---
        use_sde=args.use_sde,     # False: 标准 DiagGaussianDistribution | True: 广义状态依赖探索 gSDE
        sde_sample_freq=4 if args.use_sde else -1,  # 开启 gSDE 时，每 4 步重新采样一次状态依赖噪声权重

        # --- 网络结构与日志 ---
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
    )

    # 直观打印底层动作分布类型与初始高斯标准差，对比离散版 CategoricalDistribution
    dist_cls_name = type(model.policy.action_dist).__name__
    print(f"  底层动作分布类型 (action_dist): {dist_cls_name} (use_sde={model.use_sde})")
    print(f"  训练前初始探索标准差:           {format_policy_std(model)}")

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
    print(f"\n[第 4 步] 开始训练连续动作 PPO（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, (args.timesteps // 5) // args.n_envs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    tb_log_name = "PPO_Pendulum_gSDE" if args.use_sde else "PPO_Pendulum"
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=tb_log_name,
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估与高斯标准差收敛观察
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励:       {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励:       {mean_reward_after:.2f} +/- {std_reward_after:.2f}")
    print(f"  训练后收敛探索标准差: {format_policy_std(model)}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载
    # ==========================================================================
    final_model_path = model_dir / "ppo_pendulum_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    del model
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = PPO.load(load_path, env=eval_env)

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
