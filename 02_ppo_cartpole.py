"""
================================================================================
CleanRL 实战教程 02：PPO (Proximal Policy Optimization) 求解 CartPole-v1
================================================================================

【算法定位】
- 算法名称：PPO (近端策略优化 —— CleanRL 单文件 PyTorch 白盒实现)
- 策略类型：On-Policy（同策略）—— 只能用“当前最新策略”采样的数据进行更新，更新完后数据即丢弃
- 学习范式：Actor-Critic（行动者-评论家）
  - Actor (Policy Network π): 负责输出动作概率分布（决定怎么做）
  - Critic (Value Network V):  负责估计当前状态的价值 V(s)（评价当前状态有多好，用来计算优势函数）
- 支持动作空间：**通用全能王！同时支持离散 (Discrete) 与连续 (Box) 动作空间**

【On-Policy 为什么需要多环境并行 (SyncVectorEnv / make_vec_env)？】
因为 PPO 是同策略算法，没有庞大的 Replay Buffer 存老数据。
为了在每次迭代中快速收集到足够多样化、低相关性的训练样本，工程上通常会同时开多个并行环境
（例如 n_envs=4 个子环境同时运行）。
每次采样迭代收集的总样本量计算公式为：
    RolloutBuffer 总样本数 = n_steps * n_envs
例如 n_steps=512, n_envs=4，则每次采样共收集 512 * 4 = 2048 步数据，
随后将这 2048 条数据打乱切分为若干个 batch_size=64 的小批次，反复训练 n_epochs=10 轮。

【PPO 的核心稳定机制：Clipped Surrogate Objective】
策略梯度算法最怕“步子迈太大扯着蛋”（一次更新过大导致策略崩溃，之后再也采不到好样本）。
PPO 通过限制新旧策略概率比值 r_t(θ) = π_new(a|s) / π_old(a|s) 只能在 [1 - clip_range, 1 + clip_range]
（通常为 [0.8, 1.2]）范围内获得收益，从而实现稳定、单调的策略提升。

【CleanRL 白盒实现 vs Stable-Baselines3 (SB3) 一一对应关系】
- SB3 `make_vec_env(env_id, n_envs=4)`       <--> CleanRL `gym.vector.SyncVectorEnv` (SAME_STEP 自动重置模式)
- SB3 `ActorCriticPolicy(pi=[64,64], vf=[64,64])` <--> CleanRL 显式定义 `MlpExtractor` + 正交初始化双分支网络
- SB3 `RolloutBuffer` (GAE + 截断自举修正)    <--> CleanRL 显式实现 `RolloutBuffer.compute_returns_and_advantage`
- SB3 `PPO.train()` 联合损失反向传播          <--> CleanRL 显式展开 Minibatch 打乱、Clip 策略损失 + 价值损失 + 熵正则

运行方式：
    python 02_ppo_cartpole.py                  # 使用默认步数 (50000 步) 完整训练
    python 02_ppo_cartpole.py --timesteps 15000 # 快速测试全流程
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    """解析命令行参数（与 SB3 02_ppo_cartpole.py 完全一致）。"""
    parser = argparse.ArgumentParser(description="CleanRL PPO 并行环境训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练步数（注意：在 4 个并行环境下，50000 步实际只需约 12500 次环境并行 step）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量（默认 4）")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_run_dir(log_dir: Path, tb_log_name: str) -> Path:
    """模仿 SB3 自动递增生成 TensorBoard 子目录（如 PPO_CartPole_1, PPO_CartPole_2）。"""
    idx = 1
    while (log_dir / f"{tb_log_name}_{idx}").exists():
        idx += 1
    run_dir = log_dir / f"{tb_log_name}_{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_vec_env(env_id: str, n_envs: int, seed: int) -> gym.vector.SyncVectorEnv:
    """
    CleanRL 白盒实现的并行环境构造函数（对应 SB3 `make_vec_env`）：
    为每个子环境套上 RecordEpisodeStatistics 包装器，并使用 SAME_STEP 自动重置模式，
    确保子环境结束时立即重置并将终止状态保存在 infos["final_obs"] 中。
    """

    def make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
            env.reset(seed=seed + rank)
            return env

        return _init

    vec_kwargs = {}
    if hasattr(gym.vector, "AutoresetMode"):
        vec_kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    envs = gym.vector.SyncVectorEnv([make_single_env(i) for i in range(n_envs)], **vec_kwargs)
    return envs


# ==============================================================================
# CleanRL 白盒组件 1：正交初始化与 ActorCriticPolicy 双分支解耦网络
# 严格对应 SB3 `ActorCriticPolicy(net_arch=dict(pi=[64, 64], vf=[64, 64]))`
# ==============================================================================
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    PPO 核心工程技巧：正交初始化 (Orthogonal Initialization)。
    - 隐藏层使用 gain=sqrt(2) 配合 Tanh 激活函数保持方差稳定；
    - 策略输出头 action_net 使用极小的 gain=0.01，使初始 Logits 接近 0，初始动作分布接近均匀分布（最大熵探索）；
    - 价值输出头 value_net 使用 gain=1.0。
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MlpExtractor(nn.Module):
    """对应 SB3 的 MlpExtractor：分别构建互不共享参数的 Actor 特征主干 (policy_net) 与 Critic 特征主干 (value_net)。"""

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


class ActorCriticPolicy(nn.Module):
    """
    离散动作空间 Actor-Critic 策略网络（对应 SB3 `ActorCriticPolicy`）：
    - Actor 分支输出每个离散动作的未归一化对数几率 Logits，经 Softmax 构成 CategoricalDistribution；
    - Critic 分支输出标量状态价值估计 V(s)。
    """

    def __init__(self, obs_dim: int, n_actions: int, net_arch: dict[str, list[int]]) -> None:
        super().__init__()
        pi_arch = net_arch.get("pi", [64, 64])
        vf_arch = net_arch.get("vf", [64, 64])

        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)
        self.mlp_extractor = MlpExtractor(obs_dim, pi_arch=pi_arch, vf_arch=vf_arch)
        self.action_net = layer_init(nn.Linear(pi_arch[-1], n_actions), std=0.01)
        self.value_net = layer_init(nn.Linear(vf_arch[-1], 1), std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """仅计算状态价值 V(s)，形状 (batch_size, 1)。"""
        features = self.features_extractor(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CleanRL 经典四合一前向接口：
        同时返回 (动作 action, 动作对数概率 log_prob, 分布熵 entropy, 状态价值 value)。
        """
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value_net(latent_vf)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int | np.ndarray, None]:
        """与 SB3 完全一致的推理接口：deterministic=True 时取概率最大的离散动作 argmax。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            single_env = obs_tensor.ndim == 1
            if single_env:
                obs_tensor = obs_tensor.unsqueeze(0)
            features = self.features_extractor(obs_tensor)
            latent_pi, _ = self.mlp_extractor(features)
            logits = self.action_net(latent_pi)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = Categorical(logits=logits).sample()
            if single_env:
                return int(actions.item()), None
            return actions.cpu().numpy(), None


# ==============================================================================
# CleanRL 白盒组件 2：同策略轨迹缓冲区 (RolloutBuffer) 与 GAE 计算
# 严格对应 SB3 `stable_baselines3.common.buffers.RolloutBuffer`
# ==============================================================================
class RolloutBuffer:
    """
    On-Policy 轨迹缓冲区：存储 n_steps * n_envs 条当前策略采集的样本，
    并在采样结束时通过广义优势估计 (GAE) 逆序递推计算优势函数与回报目标。
    """

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.n_steps, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
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
        """
        从最后一步 t = n_steps - 1 逆序向前递推计算广义优势估计 (GAE)：
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
        注：对于超时截断 (truncated=True)，在存入 reward 时已按照 SB3 标准加上了 γ * V(s_final)，
        因此此处 (1 - done_t) 置 0 即可同时阻断跨回合优势传播并保留正确的截断自举价值。
        """
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
        # Critic 的回归目标值：R_t = A_t + V(s_t)
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
    env_id = "CartPole-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "ppo_cartpole"
    log_dir = base_dir / "logs" / "ppo_cartpole"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：创建矢量化并行环境与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 {args.n_envs} 个并行的 Gymnasium 环境: {env_id}")
    # SyncVectorEnv 会同步执行多个子环境（对应 SB3 的 DummyVecEnv + make_vec_env）
    train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)
    # 验证环境保持单个环境即可
    eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

    print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
    print(f"  观测空间 (Observation Space): {train_env.single_observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.single_action_space}")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置（CleanRL 白盒实例化）
    # ==========================================================================
    print("\n[第 2 步] 初始化 CleanRL PPO 模型（ActorCriticPolicy + RolloutBuffer）与核心超参数...")

    # 对于 Actor-Critic 架构（PPO / A2C），net_arch 用字典分别指定策略网络(pi)和价值网络(vf)的隐藏层
    # 这样 Actor 和 Critic 不共享特征层，在许多控制任务上训练更稳定
    net_arch = dict(pi=[64, 64], vf=[64, 64])

    # --- 学习率与采样批次控制（与 SB3 02_ppo_cartpole.py 严格 1:1 对齐）---
    learning_rate = 3e-4       # PPO 最经典的默认学习率（3e-4）
    n_steps = 512              # 每个并行环境每次采集多少步数据后触发一次网络更新
    batch_size = 64            # 小批次大小：必须尽量能整除 (n_steps * n_envs)
    n_epochs = 10              # 每次采集完一批 Rollout 数据后，重复遍历训练 10 轮 (epochs)

    # --- 折扣因子与广义优势估计 (GAE) ---
    gamma = 0.99               # 折扣因子 γ
    gae_lambda = 0.95          # GAE 平滑参数 λ ∈ [0, 1]：用于平衡优势估计的方差与偏差

    # --- PPO 核心裁剪、联合 Loss 系数与工程稳定项 ---
    clip_range = 0.2           # 对应伪代码裁剪超参数 ε：限制新旧策略概率比在 [1-ε, 1+ε] = [0.8, 1.2]
    ent_coef = 0.01            # 熵正则化系数：总 Loss 中减去 ent_coef*Entropy，鼓励探索
    vf_coef = 0.5              # 价值函数 (Critic) Loss 权重：平衡 Critic 梯度与 Actor 梯度尺度
    max_grad_norm = 0.5        # 梯度裁剪：反向传播后若梯度 L2 范数超过 0.5 则等比缩放

    rollout_buffer_size = n_steps * args.n_envs
    print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

    obs_dim = int(np.prod(train_env.single_observation_space.shape))
    n_actions = int(train_env.single_action_space.n)

    policy = ActorCriticPolicy(obs_dim=obs_dim, n_actions=n_actions, net_arch=net_arch).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    rollout_buffer = RolloutBuffer(
        n_steps=n_steps,
        n_envs=args.n_envs,
        obs_shape=train_env.single_observation_space.shape,
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
    # 第 4 步：启动 CleanRL PPO 白盒训练主循环与定期评估
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 PPO（总步数 total_timesteps={args.timesteps}）...")
    # 在 VecEnv 下，SB3 EvalCallback 的 eval_freq 是按“每次调用 env.step()（即 n_envs 步一起算 1 次）”计数的
    eval_freq_vec_steps = max(500, (args.timesteps // 5) // args.n_envs)
    best_mean_reward = -float("inf")
    best_model_path = model_dir / "best_model.pt"

    tb_run_dir = get_next_run_dir(log_dir, "PPO_CartPole")
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
        # 阶段 A：并行环境采样 (Collect Rollouts)
        # ----------------------------
        rollout_buffer.reset()
        for _ in range(n_steps):
            vec_step_count += 1
            global_step += args.n_envs

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions, log_probs, _, values = policy.get_action_and_value(obs_tensor)

            cpu_actions = actions.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = train_env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)

            # 关键细节：处理超时截断 (truncated=True) 的自举价值修正（与 SB3 RolloutBuffer 完全一致）
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

            # 记录已结束子环境的回合总奖励和长度
            if "final_info" in infos and "episode" in infos["final_info"]:
                ep_info = infos["final_info"]["episode"]
                mask = infos["final_info"].get("_episode", np.ones(args.n_envs, dtype=bool))
                for idx in range(args.n_envs):
                    if mask[idx]:
                        recent_ep_rewards.append(float(ep_info["r"][idx]))
                        recent_ep_lengths.append(int(ep_info["l"][idx]))
            elif "episode" in infos:
                ep_info = infos["episode"]
                mask = infos.get("_episode", np.ones(args.n_envs, dtype=bool))
                for idx in range(args.n_envs):
                    if mask[idx]:
                        recent_ep_rewards.append(float(ep_info["r"][idx]))
                        recent_ep_lengths.append(int(ep_info["l"][idx]))

            rollout_buffer.add(
                obs=obs,
                action=cpu_actions,
                reward=step_rewards,
                done=dones.astype(np.float32),
                value=values.flatten().cpu().numpy(),
                log_prob=log_probs.cpu().numpy(),
            )
            obs = next_obs

            # 定期在独立 eval_env 上评估当前策略（对应 SB3 EvalCallback）
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

        # 计算最后一步之后的 Bootstrap Value V(s_{T}) 并递推计算 GAE 优势与回报
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
        b_actions = torch.as_tensor(rollout_buffer.actions.reshape(-1), device=device)
        b_logprobs = torch.as_tensor(rollout_buffer.log_probs.reshape(-1), device=device)
        b_advantages = torch.as_tensor(rollout_buffer.advantages.reshape(-1), device=device)
        b_returns = torch.as_tensor(rollout_buffer.returns.reshape(-1), device=device)

        b_inds = np.arange(rollout_buffer_size)
        pg_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

        for _ in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, rollout_buffer_size, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Minibatch 优势标准化（均值为 0、标准差为 1，大幅提升策略梯度稳定性）
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 1. PPO Clipped Surrogate Loss（取负号用于梯度下降最小化）
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 2. Critic 状态价值均方误差损失
                v_loss = F.mse_loss(new_value.flatten(), b_returns[mb_inds])

                # 3. 策略熵正则项（鼓励探索）
                entropy_loss = -entropy.mean()

                # 4. 联合总损失：L_total = L_clip + vf_coef * L_vf + ent_coef * (-Entropy)
                loss = pg_loss + vf_coef * v_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_range).float().mean().item()))

        # 记录 TensorBoard 训练指标
        if recent_ep_rewards:
            writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
            writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
        writer.add_scalar("train/policy_gradient_loss", float(np.mean(pg_losses)), global_step)
        writer.add_scalar("train/value_loss", float(np.mean(value_losses)), global_step)
        writer.add_scalar("train/entropy_loss", float(np.mean(entropy_losses)), global_step)
        writer.add_scalar("train/clip_fraction", float(np.mean(clip_fractions)), global_step)
        writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

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
    final_model_path = model_dir / "ppo_cartpole_final.pt"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    torch.save(policy.state_dict(), final_model_path)

    del policy
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = ActorCriticPolicy(obs_dim=obs_dim, n_actions=n_actions, net_arch=net_arch).to(device)
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
        # deterministic=True：取策略分布中概率最大的动作，评估/部署时通常设为 True
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        if terminated or truncated:
            reason = "达到最大步数截断 (truncated)" if truncated else "摆杆倾倒/出界终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
