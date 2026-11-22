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
本脚本的 `main()` 主函数与 `stable-baseline3/02_ppo_cartpole.py` **逐行 100% 一致**，
并在文件上半部分将 SB3 封装的高层组件用纯 PyTorch 进行了透明白盒实现：
- `make_vec_env(env_id, n_envs=4)` <--> `gym.vector.SyncVectorEnv` (SAME_STEP 自动重置模式 + 单环境空间属性对齐)
- `ActorCriticPolicy`              <--> 显式定义正交初始化双分支解耦网络 (pi=[64,64], vf=[64,64])
- `RolloutBuffer`                  <--> 显式实现 GAE 逆序递推与 `truncated` 超时截断自举修正
- `PPO` (`learn`/`save`/`load`)    <--> 显式展开并行环境 Rollout 采样、Minibatch 打乱与 PPO-Clip 联合损失反传
- `EvalCallback` & `evaluate_policy` <--> 定期评估并自动保存 `best_model.zip` 与 `evaluations.npz`

运行方式：
    python 02_ppo_cartpole.py                  # 使用默认步数 (50000 步) 完整训练
    python 02_ppo_cartpole.py --timesteps 15000 # 快速测试全流程
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
from torch.distributions.categorical import Categorical
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
    """
    CleanRL 白盒实现的并行环境构造函数（对应 SB3 `make_vec_env`）：
    为每个子环境套上 RecordEpisodeStatistics 包装器，并使用 SAME_STEP 自动重置模式，
    确保子环境结束时立即重置并将终止状态保存在 infos["final_obs"] 中。
    """
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
# CleanRL 白盒组件 1：正交初始化与 ActorCriticPolicy 双分支解耦网络
# 严格对应 SB3 `ActorCriticPolicy(net_arch=dict(pi=[64, 64], vf=[64, 64]))`
# ==============================================================================
def layer_init(layer, std = float(np.sqrt(2)), bias_const = 0.0):
    """
    PPO 核心工程技巧：正交初始化 (Orthogonal Initialization)。
    - 隐藏层使用 gain=sqrt(2) 配合 Tanh 激活函数保持方差稳定；
    - 策略输出头 action_net 使用极小的 gain=0.01，使初始 Logits 接近 0，初始动作分布接近均匀分布（最大熵探索）；
    - 价值输出头 value_net 使用 gain=1.0。
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(in_dim, hidden_dims, out_dim, out_std):
    """构建带 Tanh 激活与正交初始化的多层感知机（MLP）。"""
    layers = []
    for h_dim in hidden_dims:
        layers.append(layer_init(nn.Linear(in_dim, h_dim), std=float(np.sqrt(2))))  # 隐藏层用 sqrt(2) 补偿 Tanh 压缩
        layers.append(nn.Tanh())
        in_dim = h_dim
    layers.append(layer_init(nn.Linear(in_dim, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCriticPolicy(nn.Module):
    """
    离散动作空间 Actor-Critic 策略网络：
    - self.actor:  (batch, obs_dim) -> (batch, n_actions)，输出动作 Logits 经 Softmax 构成策略分布 π(a|s, θ)
    - self.critic: (batch, obs_dim) -> (batch, 1)，输出状态价值估计 v(s, w)
    """

    def __init__(self, obs_dim, n_actions, net_arch = None):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch = list(net_arch.get("pi", [64, 64]))
            vf_arch = list(net_arch.get("vf", [64, 64]))
        elif isinstance(net_arch, list):
            pi_arch = list(net_arch)
            vf_arch = list(net_arch)
        else:
            pi_arch, vf_arch = [64, 64], [64, 64]

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.actor = _build_mlp(obs_dim, pi_arch, n_actions, out_std=0.01)  # 输出头用极小 0.01，初始概率接近均匀探索
        self.critic = _build_mlp(obs_dim, vf_arch, 1, out_std=1.0)           # 价值输出头直接输出实数 V(s)

    def get_value(self, obs):
        """对应伪代码价值函数 v(s, w_k)：仅计算状态价值，输出形状 (batch_size, 1)。"""
        features = self.flatten(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action = None):
        """
        CleanRL 经典四合一前向接口（同时覆盖伪代码第 1 步采样与第 3 步重评估）：
        - action=None 时（第 1 步）：按 π(·|s, θ_k) 采样动作，返回 (action, log_prob, entropy, value)
        - 传入历史 action 时（第 3 步）：在新参数 (θ', w') 下重新计算该动作的 log_prob、entropy 与 value
        """
        features = self.flatten(obs)
        logits = self.actor(features)          # (batch, n_actions)
        dist = Categorical(logits=logits)      # 策略分布 π(·|s, θ)，Categorical 内部自动做 Softmax
        if action is None:
            action = dist.sample()             # 采样动作 a ~ π
        return action, dist.log_prob(action), dist.entropy(), self.critic(features)

    def predict(self, obs, deterministic = True):
        """评估/演示推理接口（同时兼容单环境 1D 与并行环境 2D 输入）。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            single_env = obs_tensor.ndim == 1
            if single_env:
                obs_tensor = obs_tensor.unsqueeze(0)

            logits = self.actor(self.flatten(obs_tensor))
            actions = torch.argmax(logits, dim=-1) if deterministic else Categorical(logits=logits).sample()

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

    def __init__(self, n_steps, n_envs, obs_shape, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)  # GAE 优势估计 A_t
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
        """
        从最后一步 t = n_steps - 1 逆序向前递推计算广义优势估计 (GAE)：
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
        注：对于超时截断 (truncated=True)，在存入 reward 时已按照 SB3 标准加上了 γ * V(s_final)，
        因此此处 (1 - done_t) 置 0 即可同时阻断跨回合优势传播并保留正确的截断自举价值。
        """
        last_gae_lam = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.n_steps)):  # 从最后一步开始，倒序往前遍历每一步
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
    """
    CleanRL 白盒实现的定期评估回调（对应 SB3 `EvalCallback`）：
    在 VecEnv 下，n_calls 按每次调用并行环境 env.step()（即 n_envs 步一起算 1 次）计数，
    每隔 eval_freq 次调用在 eval_env 上评估一次，保存 evaluations.npz 与 best_model.zip。
    """

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
# CleanRL 白盒组件 4：PPO 轨迹收集、截断自举、损失更新与 PPO 主类
# ==============================================================================
def bootstrap_truncated_rewards(rewards, terminations, truncations, infos, policy, gamma, device):
    """
    超时截断 (truncated=True 且 terminated=False) 的自举价值修正：
    将终态价值 γ * V(s_final) 加回本步奖励，避免因 done=1 误把未失败截断的未来价值清零。
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
    CleanRL 白盒实现的离散动作 PPO 算法类（接口与 `stable_baselines3.PPO` 完全一致）：
    封装 `collect_rollouts`（并行采样 + GAE 计算）与 `update_policy`（小批次 PPO-Clip 联合更新）。
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
        policy_kwargs = None,
        tensorboard_log = None,
        seed = None,
        verbose = 1,
        _obs_shape = None,
        _n_actions = None,
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
            self.n_actions = int(act_space.n)
        else:
            assert _obs_shape is not None and _n_actions is not None
            self.n_envs = 1
            self.obs_shape = tuple(_obs_shape)
            self.n_actions = int(_n_actions)

        self.obs_dim = int(np.prod(self.obs_shape))
        self.net_arch = self.policy_kwargs.get("net_arch", dict(pi=[64, 64], vf=[64, 64]))

        self.policy = ActorCriticPolicy(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            net_arch=self.net_arch,
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            obs_shape=self.obs_shape,
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
        1. 用当前策略 π(·|s, θ_k) 在 n_envs 个并行环境各交互 n_steps 步，收集轨迹数据集 D_k；
        2. 采样结束后用 v(s_T, w_k) 作为引导值，通过 GAE 计算优势 A_t 与回报目标 R_t。
        """
        assert self.env is not None
        self.rollout_buffer.reset()
        for _ in range(self.n_steps):
            global_step += self.n_envs

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # (n_envs, 4)
                actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)

            cpu_actions = actions.cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = self.env.step(cpu_actions)
            dones = np.logical_or(terminations, truncations)

            step_rewards = bootstrap_truncated_rewards(
                rewards, terminations, truncations, infos, self.policy, self.gamma, self.device
            )
            record_episode_stats(infos, self.n_envs, recent_ep_rewards, recent_ep_lengths)

            self.rollout_buffer.add(
                obs=obs,
                action=cpu_actions,
                reward=step_rewards,
                done=dones.astype(np.float32),
                value=values.flatten().cpu().numpy(),
                log_prob=log_probs.cpu().numpy(),
            )
            obs = next_obs

            if callback is not None:
                callback.on_step(self, global_step, writer=writer)

        # 伪代码第 2 步：估算末尾状态价值 v(s_T, w_k) 并逆序计算 GAE 优势与回报目标
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            last_values = self.policy.get_value(next_obs_tensor).flatten().cpu().numpy()
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        return obs, global_step

    def _update_policy(self):
        """
        对应 PPO 伪代码【第 3 步】：
        展平轨迹缓冲区 (n_steps * n_envs, ...)，在 n_epochs 轮内随机打乱切分为小批次，
        联合最小化 PPO-Clip 策略损失、Critic 均方误差损失与策略熵正则项。
        """
        buffer_size = self.rollout_buffer.n_steps * self.rollout_buffer.n_envs
        b_obs = torch.as_tensor(
            self.rollout_buffer.observations.reshape((-1, *self.rollout_buffer.obs_shape)), device=self.device
        )
        b_actions = torch.as_tensor(self.rollout_buffer.actions.reshape(-1), device=self.device)
        b_logprobs = torch.as_tensor(self.rollout_buffer.log_probs.reshape(-1), device=self.device)
        b_advantages = torch.as_tensor(self.rollout_buffer.advantages.reshape(-1), device=self.device)
        b_returns = torch.as_tensor(self.rollout_buffer.returns.reshape(-1), device=self.device)

        b_inds = np.arange(buffer_size)
        pg_losses, value_losses, entropy_losses, clip_fractions = [], [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, buffer_size, self.batch_size):
                mb_inds = b_inds[start : start + self.batch_size]

                _, new_logprob, entropy, new_value = self.policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                ratio = (new_logprob - b_logprobs[mb_inds]).exp()  # π(a|s, θ') / π(a|s, θ_k)

                # Minibatch 优势标准化（均值 0、标准差 1），提升策略梯度稳定性
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 1. Actor PPO-Clip 损失（伪代码目标函数取负号以便梯度下降最小化）
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 2. Critic 价值均方误差损失：MSE(v(s, w'), v_hat)
                v_loss = F.mse_loss(new_value.flatten(), b_returns[mb_inds])

                # 3. 策略熵正则项（最大化熵等价于最小化 -entropy）
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
            # 阶段 A（伪代码第 1-2 步）：并行环境采样 + 计算 GAE 优势与回报目标
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
                writer.add_scalar("time/fps", int(global_step / max(1e-6, time.time() - start_time)), global_step)

        if writer is not None:
            writer.close()
        return self

    def save(self, path):
        checkpoint = {
            "policy_name": self.policy_name,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
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
            n_steps=checkpoint.get("n_steps", 512),
            batch_size=checkpoint.get("batch_size", 64),
            n_epochs=checkpoint.get("n_epochs", 10),
            gamma=checkpoint.get("gamma", 0.99),
            gae_lambda=checkpoint.get("gae_lambda", 0.95),
            clip_range=checkpoint.get("clip_range", 0.2),
            ent_coef=checkpoint.get("ent_coef", 0.01),
            vf_coef=checkpoint.get("vf_coef", 0.5),
            max_grad_norm=checkpoint.get("max_grad_norm", 0.5),
            policy_kwargs=checkpoint.get("policy_kwargs", dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))),
            verbose=0,
            _obs_shape=checkpoint.get("obs_shape"),
            _n_actions=checkpoint.get("n_actions"),
        )
        model.policy.load_state_dict(checkpoint["state_dict"])
        model.policy.eval()
        return model


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SB3 PPO 并行环境训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练步数（注意：在 4 个并行环境下，50000 步实际只需约 12500 次环境并行 step）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量（默认 4）")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "CartPole-v1"

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
    # make_vec_env 是 SB3 的神器：它会自动为每个环境套上 Monitor 包装器，并打包成 VecEnv
    # 默认使用 DummyVecEnv（单进程顺序执行多个环境，对轻量级环境如 CartPole 开销最小）
    # 若环境物理仿真很重（如 MuJoCo/PyBullet），可传入 vec_env_cls=SubprocVecEnv 开启真正的多进程并行
    train_env = make_vec_env(env_id, n_envs=args.n_envs, seed=args.seed)

    # 验证环境保持单个环境即可
    eval_env = Monitor(gym.make(env_id))

    print(f"  并行环境数 (num_envs):        {train_env.num_envs}")
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置
    # ==========================================================================
    print("\n[第 2 步] 初始化 PPO 模型与核心超参数...")

    # 对于 Actor-Critic 架构（PPO / A2C），net_arch 可以用字典分别指定策略网络(pi)和价值网络(vf)的隐藏层
    # 这样 Actor 和 Critic 不共享特征层，在许多控制任务上训练更稳定
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    # 攒够这 2048 条数据后，PPO 会把它们打乱；
    #按照 batch_size = 64 切分成 2048 / 64 = 32 个小批次，依次做 32 次梯度更新（这算 1 个 epoch）；
    #重复这个过程 n_epochs = 10 轮，也就是这批数据总共会触发 32 × 10 = 320 次网络参数更新；
    #更新完后清空 RolloutBuffer（因为 PPO 是 On-Policy 算法，旧策略的数据不能留用），再重新采样下一批 2048 条数据。
    n_steps = 512
    batch_size = 64
    rollout_buffer_size = n_steps * args.n_envs
    print(f"  每次采样迭代 RolloutBuffer 大小 = n_steps({n_steps}) * n_envs({args.n_envs}) = {rollout_buffer_size}")

    model = PPO( 
        policy="MlpPolicy",
        env=train_env,

        # --- 学习率与采样批次控制 ---
        learning_rate=3e-4,       # PPO 最经典的默认学习率（3e-4）
        n_steps=n_steps,          # 每个并行环境每次采集多少步数据后触发一次网络更新
        batch_size=batch_size,    # 小批次大小：必须尽量能整除 (n_steps * n_envs)
        n_epochs=10,              # 每次采集完一批 Rollout 数据后，重复遍历训练 10 轮 (epochs)

        # --- 折扣因子与广义优势估计 (GAE) ---
        gamma=0.99,               # 折扣因子 γ
        gae_lambda=0.95,          # GAE 平滑参数 λ ∈ [0, 1]：用于平衡优势估计的方差与偏差（0.9~0.98 常用）

        # --- PPO 核心裁剪、联合 Loss 系数与工程稳定项 ---
        # 注：教材基础伪代码常把 Actor/Critic 拆成两个独立优化器，故只会出现 ε (clip_range)；
        # SB3 工程实现使用单个 Adam 优化器优化联合总损失：
        #   L_total = L_clip + vf_coef * L_vf - ent_coef * Entropy
        clip_range=0.2,           # 对应伪代码裁剪超参数 ε：限制新旧策略概率比在 [1-ε, 1+ε] = [0.8, 1.2]，防止策略剧烈震荡
        ent_coef=0.01,            # 熵正则化系数（教材极简版常省略）：总 Loss 中减去 ent_coef*Entropy，鼓励探索、防止策略过早塌缩到次优动作
        vf_coef=0.5,              # 价值函数 (Critic) Loss 权重：单优化器共用 learning_rate 时，用于平衡 Critic 梯度与 Actor 梯度的尺度
        max_grad_norm=0.5,        # 工程防爆技巧（梯度裁剪）：反向传播后若梯度 L2 范数超过 0.5 则等比缩放，防止异常样本冲坏权重

        # --- 网络架构与日志 ---
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
    print(f"\n[第 4 步] 开始训练 PPO（总步数 total_timesteps={args.timesteps}）...")
    # 注意：在 VecEnv 下，EvalCallback 的 eval_freq 是按“每次调用 env.step()（即 n_envs 步一起算 1 次）”计数的！
    # 因此若希望大约每 total_timesteps / 5 个总步数评估一次，需要除以 n_envs
    eval_freq = max(500, (args.timesteps // 5) // args.n_envs)
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
        tb_log_name="PPO_CartPole",
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
    final_model_path = model_dir / "ppo_cartpole_final.zip"
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
        # deterministic=True：取策略分布中概率最大的动作（或连续动作的均值），评估/部署时通常设为 True
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
