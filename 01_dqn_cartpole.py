"""
================================================================================
CleanRL 实战教程 01：DQN (Deep Q-Network) 求解 CartPole-v1
================================================================================

【算法定位】
- 算法名称：DQN (深度 Q 网络 —— CleanRL 单文件 PyTorch 白盒实现)
- 策略类型：Off-Policy（异策略）—— 可以把历史交互经验存入“经验回放池”并重复利用
- 学习范式：Value-Based（基于价值）—— 不直接输出动作概率，而是估计每个动作的 Q 值
- 支持动作空间：**仅支持离散动作空间 (Discrete)**，例如“向左/向右”、“上/下/左/右”

【为什么 DQN 只能处理离散动作空间？】
DQN 更新网络时的目标值（TD Target）计算公式为：
    y = r + gamma * (1 - terminated) * max_{a'} Q_target(s', a')
注意其中的 `max_{a'}` 操作：它需要把下一状态 s' 下所有可能动作的 Q 值全部算出来再取最大值。
如果是连续动作空间（比如力矩在 [-2.0, 2.0] 之间有无穷多个实数），根本无法穷举求 max。
因此连续动作空间需要使用 PPO、TD3 或 SAC 等 Actor-Critic 算法。

【DQN 的两大核心稳定机制】
1. 经验回放池 (Replay Buffer)：打破连续时间步样本之间的强相关性，让数据近似独立同分布（i.i.d.），并大幅提升样本利用率。
2. 目标网络 (Target Network)：使用一个参数更新较慢的目标 Q 网络计算 TD Target，避免“自己追自己”引起的训练震荡与发散。

【CleanRL 白盒实现 vs Stable-Baselines3 (SB3) 一一对应关系】
本脚本的 `main()` 主函数与 `stable-baseline3/01_dqn_cartpole.py` **逐行 100% 一致**，
并在文件上半部分将 SB3 封装的高层类用纯 PyTorch 进行了透明白盒实现：
- `Monitor(env)`                  <--> `gym.wrappers.RecordEpisodeStatistics(env)`
- `ReplayBuffer`                  <--> 显式实现环形 NumPy/Tensor 经验回放池
- `QNetwork` & `DQNPolicy`        <--> 显式定义主 Q 网络 `q_net` 与目标 Q 网络 `q_net_target`
- `DQN` (`learn`/`save`/`load`)   <--> 显式展开环境交互、ε-greedy 衰减、Huber TD Loss 反传与软/硬更新
- `EvalCallback` & `evaluate_policy` <--> 定期在 `eval_env` 上评估并保存 `best_model.zip` 与 `evaluations.npz`

【实验环境：CartPole-v1（倒立摆车）】
- 观测空间 (Observation Space): Box(4,) —— [小车位置, 小车速度, 摆杆角度, 摆杆角速度]
- 动作空间 (Action Space): Discrete(2) —— 0 代表向左推小车，1 代表向右推小车
- 奖励机制 (Reward): 摆杆每保持直立 1 步，获得 +1 奖励；每回合最多 500 步（满分 500 分）

运行方式：
    python 01_dqn_cartpole.py                  # 使用默认步数 (50000 步) 完整训练
    python 01_dqn_cartpole.py --timesteps 15000 # 快速测试全流程
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
    """固定 Python、NumPy 与 PyTorch 随机种子，保证实验结果可复现。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def Monitor(env):
    """
    CleanRL 白盒实现的 Monitor 包装器（对应 SB3 `stable_baselines3.common.monitor.Monitor`）：
    使用 Gymnasium 官方的 `RecordEpisodeStatistics` 自动记录每个回合的总奖励 `info["episode"]["r"]` 和长度 `info["episode"]["l"]`。
    """
    return gym.wrappers.RecordEpisodeStatistics(env)


# ==============================================================================
# CleanRL 白盒组件 1：经验回放池 (Replay Buffer)
# 对应 SB3 内部封装的 `stable_baselines3.common.buffers.ReplayBuffer`
# ==============================================================================
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """固定容量的环形经验回放池，存储 (s, a, r, s', terminated) 五元组转移样本。"""

    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, terminated):
        """写入一条单步转移经验；注意 done 仅记录真实终止 terminated（超时截断 truncated 不置零）。"""
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
        """均匀随机采样 batch_size 条经验并转换为 PyTorch 张量。"""
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
# CleanRL 白盒组件 2：Q 网络与 DQNPolicy 封装
# 严格对应 SB3 `DQNPolicy(net_arch=[128, 128])` 的底层 PyTorch 模块树
# ==============================================================================
class QNetwork(nn.Module):
    """
    多层感知机 Q 网络：输入状态 s ∈ R^4，一次前向传播并行输出所有离散动作的 Q 值 [Q(s, a_0), Q(s, a_1)]。
    注意：隐藏层使用 ReLU 激活，输出层绝不能加激活函数（因为 Q 值可能是任意实数）。
    """

    def __init__(self, obs_dim, n_actions, net_arch):
        super().__init__()
        # 保留第 0 维（batch 批量维），把第 1 维到最后一维拉成一条一维向量。
        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)
        layers = []
        in_dim = obs_dim  # 当前层的输入维度 4
        # net_arch：[128, 128]，这里循环 2 次，hidden_dim 两次都为 128。
        for hidden_dim in net_arch:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))  # n_actions（2）
        # 使用 *layers 把列表里的 5 个对象展开作为独立参数传给 nn.Sequential，串联并赋值给 self.q_net
        self.q_net = nn.Sequential(*layers)

    def forward(self, obs):
        features = self.features_extractor(obs)  # (batch_size, 4) -> (batch_size, 4)
        return self.q_net(features)  # (batch_size, 4) -> (batch_size, 2)


class DQNPolicy(nn.Module):
    """
    包含主 Q 网络 (q_net) 与目标 Q 网络 (q_net_target) 的完整策略容器。
    提供与 SB3 完全一致的 `.predict(obs, deterministic=True)` 推理接口。
    4 维状态输入 -> 128(ReLU) -> 128(ReLU) -> 2 个 Q 值输出
    """

    def __init__(self, obs_dim, n_actions, net_arch):
        super().__init__()
        """
        每次创建时 PyTorch 都会随机生成一套初始权重，
        所以这两个网络刚建好时内部参数是不同的，需要手动同步一次起点。
        """
        self.q_net = QNetwork(obs_dim, n_actions, net_arch)
        self.q_net_target = QNetwork(obs_dim, n_actions, net_arch)
        # 初始时刻将主网络权重完整复制到目标网络
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def predict(
        self,
        obs,
        deterministic = True,
        epsilon = 0.0,
        action_space = None,
    ):
        """
        根据当前观测选择动作：评估时 deterministic=True 贪婪选最大 Q 值；
        训练时 deterministic=False 按 ε-greedy 探索。
        """
        if not deterministic and random.random() < epsilon:
            assert action_space is not None
            return int(action_space.sample()), None  # 等概率随机抽取一个合法动作。

        device = next(self.parameters()).device
        with torch.no_grad():  # 临时关闭 PyTorch 的自动求导，只是在做推理选动作。
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # 把形状从 (4,) 变成 (1, 4)
            q_values = self.q_net(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action, None


def linear_schedule(initial_eps, final_eps, duration, current_step):
    """计算 ε-greedy 的线性衰减探索率 ε。"""
    slope = (final_eps - initial_eps) / max(1, duration)
    return max(final_eps, initial_eps + slope * current_step)


# ==============================================================================
# CleanRL 白盒组件 3：策略评估函数 (evaluate_policy) 与评估回调 (EvalCallback)
# 对应 SB3 `evaluate_policy` 与 `EvalCallback`
# ==============================================================================
def evaluate_policy(
    model,
    eval_env,
    n_eval_episodes = 10,
    deterministic = True,
    return_episode_rewards = False,
):
    """
    CleanRL 白盒实现的策略评估函数（对应 SB3 `evaluate_policy`）：
    在独立的 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差。
    """
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
    CleanRL 白盒实现的定期评估回调（对应 SB3 `stable_baselines3.common.callbacks.EvalCallback`）：
    每隔 eval_freq 次环境 step，在 eval_env 上评估当前策略，
    记录 evaluations.npz 并在平均奖励刷新最高记录时自动保存 best_model.zip。
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
# CleanRL 白盒组件 4：DQN 算法主类（与 SB3 `DQN` 接口 1:1 完全对齐）
# 将 CleanRL 的采样、ε-greedy 探索、TD Target 计算、Huber Loss 更新封装进 `.learn()`
# ==============================================================================
class DQN:
    """
    CleanRL 白盒实现的 DQN 算法类（接口与 `stable_baselines3.DQN` 完全一致）：
    对外暴露 `__init__`、`learn`、`predict`、`save`、`load` 方法；
    内部直接展示完整的 CleanRL 单文件 PyTorch 训练循环。
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate = 1e-3,
        gamma = 0.99,
        buffer_size = 50_000,
        learning_starts = 1_000,
        batch_size = 64,
        train_freq = 4,
        gradient_steps = 1,
        target_update_interval = 500,
        tau = 1.0,
        exploration_initial_eps = 1.0,
        exploration_fraction = 0.2,
        exploration_final_eps = 0.05,
        max_grad_norm = 10.0,
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
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.policy_kwargs = policy_kwargs or {}
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if env is not None:
            self.obs_shape = tuple(env.observation_space.shape)
            self.n_actions = int(env.action_space.n)
        else:
            assert _obs_shape is not None and _n_actions is not None
            self.obs_shape = tuple(_obs_shape)
            self.n_actions = int(_n_actions)

        self.obs_dim = int(np.prod(self.obs_shape))
        self.net_arch = list(self.policy_kwargs.get("net_arch", [64, 64]))

        # 初始化主 Q 网络与目标 Q 网络（DQNPolicy）及 Adam 优化器、经验回放池
        self.policy = DQNPolicy(obs_dim=self.obs_dim, n_actions=self.n_actions, net_arch=self.net_arch).to(self.device)
        self.optimizer = optim.Adam(self.policy.q_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            obs_shape=self.obs_shape,
            device=self.device,
        )

    def predict(self, obs, deterministic = True):
        """推理接口：返回 (action, state)，与 SB3 `model.predict(obs, deterministic=True)` 完全一致。"""
        return self.policy.predict(obs, deterministic=deterministic)

    def learn(
        self,
        total_timesteps,
        callback = None,
        tb_log_name = "DQN",
        progress_bar = False,
    ):
        """
        CleanRL DQN 核心白盒训练主循环（对应 SB3 `model.learn()`）：
        1. 按线性衰减计算当前探索率 ε，执行 ε-greedy 动作选择；
        2. 与环境交互一步并将 (s, a, r, s', terminated) 存入 ReplayBuffer；
        3. 预热期后每隔 train_freq 步采样小批次，计算 TD Target 与 Smooth L1 (Huber) Loss 更新主 Q 网络；
        4. 每隔 target_update_interval 步将主 Q 网络权重同步至目标 Q 网络；
        5. 每步触发 EvalCallback 进行定期评估与最优模型保存。
        """
        assert self.env is not None, "训练前必须绑定环境 env"
        writer = None
        if self.tensorboard_log is not None:
            tb_run_dir = Path(self.tensorboard_log) / tb_log_name
            tb_run_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_run_dir))

        obs, _ = self.env.reset(seed=self.seed)
        start_time = time.time()
        exploration_duration = int(self.exploration_fraction * total_timesteps)
        recent_ep_rewards = []
        recent_ep_lengths = []

        for global_step in range(1, total_timesteps + 1):
            # 1. 计算当前步的 ε-greedy 探索率并选择动作
            epsilon = linear_schedule(
                self.exploration_initial_eps,
                self.exploration_final_eps,
                exploration_duration,
                global_step,
            )
            if global_step <= self.learning_starts:
                action = int(self.env.action_space.sample())
            else:
                action, _ = self.policy.predict(
                    obs,
                    deterministic=False,
                    epsilon=epsilon,
                    action_space=self.env.action_space,
                )

            # 2. 与环境交互一步，获取五元组
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # 3. 存入经验回放池：注意只有真正失败 terminated=True 才将未来价值置零！
            self.replay_buffer.add(obs, next_obs, action, float(reward), terminated)
            obs = next_obs

            # 若当前回合结束（terminated 或 truncated），记录统计信息并重置环境
            if terminated or truncated:
                if "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    ep_l = int(info["episode"]["l"])
                    recent_ep_rewards.append(ep_r)
                    recent_ep_lengths.append(ep_l)
                    if writer is not None:
                        writer.add_scalar("rollout/ep_rew_mean", float(np.mean(recent_ep_rewards[-100:])), global_step)
                        writer.add_scalar("rollout/ep_len_mean", float(np.mean(recent_ep_lengths[-100:])), global_step)
                        writer.add_scalar("rollout/exploration_rate", epsilon, global_step)
                obs, _ = self.env.reset()

            # 4. 达到预热步数后，每隔 train_freq 步执行一次 TD 梯度更新
            if global_step > self.learning_starts and global_step % self.train_freq == 0:
                loss = torch.tensor(0.0, device=self.device)
                current_q_values = torch.tensor(0.0, device=self.device)
                for _ in range(self.gradient_steps):
                    batch = self.replay_buffer.sample(self.batch_size)

                    # (a) 使用目标 Q 网络计算 TD Target（不计算梯度）：
                    #     y = r + γ * (1 - terminated) * max_{a'} Q_target(s', a')
                    with torch.no_grad():
                        next_q_values = self.policy.q_net_target(batch.next_observations)
                        max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
                        td_target = batch.rewards + (1.0 - batch.dones) * self.gamma * max_next_q

                    # (b) 使用主 Q 网络计算当前执行动作 a 对应的预测值 Q(s, a)
                    current_q_values = self.policy.q_net(batch.observations).gather(1, batch.actions)

                    # (c) 计算 Smooth L1 (Huber) 损失并反向传播更新主网络（与 SB3 DQN 完全一致）
                    loss = F.smooth_l1_loss(current_q_values, td_target)
                    self.optimizer.zero_grad()
                    loss.backward()  # 梯度只会传回主网络 policy.q_net
                    nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # 每隔 100 步向 TensorBoard 记录一次训练状态指标
                if writer is not None and global_step % 100 == 0:
                    sps = int(global_step / max(1e-6, time.time() - start_time))
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/q_values", current_q_values.mean().item(), global_step)
                    writer.add_scalar("time/fps", sps, global_step)

            # 5. 每隔 target_update_interval 步同步一次目标网络权重
            if global_step % self.target_update_interval == 0:
                for param, target_param in zip(
                    self.policy.q_net.parameters(), self.policy.q_net_target.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            # 6. 触发回调函数（如 EvalCallback 定期评估与保存最优模型）
            if callback is not None:
                callback.on_step(self, global_step, writer=writer)

        if writer is not None:
            writer.close()
        return self

    def save(self, path):
        """将模型结构超参数与 PyTorch 权重打包保存到磁盘文件（原生 ZIP 容器格式）。"""
        checkpoint = {
            "policy_name": self.policy_name,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "tau": self.tau,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_fraction": self.exploration_fraction,
            "exploration_final_eps": self.exploration_final_eps,
            "policy_kwargs": self.policy_kwargs,
            "state_dict": self.policy.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, env = None):
        """从磁盘文件恢复 DQN 模型实例与网络权重（对应 SB3 `DQN.load(path, env=...)`）。"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            policy=checkpoint.get("policy_name", "MlpPolicy"),
            env=env,
            learning_rate=checkpoint.get("learning_rate", 1e-3),
            gamma=checkpoint.get("gamma", 0.99),
            buffer_size=checkpoint.get("buffer_size", 50_000),
            learning_starts=checkpoint.get("learning_starts", 1_000),
            batch_size=checkpoint.get("batch_size", 64),
            train_freq=checkpoint.get("train_freq", 4),
            gradient_steps=checkpoint.get("gradient_steps", 1),
            target_update_interval=checkpoint.get("target_update_interval", 500),
            tau=checkpoint.get("tau", 1.0),
            exploration_initial_eps=checkpoint.get("exploration_initial_eps", 1.0),
            exploration_fraction=checkpoint.get("exploration_fraction", 0.2),
            exploration_final_eps=checkpoint.get("exploration_final_eps", 0.05),
            policy_kwargs=checkpoint.get("policy_kwargs", {"net_arch": [128, 128]}),
            verbose=0,
            _obs_shape=checkpoint.get("obs_shape"),
            _n_actions=checkpoint.get("n_actions"),
        )
        model.policy.load_state_dict(checkpoint["state_dict"])
        model.policy.eval()
        return model


def parse_args():
    """解析命令行参数，方便指定训练总步数与随机种子。"""
    parser = argparse.ArgumentParser(description="SB3 DQN 训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练环境交互步数（默认 50000 步，CartPole-v1 通常在 2~5 万步即可接近满分）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证实验可复现")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "CartPole-v1"

    # 定义模型与日志保存目录（当前脚本所在目录下的 models/ 和 logs/）
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "dqn_cartpole"
    log_dir = base_dir / "logs" / "dqn_cartpole"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 第 1 步：环境初始化与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 环境: {env_id}")
    # Monitor 包装器用于自动记录每个回合 (Episode) 的总奖励和回合长度，供日志和回调函数使用
    train_env = Monitor(gym.make(env_id))
    # 独立创建验证环境，避免评估过程污染训练环境状态
    eval_env = Monitor(gym.make(env_id))

    # 查看观测空间与动作空间（新手写强化学习代码前务必先打印这两个属性！）
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  观测维度 (Shape):             {train_env.observation_space.shape}")
    print(f"  观测下界 (Low):               {train_env.observation_space.low}")
    print(f"  观测上界 (High):              {train_env.observation_space.high}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}  (离散动作数量 n={train_env.action_space.n})")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置
    # ==========================================================================
    print("\n[第 2 步] 初始化 DQN 模型与核心超参数...")

    # policy_kwargs 用于自定义神经网络结构
    # 对于 DQN，net_arch=[128, 128] 表示特征输入后接两层全连接网络，每层 128 个神经元（默认是 [64, 64]）
    policy_kwargs = dict(net_arch=[128, 128])

    model = DQN(
        # 策略类型："MlpPolicy" 表示多层感知机（全连接网络），适用于向量观测；
        # 若输入是图像像素，则需改为 "CnnPolicy"；若观测是字典类型，改为 "MultiInputPolicy"。
        policy="MlpPolicy",
        env=train_env,

        # --- 优化器与折扣因子 ---
        learning_rate=1e-3,             # 学习率：DQN 常用 1e-4 ~ 1e-3
        gamma=0.99,                     # 折扣因子 γ ∈ [0, 1)：越接近 1 越重视长远未来奖励

        # --- 经验回放池 (Replay Buffer) 相关参数 ---
        buffer_size=50_000,             # 经验池最大容量：最多存储 5 万条 (s, a, r, s', done) 转移样本
        learning_starts=1_000,          # 预热步数：先随机探索收集满 1000 条样本后再开始梯度更新，避免前期数据过少导致过拟合
        batch_size=64,                  # 每次更新从经验池中随机采样的批量大小

        # --- 训练频率与目标网络更新 ---
        train_freq=4,                   # 每与环境交互 4 步，执行一次训练更新
        gradient_steps=1,               # 每次触发训练时，做 1 次梯度下降更新
        target_update_interval=500,     # 目标网络同步间隔：每 500 个环境步将主 Q 网络权重同步到目标 Q 网络
        tau=1.0,                        # 软更新系数：tau=1.0 表示硬更新（直接覆盖复制权重）；若 <1.0 则做指数滑动平均

        # --- ε-greedy（epsilon-贪婪）探索策略衰减 ---
        # 智能体在训练初期以高概率随机探索，后期逐渐减小随机性并依赖学到的 Q 网络决策
        exploration_initial_eps=1.0,    # 初始探索率 ε=1.0（100% 随机动作）
        exploration_fraction=0.2,       # 在前 20% 的总训练步数内，将 ε 从 initial 线性衰减到 final
        exploration_final_eps=0.05,     # 最终探索率 ε=0.05（保留 5% 随机探索防止陷入局部最优）

        # --- 网络结构、日志与随机种子 ---
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),   # TensorBoard 日志保存路径
        seed=args.seed,
        verbose=1,                      # 日志详尽程度：0=静默，1=输出训练指标表，2=输出调试信息
    )

    # ==========================================================================
    # 第 3 步：训练前基线评估 (Baseline Evaluation)
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估（未训练模型的随机/初始表现）...")
    # evaluate_policy 会在 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差
    mean_reward_before, std_reward_before = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：配置回调函数 (Callback) 与启动训练
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 DQN（总步数 total_timesteps={args.timesteps}）...")
    # EvalCallback 会每隔 eval_freq 步在 eval_env 上评估一次当前策略，
    # 如果平均奖励刷新了历史最高记录，就自动把最优权重保存为 best_model.zip。
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
        tb_log_name="DQN_CartPole",
        progress_bar=False,
    )

    # ==========================================================================
    # 第 5 步：训练后定量评估 (Post-Training Evaluation)
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载 (Save & Load)
    # ==========================================================================
    final_model_path = model_dir / "dqn_cartpole_final.zip"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    model.save(final_model_path)

    # 删除内存中的旧模型对象，演示从磁盘文件全新加载模型
    del model
    # 如果训练中保存了表现更好的 best_model.zip，优先加载它，否则加载 final 模型
    best_model_path = model_dir / "best_model.zip"
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = DQN.load(load_path, env=eval_env)

    # ==========================================================================
    # 第 7 步：Gymnasium 标准五元组推理闭环演示
    # ==========================================================================
    print("\n[第 7 步] 使用加载的模型运行 1 个完整推理回合（Gymnasium 标准循环）...")
    demo_env = gym.make(env_id)
    # Gymnasium reset() 返回二元组: (初始观测 obs, 辅助信息字典 info)
    obs, info = demo_env.reset(seed=args.seed)

    episode_reward = 0.0
    step_count = 0

    while True:
        # deterministic=True 表示推理阶段关闭 ε-greedy 随机探索，总是选择 Q 值最大的最优动作
        action, _states = loaded_model.predict(obs, deterministic=True)

        # Gymnasium step() 返回五元组:
        # - obs:        下一时刻状态观测
        # - reward:     本步获得的标量奖励
        # - terminated: 任务本质终止（例如在 CartPole 中摆杆倾倒超过阈值角度，或小车驶出边界）
        # - truncated:  时间步截断（例如达到环境设定的最大步数 500 步，并非失败）
        # - info:       调试/诊断附加信息字典
        obs, reward, terminated, truncated, info = demo_env.step(action)

        episode_reward += float(reward)
        step_count += 1

        # 只要 terminated 或 truncated 任一为 True，说明当前回合结束，必须退出循环并 reset()
        if terminated or truncated:
            reason = "达到最大步数截断 (truncated)" if truncated else "摆杆倾倒/出界终止 (terminated)"
            print(f"  推理回合结束 | 总步数: {step_count} | 累计奖励: {episode_reward:.2f} | 结束原因: {reason}")
            break

    # 释放环境资源
    train_env.close()
    eval_env.close()
    demo_env.close()
    print("=" * 70)


if __name__ == "__main__":
    main()
