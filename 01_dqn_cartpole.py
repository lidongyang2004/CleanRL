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
- SB3 `Monitor(gym.make(env_id))`       <--> CleanRL `gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))`
- SB3 `DQN("MlpPolicy", net_arch=[128,128])` <--> CleanRL 显式定义 `QNetwork` 与 `DQNPolicy` (q_net + q_net_target)
- SB3 `ReplayBuffer(buffer_size=50_000)` <--> CleanRL 显式实现 `ReplayBuffer` 环形张量经验池
- SB3 `model.learn()` + `EvalCallback`   <--> CleanRL 显式展开环境交互、ε-greedy 采样、TD Loss 反传与定期最优模型评估保存

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
    """解析命令行参数，方便指定训练总步数与随机种子（与 SB3 脚本参数完全一致）。"""
    parser = argparse.ArgumentParser(description="CleanRL DQN 训练 CartPole-v1 示例")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练环境交互步数（默认 50000 步，CartPole-v1 通常在 2~5 万步即可接近满分）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证实验可复现")
    return parser.parse_args()


def set_seed(seed):
    """固定 Python、NumPy 与 PyTorch 随机种子，保证实验结果可复现。"""
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
        in_dim = obs_dim # 当前层的输入维度4
        # net_arch：[128, 128]）,这里循环 2 次，hidden_dim 两次都为 128。
        for hidden_dim in net_arch: 
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions)) # n_actions（2）
        # 使用 *layers把列表里的 5 个对象展开作为独立参数传给 nn.Sequential，串联并赋值给 self.q_net
        self.q_net = nn.Sequential(*layers)

    def forward(self, obs):
        features = self.features_extractor(obs) # (batch_size, 4) -> (batch_size, 4)
        return self.q_net(features) # (batch_size, 4) -> (batch_size, 2)


class DQNPolicy(nn.Module):
    """
    包含主 Q 网络 (q_net) 与目标 Q 网络 (q_net_target) 的完整策略容器。
    提供与 SB3 完全一致的 `.predict(obs, deterministic=True)` 推理接口。
    4 维状态输入 -> 128(ReLU) -> 128(ReLU) -> 2 个 Q 值输出
    """

    def __init__(self, obs_dim, n_actions, net_arch):
        super().__init__()
        '''
        每次创建时 PyTorch 都会随机生成一套初始权重，
        所以这两个网络刚建好时内部参数是不同的，需要手动同步一次起点。
        '''
        self.q_net = QNetwork(obs_dim, n_actions, net_arch)
        self.q_net_target = QNetwork(obs_dim, n_actions, net_arch)
        # 初始时刻将主网络权重完整复制到目标网络
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def predict(
        self,
        obs,
        deterministic=True,
        epsilon=0.0,
        action_space=None,
    ):
        '''
        根据当前观测选择动作：评估时 deterministic=True 贪婪选最大 Q 值；
        训练时 deterministic=False 按 ε-greedy 探索。
        '''
        if not deterministic and random.random() < epsilon:
            assert action_space is not None
            return int(action_space.sample()), None # 等概率随机抽取一个合法动作。

        device = next(self.parameters()).device
        with torch.no_grad(): # 临时关闭 PyTorch 的自动求导，只是在做推理选动作。
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # 把形状从 (4,) 变成 (1, 4)
            q_values = self.q_net(obs_tensor) 
            action = int(torch.argmax(q_values, dim=1).item())
        return action, None


def linear_schedule(initial_eps, final_eps, duration, current_step):
    """计算 ε-greedy 的线性衰减探索率 ε。"""
    slope = (final_eps - initial_eps) / max(1, duration)
    return max(final_eps, initial_eps + slope * current_step)


def evaluate_policy(
    policy,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    return_episode_rewards=False,
):
    """
    CleanRL 白盒实现的策略评估函数（对应 SB3 `evaluate_policy`）：
    在独立的 eval_env 上运行 n_eval_episodes 个完整回合，返回平均奖励与标准差。
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_eval_episodes):
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
    env_id = "CartPole-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型与日志保存目录（当前脚本所在目录下的 models/ 和 logs/）
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / "dqn_cartpole"
    log_dir = base_dir / "logs" / "dqn_cartpole"
    reset_dir(model_dir)
    reset_dir(log_dir)

    # ==========================================================================
    # 第 1 步：环境初始化与空间剖析
    # ==========================================================================
    print("=" * 70)
    print(f"[第 1 步] 创建 Gymnasium 环境: {env_id}")
    # RecordEpisodeStatistics 包装器（对应 SB3 的 Monitor）用于自动统计每个回合的总奖励和回合长度
    train_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
    # 独立创建验证环境，避免评估过程污染训练环境状态
    eval_env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

    # 查看观测空间与动作空间（新手写强化学习代码前务必先打印这两个属性！）
    print(f"  观测空间 (Observation Space): {train_env.observation_space}")
    print(f"  观测维度 (Shape):             {train_env.observation_space.shape}")
    print(f"  观测下界 (Low):               {train_env.observation_space.low}")
    print(f"  观测上界 (High):              {train_env.observation_space.high}")
    print(f"  动作空间 (Action Space):      {train_env.action_space}  (离散动作数量 n={train_env.action_space.n})")

    # ==========================================================================
    # 第 2 步：模型创建与超参数详解配置（CleanRL 白盒实例化）
    # ==========================================================================
    print("\n[第 2 步] 初始化 CleanRL DQN 模型（QNetwork + TargetNetwork + ReplayBuffer）与核心超参数...")

    # --- 核心超参数配置（与 SB3 01_dqn_cartpole.py 严格 1:1 对齐）---
    learning_rate = 1e-3             # 学习率：DQN 常用 1e-4 ~ 1e-3
    gamma = 0.99                     # 折扣因子 γ ∈ [0, 1)：越接近 1 越重视长远未来奖励

    buffer_size = 50_000             # 经验池最大容量：最多存储 5 万条 (s, a, r, s', done) 转移样本
    learning_starts = 1_000          # 预热步数：先随机探索收集满 1000 条样本后再开始梯度更新
    batch_size = 64                  # 每次更新从经验池中随机采样的批量大小

    train_freq = 4                   # 每与环境交互 4 步，执行一次训练更新
    gradient_steps = 1               # 每次触发训练时，做 1 次梯度下降更新
    target_update_interval = 500     # 目标网络同步间隔：每 500 个环境步将主 Q 网络权重同步到目标 Q 网络
    tau = 1.0                        # 软更新系数：tau=1.0 表示硬更新（直接覆盖复制权重）
    max_grad_norm = 10.0             # SB3 DQN 默认梯度裁剪阈值 10.0

    exploration_initial_eps = 1.0    # 初始探索率 ε=1.0（100% 随机动作）
    exploration_fraction = 0.2       # 在前 20% 的总训练步数内，将 ε 从 initial 线性衰减到 final
    exploration_final_eps = 0.05     # 最终探索率 ε=0.05（保留 5% 随机探索防止陷入局部最优）

    # 构造双层 128 维全连接 Q 网络（对应 SB3 policy_kwargs=dict(net_arch=[128, 128])）
    obs_dim = int(np.prod(train_env.observation_space.shape))
    n_actions = int(train_env.action_space.n)
    net_arch = [128, 128]

    policy = DQNPolicy(obs_dim=obs_dim, n_actions=n_actions, net_arch=net_arch).to(device)
    optimizer = optim.Adam(policy.q_net.parameters(), lr=learning_rate) #训练时更新主 Q 网络（policy.q_net）的参数
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        obs_shape=train_env.observation_space.shape,
        device=device,
    )

    print(f"  运行设备 (Device):            {device}")
    print(f"  主/目标网络结构 (net_arch):   {net_arch} (ReLU 激活, 输出维度={n_actions})")
    print(f"  经验回放池容量 (buffer_size): {buffer_size} (预热步数 learning_starts={learning_starts})")

    # ==========================================================================
    # 第 3 步：训练前基线评估 (Baseline Evaluation)
    # ==========================================================================
    print("\n[第 3 步] 训练前基线评估（未训练模型的随机/初始表现）...")
    mean_reward_before, std_reward_before = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")

    # ==========================================================================
    # 第 4 步：启动 CleanRL 白盒训练主循环与定期评估 (对应 model.learn + EvalCallback)
    # ==========================================================================
    print(f"\n[第 4 步] 开始训练 DQN（总步数 total_timesteps={args.timesteps}）...")
    eval_freq = max(1_000, args.timesteps // 5)
    best_mean_reward = -float("inf")
    best_model_path = model_dir / "best_model.pt"

    tb_run_dir = log_dir / "DQN_CartPole"
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    # 用于保存与 SB3 EvalCallback 完全兼容的 evaluations.npz 记录
    eval_timesteps = []
    eval_results = []
    eval_ep_lengths = []

    obs, _ = train_env.reset(seed=args.seed)
    start_time = time.time()
    exploration_duration = int(exploration_fraction * args.timesteps)
    recent_ep_rewards = []
    recent_ep_lengths = []

    for global_step in range(1, args.timesteps + 1):
        # 1. 计算当前步的 ε-greedy 探索率并选择动作
        epsilon = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            exploration_duration,
            global_step,
        )
        if global_step <= learning_starts:
            action = int(train_env.action_space.sample())
        else:
            action, _ = policy.predict(
                obs,
                deterministic=False,
                epsilon=epsilon,
                action_space=train_env.action_space,
            )

        # 2. 与环境交互一步，获取五元组
        next_obs, reward, terminated, truncated, info = train_env.step(action)

        # 3. 存入经验回放池：注意只有真正失败 terminated=True 才将未来价值置零！
        replay_buffer.add(obs, next_obs, action, float(reward), terminated)
        obs = next_obs

        # 若当前回合结束（terminated 或 truncated），记录统计信息并重置环境
        if terminated or truncated:
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                ep_l = int(info["episode"]["l"])
                recent_ep_rewards.append(ep_r)
                recent_ep_lengths.append(ep_l)
                writer.add_scalar("rollout/ep_rew_mean", np.mean(recent_ep_rewards[-100:]), global_step)
                writer.add_scalar("rollout/ep_len_mean", np.mean(recent_ep_lengths[-100:]), global_step)
                writer.add_scalar("rollout/exploration_rate", epsilon, global_step)
            obs, _ = train_env.reset()

        # 4. 达到预热步数后，每隔 train_freq 步执行一次 TD 梯度更新
        if global_step > learning_starts and global_step % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay_buffer.sample(batch_size)

                # (a) 使用目标 Q 网络计算 TD Target（不计算梯度）：
                #     y = r + γ * (1 - terminated) * max_{a'} Q_target(s', a')
                with torch.no_grad():
                    next_q_values = policy.q_net_target(batch.next_observations)
                    max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
                    td_target = batch.rewards + (1.0 - batch.dones) * gamma * max_next_q

                # (b) 使用主 Q 网络计算当前执行动作 a 对应的预测值 Q(s, a)
                current_q_values = policy.q_net(batch.observations).gather(1, batch.actions)

                # (c) 计算 Smooth L1 (Huber) 损失并反向传播更新主网络（与 SB3 DQN 完全一致）
                loss = F.smooth_l1_loss(current_q_values, td_target)
                optimizer.zero_grad()
                loss.backward() # 梯度只会传回主网络 policy.q_net
                nn.utils.clip_grad_norm_(policy.q_net.parameters(), max_grad_norm)
                optimizer.step()

            # 每隔 100 步向 TensorBoard 记录一次训练状态指标
            if global_step % 100 == 0:
                sps = int(global_step / max(1e-6, time.time() - start_time))
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/q_values", current_q_values.mean().item(), global_step)
                writer.add_scalar("time/fps", sps, global_step)

        # 5. 每隔 target_update_interval 步同步一次目标网络权重
        if global_step % target_update_interval == 0:
            for param, target_param in zip(policy.q_net.parameters(), policy.q_net_target.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # 6. 定期评估当前策略并保存历史最优模型（对应 SB3 EvalCallback）
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
    # 第 5 步：训练后定量评估 (Post-Training Evaluation)
    # ==========================================================================
    print("\n[第 5 步] 训练完成！正在评估最终模型表现...")
    mean_reward_after, std_reward_after = evaluate_policy(
        policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  训练前平均奖励: {mean_reward_before:.2f} +/- {std_reward_before:.2f}")
    print(f"  训练后平均奖励: {mean_reward_after:.2f} +/- {std_reward_after:.2f}")

    # ==========================================================================
    # 第 6 步：模型保存与重新加载 (Save & Load)
    # ==========================================================================
    final_model_path = model_dir / "dqn_cartpole_final.pt"
    print(f"\n[第 6 步] 保存最终模型到: {final_model_path}")
    torch.save(policy.state_dict(), final_model_path)

    # 删除内存中的旧模型对象，演示从磁盘文件全新加载模型
    del policy
    # 如果训练中保存了表现更好的 best_model.pt，优先加载它，否则加载 final 模型
    load_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"  从磁盘重新加载模型: {load_path}")
    loaded_model = DQNPolicy(obs_dim=obs_dim, n_actions=n_actions, net_arch=net_arch).to(device)
    loaded_model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    loaded_model.eval()

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
