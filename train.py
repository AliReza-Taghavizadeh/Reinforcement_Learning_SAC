"""
Training loop for SAC on CartPole-v1.

Run with custom seed:  python train.py --seed 8000
"""

import argparse
import time
import numpy as np
import torch
import gymnasium as gym
import os

from replay_buffer import ReplayBuffer
from sac_agent import SACAgent


def evaluate(agent, env_name, num_episodes=5, seed=1):
    """Run a few episodes greedily to measure current policy quality."""
    env = gym.make(env_name)
    returns = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    device = "cpu" # no point in using cuda/gpu

    agent = SACAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        device=device,
    )

    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=state_dim)

    # Logging
    episode_returns = []
    eval_returns = []
    losses_log = []

    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_steps = 0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        if step < args.start_steps:
            # Warm-up
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_return += reward
        episode_steps += 1

        # --- Episode boundary ---
        if done:
            episode_returns.append(episode_return)
            if len(episode_returns) % args.log_every == 0:
                recent = np.mean(episode_returns[-args.log_every:])
                elapsed = time.time() - start_time
                print(f"step {step:6d} | episodes {len(episode_returns):4d} "
                      f"| recent return {recent:6.1f} | alpha {agent.alpha.item():.3f} "
                      f"| time {elapsed:.1f}s")
            state, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0

        # Learning
        if step >= args.start_steps and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            losses = agent.update(batch)
            if step % args.log_losses_every == 0:
                losses_log.append((step, losses))

        # Periodic evaluation
        if step % args.eval_every == 0:
            mean_ret, std_ret = evaluate(agent, args.env, num_episodes=5, seed=args.seed + 1000)
            eval_returns.append((step, mean_ret, std_ret))
            print(f"  [eval @ step {step}] mean return {mean_ret:.1f} ± {std_ret:.1f}")

    env.close()

    # Save logs
    save_dir = args.save_dir if args.save_dir else "./logs/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sac_run_seed{args.seed}.npz")
    np.savez(
        save_path,
        episode_returns=np.array(episode_returns),
        eval_returns=np.array(eval_returns),
    )
    print(f"Saved logs to {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=50000)
    p.add_argument("--start-steps", type=int, default=1000)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument("--no-auto-alpha", dest="auto_alpha", action="store_false")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-losses-every", type=int, default=500)
    p.add_argument("--cuda", action="store_true", default=False)
    p.add_argument("--save-dir", type=str, default=None)
    args = p.parse_args()
    train(args)