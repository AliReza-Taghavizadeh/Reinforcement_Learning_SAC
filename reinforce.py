import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_returns(rewards, gamma):
    G = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.insert(0, running)
    return torch.tensor(G, dtype=torch.float32)


def get_smooth(values, window):
    if len(values) == 0:
        return []
    return [float(np.mean(values[max(0, i - window + 1):i + 1])) for i in range(len(values))]


def save_aggregate(csv_list, out_file, step_size):
    all_runs = []
    biggest_step = 0
    for csv_file in csv_list:
        data = np.genfromtxt(csv_file, delimiter=",", names=True)
        if data.size == 0:
            continue
        if data.shape == ():
            data = np.array([data], dtype=data.dtype)
        steps = data["env_step"].astype(np.float32)
        smooth = data["Episode_Return_smooth"].astype(np.float32)
        all_runs.append((steps, smooth))
        biggest_step = max(biggest_step, int(steps[-1]))
    if not all_runs:
        return
    grid = np.arange(step_size, biggest_step + step_size, step_size, dtype=np.float32)
    lines = []
    for steps, returns in all_runs:
        u_steps, u_idx = np.unique(steps, return_index=True)
        interp = np.interp(grid, u_steps, returns[u_idx],
                           left=returns[u_idx][0], right=returns[u_idx][-1])
        lines.append(interp)
    lines = np.array(lines, dtype=np.float32)
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_step", "mean_return", "std_return"])
        for i in range(len(grid)):
            writer.writerow([int(grid[i]), float(np.mean(lines[:, i])), float(np.std(lines[:, i]))])


def run_one_seed(args, seed, out_dir):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(state_size, action_size, args.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    total_steps = 0
    returns = []
    rows = []
    report_interval = 100_000

    while total_steps < args.total_steps:
        # ---- Collect one complete episode ----
        ep_states, ep_actions, ep_rewards = [], [], []
        s = state
        ep_return = 0.0

        while True:
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = policy(s_tensor)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            ns, r, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            ep_states.append(s)
            ep_actions.append(action.item())
            ep_rewards.append(r)
            ep_return += r
            total_steps += 1
            s = ns

            if done:
                state, _ = env.reset()
                break
            if total_steps >= args.total_steps:
                state = s
                break

        if total_steps % report_interval < len(ep_rewards):
            avg = np.mean(returns[-100:]) if returns else 0.0
            print(f"  step {total_steps:,} / {args.total_steps:,} | avg_return={avg:.1f}")

        if len(ep_rewards) == 0:
            continue

        # ---- Compute reward-to-go ----
        G = compute_returns(ep_rewards, args.gamma)

        # ---- Normalise returns (reduces gradient variance) ----
        if G.std() > 1e-8:
            G = (G - G.mean()) / (G.std() + 1e-8)

        # ---- Recompute log-probs for the collected episode ----
        states_tensor = torch.tensor(np.array(ep_states), dtype=torch.float32)
        actions_tensor = torch.tensor(ep_actions, dtype=torch.long)
        probs_all = policy(states_tensor)
        log_probs = torch.log(probs_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) + 1e-8)

        # ---- Policy gradient loss (negated for gradient ascent) ----
        loss = -(log_probs * G).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        returns.append(ep_return)
        smooth = get_smooth(returns, args.smoothing_window)
        rows.append((ep_return, smooth[-1], total_steps))

    env.close()

    csv_file = out_dir / f"run_seed_{seed}.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode_Return", "Episode_Return_smooth", "env_step"])
        writer.writerows(rows)
    return csv_file


def get_args():
    parser = argparse.ArgumentParser(description="REINFORCE on CartPole-v1")
    parser.add_argument("--output-dir",  type=str,   default="results/reinforce")
    parser.add_argument("--total-steps", type=int,   default=1_000_000)
    parser.add_argument("--seeds",       type=int,   nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--smoothing-window",    type=int, default=20)
    parser.add_argument("--aggregate-step-size", type=int, default=2500)
    return parser.parse_args()


def main():
    args = get_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_list = []
    for seed in args.seeds:
        print(f"[REINFORCE] seed {seed}")
        csv_list.append(run_one_seed(args, seed, out_dir))
    save_aggregate(csv_list, out_dir / "aggregate.csv", args.aggregate_step_size)
    print(f"Done. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
