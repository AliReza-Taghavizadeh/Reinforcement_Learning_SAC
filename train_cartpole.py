import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# q network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# replay buffer stores transitions
class ReplayBuffer:
    def __init__(self, size):
        self.data = deque(maxlen=size)

    def add(self, s, a, r, sp, done):
        self.data.append((s, a, r, sp, done))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for item in batch:
            s, a, r, sp, done = item
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(sp)
            dones.append(done)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.data)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# decay epsilon over time
def get_epsilon(step, eps_start, eps_end, decay_steps):
    if decay_steps <= 0:
        return eps_end

    if step >= decay_steps:
        return eps_end

    frac = step / decay_steps
    eps = eps_start + frac * (eps_end - eps_start)
    return eps


def get_smooth(values, window):
    out = []

    if len(values) == 0:
        return out

    for i in range(len(values)):
        left = i - window + 1
        if left < 0:
            left = 0

        chunk = values[left:i + 1]
        avg = float(np.mean(chunk))
        out.append(avg)

    return out


# pick action, random if below epsilon
def epsilon_greedy(q_net, s, eps, action_size):
    rand_val = random.random()

    if rand_val < eps:
        return random.randint(0, action_size - 1)

    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_vals = q_net(s_tensor)

    best_a = int(torch.argmax(q_vals, dim=1).item())
    return best_a


# do one update step
def update_network(batch, q_net, target_net, optimizer, gamma):
    states, actions, rewards, next_states, dones = batch

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # get q values for what we did
    all_q = q_net(states)
    output = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

    # bellman target
    with torch.no_grad():
        if target_net is None:
            next_q = q_net(next_states)
        else:
            next_q = target_net(next_states)

        max_next_q = next_q.max(dim=1).values
        target = rewards + gamma * (1.0 - dones) * max_next_q

    loss = F.mse_loss(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


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
        smooth_returns = data["Episode_Return_smooth"].astype(np.float32)

        all_runs.append((steps, smooth_returns))

        if int(steps[-1]) > biggest_step:
            biggest_step = int(steps[-1])

    if len(all_runs) == 0:
        return

    grid = np.arange(step_size, biggest_step + step_size, step_size, dtype=np.float32)
    lines = []

    for steps, returns in all_runs:
        unique_steps, unique_index = np.unique(steps, return_index=True)
        unique_returns = returns[unique_index]

        new_line = np.interp(
            grid,
            unique_steps,
            unique_returns,
            left=unique_returns[0],
            right=unique_returns[-1],
        )
        lines.append(new_line)

    lines = np.array(lines, dtype=np.float32)
    mean_line = np.mean(lines, axis=0)
    std_line = np.std(lines, axis=0)

    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_step", "mean_return", "std_return"])

        for i in range(len(grid)):
            writer.writerow([int(grid[i]), float(mean_line[i]), float(std_line[i])])


def run_one_seed(args, seed, out_dir):
    set_seed(seed)

    env = gym.make("CartPole-v1")
    s, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = QNetwork(state_size, action_size, args.hidden_size)

    target_net = None
    if args.use_target_network:
        target_net = QNetwork(state_size, action_size, args.hidden_size)
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)

    memory = None
    if args.use_replay:
        memory = ReplayBuffer(args.replay_size)

    total_steps = 0
    update_count = 0
    ep_return = 0.0
    returns = []
    rows = []

    # need these for the no-replay case
    last_s = None
    last_a = None
    last_r = None
    last_sp = None
    last_done = None

    # progress reporting
    report_interval = 100000

    while total_steps < args.total_steps:
        eps = get_epsilon(
            total_steps,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )

        # pick action
        a = epsilon_greedy(q_net, s, eps, action_size)
        sp, r, terminated, truncated, _ = env.step(a)

        done = terminated or truncated

        last_s = s
        last_a = a
        last_r = r
        last_sp = sp
        last_done = float(done)

        if memory is not None:
            memory.add(s, a, r, sp, float(done))

        ep_return += r
        total_steps += 1

        # progress report
        if total_steps % report_interval == 0:
            avg_return = np.mean(returns[-100:]) if len(returns) >= 100 else (np.mean(returns) if returns else 0)
            print(f"    step {total_steps:,} / {args.total_steps:,} | eps={eps:.3f} | avg_return={avg_return:.1f}")

        # only train after learning_starts steps
        if total_steps >= args.learning_starts and total_steps % args.train_frequency == 0:
            for _ in range(args.gradient_steps):
                if memory is None:
                    # just use last transition if no buffer
                    s_b = np.array([last_s], dtype=np.float32)
                    a_b = np.array([last_a], dtype=np.int64)
                    r_b = np.array([last_r], dtype=np.float32)
                    sp_b = np.array([last_sp], dtype=np.float32)
                    d_b = np.array([last_done], dtype=np.float32)
                    batch = (s_b, a_b, r_b, sp_b, d_b)
                else:
                    if len(memory) < args.batch_size:
                        break
                    batch = memory.sample(args.batch_size)

                update_network(batch, q_net, target_net, optimizer, args.gamma)
                update_count += 1

                # update target net periodically
                if target_net is not None:
                    if update_count % args.target_update_frequency == 0:
                        target_net.load_state_dict(q_net.state_dict())

        s = sp

        if done:
            returns.append(float(ep_return))
            smooth_returns = get_smooth(returns, args.smoothing_window)
            rows.append((float(ep_return), float(smooth_returns[-1]), int(total_steps)))

            ep_return = 0.0
            s, _ = env.reset()

    env.close()

    csv_file = out_dir / f"run_seed_{seed}.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode_Return", "Episode_Return_smooth", "env_step"])
        writer.writerows(rows)

    return csv_file


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=str, default="results/default")
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=10000)
    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--smoothing-window", type=int, default=20)
    parser.add_argument("--aggregate-step-size", type=int, default=2500)
    parser.add_argument("--use-replay", action="store_true")
    parser.add_argument("--use-target-network", action="store_true")
    parser.add_argument("--run-all", action="store_true", help="Run all 4 configurations for assignment")
    args = parser.parse_args()
    return args


def run_single_config(args, out_dir):
    """Run training for a single configuration."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(out_dir)
    config_file = out_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    csv_list = []
    for seed in args.seeds:
        print(f"  seed {seed}")
        csv_file = run_one_seed(args, seed, out_dir)
        csv_list.append(csv_file)

    save_aggregate(csv_list, out_dir / "aggregate.csv", args.aggregate_step_size)


def main():
    args = get_args()

    if args.run_all:
        # run all 4 configurations required by assignment
        configs = [
            ("naive", False, False),
            ("only_tn", True, False),
            ("only_er", False, True),
            ("tn_er", True, True),
        ]

        for name, use_tn, use_er in configs:
            print(f"\n=== Running {name} ===")
            args.use_target_network = use_tn
            args.use_replay = use_er
            run_single_config(args, Path("results") / name)

        print("\n=== All configurations complete ===")
        print("Run 'python plot_results.py' to generate the comparison plot")
    else:
        # run single configuration
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = vars(args)
        config_file = out_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        csv_list = []
        for seed in args.seeds:
            print("running seed", seed)
            csv_file = run_one_seed(args, seed, out_dir)
            csv_list.append(csv_file)

        save_aggregate(csv_list, out_dir / "aggregate.csv", args.aggregate_step_size)
        print("done")
        print("saved results in", out_dir)


if __name__ == "__main__":
    main()
