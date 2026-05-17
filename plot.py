import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHODS = {
    "reinforce": {
        "label": "REINFORCE (Asg 3)",
        "color": "#2ca02c",   # green
        "dir":   "results/reinforce",
        "format": "csv",
    },
    "dqn": {
        "label": "DQN (Asg 2)",
        "color": "#1f77b4",   # blue
        "dir":   "results/dqn",
        "format": "csv",
    },
    "a2c": {
        "label": "A2C (Asg 3)",
        "color": "#9467bd",   # purple
        "dir":   "results/a2c",
        "format": "csv",
    },
    "sac": {
        "label": r"SAC auto-$\alpha$ (ratio 0.70)",
        "color": "#ff0000",   # red
        "dir":   "results/sac_ratio_070",
        "format": "npz",
    },
    "sac98": {
        "label": r"SAC auto-$\alpha$ (ratio 0.98)",
        "color": "#000000",  # black
        "dir": "results/sac_ratio_098",
        "format": "npz",
    },
}


def bin_episode_returns(episode_steps, episode_returns, bin_edges):
    idx = np.digitize(episode_steps, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            binned[b] = episode_returns[mask].mean()
    return binned


def forward_fill_nan(arr):
    out = arr.copy()
    last_valid = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last_valid
        else:
            last_valid = out[i]
    return out


def load_csv_method(directory, bin_edges):
    paths = sorted(glob.glob(os.path.join(directory, "run_seed_*.csv")))
    if not paths:
        return None
    binned_per_seed = []
    for path in paths:
        df = pd.read_csv(path)
        binned = bin_episode_returns(
            df["env_step"].to_numpy(),
            df["Episode_Return"].to_numpy(),
            bin_edges,
        )
        binned = forward_fill_nan(binned)
        binned_per_seed.append(binned)
    return np.array(binned_per_seed)


def load_npz_method(directory, bin_edges):
    paths = sorted(glob.glob(os.path.join(directory, "sac_run_seed*.npz")))
    if not paths:
        return None
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    interp_per_seed = []
    for path in paths:
        data = np.load(path)
        eval_returns = data["eval_returns"]  # shape (n_evals, 3): step, mean, std
        if eval_returns.size == 0:
            continue
        steps = eval_returns[:, 0]
        means = eval_returns[:, 1]
        interp = np.interp(bin_centers, steps, means,
                           left=means[0], right=means[-1])
        interp[bin_centers < steps[0]] = np.nan
        interp_per_seed.append(interp)
    return np.array(interp_per_seed)


def plot_method(ax, bin_centers, data_per_seed, label, color):
    if data_per_seed is None or data_per_seed.shape[0] == 0:
        print(f"  No data for {label}, skipping.")
        return
    mean = np.nanmean(data_per_seed, axis=0)
    std = np.nanstd(data_per_seed, axis=0)
    valid = ~np.isnan(mean)
    ax.plot(bin_centers[valid], mean[valid],
            label=f"{label} (n={data_per_seed.shape[0]})",
            color=color, linewidth=2)
    ax.fill_between(bin_centers[valid],
                    (mean - std)[valid], (mean + std)[valid],
                    color=color, alpha=0.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=400000,
                   help="Should match what you trained with.")
    p.add_argument("--num-bins", type=int, default=80,
                   help="Number of bins along the env-step axis.")
    p.add_argument("--out", type=str, default="cartpole_comparison.png")
    args = p.parse_args()

    bin_edges = np.linspace(0, args.total_steps, args.num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method_name, cfg in METHODS.items():
        if cfg["format"] == "csv":
            data = load_csv_method(cfg["dir"], bin_edges)
        else:
            data = load_npz_method(cfg["dir"], bin_edges)
        if data is None:
            print(f"No data found for {method_name} in {cfg['dir']}")
            continue
        print(f"Loaded {method_name}: {data.shape[0]} seeds")
        plot_method(ax, bin_centers, data, cfg["label"], cfg["color"])

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (binned mean)")
    ax.set_title("CartPole-v1: DQN vs A2C vs SAC across 5 seeds")
    ax.axhline(500, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Solved (500)")
    ax.set_ylim(0, 540)
    ax.set_xlim(0, args.total_steps)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")

    # Print numerical summary for the report
    print("\n=== Final performance (last 5% of training, mean across seeds) ===")
    n_tail = max(1, args.num_bins // 20)
    for method_name, cfg in METHODS.items():
        if cfg["format"] == "csv":
            data = load_csv_method(cfg["dir"], bin_edges)
        else:
            data = load_npz_method(cfg["dir"], bin_edges)
        if data is None:
            continue
        tail = data[:, -n_tail:]
        per_seed_tail_mean = np.nanmean(tail, axis=1)
        print(f"  {cfg['label']}: "
              f"{np.nanmean(per_seed_tail_mean):.1f} ± "
              f"{np.nanstd(per_seed_tail_mean):.1f}")


if __name__ == "__main__":
    main()