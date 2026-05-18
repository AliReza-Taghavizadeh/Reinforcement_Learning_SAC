"""
Compute the numbers for the results table in the report.

For each method:
  - Final return: mean over last 10% of training, averaged across seeds (+/- std).
  - Steps-to-475: first env-step where the seed's smoothed return crosses 475.
    Reported as the mean of that crossing step over seeds that reached it,
    plus how many seeds out of 5 reached it.

Run:  python compute_table.py
"""

import glob
import os
import numpy as np
import pandas as pd


# Same config as plot_comparison.py, but only the data dirs and formats matter.
METHODS = [
    ("SAC (rho=0.70)", "results/sac_ratio_070", "npz"),
    ("A2C",            "results/a2c",           "csv"),
    ("REINFORCE",      "results/reinforce",     "csv"),
    ("DQN",            "results/dqn",           "csv"),
]

TOTAL_STEPS = 400_000
THRESHOLD = 475


def load_seed_curves_csv(directory):
    """Return list of (steps, returns) per seed. Uses smoothed returns."""
    paths = sorted(glob.glob(os.path.join(directory, "run_seed_*.csv")))
    curves = []
    for path in paths:
        df = pd.read_csv(path)
        steps = df["env_step"].to_numpy()
        # Use the smoothed column for stability; it is what the report curves show.
        returns = df["Episode_Return_smooth"].to_numpy()
        curves.append((steps, returns))
    return curves


def load_seed_curves_npz(directory):
    """SAC: eval_returns has columns (step, mean, std)."""
    paths = sorted(glob.glob(os.path.join(directory, "sac_run_seed*.npz")))
    curves = []
    for path in paths:
        data = np.load(path)
        eval_returns = data["eval_returns"]  # (n_evals, 3)
        if eval_returns.size == 0:
            continue
        steps = eval_returns[:, 0]
        returns = eval_returns[:, 1]
        curves.append((steps, returns))
    return curves


def final_return(curves, tail_fraction=0.1, total_steps=TOTAL_STEPS):
    """Mean over last (tail_fraction) of training, averaged across seeds."""
    cutoff = total_steps * (1.0 - tail_fraction)
    per_seed = []
    for steps, returns in curves:
        mask = steps >= cutoff
        if mask.any():
            per_seed.append(returns[mask].mean())
    per_seed = np.array(per_seed)
    return per_seed.mean(), per_seed.std()


def steps_to_threshold(curves, threshold=THRESHOLD):
    """First env-step at which return crosses the threshold, per seed.

    Returns (mean_step_in_thousands, num_seeds_that_reached, total_seeds).
    For seeds that never crossed, they are not counted in the mean.
    """
    crossing_steps = []
    for steps, returns in curves:
        above = np.where(returns >= threshold)[0]
        if len(above) > 0:
            crossing_steps.append(steps[above[0]])
    n_total = len(curves)
    n_reached = len(crossing_steps)
    if n_reached == 0:
        return None, 0, n_total
    return np.mean(crossing_steps) / 1000.0, n_reached, n_total


def main():
    print(f"{'Method':<20} {'Final return':<20} {'Steps-to-475':<25}")
    print("-" * 65)
    for name, directory, fmt in METHODS:
        if fmt == "csv":
            curves = load_seed_curves_csv(directory)
        else:
            curves = load_seed_curves_npz(directory)
        if not curves:
            print(f"{name:<20} no data found in {directory}")
            continue

        mean_r, std_r = final_return(curves)
        crossing_k, n_reached, n_total = steps_to_threshold(curves)

        final_str = f"{mean_r:.1f} +/- {std_r:.1f}"
        if crossing_k is None:
            crossing_str = f"never ({n_reached}/{n_total})"
        else:
            crossing_str = f"{crossing_k:.0f}k ({n_reached}/{n_total})"

        print(f"{name:<20} {final_str:<20} {crossing_str:<25}")

    print()
    print("LaTeX-ready rows:")
    print()
    for name, directory, fmt in METHODS:
        if fmt == "csv":
            curves = load_seed_curves_csv(directory)
        else:
            curves = load_seed_curves_npz(directory)
        if not curves:
            continue
        mean_r, std_r = final_return(curves)
        crossing_k, n_reached, n_total = steps_to_threshold(curves)
        if crossing_k is None:
            ck = "--"
        else:
            ck = f"{crossing_k:.0f}"
        print(f"{name:<20} & ${mean_r:.0f} \\pm {std_r:.0f}$ & {ck}\\,k\\ ({n_reached}/{n_total}) \\\\")


if __name__ == "__main__":
    main()