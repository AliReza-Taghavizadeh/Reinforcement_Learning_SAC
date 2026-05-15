"""
Run the SAC ablation study: 3 entropy-coefficient conditions over 5 seeds.
"""

import argparse
import os
import subprocess
import sys
import time


CONDITIONS = {
    # "auto_alpha_ratio_098": [
    #     "--auto-alpha",
    #     "--target-entropy-ratio", "0.98",
    # ],
    # "auto_alpha_ratio_070": [
    #     "--auto-alpha",
    #     "--target-entropy-ratio", "0.7",
    # ],
    "auto_alpha_ratio_080": [
        "--auto-alpha",
        "--target-entropy-ratio", "0.8",
    ],
    # "auto_alpha_ratio_050": [
    #     "--auto-alpha",
    #     "--target-entropy-ratio", "0.50",
    # ],
    # "fixed_alpha_005": [
    #     "--no-auto-alpha",
    #     "--alpha", "0.05",
    # ],
}

SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 50000


def run_one(condition_name, extra_args, seed):
    save_dir = os.path.join("results", condition_name)

    cmd = [
        sys.executable, "train.py",
        "--seed", str(seed),
        "--total-steps", str(TOTAL_STEPS),
        "--save-dir", save_dir,
    ] + extra_args

    print(f"\n{'='*70}")
    print(f"Running condition={condition_name}  seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"!! Run FAILED: {condition_name} seed={seed} (rc={result.returncode})")
        return False
    print(f"Done in {elapsed:.1f}s")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   help="Run only one condition by name (skip the others).")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="Which seeds to run.")
    args = p.parse_args()

    conditions = CONDITIONS
    if args.only is not None:
        if args.only not in CONDITIONS:
            print(f"Unknown condition: {args.only}")
            print(f"Available: {list(CONDITIONS.keys())}")
            sys.exit(1)
        conditions = {args.only: CONDITIONS[args.only]}

    total_runs = len(conditions) * len(args.seeds)
    print(f"Planning {total_runs} runs total "
          f"({len(conditions)} conditions x {len(args.seeds)} seeds)")
    print(f"Estimated time at ~330s/run: {total_runs * 330 / 60:.0f} minutes")

    overall_start = time.time()
    failures = []
    for condition_name, extra_args in conditions.items():
        for seed in args.seeds:
            ok = run_one(condition_name, extra_args, seed)
            if not ok:
                failures.append((condition_name, seed))

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"All runs done in {elapsed/60:.1f} minutes")
    if failures:
        print(f"Failures: {failures}")
    else:
        print("All runs succeeded.")


if __name__ == "__main__":
    main()