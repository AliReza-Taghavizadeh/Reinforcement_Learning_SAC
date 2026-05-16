import argparse
import os
import subprocess
import sys
import time


SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 400000


def run_subprocess(cmd, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print('='*70)
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"!! FAILED ({label}) rc={result.returncode}")
        return False
    print(f"  Done in {elapsed/60:.1f} min")
    return True


def run_dqn(total_steps, seeds):
    cmd = [
        sys.executable, "train_cartpole.py",
        "--output-dir", "results/dqn",
        "--total-steps", str(total_steps),
        "--seeds", *[str(s) for s in seeds],
        "--use-replay",
        "--use-target-network",
    ]
    return run_subprocess(cmd, f"DQN ({total_steps} steps x {len(seeds)} seeds)")


def run_a2c(total_steps, seeds):
    """A2C: also handles all seeds internally."""
    cmd = [
        sys.executable, "a2c.py",
        "--output-dir", "results/a2c",
        "--total-steps", str(total_steps),
        "--seeds", *[str(s) for s in seeds],
    ]
    return run_subprocess(cmd, f"A2C ({total_steps} steps x {len(seeds)} seeds)")

def run_reinforce(total_steps, seeds):
    """REINFORCE: handles all seeds internally."""
    cmd = [
        sys.executable, "reinforce.py",
        "--output-dir", "results/reinforce",
        "--total-steps", str(total_steps),
        "--seeds", *[str(s) for s in seeds],
    ]
    return run_subprocess(cmd, f"REINFORCE ({total_steps} steps x {len(seeds)} seeds)")

def run_sac(total_steps, seeds):
    all_ok = True
    for seed in seeds:
        cmd = [
            sys.executable, "train.py",
            "--seed", str(seed),
            "--total-steps", str(total_steps),
            "--save-dir", "results/sac_ratio_070",
            "--auto-alpha",
            "--target-entropy-ratio", "0.70",
        ]
        ok = run_subprocess(cmd, f"SAC ratio 0.70, seed {seed}")
        if not ok:
            all_ok = False
    return all_ok


METHODS = {
    "reinforce": run_reinforce,
    "dqn": run_dqn,
    "a2c": run_a2c,
    "sac": run_sac,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--only", type=str, default=None,
                   choices=list(METHODS.keys()),
                   help="Run only one method by name.")
    args = p.parse_args()

    if args.only is not None:
        methods_to_run = {args.only: METHODS[args.only]}
    else:
        methods_to_run = METHODS

    print(f"Running {list(methods_to_run.keys())}")
    print(f"Total steps per run: {args.total_steps}")
    print(f"Seeds: {args.seeds}")

    overall_start = time.time()
    failures = []
    for name, fn in methods_to_run.items():
        ok = fn(args.total_steps, args.seeds)
        if not ok:
            failures.append(name)

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"All done in {elapsed/60:.1f} minutes")
    if failures:
        print(f"Failed methods: {failures}")
    else:
        print("All methods completed successfully.")


if __name__ == "__main__":
    main()