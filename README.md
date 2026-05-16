# SAC on CartPole-v1

Implementation of discrete Soft Actor-Critic with comparison against DQN
(Assignment 2) and A2C (Assignment 3) baselines.

## Files

**SAC implementation**
- `networks.py` - policy and Q-network definitions
- `replay_buffer.py` - experience replay
- `sac_agent.py` - SAC losses, updates, target-network handling
- `train.py` - training loop for a single SAC run

**Experiment runners**
- `run_experiments.py` - entropy-coefficient ablation (5 SAC conditions x 5 seeds)

## Setup

```
pip install -r requirements.txt
```

### Single SAC run
```
python train.py --seed 0 --total-steps 50000 --target-entropy-ratio 0.70
```

### Entropy-coefficient ablation
```
python run_experiments.py
```
Runs 5 conditions (auto-alpha at ratios 0.98, 0.80, 0.70, 0.50, and
fixed alpha=0.05), 5 seeds each, 50000 steps per run.