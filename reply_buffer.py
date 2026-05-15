"""
Replay buffer for SAC

Stores transitions (s, a, r, s', done) and lets us sample random
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.buffer = deque(maxlen=capacity) # Maxlen for overflow problems
        self.state_dim = state_dim

    def push(self, state, action, reward, next_state, done):
        """Add one transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Pull a random minibatch and return it as numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)