"""
Networks for discrete SAC.

PolicyNetwork
Q-Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits  # raw scores, not probabilities yet

    def get_action_probs(self, state):
        """Return action probabilities and log probabilities."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample_action(self, state):
        """Sample one action from the policy. Used when interacting with env."""
        probs, _ = self.get_action_probs(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()  # convert tensor to plain Python int

class QNetwork(nn.Module):
        def __init__(self, state_dim, num_actions, hidden_dim=64):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_actions)

        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            q_values = self.fc3(x)
            return q_values  # (batch, num_actions)