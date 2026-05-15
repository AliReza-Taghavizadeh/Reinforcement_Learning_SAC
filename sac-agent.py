"""
SAC agent for discrete actions.
"""

import torch
import numpy as np

from networks import PolicyNetwork, QNetwork


class SACAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        target_entropy_ratio=0.98,
        device="cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions
        self.device = device

        # Policy network
        self.policy = PolicyNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Two Q-networks (clipped double Q trick)
        self.q1 = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Target Q-networks
        self.q1_target = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Freeze target networks
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Entropy coefficient (alpha).
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy: fraction of the max possible entropy log(2) ~= 0.693
            self.target_entropy = -target_entropy_ratio * np.log(1.0 / num_actions)
            # Learn log(alpha) so alpha stays positive automatically
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.log(torch.tensor(alpha, device=device))
            self.target_entropy = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        """Pick an action for the current state"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            # Pick the most likely action (greedy)
            with torch.no_grad():
                probs, _ = self.policy.get_action_probs(state_t)
                return probs.argmax(dim=-1).item()
        else:
            return self.policy.sample_action(state_t)