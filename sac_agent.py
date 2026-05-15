"""
SAC agent for discrete actions.
"""

import torch
import numpy as np

from networks import PolicyNetwork, QNetwork
import torch.nn.functional as F

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
        target_entropy_ratio=0.5,
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


    def update_q(self, states, actions, rewards, next_states, dones):
        """One gradient step on both Q-networks."""

        with torch.no_grad():
            next_probs, next_log_probs = self.policy.get_action_probs(next_states)

            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q_min = torch.min(next_q1, next_q2)  # Clipped double Q

            next_v = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=-1)

            # Bellman target
            target_q = rewards + self.gamma * (1.0 - dones) * next_v

        # Compute the current Q-values for the (s, a) pairs in the batch
        q1_pred = self.q1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.q2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # MSE loss against the target.
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        # PyTorch update pattern
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        return q1_loss.item(), q2_loss.item()

    def update_policy(self, states):
        """One gradient step on the policy."""

        probs, log_probs = self.policy.get_action_probs(states) #(batch, num_actions)

        # Get Q-values for all actions (no gradient into Q)
        with torch.no_grad():
            q1 = self.q1(states)
            q2 = self.q2(states)
            q_min = torch.min(q1, q2)

        # Policy loss
        inside = self.alpha * log_probs - q_min
        policy_loss = (probs * inside).sum(dim=-1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return policy_loss.item(), probs.detach(), log_probs.detach()

    def update_alpha(self, probs, log_probs):
        """One gradient step on log_alpha."""
        if not self.auto_alpha:
            return 0.0

        entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return alpha_loss.item()

    def soft_update_targets(self):
        """Soft update target Q-networks"""
        with torch.no_grad():
            for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)
            for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)

    def update(self, batch):
        """One full SAC update step on a minibatch"""
        states, actions, rewards, next_states, dones = batch

        # Move data to torch tensors on the right device
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        q1_loss, q2_loss = self.update_q(states, actions, rewards, next_states, dones)
        policy_loss, probs, log_probs = self.update_policy(states)
        alpha_loss = self.update_alpha(probs, log_probs)
        self.soft_update_targets()

        return {
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item(),
        }