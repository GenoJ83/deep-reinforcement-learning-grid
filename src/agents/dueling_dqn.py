"""Dueling Deep Q-Network (Dueling DQN) Architecture."""
import torch
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int = 25, action_dim: int = 4):
        super().__init__()
        self.feature_stream = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        feat = self.feature_stream(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_vals = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals
