"""Deep Q-Network (DQN) policy neural network."""
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 25, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
