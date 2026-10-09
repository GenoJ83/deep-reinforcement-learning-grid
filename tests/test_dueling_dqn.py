"""Tests for Dueling DQN."""
import torch
from src.agents.dueling_dqn import DuelingQNetwork

def test_dueling_dqn_forward():
    net = DuelingQNetwork(state_dim=25, action_dim=4)
    x = torch.randn(2, 25)
    out = net(x)
    assert out.shape == (2, 4)
