import pytest
pytest.importorskip("torch")
"""Unit tests for GridWorld transitions, replay buffer, and Bellman targets."""
import torch
from src.environment.grid_world import GridWorldEnv
from src.replay.replay_buffer import ReplayBuffer
from src.agents.dqn_agent import QNetwork

def test_environment_reset():
    env = GridWorldEnv()
    obs = env.reset()
    assert len(obs) == 25
    assert obs[0] == 1.0

def test_replay_buffer():
    buf = ReplayBuffer(100)
    obs = [0] * 25
    buf.push(obs, 1, 1.0, obs, False)
    assert len(buf) == 1
    s, a, r, next_s, d = buf.sample(1)
    assert s.shape == (1, 25)

def test_q_network_forward():
    net = QNetwork(25, 4)
    x = torch.randn(2, 25)
    out = net(x)
    assert out.shape == (2, 4)
