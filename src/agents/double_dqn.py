"""Double DQN agent decoupling action selection from target evaluation."""
import torch
import torch.nn as nn
from src.agents.dqn_agent import QNetwork
from src.replay.replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    def __init__(self, state_dim: int = 25, action_dim: int = 4, lr: float = 1e-3, gamma: float = 0.99):
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def update(self, buffer: ReplayBuffer, batch_size: int = 32):
        if len(buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
