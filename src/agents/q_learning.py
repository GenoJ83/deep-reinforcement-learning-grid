"""Tabular Q-Learning with epsilon-greedy decay."""
import numpy as np

class TabularQLearning:
    def __init__(self, n_states: int = 25, n_actions: int = 4, lr: float = 0.1, gamma: float = 0.95):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma

    def select_action(self, state_idx: int, epsilon: float = 0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, s: int, a: int, r: float, next_s: int, done: bool):
        best_next_a = np.max(self.q_table[next_s]) if not done else 0.0
        target = r + self.gamma * best_next_a
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])
