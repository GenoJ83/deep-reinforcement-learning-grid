"""Episode reward logging and moving average tracking."""
import collections

class RewardLogger:
    def __init__(self, window_size: int = 20):
        self.history = []
        self.window = collections.deque(maxlen=window_size)

    def record_episode(self, ep_reward: float):
        self.history.append(ep_reward)
        self.window.append(ep_reward)

    def get_moving_average(self):
        return sum(self.window) / len(self.window) if self.window else 0.0
