"""Custom 2D GridWorld navigation environment with Gymnasium API."""
import numpy as np

class GridWorldEnv:
    # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 5, goal=(4, 4), obstacles=None):
        self.size = size
        self.goal = goal
        self.obstacles = obstacles or [(2, 2), (2, 3), (3, 2)]
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self._get_obs()

    def _get_obs(self):
        # Flatten state into one-hot vector representation
        obs = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.state[0] * self.size + self.state[1]
        obs[idx] = 1.0
        return obs

    def step(self, action: int):
        move = self.ACTIONS[action]
        next_r = max(0, min(self.size - 1, self.state[0] + move[0]))
        next_c = max(0, min(self.size - 1, self.state[1] + move[1]))
        next_state = (next_r, next_c)

        if next_state in self.obstacles:
            reward = -5.0
            done = False
        elif next_state == self.goal:
            reward = 20.0
            done = True
            self.state = next_state
        else:
            reward = -0.1 # Step cost
            done = False
            self.state = next_state

        return self._get_obs(), reward, done, {}
