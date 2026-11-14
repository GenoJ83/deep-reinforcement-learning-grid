"""Random policy baseline agent."""
import numpy as np
from src.environment.grid_world import GridWorldEnv

def evaluate_random_policy(episodes: int = 50):
    env = GridWorldEnv()
    rewards = []
    for _ in range(episodes):
        env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < 100:
            action = np.random.randint(0, 4)
            _, r, done, _ = env.step(action)
            ep_reward += r
            steps += 1
        rewards.append(ep_reward)
    mean_r = sum(rewards) / len(rewards)
    print(f"Random Baseline Average Reward: {mean_r:.2f}")
    return mean_r

if __name__ == "__main__":
    evaluate_random_policy()
