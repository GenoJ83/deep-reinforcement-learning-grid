"""Policy rollout evaluation engine."""
import torch
import numpy as np
from src.environment.grid_world import GridWorldEnv

def run_policy_rollout(policy_net, max_steps: int = 20):
    env = GridWorldEnv()
    obs = env.reset()
    trajectory = [env.state]
    done = False
    total_reward = 0.0

    for _ in range(max_steps):
        if done:
            break
        with torch.no_grad():
            q = policy_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax().item())
        obs, r, done, _ = env.step(action)
        trajectory.append(env.state)
        total_reward += r

    return {"trajectory": trajectory, "total_reward": round(total_reward, 2), "reached_goal": done}
