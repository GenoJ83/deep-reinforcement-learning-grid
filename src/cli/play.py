"""Interactive CLI demonstrating trained agent navigation."""
from src.environment.grid_world import GridWorldEnv
from src.agents.dqn_agent import QNetwork
from src.inference.evaluate_policy import run_policy_rollout
from src.visualization.render import render_ascii_grid

def main():
    print("Navigating GridWorld with Deep Q-Network...")
    policy = QNetwork(25, 4)
    result = run_policy_rollout(policy)
    print(f"Rollout Complete. Reward: {result['total_reward']}, Reached Goal: {result['reached_goal']}")
    print("Final Grid:")
    print(render_ascii_grid(result["trajectory"][-1], (4, 4), [(2, 2), (2, 3), (3, 2)]))

if __name__ == "__main__":
    main()
