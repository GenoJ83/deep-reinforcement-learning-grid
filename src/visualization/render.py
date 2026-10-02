"""ASCII and matrix state-value rendering utilities."""
import numpy as np

def render_ascii_grid(agent_pos, goal_pos, obstacles, size=5):
    grid = [["." for _ in range(size)] for _ in range(size)]
    for (r, c) in obstacles:
        grid[r][c] = "X"
    grid[goal_pos[0]][goal_pos[1]] = "G"
    grid[agent_pos[0]][agent_pos[1]] = "A"
    return "
".join(" ".join(row) for row in grid)
