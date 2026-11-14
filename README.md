# Deep Reinforcement Learning: Autonomous Grid Navigator

![PyTorch](https://img.shields.io/badge/PyTorch-RL-red)
![Gymnasium](https://img.shields.io/badge/API-Gymnasium-green)
![Algorithms](https://img.shields.io/badge/Algorithms-Q--Learning%20%7C%20DQN%20%7C%20Double--DQN-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A modular deep reinforcement learning framework demonstrating value-based policy learning with experience replay, target networks, and spatial trajectory visualization.

## Architecture
```mermaid
flowchart LR
    State[Grid State Vector] --> Policy[Policy Q-Network]
    Policy --> Action[Epsilon-Greedy Action]
    Action --> Env[GridWorld Environment]
    Env --> Transition[S, A, R, S']
    Transition --> Buffer[Replay Memory]
    Buffer --> DoubleDQN[Bellman Loss Update]
```

## Run Simulation
```bash
python src/cli/play.py
```
