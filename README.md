# CliffWalking 🧗‍♂️  
Reinforcement Learning Agents solving the classic **CliffWalking-v0** environment from OpenAI Gym.

---

## Overview

This repository implements fundamental reinforcement learning algorithms — **Q-Learning** and **SARSA** — to train agents to navigate the CliffWalking gridworld environment. The goal is to move an agent from the start state to the goal without stepping into the "cliff" region, which results in a heavy penalty.

The environment provides an intuitive way to explore **on-policy** (SARSA) vs **off-policy** (Q-Learning) learning.

A **Random agent** is also included as a baseline for comparison.

Visualization with OpenCV helps track the agent's position on the grid during training.

---

## 📌 Environment: CliffWalking-v0

- **Grid size:** 4 rows × 12 columns  
- **Start state:** Bottom-left cell (state 36 in flattened form)  
- **Goal:** Bottom-right cell  
- **Cliff:** Bottom row cells between start and goal (positions 37 to 46)  
- **Actions:** {Up, Right, Down, Left} represented as {0, 1, 2, 3}  
- **Rewards:**  
  - `-1` for each step (to encourage shortest paths)  
  - `-100` for stepping into the cliff (episode terminates)  

The goal is to maximize cumulative reward by reaching the goal safely.

---

## 🔹 Algorithms Implemented

| Algorithm  | Type           | Description                                       | Update Formula                                      |
|------------|----------------|-------------------------------------------------|----------------------------------------------------|
| Q-Learning | Off-policy TD  | Learns optimal policy independent of behavior. More exploratory, faster convergence. | `Q[state][action] += α * (reward + γ * max(Q[next_state]) - Q[state][action])` |
| SARSA      | On-policy TD   | Learns policy based on agent's own trajectory. More conservative, avoids risky paths near cliff. | `Q[state][action] += α * (reward + γ * Q[next_state][next_action] - Q[state][action])` |
| Random Agent | Baseline     | Selects random actions, no learning involved.   | N/A                                                |

---

## 🗂️ Repository Structure

| File Name       | Description                                |
|-----------------|--------------------------------------------|
| `cv_phython.py`  | Utility functions shared by all agents     |
| `qlearning.py`   | Q-Learning algorithm implementation        |
| `sarsaagent.py`  | SARSA algorithm implementation             |
| `randomagent.py` | Random action agent for baseline performance |

---

## 📋 How to Run

### Prerequisites

- Python 3.x  
- OpenAI Gym (version with CliffWalking-v0)  
- NumPy  
- OpenCV (for visualization)  

Install dependencies using pip:

```bash
pip install gymnasium numpy opencv-python

