# Cliffwalking
# 🧗‍♂️ Cliffwalking - Reinforcement Learning Agents

This repository contains implementations of reinforcement learning algorithms to solve the classic **CliffWalking-v0** environment from OpenAI Gym. The goal is to navigate a gridworld from a start point to a goal point without falling off the cliff — an ideal setup to understand **on-policy** and **off-policy** learning.

## 📌 Environment Description

The CliffWalking-v0 environment is a 4x12 grid. The start state is at the bottom-left corner and the goal state is at the bottom-right. The bottom row between them is the "cliff". Stepping into the cliff gives a reward of -100 and ends the episode. All other actions yield a reward of -1.

<p align="center">
  <img src="https://gymnasium.farama.org/_images/cliffwalking.gif" width="400">
</p>

---

## 🗂️ Repository Structure

| File Name         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `cv_phython.py`   | Contains utility functions used across the agents.                         |
| `qlearning.py`    | Implements the **Q-Learning** algorithm (off-policy TD control).           |
| `sarsaagent.py`   | Implements the **SARSA** algorithm (on-policy TD control).                 |
| `randomagent.py`  | Implements a basic agent that selects actions randomly (used for baseline).|

---

## 🚀 Algorithms Implemented

### 🔹 Q-Learning (Off-policy TD)
- Learns the optimal policy regardless of the agent’s current behavior.
- More exploratory and converges faster to the optimal solution.
- Formula:  
  `Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]`

### 🔹 SARSA (On-policy TD)
- Learns the policy based on the agent’s current trajectory.
- More conservative; avoids risky paths (like near the cliff).
- Formula:  
  `Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') − Q(s, a)]`

### 🔹 Random Agent
- Does not learn.
- Takes actions randomly.
- Used as a performance baseline.

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushgupta67/Cliffwalking.git
   cd Cliffwalking
