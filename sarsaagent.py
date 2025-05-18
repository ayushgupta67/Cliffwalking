import gym
import numpy as np
import cv2
from collections import defaultdict

# Create the environment
cliffEnv = gym.make("CliffWalking-v0", render_mode="ansi")

# SARSA parameters
alpha = 0.5
gamma = 0.9
epsilon = 0.1
Q = defaultdict(lambda: np.zeros(cliffEnv.action_space.n))

# Initialize the frame (grid + cliff + goal)
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    img = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img

# Draw agent at current state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img

# Epsilon-greedy action selection
def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        return cliffEnv.action_space.sample()
    return np.argmax(Q[state])

# Run multiple episodes of SARSA with optional visualization
num_episodes = 10
rewards_per_episode = []

for episode in range(num_episodes):
    done = False
    state = cliffEnv.reset()[0]
    action = epsilon_greedy(state)
    frame = initialize_frame()
    total_reward = 0

    while not done:
        frame2 = put_agent(frame.copy(), state)

        # Visualize only last few episodes
        if episode >= num_episodes - 5:
            cv2.imshow("Cliff Walking - SARSA", frame2)
            cv2.waitKey(100)

        # Take action and observe result
        next_state, reward, terminated, _, _ = cliffEnv.step(action)
        next_action = epsilon_greedy(next_state)

        # SARSA update
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

        # Move to next state
        state = next_state
        action = next_action
        done = terminated
        total_reward += reward

    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1} completed | Total Reward: {total_reward}")

cliffEnv.close()
cv2.destroyAllWindows()
