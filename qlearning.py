import gym
import numpy as np
import cv2
from collections import defaultdict

# Environment
cliffEnv = gym.make("CliffWalking-v0", render_mode="ansi")

# Hyperparameters
alpha = 0.5
gamma = 0.9
epsilon = 0.1
episodes = 50

# Initialize frame
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), (0, 0, 0), 1)
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), (0, 0, 0), 1)
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), (255, 0, 255), -1)
    img = cv2.putText(img, "Cliff", (49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, "G", (49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

# Draw agent
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(state, (4, 12))
    cv2.putText(img, "A", (49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

# Epsilon-greedy policy
def epsilon_greedy(Q, state):
    if np.random.rand() < epsilon:
        return cliffEnv.action_space.sample()
    return np.argmax(Q[state])

# Q-learning training
Q = defaultdict(lambda: np.zeros(cliffEnv.action_space.n))

for ep in range(episodes):
    state = cliffEnv.reset()[0]
    done = False
    frame = initialize_frame()

    while not done:
        frame2 = put_agent(frame.copy(), state)
        cv2.imshow("Q-Learning Cliff Walking", frame2)
        cv2.waitKey(50)

        action = epsilon_greedy(Q, state)
        next_state, reward, done, _, _ = cliffEnv.step(action)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

        state = next_state

print("✅ Q-Learning training complete!")
cliffEnv.close()
cv2.destroyAllWindows()
