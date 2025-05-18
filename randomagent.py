import gym
import numpy as np
import cv2

cliffEnv = gym.make("CliffWalking-v0", render_mode="ansi")

def initialize_frame():
    width, height = 600, 200
    #render_mode="ansi" means the environment will render its output as simple text in the background, instead of using graphical or interactive visualizations.
    #But it is not used
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2
    # Iterates through grid columns (12 spaces + 1 for the boundary)
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
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame

def put_agent(img, state):
    #put_agent(): Updates the frame with the agent's current position.
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img

done = False
frame = initialize_frame()
state = cliffEnv.reset()[0]
#state = cliffEnv.reset()[0]: Resets the environment to its initial state and retrieves the starting position of the agent.


while not done:
    frame2 = put_agent(frame.copy(), state)
    #frame.copy(): Creates a copy of the base frame to modify without overwriting.
    cv2.imshow("Cliff Walking", frame2)

    cv2.waitKey(100)
    action = int(np.random.randint(low=0, high=4, size=1))
    #np.random.randint: Selects a random action (0: up, 1: right, 2: down, 3: left).
    next_state, reward, terminated,_, _ = cliffEnv.step(action)
    done = terminated
    #In the gym environment's step() function, it typically returns five values:
    state = next_state
#The fifth value info contains metadata about the environment, but since it's not used in your code, I replaced it with _ to indicate that it’s intentionally ignored.
cliffEnv.close()
cv2.destroyAllWindows()
