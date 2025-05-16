import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define 3D grid world size (X, Y, Z)
grid_size = (7, 10, 5)  # (rows, columns, height levels)

# Define wind effect per column (only affects Y-axis)
wind_strength = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0, 0])

# Actions: (Up, Down, Left, Right, Ascend, Descend)
actions = {'up': (-1, 0, 0), 'down': (1, 0, 0), 'left': (0, -1, 0), 'right': (0, 1, 0), 
           'ascend': (0, 0, 1), 'descend': (0, 0, -1)}

action_list = list(actions.keys())
num_actions = len(action_list)

# Initialize Q-tables for SARSA and Q-Learning
Q_sarsa = np.zeros((*grid_size, num_actions))
Q_qlearn = np.zeros((*grid_size, num_actions))

# Hyperparameters
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration probability
reward_step = -1  # Penalty for each step
reward_goal = 10  # Reward for reaching the goal
episodes = 1000  # Number of training episodes

# Start and Goal
start = (3, 0, 0)  # (X, Y, Z)
goal = (3, 7, 3)   # (X, Y, Z)

# Epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Random action (exploration)
    else:
        return np.argmax(Q[state])  # Best action (exploitation)

# *Training SARSA*
for episode in range(episodes):
    state = start
    action_idx = epsilon_greedy(Q_sarsa, state, epsilon)

    while state != goal:
        # Compute next state
        next_state = (state[0] + actions[action_list[action_idx]][0],  
                      state[1] + actions[action_list[action_idx]][1],  
                      state[2] + actions[action_list[action_idx]][2])

        # Apply wind effect
        next_state = (max(0, min(grid_size[0] - 1, next_state[0] - wind_strength[state[1]])),  
                      max(0, min(grid_size[1] - 1, next_state[1])),  
                      max(0, min(grid_size[2] - 1, next_state[2])))

        # Reward
        reward = reward_goal if next_state == goal else reward_step

        # Choose next action (SARSA)
        next_action_idx = epsilon_greedy(Q_sarsa, next_state, epsilon)

        # SARSA update
        Q_sarsa[state][action_idx] += alpha * (reward + gamma * Q_sarsa[next_state][next_action_idx] - Q_sarsa[state][action_idx])

        # Move to next state
        state, action_idx = next_state, next_action_idx

# *Training Q-Learning*
for episode in range(episodes):
    state = start

    while state != goal:
        action_idx = epsilon_greedy(Q_qlearn, state, epsilon)

        # Compute next state
        next_state = (state[0] + actions[action_list[action_idx]][0],  
                      state[1] + actions[action_list[action_idx]][1],  
                      state[2] + actions[action_list[action_idx]][2])

        # Apply wind effect
        next_state = (max(0, min(grid_size[0] - 1, next_state[0] - wind_strength[state[1]])),  
                      max(0, min(grid_size[1] - 1, next_state[1])),  
                      max(0, min(grid_size[2] - 1, next_state[2])))

        # Reward
        reward = reward_goal if next_state == goal else reward_step

        # Q-Learning update
        Q_qlearn[state][action_idx] += alpha * (reward + gamma * np.max(Q_qlearn[next_state]) - Q_qlearn[state][action_idx])

        # Move to next state
        state = next_state

# *Extract Best Path for SARSA*
state = start
path_sarsa = [state]
while state != goal:
    action_idx = np.argmax(Q_sarsa[state])
    next_state = (state[0] + actions[action_list[action_idx]][0],  
                  state[1] + actions[action_list[action_idx]][1],  
                  state[2] + actions[action_list[action_idx]][2])

    # Apply wind effect
    next_state = (max(0, min(grid_size[0] - 1, next_state[0] - wind_strength[state[1]])),  
                  max(0, min(grid_size[1] - 1, next_state[1])),  
                  max(0, min(grid_size[2] - 1, next_state[2])))

    state = next_state
    path_sarsa.append(state)

# *Extract Best Path for Q-Learning*
state = start
path_qlearn = [state]
while state != goal:
    action_idx = np.argmax(Q_qlearn[state])
    next_state = (state[0] + actions[action_list[action_idx]][0],  
                  state[1] + actions[action_list[action_idx]][1],  
                  state[2] + actions[action_list[action_idx]][2])

    # Apply wind effect
    next_state = (max(0, min(grid_size[0] - 1, next_state[0] - wind_strength[state[1]])),  
                  max(0, min(grid_size[1] - 1, next_state[1])),  
                  max(0, min(grid_size[2] - 1, next_state[2])))

    state = next_state
    path_qlearn.append(state)

# *3D Visualization of Both Paths*
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert paths to X, Y, Z coordinates
X_sarsa, Y_sarsa, Z_sarsa = zip(*path_sarsa)
X_qlearn, Y_qlearn, Z_qlearn = zip(*path_qlearn)

# Plot SARSA Path
ax.plot(X_sarsa, Y_sarsa, Z_sarsa, color='blue', marker='o', label="SARSA Path")

# Plot Q-Learning Path
ax.plot(X_qlearn, Y_qlearn, Z_qlearn, color='red', marker='x', label="Q-Learning Path")

# Mark Start and Goal
ax.scatter(start[0], start[1], start[2], color='green', s=100, label="Start (S)")
ax.scatter(goal[0], goal[1], goal[2], color='black', s=100, label="Goal (G)")

# Labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Height (Z)")
ax.set_title("3D Windy Grid World: SARSA vs. Q-Learning")
ax.legend()

plt.show()