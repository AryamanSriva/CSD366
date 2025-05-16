import matplotlib.pyplot as plt
import numpy as np
import random

# Environment parameters
GRID_HEIGHT = 4
GRID_WIDTH = 12
START_STATE = (3, 0)
GOAL_STATE = (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Hyperparameters
ALPHA = 0.5
GAMMA = 1.0
EPSILONS = [0.1, 0.01, 0.001]
EPISODES = 500

# Utility functions
def step(state, action):
    x, y = state
    if action == 'UP':
        x = max(x - 1, 0)
    elif action == 'DOWN':
        x = min(x + 1, GRID_HEIGHT - 1)
    elif action == 'LEFT':
        y = max(y - 1, 0)
    elif action == 'RIGHT':
        y = min(y + 1, GRID_WIDTH - 1)

    reward = -1
    if (x, y) in CLIFF:
        return START_STATE, -100
    elif (x, y) == GOAL_STATE:
        return GOAL_STATE, 0
    return (x, y), reward

# Epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[state][a])

# Initialize Q-values
def init_Q():
    return {(i, j): {a: 0 for a in ACTIONS} for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH)}

# Q-Learning algorithm
def q_learning(Q, epsilon):
    rewards = []
    for _ in range(EPISODES):
        state = START_STATE
        episode_reward = 0
        while state != GOAL_STATE:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward = step(state, action)
            best_next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += ALPHA * (reward + GAMMA * Q[next_state][best_next_action] - Q[state][action])
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

# SARSA algorithm
def sarsa(Q, epsilon):
    rewards = []
    for _ in range(EPISODES):
        state = START_STATE
        action = epsilon_greedy(Q, state, epsilon)
        episode_reward = 0
        while state != GOAL_STATE:
            next_state, reward = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state][action] += ALPHA * (reward + GAMMA * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

# Apply moving average for smoother curves
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Run algorithms
plt.figure(figsize=(8, 5))
for epsilon in EPSILONS:
    Q_q = init_Q()
    Q_sarsa = init_Q()
    q_rewards = moving_average(q_learning(Q_q, epsilon))
    sarsa_rewards = moving_average(sarsa(Q_sarsa, epsilon))
    plt.plot(sarsa_rewards, label=f'Sarsa ε={epsilon}')
    plt.plot(q_rewards, linestyle='--', label=f'Q-learning ε={epsilon}')

# Plot results
plt.xlabel("Episodes")
plt.ylabel("Reward per episode")
plt.legend()
plt.show()
