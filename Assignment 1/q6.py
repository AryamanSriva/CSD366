import numpy as np
import random
from collections import defaultdict
import pprint

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_data(num_samples, states, actions):
    data = []
    for _ in range(num_samples):
        state = random.choice(states)
        action = random.choice(actions)
        next_state = random.choice(states)
        data.append((state, action, next_state))
    return data

def compute_transition_matrix(data, states, actions):
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for state, action, next_state in data:
        transition_counts[state][action][next_state] += 1
    
    transition_matrix = {state: {action: {} for action in actions} for state in states}
    
    for state in states:
        for action in actions:
            total = sum(transition_counts[state][action].values())
            if total > 0:
                for next_state in states:
                    transition_matrix[state][action][next_state] = transition_counts[state][action][next_state] / total
            else:
                for next_state in states:
                    transition_matrix[state][action][next_state] = 0.0
    
    return transition_matrix

def value_iteration(transition_matrix, states, actions, rewards, gamma=0.9, theta=1e-6):
    V = {state: 0.0 for state in states}
    
    while True:
        delta = 0
        for state in states:
            v = V[state]
            max_value = max(
                sum(transition_matrix[state][action][next_state] * (rewards[state][action][next_state] + gamma * V[next_state]) 
                    for next_state in states) 
                for action in actions
            )
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

# Define states and actions
states = ['S1', 'S2', 'S3', 'S4', 'S5']
actions = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

# Generate datasets
data_1k = generate_data(1000, states, actions)
data_100k = generate_data(100000, states, actions)

# Compute transition probability matrices
transition_matrix_1k = compute_transition_matrix(data_1k, states, actions)
transition_matrix_100k = compute_transition_matrix(data_100k, states, actions)

# Define a simple reward function (for example, reward of 1 when reaching S5)
rewards = {state: {action: {next_state: 1 if next_state == 'S5' else 0 for next_state in states} 
                   for action in actions} for state in states}

# Compute value function for S1
V_1k = value_iteration(transition_matrix_1k, states, actions, rewards)
V_100k = value_iteration(transition_matrix_100k, states, actions, rewards)

# Print transition matrices
print("\nTransition Probability Matrix (1000 samples):")
pprint.pprint(transition_matrix_1k)

print("\nTransition Probability Matrix (100000 samples):")
pprint.pprint(transition_matrix_100k)

# Compare value function of S1
print(f"\nV(S1) with 1000 samples: {V_1k['S1']}")
print(f"V(S1) with 100000 samples: {V_100k['S1']}")
