import numpy as np

# Define rewards and actions as before
rewards = np.array([4, 7, 1])
actions = ['a1', 'a2', 'a3']

# Transition matrices for each action
P_a1 = np.array([
    [0.3, 0.4, 0.3],
    [0.6, 0.2, 0.2],
    [0.2, 0.2, 0.6]
])

P_a2 = np.array([
    [0.5, 0.4, 0.1],
    [0.3, 0.2, 0.5],
    [0.4, 0.2, 0.4]
])

P_a3 = np.array([
    [0.25, 0.25, 0.5],
    [0.4, 0.3, 0.3],
    [0.2, 0.5, 0.3]
])

def policy_evaluation(policy, rewards, gamma=0.9, theta=1e-6):
    """Evaluate a policy by computing its value function"""
    V = np.zeros(len(rewards))
    
    while True:
        delta = 0
        V_old = V.copy()
        
        for s in range(len(rewards)):
            v = V[s]
            # Get transition matrix for the action specified by policy
            if policy[s] == 'a1':
                P = P_a1
            elif policy[s] == 'a2':
                P = P_a2
            else:
                P = P_a3
                
            # Calculate new value
            V[s] = rewards[s] + gamma * np.sum(P[s] * V_old)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
            
    return V

def policy_improvement(V, gamma=0.9):
    """Improve policy based on value function"""
    policy = [''] * len(rewards)
    
    for s in range(len(rewards)):
        # Calculate value for each action
        values = []
        
        # Try action a1
        value_a1 = rewards[s] + gamma * np.sum(P_a1[s] * V)
        values.append(('a1', value_a1))
        
        # Try action a2
        value_a2 = rewards[s] + gamma * np.sum(P_a2[s] * V)
        values.append(('a2', value_a2))
        
        # Try action a3
        value_a3 = rewards[s] + gamma * np.sum(P_a3[s] * V)
        values.append(('a3', value_a3))
        
        # Choose best action
        best_action, _ = max(values, key=lambda x: x[1])
        policy[s] = best_action
        
    return policy

def policy_iteration(gamma=0.9):
    """Complete policy iteration algorithm"""
    # Start with a random policy
    policy = ['a1', 'a1', 'a1']  # Initial policy
    iterations = 0
    
    while True:
        iterations += 1
        # Policy evaluation
        V = policy_evaluation(policy, rewards, gamma)
        
        # Policy improvement
        new_policy = policy_improvement(V, gamma)
        
        # Check if policy has converged
        if new_policy == policy:
            break
            
        policy = new_policy
        
    return policy, V, iterations

# Test policy iteration with different discount factors
discount_factors = [0.1, 0.01, 0.001, 0.3]

for gamma in discount_factors:
    print(f"\nDiscount factor: {gamma}")
    optimal_policy, values, iters = policy_iteration(gamma)
    print(f"Optimal policy: {optimal_policy}")
    print(f"State values: {values.round(4)}")
    print(f"Converged in {iters} iterations")