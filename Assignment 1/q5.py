import numpy as np

def compute_value_function(P, R, gamma):
    """
    Solves the system (I - gamma * P)V = R to compute the value function.
    """
    V = np.linalg.solve(np.eye(3) - gamma * P, R)
    return V

def compute_convergence_iterations(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Iteratively computes the value function to check how many iterations it takes to converge.
    """
    V = np.zeros(3)  # Initial value function
    diff = float('inf')
    iters = 0
    while diff > tol and iters < max_iters:
        V_new = R + gamma * P @ V
        diff = np.max(np.abs(V_new - V))
        V = V_new
        iters += 1
    return iters, V

# Define the reward vector
R = np.array([4, 7, 1])

# Transition matrices under the given policy π
P = np.array([
    [0.3, 0.4, 0.3],  # Transition probabilities for S0 (a1)
    [0.6, 0.2, 0.2],  # Transition probabilities for S1 (a0)
    [0.4, 0.2, 0.4]   # Transition probabilities for S2 (a2)
])

# Discount factors to evaluate
gammas = [0.1, 0.01, 0.001, 0.3]

# Compute value functions and convergence iterations for each gamma
results = {}
convergence_iters = {}
for gamma in gammas:
    V = compute_value_function(P, R, gamma)
    iterations, _ = compute_convergence_iterations(P, R, gamma)
    results[gamma] = V
    convergence_iters[gamma] = iterations

# Print results
print("Discount Factor | V(S0)  | V(S1)  | V(S2)  | Iterations to Converge")
print("-------------------------------------------------------------")
for gamma in gammas:
    V = results[gamma]
    print(f"{gamma:<14} | {V[0]:.4f} | {V[1]:.4f} | {V[2]:.4f} | {convergence_iters[gamma]}")
