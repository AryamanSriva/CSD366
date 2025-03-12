import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.q_true = np.random.normal(0, 1, n_arms)  # True action values
    
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)  # Reward with Gaussian noise

class Agent:
    def __init__(self, n_arms=10, epsilon=0.1, optimistic_init=0.0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_estimates = np.full(n_arms, optimistic_init)  # Initial estimates
        self.action_counts = np.zeros(n_arms)
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)  # Explore
        return np.argmax(self.q_estimates)  # Exploit
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]  # Step size for sample average
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

def run_experiment(n_bandits=2000, n_steps=1000, epsilon=0.1, optimistic_init=0.0):
    bandits = [Bandit() for _ in range(n_bandits)]
    agents = [Agent(epsilon=epsilon, optimistic_init=optimistic_init) for _ in range(n_bandits)]
    rewards = np.zeros(n_steps)
    
    for t in range(n_steps):
        step_rewards = []
        for bandit, agent in zip(bandits, agents):
            action = agent.select_action()
            reward = bandit.get_reward(action)
            agent.update(action, reward)
            step_rewards.append(reward)
        rewards[t] = np.mean(step_rewards)
    
    return rewards

# Run experiments
greedy_rewards = run_experiment(epsilon=0.0, optimistic_init=0.0)
eps_greedy_rewards = run_experiment(epsilon=0.1, optimistic_init=0.0)
eps_greedy_opt_rewards = run_experiment(epsilon=0.1, optimistic_init=5.0)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(greedy_rewards, label='Greedy (ε=0, Init=0)')
plt.plot(eps_greedy_rewards, label='ε-Greedy (ε=0.1, Init=0)')
plt.plot(eps_greedy_opt_rewards, label='ε-Greedy (ε=0.1, Init=5)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('10-Armed Bandit Performance')
plt.show()
