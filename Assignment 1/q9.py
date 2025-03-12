import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.q_true = np.random.normal(0, 1, n_arms)  # True action values
        self.q_estimates = np.zeros(n_arms)  # Estimated action values
        self.action_counts = np.zeros(n_arms)  # Action counts
        self.t = 0  # Time step
    
    def select_action_ucb(self, c):
        self.t += 1
        ucb_values = self.q_estimates + c * np.sqrt(np.log(self.t) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)
    
    def select_action_eps_greedy(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_estimates)
    
    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        return reward

def run_experiment(n_bandits=2000, n_steps=1000, c_values=[1, 2, 3, 4], epsilons=[0.1, 0.01, 0.001]):
    avg_rewards_ucb = {c: np.zeros(n_steps) for c in c_values}
    avg_rewards_eps = {eps: np.zeros(n_steps) for eps in epsilons}
    
    for _ in range(n_bandits):
        bandit = Bandit()
        rewards_ucb = {c: [] for c in c_values}
        rewards_eps = {eps: [] for eps in epsilons}
        
        for t in range(n_steps):
            for c in c_values:
                action = bandit.select_action_ucb(c)
                reward = bandit.step(action)
                rewards_ucb[c].append(reward)
                avg_rewards_ucb[c][t] += reward
            
            for eps in epsilons:
                action = bandit.select_action_eps_greedy(eps)
                reward = bandit.step(action)
                rewards_eps[eps].append(reward)
                avg_rewards_eps[eps][t] += reward
    
    for c in c_values:
        avg_rewards_ucb[c] /= n_bandits
    for eps in epsilons:
        avg_rewards_eps[eps] /= n_bandits
    
    return avg_rewards_ucb, avg_rewards_eps

def plot_results(avg_rewards_ucb, avg_rewards_eps, n_steps=1000):
    plt.figure(figsize=(12, 6))
    
    for c, rewards in avg_rewards_ucb.items():
        plt.plot(rewards, label=f'UCB c={c}')
    for eps, rewards in avg_rewards_eps.items():
        plt.plot(rewards, linestyle='--', label=f'ε-greedy ε={eps}')
    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('10-Armed Bandit: UCB vs. ε-Greedy')
    plt.legend()
    plt.show()

# Run the experiment and plot results
avg_rewards_ucb, avg_rewards_eps = run_experiment()
plot_results(avg_rewards_ucb, avg_rewards_eps)
