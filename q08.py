import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_ARMS = 20
N_RUNS = 100
N_STEPS = 10000

def epsilon_greedy(epsilon, initial_estimates, steps, true_rewards):
    """Run epsilon-greedy algorithm for the bandit problem"""
    Q = np.ones(N_ARMS) * initial_estimates  # Action-value estimates
    N = np.zeros(N_ARMS)  # Action selection count
    rewards = np.zeros(steps)

    for t in range(steps):
        # Choose action based on epsilon-greedy policy
        if np.random.random() < epsilon:
            a = np.random.choice(N_ARMS)  # Explore
        else:
            a = np.argmax(Q)  # Exploit
            
        r = true_rewards[a]  # Get reward
        
        # Update estimates
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        
        rewards[t] = r
    
    return rewards

def run_experiment():
    """Run simulations for different strategies and plot results"""
    results = {
        "(a) ε-greedy with ε=0.1": np.zeros(N_STEPS),
        "(b) ε-greedy with ε=0.01": np.zeros(N_STEPS),
        "(c) greedy with init=0": np.zeros(N_STEPS),
        "(d) greedy with init=40": np.zeros(N_STEPS)
    }

    for run in range(N_RUNS):
        true_rewards = np.random.uniform(0, 20, N_ARMS)  # Regenerate rewards
        
        # Run each strategy
        results["(a) ε-greedy with ε=0.1"] += epsilon_greedy(0.1, 0, N_STEPS, true_rewards)
        results["(b) ε-greedy with ε=0.01"] += epsilon_greedy(0.01, 0, N_STEPS, true_rewards)
        results["(c) greedy with init=0"] += epsilon_greedy(0, 0, N_STEPS, true_rewards)
        results["(d) greedy with init=40"] += epsilon_greedy(0, 40, N_STEPS, true_rewards)

    # Average results over runs
    for key in results:
        results[key] /= N_RUNS

    # Calculate total rewards
    total_rewards = {method: np.sum(rewards) for method, rewards in results.items()}
    best_method = max(total_rewards, key=total_rewards.get)

    # Plot average reward over time
    plt.figure(figsize=(10, 6))
    for method, rewards in results.items():
        plt.plot(np.cumsum(rewards) / np.arange(1, N_STEPS + 1), label=f"{method}")
    
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Step for Different Methods')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("average_reward.png")  # Save figure
    plt.show()

    # Plot cumulative rewards
    plt.figure(figsize=(10, 6))
    for method, rewards in results.items():
        plt.plot(np.cumsum(rewards), label=f"{method}")

    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward for Different Methods')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("cumulative_reward.png")
    plt.show()

    # Print results
    print("\nTotal accumulated rewards over", N_STEPS, "steps:")
    for method, total in sorted(total_rewards.items(), key=lambda x: x[1], reverse=True):
        print(f"{method}: {total:.2f}")

    print(f"\nBest method: {best_method}")

if __name__ == "__main__":
    run_experiment()
