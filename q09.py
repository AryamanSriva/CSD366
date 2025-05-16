import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Bandit:
    def __init__(self, k=10):
        # True action values for each arm
        self.k = k
        self.true_values = np.random.normal(0, 1, k)

    def pull(self, action):
        # Return reward with noise
        return np.random.normal(self.true_values[action], 1)

    def reset(self):
        # Reset the true values for a new run
        self.true_values = np.random.normal(0, 1, self.k)


def ucb_action_selection(Q, N, t, c):
    """
    Upper Confidence Bound action selection

    Args:
        Q: estimated action values
        N: count of actions taken
        t: current time step
        c: confidence level
    """
    # Handle the case where actions haven't been selected yet
    if np.any(N == 0):
        return np.random.choice(np.where(N == 0)[0])

    # Calculate UCB values
    ucb_values = Q + c * np.sqrt(np.log(t) / N)
    return np.argmax(ucb_values)


def epsilon_greedy_action_selection(Q, epsilon):
    """
    Epsilon-greedy action selection

    Args:
        Q: estimated action values
        epsilon: exploration rate
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(Q))  # Explore
    else:
        return np.argmax(Q)  # Exploit


def run_experiment(bandit, num_runs=2000, num_steps=1000, c_values=[1, 2, 3, 4], epsilon_values=[0.1, 0.01, 0.001]):
    """
    Run the experiment with UCB action selection

    Args:
        bandit: the bandit environment
        num_runs: number of independent runs
        num_steps: number of steps per run
        c_values: list of confidence levels to test
        epsilon_values: list of epsilon values to test
    """
    # Dictionary to store results
    results = {}

    # Run UCB experiments for each c value
    for c in c_values:
        average_rewards = np.zeros(num_steps)
        optimal_actions = np.zeros(num_steps)

        for _ in tqdm(range(num_runs), desc=f"UCB c={c}"):
            bandit.reset()
            optimal_action = np.argmax(bandit.true_values)

            # Initialize estimates and counts
            Q = np.zeros(bandit.k)
            N = np.zeros(bandit.k)

            for t in range(num_steps):
                # Select action using UCB
                action = ucb_action_selection(Q, N, t+1, c)

                # Get reward
                reward = bandit.pull(action)

                # Update estimates and counts
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]

                # Track metrics
                average_rewards[t] += reward
                optimal_actions[t] += (action == optimal_action)

        # Average over runs
        average_rewards /= num_runs
        optimal_actions /= num_runs

        results[f"UCB c={c}"] = {
            "rewards": average_rewards,
            "optimal_actions": optimal_actions
        }

    # Run epsilon-greedy experiments for each epsilon value
    for epsilon in epsilon_values:
        average_rewards = np.zeros(num_steps)
        optimal_actions = np.zeros(num_steps)

        for _ in tqdm(range(num_runs), desc=f"ε-greedy ε={epsilon}"):
            bandit.reset()
            optimal_action = np.argmax(bandit.true_values)

            # Initialize estimates and counts
            Q = np.zeros(bandit.k)
            N = np.zeros(bandit.k)

            for t in range(num_steps):
                # Select action using epsilon-greedy
                action = epsilon_greedy_action_selection(Q, epsilon)

                # Get reward
                reward = bandit.pull(action)

                # Update estimates and counts
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]

                # Track metrics
                average_rewards[t] += reward
                optimal_actions[t] += (action == optimal_action)

        # Average over runs
        average_rewards /= num_runs
        optimal_actions /= num_runs

        results[f"ε-greedy ε={epsilon}"] = {
            "rewards": average_rewards,
            "optimal_actions": optimal_actions
        }

    return results


def plot_results(results):
    """
    Plot the results of the experiment

    Args:
        results: dictionary with experiment results
    """
    # Set up the figure and axes for plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Plot average rewards
    for method, data in results.items():
        ax1.plot(data["rewards"], label=method)

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Average Reward over Steps")
    ax1.legend()
    ax1.grid(True)



    plt.tight_layout()
    plt.savefig("ucb_comparison.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # Define parameters
    k = 10  # 10-armed bandit
    num_runs = 1000
    num_steps = 1000
    c_values = [1, 2, 3, 4]
    epsilon_values = [0.1, 0.01, 0.001]

    # Create bandit environment
    bandit = Bandit(k=k)

    # Run experiment
    results = run_experiment(
        bandit=bandit,
        num_runs=num_runs,
        num_steps=num_steps,
        c_values=c_values,
        epsilon_values=epsilon_values
    )

    # Plot results
    plot_results(results)
