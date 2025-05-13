import numpy as np
import matplotlib.pyplot as plt

def simulate_time_1D(d, v, num_trials=10000):
    times = []
    for _ in range(num_trials):
        position = 0
        time = 0
        while position < d:
            forward = np.random.uniform(1, 10)
            backward = np.random.uniform(1, 5)
            position += forward - backward
            time += (forward + backward) / v
        times.append(time)
    return np.array(times)

def simulate_time_2D(d, v, num_trials=10000):
    times = []
    for _ in range(num_trials):
        position_x, position_y = 0, 0
        time = 0
        while np.sqrt(position_x**2 + position_y**2) < d:
            forward = np.random.uniform(1, 10)
            backward = np.random.uniform(1, 5)
            angle = np.random.uniform(0, np.pi)  # Angle between 0 and 180 degrees
            
            # Move forward in random direction
            position_x += forward * np.cos(angle)
            position_y += forward * np.sin(angle)
            
            # Move backward in random direction
            angle = np.random.uniform(0, np.pi)
            position_x -= backward * np.cos(angle)
            position_y -= backward * np.sin(angle)

            time += (forward + backward) / v
        
        times.append(time)
    return np.array(times)

def plot_cdf(data, title):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('CDF')
    plt.title(f'Cumulative Distribution Function ({title})')
    plt.grid()
    plt.legend()
    plt.show()

# Parameters
d = 100  # Distance between A and B
v = 20   # Constant velocity
num_trials = 10000  # Number of simulation runs

# Simulations
times_1D = simulate_time_1D(d, v, num_trials)
times_2D = simulate_time_2D(d, v, num_trials)

# Print results
print("Average time taken in 1D:", np.mean(times_1D))
print("Average time taken in 2D:", np.mean(times_2D))

# Plot CDFs
plot_cdf(times_1D, '1D Motion')
plot_cdf(times_2D, '2D Motion')
