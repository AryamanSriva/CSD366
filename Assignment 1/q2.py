import numpy as np
import matplotlib.pyplot as plt

def simulate_fighter_bomber(num_trials=10000):
    fighter_start = np.array([0.0, 50.0])
    bomber_start = np.array([70.0, 0.0])
    hit_distance = 20
    times = []

    for _ in range(num_trials):
        fighter_pos = np.copy(fighter_start)
        bomber_pos = np.copy(bomber_start)
        time = 0

        while np.linalg.norm(fighter_pos - bomber_pos) > hit_distance:
            fighter_speed = np.random.uniform(10, 100)
            bomber_speed = np.random.uniform(10, 50)

            fighter_direction = (bomber_pos - fighter_pos) / np.linalg.norm(bomber_pos - fighter_pos)
            bomber_direction = np.random.uniform(-1, 1, 2)
            bomber_direction /= np.linalg.norm(bomber_direction)

            fighter_pos += fighter_speed * fighter_direction * 0.1
            bomber_pos += bomber_speed * bomber_direction * 0.1
            time += 0.1
        
        times.append(time)
    
    return np.array(times)



# Simulation
num_trials = 10000
times = simulate_fighter_bomber(num_trials)

# Print results
print("Average time for fighter to hit bomber:", np.mean(times))