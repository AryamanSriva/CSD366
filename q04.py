import numpy as np
import matplotlib.pyplot as plt

def simulate_mm1_queue(arrival_rate, service_rate, num_jobs=10000):
    """
    Simulate M/M/1 queue and compute key metrics
    
    Args:
    arrival_rate (float): λ - job arrival rate
    service_rate (float): μ - job service rate
    num_jobs (int): Number of jobs to simulate
    
    Returns:
    dict: Computed queue metrics
    """
    # Check stability condition
    if arrival_rate >= service_rate:
        raise ValueError("System is unstable: λ must be < μ")
    
    # Track system state
    arrival_times = np.cumsum(np.random.exponential(1/arrival_rate, num_jobs))
    service_times = np.random.exponential(1/service_rate, num_jobs)
    
    # Compute queue metrics
    departure_times = np.zeros_like(arrival_times)
    response_times = np.zeros_like(arrival_times)
    system_jobs = np.zeros_like(arrival_times)
    
    current_time = 0
    jobs_in_system = 0
    last_departure = 0
    
    for i in range(num_jobs):
        # Wait until current job can start
        start_time = max(arrival_times[i], last_departure)
        
        # Compute departure
        departure_times[i] = start_time + service_times[i]
        last_departure = departure_times[i]
        
        # Response time and system jobs
        response_times[i] = departure_times[i] - arrival_times[i]
        
        # Track jobs in system
        jobs_in_system += 1
        for j in range(i):
            if departure_times[j] <= arrival_times[i]:
                jobs_in_system -= 1
    
    # Compute aggregate metrics
    mean_response_time = np.mean(response_times)
    mean_system_jobs = np.mean(jobs_in_system)
    
    return {
        'E[T]': mean_response_time, 
        'E[N]': mean_system_jobs,
        'Little\'s Law Check': arrival_rate * mean_response_time
    }

def plot_queue_metrics():
    """Plot queue metrics for different arrival rates"""
    service_rate = 1.0  # μ = 1
    arrival_rates = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    response_times = []
    system_jobs = []
    little_law_checks = []
    
    # Compute metrics for each arrival rate
    for λ in arrival_rates:
        results = simulate_mm1_queue(λ, service_rate)
        response_times.append(results['E[T]'])
        system_jobs.append(results['E[N]'])
        little_law_checks.append(results['Little\'s Law Check'])
    
    # Plotting
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(arrival_rates, response_times, marker='o')
    plt.title('Mean Response Time E[T]')
    plt.xlabel('Arrival Rate (λ)')
    plt.ylabel('E[T]')
    
    plt.subplot(1,2,2)
    plt.plot(arrival_rates, system_jobs, marker='o')
    plt.title('Mean Number of Jobs E[N]')
    plt.xlabel('Arrival Rate (λ)')
    plt.ylabel('E[N]')
    
    plt.tight_layout()
    plt.show()
    
    # Print results for verification
    print("Arrival Rates | E[T] | E[N] | Little's Law Check")
    for λ, t, n, check in zip(arrival_rates, response_times, system_jobs, little_law_checks):
        print(f"{λ:.1f}        | {t:.4f} | {n:.4f} | {check:.4f}")

# Run the simulation and plotting
plot_queue_metrics()