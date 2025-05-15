import random

def simulate_inventory(policies, days=100):
    results = []

    for policy in policies:
        P, Q = policy['P'], policy['Q']

        inventory = 115  # Initial stock
        outstanding_order = 0
        order_day = -1

        total_carrying_cost = 0
        total_lost_sales_cost = 0
        total_order_cost = 0

        for day in range(days):
            # Check if an order arrives today
            if day == order_day + 3:
                inventory += outstanding_order
                outstanding_order = 0

            # Simulate daily demand
            demand = random.randint(0, 99)

            # Fulfill demand
            if demand <= inventory:
                inventory -= demand
            else:
                # Calculate lost sales cost
                lost_sales = demand - inventory
                total_lost_sales_cost += lost_sales * 18
                inventory = 0

            # Carrying cost for remaining inventory
            total_carrying_cost += inventory * 75

            # Place an order if inventory drops to or below P
            if inventory <= P and outstanding_order == 0:
                outstanding_order = Q
                order_day = day
                total_order_cost += 75

        # Store results for this policy
        total_cost = total_carrying_cost + total_lost_sales_cost + total_order_cost
        results.append({
            'Policy': policy['name'],
            'Total Cost': total_cost,
            'Carrying Cost': total_carrying_cost,
            'Lost Sales Cost': total_lost_sales_cost,
            'Order Cost': total_order_cost
        })

    return results

# Define policies
policies = [
    {'name': 'Policy I', 'P': 125, 'Q': 150},
    {'name': 'Policy II', 'P': 125, 'Q': 250},
    {'name': 'Policy III', 'P': 150, 'Q': 250},
    {'name': 'Policy IV', 'P': 175, 'Q': 250},
    {'name': 'Policy V', 'P': 175, 'Q': 300},
]

# Run simulation
simulation_results = simulate_inventory(policies)

# Print results
for result in simulation_results:
    print(f"{result['Policy']}:\n"
          f"  Total Cost: ${result['Total Cost']:.2f}\n"
          f"  Carrying Cost: ${result['Carrying Cost']:.2f}\n"
          f"  Lost Sales Cost: ${result['Lost Sales Cost']:.2f}\n"
          f"  Order Cost: ${result['Order Cost']:.2f}\n")
