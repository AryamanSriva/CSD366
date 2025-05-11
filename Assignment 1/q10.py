import numpy as np
import random
from collections import defaultdict

# Constants
HOLDING_COST = 75             # Per unit per night
ORDER_COST = 75               # Flat cost per order
STOCKOUT_COST = 18            # Per unit lost sale
MAX_DEMAND = 99
INITIAL_STOCK = 115
LAG_DAYS = 3
EPISODES = 10000
ALPHA = 0.1                   # Learning rate
GAMMA = 0.99                  # Discount factor

# Policies: (reorder point, reorder quantity)
POLICIES = {
    "Policy I": (125, 125),
    "Policy II": (125, 250),
    "Policy III": (150, 250),
    "Policy IV": (175, 250),
    "Policy V": (175, 300)
}


def simulate_td(policy_name, episodes=EPISODES):
    P, Q = POLICIES[policy_name]
    V = defaultdict(float)

    for episode in range(episodes):
        inventory = INITIAL_STOCK
        order_arrival_day = None
        day = 0
        state = inventory

        while inventory > 0:
            # Handle arrival
            if order_arrival_day is not None and day == order_arrival_day:
                inventory += Q
                order_arrival_day = None

            # Daily demand
            demand = random.randint(0, MAX_DEMAND)

            sold = min(inventory, demand)
            lost_sales = max(0, demand - inventory)
            inventory -= sold

            # Cost calculation
            holding_cost = inventory * HOLDING_COST
            stockout_cost = lost_sales * STOCKOUT_COST
            order_cost = 0

            # Place order if inventory <= reorder point and no outstanding order
            if inventory <= P and order_arrival_day is None:
                order_arrival_day = day + LAG_DAYS
                order_cost = ORDER_COST

            # Reward is negative of cost
            reward = -(holding_cost + stockout_cost + order_cost)

            next_state = inventory
            V[state] += ALPHA * (reward + GAMMA * V[next_state] - V[state])

            state = next_state
            day += 1

    return V


# Run simulation for all policies and compare value of initial state
if __name__ == "__main__":
    for policy in POLICIES:
        V = simulate_td(policy)
        print(f"{policy}: Estimated Value of initial state (inventory = {INITIAL_STOCK}): {V[INITIAL_STOCK]:.2f}")