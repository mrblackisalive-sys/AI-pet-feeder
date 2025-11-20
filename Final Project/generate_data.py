"""
Generate synthetic episode data using the PetEnv and baseline policy.
Logs observations and actions to CSV.
"""

import csv
from env.pet_env import PetEnv
from baseline.baseline_policy import baseline_policy

# Define pets
pets = [
    {'id': 1, 'name': 'Fluffy', 'weight': 5.0, 'appetite': 1.0},  # appetite in arbitrary units
    {'id': 2, 'name': 'Spot', 'weight': 8.0, 'appetite': 1.5}
]

# Initialize environment
env = PetEnv(pets)
num_episodes = 5  # small number for test; increase for real data

with open('data/week1_episode_log.csv', 'w', newline='') as csvfile:
    fieldnames = ['episode', 'time', 'pet_at_bowl', 'bowl_weight', 'action_dispense', 'reward']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = baseline_policy(pets, obs)
            obs, reward, done, info = env.step(action)
            writer.writerow({
                'episode': ep,
                'time': obs['time'],
                'pet_at_bowl': info['events']['pet_id'],
                'bowl_weight': round(obs['bowl_weight'],2),
                'action_dispense': action['dispense'],
                'reward': reward
            })

print("Week 1 data generation complete. Log saved to data/week1_episode_log.csv")
