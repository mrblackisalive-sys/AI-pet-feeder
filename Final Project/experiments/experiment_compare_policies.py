"""
This script compares:
1. Baseline policy (Week 1)
2. RL agent (Week 3)

It runs one episode with each policy and prints the total reward.

Note:
Better reward means better feeding behavior (in theory).
"""

from env.pet_env import PetEnv
from baseline.baseline_policy import baseline_policy
from rl.dqn_agent import DQNAgent
import numpy as np

# Same pet setup as Week 1
pets = [
    {'id': 1, 'name': 'Fluffy', 'weight': 5,  'appetite': 1},
    {'id': 2, 'name': 'Spot',   'weight': 8,  'appetite': 1.5}
]

env = PetEnv(pets)

# Create RL agent
agent = DQNAgent(state_size=3, action_size=3)

def run_policy(policy_name):
    """
    Run either the baseline or RL policy for one full episode.
    """
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        if policy_name == "baseline":
            # Use Week 1 baseline method
            action = baseline_policy(pets, obs)
        else:
            # RL policy
            state = np.array([obs['time'], obs['bowl_weight'], int(obs['proximity'])])
            a_idx = agent.act(state)
            grams = [0, 10, 20][a_idx]
            action = {'dispense': {1: grams, 2: grams}, 'lock': False}

        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward

# Compare policies
print("Baseline Policy Reward:", run_policy("baseline"))
print("RL Policy Reward:", run_policy("rl"))
