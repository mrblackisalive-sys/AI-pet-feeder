"""
This script trains the DQN agent inside our PetEnv environment.

Goal:
Teach the agent how much food to dispense at each time step.

Actions:
0 - dispense 0 grams
1 - dispense 10 grams
2 - dispense 20 grams

The agent learns based on rewards.

NOTE:
The reward system in Week 1 env is simple and placeholder-like.
In a real project, you would fine-tune reward shaping.
"""

import numpy as np
from env.pet_env import PetEnv
from dqn_agent import DQNAgent

# Define pets (same as Week 1)
pets = [
    {'id': 1, 'name': 'Fluffy', 'weight': 5,  'appetite': 1},
    {'id': 2, 'name': 'Spot',   'weight': 8,  'appetite': 1.5}
]

env = PetEnv(pets)

# DQN agent has:
# state = [time, bowl_weight, proximity]
agent = DQNAgent(state_size=3, action_size=3)

def state_from_obs(obs):
    """
    Convert observation dictionary from env into a numeric list.
    obs contains:
    - time
    - bowl_weight
    - proximity
    """
    return np.array([
        obs['time'] % 1440,     # Normalize time to 24 hours
        obs['bowl_weight'],     
        int(obs['proximity'])   # Convert True/False to 1/0
    ])

EPISODES = 30  # Train for 30 simulated days

for ep in range(EPISODES):
    obs = env.reset()
    state = state_from_obs(obs)

    done = False
    total_reward = 0

    while not done:
        # Agent chooses an action
        action_idx = agent.act(state)

        # Map the action index to a real-world action (grams)
        if action_idx == 0:
            amount = 0
        elif action_idx == 1:
            amount = 10
        else:
            amount = 20

        # Feed both pets equally for now (simple setup)
        action = {'dispense': {1: amount, 2: amount}, 'lock': False}

        # Take a step in the environment
        obs2, reward, done, _ = env.step(action)

        # Convert new observation to state vector
        next_state = state_from_obs(obs2)

        # Store experience
        agent.remember(state, action_idx, reward, next_state)

        # Train agent with random batch
        agent.train()

        # Move to next state
        state = next_state
        total_reward += reward

    print(f"Episode {ep}: Total Reward = {total_reward}")

print("[DONE] RL training finished.")
