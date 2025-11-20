"""
reward_shaping.py
Defines a better reward system for RL.
"""

def shaped_reward(ate, grams_dispensed):
    reward = 0

    # Reward correct eating
    if ate > 0:
        reward += 1

    # Penalty for waste
    waste = grams_dispensed - ate
    if waste > 5:
        reward -= waste * 0.1

    return reward
