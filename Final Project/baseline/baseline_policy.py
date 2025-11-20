"""
Baseline rule-based feeder policy
Dispenses food at fixed schedule based on pet appetite.
"""

def baseline_policy(pets, obs):
    """
    Simple rule-based policy:
    - Dispense each pet's appetite*10 grams if bowl is empty
    - Otherwise, do nothing
    """
    action = {'dispense': {}, 'lock': False}
    if obs['bowl_weight'] < 1.0:  # bowl is empty
        for pet in pets:
            action['dispense'][pet['id']] = pet['appetite'] * 10  # grams
    return action
