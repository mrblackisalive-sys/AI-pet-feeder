"""
pet_behavior.py
Defines extra behavior mechanics for pets.
"""

import numpy as np

def update_hunger(pet):
    # Hunger increases randomly
    pet["hunger"] += np.random.uniform(0.05, 0.2)
    pet["hunger"] = min(pet["hunger"], 1.0)
