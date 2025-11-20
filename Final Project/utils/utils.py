"""
utils.py
Small helper functions.
"""

import numpy as np

def normalize(value, max_value):
    return value / max_value

def choose_random_pet(pets):
    return np.random.choice(pets)
