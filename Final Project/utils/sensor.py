"""
sensors.py
Simulates camera, RFID, and weight sensors.
"""

import numpy as np

def simulate_camera(pet_name):
    """Return a fake 'image' matrix."""
    if pet_name == "Fluffy":
        return np.random.normal(0.2, 0.05, (32,32))
    return np.random.normal(0.8, 0.05, (32,32))

def simulate_rfid(pet):
    return pet["id"]

def simulate_weight(actual_weight):
    return actual_weight + np.random.normal(0,0.2)
