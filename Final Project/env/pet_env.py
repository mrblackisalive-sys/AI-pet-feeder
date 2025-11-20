"""
Pet Environment Simulation (Week 1)
Simulates multiple pets, food bowl, and basic sensors for AI Smart Feeder 
"""

import numpy as np
import random

class PetEnv:
    def __init__(self, pets, seed=0):
        """
        Initialize environment with a list of pets.
        Each pet is a dict: {'id': int, 'name': str, 'weight': float, 'appetite': float}
        """
        random.seed(seed)
        np.random.seed(seed)
        self.pets = pets
        self.time = 0
        self.bowl_weight = 0.0
        self.pet_at_bowl = None

    def reset(self):
        """
        Reset the environment to initial state.
        Returns initial observation.
        """
        self.time = 0
        self.bowl_weight = 0.0
        for pet in self.pets:
            pet['last_meal_time'] = -999
            pet['hunger'] = np.random.uniform(0, 1)  # 0=full, 1=very hungry
        return self._get_obs()

    def step(self, action):
        """
        Apply action and simulate environment for one timestep.
        action: dict, e.g. {'dispense': {pet_id: grams}, 'lock': False}
        Returns: obs, reward, done, info
        """
        # Apply dispense action
        dispensed = sum(action.get('dispense', {}).values())
        # Simulate slight noise in dispensing
        self.bowl_weight += dispensed + np.random.normal(0, 0.2)

        # Simulate pets behavior
        events = self._simulate_pets()

        # Compute reward for baseline logging (placeholder)
        reward = self._compute_reward(action, events)

        # Increment time
        self.time += 1

        # Observation
        obs = self._get_obs()
        done = self.time >= 24*60  # 1-day episode (1440 mins)

        # Info dictionary (can include events, etc.)
        info = {'events': events}

        return obs, reward, done, info

    def _simulate_pets(self):
        """
        Simulate which pet approaches the bowl and eats.
        Returns a dict of pet_id and amount eaten.
        """
        # Simple approach: pet with highest hunger goes first
        probs = [p['hunger'] for p in self.pets]
        idx = np.argmax(probs)
        pet = self.pets[idx]

        # Pet eats minimum of bowl_weight and appetite * 10 grams
        ate = min(self.bowl_weight, pet['appetite'] * 10)
        self.bowl_weight -= ate

        # Update pet hunger
        pet['hunger'] = max(0, pet['hunger'] - 0.7)
        pet['last_meal_time'] = self.time

        # Small chance of stealing from another pet
        for other in self.pets:
            if other['id'] != pet['id'] and random.random() < 0.05:
                steal_amt = min(self.bowl_weight, 2)
                self.bowl_weight -= steal_amt
                ate += steal_amt

        self.pet_at_bowl = pet['id']
        return {'pet_id': pet['id'], 'ate': ate}

    def _get_obs(self):
        """
        Returns a dictionary representing the current observation
        """
        obs = {
            'time': self.time,
            'bowl_weight': self.bowl_weight + np.random.normal(0, 0.1),
            'proximity': bool(random.random() < 0.3)  # simulate proximity sensor
        }
        return obs

    def _compute_reward(self, action, events):
        """
        Placeholder reward function for logging purposes
        """
        return 0
