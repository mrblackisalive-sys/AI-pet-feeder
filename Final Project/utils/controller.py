"""
controller.py
This file decides what feeding action to take by combining:
- CV model (pet identification)
- Eating predictor
- RL agent
- Anomaly detector
"""

import numpy as np
import pickle

class Controller:
    def __init__(self, cv_model_path, eating_model_path, rl_agent):
        # Load trained models
        with open(cv_model_path, "rb") as f:
            self.cv_model = pickle.load(f)
        with open(eating_model_path, "rb") as f:
            self.eating_model = pickle.load(f)

        self.rl_agent = rl_agent
        self.last_predicted_pet = None

    def detect_pet(self, img):
        """Use CV model to detect which pet is at the bowl."""
        pet_name = self.cv_model.predict(img)
        self.last_predicted_pet = pet_name
        return pet_name

    def predict_eating(self, bowl_weight, pet_id):
        """Predict whether the pet will finish the meal."""
        features = [bowl_weight, pet_id]
        return self.eating_model.predict(features)

    def choose_action(self, obs):
        """
        Decide action using RL agent.
        The RL chooses between 0g, 10g, 20g.
        """
        state = np.array([
            obs["time"],
            obs["bowl_weight"],
            int(obs["proximity"])
        ])

        action_idx = self.rl_agent.act(state)
        grams = [0, 10, 20][action_idx]

        return grams
