"""
This model predicts if a pet will finish their meal.
It uses a small Neural Network (MLPClassifier from sklearn).

Input features:
- bowl_weight
- pet_id

Output:
- 1 = pet is likely to finish meal
- 0 = pet is not likely to finish meal
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

class EatingPredictor:
    def __init__(self):
        # Simple neural network with 1 hidden layer of 16 neurons
        self.model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500)

    def train(self, X, y):
        # Fit the model
        self.model.fit(X, y)

    def predict(self, features):
        # Predict single sample
        return self.model.predict([features])[0]
