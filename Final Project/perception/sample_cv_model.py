"""
This is a VERY simple computer vision model.
Instead of real deep learning, we use Logistic Regression.

The model:
- Takes a flattened (1D) 32x32 image = 1024 numbers
- Learns to classify whether the image is Fluffy or Spot
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

class SimpleCVModel:
    def __init__(self):
        # Create logistic regression model
        self.model = LogisticRegression(max_iter=500)

    def train(self, X, y):
        """
        Train the model.
        X = list of flattened images
        y = pet names (labels)
        """
        self.model.fit(X, y)

    def predict(self, img_array):
        """
        Predict which pet the image belongs to.
        img_array = 32x32 matrix
        """
        X = img_array.reshape(1, -1)  # Convert to row vector
        return self.model.predict(X)[0]

