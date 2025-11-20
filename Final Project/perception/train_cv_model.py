"""
This script trains the simple CV model using the fake image dataset created earlier.

Steps:
1. Load all .npy images
2. Flatten them into vectors
3. Load labels from CSV
4. Train the SimpleCVModel
5. Save trained model as .pkl file
"""

import os
import numpy as np
from simple_cv_model import SimpleCVModel

def load_dataset(path):
    X, y = [], []

    # Read label file
    with open(os.path.join(path, "labels.csv")) as f:
        for line in f:
            file, label = line.strip().split(",")

            # Load the image stored in .npy format
            img = np.load(os.path.join(path, file))

            # Flatten image into a single long line of numbers
            X.append(img.flatten())

            # Store label (pet name)
            y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_dir = "data/sim_images"

    # Load dataset
    X, y = load_dataset(data_dir)

    # Create the model
    model = SimpleCVModel()

    # Train it
    model.train(X, y)

    # Save the model to file
    import pickle
    with open("perception/cv_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("[DONE] CV model trained and saved.")