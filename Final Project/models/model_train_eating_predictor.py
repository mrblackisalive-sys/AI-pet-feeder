"""
This script trains the Eating Predictor model using the data
generated in Week 1 (week1_episode_log.csv).

We use:
- bowl weight
- pet_at_bowl

We create labels:
- If bowl_weight < 5 - we assume pet finished most of the food (label = 1)
- Otherwise - pet didn't finish (label = 0)
"""

import csv
import numpy as np
from eating_predictor import EatingPredictor

if __name__ == "__main__":
    X, y = [], []

    # Load Week 1 log file
    with open("data/week1_episode_log.csv") as f:
        reader = csv.DictReader(f)

        for row in reader:
            bowl_w = float(row["bowl_weight"])
            pet = int(row["pet_at_bowl"])

            # Simple ground truth assumption
            success = 1 if bowl_w < 5 else 0

            X.append([bowl_w, pet])
            y.append(success)

    X = np.array(X)
    y = np.array(y)

    # Create model
    model = EatingPredictor()

    # Train model
    model.train(X, y)

    # Save model
    import pickle
    with open("models/eating_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("[DONE] Eating predictor trained and saved.")
