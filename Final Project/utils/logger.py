"""
logger.py
Saves every step of the feeding cycle into a CSV file.
"""

import csv
import os

class Logger:
    def __init__(self, filepath="data/final_log.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "bowl_weight", "pet_detected",
                "dispensed", "reward", "anomaly"
            ])

    def log_step(self, obs, detected_pet, grams, reward, anomaly):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                obs["time"],
                obs["bowl_weight"],
                detected_pet,
                grams,
                reward,
                anomaly
            ])
