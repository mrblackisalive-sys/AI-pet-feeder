"""
visualization.py
Generates graphs using matplotlib for the final report.
"""

import matplotlib.pyplot as plt
import csv

def plot_food_dispensed(log_path="data/final_log.csv"):
    times = []
    dispensed = []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(int(row["time"]))
            dispensed.append(int(row["dispensed"]))

    plt.plot(times, dispensed)
    plt.title("Food Dispensed Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Grams Dispensed")
    plt.show()
