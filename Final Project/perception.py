"""
This script creates a FAKE image dataset for Week 2.
Since we don't have real pet images, we generate small numeric matrices (32x32)
that represent 'images'. Each image belongs to either Fluffy or Spot.

Fluffy images are generated with lower values.
Spot images are generated with higher values.
This difference makes it easy for the computer to learn.
"""

import numpy as np
import os

def generate_simulated_images(output_dir, num_images=200):
    # Create folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Two pets that we want to classify
    pets = ["Fluffy", "Spot"]

    labels = []  # Store labels for each image

    for i in range(num_images):
        # Randomly choose a pet for this image
        pet = np.random.choice(pets)

        # Generate simple numeric "image" patterns
        if pet == "Fluffy":
            # Fluffy = numbers around 0.2
            img = np.random.normal(0.2, 0.05, (32, 32))
        else:
            # Spot = numbers around 0.8
            img = np.random.normal(0.8, 0.05, (32, 32))

        # Save image as .npy file
        np.save(os.path.join(output_dir, f"img_{i}.npy"), img)

        # Add label (image name + pet name)
        labels.append([f"img_{i}.npy", pet])

    # Write label file
    with open(os.path.join(output_dir, "labels.csv"), "w") as f:
        for file, pet in labels:
            f.write(f"{file},{pet}\n")

    print(f"[DONE] Generated {num_images} simulated images.")
