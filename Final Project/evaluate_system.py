"""
evaluate_system.py
Runs multiple episodes and collects performance metrics.
"""

from system_pipeline import SystemPipeline
import numpy as np

def evaluate(env, pipeline, episodes=20):
    rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        total_reward = 0

        while not done:
            img = np.random.normal(0.5,0.2,(32,32))
            obs, done = pipeline.run_step(obs, img, 1)
            total_reward += 1

        rewards.append(total_reward)

    print("Average reward:", np.mean(rewards))
