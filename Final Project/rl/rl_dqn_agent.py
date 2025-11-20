"""
This is a SIMPLE version of a Deep Q-Learning (DQN) agent.

Important:
We do NOT use deep learning here (no PyTorch/TensorFlow).
Instead, we use a small matrix of weights to keep things easy.

The agent:
- Takes a state (3 numbers)
- Chooses an action (0 = no food, 1 = small food, 2 = medium food)
- Learns from experience using Q-learning updates
"""

import numpy as np
import random

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.95):
        """
        state_size : number of inputs (time, bowl_weight, proximity)
        action_size: number of possible actions
        lr         : learning rate
        gamma      : discount factor for future rewards
        """

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.memory = []  # Stores experiences for replay training

        # Weight matrix for linear Q function: (state_size x action_size)
        self.weights = np.random.randn(state_size, action_size) * 0.01

    def act(self, state):
        """
        Choose an action.
        10% of the time we choose randomly (exploration).
        90% of the time we choose the best predicted action.
        """
        if random.random() < 0.1:
            return random.randint(0, self.action_size - 1)

        q_values = state @ self.weights  # Predict Q-values
        return np.argmax(q_values)

    def remember(self, s, a, r, s2):
        """
        Store an experience for training later.
        Each experience is:
        (state, action, reward, next_state)
        """
        self.memory.append((s, a, r, s2))

    def train(self):
        """
        Train the agent using random samples from memory.
        This is similar to experience replay.
        """

        # Only start training after we have enough samples
        if len(self.memory) < 100:
            return

        # Draw a random mini-batch of 32 experiences
        batch = random.sample(self.memory, 32)

        for s, a, r, s2 in batch:
            # TD target:
            # reward + discounted future best Q-value
            target = r + self.gamma * np.max(s2 @ self.weights)

            # Our predicted Q-value
            q_value = s @ self.weights[:, a]

            # Error used to adjust weights
            error = target - q_value

            # Update weights for chosen action column
            self.weights[:, a] += self.lr * error * s
