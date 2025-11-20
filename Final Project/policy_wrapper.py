"""
policy_wrapper.py
Allows system to switch between:
- baseline
- rl
- hybrid (predictor + rl)
"""

from baseline_policy import baseline_policy

class PolicyWrapper:
    def __init__(self, mode, rl_agent=None):
        self.mode = mode
        self.rl_agent = rl_agent

    def get_action(self, pets, obs):
        if self.mode == "baseline":
            return baseline_policy(pets, obs)

        elif self.mode == "rl":
            # RL chooses simple amount
            state = [
                obs["time"],
                obs["bowl_weight"],
                int(obs["proximity"])
            ]
            idx = self.rl_agent.act(state)
            grams = [0,10,20][idx]
            return {"dispense": {1: grams, 2: grams}, "lock": False}

        elif self.mode == "hybrid":
            # Intelligent combination
            if obs["bowl_weight"] < 5:
                return baseline_policy(pets, obs)
            else:
                return {"dispense": {1: 10}, "lock": False}
