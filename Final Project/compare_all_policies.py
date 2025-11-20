"""
compare_all_policies.py
Compares multiple policy types.
"""

from policy_wrapper import PolicyWrapper

def compare(env, pets, policies):
    for mode in policies:
        wrapper = PolicyWrapper(mode)
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = wrapper.get_action(pets, obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"{mode} total reward = {total_reward}")
