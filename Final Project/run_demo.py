"""
run_demo.py
Runs the full final Smart Feeder system end-to-end.
"""

from env.pet_env import PetEnv
from rl_dqn_agent import DQNAgent
from controller import Controller
from logger import Logger
from system_pipeline import SystemPipeline
from sensors import simulate_camera
from config import PETS

def main():
    print("Starting Final AI Smart Feeder Demo...")

    env = PetEnv(PETS)
    agent = DQNAgent(3,3)

    controller = Controller(
        cv_model_path="perception/cv_model.pkl",
        eating_model_path="models/eating_model.pkl",
        rl_agent=agent
    )

    logger = Logger()
    pipeline = SystemPipeline(env, controller, logger)

    obs = env.reset()
    done = False

    while not done:
        img = simulate_camera("Fluffy")
        obs, done = pipeline.run_step(obs, img, 1)

    print("Demo finished. Log saved in data/final_log.csv")

if __name__ == "__main__":
    main()
