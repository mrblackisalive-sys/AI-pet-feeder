"""
system_pipeline.py
Runs the full feeding cycle:
1. Sensor input
2. Perception (detect pet)
3. Predict eating
4. RL-based action selection
5. Apply to environment
6. Log results
7. Detect anomalies
"""

from controller import Controller
from health_anomaly_detector import AnomalyDetector

class SystemPipeline:
    def __init__(self, env, controller, logger):
        self.env = env
        self.controller = controller
        self.logger = logger
        self.detector = AnomalyDetector()

    def run_step(self, obs, img, pet_id):
        # 1. Detect which pet is seen
        detected_pet = self.controller.detect_pet(img)

        # 2. Predict eating behavior
        pred_finish = self.controller.predict_eating(obs["bowl_weight"], pet_id)

        # 3. RL decides how much food to give
        grams = self.controller.choose_action(obs)

        # 4. Take environment action
        action = {"dispense": {pet_id: grams}, "lock": False}
        obs2, reward, done, info = self.env.step(action)

        # 5. Anomaly check
        anomaly = self.detector.detect(info["events"]["ate"])

        # 6. Log everything
        self.logger.log_step(obs2, detected_pet, grams, reward, anomaly)

        return obs2, done
