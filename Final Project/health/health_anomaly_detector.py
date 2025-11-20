"""
This is a simple anomaly detector.

It checks if a pet is being:
- Underfed (too little food)
- Overfed (too much food)
- Normal

Very basic version just using thresholds.
"""

class AnomalyDetector:
    def __init__(self, threshold_low=5, threshold_high=50):
        """
        threshold_low - minimum acceptable grams eaten
        threshold_high - maximum acceptable grams eaten
        """
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def detect(self, grams_eaten):
        """
        Return simple alerts based on amount of food eaten.
        """
        if grams_eaten < self.threshold_low:
            return "UNDERFEEDING ALERT"
        elif grams_eaten > self.threshold_high:
            return "OVERFEEDING ALERT"
        else:
            return "NORMAL"
