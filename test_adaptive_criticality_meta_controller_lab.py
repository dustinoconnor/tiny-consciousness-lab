import unittest

import numpy as np

from adaptive_criticality_meta_controller_lab import (
    AdaptiveGainController,
    adaptation_latency,
    delayed_cue_accuracy_at_noise,
)
from network_criticality_lab import make_reservoir


class AdaptiveCriticalityMetaControllerTests(unittest.TestCase):
    def test_noisy_delayed_accuracy_is_bounded(self):
        weights, inputs = make_reservoir(3, units=16)
        accuracy = delayed_cue_accuracy_at_noise(
            weights, inputs, 1.0, 4, noise=0.3, delay=8
        )
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_controller_reexplores_after_reward_drop(self):
        controller = AdaptiveGainController(3, np.random.default_rng(5))
        controller.relearn_queue["high_noise"].clear()
        controller.history["high_noise"][0].extend([0.9, 0.9, 0.9, 0.9])
        controller.observe(0, 0.2, "high_noise")
        self.assertEqual(controller.change_detections, 1)
        self.assertEqual(len(controller.relearn_queue["high_noise"]), 3)

    def test_latency_detects_sustained_low_regret(self):
        trace = [{"regret": 0.5} for _ in range(8)]
        trace += [{"regret": 0.2}, {"regret": 0.1}]
        trace += [{"regret": 0.02} for _ in range(6)]
        self.assertEqual(adaptation_latency(trace, phase_length=8, hold=5), [2])


if __name__ == "__main__":
    unittest.main()
