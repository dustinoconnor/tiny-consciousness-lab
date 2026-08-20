import unittest

import numpy as np

from gnw_ignition_lab import (
    GNWState,
    raw_bus_probabilities,
    run_benchmark,
)


class GNWIgnitionLabTests(unittest.TestCase):
    def test_weak_bid_does_not_ignite(self):
        workspace = GNWState(4, ignition_threshold=0.6)
        winner, ignition = workspace.update([0.3, 0.3, 0.2, 0.2])
        self.assertIsNone(winner)
        self.assertFalse(ignition)

    def test_repeated_strong_bid_ignites_and_persists(self):
        workspace = GNWState(
            4,
            ignition_threshold=0.35,
            persistence_steps=2,
        )
        winner, ignition = workspace.update([0.9, 0.04, 0.03, 0.03])
        self.assertEqual(winner, 0)
        self.assertTrue(ignition)
        winner, _ = workspace.update([0.05, 0.85, 0.05, 0.05])
        self.assertEqual(winner, 0)

    def test_raw_bus_overload_is_bounded(self):
        rng = np.random.default_rng(2)
        probabilities, overload = raw_bus_probabilities(
            [0.7, 0.1, 0.1, 0.1],
            np.zeros(7),
            rng,
        )
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertGreater(overload, 0.0)

    def test_quick_benchmark_has_causal_broadcast_effect(self):
        payload = run_benchmark(
            seed_count=3,
            train_steps=5000,
            blocks=35,
            steps_per_block=8,
        )
        self.assertGreater(
            payload["summary"]["gnw_ignition"]["global_coordination_rate"],
            payload["summary"]["gnw_broadcast_lesion"][
                "global_coordination_rate"
            ],
        )
        self.assertGreater(
            payload["summary"]["gnw_ignition"]["utility_per_step"],
            payload["summary"]["gnw_broadcast_scramble"][
                "utility_per_step"
            ],
        )


if __name__ == "__main__":
    unittest.main()
