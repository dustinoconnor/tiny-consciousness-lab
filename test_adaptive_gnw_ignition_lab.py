import unittest

from adaptive_gnw_ignition_lab import (
    AdaptiveIgnitionGovernor,
    GovernorConfig,
    run_benchmark,
)


class AdaptiveGNWIgnitionTests(unittest.TestCase):
    def config(self, streak=2):
        return GovernorConfig(0.60, 0.05, streak, 0.0)

    def test_single_challenger_is_held(self):
        governor = AdaptiveIgnitionGovernor(4, self.config(streak=2))
        governor.select([0.9, 0.04, 0.03, 0.03])
        winner, ignition = governor.select([0.03, 0.9, 0.04, 0.03])
        self.assertEqual(winner, 0)
        self.assertFalse(ignition)

    def test_sustained_challenger_releases_broadcast(self):
        governor = AdaptiveIgnitionGovernor(4, self.config(streak=2))
        governor.select([0.9, 0.04, 0.03, 0.03])
        governor.select([0.03, 0.9, 0.04, 0.03])
        winner, ignition = governor.select([0.03, 0.9, 0.04, 0.03])
        self.assertEqual(winner, 1)
        self.assertTrue(ignition)

    def test_failure_feedback_can_release_before_streak(self):
        governor = AdaptiveIgnitionGovernor(4, self.config(streak=3))
        governor.select([0.9, 0.04, 0.03, 0.03])
        governor.observe_utility(-1.0)
        winner, _ = governor.select([0.03, 0.9, 0.04, 0.03])
        self.assertEqual(winner, 1)

    def test_quick_benchmark_beats_fixed_gnw(self):
        payload = run_benchmark(
            seed_count=2,
            train_steps=4000,
            train_blocks=25,
            eval_blocks=30,
            steps_per_block=8,
        )
        self.assertGreater(
            payload["summary"]["adaptive_gnw"]["utility_per_step"],
            payload["summary"]["fixed_gnw"]["utility_per_step"],
        )


if __name__ == "__main__":
    unittest.main()
