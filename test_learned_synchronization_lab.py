import math
import unittest

import torch

from learned_synchronization_lab import OscillatoryBus, train_phases


class LearnedSynchronizationLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_phases(seed=101, steps=600, bottleneck=True)

    def test_training_discovers_high_utility_phase_policy(self):
        self.assertGreater(self.result.metrics["learned"]["mean_utility"], 0.98)
        self.assertLess(self.result.metrics["initial"]["mean_utility"], 0.15)

    def test_scramble_collapses_and_restore_recovers(self):
        self.assertLess(self.result.metrics["scrambled"]["mean_utility"], 0.15)
        self.assertAlmostEqual(
            self.result.metrics["restored"]["mean_utility"],
            self.result.metrics["learned"]["mean_utility"],
            places=6,
        )

    def test_global_rotation_preserves_relative_timing(self):
        self.assertAlmostEqual(
            self.result.metrics["global_rotation"]["mean_utility"],
            self.result.metrics["learned"]["mean_utility"],
            places=5,
        )

    def test_learned_contexts_use_distinct_phase_structures(self):
        phase = self.result.metrics["phase_structure"]
        self.assertGreater(phase["binding_order_parameter"], 0.98)
        self.assertGreater(phase["multiplex_group_a_order"], 0.98)
        self.assertGreater(phase["multiplex_group_b_order"], 0.98)
        self.assertGreater(phase["multiplex_group_separation_fraction_of_pi"], 0.70)

    def test_bus_bottleneck_penalizes_overlapping_packets(self):
        bus = OscillatoryBus()
        aligned = torch.zeros(6)
        separated = torch.tensor([0.0, 0.0, 0.0, math.pi, math.pi, math.pi])
        aligned_score, _, aligned_collision = bus.multiplex_score(aligned)
        separated_score, _, separated_collision = bus.multiplex_score(separated)
        self.assertGreater(float(aligned_collision), 0.95)
        self.assertLess(float(separated_collision), 0.01)
        self.assertGreater(float(separated_score), float(aligned_score) + 0.90)


if __name__ == "__main__":
    unittest.main()
