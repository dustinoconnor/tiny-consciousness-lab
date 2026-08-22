import unittest

import numpy as np

from bunge_systemic_emergence_lab import (
    fixed_derangement,
    paired_summary,
    run_benchmark,
    select_system_specialist,
)
from four_context_conductor_lab import (
    FEATURE_NAMES,
    SPECIALISTS,
    ContextualConductor,
)


class BungeSystemicEmergenceLabTests(unittest.TestCase):
    def test_derangement_preserves_parts_without_identity_connections(self):
        permutation = fixed_derangement(4, np.random.default_rng(7))
        self.assertEqual(sorted(permutation.tolist()), [0, 1, 2, 3])
        self.assertTrue(np.all(permutation != np.arange(4)))

    def test_output_scramble_changes_selected_component(self):
        conductor = ContextualConductor(3)
        conductor.weights[0] = 2.0
        features = np.full(len(FEATURE_NAMES), 0.5)
        permutation = np.array([1, 2, 3, 0])
        selected, _ = select_system_specialist(
            "output_scrambled",
            conductor,
            features,
            flat_step=0,
            input_permutation=np.arange(len(FEATURE_NAMES)),
            output_permutation=permutation,
        )
        self.assertEqual(selected, 1)

    def test_static_control_uses_all_specialists_equally(self):
        conductor = ContextualConductor(5)
        selected = []
        for step in range(40):
            specialist, _ = select_system_specialist(
                "static_balanced",
                conductor,
                np.zeros(len(FEATURE_NAMES)),
                flat_step=step,
                input_permutation=np.arange(len(FEATURE_NAMES)),
                output_permutation=np.arange(len(SPECIALISTS)),
            )
            selected.append(specialist)
        self.assertEqual([selected.count(index) for index in range(4)], [10] * 4)

    def test_paired_summary_reports_all_positive_seeds(self):
        result = paired_summary([0.2, 0.3, 0.1, 0.4])
        self.assertEqual(result["positive_seeds"], 4)
        self.assertGreater(result["ci95_low"], 0.0)

    def test_quick_benchmark_supports_organizational_dependence(self):
        payload = run_benchmark(
            seed_count=3,
            train_steps=7000,
            blocks=45,
            steps_per_block=8,
        )
        criteria = payload["criteria"]
        self.assertTrue(criteria["whole_beats_every_isolated_component"])
        self.assertTrue(criteria["whole_beats_equal_component_static_control"])
        self.assertTrue(criteria["whole_beats_all_connection_scrambles"])
        self.assertTrue(criteria["each_targeted_lesion_largest_in_matching_context"])


if __name__ == "__main__":
    unittest.main()
