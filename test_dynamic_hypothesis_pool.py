import unittest

from dynamic_hypothesis_pool import (
    DynamicHypothesisPool,
    compile_l1_hypothesis,
)


DISCOVERY = {
    "feature_outcomes": {
        "red": {
            "isolated_episodes": 2,
            "mean_pressure_delta": 0.34,
            "positive_pressure_fraction": 1.0,
            "mean_positive_delay_seconds": 10.0,
        },
        "blue": {
            "isolated_episodes": 6,
            "mean_pressure_delta": 0.0,
            "positive_pressure_fraction": 0.0,
            "mean_positive_delay_seconds": None,
        },
    }
}


class DynamicHypothesisPoolTests(unittest.TestCase):
    def proposal(self):
        return compile_l1_hypothesis(
            "L1 c red k blue e + t 10 q 0.8", DISCOVERY, 8
        )

    def test_compiler_admits_zero_semantic_authority_candidate(self):
        candidate = self.proposal()
        self.assertEqual(candidate.target_cause, "red")
        self.assertEqual(candidate.likelihood_rise, (0.92, 0.08, 0.05))
        self.assertEqual(candidate.admitted_after_observation, 8)

    def test_discovery_observation_cannot_verify_new_candidate(self):
        pool = DynamicHypothesisPool()
        pool.admit(self.proposal())
        before = pool.posterior.copy()
        accepted = pool.update(0, True, observation_index=8)
        self.assertFalse(accepted)
        self.assertTrue((pool.posterior == before).all())
        self.assertEqual(pool.held_out_updates, 0)
        self.assertEqual(pool.pre_admission_evidence_rejections, 1)

    def test_held_out_red_positive_and_blue_negative_verify_candidate(self):
        pool = DynamicHypothesisPool(candidate_prior=0.20)
        pool.admit(self.proposal())
        observation = 9
        for _ in range(3):
            pool.update(0, True, observation)
            observation += 1
            pool.update(1, False, observation)
            observation += 1
        self.assertEqual(pool.map_hypothesis, "dsl:red_causes_probe_rise@10s")
        self.assertGreater(pool.map_confidence, 0.95)
        self.assertEqual(pool.held_out_updates, 6)

    def test_dynamic_candidate_changes_epistemic_experiment_selection(self):
        pool = DynamicHypothesisPool()
        baseline_action, _ = pool.select_experiment()
        pool.admit(self.proposal())
        proposed_action, scores = pool.select_experiment()
        self.assertIn(baseline_action, range(3))
        self.assertIn(proposed_action, range(3))
        self.assertGreater(max(scores), 0.0)

    def test_duplicate_candidate_is_rejected(self):
        pool = DynamicHypothesisPool()
        pool.admit(self.proposal())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pool.admit(self.proposal())


if __name__ == "__main__":
    unittest.main()
