import unittest

import numpy as np

from pgnw_hypothesis_selection_lab import (
    HYPOTHESES,
    LIKELIHOOD_RISE,
    action_scores,
    expected_information_gain,
    posterior_after,
    run_benchmark,
    run_episode,
)


class PredictiveGNWHypothesisSelectionTests(unittest.TestCase):
    def test_initial_red_and_blue_information_are_symmetric(self):
        prior = np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))
        self.assertAlmostEqual(
            expected_information_gain(prior, 0),
            expected_information_gain(prior, 1),
        )

    def test_red_rise_increases_red_compatible_models(self):
        prior = np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))
        posterior = posterior_after(prior, action=0, rise=True)
        self.assertGreater(posterior[0], posterior[1])
        self.assertGreater(posterior[2], posterior[4])
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)

    def test_selector_signature_has_no_true_hypothesis_argument(self):
        prior = np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))
        scores = action_scores("pgnw_efe", prior)
        self.assertEqual(scores.shape, (LIKELIHOOD_RISE.shape[1],))

    def test_scrambled_broadcast_executes_different_experiment(self):
        episode = run_episode(
            "pgnw_broadcast_scramble", true_hypothesis=0, seed=44, budget=6
        )
        self.assertTrue(
            all(
                row["broadcast_execution_agreement"] == 0.0
                for row in episode["rows"]
            )
        )

    def test_quick_benchmark_is_counterbalanced(self):
        payload = run_benchmark(seed_count=2, budget=5, seed_start=7)
        self.assertEqual(payload["seed_start"], 7)
        self.assertEqual(payload["episode_count_per_condition"], 10)
        self.assertEqual(set(payload["criteria"]), {
            "pgnw_beats_pragmatic_only",
            "pgnw_beats_random",
            "bound_broadcast_beats_scramble",
            "pgnw_information_noninferior_to_epistemic",
            "pgnw_pragmatic_value_beats_epistemic",
        })


if __name__ == "__main__":
    unittest.main()
