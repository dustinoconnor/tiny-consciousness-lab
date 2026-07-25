import unittest

import numpy as np

from four_context_conductor_lab import (
    CONTEXTS,
    FEATURE_NAMES,
    OPTIMAL_SPECIALIST,
    SPECIALISTS,
    ContextualConductor,
    expected_utility,
    expand_features,
    gate_entropy,
    run_benchmark,
    sample_features,
    softmax,
    train_conductor,
)


class FourContextConductorLabTests(unittest.TestCase):
    def test_declared_specialists_are_expected_utility_optima(self):
        for context in range(len(CONTEXTS)):
            utilities = [
                expected_utility(context, specialist)
                for specialist in range(len(SPECIALISTS))
            ]
            self.assertEqual(int(np.argmax(utilities)), int(OPTIMAL_SPECIALIST[context]))

    def test_features_are_bounded_and_do_not_include_context_one_hot(self):
        rng = np.random.default_rng(8)
        for context in range(len(CONTEXTS)):
            features = sample_features(context, rng, domain_shift=0.04)
            self.assertEqual(features.shape, (len(FEATURE_NAMES),))
            self.assertTrue(np.all(features >= 0.0))
            self.assertTrue(np.all(features <= 1.0))
            self.assertLess(len(features), len(expand_features(features)))

    def test_probability_and_entropy_are_well_formed(self):
        probabilities = softmax([1.0, 0.5, -0.5, -1.0])
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertGreaterEqual(gate_entropy(probabilities), 0.0)
        self.assertLessEqual(gate_entropy(probabilities), 2.0)

    def test_targeted_lesion_masks_selected_specialist(self):
        conductor = ContextualConductor(2)
        conductor.weights[0] = 2.0
        features = np.full(len(FEATURE_NAMES), 0.5)
        unrestricted = conductor.choose(features)
        restricted = conductor.choose(features, allowed=[1, 2, 3])
        self.assertEqual(unrestricted, 0)
        self.assertNotEqual(restricted, 0)

    def test_reward_training_learns_all_four_contexts(self):
        conductor = train_conductor(91, steps=12000)
        rng = np.random.default_rng(92)
        correct = []
        for context in range(len(CONTEXTS)):
            selections = [
                conductor.choose(sample_features(context, rng))
                for _ in range(300)
            ]
            correct.append(selections.count(int(OPTIMAL_SPECIALIST[context])) / 300)
        self.assertTrue(all(rate >= 0.80 for rate in correct), correct)

    def test_quick_benchmark_beats_fixed_policies(self):
        payload = run_benchmark(
            seed_count=3,
            train_steps=6000,
            blocks=40,
            steps_per_block=8,
        )
        self.assertTrue(payload["criteria"]["learned_beats_every_static_utility"])
        self.assertGreater(
            payload["summary"]["learned_conductor"]["optimal_routing_rate"],
            0.80,
        )


if __name__ == "__main__":
    unittest.main()
