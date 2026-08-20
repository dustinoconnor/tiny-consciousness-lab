import unittest

from train_embodied_conductor_offline import evaluate_pairs, fit_context_values


def pair(seed, recurrent, episodic):
    return {
        "key": {"seed": seed, "variant": "lwall_original", "occurrence": 0},
        "recurrent": {"utility": recurrent},
        "episodic": {"utility": episodic},
    }


class OfflineConductorTrainingTests(unittest.TestCase):
    def test_fit_excludes_held_out_seed(self):
        pairs = [pair(11, -1.0, 1.0), pair(23, 0.2, 0.6)]
        values, visits = fit_context_values(pairs, excluded_seed=11)
        self.assertEqual(values, {"recurrent": 0.2, "episodic": 0.6})
        self.assertEqual(visits, {"recurrent": 1, "episodic": 1})

    def test_evaluation_reports_utility_gain_without_forcing_winner(self):
        pairs = [pair(11, 0.1, 0.8), pair(11, 0.7, 0.2)]
        result = evaluate_pairs(
            pairs,
            {"recurrent": 0.0, "episodic": 0.5},
            seed=11,
        )
        self.assertEqual(result["selected_specialist"], "episodic")
        self.assertEqual(result["selection_accuracy"], 0.5)
        self.assertAlmostEqual(result["utility_gain_over_recurrent"], 0.1)


if __name__ == "__main__":
    unittest.main()
