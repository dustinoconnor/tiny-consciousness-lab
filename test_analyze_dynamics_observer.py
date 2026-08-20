import unittest

from analyze_dynamics_observer import roc_auc, standardized_difference


class ObserverAnalysisTests(unittest.TestCase):
    def test_auc_detects_perfect_positive_and_negative_ordering(self):
        labels = [False, False, True, True]
        self.assertEqual(roc_auc([0.0, 0.1, 0.8, 1.0], labels), 1.0)
        self.assertEqual(roc_auc([1.0, 0.8, 0.1, 0.0], labels), 0.0)

    def test_standardized_difference_preserves_direction(self):
        self.assertGreater(standardized_difference([2.0, 2.2, 2.4], [0.0, 0.2, 0.4]), 0.0)


if __name__ == "__main__":
    unittest.main()
