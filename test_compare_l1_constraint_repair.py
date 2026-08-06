import unittest

from compare_l1_constraint_repair import paired_summary


class L1ConstraintRepairTests(unittest.TestCase):
    def test_paired_summary_counts_repairs_and_regressions(self):
        greedy = {
            "semantic_accuracy": 0.5,
            "records": [{"semantic_match": True}, {"semantic_match": False}],
        }
        constrained = {
            "semantic_accuracy": 0.5,
            "records": [{"semantic_match": False}, {"semantic_match": True}],
        }
        result = paired_summary(greedy, constrained)
        self.assertEqual(result["greedy_only_correct"], 1)
        self.assertEqual(result["constrained_only_correct"], 1)
        self.assertEqual(result["accuracy_difference_constrained_minus_greedy"], 0.0)


if __name__ == "__main__":
    unittest.main()
