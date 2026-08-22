import unittest

from compare_l1_mask_controls import pairwise_summary


class L1MaskControlTests(unittest.TestCase):
    def test_pairwise_summary_separates_repairs_and_regressions(self):
        left = {
            "semantic_accuracy": 0.25,
            "records": [
                {"semantic_match": True},
                {"semantic_match": False},
                {"semantic_match": False},
                {"semantic_match": False},
            ],
        }
        right = {
            "semantic_accuracy": 0.5,
            "records": [
                {"semantic_match": False},
                {"semantic_match": True},
                {"semantic_match": True},
                {"semantic_match": False},
            ],
        }
        result = pairwise_summary(left, right)
        self.assertEqual(result["left_only_correct"], 1)
        self.assertEqual(result["right_only_correct"], 2)
        self.assertEqual(result["both_wrong"], 1)
        self.assertEqual(result["accuracy_difference_right_minus_left"], 0.25)


if __name__ == "__main__":
    unittest.main()
