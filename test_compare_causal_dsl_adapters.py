import unittest

from compare_causal_dsl_adapters import paired_summary


class CompareCausalDslAdaptersTests(unittest.TestCase):
    def test_paired_summary_counts_discordance_and_token_reduction(self):
        json_result = {
            "semantic_accuracy": 0.5,
            "mean_total_tokens": 200,
            "records": [
                {"semantic_match": True},
                {"semantic_match": False},
            ],
        }
        l1_result = {
            "semantic_accuracy": 0.5,
            "mean_total_tokens": 120,
            "records": [
                {"semantic_match": False},
                {"semantic_match": True},
            ],
        }
        result = paired_summary(json_result, l1_result)
        self.assertEqual(result["json_only_correct"], 1)
        self.assertEqual(result["l1_only_correct"], 1)
        self.assertEqual(result["l1_visible_token_reduction_percent"], 40.0)


if __name__ == "__main__":
    unittest.main()
