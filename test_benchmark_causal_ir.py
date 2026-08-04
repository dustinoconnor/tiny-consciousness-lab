import unittest

from benchmark_causal_ir import summarize_trials


class CausalIRBenchmarkTests(unittest.TestCase):
    def test_summary_uses_retry_adjusted_cost_per_accepted_result(self):
        trials = [
            {
                "model": "m",
                "contract": "bound_schema",
                "accepted": True,
                "metrics": {"total_tokens": 100, "elapsed_seconds": 2.0},
            },
            {
                "model": "m",
                "contract": "bound_schema",
                "accepted": False,
                "metrics": {"total_tokens": 100, "elapsed_seconds": 2.0},
            },
            {
                "model": "m",
                "contract": "causal_ir",
                "accepted": True,
                "metrics": {"total_tokens": 50, "elapsed_seconds": 1.0},
            },
            {
                "model": "m",
                "contract": "causal_ir",
                "accepted": True,
                "metrics": {"total_tokens": 50, "elapsed_seconds": 1.0},
            },
        ]
        result = summarize_trials(trials)["m"]
        self.assertEqual(
            result["contracts"]["bound_schema"][
                "retry_adjusted_tokens_per_accepted"
            ],
            200,
        )
        self.assertEqual(
            result["contracts"]["causal_ir"][
                "retry_adjusted_tokens_per_accepted"
            ],
            50,
        )
        self.assertEqual(
            result["causal_ir_retry_adjusted_token_savings_percent"], 75.0
        )


if __name__ == "__main__":
    unittest.main()
