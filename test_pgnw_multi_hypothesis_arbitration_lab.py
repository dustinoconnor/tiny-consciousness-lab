import math
import unittest

from pgnw_multi_hypothesis_arbitration_lab import (
    CandidateRoute,
    arbitrate,
    candidate_score,
    counterbalanced_matrix,
    pool_with_posterior,
)


class PGNWMultiHypothesisArbitrationTests(unittest.TestCase):
    def test_selector_is_invariant_to_candidate_presentation_order(self):
        pool = pool_with_posterior((0.70, 0.05, 0.15, 0.10))
        yellow = CandidateRoute("yellow", 36.0, 0.60)
        blue = CandidateRoute("blue", 12.0, 0.60)

        forward = arbitrate(pool, [yellow, blue])
        reverse = arbitrate(pool, [blue, yellow])

        self.assertEqual(forward["selected_feature"], "yellow")
        self.assertEqual(reverse["selected_feature"], "yellow")
        self.assertEqual(forward["records"], reverse["records"])

    def test_color_swapped_posterior_swaps_decision(self):
        yellow_pool = pool_with_posterior((0.70, 0.05, 0.15, 0.10))
        blue_pool = pool_with_posterior((0.05, 0.70, 0.15, 0.10))
        candidates = [
            CandidateRoute("yellow", 20.0, 0.60),
            CandidateRoute("blue", 20.0, 0.60),
        ]

        self.assertEqual(
            arbitrate(yellow_pool, candidates)["selected_feature"], "yellow"
        )
        self.assertEqual(
            arbitrate(blue_pool, candidates)["selected_feature"], "blue"
        )

    def test_safety_gate_cannot_be_outvoted_by_posterior(self):
        pool = pool_with_posterior((0.99, 0.0, 0.01, 0.0))
        result = arbitrate(
            pool,
            [
                CandidateRoute("yellow", 5.0, 1.0, safety_allowed=False),
                CandidateRoute("blue", 30.0, 0.5, safety_allowed=True),
            ],
        )

        self.assertEqual(result["selected_feature"], "blue")
        denied = next(
            record for record in result["records"]
            if record["candidate"]["feature"] == "yellow"
        )
        self.assertFalse(denied["eligible"])
        self.assertTrue(math.isinf(denied["score"]))

    def test_score_exposes_pragmatic_epistemic_and_route_terms(self):
        pool = pool_with_posterior((0.70, 0.05, 0.15, 0.10))
        metrics = candidate_score(
            pool, CandidateRoute("yellow", 36.0, 0.60)
        )

        self.assertGreater(metrics["pragmatic_value"], 0.0)
        self.assertGreaterEqual(metrics["epistemic_value"], 0.0)
        self.assertGreater(metrics["route_cost"], 0.0)
        self.assertAlmostEqual(
            metrics["score"],
            metrics["pragmatic_value"]
            + metrics["epistemic_value"]
            - metrics["route_cost"],
        )

    def test_deadline_can_change_choice_without_changing_posterior(self):
        pool = pool_with_posterior((0.9763185, 0.0000204, 0.0207773, 0.0028838))
        candidates = [
            CandidateRoute("yellow", 50.0, 0.60),
            CandidateRoute("blue", 12.0, 0.60),
        ]

        long_deadline = arbitrate(
            pool, candidates, deadline_remaining=240.0
        )
        short_deadline = arbitrate(
            pool, candidates, deadline_remaining=10.0
        )

        self.assertEqual(long_deadline["selected_feature"], "yellow")
        self.assertEqual(short_deadline["selected_feature"], "blue")

    def test_registered_matrix_balances_identity_distance_and_order(self):
        cases = counterbalanced_matrix()

        self.assertEqual(len(cases), 8)
        self.assertTrue(all(case["passed"] for case in cases))
        self.assertEqual(
            sum(case["dominant_feature"] == "yellow" for case in cases), 4
        )
        self.assertEqual(
            sum(case["distance_order"] == "dominant_far" for case in cases), 4
        )


if __name__ == "__main__":
    unittest.main()
