import unittest

from remote_perception_agent import (
    aggregate_impressions,
    impression_accuracy,
    intuition_prompt,
    parse_impression,
    update_slot_weights,
)


class RemotePerceptionAgentTests(unittest.TestCase):
    def impression(self, color="blue", shape="circle"):
        return {
            "dominant_color": color,
            "shape": shape,
            "count": 1,
            "arrangement": "horizontal",
            "texture": "solid",
            "background": "light",
        }

    def test_prompt_contains_identifier_but_no_target_or_candidate_data(self):
        prompt = intuition_prompt("RV-opaque")
        self.assertIn("RV-opaque", prompt)
        self.assertNotIn("target.png", prompt)
        self.assertNotIn("candidate", prompt.lower())

    def test_strict_impression_parser(self):
        raw = "first impression: " + __import__("json").dumps(self.impression())
        self.assertEqual(parse_impression(raw), self.impression())

    def test_workspace_consensus_uses_weighted_recurring_elements(self):
        impressions = [
            self.impression("red", "star"),
            self.impression("blue", "circle"),
            self.impression("blue", "circle"),
        ]
        result, support = aggregate_impressions(impressions)
        self.assertEqual(result["dominant_color"], "blue")
        self.assertEqual(result["shape"], "circle")
        self.assertAlmostEqual(support["shape"], 2 / 3)

    def test_valence_update_rewards_more_accurate_impression_slot(self):
        target = self.impression("yellow", "triangle")
        impressions = [target, self.impression("red", "star")]
        weights = update_slot_weights([1.0, 1.0], impressions, target)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertEqual(impression_accuracy(target, target), 1.0)


if __name__ == "__main__":
    unittest.main()
