import unittest

from analyze_food_handoff import analyze_rows


class FoodHandoffAnalysisTests(unittest.TestCase):
    def test_reports_yaw_loss_and_intercept_coverage(self):
        rows = [
            {"trap_episode": 1, "yaw": 0.0, "food_visible": False},
            {
                "trap_episode": 1,
                "yaw": 10.0,
                "food_visible": True,
                "hidden_goal_food_latched": True,
                "shadow_continuous_intercept": True,
            },
            {
                "trap_episode": 1,
                "yaw": 30.0,
                "food_visible": False,
                "hidden_goal_food_latched": True,
                "shadow_continuous_intercept": True,
                "trap_outcome": "success",
            },
        ]
        result = analyze_rows(rows)
        self.assertEqual(result["episodes_with_food_sighting"], 1)
        self.assertEqual(result["mean_yaw_travel_degrees"], 20.0)
        self.assertEqual(result["total_visibility_losses"], 0)
        self.assertEqual(result["total_latched_frames"], 2)
        self.assertEqual(result["total_continuous_intercept_frames"], 2)


if __name__ == "__main__":
    unittest.main()
