import json
import tempfile
import unittest
from pathlib import Path

from analyze_pgnw_reserved_side_blocked import (
    mcnemar_exact_two_sided,
    summarize_trial,
)


class ReservedSideBlockedAnalysisTests(unittest.TestCase):
    def test_mcnemar_exact_extremes_and_tie(self):
        self.assertEqual(mcnemar_exact_two_sided(0, 0), 1.0)
        self.assertEqual(mcnemar_exact_two_sided(2, 2), 1.0)
        self.assertEqual(mcnemar_exact_two_sided(4, 0), 0.125)

    def test_rejects_nonzero_frame_zero_pickup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contaminated.jsonl"
            path.write_text(json.dumps({
                "time": 0.0,
                "red_mushroom_pickups_total": 1,
                "blue_mushroom_pickups_total": 0,
                "yellow_flower_pickups_total": 0,
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "contaminated frame-zero"):
                summarize_trial(path, "x", "blue_left", "active", 1)

    def test_blue_increment_after_red_is_primary_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.jsonl"
            rows = [
                {"time": 0.0, "red_mushroom_pickups_total": 0, "blue_mushroom_pickups_total": 0, "yellow_flower_pickups_total": 0},
                {"time": 1.0, "red_mushroom_pickups_total": 1, "blue_mushroom_pickups_total": 0, "yellow_flower_pickups_total": 0},
                {"time": 2.0, "red_mushroom_pickups_total": 1, "blue_mushroom_pickups_total": 1, "yellow_flower_pickups_total": 0, "pgnw_experiment": {"arbitration_action_influence": 3}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = summarize_trial(path, "x", "blue_left", "active", 1)
            self.assertTrue(result["blue_first"])
            self.assertEqual(result["blue_action_influence"], 3)
            self.assertEqual(result["prechoice_action_influence"], 0)


if __name__ == "__main__":
    unittest.main()
