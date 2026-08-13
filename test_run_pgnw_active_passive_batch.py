import json
import tempfile
import unittest
from pathlib import Path

from run_pgnw_active_passive_batch import (
    build_command,
    condition_order,
    inspect_recording,
    parse_seeds,
)


class PGNWBatchTests(unittest.TestCase):
    def test_seed_parser_and_counterbalanced_order(self):
        self.assertEqual(parse_seeds("143-145,150"), [143, 144, 145, 150])
        self.assertEqual(condition_order(143), ("bounded", "passive"))
        self.assertEqual(condition_order(144), ("passive", "bounded"))
        self.assertEqual(condition_order(151, "committed"), ("committed", "passive"))
        self.assertEqual(condition_order(152, "committed"), ("passive", "committed"))
        with self.assertRaisesRegex(ValueError, "duplicate_seed"):
            parse_seeds("143,143")

    def test_recording_requires_duration_identity_and_fresh_unity_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.jsonl"
            rows = [
                {"time": 100.0, "controller_seed": 143, "mushroom_pickups_total": 0, "position": [0.0, -2.856, 0.0], "pgnw_experiment": {"mode": "bounded"}},
                {"time": 1295.0, "controller_seed": 143, "mushroom_pickups_total": 8, "position": [4.0, -2.856, 5.0], "pgnw_experiment": {"mode": "bounded"}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertTrue(inspect_recording(path, "bounded", 143, 1200).complete)
            self.assertFalse(inspect_recording(path, "passive", 143, 1200).complete)
            rows[0]["mushroom_pickups_total"] = 4
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertFalse(inspect_recording(path, "bounded", 143, 1200).complete)

    def test_recording_can_enforce_protocol_start_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.jsonl"
            rows = [
                {"time": 0.0, "controller_seed": 145, "mushroom_pickups_total": 0, "position": [0.0, -2.856, 0.0], "pgnw_experiment": {"mode": "passive"}},
                {"time": 1200.0, "controller_seed": 145, "mushroom_pickups_total": 4, "position": [2.0, -2.856, 3.0], "pgnw_experiment": {"mode": "passive"}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertTrue(inspect_recording(path, "passive", 145, 1200, (0.0, -2.856, 0.0)).complete)
            self.assertFalse(inspect_recording(path, "passive", 145, 1200, (0.0, 0.0, 0.0)).complete)

    def test_command_freezes_condition_and_output(self):
        command = build_command("python3", 1200, 144, "passive", "/tmp/passive.jsonl")
        self.assertEqual(command[:2], ["python3", "embodied_unity_loop.py"])
        self.assertEqual(command[command.index("--seed") + 1], "144")
        self.assertEqual(command[command.index("--tiny-scientist-experiment-control") + 1], "passive")
        self.assertEqual(command[command.index("--shadow-log") + 1], "/tmp/passive.jsonl")


if __name__ == "__main__":
    unittest.main()
