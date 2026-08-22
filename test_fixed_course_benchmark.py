import json
import tempfile
import unittest
from pathlib import Path

from run_fixed_course_benchmark import completed_episodes


class FixedCourseBenchmarkTests(unittest.TestCase):
    def test_counts_only_running_to_terminal_transitions(self):
        rows = [
            {"trap_episode": 1, "trap_course": "utrap", "trap_outcome": "success"},
            {"trap_episode": 2, "trap_course": "ctrap", "trap_outcome": "running"},
            {"trap_episode": 2, "trap_course": "ctrap", "trap_outcome": "running"},
            {"trap_episode": 2, "trap_course": "ctrap", "trap_outcome": "timeout"},
            {"trap_episode": 3, "trap_course": "lwall", "trap_outcome": "running"},
            {"trap_episode": 3, "trap_course": "lwall", "trap_outcome": "success"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(
                completed_episodes(path),
                [
                    {"episode": 2, "course": "ctrap", "outcome": "timeout"},
                    {"episode": 3, "course": "lwall", "outcome": "success"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
