import json
import tempfile
import unittest
from pathlib import Path

from remote_perception_lab import generate_deck
from run_remote_perception_evaluation import (
    CONDITIONS,
    commit_all,
    evaluation_summary,
    load_schedule,
    prepare_public_schedule,
    reveal_all,
    validate_records,
)
from run_remote_perception_training import load_records


class FakeSampler:
    def sample(self, trial_id, count=7, temperature=0.9):
        del trial_id, temperature
        prediction = {
            "dominant_color": "blue",
            "shape": "circle",
            "count": 1,
            "arrangement": "horizontal",
            "texture": "solid",
            "background": "light",
        }
        return [dict(prediction) for _ in range(count)], [json.dumps(prediction)] * count, []


class RemotePerceptionEvaluationTests(unittest.TestCase):
    def setup_deck(self, root):
        generate_deck(root / "deck", training=4, reserved=8, seed=71)
        manifest = root / "deck" / "private_manifest.json"
        public = root / "deck" / "public_commitment.json"
        schedule_path = root / "schedule.json"
        prepare_public_schedule(manifest, public, schedule_path)
        return manifest, public, schedule_path, load_schedule(schedule_path)

    def test_public_schedule_contains_only_opaque_trial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _manifest, _public, path, schedule = self.setup_deck(root)
            text = path.read_text()
            self.assertEqual(len(schedule["trial_ids"]), 8)
            for forbidden in ("target_id", "candidate_ids", "dominant_color", "shape"):
                self.assertNotIn(forbidden, text)

    def test_all_conditions_commit_before_any_reveal_and_hide_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, public, _path, schedule = self.setup_deck(root)
            log = root / "evaluation.jsonl"
            weights = [1.0] * 7
            commits = commit_all(FakeSampler(), schedule, log, weights, "frozen", samples=7)
            self.assertEqual(len(commits), 8 * len(CONDITIONS))
            records = load_records(log)
            self.assertTrue(all(row["event"] == "commit" for row in records))
            self.assertTrue(all("target" not in row and "candidate_ids" not in row for row in records))
            reveals = reveal_all(manifest, public, schedule, log)
            self.assertEqual(len(reveals), len(commits))
            self.assertNotEqual(
                reveals[(schedule["trial_ids"][0], "sham")]["assigned_trial_id"],
                schedule["trial_ids"][0],
            )
            summary = evaluation_summary(log, schedule)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["commits"], 24)
            self.assertEqual(summary["reveals"], 24)

    def test_partial_commits_cannot_be_revealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, public, _path, schedule = self.setup_deck(root)
            log = root / "evaluation.jsonl"
            log.write_text(json.dumps({
                "event": "commit", "trial_id": schedule["trial_ids"][0],
                "condition": "active",
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "before_all_commits"):
                reveal_all(manifest, public, schedule, log)


if __name__ == "__main__":
    unittest.main()
