import json
import tempfile
import unittest
from pathlib import Path

from remote_perception_lab import generate_deck, prediction_commitment
from run_remote_perception_training import (
    completed_trials,
    load_records,
    load_state,
    run_one_training_trial,
)


class FakeSampler:
    def sample(self, _trial_id, count=7, temperature=0.9):
        del temperature
        prediction = {
            "dominant_color": "blue",
            "shape": "circle",
            "count": 1,
            "arrangement": "horizontal",
            "texture": "solid",
            "background": "light",
        }
        return [dict(prediction) for _ in range(count)], [json.dumps(prediction)] * count, []


class RemotePerceptionTrainingTests(unittest.TestCase):
    def test_commit_is_durable_before_reveal_and_recomputes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, _public = generate_deck(root / "deck", training=4, reserved=8, seed=29)
            manifest = root / "deck" / "private_manifest.json"
            log = root / "training.jsonl"
            state = load_state(root / "missing.json", 3)
            trial_id = private["training_trials"][0]["trial_id"]
            reveal = run_one_training_trial(
                FakeSampler(), trial_id, manifest, log, state, samples=3
            )
            records = load_records(log)
            self.assertEqual([row["event"] for row in records], ["commit", "reveal"])
            commit = records[0]
            self.assertEqual(
                prediction_commitment(trial_id, commit["prediction"], commit["nonce"]),
                commit["prediction_sha256"],
            )
            self.assertEqual(reveal["prediction_sha256"], commit["prediction_sha256"])
            self.assertEqual(completed_trials(records), {trial_id})
            self.assertNotIn("target", commit)

    def test_reveal_without_commit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without_commit"):
            completed_trials([{"event": "reveal", "trial_id": "RV-bad"}])


if __name__ == "__main__":
    unittest.main()
