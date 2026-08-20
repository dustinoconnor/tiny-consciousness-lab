import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from remote_perception_lab import (
    DESCRIPTOR_FIELDS,
    descriptor_score,
    generate_deck,
    prediction_commitment,
    rank_candidates,
    sha256_json,
    validate_prediction,
)


class RemotePerceptionLabTests(unittest.TestCase):
    def prediction(self):
        return {
            "dominant_color": "blue",
            "shape": "circle",
            "count": 2,
            "arrangement": "vertical",
            "texture": "striped",
            "background": "dark",
        }

    def test_prediction_schema_is_closed_and_objective(self):
        prediction = self.prediction()
        self.assertEqual(validate_prediction(prediction), prediction)
        self.assertEqual(descriptor_score(prediction, prediction), len(DESCRIPTOR_FIELDS))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_prediction({**prediction, "free_text": "looks watery"})

    def test_commitment_changes_with_prediction_or_nonce(self):
        first = prediction_commitment("RV-test", self.prediction(), "a")
        second = prediction_commitment("RV-test", self.prediction(), "b")
        changed = self.prediction()
        changed["shape"] = "star"
        third = prediction_commitment("RV-test", changed, "a")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_candidate_ranking_uses_only_frozen_descriptors(self):
        target = {"target_id": "T-a", **self.prediction()}
        decoy = {"target_id": "T-b", **self.prediction(), "shape": "square"}
        ranked = rank_candidates(self.prediction(), [decoy, target])
        self.assertEqual(ranked[0], {"target_id": "T-a", "score": 6})

    def test_generated_deck_has_opaque_names_and_valid_commitment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "deck"
            private, public = generate_deck(root, training=4, reserved=8, seed=17)
            self.assertEqual(len(private["training_cards"]), 4)
            self.assertEqual(len(private["reserved_cards"]), 8)
            self.assertEqual(len(private["reserved_trials"]), 8)
            self.assertEqual(public["private_manifest_sha256"], sha256_json(private))
            for card in private["training_cards"] + private["reserved_cards"]:
                filename = root / card["split"] / f"{card['target_id']}.png"
                self.assertTrue(filename.exists())
                self.assertNotIn(card["shape"], filename.name)
                with Image.open(filename) as image:
                    self.assertEqual(image.size, (384, 384))
            disk_private = json.loads((root / "private_manifest.json").read_text())
            self.assertEqual(sha256_json(disk_private), public["private_manifest_sha256"])

    def test_nonempty_target_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "deck"
            root.mkdir()
            (root / "existing").write_text("do not overwrite")
            with self.assertRaisesRegex(FileExistsError, "not_empty"):
                generate_deck(root, training=4, reserved=8, seed=17)


if __name__ == "__main__":
    unittest.main()
