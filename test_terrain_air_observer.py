import json
import tempfile
import unittest
from pathlib import Path

from terrain_air_observer import PassiveTerrainAirObserver


class TerrainAirObserverTests(unittest.TestCase):
    def checkpoint(self, path):
        payload = {
            "format": "terrain_air_memory_v1",
            "protocol": {"retrieval_threshold": 0.12},
            "memory": [
                {
                    "rays": [0.8] * 8,
                    "body_clearance": [1.0] * 8,
                    "hunger": 0.4,
                    "action": "up_right",
                    "utility": 1.0,
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_retrieval_is_resonant_but_has_no_action_influence(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "memory.json"
            self.checkpoint(checkpoint)
            observer = PassiveTerrainAirObserver(checkpoint)
            observer.update(
                {
                    "directional_rays": [0.8] * 8,
                    "directional_body_clearance": [1.0] * 8,
                },
                hunger=0.4,
                active_direction="up",
            )
            self.assertTrue(observer.resonance)
            self.assertTrue(observer.agreement)
            self.assertEqual(observer.recalled_action, "up_right")
            self.assertEqual(observer.action_influence, 0.0)


if __name__ == "__main__":
    unittest.main()
