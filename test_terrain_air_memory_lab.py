import json
import tempfile
import unittest
from pathlib import Path

from terrain_air_memory_lab import build_packets, load_rows, run


def synthetic_rows(count=180):
    rows = []
    pickups = 0
    for index in range(count):
        visible = 30 <= index % 60 < 48
        distance = max(0.5, 10.0 - (index % 60 - 30) * 0.5) if visible else 0.0
        if index % 60 == 48:
            pickups += 1
        rows.append(
            {
                "time": index * 0.2,
                "step": index,
                "rays": [0.8, 0.7, 0.6, 0.5, 0.2, 0.5, 0.6, 0.7],
                "body_clearance": [1.0] * 8,
                "active_action": "up" if visible else "up_right",
                "food_visible": visible,
                "food_distance": distance,
                "hunger": 0.4,
                "body_collision": False,
                "physics_wedge_seconds": 0.0,
                "mushroom_pickups_total": pickups,
                "mushroom_reward_total": float(pickups),
                "survival_failures": 0,
            }
        )
    return rows


class TerrainAirMemoryLabTests(unittest.TestCase):
    def test_packet_labels_use_delayed_outcomes(self):
        packets = build_packets(synthetic_rows())
        self.assertTrue(any(packet.useful for packet in packets))
        self.assertEqual(len(packets[0].features), 6)

    def test_export_is_passive_and_fixed_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "recording.jsonl"
            output = Path(directory) / "memory.json"
            recording.write_text(
                "".join(json.dumps(row) + "\n" for row in synthetic_rows()),
                encoding="utf-8",
            )
            payload = run(recording, output, capacity=12)
            self.assertTrue(output.exists())
            self.assertFalse(payload["protocol"]["motor_control_enabled"])
            self.assertFalse(payload["protocol"]["future_outcome_present_in_packet"])
            self.assertEqual(payload["protocol"]["memory_capacity"], 12)
            self.assertEqual(len(load_rows(recording)), 180)


if __name__ == "__main__":
    unittest.main()
