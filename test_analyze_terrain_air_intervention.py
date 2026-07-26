import unittest

from analyze_terrain_air_intervention import (
    aggregate_events,
    baseline_gate_events,
    summarize,
)


def row(step, pickups=0, orbit=False):
    return {
        "time": float(step),
        "rays": [1.0] * 8,
        "body_clearance": [1.0] * 8,
        "mushroom_pickups_total": pickups,
        "body_collision": False,
        "orbit_path": 5.0 if orbit else 0.0,
        "orbit_efficiency": 0.2 if orbit else 1.0,
        "physics_wedge_seconds": 0.0,
        "trap_accumulation_seconds": 0.0,
        "fallback_active": False,
        "survival_failures": 0,
    }


class TerrainAirInterventionAnalysisTests(unittest.TestCase):
    def test_summary_normalizes_pickups_by_runtime(self):
        rows = [row(0), row(10, 1), row(20, 2)]
        self.assertAlmostEqual(summarize(rows)["pickups_per_hour"], 360.0)

    def test_baseline_events_observe_cooldown(self):
        rows = [row(index, orbit=True) for index in range(25)]
        self.assertEqual(baseline_gate_events(rows, cooldown_seconds=12.0), [0, 12, 24])

    def test_event_aggregation_is_bounded(self):
        result = aggregate_events(
            [
                {
                    "recovered_within_8_seconds": True,
                    "recovery_seconds": 3.0,
                    "pickup_within_20_seconds": True,
                    "collision_frames_within_20_seconds": 2,
                    "orbit_efficiency_after_8_seconds": 0.8,
                }
            ]
        )
        self.assertEqual(result["recovery_rate_within_8_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
