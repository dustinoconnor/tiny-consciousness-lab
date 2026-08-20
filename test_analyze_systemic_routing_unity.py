import json
import tempfile
import unittest
from pathlib import Path

from analyze_systemic_routing_unity import compare, summarize


class SystemicRoutingAnalysisTests(unittest.TestCase):
    def recording(self, rows):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def rows(self, active=False):
        rows = []
        for index in range(20):
            rows.append(
                {
                    "time": index,
                    "mushroom_pickups_total": index // 4,
                    "survival_failures": 0,
                    "stuck_events": 0,
                    "unstuck_respawns": 0,
                    "body_collision": index == 3,
                    "shadow_mpc_engaged": index % 2 == 0,
                    "fallback_active": False,
                    "systemic_router_handoffs": 2 if active else 0,
                    "systemic_router_chatter_events": 0,
                    "systemic_router_safety_overrides": 1 if active else 0,
                    "systemic_router_influence_frames": 4 if active else 0,
                }
            )
        return rows

    def test_summarize_counts_rates_and_router_effects(self):
        metrics = summarize(self.rows(active=True))
        self.assertEqual(metrics["frames"], 20)
        self.assertEqual(metrics["router_influence_frames"], 4)
        self.assertGreater(metrics["pickups_per_hour"], 0.0)

    def test_equal_active_run_passes_bounded_criteria(self):
        baseline = self.recording(self.rows(active=False))
        active = self.recording(self.rows(active=True))
        metrics = compare(baseline, active)
        self.assertTrue(metrics["bounded_routing_supported"])


if __name__ == "__main__":
    unittest.main()
