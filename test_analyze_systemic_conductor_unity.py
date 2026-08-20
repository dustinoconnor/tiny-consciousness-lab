import json
import tempfile
import unittest
from pathlib import Path

from analyze_systemic_conductor_unity import (
    analyze,
    exact_mcnemar_one_sided,
    load_rows,
)
from embodied_systemic_conductor import CONTEXTS, FEATURE_NAMES, SPECIALISTS


class AnalyzeSystemicConductorUnityTests(unittest.TestCase):
    def checkpoint(self, directory):
        path = directory / "checkpoint.json"
        weights = [[0.0] * 13 for _ in SPECIALISTS]
        for index in range(len(SPECIALISTS)):
            weights[index][index] = 4.0
        path.write_text(
            json.dumps(
                {
                    "checkpoint_type": "four_context_conductor",
                    "contexts": list(CONTEXTS),
                    "specialists": list(SPECIALISTS),
                    "feature_names": list(FEATURE_NAMES),
                    "weights": weights,
                }
            )
        )
        return path

    def recording(self, directory):
        path = directory / "recording.jsonl"
        rows = []
        templates = {
            "clear_terrain": [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9],
            "familiar_hidden_goal": [0.0, 1.0, 0.6, 0.1, 0.2, 0.2, 0.3],
            "visible_target": [1.0, 0.2, 0.2, 0.0, 0.0, 0.3, 0.8],
            "genuine_wedge": [0.0, 0.1, 0.9, 1.0, 1.0, 0.1, 0.1],
        }
        for context, features in templates.items():
            for _ in range(30):
                rows.append(
                    {
                        "systemic_conductor_features": features,
                        "systemic_conductor_proxy_context": context,
                        "systemic_conductor_action_influence": 0,
                    }
                )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return path

    def test_loader_filters_unrelated_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(
                json.dumps({"other": True})
                + "\n"
                + json.dumps(
                    {
                        "systemic_conductor_features": [0.0] * 7,
                        "systemic_conductor_proxy_context": "clear_terrain",
                    }
                )
                + "\n"
            )
            self.assertEqual(len(load_rows(path)), 1)

    def test_exact_mcnemar_prefers_intact_only_wins(self):
        p_value = exact_mcnemar_one_sided(
            [True] * 10,
            [False] * 10,
        )
        self.assertLess(p_value, 0.01)

    def test_analysis_preserves_zero_action_influence(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = analyze(
                self.recording(directory),
                self.checkpoint(directory),
            )
            self.assertEqual(result["frames"], 120)
            self.assertTrue(
                result["criteria"]["all_four_contexts_have_at_least_25_frames"]
            )
            self.assertTrue(
                result["criteria"]["observer_had_zero_action_influence"]
            )


if __name__ == "__main__":
    unittest.main()
