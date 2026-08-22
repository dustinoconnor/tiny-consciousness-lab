import unittest

from make_conductor_lesion_checkpoint import force_recurrent


class ConductorLesionCheckpointTests(unittest.TestCase):
    def test_swaps_only_target_context_specialists(self):
        payload = {
            "q_values": {
                "familiar_hidden_goal": {
                    "recurrent": 0.1,
                    "episodic": 0.8,
                    "predictive": 0.0,
                    "fallback": 0.0,
                },
                "visible_target": {
                    "recurrent": 0.2,
                    "episodic": 0.3,
                },
            }
        }
        lesion = force_recurrent(payload)
        self.assertEqual(
            lesion["q_values"]["familiar_hidden_goal"]["recurrent"],
            0.8,
        )
        self.assertEqual(
            lesion["q_values"]["familiar_hidden_goal"]["episodic"],
            0.1,
        )
        self.assertEqual(
            lesion["q_values"]["visible_target"],
            payload["q_values"]["visible_target"],
        )


if __name__ == "__main__":
    unittest.main()
