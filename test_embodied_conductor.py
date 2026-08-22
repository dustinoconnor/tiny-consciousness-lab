import unittest
import json
import tempfile
from pathlib import Path

from embodied_conductor import PassiveEmbodiedConductor


def snapshot(**overrides):
    values = {
        "trap_course": "lwall",
        "trap_course_variant": "lwall_original",
        "trap_episode": 1,
        "trap_outcome": "running",
        "x": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "food_visible": False,
        "food_distance": 0.0,
        "blocked": False,
        "body_collision": False,
        "stuck": False,
        "body_safe_actions": 8,
        "physics_wedge_seconds": 0.0,
        "trap_accumulation_seconds": 0.0,
        "fallback_active": False,
        "hidden_goal_active": False,
        "hidden_goal_route_selected_id": "none",
        "route_index": 0,
        "route_distance": 0.0,
        "mpc_engaged": False,
    }
    values.update(overrides)
    return values


class PassiveEmbodiedConductorTests(unittest.TestCase):
    def test_context_priority_tracks_recovery_target_and_precedent(self):
        classify = PassiveEmbodiedConductor.classify_context
        self.assertEqual(classify(snapshot()), "clear_exploration")
        self.assertEqual(
            classify(snapshot(hidden_goal_route_selected_id="route-a")),
            "familiar_hidden_goal",
        )
        self.assertEqual(classify(snapshot(food_visible=True)), "visible_target")
        self.assertEqual(
            classify(snapshot(food_visible=True, physics_wedge_seconds=2.0)),
            "recovery_crisis",
        )

    def test_specialist_identification_matches_active_controller(self):
        identify = PassiveEmbodiedConductor.identify_specialist
        self.assertEqual(identify(snapshot()), "recurrent")
        self.assertEqual(identify(snapshot(mpc_engaged=True)), "predictive")
        self.assertEqual(
            identify(snapshot(mpc_engaged=True, hidden_goal_active=True)),
            "episodic",
        )
        self.assertEqual(
            identify(snapshot(hidden_goal_active=True, fallback_active=True)),
            "fallback",
        )

    def test_positive_route_progress_teaches_episodic_recommendation(self):
        observer = PassiveEmbodiedConductor(enabled=True)
        for _ in range(4):
            observer._update("familiar_hidden_goal", "episodic", 0.6)
            observer._update("familiar_hidden_goal", "recurrent", -0.2)
        observer.observe(
            snapshot(
                hidden_goal_active=True,
                hidden_goal_route_selected_id="route-a",
            )
        )
        self.assertEqual(observer.context, "familiar_hidden_goal")
        self.assertEqual(observer.recommendation, "episodic")
        self.assertGreater(observer.confidence, 0.0)
        self.assertEqual(observer.action_influence, 0)

    def test_single_specialist_coverage_cannot_support_comparison(self):
        observer = PassiveEmbodiedConductor(enabled=True)
        for _ in range(8):
            observer._update("visible_target", "predictive", 0.5)
        observer.observe(snapshot(food_visible=True, mpc_engaged=True))
        self.assertEqual(observer.recommendation, "insufficient_evidence")
        self.assertEqual(observer.confidence, 0.0)

    def test_episode_change_clears_transition_credit(self):
        observer = PassiveEmbodiedConductor(enabled=True)
        observer.observe(snapshot(hidden_goal_active=True))
        observer.observe(snapshot(trap_episode=2, body_collision=True))
        self.assertEqual(observer.updates, 0)
        self.assertEqual(observer.episode_resets, 1)

    def test_offline_checkpoint_supplies_comparative_coverage(self):
        payload = {
            "checkpoint_type": "passive_embodied_conductor",
            "q_values": {
                "familiar_hidden_goal": {
                    "recurrent": 0.1,
                    "episodic": 0.8,
                }
            },
            "visits": {
                "familiar_hidden_goal": {
                    "recurrent": 12,
                    "episodic": 12,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "conductor.json"
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            observer = PassiveEmbodiedConductor(
                enabled=True,
                checkpoint=checkpoint,
            )
        observer.observe(
            snapshot(
                hidden_goal_active=True,
                hidden_goal_route_selected_id="route-a",
            )
        )
        self.assertEqual(observer.recommendation, "episodic")
        self.assertGreater(observer.confidence, 0.5)
        self.assertEqual(observer.action_influence, 0)
        self.assertFalse(observer.learning_enabled)
        q_before = dict(observer.q["familiar_hidden_goal"])
        observer.observe(
            snapshot(
                x=1.0,
                hidden_goal_active=True,
                hidden_goal_route_selected_id="route-a",
            )
        )
        self.assertEqual(observer.updates, 0)
        self.assertEqual(observer.q["familiar_hidden_goal"], q_before)


if __name__ == "__main__":
    unittest.main()
