import json
import tempfile
import unittest
from pathlib import Path

from embodied_systemic_conductor import (
    BoundedExecutiveRouter,
    CONTEXTS,
    FEATURE_NAMES,
    SPECIALISTS,
    PassiveAdaptiveIgnitionGate,
    PassiveSystemicConductor,
    live_features,
    proxy_context,
)


class PassiveSystemicConductorTests(unittest.TestCase):
    def checkpoint(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        payload = {
            "checkpoint_type": "four_context_conductor",
            "contexts": list(CONTEXTS),
            "specialists": list(SPECIALISTS),
            "feature_names": list(FEATURE_NAMES),
            "weights": [[0.0] * 13 for _ in SPECIALISTS],
        }
        payload["weights"][2][-1] = 1.0
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_live_features_are_bounded(self):
        values = live_features(
            {
                "food_visible": True,
                "art_match": 0.8,
                "art_resonance": True,
                "trap_pressure": 0.4,
                "blocked": True,
                "physics_wedge_seconds": 0.5,
                "art_novel": False,
                "directional_body_clearance": [0.2] * 8,
            }
        )
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_proxy_labels_four_grounded_contexts(self):
        base = [0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.9]
        self.assertEqual(proxy_context({}, base), "clear_terrain")
        visible = list(base)
        visible[0] = 1.0
        self.assertEqual(proxy_context({}, visible), "visible_target")
        hidden = [0.0, 0.9, 0.7, 0.0, 0.1, 0.2, 0.2]
        self.assertEqual(proxy_context({}, hidden), "familiar_hidden_goal")
        wedge = list(base)
        wedge[4] = 0.9
        self.assertEqual(proxy_context({}, wedge), "genuine_wedge")

    def test_observer_is_passive(self):
        observer = PassiveSystemicConductor(self.checkpoint())
        observer.observe(
            {
                "food_visible": True,
                "directional_body_clearance": [0.8] * 8,
            }
        )
        self.assertEqual(observer.recommendation, "predictive")
        self.assertEqual(observer.proxy_context, "visible_target")
        self.assertEqual(observer.action_influence, 0)

    def test_lesion_masks_specialist(self):
        observer = PassiveSystemicConductor(self.checkpoint())
        selected, _scores, _probabilities = observer.predict(
            [0.5] * len(FEATURE_NAMES), lesion=2
        )
        self.assertNotEqual(selected, 2)


class PassiveAdaptiveIgnitionGateTests(unittest.TestCase):
    def test_default_gate_releases_confident_challenger_immediately(self):
        gate = PassiveAdaptiveIgnitionGate(enabled=True)
        gate.observe([0.90, 0.04, 0.03, 0.03], "recurrent")
        gate.observe([0.03, 0.90, 0.04, 0.03], "episodic")
        self.assertEqual(gate.active_specialist, "episodic")

    def test_transient_challenger_is_held(self):
        gate = PassiveAdaptiveIgnitionGate(enabled=True, release_frames=2)
        gate.observe([0.90, 0.04, 0.03, 0.03], "recurrent")
        gate.observe([0.03, 0.90, 0.04, 0.03], "recurrent")
        self.assertEqual(gate.active_specialist, "recurrent")
        self.assertEqual(gate.last_event, "transient_challenger_held")
        self.assertEqual(gate.action_influence, 0)

    def test_sustained_challenger_releases_broadcast(self):
        gate = PassiveAdaptiveIgnitionGate(enabled=True, release_frames=2)
        gate.observe([0.90, 0.04, 0.03, 0.03], "recurrent")
        gate.observe([0.03, 0.90, 0.04, 0.03], "episodic")
        gate.observe([0.03, 0.90, 0.04, 0.03], "episodic")
        self.assertEqual(gate.active_specialist, "episodic")
        self.assertEqual(gate.last_event, "sustained_challenger_release")
        self.assertEqual(gate.releases, 1)

    def test_reset_clears_broadcast_but_preserves_totals(self):
        gate = PassiveAdaptiveIgnitionGate(enabled=True)
        gate.observe([0.90, 0.04, 0.03, 0.03], "recurrent")
        gate.reset()
        self.assertEqual(gate.active_specialist, "unobserved")
        self.assertEqual(gate.ignitions, 1)


class BoundedExecutiveRouterTests(unittest.TestCase):
    def test_predictive_recommendation_can_start_mpc(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc", confidence_threshold=0.5
        )
        selected = router.select_mpc(
            "predictive", 0.8, baseline_mpc=False
        )
        self.assertTrue(selected)
        self.assertEqual(router.active_specialist, "predictive")
        self.assertEqual(router.influence_frames, 1)

    def test_hysteresis_prevents_immediate_reversal(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc",
            hz=5.0,
            confidence_threshold=0.5,
            hold_seconds=1.0,
        )
        self.assertTrue(router.select_mpc("predictive", 0.8, False))
        self.assertTrue(router.select_mpc("recurrent", 0.8, False))
        self.assertEqual(router.handoffs, 1)
        self.assertGreater(router.denials, 0)

    def test_mandatory_mpc_overrides_recurrent_recommendation(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc", confidence_threshold=0.5
        )
        selected = router.select_mpc(
            "recurrent",
            0.9,
            baseline_mpc=True,
            mandatory_mpc=True,
        )
        self.assertTrue(selected)
        self.assertEqual(router.active_specialist, "predictive")
        self.assertEqual(router.safety_overrides, 1)

    def test_mandatory_mpc_persists_then_requires_recurrent_confirmation(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc",
            hz=2.0,
            confidence_threshold=0.5,
            hold_seconds=0.5,
            mandatory_persistence_seconds=1.0,
            recurrent_release_frames=2,
        )
        self.assertTrue(
            router.select_mpc(
                "recurrent", 0.9, baseline_mpc=True, mandatory_mpc=True
            )
        )
        self.assertTrue(router.select_mpc("recurrent", 0.9, False))
        self.assertTrue(router.select_mpc("recurrent", 0.9, False))
        self.assertTrue(router.select_mpc("recurrent", 0.9, False))
        self.assertFalse(router.select_mpc("recurrent", 0.9, False))

    def test_fallback_clears_mpc_persistence_latch(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc",
            confidence_threshold=0.5,
        )
        router.select_mpc(
            "predictive", 0.9, baseline_mpc=True, mandatory_mpc=True
        )
        self.assertGreater(router.mandatory_latch_ticks, 0)
        router.select_mpc(
            "predictive", 0.9, baseline_mpc=False, fallback_active=True
        )
        self.assertEqual(router.mandatory_latch_ticks, 0)
        self.assertEqual(router.active_specialist, "fallback")

    def test_unenabled_branch_uses_baseline(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc", confidence_threshold=0.5
        )
        selected = router.select_mpc(
            "episodic", 0.9, baseline_mpc=False
        )
        self.assertFalse(selected)
        self.assertEqual(router.active_specialist, "recurrent")
        self.assertEqual(router.denials, 1)

    def test_fallback_remains_external_priority(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc", confidence_threshold=0.5
        )
        router.select_mpc(
            "predictive",
            0.9,
            baseline_mpc=False,
            fallback_active=True,
        )
        self.assertEqual(router.active_specialist, "fallback")
        self.assertEqual(router.last_reason, "fallback_priority")

    def test_air_mode_requires_confident_episodic_recommendation(self):
        router = BoundedExecutiveRouter(
            mode="recurrent_mpc_air", confidence_threshold=0.5
        )
        self.assertFalse(router.authorizes_episodic("recurrent", 0.9))
        self.assertFalse(router.authorizes_episodic("episodic", 0.4))
        self.assertTrue(router.authorizes_episodic("episodic", 0.8))


if __name__ == "__main__":
    unittest.main()
