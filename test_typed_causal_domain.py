import unittest
import tempfile
import time
from pathlib import Path

import numpy as np

from typed_causal_domain import (
    ORDERED_ACTIONS,
    OrderedCausalProbeWorld,
    PassiveOrderedEpisodeLearner,
    TypedInteractionPool,
    validate_sealed_discovery_memory,
    validate_consume_action,
)


class TypedCausalDomainTests(unittest.TestCase):
    def test_grounded_yellow_is_typed_as_pickup_without_causal_label(self):
        action = validate_consume_action("yellow")
        self.assertEqual(
            action,
            {"operator": "consume", "argument": "yellow", "argument_type": "pickup"},
        )
        self.assertNotIn("antidote", str(action))
        with self.assertRaisesRegex(ValueError, "grounded_pickup"):
            validate_consume_action("unseen_triangle")

    def test_yellow_after_red_cancels_one_pending_event(self):
        world = OrderedCausalProbeWorld(
            hz=5.0, delay_seconds=30.0, cancellation_feature="yellow"
        )
        world.register_pickups(step=10, red=1)
        self.assertEqual(len(world.pending_due_steps), 1)
        cancelled = world.register_pickups(step=40, yellow=1)
        self.assertEqual(cancelled, 1)
        self.assertEqual(world.pop_due(step=200), 0)
        self.assertEqual(world.audit()["cancelled_events"], 1)

    def test_yellow_before_red_and_blue_after_red_do_not_cancel(self):
        world = OrderedCausalProbeWorld(
            hz=5.0, delay_seconds=30.0, cancellation_feature="yellow"
        )
        self.assertEqual(world.register_pickups(step=5, yellow=1), 0)
        world.register_pickups(step=10, red=1)
        self.assertEqual(world.register_pickups(step=40, blue=1), 0)
        self.assertEqual(world.pop_due(step=160), 1)

    def test_same_frame_red_yellow_is_ambiguous_not_cancelled(self):
        world = OrderedCausalProbeWorld(
            hz=5.0, delay_seconds=30.0, cancellation_feature="yellow"
        )
        self.assertEqual(world.register_pickups(step=10, red=1, yellow=1), 0)
        self.assertEqual(len(world.pending_due_steps), 1)
        self.assertEqual(world.audit()["ambiguous_pickup_frames"], 1)

    def test_symmetric_pool_does_not_begin_with_answer_preferred(self):
        pool = TypedInteractionPool()
        self.assertTrue((pool.posterior == 0.25).all())
        selected, scores = pool.select_experiment()
        self.assertEqual(ORDERED_ACTIONS[selected], "red_then_yellow")
        self.assertGreater(scores[selected], scores[0])

    def test_clean_ordered_controls_verify_yellow_specific_rule(self):
        pool = TypedInteractionPool()
        for _ in range(3):
            pool.update(1, True)
            pool.update(2, False)
        self.assertEqual(
            pool.map_hypothesis, "yellow_after_red_suppresses_probe"
        )
        self.assertGreater(pool.map_confidence, 0.95)

    def test_passive_episode_waits_for_deadline_before_learning(self):
        learner = PassiveOrderedEpisodeLearner(hz=5.0, delay_seconds=60.0, enabled=True)
        learner.observe_pickups(10, red=1)
        learner.observe_pickups(150, yellow=1, can_start=False)
        self.assertFalse(learner.observe_deadline(309, observed_probe_events=0))
        self.assertEqual(learner.pool.updates, 0)
        self.assertTrue(learner.observe_deadline(310, observed_probe_events=0))
        self.assertEqual(learner.last_action, "red_then_yellow")
        self.assertEqual(learner.last_outcome, "suppressed")

    def test_passive_episode_uses_observation_and_discards_contamination(self):
        learner = PassiveOrderedEpisodeLearner(hz=5.0, delay_seconds=10.0, enabled=True)
        learner.observe_pickups(0, red=1)
        learner.observe_pickups(10, blue=1, can_start=False)
        learner.observe_deadline(50, observed_probe_events=1)
        self.assertEqual(learner.last_action, "red_then_blue")
        self.assertEqual(learner.last_outcome, "probe_observed")
        learner.observe_pickups(100, red=1)
        learner.observe_pickups(105, yellow=1, can_start=False)
        learner.observe_pickups(106, blue=1, can_start=False)
        self.assertFalse(learner.observe_deadline(150, observed_probe_events=0))
        self.assertEqual(learner.pool.updates, 1)
        self.assertEqual(learner.discarded_episodes, 1)

    def test_passive_episode_memory_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "typed.json"
            learner = PassiveOrderedEpisodeLearner(
                hz=1.0,
                delay_seconds=2.0,
                enabled=True,
                memory_path=path,
            )
            learner.observe_pickups(0, red=1)
            learner.observe_pickups(1, blue=1, can_start=False)
            learner.observe_deadline(2, observed_probe_events=1)
            restored = PassiveOrderedEpisodeLearner(
                hz=1.0,
                delay_seconds=2.0,
                enabled=True,
                memory_path=path,
            )
            self.assertTrue(restored.memory_loaded)
            self.assertEqual(restored.pool.updates, 1)
            self.assertTrue(np.allclose(restored.pool.posterior, learner.pool.posterior))

    def test_read_only_discovery_loads_but_only_run_memory_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            discovery = Path(directory) / "discovery.json"
            run_memory = Path(directory) / "run.json"
            source = Path(
                "checkpoints/typed_interaction/discovery_three_20260817.json"
            ).read_text()
            discovery.write_text(source)
            original = discovery.read_bytes()
            learner = PassiveOrderedEpisodeLearner(
                hz=1.0,
                delay_seconds=2.0,
                enabled=True,
                memory_path=run_memory,
                discovery_memory_path=discovery,
            )
            self.assertEqual(learner.pool.updates, 3)
            self.assertEqual(len(learner.observations), 3)
            self.assertTrue(run_memory.exists())
            self.assertEqual(discovery.read_bytes(), original)
            learner.observe_pickups(0, red=1)
            learner.observe_pickups(1, yellow=1, can_start=False)
            self.assertTrue(learner.observe_deadline(2, observed_probe_events=0))
            self.assertEqual(discovery.read_bytes(), original)
            self.assertTrue(run_memory.exists())
            self.assertEqual(len(__import__("json").loads(run_memory.read_text())["observations"]), 4)

    def test_discovery_and_run_memory_must_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.json"
            with self.assertRaisesRegex(ValueError, "read_only"):
                PassiveOrderedEpisodeLearner(
                    memory_path=path, discovery_memory_path=path
                )

    def test_sealed_discovery_checkpoint_validates_exact_profile(self):
        path = Path("checkpoints/typed_interaction/discovery_three_20260817.json")
        payload = validate_sealed_discovery_memory(path)
        self.assertEqual(payload["updates"], 3)
        self.assertEqual(len(payload["observations"]), 3)

    def test_live_formulation_runs_asynchronously_from_observations(self):
        def proposer(summary, cutoff):
            self.assertEqual(cutoff, 3)
            self.assertIn("red_then_yellow", summary)
            return {"second_action": "yellow"}, "yellow", {"ok": True}

        learner = PassiveOrderedEpisodeLearner(
            enabled=True, hypothesis_proposer=proposer, formulation_minimum=3
        )
        learner.observations = [
            {"ordered_action": "red_then_blue", "probe_suppressed": False},
            {"ordered_action": "red_then_blue", "probe_suppressed": False},
            {"ordered_action": "red_then_yellow", "probe_suppressed": True},
        ]
        learner.poll_formulation()
        for _ in range(100):
            learner.poll_formulation()
            if learner.formulated_hypothesis:
                break
            time.sleep(0.001)
        self.assertEqual(learner.formulation_status, "admitted_unverified")
        self.assertEqual(learner.formulated_hypothesis["second_action"], "yellow")

    def test_matching_post_cutoff_episode_promotes_held_out_status(self):
        def proposer(_summary, cutoff):
            return (
                {
                    "initiator": "red",
                    "second_action": "yellow",
                    "observed_effect": "suppresses_probe",
                    "admitted_after_observation": cutoff,
                },
                "yellow",
                {},
            )

        learner = PassiveOrderedEpisodeLearner(
            hz=1.0,
            delay_seconds=2.0,
            enabled=True,
            hypothesis_proposer=proposer,
            formulation_minimum=3,
        )
        learner.observations = [
            {"index": 1, "ordered_action": "red_then_blue", "probe_suppressed": False},
            {"index": 2, "ordered_action": "red_then_blue", "probe_suppressed": False},
            {"index": 3, "ordered_action": "red_then_yellow", "probe_suppressed": True},
        ]
        learner.pool.updates = 3
        learner.poll_formulation()
        for _ in range(100):
            learner.poll_formulation()
            if learner.formulated_hypothesis:
                break
            time.sleep(0.001)
        self.assertEqual(learner.formulation_status, "admitted_unverified")
        learner.observe_pickups(10, red=1)
        learner.observe_pickups(11, yellow=1, can_start=False)
        self.assertTrue(learner.observe_deadline(12, observed_probe_events=0))
        self.assertEqual(learner.formulation_status, "verified_held_out")
        self.assertEqual(learner.formulation_held_out_evaluations, 1)
        self.assertEqual(learner.formulation_held_out_confirmations, 1)


if __name__ == "__main__":
    unittest.main()
