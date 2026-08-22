import unittest
import tempfile
from pathlib import Path

import numpy as np

from embodied_unity_loop import (
    EmbodiedAdaptiveResonanceObserver,
    EmbodiedFunctionalEgo,
    art_route_selection_score,
    art_route_context,
    conductor_gate_allows,
    fuzzy_art_similarity,
    orbit_teacher_due,
    orbit_recovery_should_finish,
    pgnw_experiment_guidance_allowed,
    pgnw_constrained_target_action,
    route_memory_adapter,
    route_reversal_count,
    route_waypoint_radius,
    smooth_target_intercept,
    stable_recovery_due,
    trajectory_orbit_metrics,
)
from terrain_resource_memory import PassiveTerrainResourceMemory


class FakeShadowPolicy:
    def initial_state(self, batch_size):
        return ("clean", batch_size)


class StableRecoveryGateTests(unittest.TestCase):
    def test_physical_wedge_independently_triggers_recovery(self):
        self.assertTrue(stable_recovery_due(40, 0, 40))

    def test_critical_orbit_independently_triggers_recovery(self):
        self.assertTrue(stable_recovery_due(0, 40, 40))

    def test_subthreshold_signals_do_not_combine(self):
        self.assertFalse(stable_recovery_due(20, 20, 40))

    def test_sustained_orbit_independently_triggers_recovery(self):
        self.assertTrue(stable_recovery_due(0, 0, 40, sustained_orbit=True))

    def test_orbit_teacher_waits_for_bounded_episodic_attempt(self):
        self.assertFalse(orbit_teacher_due(True, route_active=True))
        self.assertFalse(orbit_teacher_due(True, route_pending=True))
        self.assertTrue(orbit_teacher_due(True))


class MetabolicObservationTests(unittest.TestCase):
    @staticmethod
    def body(**updates):
        body = {
            "mushroom_pickups_total": 0,
            "mushroom_reward_total": 0.0,
            "red_mushroom_pickups_total": 0,
            "blue_mushroom_pickups_total": 0,
            "yellow_flower_pickups_total": 0,
            "mushroom_feature": "none",
            "grounded": True,
            "forward_clear": True,
            "left_clear": True,
            "right_clear": True,
            "directional_rays": [1.0] * 8,
            "directional_body_clearance": [1.0] * 8,
        }
        body.update(updates)
        return body

    def test_red_pickup_has_delayed_passive_causal_probe_effect(self):
        ego = EmbodiedFunctionalEgo(hz=5.0)
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                red_mushroom_pickups_total=1,
                mushroom_feature="red",
            )
        )
        self.assertEqual(ego.causal_probe_events, 0)
        self.assertEqual(len(ego.causal_probe_due_steps), 1)
        for _ in range(ego.causal_probe_delay_ticks):
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                )
            )
        self.assertEqual(ego.causal_probe_events, 1)
        self.assertGreater(ego.causal_probe_signal, 0.30)
        self.assertEqual(ego.metabolic_pressure, ego.causal_probe_signal)

    def test_blue_pickup_never_schedules_red_challenge(self):
        ego = EmbodiedFunctionalEgo(hz=5.0)
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                mushroom_feature="blue",
            )
        )
        self.assertFalse(ego.causal_probe_due_steps)

    def test_live_metabolic_role_mode_learns_and_verifies_blue(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory) / "metabolic.json"
            ego = EmbodiedFunctionalEgo(
                hz=5.0,
                typed_metabolic_role_learning=True,
                typed_metabolic_role_memory=memory,
            )
            ego.update_from_body(self.body())
            totals = {"red": 0, "blue": 0, "yellow": 0}
            for index, feature in enumerate(
                ("red", "yellow", "blue", "blue", "blue", "yellow"),
                start=1,
            ):
                ego.hunger = 0.60
                totals[feature] += 1
                ego.steps += 1
                ego.update_from_body(self.body(
                    mushroom_pickups_total=index,
                    mushroom_reward_total=0.35 * index,
                    red_mushroom_pickups_total=totals["red"],
                    blue_mushroom_pickups_total=totals["blue"],
                    yellow_flower_pickups_total=totals["yellow"],
                    mushroom_feature=feature,
                ))
                if feature == "blue":
                    self.assertLess(ego.hunger, 0.30)
                else:
                    self.assertGreater(ego.hunger, 0.59)

            audit = ego.typed_metabolic_role_learner.audit()
            self.assertEqual(audit["status"], "verified_held_out")
            self.assertEqual(audit["admitted_nutrient"], "blue")
            self.assertEqual(audit["admission_cutoff"], 4)
            self.assertEqual(audit["held_out_positive"], 1)
            self.assertEqual(audit["held_out_negative"], 1)
            self.assertTrue(memory.exists())

    def test_typed_yellow_after_red_cancels_pending_probe(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            causal_probe_delay_seconds=3.0,
            causal_probe_cancellation_feature="yellow",
        )
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                red_mushroom_pickups_total=1,
                mushroom_feature="red",
            )
        )
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=2,
                mushroom_reward_total=0.70,
                red_mushroom_pickups_total=1,
                yellow_flower_pickups_total=1,
                mushroom_feature="yellow",
            )
        )
        self.assertFalse(ego.causal_probe_due_steps)
        for _ in range(4):
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    mushroom_pickups_total=2,
                    mushroom_reward_total=0.70,
                    red_mushroom_pickups_total=1,
                    yellow_flower_pickups_total=1,
                )
            )
        self.assertEqual(ego.causal_probe_events, 0)
        self.assertEqual(ego.causal_probe_world.cancelled_events, 1)

    def test_typed_yellow_before_red_does_not_cancel_probe(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            causal_probe_delay_seconds=2.0,
            causal_probe_cancellation_feature="yellow",
        )
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                yellow_flower_pickups_total=1,
                mushroom_feature="yellow",
            )
        )
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=2,
                mushroom_reward_total=0.70,
                red_mushroom_pickups_total=1,
                yellow_flower_pickups_total=1,
                mushroom_feature="red",
            )
        )
        for _ in range(2):
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    mushroom_pickups_total=2,
                    mushroom_reward_total=0.70,
                    red_mushroom_pickups_total=1,
                    yellow_flower_pickups_total=1,
                )
            )
        self.assertEqual(ego.causal_probe_events, 1)
        self.assertEqual(ego.causal_probe_world.cancelled_events, 0)

    def test_causal_probe_has_zero_workspace_and_navigation_influence(self):
        ego = EmbodiedFunctionalEgo(hz=5.0)
        ego.tiny_scientist_memory.rules = [
            {"rule_id": "red_pickup_delayed_pressure_increase_v1"}
        ]
        ego.causal_probe_signal = 0.60
        before = (
            ego.prediction_error,
            ego.crosstalk,
            ego.complexity,
            ego.tiny_scientist_rule_action_influence,
        )
        ego.update_tiny_scientist_rule_targeting(
            self.body(
                red_food_visible=True,
                blue_food_visible=True,
                red_food_distance=8.0,
                blue_food_distance=8.8,
                red_food_world_x=0.0,
                red_food_world_z=1.0,
                blue_food_world_x=1.0,
                blue_food_world_z=0.0,
            )
        )
        after = (
            ego.prediction_error,
            ego.crosstalk,
            ego.complexity,
            ego.tiny_scientist_rule_action_influence,
        )
        self.assertFalse(ego.tiny_scientist_rule_active)
        self.assertEqual(ego.tiny_scientist_rule_target_feature, "none")
        self.assertEqual(ego.tiny_scientist_rule_guidance_weight, 0.0)
        self.assertEqual(after, before)

    def test_delayed_probe_does_not_change_functional_ego_state(self):
        red = EmbodiedFunctionalEgo(hz=5.0)
        blue = EmbodiedFunctionalEgo(hz=5.0)
        red.update_from_body(self.body())
        blue.update_from_body(self.body())
        red.steps += 1
        blue.steps += 1
        red.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                red_mushroom_pickups_total=1,
                mushroom_feature="red",
            )
        )
        blue.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                mushroom_feature="blue",
            )
        )
        for _ in range(red.causal_probe_delay_ticks):
            red.steps += 1
            blue.steps += 1
            red.update_from_body(
                self.body(
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                )
            )
            blue.update_from_body(
                self.body(
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                )
            )
        self.assertGreater(red.causal_probe_signal, 0.30)
        self.assertEqual(blue.causal_probe_signal, 0.0)
        self.assertEqual(
            (
                red.prediction_error,
                red.crosstalk,
                red.complexity,
                red.hunger,
                red.dopamine,
                red.last_action,
                red.workspace_packet,
            ),
            (
                blue.prediction_error,
                blue.crosstalk,
                blue.complexity,
                blue.hunger,
                blue.dopamine,
                blue.last_action,
                blue.workspace_packet,
            ),
        )

    def test_active_tiny_scientist_rule_control_is_retired(self):
        with self.assertRaisesRegex(ValueError, "passive_only"):
            EmbodiedFunctionalEgo(
                hz=5.0, tiny_scientist_rule_control="pressure_avoidance"
            )

    def test_verified_protective_control_requires_learning_and_committed_pgnw(self):
        with self.assertRaisesRegex(ValueError, "requires_typed_learning"):
            EmbodiedFunctionalEgo(
                hz=5.0,
                shadow_mpc=True,
                tiny_scientist_experiment_control="committed",
                typed_interaction_rule_control="verified_protective",
            )

    def test_verified_rule_retrieves_unseen_yellow_from_typed_memory(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            shadow_mpc=True,
            tiny_scientist_experiment_control="committed",
            typed_interaction_learning=True,
            typed_interaction_rule_control="verified_protective",
        )
        with tempfile.TemporaryDirectory() as directory:
            ego.resource_memory = PassiveTerrainResourceMemory(
                Path(directory) / "memory.json", control_mode="guided"
            )
            ego.resource_memory.record_typed_reward("yellow", 30.0, 40.0)
            ego.typed_interaction_learner.pool.posterior = np.array(
                [0.97, 0.01, 0.01, 0.01]
            )
            ego.update_from_body(self.body(x=0.0, z=0.0))
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    x=0.0,
                    z=0.0,
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                    mushroom_feature="red",
                    yellow_food_visible=False,
                )
            )

            self.assertEqual(ego.resource_memory.active_feature, "yellow")
            self.assertEqual(ego.resource_memory.target, (30.0, 40.0))
            self.assertTrue(
                ego.pgnw_experiment_planner.protective_memory_active
            )
            self.assertEqual(
                ego.pgnw_experiment_planner.guidance_vector, (0.6, 0.8)
            )
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    x=2.0,
                    z=0.0,
                    mushroom_pickups_total=2,
                    mushroom_reward_total=0.70,
                    red_mushroom_pickups_total=1,
                    blue_mushroom_pickups_total=1,
                    mushroom_feature="blue",
                    yellow_food_visible=False,
                )
            )

            self.assertTrue(ego.causal_probe_world.pending_due_steps)
            self.assertEqual(
                ego.typed_interaction_learner.active["second"], "blue"
            )
            self.assertEqual(ego.resource_memory.active_feature, "yellow")
            self.assertTrue(
                ego.pgnw_experiment_planner.protective_memory_active
            )
        with self.assertRaisesRegex(ValueError, "requires_committed_pgnw"):
            EmbodiedFunctionalEgo(
                hz=5.0,
                typed_interaction_learning=True,
                typed_interaction_rule_control="verified_protective",
            )

    def test_passive_arbitration_logs_candidates_without_motor_authority(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            shadow_mpc=True,
            tiny_scientist_experiment_control="committed",
            typed_interaction_learning=True,
            typed_interaction_rule_control="verified_protective",
            pgnw_multi_hypothesis_arbitration="passive",
            causal_probe_hunger_cost=0.25,
            causal_probe_delay_seconds=240.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            ego.resource_memory = PassiveTerrainResourceMemory(
                Path(directory) / "memory.json", control_mode="guided"
            )
            ego.resource_memory.record_typed_reward("yellow", 30.0, 40.0)
            ego.resource_memory.record_typed_reward("blue", 12.0, 0.0)
            ego.typed_interaction_learner.pool.posterior = np.array(
                [0.9763185, 0.0000204, 0.0207773, 0.0028838]
            )
            ego.update_from_body(self.body(x=0.0, z=0.0))
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    x=0.0,
                    z=0.0,
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                    mushroom_feature="red",
                    yellow_food_visible=False,
                )
            )

            audit = ego.pgnw_experiment_planner.audit()
            self.assertTrue(audit["arbitration_active"])
            self.assertEqual(
                audit["arbitration_selected_feature"], "yellow"
            )
            self.assertEqual(len(audit["arbitration_records"]), 2)
            self.assertEqual(audit["arbitration_authority"], 0.0)
            self.assertEqual(audit["arbitration_action_influence"], 0)
            self.assertTrue(
                ego.pgnw_experiment_planner.protective_memory_active
            )

    def test_uncancelled_probe_applies_bounded_hunger_cost(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            causal_probe_delay_seconds=2.0,
            causal_probe_cancellation_feature="yellow",
            causal_probe_hunger_cost=0.25,
        )
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(self.body(
            mushroom_pickups_total=1,
            mushroom_reward_total=0.35,
            red_mushroom_pickups_total=1,
        ))
        hunger_before = ego.hunger
        for _ in range(ego.causal_probe_delay_ticks):
            ego.steps += 1
            ego.update_from_body(self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                red_mushroom_pickups_total=1,
            ))
        self.assertEqual(ego.causal_probe_hunger_cost_events, 1)
        self.assertAlmostEqual(ego.causal_probe_hunger_cost_total, 0.25)
        self.assertGreaterEqual(ego.hunger, hunger_before + 0.25)

    def test_yellow_cancellation_avoids_hunger_cost(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            causal_probe_delay_seconds=3.0,
            causal_probe_cancellation_feature="yellow",
            causal_probe_hunger_cost=0.25,
        )
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(self.body(
            mushroom_pickups_total=1,
            mushroom_reward_total=0.35,
            red_mushroom_pickups_total=1,
        ))
        ego.steps += 1
        ego.update_from_body(self.body(
            mushroom_pickups_total=2,
            mushroom_reward_total=0.70,
            red_mushroom_pickups_total=1,
            yellow_flower_pickups_total=1,
        ))
        for _ in range(ego.causal_probe_delay_ticks):
            ego.steps += 1
            ego.update_from_body(self.body(
                mushroom_pickups_total=2,
                mushroom_reward_total=0.70,
                red_mushroom_pickups_total=1,
                yellow_flower_pickups_total=1,
            ))
        self.assertEqual(ego.causal_probe_hunger_cost_events, 0)
        self.assertEqual(ego.causal_probe_hunger_cost_total, 0.0)

    def test_bounded_pgnw_requires_mpc(self):
        with self.assertRaisesRegex(ValueError, "requires_shadow_mpc"):
            EmbodiedFunctionalEgo(
                hz=5.0,
                tiny_scientist_experiment_control="bounded",
            )

    def test_passive_pgnw_updates_from_delayed_red_probe(self):
        ego = EmbodiedFunctionalEgo(
            hz=1.0,
            tiny_scientist_experiment_control="passive",
        )
        planner = ego.pgnw_experiment_planner
        planner.current_action = 0
        planner.phase = "seeking_target"
        ego.update_from_body(self.body())
        ego.steps += 1
        ego.update_from_body(
            self.body(
                mushroom_pickups_total=1,
                mushroom_reward_total=0.35,
                red_mushroom_pickups_total=1,
                mushroom_feature="red",
            )
        )
        for _ in range(ego.causal_probe_delay_ticks):
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                )
            )
        self.assertEqual(planner.experiments_completed, 1)
        self.assertEqual(planner.last_outcome, "probe_rise")
        self.assertGreater(planner.posterior[0], planner.posterior[1])
        self.assertEqual(planner.action_influence, 0)

    def test_pgnw_guidance_yields_to_every_safety_gate(self):
        base = dict(
            mode="bounded",
            guidance_active=True,
            fallback_active=False,
            stuck=False,
            hunger=0.50,
            air_guided=False,
            resource_guided=False,
        )
        self.assertTrue(pgnw_experiment_guidance_allowed(**base))
        committed = dict(base, mode="committed")
        self.assertTrue(pgnw_experiment_guidance_allowed(**committed))
        for override in (
            {"fallback_active": True},
            {"stuck": True},
            {"hunger": 0.92},
            {"air_guided": True},
            {"resource_guided": True},
            {"mode": "passive"},
            {"guidance_active": False},
        ):
            values = dict(base)
            values.update(override)
            self.assertFalse(pgnw_experiment_guidance_allowed(**values))

        rescue = dict(base, hunger=0.99, verified_metabolic_rescue=True)
        self.assertTrue(pgnw_experiment_guidance_allowed(**rescue))
        for override in (
            {"fallback_active": True},
            {"stuck": True},
            {"air_guided": True},
            {"resource_guided": True},
            {"mode": "passive"},
            {"guidance_active": False},
        ):
            values = dict(rescue)
            values.update(override)
            self.assertFalse(pgnw_experiment_guidance_allowed(**values))

    def test_committed_target_choice_respects_safety_score_regret(self):
        scores = [1.00, 0.91, 0.76, float("-inf")]
        alignments = [0.1, 0.9, 1.0, 1.0]
        self.assertEqual(
            pgnw_constrained_target_action(scores, alignments, 0.18),
            1,
        )
        self.assertEqual(
            pgnw_constrained_target_action(scores, alignments, 0.30),
            2,
        )


class ArtRouteRetrievalTests(unittest.TestCase):
    def test_directional_context_distinguishes_mirrored_geometry(self):
        original = [0.44, 0.57, 0.40, 0.57, 1.0, 1.0, 1.0, 0.62]
        mirrored = [0.43, 0.61, 1.0, 1.0, 1.0, 0.58, 0.41, 0.58]
        self.assertGreater(fuzzy_art_similarity(original, original), 0.99)
        self.assertLess(fuzzy_art_similarity(original, mirrored), 0.86)

    def test_context_requires_complete_directional_rays(self):
        self.assertIsNone(art_route_context({"directional_rays": [1.0] * 7}))
        self.assertEqual(
            art_route_context({"directional_rays": [1.2] + [0.5] * 7}),
            [1.0] + [0.5] * 7,
        )

    def test_art_and_single_routes_share_target_latching(self):
        self.assertTrue(route_memory_adapter("episodic_route"))
        self.assertTrue(route_memory_adapter("art_route_library"))
        self.assertFalse(route_memory_adapter("temporal_gru"))

    def test_intercept_turns_from_prior_motion_without_heading_snap(self):
        move = smooth_target_intercept((1.0, 0.0), (-1.0, 0.0), 3.0)
        angle = abs(__import__("math").degrees(__import__("math").atan2(move[1], move[0])))
        self.assertAlmostEqual(angle, 55.0, places=5)
        self.assertLess(__import__("math").hypot(*move), 0.6)

    def test_terminal_funnel_keeps_penultimate_waypoint_reachable(self):
        self.assertAlmostEqual(route_waypoint_radius(34, 36, 0.85, 0.30, 5), 0.55)
        self.assertAlmostEqual(route_waypoint_radius(35, 36, 0.85, 0.30, 5), 0.30)

    def test_route_geometry_detects_multiple_direction_reversals(self):
        smooth = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.2), (3.0, 0.2)]
        reversing = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        self.assertEqual(route_reversal_count(smooth), 0)
        self.assertEqual(route_reversal_count(reversing), 2)

    def test_route_selection_rewards_grounded_terminal_alignment(self):
        self.assertGreater(
            art_route_selection_score(0.90, 0.40, {"terminal_alignment": 0.9}),
            art_route_selection_score(0.90, 0.40, {"terminal_alignment": 0.2}),
        )

    def test_live_conductor_only_gates_its_validated_context(self):
        self.assertTrue(
            conductor_gate_allows(
                "familiar_hidden_goal",
                "familiar_hidden_goal",
                "episodic",
                0.95,
            )
        )
        self.assertFalse(
            conductor_gate_allows(
                "familiar_hidden_goal",
                "familiar_hidden_goal",
                "recurrent",
                0.95,
            )
        )
        self.assertTrue(
            conductor_gate_allows(
                "familiar_hidden_goal",
                "visible_target",
                "insufficient_evidence",
                0.0,
            )
        )


class TrajectoryOrbitTests(unittest.TestCase):
    def test_detects_repeated_low_efficiency_motion_with_grounded_evidence(self):
        samples = []
        for cycle in range(8):
            for x, z in ((0, 0), (2, 0), (2, 2), (0, 2), (0, 0)):
                samples.append((x, z, cycle % 2 == 0, True))
        metrics = trajectory_orbit_metrics(samples)
        self.assertTrue(metrics["detected"])
        self.assertLess(metrics["efficiency"], 0.22)

    def test_rejects_sustained_forward_progress(self):
        samples = [(float(index), 0.0, True, False) for index in range(20)]
        self.assertFalse(trajectory_orbit_metrics(samples)["detected"])


class OrbitRecoveryLatchTests(unittest.TestCase):
    def test_does_not_release_on_first_clear_frame(self):
        self.assertFalse(orbit_recovery_should_finish(5, 0.5, True, 25, 60))

    def test_releases_after_sustained_clear_displacement(self):
        self.assertTrue(orbit_recovery_should_finish(25, 4.1, True, 25, 60))

    def test_collision_prevents_early_release(self):
        self.assertFalse(orbit_recovery_should_finish(40, 5.0, False, 25, 60))

    def test_time_bound_always_releases(self):
        self.assertTrue(orbit_recovery_should_finish(60, 0.0, False, 25, 60))

    def test_rejects_ungrounded_rotation(self):
        samples = [(x, z, False, False) for _ in range(8) for x, z in ((0, 0), (2, 0), (2, 2), (0, 2), (0, 0))]
        self.assertFalse(trajectory_orbit_metrics(samples)["detected"])


class ShadowEpisodeBoundaryTests(unittest.TestCase):
    def make_ego(self):
        ego = EmbodiedFunctionalEgo.__new__(EmbodiedFunctionalEgo)
        ego.hz = 5.0
        ego.shadow_policy = FakeShadowPolicy()
        ego.shadow_hidden = ("dirty", 1)
        ego.shadow_episode_key = None
        ego.shadow_episode_resets = 0
        ego.shadow_episode_idle_active = False
        ego.shadow_episode_idle_ticks = 0
        ego.shadow_previous_action = 5
        ego.shadow_previous_position = (4.0, 6.0)
        ego.shadow_previous_proposal = "down"
        ego.shadow_last_reward = 1.0
        ego.shadow_mpc_engaged = True
        ego.shadow_mpc_hold_ticks = 12
        ego.shadow_fallback_hold_ticks = 8
        ego.orbit_history = __import__("collections").deque([(1.0, 1.0, True, False)])
        ego.orbit_path = 4.0
        ego.orbit_net = 0.5
        ego.orbit_efficiency = 0.1
        ego.orbit_adapter_active = True
        ego.orbit_adapter_action = "left"
        ego.orbit_adapter_confidence = 0.9
        ego.orbit_adapter_hold_ticks = 20
        ego.orbit_adapter_action_ticks = 4
        ego.orbit_adapter_elapsed_ticks = 10
        ego.orbit_adapter_selected = 6
        ego.orbit_adapter_start_position = (0.0, 0.0)
        ego.orbit_adapter_displacement = 2.0
        ego.hidden_goal_adapter_history = __import__("collections").deque(
            [("old", 1)], maxlen=12
        )
        ego.hidden_goal_route_index = 7
        ego.hidden_goal_route_origin = (1.0, 2.0)
        ego.hidden_goal_route_distance = 0.4
        ego.hidden_goal_route_vetoes = 3
        ego.hidden_goal_route_terminal_holds = 2
        ego.hidden_goal_route_terminal_active = True
        ego.hidden_goal_food_latch_ticks = 5
        ego.hidden_goal_food_target = (3.0, 4.0)
        ego.hidden_goal_adapter_active = True
        ego.hidden_goal_adapter_action = "up"
        ego.hidden_goal_adapter_confidence = 0.9
        ego.hidden_goal_adapter_selected = 0
        ego.hidden_goal_adapter_hold_ticks = 2
        ego.hidden_goal_adapter_lateral_sign = 1
        ego.hidden_goal_adapter_lateral_start_position = (0.0, 0.0)
        ego.hidden_goal_adapter_lateral_elapsed_ticks = 3
        return ego

    def test_first_packet_establishes_episode_without_reset(self):
        ego = self.make_ego()
        self.assertFalse(ego.synchronize_shadow_episode({"trap_course": "utrap", "trap_episode": 1}))
        self.assertEqual(ego.shadow_episode_resets, 0)
        self.assertEqual(ego.shadow_hidden, ("dirty", 1))

    def test_course_change_resets_transient_navigation_state(self):
        ego = self.make_ego()
        ego.synchronize_shadow_episode({"trap_course": "utrap", "trap_episode": 1})
        self.assertTrue(ego.synchronize_shadow_episode({"trap_course": "ctrap", "trap_episode": 2}))
        self.assertEqual(ego.shadow_hidden, ("clean", 1))
        self.assertEqual(ego.shadow_world_move, (0.0, 0.0))
        self.assertGreater(ego.shadow_episode_idle_ticks, 0)
        self.assertEqual(ego.shadow_previous_position, None)
        self.assertEqual(len(ego.orbit_history), 0)
        self.assertEqual(ego.shadow_mpc_hold_ticks, 0)
        self.assertEqual(ego.orbit_adapter_selected, None)
        self.assertEqual(len(ego.hidden_goal_adapter_history), 0)
        self.assertEqual(ego.hidden_goal_route_index, 0)
        self.assertFalse(ego.hidden_goal_route_terminal_active)
        self.assertEqual(ego.hidden_goal_food_latch_ticks, 0)
        self.assertIsNone(ego.hidden_goal_food_target)
        self.assertEqual(ego.shadow_episode_resets, 1)

    def test_repeated_packet_does_not_reset_memory(self):
        ego = self.make_ego()
        packet = {"trap_course": "ctrap", "trap_episode": 2}
        ego.synchronize_shadow_episode(packet)
        self.assertFalse(ego.synchronize_shadow_episode(packet))
        self.assertEqual(ego.shadow_episode_resets, 0)


class EmbodiedAdaptiveResonanceObserverTests(unittest.TestCase):
    def test_similar_telemetry_resonates_with_stable_category(self):
        observer = EmbodiedAdaptiveResonanceObserver(vigilance=0.80)
        observer.update([0.80, 0.20, 0.65, 0.35], "clear_traversal")
        first_category = observer.category
        self.assertTrue(observer.novel)

        observer.update([0.79, 0.21, 0.64, 0.36], "clear_traversal")
        self.assertTrue(observer.resonance)
        self.assertFalse(observer.novel)
        self.assertEqual(observer.category, first_category)
        self.assertEqual(observer.category_label, "clear_traversal")

    def test_mismatch_search_creates_new_category_without_steering(self):
        observer = EmbodiedAdaptiveResonanceObserver(vigilance=0.90)
        observer.update([0.95, 0.95, 0.05, 0.05], "clear_traversal")
        first_category = observer.category
        observer.update([0.05, 0.05, 0.95, 0.95], "contact_obstacle")

        self.assertTrue(observer.novel)
        self.assertNotEqual(observer.category, first_category)
        self.assertGreaterEqual(observer.mismatch_resets, 1)
        self.assertEqual(len(observer.templates), 2)

    def test_capacity_limit_reports_unknown_instead_of_overwriting(self):
        observer = EmbodiedAdaptiveResonanceObserver(vigilance=0.95, max_categories=1)
        observer.update([0.95, 0.95, 0.05, 0.05], "clear_traversal")
        observer.update([0.05, 0.05, 0.95, 0.95], "contact_obstacle")

        self.assertTrue(observer.unknown)
        self.assertEqual(observer.category, "unknown")
        self.assertEqual(len(observer.templates), 1)


if __name__ == "__main__":
    unittest.main()
