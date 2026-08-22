import unittest
import time
from types import SimpleNamespace

import numpy as np

from embodied_pgnw_planner import EmbodiedPGNWExperimentPlanner
from pgnw_hypothesis_selection_lab import HYPOTHESES
from typed_causal_domain import PassiveOrderedEpisodeLearner
from typed_metabolic_domain import HeldOutMetabolicRoleLearner


class EmbodiedPGNWPlannerTests(unittest.TestCase):
    def planner(self, mode="passive", seed=2):
        return EmbodiedPGNWExperimentPlanner(
            mode=mode,
            hz=1.0,
            seed=seed,
            delay_seconds=2.0,
        )

    def test_passive_mode_never_emits_guidance(self):
        planner = self.planner("passive")
        planner.update_guidance(
            {
                "red_food_visible": True,
                "red_food_world_x": 1.0,
                "red_food_world_z": 0.0,
            }
        )
        self.assertFalse(planner.guidance_active)
        self.assertEqual(planner.action_influence, 0)

    def test_matching_red_observation_updates_posterior_after_delay(self):
        planner = self.planner(seed=1)
        planner.current_action = 0
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0)
        planner.update(1, 0.0, 1, 1)
        before = planner.posterior.copy()
        planner.update(2, 0.0, 1, 1)
        planner.update(3, 0.34, 1, 1)
        self.assertEqual(planner.experiments_completed, 1)
        self.assertGreater(planner.posterior[0], before[0])
        self.assertGreater(planner.posterior[2], before[2])

    def test_blue_remains_negative_control_evidence(self):
        planner = self.planner(seed=1)
        planner.current_action = 1
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0)
        planner.update(1, 0.0, 1, 0)
        planner.update(3, 0.0, 1, 0)
        red_index = HYPOTHESES.index("red_causes_probe")
        blue_index = HYPOTHESES.index("blue_causes_probe")
        self.assertGreater(
            planner.posterior[red_index], planner.posterior[blue_index]
        )

    def test_legacy_red_blue_planner_never_collapses_yellow_into_blue(self):
        planner = self.planner(seed=1)
        planner.current_action = 1
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0, 0, 0)
        planner.update(1, 0.0, 1, 0, 0, 1)
        self.assertEqual(planner.last_outcome, "unexpected_yellow")
        self.assertEqual(planner.protocol_mismatches, 1)
        self.assertEqual(planner.experiments_discarded, 1)
        self.assertEqual(planner.posterior_updates, 0)
        self.assertTrue(np.allclose(planner.posterior, 0.2))

    def test_intervening_pickup_discards_trial(self):
        planner = self.planner(seed=1)
        planner.current_action = 0
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0)
        planner.update(1, 0.0, 1, 1)
        planner.update(2, 0.0, 2, 1)
        planner.update(3, 0.34, 2, 1)
        self.assertEqual(planner.experiments_discarded, 1)
        self.assertEqual(planner.posterior_updates, 0)
        self.assertTrue(np.allclose(planner.posterior, 0.2))

    def test_bounded_guidance_targets_requested_visible_color(self):
        planner = self.planner("bounded", seed=1)
        planner.current_action = 1
        planner.phase = "seeking_target"
        planner.update_guidance(
            {
                "blue_food_visible": True,
                "blue_food_distance": 7.0,
                "blue_food_world_x": 3.0,
                "blue_food_world_z": 4.0,
            }
        )
        self.assertTrue(planner.guidance_active)
        self.assertEqual(planner.guidance_vector, (0.6, 0.8))
        self.assertLessEqual(planner.guidance_weight, 0.03)

    def test_stronger_guidance_configuration_is_explicitly_bounded(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="bounded",
            hz=5.0,
            seed=3,
            max_guidance_weight=0.12,
            guidance_margin_threshold=0.25,
        )
        planner.current_action = 0
        planner.phase = "seeking_target"
        planner.update_guidance(
            {
                "red_food_visible": True,
                "red_food_distance": 6.0,
                "red_food_world_x": 1.0,
                "red_food_world_z": 0.0,
            }
        )
        self.assertEqual(planner.guidance_weight, 0.12)
        self.assertEqual(planner.guidance_margin_threshold, 0.25)
        self.assertLessEqual(planner.max_guidance_weight, 0.20)

    def test_committed_mode_has_bounded_duty_cycle(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            seed=3,
            commitment_seconds=2.0,
            commitment_cooldown_seconds=1.0,
        )
        planner.current_action = 0
        planner.phase = "seeking_target"
        state = {
            "red_food_visible": True,
            "red_food_distance": 5.0,
            "red_food_world_x": 1.0,
            "red_food_world_z": 0.0,
        }
        planner.update_guidance(state)
        self.assertTrue(planner.guidance_active)
        self.assertEqual(planner.commitment_starts, 1)
        planner.update_guidance(state)
        self.assertEqual(planner.commitment_expirations, 1)
        planner.update_guidance(state)
        self.assertFalse(planner.guidance_active)
        planner.update_guidance(state)
        self.assertTrue(planner.guidance_active)
        self.assertEqual(planner.commitment_starts, 2)

    def test_observation_isolation_points_away_from_visible_food(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed", hz=1.0, seed=3
        )
        planner.phase = "observing_delay"
        planner.update_guidance(
            {
                "red_food_visible": False,
                "blue_food_visible": True,
                "blue_food_distance": 2.0,
                "blue_food_world_x": 1.0,
                "blue_food_world_z": 0.0,
            }
        )
        self.assertTrue(planner.isolation_active)
        self.assertEqual(planner.guidance_vector, (-1.0, 0.0))

    def test_observation_isolation_also_repels_visible_yellow(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed", hz=1.0, seed=3
        )
        planner.phase = "observing_delay"
        planner.update_guidance(
            {
                "yellow_food_visible": True,
                "yellow_food_distance": 2.0,
                "yellow_food_world_x": 0.0,
                "yellow_food_world_z": 1.0,
            }
        )
        self.assertTrue(planner.isolation_active)
        self.assertEqual(planner.guidance_vector, (0.0, -1.0))

    def test_typed_guidance_requests_red_then_yellow_without_answer_prior(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed", hz=1.0, typed_learner=learner
        )
        self.assertEqual(planner.requested_experiment, "red_then_yellow")
        planner.update(0, 0.0, 0, 0, 0, 0)
        planner.update_guidance(
            {
                "red_food_visible": True,
                "red_food_distance": 4.0,
                "red_food_world_x": 1.0,
                "red_food_world_z": 0.0,
            }
        )
        self.assertEqual(planner.phase, "seeking_red")
        self.assertEqual(planner.guidance_vector, (1.0, 0.0))
        learner.observe_pickups(1, red=1)
        planner.update(1, 0.0, 1, 1, 0, 0)
        planner.update_guidance(
            {
                "yellow_food_visible": True,
                "yellow_food_distance": 3.0,
                "yellow_food_world_x": 0.0,
                "yellow_food_world_z": 1.0,
            }
        )
        self.assertEqual(planner.phase, "seeking_second")
        self.assertEqual(planner.guidance_vector, (0.0, 1.0))

    def test_typed_guidance_isolates_after_second_pickup(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed", hz=1.0, typed_learner=learner
        )
        learner.observe_pickups(1, red=1)
        learner.observe_pickups(2, yellow=1, can_start=False)
        planner.update(2, 0.0, 2, 1, 0, 1)
        planner.update_guidance(
            {
                "blue_food_visible": True,
                "blue_food_distance": 2.0,
                "blue_food_world_x": 1.0,
                "blue_food_world_z": 0.0,
            }
        )
        self.assertEqual(planner.phase, "observing_delay")
        self.assertTrue(planner.isolation_active)
        self.assertEqual(planner.guidance_vector, (-1.0, 0.0))

    def test_isolation_counts_intervening_pickup(self):
        planner = self.planner("committed", seed=1)
        planner.current_action = 0
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0)
        planner.update(1, 0.0, 1, 1)
        planner.update(2, 0.0, 2, 1)
        self.assertEqual(planner.isolation_intervening_pickups, 1)
        planner.update_guidance(
            {
                "blue_food_visible": True,
                "blue_food_distance": 2.0,
                "blue_food_world_x": 1.0,
                "blue_food_world_z": 0.0,
            }
        )
        self.assertFalse(planner.isolation_active)

    def test_verified_protective_rule_is_inert_until_red_is_pending(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        learner.pool.posterior = np.array([0.97, 0.01, 0.01, 0.01])
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )
        yellow = {
            "yellow_food_visible": True,
            "yellow_food_distance": 3.0,
            "yellow_food_world_x": 0.0,
            "yellow_food_world_z": 1.0,
        }
        planner.update_guidance(yellow)
        self.assertFalse(planner.guidance_active)
        learner.observe_pickups(1, red=1)
        planner.update(1, 0.0, 1, 1, 0, 0)
        planner.update_guidance(yellow)
        self.assertTrue(planner.guidance_active)
        self.assertTrue(planner.protective_rule_active)
        self.assertEqual(planner.protective_target_feature, "yellow")
        self.assertEqual(planner.guidance_vector, (0.0, 1.0))

    def test_verified_protective_rule_can_recall_unseen_yellow_location(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        learner.pool.posterior = np.array([0.97, 0.01, 0.01, 0.01])
        learner.observe_pickups(1, red=1)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )
        memory = SimpleNamespace(
            active=True,
            active_feature="yellow",
            target=(30.0, 40.0),
            guidance_vector=(0.6, 0.8),
            distance=50.0,
        )

        planner.update_guidance(
            {"yellow_food_visible": False}, resource_memory=memory
        )

        self.assertTrue(planner.protective_rule_active)
        self.assertTrue(planner.protective_memory_active)
        self.assertTrue(planner.guidance_active)
        self.assertEqual(planner.protective_target_feature, "yellow")
        self.assertEqual(planner.guidance_vector, (0.6, 0.8))

    def test_verified_protection_survives_irrelevant_blue_intervention(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        learner.pool.posterior = np.array([0.97, 0.01, 0.01, 0.01])
        learner.observe_pickups(1, red=1)
        learner.observe_pickups(2, blue=1, can_start=False)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )
        planner.phase = "observing_delay"
        memory = SimpleNamespace(
            active=True,
            active_feature="yellow",
            target=(30.0, 40.0),
            guidance_vector=(0.6, 0.8),
            distance=50.0,
        )

        planner.update_guidance(
            {"yellow_food_visible": False},
            resource_memory=memory,
            protective_need_active=True,
        )

        self.assertTrue(planner.protective_rule_active)
        self.assertTrue(planner.protective_need_active)
        self.assertTrue(planner.protective_memory_active)
        self.assertTrue(planner.guidance_active)
        self.assertFalse(planner.isolation_active)

    def test_verified_protection_stops_when_metabolic_need_is_resolved(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        learner.pool.posterior = np.array([0.97, 0.01, 0.01, 0.01])
        learner.observe_pickups(1, red=1)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )

        planner.update_guidance(
            {"yellow_food_visible": True}, protective_need_active=False
        )

        self.assertFalse(planner.protective_rule_active)

    def test_passive_multi_hypothesis_arbitration_has_zero_authority(self):
        learner = PassiveOrderedEpisodeLearner(enabled=True)
        learner.pool.posterior = np.asarray(
            [0.9763185, 0.0000204, 0.0207773, 0.0028838]
        )
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
            multi_hypothesis_arbitration="passive",
            arbitration_hazard_cost=0.25,
        )
        memory = SimpleNamespace(
            enabled=True,
            active=True,
            active_feature="yellow",
            target=(30.0, 40.0),
            guidance_vector=(0.6, 0.8),
            distance=50.0,
        )
        entries = {
            "yellow": SimpleNamespace(confidence=0.60),
            "blue": SimpleNamespace(confidence=0.80),
        }
        memory.best_typed_region = lambda feature, _x, _z, **_options: (
            (0.0, -36.0, (feature, 0, 0), entries[feature])
            if feature == "yellow"
            else (0.0, -12.0, (feature, 0, 0), entries[feature])
        )
        baseline = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )
        baseline.update_guidance(
            {"x": 0.0, "z": 0.0, "yellow_food_visible": False},
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
        )

        planner.update_guidance(
            {"x": 0.0, "z": 0.0, "yellow_food_visible": False},
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
        )

        audit = planner.audit()
        self.assertTrue(audit["arbitration_active"])
        self.assertEqual(audit["arbitration_selected_feature"], "yellow")
        self.assertEqual(audit["arbitration_authority"], 0.0)
        self.assertEqual(audit["arbitration_action_influence"], 0)
        self.assertTrue(planner.guidance_active)
        self.assertEqual(planner.guidance_vector, (0.6, 0.8))
        self.assertEqual(planner.guidance_vector, baseline.guidance_vector)
        self.assertEqual(planner.guidance_weight, baseline.guidance_weight)

    def test_bounded_arbitration_requires_and_uses_verified_agreement(self):
        learner = PassiveOrderedEpisodeLearner(enabled=True)
        learner.pool.posterior = np.asarray(
            [0.9763185, 0.0000204, 0.0207773, 0.0028838]
        )
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
            multi_hypothesis_arbitration="bounded_verified",
            arbitration_hazard_cost=0.25,
        )
        memory = SimpleNamespace(
            enabled=True,
            active=True,
            active_feature="yellow",
            target=(30.0, 40.0),
            guidance_vector=(0.6, 0.8),
            distance=50.0,
        )
        entries = {
            "yellow": SimpleNamespace(confidence=0.60),
            "blue": SimpleNamespace(confidence=0.60),
        }
        memory.best_typed_region = lambda feature, _x, _z, **_options: (
            (0.0, -36.0, (feature, 0, 0), entries[feature])
            if feature == "yellow"
            else (0.0, -12.0, (feature, 0, 0), entries[feature])
        )

        planner.update_guidance(
            {"x": 0.0, "z": 0.0, "yellow_food_visible": False},
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
        )

        audit = planner.audit()
        self.assertEqual(audit["arbitration_selected_feature"], "yellow")
        self.assertEqual(audit["arbitration_authority"], 1.0)
        self.assertEqual(audit["arbitration_denial_reason"], "none")
        self.assertEqual(audit["arbitration_control_frames"], 1)
        self.assertGreater(audit["arbitration_score_margin"], 0.02)
        self.assertGreater(
            audit["arbitration_suppression_probability"], 0.50
        )
        self.assertEqual(planner.protective_target_feature, "yellow")

    def test_dual_verified_conflict_selects_blue_then_returns_to_yellow(self):
        protective = PassiveOrderedEpisodeLearner(enabled=True)
        protective.pool.posterior = np.asarray(
            [0.9763185, 0.0000204, 0.0207773, 0.0028838]
        )
        metabolic = HeldOutMetabolicRoleLearner()
        for feature, relieved in (
            ("red", False), ("yellow", False), ("blue", True),
            ("blue", True), ("yellow", False), ("blue", True),
        ):
            metabolic.observe(feature, relieved)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=protective,
            typed_rule_control="verified_protective",
            multi_hypothesis_arbitration="bounded_dual_verified",
            arbitration_hazard_cost=0.25,
            metabolic_learner=metabolic,
        )
        entries = {
            "yellow": SimpleNamespace(confidence=0.80, x=20.0, z=0.0),
            "blue": SimpleNamespace(confidence=0.80, x=0.0, z=20.0),
        }
        memory = SimpleNamespace(
            enabled=True, active=True, active_feature="yellow",
            target=(20.0, 0.0), guidance_vector=(1.0, 0.0), distance=20.0,
        )
        memory.best_typed_region = lambda feature, _x, _z, **_options: (
            0.0, -20.0, (feature, 0, 0), entries[feature]
        )
        state = {"x": 0.0, "z": 0.0}

        planner.update_guidance(
            state,
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
            metabolic_need_urgency=0.80,
        )
        self.assertEqual(planner.arbitration_selected_feature, "blue")
        self.assertEqual(planner.arbitration_authority, 1.0)
        self.assertEqual(planner.protective_target_feature, "blue")
        self.assertEqual(planner.guidance_vector, (0.0, 1.0))

        planner.update_guidance(
            state,
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=220.0,
            metabolic_need_urgency=0.0,
        )
        self.assertFalse(planner.arbitration_active)
        self.assertEqual(planner.protective_target_feature, "yellow")
        self.assertEqual(planner.guidance_vector, (1.0, 0.0))

    def test_visible_blue_remains_candidate_inside_memory_arrival_radius(self):
        protective = PassiveOrderedEpisodeLearner(enabled=True)
        protective.pool.posterior = np.asarray(
            [0.9763185, 0.0000204, 0.0207773, 0.0028838]
        )
        metabolic = HeldOutMetabolicRoleLearner()
        for feature, relieved in (
            ("red", False), ("yellow", False), ("blue", True),
            ("blue", True), ("yellow", False), ("blue", True),
        ):
            metabolic.observe(feature, relieved)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=protective,
            typed_rule_control="verified_protective",
            multi_hypothesis_arbitration="bounded_dual_verified",
            arbitration_hazard_cost=0.25,
            metabolic_learner=metabolic,
        )
        entries = {
            "yellow": SimpleNamespace(confidence=0.80, x=20.0, z=0.0),
            "blue": SimpleNamespace(confidence=0.80, x=2.0, z=0.0),
        }
        memory = SimpleNamespace(enabled=True)

        def best_typed(feature, _x, _z, retain_arrived=False):
            if feature == "blue" and not retain_arrived:
                return None
            distance = 2.0 if feature == "blue" else 20.0
            return 0.0, -distance, (feature, 0, 0), entries[feature]

        memory.best_typed_region = best_typed
        planner.update_guidance(
            {
                "x": 0.0,
                "z": 0.0,
                "blue_food_visible": True,
            },
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
            metabolic_need_urgency=0.80,
        )

        self.assertEqual(planner.arbitration_selected_feature, "blue")
        self.assertEqual(planner.arbitration_authority, 1.0)
        self.assertEqual(planner.protective_target_feature, "blue")


    def test_bounded_arbitration_abstains_without_verified_production(self):
        learner = PassiveOrderedEpisodeLearner(enabled=True)
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
            multi_hypothesis_arbitration="bounded_verified",
        )
        entries = {
            "yellow": SimpleNamespace(confidence=0.60),
            "blue": SimpleNamespace(confidence=0.60),
        }
        memory = SimpleNamespace(enabled=True)
        memory.best_typed_region = lambda feature, _x, _z, **_options: (
            0.0, -12.0, (feature, 0, 0), entries[feature]
        )

        planner.update_guidance(
            {"x": 0.0, "z": 0.0},
            resource_memory=memory,
            protective_need_active=True,
            protective_deadline_remaining_seconds=240.0,
        )

        audit = planner.audit()
        self.assertTrue(audit["arbitration_active"])
        self.assertEqual(audit["arbitration_authority"], 0.0)
        self.assertEqual(
            audit["arbitration_denial_reason"], "unverified_production"
        )
        self.assertFalse(planner.guidance_active)

    def test_unverified_or_nonspecific_rule_cannot_gain_protective_authority(self):
        learner = PassiveOrderedEpisodeLearner(
            hz=1.0, delay_seconds=10.0, enabled=True
        )
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=1.0,
            typed_learner=learner,
            typed_rule_control="verified_protective",
        )
        learner.observe_pickups(1, red=1)
        state = {
            "yellow_food_visible": True,
            "yellow_food_distance": 3.0,
            "yellow_food_world_x": 0.0,
            "yellow_food_world_z": 1.0,
        }
        learner.pool.posterior = np.array([0.94, 0.02, 0.02, 0.02])
        planner.update_guidance(state)
        self.assertFalse(planner.guidance_active)
        learner.pool.posterior = np.array([0.01, 0.01, 0.97, 0.01])
        planner.update_guidance(state)
        self.assertFalse(planner.guidance_active)

    def test_red_pickup_arms_brief_stronger_isolation_retreat(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="committed",
            hz=5.0,
            seed=1,
            isolation_retreat_seconds=2.0,
            isolation_retreat_max_score_regret=0.35,
        )
        planner.current_action = 0
        planner.phase = "seeking_target"
        planner.update(0, 0.0, 0, 0)
        planner.update(1, 0.0, 1, 1)
        self.assertEqual(planner.isolation_retreat_remaining_ticks, 10)
        self.assertEqual(planner.isolation_retreat_max_score_regret, 0.35)

    def test_dynamic_mode_admits_async_l1_after_balanced_discovery(self):
        def proposer(_summary):
            return "L1 c red k blue e + t 2.0 q 0.5", {"accepted": True}

        planner = EmbodiedPGNWExperimentPlanner(
            mode="dynamic_committed",
            hz=1.0,
            seed=3,
            delay_seconds=2.0,
            hypothesis_proposer=proposer,
            dynamic_discovery_minimum=4,
        )
        planner.dynamic_observations.extend(
            [(0, True, 0.34), (1, False, 0.0), (2, False, 0.0), (1, False, 0.0)]
        )
        planner._poll_dynamic_proposal()
        for _ in range(50):
            planner._poll_dynamic_proposal()
            if planner.dynamic_pool.admission_count:
                break
            time.sleep(0.001)
        self.assertEqual(planner.dynamic_pool.admission_count, 1)
        self.assertEqual(planner.dynamic_proposal_status, "admitted_unverified")
        self.assertEqual(planner.dynamic_pool.held_out_updates, 0)
        self.assertEqual(
            planner.dynamic_pool.hypotheses[-1].admitted_after_observation, 4
        )

    def test_dynamic_discovery_cycles_actions_before_admission(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="dynamic_committed",
            hz=1.0,
            seed=3,
            hypothesis_proposer=lambda _summary: ("", {}),
        )
        selected = []
        for _ in range(3):
            selected.append(planner.current_action)
            planner.select_experiment()
        self.assertEqual(set(selected), {0, 1, 2})

    def test_dynamic_summary_has_frozen_canonical_feature_order(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="dynamic_committed",
            hz=1.0,
            hypothesis_proposer=lambda _summary: ("", {}),
        )
        planner.dynamic_observations.extend([(1, False, 0.0), (0, True, 0.34)])
        self.assertEqual(
            list(planner._dynamic_summary()["feature_outcomes"]), ["red", "blue"]
        )
        self.assertEqual(
            planner._dynamic_summary()["feature_outcomes"]["blue"][
                "mean_pressure_delta"
            ],
            0.0,
        )

    def test_dynamic_rejection_is_terminal(self):
        calls = []

        def rejected(_summary):
            calls.append(1)
            raise ValueError("deliberate_rejection")

        planner = EmbodiedPGNWExperimentPlanner(
            mode="dynamic_committed",
            hz=1.0,
            hypothesis_proposer=rejected,
            dynamic_discovery_minimum=2,
        )
        planner.dynamic_observations.extend([(1, False, 0.0), (0, True, 0.34)])
        planner._poll_dynamic_proposal()
        for _ in range(50):
            planner._poll_dynamic_proposal()
            if planner.dynamic_proposal_status == "rejected":
                break
            time.sleep(0.001)
        for _ in range(5):
            planner._poll_dynamic_proposal()
        self.assertEqual(planner.dynamic_proposal_status, "rejected")
        self.assertEqual(len(calls), 1)

    def test_dynamic_status_reports_held_out_verification(self):
        planner = EmbodiedPGNWExperimentPlanner(
            mode="dynamic_committed",
            hz=1.0,
            hypothesis_proposer=lambda _summary: ("", {}),
        )
        planner.dynamic_observations.extend(
            [(2, False, 0.0), (0, True, 0.34), (2, False, 0.0), (1, False, 0.0)]
        )
        from dynamic_hypothesis_pool import compile_l1_hypothesis

        summary = planner._dynamic_summary()
        planner.dynamic_pool.admit(
            compile_l1_hypothesis(
                "L1 c red k blue e + t 10.0 q 0.5", summary, 4
            )
        )
        planner.dynamic_proposal_status = "admitted_unverified"
        # Exercise the same state transition made after live held-out updates.
        for index in range(5, 13):
            planner.dynamic_pool.update(0, True, index)
        if planner.dynamic_pool.map_confidence >= 0.95:
            planner.dynamic_proposal_status = "verified_held_out"
        self.assertEqual(planner.dynamic_proposal_status, "verified_held_out")


if __name__ == "__main__":
    unittest.main()
