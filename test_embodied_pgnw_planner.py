import unittest
import time

import numpy as np

from embodied_pgnw_planner import EmbodiedPGNWExperimentPlanner
from pgnw_hypothesis_selection_lab import HYPOTHESES


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
