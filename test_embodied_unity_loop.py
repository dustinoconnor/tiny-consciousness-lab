import unittest

from embodied_unity_loop import (
    EmbodiedAdaptiveResonanceObserver,
    EmbodiedDynamicsObserver,
    EmbodiedFunctionalEgo,
    art_route_selection_score,
    art_route_context,
    conductor_gate_allows,
    fuzzy_art_similarity,
    orbit_teacher_due,
    orbit_recovery_should_finish,
    route_memory_adapter,
    route_reversal_count,
    route_waypoint_radius,
    smooth_target_intercept,
    stable_recovery_due,
    trajectory_orbit_metrics,
)


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

    def test_red_pickup_has_delayed_not_immediate_internal_effect(self):
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
        self.assertEqual(ego.metabolic_challenge_events, 0)
        self.assertEqual(len(ego.metabolic_challenge_due_steps), 1)
        for _ in range(ego.metabolic_challenge_delay_ticks):
            ego.steps += 1
            ego.update_from_body(
                self.body(
                    mushroom_pickups_total=1,
                    mushroom_reward_total=0.35,
                    red_mushroom_pickups_total=1,
                )
            )
        self.assertEqual(ego.metabolic_challenge_events, 1)
        self.assertGreater(ego.metabolic_pressure, 0.30)

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
        self.assertFalse(ego.metabolic_challenge_due_steps)


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


class EmbodiedDynamicsObserverTests(unittest.TestCase):
    def test_simultaneous_module_events_have_high_coherence(self):
        observer = EmbodiedDynamicsObserver(5.0)
        observer.update([0.8] * 6, 0.0)
        self.assertGreater(observer.coherence, 0.95)
        self.assertEqual(observer.active_modules, 6)

    def test_observer_is_telemetry_only_and_recommends_bounded_gain(self):
        observer = EmbodiedDynamicsObserver(5.0)
        for index in range(12):
            values = [0.9 if module == index % 6 else 0.1 for module in range(6)]
            observer.update(values, 1.0)
        self.assertGreaterEqual(observer.recommended_gain, 1.16)
        self.assertLessEqual(observer.recommended_gain, 1.33)
        self.assertIn(observer.criticality_regime, {
            "subcritical_proxy", "near_critical_proxy", "supercritical_proxy"
        })


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
