import json
import tempfile
import unittest
from pathlib import Path

from terrain_air_route_controller import (
    TerrainAirRouteController,
    intervention_reason,
    local_to_world,
    score_margin_ambiguity,
)


ACTIONS = ["up", "right", "down", "left"]
VECTORS = {
    "up": (0.0, 1.0),
    "right": (1.0, 0.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
}


def body(food_visible=False):
    return {
        "x": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "food_visible": food_visible,
        "directional_rays": [1.0] * 8,
        "directional_body_clearance": [1.0] * 8,
    }


class TerrainAirRouteControllerTests(unittest.TestCase):
    def checkpoint(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {
                "format": "terrain_air_art_route_library_v1",
                "art_vigilance": 0.75,
                "routes": [
                    {
                        "episode_id": 7,
                        "prototype": [1.0] * 16
                        + [0.5, 0.0, 0.2, 0.25, 0.0, 0.0],
                        "waypoints": [[0.0, 2.0], [0.0, 4.0]],
                    }
                ],
            },
            handle,
        )
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_large_area_orbit_is_independently_necessary(self):
        self.assertEqual(
            intervention_reason(0.0, 0.0, 5.0, 0.2),
            "low_efficiency_orbit",
        )

    def test_local_route_rotates_into_world_heading(self):
        self.assertAlmostEqual(local_to_world((0.0, 2.0), 90.0)[0], 2.0)

    def test_score_margin_only_allows_ambiguous_guidance(self):
        self.assertEqual(score_margin_ambiguity(0.05, 0.05), 0.0)
        self.assertEqual(score_margin_ambiguity(0.0, 0.05), 1.0)
        self.assertAlmostEqual(score_margin_ambiguity(0.025, 0.05), 0.5)

    def test_passive_mode_recommends_without_action_influence(self):
        controller = TerrainAirRouteController(self.checkpoint(), "passive", hz=5)
        selected = controller.update(
            body(),
            0.5,
            5.0,
            0.2,
            0.0,
            0.0,
            ACTIONS,
            VECTORS,
            [1.0] * 4,
        )
        self.assertIsNone(selected)
        self.assertEqual(controller.recommendations, 1)
        self.assertEqual(controller.action_influence, 0)

    def test_bounded_mode_controls_then_times_out(self):
        controller = TerrainAirRouteController(
            self.checkpoint(), "bounded", hz=1, max_control_seconds=1
        )
        selected = controller.update(
            body(),
            0.5,
            5.0,
            0.2,
            0.0,
            0.0,
            ACTIONS,
            VECTORS,
            [1.0] * 4,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(controller.action_influence, 1)
        selected = controller.update(
            body(),
            0.5,
            5.0,
            0.2,
            0.0,
            0.0,
            ACTIONS,
            VECTORS,
            [1.0] * 4,
        )
        self.assertIsNone(selected)
        self.assertFalse(controller.active)
        self.assertEqual(controller.release_reason, "bounded_timeout")

    def test_guided_mode_supplies_prior_without_motor_action(self):
        controller = TerrainAirRouteController(self.checkpoint(), "guided", hz=5)
        selected = controller.update(
            body(),
            0.5,
            5.0,
            0.2,
            0.0,
            0.0,
            ACTIONS,
            VECTORS,
            [1.0] * 4,
        )
        self.assertIsNone(selected)
        self.assertTrue(controller.active)
        self.assertGreater(controller.guidance_weight, 0.0)
        self.assertEqual(controller.guidance_vector, (0.0, 1.0))
        self.assertEqual(controller.action_influence, 0)
        controller.last_effective_guidance_weight = 0.03
        controller.reset("test")
        self.assertEqual(controller.last_effective_guidance_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
