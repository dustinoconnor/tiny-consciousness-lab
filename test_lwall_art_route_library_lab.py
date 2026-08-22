import unittest

from lwall_art_route_library_lab import (
    route_signature,
    select_diverse_routes,
    trajectory_metrics,
)


def row(x, z):
    return {"position": [x, 0.0, z]}


class RouteQualityTests(unittest.TestCase):
    def test_straight_route_has_less_waste_and_jerk_than_zigzag(self):
        straight = [row(0, 0), row(1, 0), row(2, 0), row(3, 0)]
        zigzag = [row(0, 0), row(1, 1), row(2, -1), row(3, 0)]

        straight_metrics = trajectory_metrics(straight)
        zigzag_metrics = trajectory_metrics(zigzag)

        self.assertLess(straight_metrics["path_waste"], zigzag_metrics["path_waste"])
        self.assertLess(
            straight_metrics["steering_jerk"],
            zigzag_metrics["steering_jerk"],
        )

    def test_selector_preserves_geometrically_distinct_routes(self):
        routes = [
            {
                "route_id": "cheap-right",
                "cost": 1.0,
                "quality": 1.0,
                "signature": route_signature([[1, 0], [2, 0], [3, 0]]).tolist(),
            },
            {
                "route_id": "duplicate-right",
                "cost": 1.1,
                "quality": 0.95,
                "signature": route_signature([[1, 0.05], [2, 0.05], [3, 0]]).tolist(),
            },
            {
                "route_id": "clean-left",
                "cost": 1.5,
                "quality": 0.90,
                "signature": route_signature([[-1, 0], [-2, 0], [-3, 0]]).tolist(),
            },
        ]

        selected = select_diverse_routes(routes, count=2, minimum_distance=1.0)

        self.assertEqual(
            [route["route_id"] for route in selected],
            ["cheap-right", "clean-left"],
        )

    def test_selector_rejects_diverse_but_low_quality_route(self):
        routes = [
            {
                "route_id": "clean",
                "cost": 1.0,
                "quality": 1.0,
                "signature": route_signature([[1, 0], [2, 0], [3, 0]]).tolist(),
            },
            {
                "route_id": "different-but-poor",
                "cost": 5.0,
                "quality": 0.40,
                "signature": route_signature([[-1, 0], [-2, 0], [-3, 0]]).tolist(),
            },
        ]

        selected = select_diverse_routes(
            routes,
            count=2,
            minimum_distance=1.0,
            minimum_relative_quality=0.65,
        )

        self.assertEqual([route["route_id"] for route in selected], ["clean"])


if __name__ == "__main__":
    unittest.main()
