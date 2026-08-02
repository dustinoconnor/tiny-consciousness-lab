import unittest

from terrain_escape_teacher import TerrainEscapeTeacher, widest_safe_action


class FakeRouteController:
    def __init__(self):
        self.routes = []

    def add_teacher_route(self, prototype, waypoints, metadata):
        self.routes.append((prototype, waypoints, metadata))
        return f"teacher_exit_{len(self.routes) - 1:04d}"


def body(x=0.0, z=0.0, collision=False):
    return {
        "x": x,
        "z": z,
        "yaw": 0.0,
        "horizontal_collision": collision,
    }


class EscapeTeacherTests(unittest.TestCase):
    def test_widest_safe_direction_wins(self):
        rays = [0.2, 0.4, 0.9, 0.6]
        clearance = [1.0, 1.0, 0.8, 1.0]
        self.assertEqual(widest_safe_action(rays, clearance), 2)

    def test_unsafe_widest_direction_is_excluded(self):
        rays = [0.2, 0.99, 0.8, 0.6]
        clearance = [1.0, 0.2, 0.9, 1.0]
        self.assertEqual(widest_safe_action(rays, clearance), 2)

    def test_successful_exit_is_recorded(self):
        controller = FakeRouteController()
        teacher = TerrainEscapeTeacher(
            hz=1,
            min_commit_seconds=2,
            max_commit_seconds=4,
            min_displacement=2.0,
            min_efficiency=0.8,
        )
        actions = ["up", "right", "down", "left"]
        rays = [0.2, 0.9, 0.3, 0.4]
        clearance = [1.0] * 4
        self.assertTrue(
            teacher.start(body(), [0.5] * 22, actions, rays, clearance)
        )
        self.assertEqual(
            teacher.update(body(1.0, 0.0), actions, rays, clearance, controller),
            1,
        )
        self.assertIsNone(
            teacher.update(body(2.2, 0.0), actions, rays, clearance, controller)
        )
        self.assertEqual(teacher.successes, 1)
        self.assertEqual(len(controller.routes), 1)
        self.assertGreaterEqual(len(controller.routes[0][1]), 2)

    def test_timeout_is_not_recorded(self):
        controller = FakeRouteController()
        teacher = TerrainEscapeTeacher(
            hz=1,
            min_commit_seconds=2,
            max_commit_seconds=2,
            min_displacement=2.0,
        )
        actions = ["up", "right", "down", "left"]
        rays = [0.9, 0.2, 0.2, 0.2]
        clearance = [1.0] * 4
        teacher.start(body(), [0.5] * 22, actions, rays, clearance)
        teacher.update(body(0.1, 0.0), actions, rays, clearance, controller)
        self.assertIsNone(
            teacher.update(body(0.2, 0.0), actions, rays, clearance, controller)
        )
        self.assertEqual(teacher.failures, 1)
        self.assertEqual(controller.routes, [])


if __name__ == "__main__":
    unittest.main()
