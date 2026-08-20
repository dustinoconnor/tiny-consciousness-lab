import unittest

import numpy as np

from unity_geometry_navigation_lab import UnityGeometryCourse


class UnityGeometryCourseTests(unittest.TestCase):
    def test_hidden_goals_match_unity_course_start(self):
        for family in ("utrap", "ctrap"):
            self.assertFalse(UnityGeometryCourse(family).food_visible())

    def test_teacher_moves_away_from_occluded_u_goal_first(self):
        env = UnityGeometryCourse("utrap")
        action = env.teacher_action()
        self.assertLess(np.dot(env.food - env.pos, np.asarray([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])[action]), 0)

    def test_teacher_completes_both_courses(self):
        for family in ("utrap", "ctrap"):
            env = UnityGeometryCourse(family)
            for _ in range(450):
                _, _, done, ate, _ = env.step(env.teacher_action())
                if done:
                    break
            self.assertTrue(ate, family)


if __name__ == "__main__":
    unittest.main()
