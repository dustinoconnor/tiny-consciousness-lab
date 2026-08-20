import math
import unittest

from back_goal_navigation_lab import topological_oracle_action
from unity_posttraining_lab import UnityContinuousCourse
from upgraded_foraging_pipeline import MOVES, generate_layout, reachable


class BackGoalGeometryTests(unittest.TestCase):
    def test_randomized_layouts_are_reachable_and_initially_occluded(self):
        for seed in range(12):
            blocked, start, food = generate_layout("c_back_goal", 20_000 + seed)
            self.assertTrue(reachable(blocked, start, food))
            env = UnityContinuousCourse("c_back_goal", 20_000 + seed)
            self.assertFalse(env.food_visible())

    def test_oracle_initially_moves_away_from_hidden_goal(self):
        for seed in range(12):
            env = UnityContinuousCourse("c_back_goal", 30_000 + seed)
            old_distance = float(math.dist(env.pos, env.food))
            action = topological_oracle_action(env)
            direction = MOVES[action].astype(float)
            direction /= math.hypot(direction[0], direction[1])
            candidate = env.pos + direction * 0.44
            self.assertGreater(float(math.dist(candidate, env.food)), old_distance)


if __name__ == "__main__":
    unittest.main()
