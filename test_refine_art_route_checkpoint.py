import unittest

from refine_art_route_checkpoint import prune_reversal_waypoints


class RefineArtRouteCheckpointTests(unittest.TestCase):
    def test_prunes_backtracking_point_and_preserves_endpoints(self):
        route = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        refined, removed = prune_reversal_waypoints(route)
        self.assertEqual(removed, 1)
        self.assertEqual(refined[0], route[0])
        self.assertEqual(refined[-1], route[-1])


if __name__ == "__main__":
    unittest.main()
