import unittest

import numpy as np

from computational_invariance_lab import Observation, SymbolicEgo
from experience_structure_correspondence_lab import (
    distance_correlation,
    neighborhood_overlap,
    pairwise_distances,
    rankdata,
    vectors,
)


class ExperienceStructureCorrespondenceTests(unittest.TestCase):
    def test_distance_geometry_is_invariant_to_coordinate_permutation(self):
        rng = np.random.default_rng(2)
        matrix = rng.normal(size=(30, 5))
        permuted = matrix[:, [3, 0, 4, 1, 2]]
        self.assertAlmostEqual(distance_correlation(matrix, permuted), 1.0)

    def test_neighborhood_overlap_is_one_for_identical_spaces(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(24, 4))
        self.assertAlmostEqual(neighborhood_overlap(matrix, matrix, k=5), 1.0)

    def test_rankdata_assigns_average_tie_ranks(self):
        ranks = rankdata(np.array([3.0, 1.0, 1.0, 2.0]))
        self.assertTrue(np.allclose(ranks, [3.0, 0.5, 0.5, 2.0]))

    def test_report_vector_changes_with_forced_workspace(self):
        observation = Observation(1.0, 0.0, 0.0, 0.8, 0.0)
        baseline = SymbolicEgo().step(observation)
        forced = SymbolicEgo().step(observation, {"force_workspace": "danger"})
        baseline_report = vectors(observation, baseline).report
        forced_report = vectors(observation, forced).report
        self.assertFalse(np.allclose(baseline_report, forced_report))
        self.assertNotEqual(baseline.action, forced.action)


if __name__ == "__main__":
    unittest.main()
