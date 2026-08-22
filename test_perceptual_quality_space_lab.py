import unittest

import numpy as np
import torch

from perceptual_quality_space_lab import (
    INPUT_DIM,
    STEPS,
    PerceptualModel,
    distance_correlation,
    generate_episodes,
)


class PerceptualQualitySpaceLabTests(unittest.TestCase):
    def test_generator_shapes_and_answer_is_not_in_observation(self):
        batch = generate_episodes(12, 16)
        self.assertEqual(batch.observations.shape, (16, STEPS, INPUT_DIM))
        self.assertEqual(batch.qualities.shape, (16, 8))
        self.assertEqual(set(batch.choices.tolist()) - {0, 1}, set())
        self.assertFalse(torch.equal(batch.choices.float(), batch.qualities[:, 0]))

    def test_final_frame_hides_all_quality_channels(self):
        batch = generate_episodes(13, 8)
        final = batch.observations[:, -1]
        for offset in (0, 9):
            self.assertTrue(torch.all(final[:, offset + 1 : offset + 9] == 0))

    def test_all_conditions_return_matched_outputs(self):
        batch = generate_episodes(14, 7)
        for recurrent in (False, True):
            hidden = 40 if recurrent else 59
            model = PerceptualModel(recurrent=recurrent, hot4=True, hidden=hidden)
            logits, code, reconstruction, state = model(batch.observations, batch.needs)
            self.assertEqual(logits.shape, (7, 2))
            self.assertEqual(code.shape, (7, 12))
            self.assertEqual(reconstruction.shape, (7, 8))
            self.assertEqual(state.shape, (7, hidden))

    def test_flat_and_recurrent_parameter_counts_are_matched(self):
        flat = PerceptualModel(recurrent=False, hot4=False, hidden=59)
        recurrent = PerceptualModel(recurrent=True, hot4=False, hidden=40)
        flat_count = sum(parameter.numel() for parameter in flat.parameters())
        recurrent_count = sum(parameter.numel() for parameter in recurrent.parameters())
        self.assertLess(abs(flat_count - recurrent_count) / recurrent_count, 0.01)

    def test_distance_correlation_detects_geometry_and_permutation(self):
        rng = np.random.default_rng(15)
        quality = rng.normal(size=(256, 8))
        aligned = distance_correlation(quality, quality, seed=3)
        permuted = distance_correlation(quality[rng.permutation(len(quality))], quality, seed=3)
        self.assertGreater(aligned, 0.99)
        self.assertLess(abs(permuted), 0.15)


if __name__ == "__main__":
    unittest.main()
