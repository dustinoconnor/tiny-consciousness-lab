import unittest

import numpy as np

from resource_memory_lab import (
    FAILED_MEMORY_SUPPRESSION_STEPS,
    MEMORY_REFRACTORY_STEPS,
    ResourceMemory,
    shuffled_target,
)


class ResourceMemoryTests(unittest.TestCase):
    def test_rewarded_region_becomes_recallable_after_refractory_period(self):
        memory = ResourceMemory()
        memory.record_reward(np.asarray([8.0, 12.0]), step=10)
        self.assertIsNone(memory.recall(np.asarray([20.0, 20.0]), step=20))
        recalled = memory.recall(
            np.asarray([20.0, 20.0]),
            step=10 + MEMORY_REFRACTORY_STEPS,
        )
        self.assertIsNotNone(recalled)
        _key, position = recalled
        np.testing.assert_allclose(position, [8.0, 12.0])

    def test_failed_recall_is_temporarily_suppressed(self):
        memory = ResourceMemory()
        position = np.asarray([8.0, 12.0])
        memory.record_reward(position, step=0)
        key = memory.cell(position)
        memory.record_failure(key, step=100)
        self.assertIsNone(memory.recall(np.asarray([20.0, 20.0]), step=101))
        recalled = memory.recall(
            np.asarray([20.0, 20.0]),
            step=100 + FAILED_MEMORY_SUPPRESSION_STEPS,
        )
        self.assertIsNotNone(recalled)

    def test_shuffled_control_changes_spatial_correspondence(self):
        original = np.asarray([8.0, 12.0])
        self.assertFalse(np.allclose(original, shuffled_target(original)))


if __name__ == "__main__":
    unittest.main()
