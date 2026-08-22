import numpy as np

from primed_resource_memory_lab import (
    PrimedResourceMemory,
    final_visible_anchor,
)


def test_final_visible_anchor_uses_last_uninterrupted_visible_run():
    history = [
        (np.asarray([0.0, 0.0]), False),
        (np.asarray([1.0, 0.0]), True),
        (np.asarray([2.0, 0.0]), False),
        (np.asarray([3.0, 0.0]), True),
        (np.asarray([4.0, 0.0]), True),
    ]

    anchor = final_visible_anchor(history)

    assert np.allclose(anchor, [3.0, 0.0])


def test_penalized_precedent_is_temporarily_suppressed():
    memory = PrimedResourceMemory()
    memory.record([20.0, 20.0], [16.0, 20.0])
    entry = memory.entries[0]
    memory.penalize(entry, step=100)

    assert memory.recall(np.asarray([2.0, 2.0]), step=101) is None
    assert memory.recall(
        np.asarray([2.0, 2.0]),
        step=entry.suppressed_until,
    ) is entry


def test_clear_models_memory_reset_control():
    memory = PrimedResourceMemory()
    memory.record([20.0, 20.0], [16.0, 20.0])

    memory.clear()

    assert memory.entries == []
    assert memory.recall(np.asarray([2.0, 2.0]), step=1000) is None
