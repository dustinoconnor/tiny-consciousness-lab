import numpy as np

from route_bound_resource_memory_lab import (
    RouteResourceMemory,
    compress_approach,
    route_efficiency,
    shuffle_point,
)


def test_compress_approach_keeps_reward_endpoint():
    history = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([4.0, 0.0], dtype=np.float32),
        np.asarray([8.0, 0.0], dtype=np.float32),
        np.asarray([10.0, 0.0], dtype=np.float32),
    ]
    reward = np.asarray([10.4, 0.0], dtype=np.float32)

    route = compress_approach(history, reward)

    assert route
    assert np.allclose(route[-1], reward)


def test_route_efficiency_prefers_direct_approach():
    direct = [
        np.asarray([0.0, 0.0]),
        np.asarray([2.0, 0.0]),
        np.asarray([4.0, 0.0]),
    ]
    looping = [
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 2.0]),
        np.asarray([2.0, 2.0]),
        np.asarray([2.0, 0.0]),
        np.asarray([4.0, 0.0]),
    ]

    assert route_efficiency(direct) > route_efficiency(looping)


def test_route_memory_records_and_recalls_approach():
    memory = RouteResourceMemory()
    reward = np.asarray([20.0, 20.0], dtype=np.float32)
    history = [
        np.asarray([8.0, 20.0], dtype=np.float32),
        np.asarray([12.0, 20.0], dtype=np.float32),
        np.asarray([16.0, 20.0], dtype=np.float32),
        reward.copy(),
    ]
    memory.record_reward(reward, step=10, history=history)

    recalled = memory.recall(
        np.asarray([4.0, 20.0], dtype=np.float32),
        step=200,
    )

    assert recalled is not None
    _key, entry = recalled
    assert len(entry.approach) >= 2
    assert np.allclose(entry.approach[-1], reward)


def test_shuffled_control_moves_location_and_route_points():
    point = np.asarray([12.0, 23.0], dtype=np.float32)

    shuffled = shuffle_point(point)

    assert not np.allclose(shuffled, point)
    assert np.all(shuffled >= 0.0)
    assert np.all(shuffled < 64.0)
