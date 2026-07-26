import math

import numpy as np

from terrain_air_art_route_library_lab import (
    RouteEpisode,
    evaluate_library,
    intervention_needed,
    local_offset,
    pickup_segments,
    route_context,
)


def row(step, pickups=0, orbit_path=0.0, orbit_efficiency=1.0):
    return {
        "time": float(step),
        "step": step,
        "position": [float(step), 0.0, 0.0],
        "yaw": 90.0,
        "rays": [1.0] * 8,
        "body_clearance": [1.0] * 8,
        "hunger": 0.5,
        "food_visible": False,
        "orbit_path": orbit_path,
        "orbit_efficiency": orbit_efficiency,
        "physics_wedge_seconds": 0.0,
        "trap_accumulation_seconds": 0.0,
        "mushroom_pickups_total": pickups,
    }


def episode(identifier, context, heading):
    endpoint = [math.sin(heading), math.cos(heading)]
    return RouteEpisode(
        episode_id=identifier,
        start_index=0,
        trigger_index=1,
        pickup_index=2,
        context=np.asarray(context, dtype=float),
        waypoints=[endpoint, [2.0 * endpoint[0], 2.0 * endpoint[1]]],
        signature=np.zeros((24, 2)),
        air_score=1.0,
        duration=2.0,
        collisions=0,
        orbit_frames=1,
        path_length=2.0,
        displacement=2.0,
    )


def test_pickup_segments_and_necessity_gate():
    rows = [row(0), row(1, orbit_path=4.5, orbit_efficiency=0.2), row(2, 1)]
    segments = pickup_segments(rows)
    assert len(segments) == 1
    assert intervention_needed(rows[1])
    assert not intervention_needed(rows[0])


def test_context_is_fixed_width_and_grounded():
    context = route_context(row(0))
    assert context.shape == (22,)
    assert np.all((context >= 0.0) & (context <= 1.0))


def test_local_offset_uses_egocentric_heading():
    local = local_offset(np.array([0.0, 0.0]), 90.0, np.array([2.0, 0.0]))
    assert np.allclose(local, [0.0, 2.0], atol=1e-6)


def test_passive_retrieval_scores_matching_successful_direction():
    north = episode(1, [0.0, 0.0], 0.0)
    east = episode(2, [1.0, 1.0], math.pi / 2)
    query = episode(3, [0.02, 0.01], 0.05)
    result = evaluate_library([north, east], [query], vigilance=0.75)
    assert result["accepted"] == 1
    assert result["directional_agreement_all_queries"] == 1.0
    assert result["mean_normalized_route_shape_distance"] < 0.05
