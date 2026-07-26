#!/usr/bin/env python3
"""Compare AIR-selected and random ART route libraries on Unity terrain.

Successful pickup episodes are split chronologically. Early difficult routes
form fixed-capacity precedent libraries; later difficult routes are untouched
queries. Retrieval is evaluated passively at the first grounded intervention-
necessity frame. No recommendation is sent to Unity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lwall_art_route_library_lab import route_signature
from terrain_air_counterfactual_gate_lab import intervention_needed
from terrain_air_memory_lab import LogisticAttentionGate, build_packets, load_rows


@dataclass
class RouteEpisode:
    episode_id: int
    start_index: int
    trigger_index: int
    pickup_index: int
    context: np.ndarray
    waypoints: list[list[float]]
    signature: np.ndarray
    air_score: float
    duration: float
    collisions: int
    orbit_frames: int
    path_length: float
    displacement: float

    def as_json(self):
        return {
            "episode_id": self.episode_id,
            "source_indices": {
                "start": self.start_index,
                "necessity_trigger": self.trigger_index,
                "pickup": self.pickup_index,
            },
            "prototype": self.context.tolist(),
            "waypoints": self.waypoints,
            "air_score": self.air_score,
            "duration_seconds": self.duration,
            "collisions": self.collisions,
            "orbit_frames": self.orbit_frames,
            "path_length": self.path_length,
            "displacement": self.displacement,
        }


def position(row):
    values = row.get("position", [0.0, 0.0, 0.0])
    return np.asarray([float(values[0]), float(values[2])], dtype=np.float64)


def local_offset(origin, yaw_degrees, point):
    delta = point - origin
    yaw = math.radians(float(yaw_degrees))
    right = np.asarray([math.cos(yaw), -math.sin(yaw)])
    forward = np.asarray([math.sin(yaw), math.cos(yaw)])
    return np.asarray([np.dot(delta, right), np.dot(delta, forward)])


def route_context(row):
    rays = np.clip(np.asarray(row.get("rays", [1.0] * 8)), 0.0, 1.0)
    clearance = np.clip(
        np.asarray(row.get("body_clearance", [1.0] * 8)), 0.0, 1.0
    )
    if rays.size != 8 or clearance.size != 8:
        raise ValueError("route context requires eight rays and clearances")
    return np.r_[
        rays,
        clearance,
        np.clip(float(row.get("hunger", 0.0) or 0.0), 0.0, 1.0),
        float(bool(row.get("food_visible", False))),
        np.clip(float(row.get("orbit_efficiency", 1.0) or 1.0), 0.0, 1.0),
        np.clip(float(row.get("orbit_path", 0.0) or 0.0) / 20.0, 0.0, 1.0),
        np.clip(float(row.get("physics_wedge_seconds", 0.0) or 0.0) / 8.0, 0.0, 1.0),
        np.clip(float(row.get("trap_accumulation_seconds", 0.0) or 0.0) / 8.0, 0.0, 1.0),
    ]


def sparse_local_route(rows, trigger_offset, spacing=1.0):
    source = rows[trigger_offset:]
    origin = position(source[0])
    yaw = float(source[0].get("yaw", 0.0) or 0.0)
    selected = []
    last = np.zeros(2, dtype=np.float64)
    path_length = 0.0
    previous = origin
    for row in source[1:]:
        current = position(row)
        path_length += float(np.linalg.norm(current - previous))
        previous = current
        local = local_offset(origin, yaw, current)
        if float(np.linalg.norm(local - last)) >= spacing:
            selected.append(local)
            last = local
    final = local_offset(origin, yaw, position(source[-1]))
    if not selected or float(np.linalg.norm(final - selected[-1])) > 0.25:
        selected.append(final)
    waypoints = [point.tolist() for point in selected]
    return waypoints, path_length, float(np.linalg.norm(final))


def pickup_segments(rows):
    segments = []
    start = 0
    previous = int(rows[0].get("mushroom_pickups_total", 0) or 0)
    episode = 0
    for index, row in enumerate(rows):
        current = int(row.get("mushroom_pickups_total", 0) or 0)
        if current <= previous:
            continue
        segments.append((episode, start, index, rows[start : index + 1]))
        episode += 1
        start = index + 1
        previous = current
    return segments


def build_route_episodes(rows, packet_scores, spacing=1.0):
    scores_by_index = {
        packet.source_index: float(score)
        for packet, score in packet_scores
    }
    episodes = []
    for episode_id, start, pickup, segment in pickup_segments(rows):
        necessity_offsets = [
            index for index, row in enumerate(segment) if intervention_needed(row)
        ]
        if not necessity_offsets:
            continue
        trigger_offset = necessity_offsets[0]
        trigger = start + trigger_offset
        waypoints, path_length, displacement = sparse_local_route(
            segment, trigger_offset, spacing
        )
        if not waypoints or displacement < 0.5:
            continue
        nearby_scores = [
            scores_by_index[index]
            for index in range(trigger, min(pickup + 1, trigger + 12))
            if index in scores_by_index
        ]
        air_score = max(nearby_scores) if nearby_scores else -math.inf
        episodes.append(
            RouteEpisode(
                episode_id=episode_id,
                start_index=start,
                trigger_index=trigger,
                pickup_index=pickup,
                context=route_context(rows[trigger]),
                waypoints=waypoints,
                signature=route_signature(waypoints),
                air_score=air_score,
                duration=float(rows[pickup].get("time", 0.0))
                - float(rows[trigger].get("time", 0.0)),
                collisions=sum(
                    bool(row.get("body_collision", False))
                    for row in rows[trigger : pickup + 1]
                ),
                orbit_frames=sum(
                    intervention_needed(row) for row in rows[trigger : pickup + 1]
                ),
                path_length=path_length,
                displacement=displacement,
            )
        )
    return episodes


def retrieve(library, query):
    if not library:
        return None, math.inf, 0.0
    distances = [
        float(np.mean(np.abs(route.context - query.context))) for route in library
    ]
    index = int(np.argmin(distances))
    distance = distances[index]
    return library[index], distance, max(0.0, 1.0 - distance)


def direction(vector):
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-6 else np.zeros(2)


def route_direction(route, lookahead=3):
    index = min(max(lookahead - 1, 0), len(route.waypoints) - 1)
    return direction(route.waypoints[index])


def normalized_signature(route):
    # Path-length normalization compares route shape without rewarding a
    # precedent merely because its source mushroom happened to be nearby.
    return np.asarray(route.signature, dtype=np.float64) / max(
        route.path_length, 1e-6
    )


def evaluate_library(library, queries, vigilance=0.78):
    accepted = correct = 0
    cosine_values = []
    shape_distances = []
    path_length_errors = []
    matches = []
    details = []
    for query in queries:
        recalled, distance, match = retrieve(library, query)
        accepted_now = recalled is not None and match >= vigilance
        cosine = None
        agrees = False
        if accepted_now:
            accepted += 1
            cosine = float(
                np.dot(route_direction(recalled), route_direction(query))
            )
            cosine_values.append(cosine)
            agrees = cosine >= math.cos(math.radians(45.0))
            correct += agrees
            shape_distance = float(
                np.mean(
                    np.linalg.norm(
                        normalized_signature(recalled)
                        - normalized_signature(query),
                        axis=1,
                    )
                )
            )
            path_length_error = abs(
                math.log(
                    max(recalled.path_length, 1e-3)
                    / max(query.path_length, 1e-3)
                )
            )
            shape_distances.append(shape_distance)
            path_length_errors.append(path_length_error)
        else:
            shape_distance = None
            path_length_error = None
        matches.append(match)
        details.append(
            {
                "query_episode": query.episode_id,
                "recalled_episode": recalled.episode_id if recalled else None,
                "art_match": match,
                "accepted": accepted_now,
                "first_leg_cosine": cosine,
                "directional_agreement": agrees,
                "normalized_route_shape_distance": shape_distance,
                "log_path_length_error": path_length_error,
            }
        )
    return {
        "queries": len(queries),
        "accepted": accepted,
        "coverage": accepted / max(len(queries), 1),
        "directional_agreement_all_queries": correct / max(len(queries), 1),
        "directional_agreement_given_retrieval": correct / max(accepted, 1),
        "mean_first_leg_cosine_given_retrieval": (
            float(np.mean(cosine_values)) if cosine_values else None
        ),
        "mean_normalized_route_shape_distance": (
            float(np.mean(shape_distances)) if shape_distances else None
        ),
        "mean_log_path_length_error": (
            float(np.mean(path_length_errors)) if path_length_errors else None
        ),
        "mean_art_match": float(np.mean(matches)) if matches else None,
        "recommendations": details,
    }


def run(
    recording,
    output,
    checkpoint,
    capacity=8,
    train_fraction=0.6,
    vigilance=0.78,
    random_trials=500,
    seed=20260726,
):
    rows = load_rows(recording)
    packets = build_packets(rows)
    split_row = int(len(rows) * train_fraction)
    training_packets = [packet for packet in packets if packet.source_index < split_row]
    if not training_packets:
        raise ValueError("no AIR training packets")
    gate = LogisticAttentionGate()
    gate.fit(training_packets)
    all_scores = gate.scores(packets)
    episodes = build_route_episodes(
        rows, list(zip(packets, all_scores, strict=True))
    )
    train = [episode for episode in episodes if episode.pickup_index < split_row]
    queries = [episode for episode in episodes if episode.trigger_index >= split_row]
    if not train or not queries:
        raise ValueError("chronological split lacks difficult train or query routes")
    library_size = min(max(1, capacity), len(train))
    air_library = sorted(
        train, key=lambda episode: (-episode.air_score, episode.episode_id)
    )[:library_size]
    air_result = evaluate_library(air_library, queries, vigilance)

    rng = np.random.default_rng(seed)
    random_scores = []
    random_coverage = []
    random_shape_distances = []
    random_path_length_errors = []
    for _ in range(max(1, random_trials)):
        indices = rng.choice(len(train), library_size, replace=False)
        result = evaluate_library(
            [train[int(index)] for index in indices], queries, vigilance
        )
        random_scores.append(result["directional_agreement_all_queries"])
        random_coverage.append(result["coverage"])
        random_shape_distances.append(
            result["mean_normalized_route_shape_distance"]
        )
        random_path_length_errors.append(result["mean_log_path_length_error"])
    air_score = air_result["directional_agreement_all_queries"]
    random_scores_array = np.asarray(random_scores)
    random_mean = float(np.mean(random_scores_array))
    advantage = air_score - random_mean
    percentile = float(
        100.0 * np.mean(random_scores_array <= air_score)
    )
    random_shape_array = np.asarray(random_shape_distances)
    air_shape = air_result["mean_normalized_route_shape_distance"]
    shape_advantage = float(np.mean(random_shape_array) - air_shape)
    shape_percentile = float(100.0 * np.mean(random_shape_array >= air_shape))
    empirical_shape_tail = float(
        (1 + np.sum(random_shape_array <= air_shape))
        / (len(random_shape_array) + 1)
    )

    payload = {
        "experiment": "terrain AIR-selected versus random ART route libraries",
        "protocol": {
            "recording": str(Path(recording).resolve()),
            "telemetry_rows": len(rows),
            "pickup_episodes": len(pickup_segments(rows)),
            "difficult_successful_routes": len(episodes),
            "training_routes": len(train),
            "held_out_routes": len(queries),
            "chronological_split_row": split_row,
            "train_fraction": train_fraction,
            "equal_library_capacity": library_size,
            "random_library_trials": max(1, random_trials),
            "art_vigilance": vigilance,
            "motor_control_enabled": False,
            "necessity_gate": (
                "physics wedge >=1s OR trap accumulation >=2s OR "
                "(orbit path >=4m AND efficiency <=0.25)"
            ),
        },
        "air_selected": air_result,
        "random_capacity_control": {
            "mean_directional_agreement_all_queries": random_mean,
            "std_directional_agreement_all_queries": float(
                np.std(random_scores_array)
            ),
            "mean_coverage": float(np.mean(random_coverage)),
            "air_advantage": advantage,
            "air_percentile_among_random_libraries": percentile,
            "mean_normalized_route_shape_distance": float(
                np.mean(random_shape_array)
            ),
            "std_normalized_route_shape_distance": float(
                np.std(random_shape_array)
            ),
            "mean_log_path_length_error": float(
                np.mean(random_path_length_errors)
            ),
            "air_shape_distance_advantage": shape_advantage,
            "air_shape_quality_percentile": shape_percentile,
            "empirical_shape_tail_probability": empirical_shape_tail,
        },
        "air_library": [route.as_json() for route in air_library],
        "claim_boundary": (
            "All query episodes eventually reached food, so their realized "
            "trajectory supplies a successful directional precedent. The test "
            "measures passive recommendation agreement after grounded necessity "
            "triggers. Full-route shape is the discriminating endpoint because "
            "first-leg direction saturates for both libraries. It does not "
            "demonstrate counterfactual route success or justify unbounded motor "
            "control."
        ),
    }
    output = Path(output)
    checkpoint = Path(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    checkpoint.write_text(
        json.dumps(
            {
                "format": "terrain_air_art_route_library_v1",
                "mode": "passive_recommendation_only",
                "art_vigilance": vigilance,
                "route_capacity": library_size,
                "routes": [route.as_json() for route in air_library],
                "motor_control_enabled": False,
                "source_metrics": str(output.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/terrain_air_art_route_library_metrics.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/terrain_air/passive_route_library.json"),
    )
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--vigilance", type=float, default=0.78)
    parser.add_argument("--random-trials", type=int, default=500)
    args = parser.parse_args()
    payload = run(
        args.recording,
        args.output,
        args.checkpoint,
        capacity=args.capacity,
        train_fraction=args.train_fraction,
        vigilance=args.vigilance,
        random_trials=args.random_trials,
    )
    summary = {
        "protocol": payload["protocol"],
        "air_selected": {
            key: value
            for key, value in payload["air_selected"].items()
            if key != "recommendations"
        },
        "random_capacity_control": payload["random_capacity_control"],
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
