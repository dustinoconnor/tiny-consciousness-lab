#!/usr/bin/env python3
"""Build an ART-gated episodic route library from successful Unity L-wall runs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lwall_hidden_goal_adapter_lab import grouped_episodes, position


def context_vector(rows, frames=8):
    samples = []
    for row in rows[:frames]:
        rays = row.get("rays")
        if isinstance(rays, list) and len(rays) == 8:
            samples.append(np.clip(np.asarray(rays, dtype=np.float32), 0.0, 1.0))
    if not samples:
        raise ValueError("trajectory_missing_sensor_context")
    return np.mean(samples, axis=0)


def sparse_route(rows, spacing):
    points = np.stack([position(row) for row in rows])
    origin = points[0]
    waypoints = [np.zeros(2, dtype=np.float32)]
    for point in points[1:]:
        relative = point - origin
        if float(np.linalg.norm(relative - waypoints[-1])) >= spacing:
            waypoints.append(relative)
    final = points[-1] - origin
    if float(np.linalg.norm(final - waypoints[-1])) > 0.25:
        waypoints.append(final)
    return [point.tolist() for point in waypoints[1:]]


def trajectory_metrics(rows):
    points = np.stack([position(row) for row in rows])
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    path_length = float(np.sum(lengths))
    displacement = float(np.linalg.norm(points[-1] - points[0]))
    moving = deltas[lengths > 1e-5]
    if len(moving) >= 2:
        directions = moving / np.linalg.norm(moving, axis=1, keepdims=True)
        turns = np.arccos(
            np.clip(np.sum(directions[1:] * directions[:-1], axis=1), -1.0, 1.0)
        )
        steering_jerk = float(np.mean(turns) / np.pi)
    else:
        steering_jerk = 0.0
    path_waste = max(0.0, path_length / max(displacement, 1e-5) - 1.0)
    return {
        "path_length": path_length,
        "displacement": displacement,
        "path_waste": path_waste,
        "steering_jerk": steering_jerk,
    }


def route_signature(waypoints, samples=24):
    points = np.asarray([[0.0, 0.0], *waypoints], dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, samples, axis=0)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    if cumulative[-1] <= 1e-6:
        return np.repeat(points[:1], samples, axis=0)
    targets = np.linspace(0.0, cumulative[-1], samples)
    return np.stack(
        [
            np.interp(targets, cumulative, points[:, axis])
            for axis in range(2)
        ],
        axis=1,
    )


def route_distance(first, second):
    return float(
        np.mean(
            np.linalg.norm(
                np.asarray(first["signature"]) - np.asarray(second["signature"]),
                axis=1,
            )
        )
    )


def select_diverse_routes(
    candidates,
    count,
    minimum_distance,
    minimum_relative_quality=0.65,
):
    remaining = sorted(candidates, key=lambda item: (item["cost"], item["route_id"]))
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    best_quality = float(selected[0]["quality"])
    remaining = [
        item
        for item in remaining
        if float(item["quality"]) >= best_quality * minimum_relative_quality
    ]
    while remaining and len(selected) < count:
        diverse = [
            item
            for item in remaining
            if min(route_distance(item, chosen) for chosen in selected) >= minimum_distance
        ]
        if not diverse:
            break
        candidate = min(diverse, key=lambda item: (item["cost"], item["route_id"]))
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def collect_candidates(recording_dirs, spacing):
    candidates = []
    for recording_dir in recording_dirs:
        for recording in sorted(recording_dir.glob("seed_*.jsonl")):
            for episode, rows in grouped_episodes(recording).items():
                if not any(row.get("trap_outcome") == "success" for row in rows):
                    continue
                hidden = []
                for row in rows:
                    if bool(row.get("food_visible", False)):
                        break
                    hidden.append(row)
                if len(hidden) < 2:
                    continue
                try:
                    context = context_vector(hidden)
                except ValueError:
                    continue
                collisions = sum(bool(row.get("body_collision")) for row in hidden)
                variant = str(hidden[0].get("trap_course_variant", "lwall_original"))
                if variant == "standard":
                    variant = "lwall_original"
                waypoints = sparse_route(hidden, spacing)
                geometry = trajectory_metrics(hidden)
                cost = (
                    len(hidden)
                    + 20 * collisions
                    + 55.0 * geometry["steering_jerk"]
                    + 12.0 * geometry["path_waste"]
                )
                candidates.append(
                    {
                        "route_id": f"{variant}:{recording.stem}:ep{episode}",
                        "variant": variant,
                        "prototype": context.tolist(),
                        "waypoints": waypoints,
                        "signature": route_signature(waypoints).tolist(),
                        "quality": math.exp(-cost / 250.0),
                        "cost": cost,
                        "hidden_frames": len(hidden),
                        "collisions": collisions,
                        **geometry,
                        "source_recording": str(recording),
                        "source_episode": int(episode),
                    }
                )
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/lwall_hidden_goal_adapter/v5_art_route_library.pt"),
    )
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--routes-per-variant", type=int, default=3)
    parser.add_argument("--vigilance", type=float, default=0.86)
    parser.add_argument("--diversity-distance", type=float, default=1.25)
    parser.add_argument("--minimum-relative-quality", type=float, default=0.65)
    args = parser.parse_args()

    candidates = collect_candidates(args.recording_dir, args.spacing)
    selected = []
    for variant in sorted({item["variant"] for item in candidates}):
        routes = select_diverse_routes(
            [item for item in candidates if item["variant"] == variant],
            max(1, args.routes_per_variant),
            max(0.0, args.diversity_distance),
            min(1.0, max(0.0, args.minimum_relative_quality)),
        )
        selected.extend(routes)
    variants = {item["variant"] for item in selected}
    required = {"lwall_original", "lwall_mirrored"}
    if not required.issubset(variants):
        missing = ", ".join(sorted(required - variants))
        raise RuntimeError(f"Missing successful Unity experience for: {missing}")

    payload = {
        "adapter_type": "art_route_library",
        "activation_scope": "lwall_food_hidden_only",
        "routes": selected,
        "art_vigilance": float(args.vigilance),
        "waypoint_radius": 0.85,
        "terminal_waypoint_radius": 0.30,
        "terminal_waypoint_count": 5,
        "terminal_extension": 2.0,
        "target_latch_seconds": 2.0,
        "selection_exploration_rate": 0.12,
        "spacing": args.spacing,
        "diversity_distance": args.diversity_distance,
        "minimum_relative_quality": args.minimum_relative_quality,
        "claim_boundary": (
            "Successful Unity trajectories are retrieved by Fuzzy-ART-style sensory resonance. "
            "This tests adaptive precedent selection, not topology-general navigation."
        ),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.checkpoint)
    summary = {
        "checkpoint": str(args.checkpoint),
        "routes": len(selected),
        "variants": {variant: sum(r["variant"] == variant for r in selected) for variant in variants},
        "vigilance": args.vigilance,
        "sources": [route["route_id"] for route in selected],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
