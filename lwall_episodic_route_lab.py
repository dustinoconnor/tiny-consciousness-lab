#!/usr/bin/env python3
"""Export the cleanest successful Unity L-wall trajectory as episodic route memory."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lwall_hidden_goal_adapter_lab import grouped_episodes, position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/lwall_hidden_goal_adapter/v4_episodic_route.pt"),
    )
    parser.add_argument("--spacing", type=float, default=1.0)
    args = parser.parse_args()

    candidates = []
    for recording in sorted(args.recording_dir.glob("seed_*.jsonl")):
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
            collisions = sum(bool(row.get("body_collision")) for row in hidden)
            candidates.append((len(hidden) + 20 * collisions, recording.stem, episode, hidden))
    if not candidates:
        raise RuntimeError("No successful food-hidden L-wall trajectory found")

    _score, source, episode, rows = min(candidates, key=lambda item: item[0])
    points = np.stack([position(row) for row in rows])
    origin = points[0]
    waypoints = [np.zeros(2, dtype=np.float32)]
    for point in points[1:]:
        relative = point - origin
        if float(np.linalg.norm(relative - waypoints[-1])) >= args.spacing:
            waypoints.append(relative)
    final = points[-1] - origin
    if float(np.linalg.norm(final - waypoints[-1])) > 0.25:
        waypoints.append(final)

    payload = {
        "adapter_type": "episodic_route",
        "activation_scope": "lwall_food_hidden_only",
        "waypoints": [point.tolist() for point in waypoints[1:]],
        "waypoint_radius": 0.85,
        "terminal_waypoint_radius": 0.30,
        "terminal_waypoint_count": 5,
        "terminal_extension": 2.0,
        "target_latch_seconds": 2.0,
        "source_recording": source,
        "source_episode": int(episode),
        "source_hidden_frames": len(rows),
        "spacing": args.spacing,
        "claim_boundary": (
            "A sensor-vetoed memory of one successful fixed L-wall route with visually gated "
            "terminal completion; this is episodic playback, not topology generalization."
        ),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.checkpoint)
    print(json.dumps({key: value for key, value in payload.items() if key != "waypoints"}, indent=2))
    print(f"waypoints: {len(payload['waypoints'])}")
    print(f"exported: {args.checkpoint}")


if __name__ == "__main__":
    main()
