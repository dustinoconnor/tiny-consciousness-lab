#!/usr/bin/env python3
"""Measure route-to-food handoff stability from Unity JSONL telemetry."""

import argparse
import json
from pathlib import Path


def yaw_delta(first, second):
    return abs((float(second) - float(first) + 180.0) % 360.0 - 180.0)


def analyze_rows(rows, window_frames=50):
    episodes = {}
    for row in rows:
        episodes.setdefault(int(row.get("trap_episode", 0)), []).append(row)
    results = []
    for episode, samples in sorted(episodes.items()):
        first_visible = next(
            (index for index, row in enumerate(samples) if row.get("food_visible")),
            None,
        )
        if first_visible is None:
            continue
        window = samples[first_visible : first_visible + max(2, int(window_frames))]
        finished = next(
            (
                index
                for index, row in enumerate(window)
                if str(row.get("trap_outcome", "running")) in {"success", "timeout"}
            ),
            None,
        )
        if finished is not None:
            window = window[: finished + 1]
        yaw_travel = sum(
            yaw_delta(first.get("yaw", 0.0), second.get("yaw", 0.0))
            for first, second in zip(window, window[1:])
        )
        visibility_losses = sum(
            bool(first.get("food_visible"))
            and not bool(second.get("food_visible"))
            and str(second.get("trap_outcome", "running")) not in {"success", "timeout"}
            for first, second in zip(window, window[1:])
        )
        results.append(
            {
                "episode": episode,
                "variant": samples[0].get("trap_course_variant", "unknown"),
                "outcome": samples[-1].get("trap_outcome", "unknown"),
                "yaw_travel_degrees": round(yaw_travel, 3),
                "visibility_losses": int(visibility_losses),
                "latched_frames": sum(
                    bool(row.get("hidden_goal_food_latched")) for row in window
                ),
                "continuous_intercept_frames": sum(
                    bool(row.get("shadow_continuous_intercept")) for row in window
                ),
            }
        )
    count = len(results)
    return {
        "episodes_with_food_sighting": count,
        "mean_yaw_travel_degrees": round(
            sum(item["yaw_travel_degrees"] for item in results) / max(1, count), 3
        ),
        "total_visibility_losses": sum(item["visibility_losses"] for item in results),
        "total_latched_frames": sum(item["latched_frames"] for item in results),
        "total_continuous_intercept_frames": sum(
            item["continuous_intercept_frames"] for item in results
        ),
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--window-frames", type=int, default=50)
    args = parser.parse_args()
    paths = (
        sorted(args.telemetry.glob("*.jsonl"))
        if args.telemetry.is_dir()
        else [args.telemetry]
    )
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    print(json.dumps(analyze_rows(rows, args.window_frames), indent=2))


if __name__ == "__main__":
    main()
