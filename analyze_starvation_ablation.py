#!/usr/bin/env python3
"""Compare matched Unity food-deprivation and recovery telemetry runs."""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def distance(a, b):
    return math.hypot(
        float(b["position"][0]) - float(a["position"][0]),
        float(b["position"][2]) - float(a["position"][2]),
    )


def longest_local_dwell(rows, radius=5.0):
    best = 0.0
    for start in range(0, len(rows), 5):
        end = start
        while end + 1 < len(rows) and distance(rows[start], rows[end + 1]) <= radius:
            end += 1
        best = max(best, float(rows[end]["time"]) - float(rows[start]["time"]))
    return best


def phase_metrics(rows):
    duration = float(rows[-1]["time"]) - float(rows[0]["time"])
    path = sum(
        distance(before, after)
        for before, after in zip(rows, rows[1:])
        if float(after["time"]) - float(before["time"]) < 1.0 and distance(before, after) < 3.0
    )
    pickup_start = int(rows[0].get("mushroom_pickups_total", 0))
    pickup_end = int(rows[-1].get("mushroom_pickups_total", 0))
    return {
        "duration_seconds": round(duration, 2),
        "path_meters_excluding_jumps": round(path, 1),
        "net_displacement_meters": round(distance(rows[0], rows[-1]), 1),
        "unique_1m_cells": len({
            (math.floor(float(row["position"][0])), math.floor(float(row["position"][2])))
            for row in rows
        }),
        "unique_10m_cells": len({
            (math.floor(float(row["position"][0]) / 10.0), math.floor(float(row["position"][2]) / 10.0))
            for row in rows
        }),
        "longest_within_5m_seconds": round(longest_local_dwell(rows), 1),
        "respawn_like_position_jumps": sum(distance(a, b) > 10.0 for a, b in zip(rows, rows[1:])),
        "stuck_seconds": round(duration * sum(bool(row.get("stuck")) for row in rows) / len(rows), 1),
        "collision_seconds": round(duration * sum(bool(row.get("body_collision")) for row in rows) / len(rows), 1),
        "fallback_seconds": round(duration * sum(bool(row.get("fallback_active")) for row in rows) / len(rows), 1),
        "learned_takeover_percent": round(100.0 * sum(bool(row.get("shadow_takeover")) for row in rows) / len(rows), 2),
        "failure_count_start_end": [
            int(rows[0].get("survival_failures", 0)),
            int(rows[-1].get("survival_failures", 0)),
        ],
        "respawn_count_start_end": [
            int(rows[0].get("unstuck_respawns", 0)),
            int(rows[-1].get("unstuck_respawns", 0)),
        ],
        "hunger_start_end": [
            round(float(rows[0].get("hunger", 0.0)), 3),
            round(float(rows[-1].get("hunger", 0.0)), 3),
        ],
        "pickups": max(0, pickup_end - pickup_start),
        "food_visible_seconds": round(duration * sum(bool(row.get("food_visible")) for row in rows) / len(rows), 1),
        "shadow_actions": dict(Counter(row.get("shadow_action", "none") for row in rows).most_common()),
    }


def analyze_run(path, deprivation_seconds):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    split_time = float(rows[0]["time"]) + deprivation_seconds
    deprivation = [row for row in rows if float(row["time"]) < split_time]
    recovery = [row for row in rows if float(row["time"]) >= split_time]
    base_pickups = int(recovery[0].get("mushroom_pickups_total", 0))
    pickup_times = []
    previous = base_pickups
    for row in recovery:
        total = int(row.get("mushroom_pickups_total", 0))
        if total > previous:
            pickup_times.extend([float(row["time"])] * (total - previous))
        previous = total
    intervals = [later - earlier for earlier, later in zip(pickup_times, pickup_times[1:])]
    all_pickup_times = []
    previous_total = int(rows[0].get("mushroom_pickups_total", 0))
    for row in rows:
        total = int(row.get("mushroom_pickups_total", 0))
        if total > previous_total:
            all_pickup_times.extend([float(row["time"])] * (total - previous_total))
        previous_total = total
    recovery_metrics = phase_metrics(recovery)
    recovery_metrics.update({
        "first_pickup_latency_seconds": (
            round(pickup_times[0] - float(recovery[0]["time"]), 2) if pickup_times else None
        ),
        "pickup_interval_median_seconds": round(statistics.median(intervals), 2) if intervals else None,
        "pickup_interval_max_seconds": round(max(intervals), 2) if intervals else None,
    })
    return {
        "source": str(path),
        "duration_minutes": round((float(rows[-1]["time"]) - float(rows[0]["time"])) / 60.0, 2),
        "first_pickup_offset_from_scheduled_transition_seconds": (
            round(all_pickup_times[0] - split_time, 2) if all_pickup_times else None
        ),
        "total_pickups": max(
            0,
            int(rows[-1].get("mushroom_pickups_total", 0))
            - int(rows[0].get("mushroom_pickups_total", 0)),
        ),
        "deprivation": phase_metrics(deprivation),
        "recovery": recovery_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--deprivation-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/unity_shadow/starvation_ablation_analysis.json"))
    args = parser.parse_args()
    report = {
        "protocol": {
            "deprivation_seconds": args.deprivation_seconds,
            "mpc_enabled": False,
            "manual_steering": False,
            "food_toggle_timing": "manual; phase boundary has approximately 15 seconds of uncertainty",
            "replicates_per_checkpoint": 1,
        },
        "old_checkpoint": analyze_run(args.old, args.deprivation_seconds),
        "starvation_posttrained_checkpoint": analyze_run(args.new, args.deprivation_seconds),
        "claim_boundary": (
            "A matched single-run comparison supports transfer and provides an effect-size estimate; "
            "multiple randomized replicates are required for a robustness or statistical-significance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
