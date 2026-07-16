#!/usr/bin/env python3
"""Stream a long Unity telemetry recording into an overnight reliability report."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def event_context(before, after, start_time):
    return {
        "hours_into_run": round((after["time"] - start_time) / 3600.0, 3),
        "reason": after.get("survival_failure_reason", "none"),
        "before_reset": {
            "position": [round(float(value), 2) for value in before["position"]],
            "hunger": round(float(before.get("hunger", 0.0)), 3),
            "seconds_since_food": round(float(before.get("seconds_since_food", 0.0)), 1),
            "food_visible": bool(before.get("food_visible", False)),
            "food_sensor_radius": float(before.get("food_sensor_radius", 0.0)),
            "stuck": bool(before.get("stuck", False)),
            "stuck_events": int(before.get("stuck_events", 0)),
            "wedge_seconds": round(float(before.get("physics_wedge_seconds", 0.0)), 1),
            "fallback_active": bool(before.get("fallback_active", False)),
            "mpc_mode": before.get("shadow_mpc_mode", "off"),
            "mpc_horizon": int(before.get("shadow_mpc_horizon", 0)),
        },
        "after_reset_position": [round(float(value), 2) for value in after["position"]],
    }


def diagnose_interval(path, start_time, end_time):
    frames = 0
    previous = None
    distance = 0.0
    xs = []
    zs = []
    cells = set()
    food_visible_frames = 0
    stuck_frames = 0
    collision_frames = 0
    modes = Counter()
    actions = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            timestamp = float(row["time"])
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            frames += 1
            x = float(row["position"][0])
            z = float(row["position"][2])
            xs.append(x)
            zs.append(z)
            cells.add((math.floor(x / 10.0), math.floor(z / 10.0)))
            food_visible_frames += bool(row.get("food_visible", False))
            stuck_frames += bool(row.get("stuck", False))
            collision_frames += bool(row.get("body_collision", False))
            modes[row.get("shadow_mpc_mode", "off")] += 1
            actions[row.get("active_action", "unknown")] += 1
            if previous is not None:
                step_distance = math.hypot(x - previous[0], z - previous[1])
                if step_distance <= 50.0:
                    distance += step_distance
            previous = (x, z)
    duration = max(0.0, end_time - start_time)

    def seconds_for(count):
        return count / frames * duration if frames else 0.0

    return {
        "frames": frames,
        "path_distance_meters": round(distance, 1),
        "net_displacement_meters": round(math.hypot(xs[-1] - xs[0], zs[-1] - zs[0]), 1),
        "x_range": [round(min(xs), 1), round(max(xs), 1)],
        "z_range": [round(min(zs), 1), round(max(zs), 1)],
        "unique_10m_cells": len(cells),
        "food_visible_seconds": round(seconds_for(food_visible_frames), 1),
        "stuck_seconds": round(seconds_for(stuck_frames), 1),
        "collision_seconds": round(seconds_for(collision_frames), 1),
        "mpc_mode_seconds": {
            mode: round(seconds_for(count), 1) for mode, count in modes.most_common()
        },
        "active_action_frames": dict(actions.most_common()),
    }


def analyze(path):
    frames = 0
    first = None
    previous = None
    last = None
    pickup_events = []
    failure_events = []
    respawn_events = []
    mode_frames = Counter()
    horizon_frames = Counter()
    radius_frames = Counter()
    action_frames = Counter()
    food_visible_frames = 0
    collision_frames = 0
    blocked_frames = 0
    stuck_frames = 0
    fallback_frames = 0
    takeover_frames = 0
    hunger_high_frames = 0
    hunger_critical_frames = 0
    shadow_action_changes = 0
    distance_meters = 0.0
    respawn_like_jumps = 0
    max_hunger = 0.0
    max_seconds_since_food = 0.0
    max_wedge_seconds = 0.0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frames += 1
            if first is None:
                first = row
            start_time = first["time"]
            hunger = float(row.get("hunger", 0.0))
            max_hunger = max(max_hunger, hunger)
            max_seconds_since_food = max(
                max_seconds_since_food, float(row.get("seconds_since_food", 0.0))
            )
            max_wedge_seconds = max(
                max_wedge_seconds, float(row.get("physics_wedge_seconds", 0.0))
            )
            food_visible_frames += bool(row.get("food_visible", False))
            collision_frames += bool(row.get("body_collision", False))
            blocked_frames += bool(row.get("blocked", False))
            stuck_frames += bool(row.get("stuck", False))
            fallback_frames += bool(row.get("fallback_active", False))
            takeover_frames += bool(row.get("shadow_takeover", False))
            hunger_high_frames += hunger >= 0.70
            hunger_critical_frames += hunger >= 0.92
            mode_frames[row.get("shadow_mpc_mode", "off")] += 1
            horizon_frames[int(row.get("shadow_mpc_horizon", 0))] += 1
            radius_frames[float(row.get("food_sensor_radius", 0.0))] += 1
            action_frames[row.get("active_action", "unknown")] += 1

            if previous is not None:
                if row.get("mushroom_pickups_total", 0) > previous.get("mushroom_pickups_total", 0):
                    pickup_events.append(
                        (float(row["time"]), [float(value) for value in row["position"]])
                    )
                if row.get("survival_failures", 0) > previous.get("survival_failures", 0):
                    failure_events.append(event_context(previous, row, start_time))
                if row.get("unstuck_respawns", 0) > previous.get("unstuck_respawns", 0):
                    respawn_events.append(event_context(previous, row, start_time))
                if row.get("shadow_action") != previous.get("shadow_action"):
                    shadow_action_changes += 1
                dx = float(row["position"][0]) - float(previous["position"][0])
                dz = float(row["position"][2]) - float(previous["position"][2])
                distance = math.hypot(dx, dz)
                if distance > 50.0:
                    respawn_like_jumps += 1
                else:
                    distance_meters += distance
            previous = row
            last = row

    if first is None or last is None or frames < 2:
        raise ValueError("Recording needs at least two telemetry frames")

    duration_seconds = float(last["time"] - first["time"])
    intervals = [
        later[0] - earlier[0] for earlier, later in zip(pickup_events, pickup_events[1:])
    ]
    longest_gap = None
    if intervals:
        gap_index = max(range(len(intervals)), key=intervals.__getitem__)
        before_gap, after_gap = pickup_events[gap_index:gap_index + 2]
        longest_gap = {
            "seconds": round(intervals[gap_index], 2),
            "start_hours": round((before_gap[0] - first["time"]) / 3600.0, 3),
            "end_hours": round((after_gap[0] - first["time"]) / 3600.0, 3),
            "start_position": [round(value, 2) for value in before_gap[1]],
            "end_position": [round(value, 2) for value in after_gap[1]],
            "diagnostics": diagnose_interval(path, before_gap[0], after_gap[0]),
        }
    pickups = int(last.get("mushroom_pickups_total", 0) - first.get("mushroom_pickups_total", 0))

    def seconds_for(count):
        return count / frames * duration_seconds

    return {
        "source": str(path),
        "frames": frames,
        "duration_hours": round(duration_seconds / 3600.0, 3),
        "sample_rate_hz": round((frames - 1) / duration_seconds, 3),
        "foraging": {
            "pickups": pickups,
            "pickups_per_hour": round(pickups / (duration_seconds / 3600.0), 2),
            "food_visible_percent": round(100.0 * food_visible_frames / frames, 2),
            "max_hunger": round(max_hunger, 3),
            "max_seconds_since_food": round(max_seconds_since_food, 1),
            "median_pickup_interval_seconds": round(percentile(intervals, 0.50), 2),
            "p95_pickup_interval_seconds": round(percentile(intervals, 0.95), 2),
            "max_pickup_interval_seconds": round(max(intervals, default=0.0), 2),
            "longest_pickup_gap": longest_gap,
            "seconds_at_hunger_0_70": round(seconds_for(hunger_high_frames), 1),
            "seconds_at_hunger_0_92": round(seconds_for(hunger_critical_frames), 1),
            "radius_seconds": {
                str(radius): round(seconds_for(count), 1)
                for radius, count in sorted(radius_frames.items())
            },
        },
        "reliability": {
            "survival_failures": int(last.get("survival_failures", 0)),
            "failure_events": failure_events,
            "stuck_events": int(last.get("stuck_events", 0)),
            "stuck_seconds": round(seconds_for(stuck_frames), 1),
            "max_wedge_seconds": round(max_wedge_seconds, 1),
            "unstuck_respawns": int(last.get("unstuck_respawns", 0)),
            "respawn_events": respawn_events,
            "respawn_like_position_jumps": respawn_like_jumps,
            "fallback_seconds": round(seconds_for(fallback_frames), 1),
            "collision_seconds": round(seconds_for(collision_frames), 1),
            "blocked_seconds": round(seconds_for(blocked_frames), 1),
            "handoff_events": int(last.get("handoff_events", 0)),
            "final_generation": int(last.get("generation", 0)),
        },
        "control": {
            "takeover_percent": round(100.0 * takeover_frames / frames, 2),
            "mpc_mode_seconds": {
                mode: round(seconds_for(count), 1) for mode, count in mode_frames.most_common()
            },
            "mpc_horizon_seconds": {
                str(horizon): round(seconds_for(count), 1)
                for horizon, count in sorted(horizon_frames.items())
            },
            "active_action_frames": dict(action_frames.most_common()),
            "shadow_action_change_percent": round(100.0 * shadow_action_changes / (frames - 1), 2),
            "distance_meters_excluding_respawns": round(distance_meters, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/unity_overnight_analysis.json"))
    args = parser.parse_args()
    summary = analyze(args.recording)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
