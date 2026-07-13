#!/usr/bin/env python3
"""Analyze active GRU control on generated Unity trap-course episodes."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


OPPOSITE = {
    "up": "down", "up_right": "down_left", "right": "left", "down_right": "up_left",
    "down": "up", "down_left": "up_right", "left": "right", "up_left": "down_right",
}


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: float(row["time"]))
    return rows


def distance(a, b):
    return math.hypot(float(b["position"][0]) - float(a["position"][0]), float(b["position"][2]) - float(a["position"][2]))


def summarize_episode(episode, course, rows):
    rows.sort(key=lambda row: float(row["time"]))
    moving_pairs = [(a, b) for a, b in zip(rows, rows[1:]) if distance(a, b) >= 0.01]
    path_length = sum(distance(a, b) for a, b in moving_pairs)
    cells = [(round(float(row["position"][0])), round(float(row["position"][2]))) for row in rows]
    revisit_ratio = 1.0 - len(set(cells)) / len(cells) if cells else 0.0
    reversals = sum(
        OPPOSITE.get(a.get("shadow_action")) == b.get("shadow_action")
        for a, b in zip(rows, rows[1:])
    )
    first_visible = next((row for row in rows if row.get("food_visible")), None)
    success_row = next((row for row in rows if row.get("trap_outcome") == "success"), None)
    pickup_delay = None
    if first_visible is not None and success_row is not None:
        pickup_delay = max(0.0, float(success_row["time"]) - float(first_visible["time"]))
    outcomes = {row.get("trap_outcome") for row in rows}
    return {
        "episode": episode,
        "course": course,
        "success": "success" in outcomes,
        "timeout": "timeout" in outcomes,
        "duration_seconds": float(rows[-1]["time"]) - float(rows[0]["time"]),
        "frames": len(rows),
        "path_length": path_length,
        "collision_frames": sum(bool(row.get("body_collision")) for row in rows),
        "stuck_frames": sum(bool(row.get("stuck")) for row in rows),
        "learned_control_fraction": sum(bool(row.get("shadow_takeover")) for row in rows) / len(rows),
        "food_visible_frames": sum(bool(row.get("food_visible")) for row in rows),
        "food_visible_to_pickup_seconds": pickup_delay,
        "revisit_ratio_1m": revisit_ratio,
        "shadow_action_reversals": reversals,
        "mean_shadow_confidence": sum(float(row.get("shadow_confidence", 0.0)) for row in rows) / len(rows),
    }


def analyze(paths):
    rows = load_rows(paths)
    grouped = defaultdict(list)
    for row in rows:
        episode = row.get("trap_episode")
        course = row.get("trap_course")
        if not episode or not course or course == "natural_terrain":
            continue
        grouped[(int(episode), str(course))].append(row)
    episodes = [summarize_episode(episode, course, selected) for (episode, course), selected in sorted(grouped.items())]
    completed = [episode for episode in episodes if episode["success"] or episode["timeout"]]
    successes = [episode for episode in completed if episode["success"]]
    return {
        "sources": [str(path) for path in paths],
        "episodes": episodes,
        "completed_episodes": len(completed),
        "successes": len(successes),
        "timeouts": sum(episode["timeout"] for episode in completed),
        "success_rate": len(successes) / len(completed) if completed else 0.0,
        "mean_success_seconds": sum(episode["duration_seconds"] for episode in successes) / len(successes) if successes else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="*", help="JSONL files; defaults to all Unity shadow recordings")
    parser.add_argument("--output", default="outputs/unity_course_analysis.json")
    args = parser.parse_args()
    paths = [Path(path) for path in args.recordings] if args.recordings else sorted(Path("outputs/unity_shadow").glob("*.jsonl"))
    if not paths:
        parser.error("no Unity recordings found")
    result = analyze(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
