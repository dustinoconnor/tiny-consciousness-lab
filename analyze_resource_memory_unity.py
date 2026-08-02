#!/usr/bin/env python3
"""Summarize passive Unity resource-memory encoding and recommendations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def delta(rows, key):
    return float(rows[-1].get(key, 0) or 0) - float(rows[0].get(key, 0) or 0)


def recommendation_outcomes(rows, horizon_seconds=180.0):
    starts = []
    previous = (False, "none")
    for index, row in enumerate(rows):
        current = (
            bool(row.get("resource_memory_active", False)),
            str(row.get("resource_memory_recommendation", "none")),
        )
        if current[0] and current != previous:
            starts.append(index)
        previous = current

    outcomes = []
    for index in starts:
        row = rows[index]
        recommendation = str(
            row.get("resource_memory_recommendation", "unknown")
        )
        started = float(row["time"])
        pickup_total = int(row.get("mushroom_pickups_total", 0) or 0)
        target = row.get("resource_memory_target")
        active_distances = []
        next_pickup = None
        for later in range(index, len(rows)):
            candidate = rows[later]
            if float(candidate["time"]) - started > horizon_seconds:
                break
            if (
                bool(candidate.get("resource_memory_active", False))
                and str(candidate.get("resource_memory_recommendation", "none"))
                == recommendation
            ):
                active_distances.append(
                    float(candidate.get("resource_memory_distance", 0.0) or 0.0)
                )
            if int(candidate.get("mushroom_pickups_total", 0) or 0) > pickup_total:
                next_pickup = candidate
                break
        pickup_target_distance = None
        if next_pickup is not None and target is not None:
            position = next_pickup.get("position", [None, None, None])
            if (
                len(position) >= 3
                and position[0] is not None
                and position[2] is not None
            ):
                pickup_target_distance = math.hypot(
                    float(position[0]) - float(target[0]),
                    float(position[2]) - float(target[1]),
                )
        start_distance = (
            active_distances[0] if active_distances else float("inf")
        )
        minimum_distance = min(active_distances, default=float("inf"))
        outcomes.append(
            {
                "recommendation": recommendation,
                "start_distance": start_distance,
                "minimum_distance": minimum_distance,
                "approached_region": (
                    minimum_distance <= max(0.0, start_distance - 3.0)
                ),
                "pickup_within_horizon": next_pickup is not None,
                "pickup_delay_seconds": (
                    None
                    if next_pickup is None
                    else float(next_pickup["time"]) - started
                ),
                "pickup_target_distance": pickup_target_distance,
                "pickup_near_recommended_region": (
                    pickup_target_distance is not None
                    and pickup_target_distance <= 12.0
                ),
            }
        )
    return outcomes


def summarize(rows):
    enabled = [row for row in rows if row.get("resource_memory_enabled", False)]
    if not enabled:
        raise ValueError("recording contains no passive resource-memory rows")
    duration = max(0.0, float(enabled[-1]["time"]) - float(enabled[0]["time"]))
    hunger_gate = float(
        enabled[0].get("resource_memory_hunger_gate", 0.55) or 0.55
    )
    active_frames = sum(bool(row.get("resource_memory_active", False)) for row in enabled)
    eligible_frames = sum(
        float(row.get("hunger", 0.0) or 0.0) > hunger_gate
        and not bool(row.get("food_visible", False))
        for row in enabled
    )
    release_reasons = Counter(
        str(row.get("resource_memory_release_reason", "unknown"))
        for row in enabled
    )
    outcomes = recommendation_outcomes(enabled)

    result = {
        "recorded_frames": len(enabled),
        "duration_seconds": round(duration, 2),
        "hunger_gate": hunger_gate,
        "pickup_delta": int(delta(enabled, "mushroom_pickups_total")),
        "region_count_start": int(enabled[0].get("resource_memory_regions", 0) or 0),
        "region_count_end": int(enabled[-1].get("resource_memory_regions", 0) or 0),
        "encoding_delta": int(delta(enabled, "resource_memory_encodings")),
        "query_delta": int(delta(enabled, "resource_memory_queries")),
        "recommendation_delta": int(
            delta(enabled, "resource_memory_recommendations")
        ),
        "recommendation_transitions_observed": len(outcomes),
        "recommendations_approached_passively": sum(
            outcome["approached_region"] for outcome in outcomes
        ),
        "pickups_within_180s_of_recommendation": sum(
            outcome["pickup_within_horizon"] for outcome in outcomes
        ),
        "pickups_near_recommended_region": sum(
            outcome["pickup_near_recommended_region"] for outcome in outcomes
        ),
        "active_recommendation_frames": active_frames,
        "eligible_hungry_food_hidden_frames": eligible_frames,
        "active_fraction_when_eligible": (
            active_frames / eligible_frames if eligible_frames else 0.0
        ),
        "pickups_after_recommendation_delta": int(
            delta(enabled, "resource_memory_pickups_after_recommendation")
        ),
        "stale_arrival_delta": int(
            delta(enabled, "resource_memory_stale_arrivals")
        ),
        "maximum_action_influence": max(
            int(row.get("resource_memory_action_influence", 0) or 0)
            for row in enabled
        ),
        "release_reasons": dict(release_reasons),
        "recommendation_outcomes": outcomes,
    }
    result["checks"] = {
        "memory_encoded_at_least_one_rewarded_region": (
            result["region_count_end"] > 0 or result["encoding_delta"] > 0
        ),
        "recommendations_observed_when_testable": (
            eligible_frames == 0 or active_frames > 0
        ),
        "passive_causal_separation_preserved": (
            result["maximum_action_influence"] == 0
        ),
    }
    result["checks"]["passive_integration_valid"] = all(
        result["checks"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording",
        nargs="?",
        help="Unity JSONL recording; defaults to the latest shadow recording.",
    )
    parser.add_argument(
        "--output",
        default="outputs/resource_memory_unity_analysis.json",
    )
    args = parser.parse_args()
    if args.recording:
        path = Path(args.recording)
    else:
        candidates = sorted(Path("outputs/unity_shadow").glob("*.jsonl"))
        if not candidates:
            parser.error("no Unity shadow recordings found")
        path = candidates[-1]
    result = summarize(load_rows(path))
    result["recording"] = str(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
