#!/usr/bin/env python3
"""Frozen analysis for the 2026-08-21 PGNW active/passive pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path


TRIALS = (
    ("a1", "active", 181),
    ("p1", "passive", 181),
    ("p2", "passive", 182),
    ("a2", "active", 182),
    ("a3", "active", 183),
    ("p3", "passive", 183),
    ("p4", "passive", 184),
    ("a4", "active", 184),
)


def first_increment(rows, key, start_index=0):
    previous = int(rows[0].get(key, 0) or 0)
    for index, row in enumerate(rows[1:], start=1):
        value = int(row.get(key, 0) or 0)
        if index >= start_index and value > previous:
            return index
        previous = value
    return None


def summarize_trial(path, trial_id, condition, controller_seed):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty telemetry: {path}")
    start_time = float(rows[0]["time"])
    red_index = first_increment(rows, "red_mushroom_pickups_total")
    search_index = red_index if red_index is not None else len(rows)
    blue_index = first_increment(
        rows, "blue_mushroom_pickups_total", search_index
    )
    yellow_index = first_increment(
        rows, "yellow_flower_pickups_total", search_index
    )
    if blue_index is not None and (
        yellow_index is None or blue_index < yellow_index
    ):
        first_competing = "blue"
    elif yellow_index is not None and (
        blue_index is None or yellow_index < blue_index
    ):
        first_competing = "yellow"
    elif blue_index is not None and blue_index == yellow_index:
        first_competing = "tie"
    else:
        first_competing = "none"
    final = rows[-1]
    final_pgnw = final.get("pgnw_experiment", {})

    def seconds(index):
        return None if index is None else round(float(rows[index]["time"]) - start_time, 6)

    blue_influence = 0
    if blue_index is not None:
        blue_influence = int(
            rows[blue_index]
            .get("pgnw_experiment", {})
            .get("arbitration_action_influence", 0)
        )
    yellow_influence = int(final_pgnw.get("protective_action_influence", 0))
    complete_order = bool(
        red_index is not None
        and blue_index is not None
        and yellow_index is not None
        and red_index < blue_index < yellow_index
    )
    return {
        "trial_id": trial_id,
        "condition": condition,
        "controller_seed": controller_seed,
        "rows": len(rows),
        "duration_seconds": round(float(final["time"]) - start_time, 6),
        "red_seconds": seconds(red_index),
        "blue_seconds": seconds(blue_index),
        "yellow_seconds": seconds(yellow_index),
        "first_competing_pickup": first_competing,
        "blue_first": first_competing == "blue",
        "complete_red_blue_yellow": complete_order,
        "blue_action_influence": blue_influence,
        "yellow_action_influence": yellow_influence,
        "hazard_cancelled_events": int(final.get("causal_probe_cancelled_events", 0)),
        "hazard_cost_events": int(final.get("causal_probe_hunger_cost_events", 0)),
        "stuck_events": int(final.get("stuck_events", 0)),
        "survival_failures": int(final.get("survival_failures", 0)),
        "respawns": int(final.get("unstuck_respawns", 0)),
    }


def fisher_two_sided(active_successes, active_total, passive_successes, passive_total):
    total_successes = active_successes + passive_successes
    total = active_total + passive_total

    def probability(active_count):
        return (
            math.comb(active_total, active_count)
            * math.comb(passive_total, total_successes - active_count)
            / math.comb(total, total_successes)
        )

    observed = probability(active_successes)
    low = max(0, total_successes - passive_total)
    high = min(active_total, total_successes)
    result = sum(
        probability(count)
        for count in range(low, high + 1)
        if probability(count) <= observed + 1e-12
    )
    return 1.0 if result >= 1.0 - 1e-12 else result


def analyze(root=Path("outputs/unity_shadow")):
    trial_results = []
    for trial_id, condition, controller_seed in TRIALS:
        filename = (
            f"pgnw_active_passive_{trial_id}_seed{controller_seed}_20260821.jsonl"
        )
        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"missing preregistered trial: {path}")
        trial_results.append(
            summarize_trial(path, trial_id, condition, controller_seed)
        )
    active = [row for row in trial_results if row["condition"] == "active"]
    passive = [row for row in trial_results if row["condition"] == "passive"]
    active_successes = sum(row["blue_first"] for row in active)
    passive_successes = sum(row["blue_first"] for row in passive)
    return {
        "trial_order": [row["trial_id"] for row in trial_results],
        "trials": trial_results,
        "primary": {
            "active_blue_first": active_successes,
            "active_total": len(active),
            "passive_blue_first": passive_successes,
            "passive_total": len(passive),
            "absolute_rate_difference": round(
                active_successes / len(active) - passive_successes / len(passive),
                6,
            ),
            "fisher_exact_two_sided_p": round(
                fisher_two_sided(
                    active_successes,
                    len(active),
                    passive_successes,
                    len(passive),
                ),
                6,
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
