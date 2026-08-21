#!/usr/bin/env python3
"""Frozen analysis for the 2026-08-21 PGNW side-blocked reserved pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path


TRIALS = (
    ("rbl_a1", "blue_left", "active", 187),
    ("rbl_p1", "blue_left", "passive", 187),
    ("ryl_p1", "blue_right", "passive", 188),
    ("ryl_a1", "blue_right", "active", 188),
    ("ryl_a2", "blue_right", "active", 189),
    ("ryl_p2", "blue_right", "passive", 189),
    ("rbl_p2", "blue_left", "passive", 190),
    ("rbl_a2", "blue_left", "active", 190),
)


def first_increment(rows, key, start_index=0):
    previous = int(rows[0].get(key, 0) or 0)
    for index, row in enumerate(rows[1:], start=1):
        value = int(row.get(key, 0) or 0)
        if index >= start_index and value > previous:
            return index
        previous = value
    return None


def mcnemar_exact_two_sided(active_wins, passive_wins):
    discordant = active_wins + passive_wins
    if discordant == 0:
        return 1.0
    smaller = min(active_wins, passive_wins)
    probability = 2.0 * sum(
        math.comb(discordant, count) / (2 ** discordant)
        for count in range(smaller + 1)
    )
    return min(1.0, probability)


def summarize_trial(path, trial_id, blue_side, condition, controller_seed):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty telemetry: {path}")
    start = rows[0]
    pickup_keys = (
        "red_mushroom_pickups_total",
        "blue_mushroom_pickups_total",
        "yellow_flower_pickups_total",
    )
    clean_start = all(int(start.get(key, 0) or 0) == 0 for key in pickup_keys)
    if not clean_start:
        raise ValueError(f"contaminated frame-zero pickup state: {path}")

    start_time = float(start["time"])
    red_index = first_increment(rows, pickup_keys[0])
    search_index = red_index if red_index is not None else len(rows)
    blue_index = first_increment(rows, pickup_keys[1], search_index)
    yellow_index = first_increment(rows, pickup_keys[2], search_index)
    if blue_index is not None and (yellow_index is None or blue_index < yellow_index):
        first_competing = "blue"
    elif yellow_index is not None and (blue_index is None or yellow_index < blue_index):
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
    competing_indices = [index for index in (blue_index, yellow_index) if index is not None]
    first_competing_index = min(competing_indices) if competing_indices else None
    prechoice_influence = 0
    if first_competing_index is not None:
        prechoice_influence = int(
            rows[max(0, first_competing_index - 1)]
            .get("pgnw_experiment", {})
            .get("arbitration_action_influence", 0)
        )
    return {
        "trial_id": trial_id,
        "blue_side": blue_side,
        "condition": condition,
        "controller_seed": controller_seed,
        "clean_start": clean_start,
        "rows": len(rows),
        "duration_seconds": round(float(final["time"]) - start_time, 6),
        "red_seconds": seconds(red_index),
        "blue_seconds": seconds(blue_index),
        "yellow_seconds": seconds(yellow_index),
        "first_competing_pickup": first_competing,
        "blue_first": first_competing == "blue",
        "blue_action_influence": blue_influence,
        "prechoice_action_influence": prechoice_influence,
        "yellow_action_influence": int(final_pgnw.get("protective_action_influence", 0)),
        "hazard_cancelled_events": int(final.get("causal_probe_cancelled_events", 0)),
        "hazard_cost_events": int(final.get("causal_probe_hunger_cost_events", 0)),
        "stuck_events": int(final.get("stuck_events", 0)),
        "survival_failures": int(final.get("survival_failures", 0)),
        "respawns": int(final.get("unstuck_respawns", 0)),
    }


def summarize_block(rows):
    active = [row for row in rows if row["condition"] == "active"]
    passive = [row for row in rows if row["condition"] == "passive"]
    return {
        "active_blue_first": sum(row["blue_first"] for row in active),
        "active_total": len(active),
        "passive_blue_first": sum(row["blue_first"] for row in passive),
        "passive_total": len(passive),
    }


def analyze(root=Path("outputs/unity_shadow")):
    results = []
    for trial_id, blue_side, condition, controller_seed in TRIALS:
        path = Path(root) / (
            f"pgnw_reserved_side_blocked_{trial_id}_seed{controller_seed}_20260821.jsonl"
        )
        if not path.exists():
            raise FileNotFoundError(f"missing preregistered trial: {path}")
        results.append(summarize_trial(path, trial_id, blue_side, condition, controller_seed))

    pairs = []
    active_wins = 0
    passive_wins = 0
    for seed in sorted({row["controller_seed"] for row in results}):
        pair_rows = [row for row in results if row["controller_seed"] == seed]
        active = next(row for row in pair_rows if row["condition"] == "active")
        passive = next(row for row in pair_rows if row["condition"] == "passive")
        if active["blue_first"] and not passive["blue_first"]:
            discordance = "active_win"
            active_wins += 1
        elif passive["blue_first"] and not active["blue_first"]:
            discordance = "passive_win"
            passive_wins += 1
        else:
            discordance = "concordant"
        pairs.append({
            "controller_seed": seed,
            "blue_side": active["blue_side"],
            "active_blue_first": active["blue_first"],
            "passive_blue_first": passive["blue_first"],
            "discordance": discordance,
        })

    active = [row for row in results if row["condition"] == "active"]
    passive = [row for row in results if row["condition"] == "passive"]
    return {
        "trial_order": [row["trial_id"] for row in results],
        "trials": results,
        "primary": {
            "active_blue_first": sum(row["blue_first"] for row in active),
            "active_total": len(active),
            "passive_blue_first": sum(row["blue_first"] for row in passive),
            "passive_total": len(passive),
            "active_motor_influence_total": sum(row["blue_action_influence"] for row in active),
            "passive_motor_influence_total": sum(row["blue_action_influence"] for row in passive),
            "active_prechoice_influence_total": sum(
                row["prechoice_action_influence"] for row in active
            ),
            "passive_prechoice_influence_total": sum(
                row["prechoice_action_influence"] for row in passive
            ),
        },
        "blocks": {
            side: summarize_block([row for row in results if row["blue_side"] == side])
            for side in ("blue_left", "blue_right")
        },
        "paired": {
            "pairs": pairs,
            "active_wins": active_wins,
            "passive_wins": passive_wins,
            "mcnemar_exact_two_sided_p": round(
                mcnemar_exact_two_sided(active_wins, passive_wins), 6
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
