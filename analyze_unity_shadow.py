#!/usr/bin/env python3
"""Summarize passive Unity shadow-policy recordings without training on them."""

import argparse
import json
from pathlib import Path

SHADOW_VECTORS = {
    "up": (0.0, 1.0), "up_right": (1.0, 1.0), "right": (1.0, 0.0), "down_right": (1.0, -1.0),
    "down": (0.0, -1.0), "down_left": (-1.0, -1.0), "left": (-1.0, 0.0), "up_left": (-1.0, 1.0),
}


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize_context(rows):
    if not rows:
        return {"frames": 0, "fraction": 0.0}
    return {
        "frames": len(rows),
        "deprecated_command_label_agreement": mean(rows, "shadow_agreement"),
        "deprecated_exact_action_agreement": sum(row["active_action"] == row["shadow_action"] for row in rows) / len(rows),
        "mean_shadow_confidence": mean(rows, "shadow_confidence"),
    }


def summarize_world_motion(rows, predicate):
    agreements = []
    for current, following in zip(rows, rows[1:]):
        if not predicate(current):
            continue
        dx = float(following["position"][0]) - float(current["position"][0])
        dz = float(following["position"][2]) - float(current["position"][2])
        distance = (dx * dx + dz * dz) ** 0.5
        if distance < 0.01:
            continue
        sx, sz = SHADOW_VECTORS[current["shadow_action"]]
        shadow_norm = (sx * sx + sz * sz) ** 0.5
        cosine = (dx * sx + dz * sz) / (distance * shadow_norm)
        agreements.append(0.5 * (cosine + 1.0))
    return {
        "moving_frames": len(agreements),
        "mean_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
        "agreement_at_0_85": sum(value >= 0.85 for value in agreements) / len(agreements) if agreements else 0.0,
    }


def analyze(path):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) < 2:
        raise ValueError("Shadow recording needs at least two rows")

    friction = lambda row: (
        row["blocked"] or row["body_collision"] or row["stuck"] or row["trap_pressure"] >= 0.35
    )
    predicates = {
        "all": lambda row: True,
        "food_visible": lambda row: row["food_visible"],
        "physical_collision": lambda row: row["body_collision"],
        "reported_stuck": lambda row: row["stuck"],
        "breakout": lambda row: row["active_intent"] == "breakout_arc",
        "open_field": lambda row: not row["food_visible"] and not friction(row),
    }
    contexts = {name: [row for row in rows if predicate(row)] for name, predicate in predicates.items()}
    context_summary = {name: summarize_context(selected) for name, selected in contexts.items()}
    for name, summary in context_summary.items():
        summary["fraction"] = summary["frames"] / len(rows)
        summary["world_motion_agreement"] = summarize_world_motion(rows, predicates[name])

    pickup_indices = []
    previous_pickups = rows[0]["mushroom_pickups_total"]
    for index, row in enumerate(rows[1:], 1):
        if row["mushroom_pickups_total"] > previous_pickups:
            pickup_indices.append(index)
        previous_pickups = row["mushroom_pickups_total"]
    pre_pickup = []
    for index in pickup_indices:
        pre_pickup.extend(row for row in rows[max(0, index - 15):index] if row["food_visible"])

    warnings = []
    if context_summary["reported_stuck"]["fraction"] > 0.50:
        warnings.append("reported_stuck exceeds 50%; do not use it as a training label until recalibrated")
    if context_summary["breakout"]["fraction"] > 0.50:
        warnings.append("breakout intent exceeds 50%; controller state is likely over-latching")
    if not any(row["blocked"] for row in rows) and any(row["body_collision"] for row in rows):
        warnings.append("blocked never fired despite physical collisions; forward-clear telemetry is not a contact label")

    return {
        "source": str(path),
        "frames": len(rows),
        "duration_minutes": (rows[-1]["time"] - rows[0]["time"]) / 60.0,
        "pickup_delta": rows[-1]["mushroom_pickups_total"] - rows[0]["mushroom_pickups_total"],
        "pickup_events": len(pickup_indices),
        "physical_collision_frames": sum(row["body_collision"] for row in rows),
        "pre_pickup_15_frame_window": summarize_context(pre_pickup),
        "contexts": context_summary,
        "telemetry_warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", help="JSONL recording; defaults to the latest Unity shadow run")
    parser.add_argument("--output", default="outputs/unity_shadow_analysis.json")
    args = parser.parse_args()
    if args.recording:
        path = Path(args.recording)
    else:
        candidates = sorted(Path("outputs/unity_shadow").glob("*.jsonl"))
        if not candidates:
            parser.error("no Unity shadow recordings found")
        path = candidates[-1]
    summary = analyze(path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
