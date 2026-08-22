#!/usr/bin/env python3
"""Compare framewise, fixed-GNW, and adaptive-GNW Unity recommendations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tiny_lab import OUT


CONDITIONS = {
    "framewise": "systemic_conductor_raw_recommendation",
    "fixed_gnw": "systemic_conductor_recommendation",
    "adaptive_gnw": "adaptive_gnw_active_specialist",
}


def rate(values):
    return sum(values) / len(values) if values else 0.0


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("adaptive_gnw_mode")
                in {"passive_reward_calibrated", "passive"}
                and row.get("systemic_conductor_proxy_optimal")
                in {"recurrent", "episodic", "predictive", "fallback"}
            ):
                rows.append(row)
    return rows


def summarize_condition(rows, key):
    recommendations = [str(row.get(key, "unobserved")) for row in rows]
    optimal = [
        str(row["systemic_conductor_proxy_optimal"]) for row in rows
    ]
    correct = [
        recommendation == target
        for recommendation, target in zip(recommendations, optimal)
    ]
    handoffs = sum(
        current != previous
        for previous, current in zip(recommendations, recommendations[1:])
    )
    boundary = [
        index == 0
        or optimal[index] != optimal[index - 1]
        or (
            index > 1
            and optimal[index - 1] != optimal[index - 2]
        )
        for index in range(len(rows))
    ]
    contexts = {}
    for context in ("recurrent", "episodic", "predictive", "fallback"):
        selected = [
            value
            for value, target in zip(correct, optimal)
            if target == context
        ]
        contexts[context] = rate(selected)
    return {
        "agreement_rate": rate(correct),
        "boundary_agreement_rate": rate(
            [value for value, edge in zip(correct, boundary) if edge]
        ),
        "steady_agreement_rate": rate(
            [value for value, edge in zip(correct, boundary) if not edge]
        ),
        "handoffs": handoffs,
        "handoffs_per_1000_frames": 1000.0 * handoffs / max(len(rows), 1),
        "context_agreement": contexts,
    }


def transient_challenge_metrics(rows):
    raw = [
        str(row.get("systemic_conductor_raw_recommendation", "unobserved"))
        for row in rows
    ]
    adaptive = [
        str(row.get("adaptive_gnw_active_specialist", "unobserved"))
        for row in rows
    ]
    confidence = [
        max(
            [
                float(value)
                for value in row.get(
                    "systemic_conductor_raw_probabilities", [0.0]
                )
            ]
        )
        for row in rows
    ]
    transient_indices = [
        index
        for index in range(1, len(rows) - 1)
        if raw[index - 1] == raw[index + 1] != raw[index]
    ]
    held = [
        adaptive[index] == adaptive[index - 1]
        for index in transient_indices
    ]
    low_confidence_held = [
        adaptive[index] == adaptive[index - 1]
        for index in transient_indices
        if confidence[index] < 0.55
    ]
    return {
        "one_frame_raw_challenges": len(transient_indices),
        "adaptive_hold_rate": rate(held),
        "low_confidence_one_frame_challenges": len(low_confidence_held),
        "low_confidence_hold_rate": rate(low_confidence_held),
    }


def analyze(path):
    rows = load_rows(path)
    if not rows:
        raise ValueError("no adaptive GNW passive rows found")
    summaries = {
        name: summarize_condition(rows, key)
        for name, key in CONDITIONS.items()
    }
    transient = transient_challenge_metrics(rows)
    final = rows[-1]
    result = {
        "source": str(Path(path).resolve()),
        "frames": len(rows),
        "conditions": summaries,
        "transient_challenges": transient,
        "adaptive_events": dict(
            Counter(str(row.get("adaptive_gnw_event")) for row in rows)
        ),
        "adaptive_ignitions": int(final.get("adaptive_gnw_ignitions", 0)),
        "adaptive_releases": int(final.get("adaptive_gnw_releases", 0)),
        "adaptive_held_challenges": int(
            final.get("adaptive_gnw_held_challenges", 0)
        ),
        "maximum_action_influence": max(
            int(row.get("adaptive_gnw_action_influence", 0)) for row in rows
        ),
    }
    result["comparisons"] = {
        "adaptive_minus_fixed_agreement": (
            summaries["adaptive_gnw"]["agreement_rate"]
            - summaries["fixed_gnw"]["agreement_rate"]
        ),
        "adaptive_minus_framewise_agreement": (
            summaries["adaptive_gnw"]["agreement_rate"]
            - summaries["framewise"]["agreement_rate"]
        ),
        "adaptive_minus_fixed_boundary_agreement": (
            summaries["adaptive_gnw"]["boundary_agreement_rate"]
            - summaries["fixed_gnw"]["boundary_agreement_rate"]
        ),
        "adaptive_handoff_reduction_vs_framewise": (
            summaries["framewise"]["handoffs_per_1000_frames"]
            - summaries["adaptive_gnw"]["handoffs_per_1000_frames"]
        ),
    }
    result["criteria"] = {
        "passive_causal_separation": result["maximum_action_influence"] == 0,
        "adaptive_improves_fixed_gnw": (
            result["comparisons"]["adaptive_minus_fixed_agreement"] > 0.0
        ),
        "adaptive_preserves_framewise_accuracy": (
            result["comparisons"]["adaptive_minus_framewise_agreement"] >= -0.03
        ),
        "adaptive_reduces_framewise_handoffs": (
            result["comparisons"]["adaptive_handoff_reduction_vs_framewise"]
            > 0.0
        ),
        "adaptive_holds_low_confidence_challenges": (
            transient["low_confidence_one_frame_challenges"] > 0
            and transient["low_confidence_hold_rate"] >= 0.80
        ),
    }
    result["all_criteria_pass"] = all(result["criteria"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument(
        "--output",
        default=str(OUT / "adaptive_gnw_unity_metrics.json"),
    )
    args = parser.parse_args()
    result = analyze(args.recording)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["conditions"].items():
        print(
            f"{name}: agreement={metrics['agreement_rate']:.3f} "
            f"boundary={metrics['boundary_agreement_rate']:.3f} "
            f"handoffs/1000={metrics['handoffs_per_1000_frames']:.1f}"
        )
    for name, passed in result["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {result['all_criteria_pass']}")
    print(f"wrote {output.resolve()}")


if __name__ == "__main__":
    main()
