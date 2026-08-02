#!/usr/bin/env python3
"""Compare passive-baseline and bounded systemic-routing Unity recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path):
    rows = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 2:
        raise ValueError(f"recording_too_short:{path}")
    return rows


def delta(rows, key):
    return float(rows[-1].get(key, 0) or 0) - float(
        rows[0].get(key, 0) or 0
    )


def summarize(rows):
    duration = max(
        float(rows[-1].get("time", 0)) - float(rows[0].get("time", 0)),
        1e-6,
    )
    hours = duration / 3600.0
    minutes = duration / 60.0
    pickups = max(0.0, delta(rows, "mushroom_pickups_total"))
    collisions = sum(bool(row.get("body_collision", False)) for row in rows)
    mpc_frames = sum(bool(row.get("shadow_mpc_engaged", False)) for row in rows)
    fallback_frames = sum(bool(row.get("fallback_active", False)) for row in rows)
    mpc_states = [
        bool(row.get("shadow_mpc_engaged", False)) for row in rows
    ]
    mpc_transitions = sum(
        previous != current
        for previous, current in zip(mpc_states, mpc_states[1:])
    )
    one_frame_reversals = sum(
        mpc_states[index] == mpc_states[index - 2]
        and mpc_states[index] != mpc_states[index - 1]
        for index in range(2, len(mpc_states))
    )
    last = rows[-1]
    return {
        "frames": len(rows),
        "duration_seconds": duration,
        "pickups": pickups,
        "pickups_per_hour": pickups / hours,
        "survival_failures": max(0.0, delta(rows, "survival_failures")),
        "stuck_events": max(0.0, delta(rows, "stuck_events")),
        "unstuck_respawns": max(0.0, delta(rows, "unstuck_respawns")),
        "collision_frame_rate": collisions / len(rows),
        "mpc_frame_rate": mpc_frames / len(rows),
        "mpc_transitions": mpc_transitions,
        "mpc_transitions_per_minute": mpc_transitions / minutes,
        "one_frame_reversals": one_frame_reversals,
        "one_frame_reversals_per_minute": one_frame_reversals / minutes,
        "fallback_frame_rate": fallback_frames / len(rows),
        "router_handoffs": int(last.get("systemic_router_handoffs", 0) or 0),
        "router_handoffs_per_minute": (
            float(last.get("systemic_router_handoffs", 0) or 0) / minutes
        ),
        "router_chatter_events": int(
            last.get("systemic_router_chatter_events", 0) or 0
        ),
        "router_safety_overrides": int(
            last.get("systemic_router_safety_overrides", 0) or 0
        ),
        "router_influence_frames": int(
            last.get("systemic_router_influence_frames", 0) or 0
        ),
    }


def compare(baseline, active):
    baseline_summary = summarize(load_rows(baseline))
    active_summary = summarize(load_rows(active))
    pickup_floor = baseline_summary["pickups_per_hour"] * 0.90
    collision_ceiling = max(
        baseline_summary["collision_frame_rate"] * 1.10,
        baseline_summary["collision_frame_rate"] + 0.002,
    )
    criteria = {
        "router_had_causal_influence": (
            active_summary["router_influence_frames"] > 0
        ),
        "no_additional_survival_failures": (
            active_summary["survival_failures"]
            <= baseline_summary["survival_failures"]
        ),
        "pickup_rate_preserved_within_10_percent": (
            active_summary["pickups_per_hour"] >= pickup_floor
        ),
        "collision_rate_not_materially_worse": (
            active_summary["collision_frame_rate"] <= collision_ceiling
        ),
        "handoff_rate_not_materially_worse": (
            active_summary["mpc_transitions_per_minute"]
            <= baseline_summary["mpc_transitions_per_minute"] * 1.25
        ),
        "one_frame_reversals_not_materially_worse": (
            active_summary["one_frame_reversals_per_minute"]
            <= max(
                baseline_summary["one_frame_reversals_per_minute"] * 1.50,
                baseline_summary["one_frame_reversals_per_minute"] + 0.50,
            )
        ),
    }
    return {
        "experiment": "bounded systemic recurrent-MPC routing",
        "baseline_source": str(baseline),
        "active_source": str(active),
        "baseline": baseline_summary,
        "active": active_summary,
        "deltas": {
            "pickups_per_hour": (
                active_summary["pickups_per_hour"]
                - baseline_summary["pickups_per_hour"]
            ),
            "collision_frame_rate": (
                active_summary["collision_frame_rate"]
                - baseline_summary["collision_frame_rate"]
            ),
            "mpc_frame_rate": (
                active_summary["mpc_frame_rate"]
                - baseline_summary["mpc_frame_rate"]
            ),
        },
        "criteria": criteria,
        "bounded_routing_supported": all(criteria.values()),
        "claim_boundary": (
            "A passing single-run comparison supports bounded executive timing "
            "under this terrain realization. It does not establish a general "
            "performance advantage without matched-seed repeated runs."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("active")
    parser.add_argument(
        "--output",
        default="outputs/systemic_routing_unity_metrics.json",
    )
    args = parser.parse_args()
    metrics = compare(args.baseline, args.active)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"baseline_pickups_h={metrics['baseline']['pickups_per_hour']:.2f} "
        f"active_pickups_h={metrics['active']['pickups_per_hour']:.2f}"
    )
    print(
        f"baseline_collision={metrics['baseline']['collision_frame_rate']:.4f} "
        f"active_collision={metrics['active']['collision_frame_rate']:.4f}"
    )
    for name, passed in metrics["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(
        "bounded_systemic_routing_supported: "
        f"{'PASS' if metrics['bounded_routing_supported'] else 'FAIL'}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
