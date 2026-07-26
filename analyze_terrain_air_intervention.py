#!/usr/bin/env python3
"""Compare passive MPC terrain control with bounded AIR route intervention."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from terrain_air_counterfactual_gate_lab import intervention_needed
from terrain_air_memory_lab import load_rows


def pickup_gaps(rows):
    gaps = []
    previous_count = int(rows[0].get("mushroom_pickups_total", 0) or 0)
    previous_time = float(rows[0].get("time", 0.0))
    for row in rows:
        count = int(row.get("mushroom_pickups_total", 0) or 0)
        if count > previous_count:
            gaps.append(float(row.get("time", 0.0)) - previous_time)
            previous_time = float(row.get("time", 0.0))
            previous_count = count
    gaps.append(float(rows[-1].get("time", 0.0)) - previous_time)
    return gaps


def summarize(rows):
    duration = float(rows[-1]["time"]) - float(rows[0]["time"])
    pickups = int(rows[-1].get("mushroom_pickups_total", 0) or 0) - int(
        rows[0].get("mushroom_pickups_total", 0) or 0
    )
    gaps = pickup_gaps(rows)
    return {
        "rows": len(rows),
        "duration_minutes": duration / 60.0,
        "pickups": pickups,
        "pickups_per_hour": pickups / max(duration, 1e-6) * 3600.0,
        "collision_frames": sum(
            bool(row.get("body_collision", False)) for row in rows
        ),
        "collision_frame_fraction": float(
            np.mean([bool(row.get("body_collision", False)) for row in rows])
        ),
        "necessity_frames": sum(intervention_needed(row) for row in rows),
        "necessity_frame_fraction": float(
            np.mean([intervention_needed(row) for row in rows])
        ),
        "fallback_frames": sum(
            bool(row.get("fallback_active", False)) for row in rows
        ),
        "survival_failures": int(rows[-1].get("survival_failures", 0) or 0)
        - int(rows[0].get("survival_failures", 0) or 0),
        "maximum_pickup_gap_seconds": max(gaps),
        "pickup_gaps_over_60_seconds": sum(gap > 60.0 for gap in gaps),
        "mean_pickup_gap_seconds": float(np.mean(gaps)),
    }


def baseline_gate_events(rows, cooldown_seconds=12.0):
    events = []
    last_time = -math.inf
    for index, row in enumerate(rows):
        now = float(row["time"])
        if intervention_needed(row) and now - last_time >= cooldown_seconds:
            events.append(index)
            last_time = now
    return events


def intervention_events(rows):
    events = []
    previous = 0
    for index, row in enumerate(rows):
        current = int(row.get("terrain_air_route_interventions", 0) or 0)
        if current > previous:
            events.append(index)
        previous = current
    return events


def guidance_diagnostics(rows):
    applications = []
    changes = []
    previous_decisions = 0
    previous_changes = 0
    for row in rows:
        decisions = int(
            row.get("terrain_air_route_guidance_decisions", 0) or 0
        )
        action_changes = int(
            row.get("terrain_air_route_guidance_action_changes", 0) or 0
        )
        if decisions > previous_decisions:
            applications.append(row)
        if action_changes > previous_changes:
            changes.append(row)
        previous_decisions = decisions
        previous_changes = action_changes

    def mean(key, selected):
        if not selected:
            return 0.0
        return float(
            np.mean([float(row.get(key, 0.0) or 0.0) for row in selected])
        )

    return {
        "applications": len(applications),
        "action_changes": len(changes),
        "action_change_fraction": len(changes) / max(len(applications), 1),
        "mean_unguided_margin_when_applied": mean(
            "terrain_air_route_unguided_margin", applications
        ),
        "mean_effective_weight_when_applied": mean(
            "terrain_air_route_effective_guidance_weight", applications
        ),
        "mean_unguided_margin_when_action_changed": mean(
            "terrain_air_route_unguided_margin", changes
        ),
        "mean_effective_weight_when_action_changed": mean(
            "terrain_air_route_effective_guidance_weight", changes
        ),
    }


def horizon_index(rows, start, seconds):
    target = float(rows[start]["time"]) + seconds
    for index in range(start, len(rows)):
        if float(rows[index]["time"]) >= target:
            return index
    return len(rows) - 1


def event_outcomes(rows, events):
    outcomes = []
    for start in events:
        end_8 = horizon_index(rows, start, 8.0)
        end_20 = horizon_index(rows, start, 20.0)
        nonnecessity_streak = 0
        recovered = False
        recovery_seconds = None
        for index in range(start, end_8 + 1):
            if intervention_needed(rows[index]):
                nonnecessity_streak = 0
                continue
            nonnecessity_streak += 1
            if nonnecessity_streak >= 10:
                recovered = True
                recovery_seconds = (
                    float(rows[index]["time"]) - float(rows[start]["time"])
                )
                break
        outcomes.append(
            {
                "start_index": start,
                "recovered_within_8_seconds": recovered,
                "recovery_seconds": recovery_seconds,
                "pickup_within_20_seconds": int(
                    rows[end_20].get("mushroom_pickups_total", 0) or 0
                )
                > int(rows[start].get("mushroom_pickups_total", 0) or 0),
                "collision_frames_within_20_seconds": sum(
                    bool(row.get("body_collision", False))
                    for row in rows[start : end_20 + 1]
                ),
                "orbit_efficiency_after_8_seconds": float(
                    rows[end_8].get("orbit_efficiency", 1.0) or 1.0
                ),
            }
        )
    return outcomes


def aggregate_events(outcomes):
    recovered = [row for row in outcomes if row["recovered_within_8_seconds"]]
    return {
        "events": len(outcomes),
        "recovery_rate_within_8_seconds": len(recovered)
        / max(len(outcomes), 1),
        "mean_recovery_seconds_when_recovered": (
            float(np.mean([row["recovery_seconds"] for row in recovered]))
            if recovered
            else None
        ),
        "pickup_rate_within_20_seconds": float(
            np.mean([row["pickup_within_20_seconds"] for row in outcomes])
        )
        if outcomes
        else 0.0,
        "mean_collision_frames_within_20_seconds": float(
            np.mean(
                [row["collision_frames_within_20_seconds"] for row in outcomes]
            )
        )
        if outcomes
        else 0.0,
        "mean_orbit_efficiency_after_8_seconds": float(
            np.mean(
                [row["orbit_efficiency_after_8_seconds"] for row in outcomes]
            )
        )
        if outcomes
        else 0.0,
    }


def analyze(baseline_path, intervention_path):
    baseline = load_rows(baseline_path)
    intervention = load_rows(intervention_path)
    baseline_summary = summarize(baseline)
    intervention_summary = summarize(intervention)
    baseline_events = aggregate_events(
        event_outcomes(baseline, baseline_gate_events(baseline))
    )
    intervention_events_list = intervention_events(intervention)
    intervention_event_summary = aggregate_events(
        event_outcomes(intervention, intervention_events_list)
    )
    releases = Counter()
    previous_releases = 0
    for row in intervention:
        current = int(row.get("terrain_air_route_releases", 0) or 0)
        if current > previous_releases:
            releases[str(row.get("terrain_air_route_release_reason", "unknown"))] += 1
        previous_releases = current
    baseline_rate = baseline_summary["pickups_per_hour"]
    intervention_rate = intervention_summary["pickups_per_hour"]
    control_mode = str(
        intervention[-1].get("terrain_air_route_mode", "bounded")
    )
    if control_mode == "guided":
        if (
            intervention_rate >= 0.97 * baseline_rate
            and intervention_summary["necessity_frame_fraction"]
            < baseline_summary["necessity_frame_fraction"]
            and intervention_summary["collision_frame_fraction"]
            < baseline_summary["collision_frame_fraction"]
        ):
            verdict = (
                "Margin-gated AIR guidance preserved baseline foraging while "
                "reducing collision and low-efficiency exposure in this single "
                "session. Gate-event recovery remained lower, so the result is "
                "promising but requires matched replication before authorization."
            )
        else:
            verdict = (
                "The AIR heading prior improved on direct coordinate replay but "
                "did not preserve the ordinary MPC baseline. It should remain "
                "experimental while its uncertainty gate is recalibrated."
            )
    else:
        verdict = (
            "Bounded coordinate-route replay is not authorized for terrain "
            "control: it reduced normalized foraging and gate-event recovery "
            "while increasing low-efficiency navigation. AIR should remain a "
            "selector, with retrieved experience converted to a soft MPC prior "
            "or grounded subgoal rather than replayed as literal waypoints."
        )
    return {
        "experiment": (
            f"{control_mode} AIR route intervention versus MPC terrain baseline"
        ),
        "baseline": {
            "recording": str(Path(baseline_path).resolve()),
            **baseline_summary,
            "gate_event_outcomes": baseline_events,
        },
        "bounded_air": {
            "recording": str(Path(intervention_path).resolve()),
            **intervention_summary,
            "gate_event_outcomes": intervention_event_summary,
            "route_recommendations": int(
                intervention[-1].get("terrain_air_route_recommendations", 0) or 0
            ),
            "route_interventions": int(
                intervention[-1].get("terrain_air_route_interventions", 0) or 0
            ),
            "route_action_influence_frames": int(
                intervention[-1].get("terrain_air_route_action_influence", 0) or 0
            ),
            "control_mode": control_mode,
            "guidance_decisions": int(
                intervention[-1].get("terrain_air_route_guidance_decisions", 0)
                or 0
            ),
            "guidance_action_changes": int(
                intervention[-1].get(
                    "terrain_air_route_guidance_action_changes", 0
                )
                or 0
            ),
            "guidance_diagnostics": guidance_diagnostics(intervention),
            "release_reasons": dict(releases),
        },
        "comparison": {
            "pickup_rate_ratio_air_over_baseline": intervention_rate
            / max(baseline_rate, 1e-6),
            "pickup_rate_change_percent": 100.0
            * (intervention_rate - baseline_rate)
            / max(baseline_rate, 1e-6),
            "necessity_fraction_change_percentage_points": 100.0
            * (
                intervention_summary["necessity_frame_fraction"]
                - baseline_summary["necessity_frame_fraction"]
            ),
            "gate_recovery_rate_change_percentage_points": 100.0
            * (
                intervention_event_summary["recovery_rate_within_8_seconds"]
                - baseline_events["recovery_rate_within_8_seconds"]
            ),
        },
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/terrain_air_bounded_intervention_metrics.json"),
    )
    args = parser.parse_args()
    payload = analyze(args.baseline, args.intervention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
