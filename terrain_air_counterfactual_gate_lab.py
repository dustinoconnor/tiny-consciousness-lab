#!/usr/bin/env python3
"""Offline conservative gate for AIR recall versus active terrain control.

The first Unity recording supplies action-conditioned empirical precedents.
The second recording is untouched evaluation. At AIR/active disagreements, a
k-nearest-neighbor outcome model estimates delayed utility for each action.
AIR is nominated only when ART resonates and AIR's conservative value exceeds
the active action. This lab never sends commands to Unity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from terrain_air_memory_lab import (
    ACTION_INDEX,
    TerrainPacket,
    build_packets,
    load_rows,
)
from terrain_air_observer import PassiveTerrainAirObserver


def state_vector(packet):
    return np.r_[
        packet.retrieval_vector(),
        packet.features[1],  # local prediction error
        packet.features[4],  # target evidence
    ]


class ActionConditionedKnn:
    def __init__(self, packets, neighbors=24):
        self.neighbors = neighbors
        self.by_action = {}
        matrix = np.stack([state_vector(packet) for packet in packets])
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        for action in ACTION_INDEX:
            selected = [packet for packet in packets if packet.action == action]
            if not selected:
                continue
            self.by_action[action] = (
                (np.stack([state_vector(packet) for packet in selected]) - self.mean)
                / self.scale,
                np.asarray([packet.utility for packet in selected]),
            )

    def estimate(self, packet, action):
        if action not in self.by_action:
            return None
        states, outcomes = self.by_action[action]
        query = (state_vector(packet) - self.mean) / self.scale
        distance = np.sqrt(np.mean((states - query) ** 2, axis=1))
        count = min(self.neighbors, len(distance))
        indices = np.argpartition(distance, count - 1)[:count]
        local_distance = distance[indices]
        weights = 1.0 / np.maximum(local_distance, 0.03)
        values = outcomes[indices]
        mean = float(np.sum(weights * values) / np.sum(weights))
        variance = float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
        return {
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)),
            "distance": float(np.mean(local_distance)),
            "support": count,
        }


def observe_recall(observer, packet):
    body = {
        "directional_rays": packet.rays.tolist(),
        "directional_body_clearance": packet.body_clearance.tolist(),
    }
    observer.update(body, packet.hunger, packet.action)
    return {
        "resonance": observer.resonance,
        "action": observer.recalled_action,
        "distance": observer.distance,
        "confidence": observer.confidence,
    }


def lower_bound(estimate):
    return (
        estimate["mean"]
        - 0.50 * estimate["std"]
        - 0.20 * estimate["distance"]
    )


def active_reference(estimate):
    return estimate["mean"] - 0.10 * estimate["std"]


def intervention_needed(row):
    wedge = float(row.get("physics_wedge_seconds", 0.0) or 0.0) >= 1.0
    trap = float(row.get("trap_accumulation_seconds", 0.0) or 0.0) >= 2.0
    inefficient_orbit = (
        float(row.get("orbit_path", 0.0) or 0.0) >= 4.0
        and float(row.get("orbit_efficiency", 1.0) or 1.0) <= 0.25
    )
    return bool(wedge or trap or inefficient_orbit)


def evaluate(
    training_recording,
    evaluation_recording,
    memory_checkpoint,
    margin=0.03,
):
    training_packets = build_packets(load_rows(training_recording))
    evaluation_rows = load_rows(evaluation_recording)
    evaluation_packets = build_packets(evaluation_rows)
    model = ActionConditionedKnn(training_packets)
    observer = PassiveTerrainAirObserver(memory_checkpoint)

    decisions = []
    active_errors = []
    for packet in evaluation_packets:
        recalled = observe_recall(observer, packet)
        if not recalled["resonance"] or recalled["action"] not in ACTION_INDEX:
            continue
        active = model.estimate(packet, packet.action)
        air = model.estimate(packet, recalled["action"])
        if active is None or air is None:
            continue
        active_errors.append(abs(active["mean"] - packet.utility))
        row = evaluation_rows[packet.source_index]
        art_grounded = (
            bool(row.get("art_resonance", False))
            and not bool(row.get("art_unknown", False))
            and float(row.get("art_match", 0.0) or 0.0) >= 0.80
        )
        necessity = intervention_needed(row)
        disagreement = recalled["action"] != packet.action
        advantage = lower_bound(air) - active_reference(active)
        select_air = (
            disagreement
            and art_grounded
            and necessity
            and recalled["confidence"] >= 0.35
            and air["distance"] <= 1.25
            and advantage >= margin
        )
        decisions.append(
            {
                "active_action": packet.action,
                "air_action": recalled["action"],
                "disagreement": disagreement,
                "art_grounded": art_grounded,
                "intervention_needed": necessity,
                "air_selected": select_air,
                "active_value": active["mean"],
                "air_value": air["mean"],
                "active_conservative": active_reference(active),
                "air_lower_bound": lower_bound(air),
                "advantage": advantage,
                "realized_active_utility": packet.utility,
            }
        )

    disagreements = [row for row in decisions if row["disagreement"]]
    selected = [row for row in decisions if row["air_selected"]]
    def estimated_value(mode):
        values = []
        for row in decisions:
            if mode == "active":
                values.append(row["active_value"])
            elif mode == "air":
                values.append(row["air_value"])
            else:
                values.append(
                    row["air_value"] if row["air_selected"] else row["active_value"]
                )
        return float(np.mean(values)) if values else 0.0

    rng = np.random.default_rng(9137)
    bootstrap = []
    if decisions:
        for _ in range(2000):
            indices = rng.integers(0, len(decisions), len(decisions))
            gain = np.mean(
                [
                    (
                        decisions[index]["air_value"]
                        - decisions[index]["active_value"]
                        if decisions[index]["air_selected"]
                        else 0.0
                    )
                    for index in indices
                ]
            )
            bootstrap.append(float(gain))

    gate_value = estimated_value("gate")
    active_value = estimated_value("active")
    return {
        "experiment": "terrain AIR conservative counterfactual gate",
        "protocol": {
            "training_recording": str(Path(training_recording).resolve()),
            "evaluation_recording": str(Path(evaluation_recording).resolve()),
            "memory_checkpoint": str(Path(memory_checkpoint).resolve()),
            "training_packets": len(training_packets),
            "evaluation_packets": len(evaluation_packets),
            "neighbors_per_action": model.neighbors,
            "gate_margin": margin,
            "motor_control_enabled": False,
            "chronologically_separate_sessions": True,
        },
        "coverage": {
            "retrieval_decisions": len(decisions),
            "air_active_disagreements": len(disagreements),
            "art_grounded_disagreements": sum(
                row["art_grounded"] for row in disagreements
            ),
            "necessity_grounded_disagreements": sum(
                row["art_grounded"] and row["intervention_needed"]
                for row in disagreements
            ),
            "air_gate_selections": len(selected),
            "gate_selection_rate": len(selected) / max(len(decisions), 1),
            "gate_selection_rate_on_disagreement": len(selected)
            / max(len(disagreements), 1),
        },
        "outcome_model": {
            "active_action_mean_absolute_error": float(np.mean(active_errors)),
        },
        "estimated_policy_value": {
            "always_active": active_value,
            "always_air_when_retrieved": estimated_value("air"),
            "conservative_gate": gate_value,
            "gate_gain_over_active": gate_value - active_value,
            "bootstrap_gain_95_percent_interval": (
                [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ]
                if bootstrap
                else [0.0, 0.0]
            ),
        },
        "selected_examples": selected[:40],
        "claim_boundary": (
            "Values are estimated from action-conditioned nearest-neighbor "
            "precedents, not observed interventions. A positive result justifies "
            "a bounded shadow recommendation test, not live motor control."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-recording", type=Path, required=True)
    parser.add_argument("--evaluation-recording", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/terrain_air_counterfactual_gate_metrics.json"),
    )
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()
    payload = evaluate(
        args.training_recording,
        args.evaluation_recording,
        args.memory,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "selected_examples"}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
