#!/usr/bin/env python3
"""Test thresholded global access in the four-specialist control stack.

This is a software analogue of Global Neuronal Workspace (GNW), not a claim
that neuronal ignition or phenomenal experience has been reproduced. Local
specialists submit compressed bids to a capacity-one workspace. Evidence must
cross an ignition threshold before one winner is persistently broadcast to
action, memory, and report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bunge_systemic_emergence_lab import paired_summary
from four_context_conductor_lab import (
    CONTEXTS,
    OPTIMAL_SPECIALIST,
    SPECIALISTS,
    make_evaluation_blocks,
    sample_features,
    sampled_outcome,
    train_conductor,
)
from tiny_lab import OUT


CONDITIONS = (
    "framewise_conductor",
    "gnw_ignition",
    "gnw_no_persistence",
    "gnw_raw_bus",
    "gnw_broadcast_lesion",
    "gnw_broadcast_scramble",
)


def entropy_bits(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    active = values[values > 1e-12]
    return float(-np.sum(active * np.log2(active)))


@dataclass
class GNWState:
    specialist_count: int
    decay: float = 0.55
    ignition_threshold: float = 0.43
    switch_margin: float = 0.12
    persistence_steps: int = 3
    refractory_steps: int = 2
    support_threshold: float = 0.45
    strong_challenger_threshold: float = 0.72
    displaced_winner_ceiling: float = 0.20

    def __post_init__(self):
        self.evidence = np.zeros(self.specialist_count, dtype=np.float64)
        self.active = None
        self.persistence = 0
        self.refractory = 0
        self.ignitions = 0
        self.switches = 0
        self.strong_challenger = None
        self.strong_challenger_streak = 0

    def update(self, probabilities, allow_persistence=True):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        self.evidence = self.decay * self.evidence + (1.0 - self.decay) * probabilities
        candidate = int(np.argmax(self.evidence))
        local_candidate = int(np.argmax(probabilities))
        candidate_evidence = float(self.evidence[candidate])
        previous = self.active

        if self.refractory > 0:
            self.refractory -= 1
        if self.persistence > 0:
            self.persistence -= 1

        ignite = False
        strong_challenger_evidence = (
            self.active is not None
            and local_candidate != self.active
            and probabilities[local_candidate]
            >= self.strong_challenger_threshold
            and probabilities[self.active] <= self.displaced_winner_ceiling
        )
        if strong_challenger_evidence:
            if self.strong_challenger == local_candidate:
                self.strong_challenger_streak += 1
            else:
                self.strong_challenger = local_candidate
                self.strong_challenger_streak = 1
        else:
            self.strong_challenger = None
            self.strong_challenger_streak = 0
        strong_challenger = self.strong_challenger_streak >= 2
        if strong_challenger:
            candidate = local_candidate
            candidate_evidence = float(probabilities[candidate])
            ignite = True
        elif self.active is None:
            ignite = candidate_evidence >= self.ignition_threshold
        elif candidate == self.active:
            if (
                allow_persistence
                and probabilities[self.active] >= self.support_threshold
            ):
                self.persistence = max(self.persistence, self.persistence_steps)
        else:
            advantage = candidate_evidence - float(self.evidence[self.active])
            persistent = allow_persistence and self.persistence > 0
            ignite = (
                not persistent
                and self.refractory <= 0
                and candidate_evidence >= self.ignition_threshold
                and advantage >= self.switch_margin
            )

        if ignite:
            self.active = candidate
            self.persistence = self.persistence_steps if allow_persistence else 0
            self.refractory = self.refractory_steps
            self.ignitions += 1
            if previous is not None and previous != candidate:
                self.switches += 1
        return self.active, ignite


def make_gnw_blocks(
    seed,
    blocks,
    steps_per_block,
    distractor_probability=0.14,
):
    """Inject matched one-frame specialist distractors after context onset."""
    result = make_evaluation_blocks(
        seed,
        blocks=blocks,
        steps_per_block=steps_per_block,
    )
    rng = np.random.default_rng(seed + 88_001)
    augmented = []
    for context, block_rows in result:
        rows = []
        for step, (features, draws) in enumerate(block_rows):
            distractor = (
                step > 1 and rng.random() < distractor_probability
            )
            if distractor:
                alternatives = [
                    index
                    for index in range(len(CONTEXTS))
                    if index != context
                ]
                wrong_context = int(rng.choice(alternatives))
                features = sample_features(
                    wrong_context,
                    rng,
                    domain_shift=0.035,
                )
            rows.append((features, draws, distractor))
        augmented.append((context, rows))
    return augmented


def raw_bus_probabilities(probabilities, features, rng, capacity=8):
    """Model raw specialist packets colliding on a capacity-limited bus."""
    raw_items = len(SPECIALISTS) * len(features)
    overload = max(0.0, (raw_items - capacity) / raw_items)
    mixed = (
        (1.0 - overload) * np.asarray(probabilities, dtype=np.float64)
        + overload * rng.dirichlet(np.ones(len(SPECIALISTS)))
    )
    return mixed / mixed.sum(), overload


def condition_seed(condition):
    return CONDITIONS.index(condition) * 100_003


def evaluate(condition, conductor, blocks, seed):
    rng = np.random.default_rng(seed + condition_seed(condition))
    workspace = GNWState(len(SPECIALISTS))
    attended = None
    previous_action = None
    workspace_memory = None
    rows = []

    for block_index, (context, block_rows) in enumerate(blocks):
        for step, (features, draws, distractor) in enumerate(block_rows):
            attended = (
                np.asarray(features, dtype=np.float64)
                if attended is None
                else 0.62 * np.asarray(features) + 0.38 * attended
            )
            probabilities = conductor.probabilities(attended)
            local_winner = int(np.argmax(probabilities))
            overload = 0.0
            ignition = False

            if condition == "framewise_conductor":
                workspace_winner = local_winner
            else:
                bids = probabilities
                if condition == "gnw_raw_bus":
                    bids, overload = raw_bus_probabilities(
                        probabilities, features, rng
                    )
                workspace_winner, ignition = workspace.update(
                    bids,
                    allow_persistence=condition != "gnw_no_persistence",
                )
                if workspace_winner is None:
                    workspace_winner = local_winner

            action_specialist = workspace_winner
            report_specialist = workspace_winner
            if condition == "gnw_broadcast_lesion":
                # The winner still ignites internally, but cannot recruit the
                # action system or update globally reportable memory.
                action_specialist = local_winner
                report_specialist = (
                    workspace_memory
                    if workspace_memory is not None
                    else local_winner
                )
            elif condition == "gnw_broadcast_scramble":
                action_specialist = int(
                    (workspace_winner + rng.integers(1, len(SPECIALISTS)))
                    % len(SPECIALISTS)
                )
                report_specialist = int(
                    (workspace_winner + rng.integers(1, len(SPECIALISTS)))
                    % len(SPECIALISTS)
                )

            broadcast_intact = condition not in {
                "gnw_broadcast_lesion",
                "gnw_broadcast_scramble",
            }
            if broadcast_intact:
                workspace_memory = workspace_winner

            success, time_cost, collision, utility = sampled_outcome(
                context, action_specialist, draws
            )
            handoff = (
                previous_action is not None
                and action_specialist != previous_action
            )
            coordinated = (
                action_specialist == report_specialist == workspace_memory
                if workspace_memory is not None
                else False
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "block": block_index,
                    "step": step,
                    "context": CONTEXTS[context],
                    "optimal": int(OPTIMAL_SPECIALIST[context]),
                    "local_winner": local_winner,
                    "workspace_winner": workspace_winner,
                    "action_specialist": action_specialist,
                    "report_specialist": report_specialist,
                    "memory_specialist": workspace_memory,
                    "success": float(success),
                    "utility": utility,
                    "time_cost": time_cost,
                    "collision": collision,
                    "routing_optimal": float(
                        action_specialist == OPTIMAL_SPECIALIST[context]
                    ),
                    "report_accurate": float(
                        report_specialist == action_specialist
                    ),
                    "globally_coordinated": float(coordinated),
                    "handoff": float(handoff),
                    "unnecessary_handoff": float(handoff and step > 1),
                    "boundary": float(step <= 1),
                    "distractor": float(distractor),
                    "ignition": float(ignition),
                    "gate_entropy": entropy_bits(probabilities),
                    "bus_overload": overload,
                    "compressed_packet_items": 4.0,
                    "raw_packet_items": float(
                        len(SPECIALISTS) * len(features)
                    ),
                }
            )
            previous_action = action_specialist
    return rows


def summarize(rows):
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    steady = [row for row in rows if row["step"] > 1]
    boundary = [row for row in rows if row["step"] <= 1]
    distractors = [row for row in rows if row["distractor"]]
    return {
        "success_rate": mean("success"),
        "utility_per_step": mean("utility"),
        "optimal_routing_rate": mean("routing_optimal"),
        "report_action_agreement": mean("report_accurate"),
        "global_coordination_rate": mean("globally_coordinated"),
        "handoff_rate": mean("handoff"),
        "unnecessary_handoff_rate": float(
            np.mean([row["unnecessary_handoff"] for row in steady])
        ),
        "boundary_optimal_routing": float(
            np.mean([row["routing_optimal"] for row in boundary])
        ),
        "steady_optimal_routing": float(
            np.mean([row["routing_optimal"] for row in steady])
        ),
        "distractor_optimal_routing": float(
            np.mean([row["routing_optimal"] for row in distractors])
        ),
        "ignition_rate": mean("ignition"),
        "mean_gate_entropy_bits": mean("gate_entropy"),
        "mean_bus_overload": mean("bus_overload"),
    }


def plot(payload, path):
    summary = payload["summary"]
    names = list(CONDITIONS)
    labels = [
        "framewise",
        "GNW",
        "no\npersistence",
        "raw\nbus",
        "broadcast\nlesion",
        "broadcast\nscramble",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(
        labels,
        [summary[name]["utility_per_step"] for name in names],
        color="#168aad",
    )
    axes[0, 0].set_title("Task utility")
    axes[0, 1].bar(
        labels,
        [summary[name]["unnecessary_handoff_rate"] for name in names],
        color="#e76f51",
    )
    axes[0, 1].set_title("Unnecessary handoffs")
    axes[1, 0].bar(
        labels,
        [summary[name]["global_coordination_rate"] for name in names],
        color="#2a9d8f",
    )
    axes[1, 0].set_title("Action-memory-report coordination")
    axes[1, 1].bar(
        labels,
        [summary[name]["optimal_routing_rate"] for name in names],
        color="#e9c46a",
    )
    axes[1, 1].set_title("Optimal specialist routing")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("GNW ignition, compression, and broadcast lesions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(
    seed_count=12,
    train_steps=16000,
    blocks=180,
    steps_per_block=10,
):
    seed_summaries = {condition: [] for condition in CONDITIONS}
    all_rows = {condition: [] for condition in CONDITIONS}
    for seed in range(seed_count):
        conductor = train_conductor(700 + seed, steps=train_steps)
        blocks_data = make_gnw_blocks(
            1700 + seed,
            blocks=blocks,
            steps_per_block=steps_per_block,
        )
        for condition in CONDITIONS:
            rows = evaluate(condition, conductor, blocks_data, 2700 + seed)
            all_rows[condition].extend(rows)
            seed_summaries[condition].append(summarize(rows))

    summary = {
        condition: summarize(rows)
        for condition, rows in all_rows.items()
    }

    def paired(metric, left, right):
        return paired_summary(
            [
                seed_summaries[left][index][metric]
                - seed_summaries[right][index][metric]
                for index in range(seed_count)
            ]
        )

    contrasts = {
        "gnw_minus_framewise_utility": paired(
            "utility_per_step", "gnw_ignition", "framewise_conductor"
        ),
        "framewise_minus_gnw_unnecessary_handoffs": paired(
            "unnecessary_handoff_rate",
            "framewise_conductor",
            "gnw_ignition",
        ),
        "gnw_minus_no_persistence_utility": paired(
            "utility_per_step", "gnw_ignition", "gnw_no_persistence"
        ),
        "gnw_minus_raw_bus_utility": paired(
            "utility_per_step", "gnw_ignition", "gnw_raw_bus"
        ),
        "gnw_minus_broadcast_lesion_coordination": paired(
            "global_coordination_rate",
            "gnw_ignition",
            "gnw_broadcast_lesion",
        ),
        "gnw_minus_broadcast_scramble_utility": paired(
            "utility_per_step", "gnw_ignition", "gnw_broadcast_scramble"
        ),
    }
    criteria = {
        "gnw_utility_not_below_framewise": (
            contrasts["gnw_minus_framewise_utility"]["ci95_low"] >= -0.01
        ),
        "gnw_reduces_unnecessary_handoffs": (
            contrasts["framewise_minus_gnw_unnecessary_handoffs"]["ci95_low"]
            > 0.0
        ),
        "persistence_has_positive_utility": (
            contrasts["gnw_minus_no_persistence_utility"]["ci95_low"] > 0.0
        ),
        "compressed_bus_beats_raw_overload": (
            contrasts["gnw_minus_raw_bus_utility"]["ci95_low"] > 0.0
        ),
        "broadcast_is_causally_required_for_coordination": (
            contrasts[
                "gnw_minus_broadcast_lesion_coordination"
            ]["ci95_low"]
            > 0.25
        ),
        "correct_broadcast_beats_scrambled_broadcast": (
            contrasts[
                "gnw_minus_broadcast_scramble_utility"
            ]["ci95_low"]
            > 0.20
        ),
    }
    return {
        "experiment": "GNW-inspired thresholded global access",
        "seed_count": seed_count,
        "steps_per_condition": seed_count * blocks * steps_per_block,
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "claim_boundary": (
            "Passing supports a functional software analogue of limited-capacity "
            "global access: thresholded competition, persistent broadcast, and "
            "coordinated downstream availability. It does not establish neuronal "
            "GNW equivalence or phenomenal consciousness."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=16000)
    parser.add_argument("--blocks", type=int, default=180)
    parser.add_argument("--steps-per-block", type=int, default=10)
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=args.seeds,
        train_steps=args.train_steps,
        blocks=args.blocks,
        steps_per_block=args.steps_per_block,
    )
    OUT.mkdir(exist_ok=True)
    metrics_path = OUT / "gnw_ignition_metrics.json"
    plot_path = OUT / "gnw_ignition_summary.png"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, plot_path)
    print(
        f"GNW utility={payload['summary']['gnw_ignition']['utility_per_step']:.3f} "
        f"framewise={payload['summary']['framewise_conductor']['utility_per_step']:.3f}"
    )
    for name, passed in payload["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {payload['all_criteria_pass']}")
    print(f"wrote {metrics_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
