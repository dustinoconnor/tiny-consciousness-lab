#!/usr/bin/env python3
"""Learn when a GNW broadcast should persist or yield.

The governor is selected only from reward on separate training episodes. It
does not receive context labels. Held-out evaluation compares it with the
framewise conductor, fixed GNW persistence, feedback lesion, and broadcast
lesions.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from bunge_systemic_emergence_lab import paired_summary
from four_context_conductor_lab import (
    CONTEXTS,
    OPTIMAL_SPECIALIST,
    SPECIALISTS,
    sampled_outcome,
    train_conductor,
)
from gnw_ignition_lab import (
    evaluate as evaluate_fixed,
    make_gnw_blocks,
    summarize,
)
from tiny_lab import OUT


CONDITIONS = (
    "framewise_conductor",
    "fixed_gnw",
    "adaptive_gnw",
    "adaptive_feedback_lesion",
    "adaptive_broadcast_lesion",
    "adaptive_broadcast_scramble",
)


@dataclass(frozen=True)
class GovernorConfig:
    challenger_confidence: float
    switch_margin: float
    release_streak: int
    failure_threshold: float
    reward_alpha: float = 0.55
    refractory_steps: int = 1


class AdaptiveIgnitionGovernor:
    """Reward-calibrated temporal release policy over compressed bids."""

    def __init__(self, specialist_count, config):
        self.specialist_count = specialist_count
        self.config = config
        self.active = None
        self.memory = None
        self.report = None
        self.active_age = 0
        self.challenger = None
        self.challenger_streak = 0
        self.refractory = 0
        self.reward_ema = 0.70
        self.ignitions = 0
        self.switches = 0

    def select(self, raw_probabilities):
        probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        candidate = int(np.argmax(probabilities))
        confidence = float(probabilities[candidate])
        if self.active is None:
            self.active = candidate
            self.memory = candidate
            self.report = candidate
            self.active_age = 1
            self.ignitions += 1
            return candidate, True

        if self.refractory > 0:
            self.refractory -= 1
        if candidate == self.active:
            self.challenger = None
            self.challenger_streak = 0
            self.active_age += 1
            self.memory = self.active
            self.report = self.active
            return self.active, False

        if self.challenger == candidate:
            self.challenger_streak += 1
        else:
            self.challenger = candidate
            self.challenger_streak = 1

        active_confidence = float(probabilities[self.active])
        margin = confidence - active_confidence
        failure_release = self.reward_ema <= self.config.failure_threshold
        sustained_release = (
            self.challenger_streak >= self.config.release_streak
        )
        switch = (
            confidence >= self.config.challenger_confidence
            and margin >= self.config.switch_margin
            and (failure_release or sustained_release)
            and (self.refractory <= 0 or failure_release)
        )
        if switch:
            self.active = candidate
            self.memory = candidate
            self.report = candidate
            self.active_age = 1
            self.challenger = None
            self.challenger_streak = 0
            self.refractory = self.config.refractory_steps
            self.ignitions += 1
            self.switches += 1
            return candidate, True

        self.active_age += 1
        return self.active, False

    def observe_utility(self, utility, enabled=True):
        if not enabled:
            return
        alpha = self.config.reward_alpha
        self.reward_ema = (
            (1.0 - alpha) * self.reward_ema + alpha * float(utility)
        )


def evaluate_adaptive(condition, conductor, blocks, seed, config):
    rng = np.random.default_rng(seed + 91_003 * CONDITIONS.index(condition))
    governor = AdaptiveIgnitionGovernor(len(SPECIALISTS), config)
    attended = None
    previous_action = None
    rows = []
    for block_index, (context, block_rows) in enumerate(blocks):
        for step, (features, draws, distractor) in enumerate(block_rows):
            attended = (
                np.asarray(features, dtype=np.float64)
                if attended is None
                else 0.62 * np.asarray(features) + 0.38 * attended
            )
            local_probabilities = conductor.probabilities(attended)
            raw_probabilities = conductor.probabilities(features)
            local_winner = int(np.argmax(local_probabilities))
            workspace_winner, ignition = governor.select(raw_probabilities)

            action = workspace_winner
            report = governor.report
            memory = governor.memory
            if condition == "adaptive_broadcast_lesion":
                action = local_winner
                report = None
                memory = None
            elif condition == "adaptive_broadcast_scramble":
                offsets = rng.integers(1, len(SPECIALISTS), size=3)
                action = int((workspace_winner + offsets[0]) % len(SPECIALISTS))
                report = int((workspace_winner + offsets[1]) % len(SPECIALISTS))
                memory = int((workspace_winner + offsets[2]) % len(SPECIALISTS))

            success, time_cost, collision, utility = sampled_outcome(
                context, action, draws
            )
            governor.observe_utility(
                utility,
                enabled=condition != "adaptive_feedback_lesion",
            )
            handoff = previous_action is not None and action != previous_action
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
                    "action_specialist": action,
                    "report_specialist": report,
                    "memory_specialist": memory,
                    "success": float(success),
                    "utility": utility,
                    "time_cost": time_cost,
                    "collision": collision,
                    "routing_optimal": float(
                        action == OPTIMAL_SPECIALIST[context]
                    ),
                    "report_accurate": float(report == action),
                    "globally_coordinated": float(
                        action == report == memory
                        if memory is not None
                        else False
                    ),
                    "handoff": float(handoff),
                    "unnecessary_handoff": float(handoff and step > 1),
                    "boundary": float(step <= 1),
                    "distractor": float(distractor),
                    "ignition": float(ignition),
                    "gate_entropy": float(
                        -np.sum(
                            raw_probabilities
                            * np.log2(
                                np.clip(raw_probabilities, 1e-12, 1.0)
                            )
                        )
                    ),
                    "bus_overload": 0.0,
                    "reward_ema": governor.reward_ema,
                    "challenger_streak": governor.challenger_streak,
                    "broadcast_age": governor.active_age,
                }
            )
            previous_action = action
    return rows


def train_governor(conductor, blocks, seed):
    candidates = [
        GovernorConfig(*values)
        for values in itertools.product(
            (0.55, 0.70, 0.82),
            (0.04, 0.10),
            (1, 2, 3),
            (-0.25, 0.05, 0.30),
        )
    ]
    scored = []
    for config in candidates:
        rows = evaluate_adaptive(
            "adaptive_gnw",
            conductor,
            blocks,
            seed,
            config,
        )
        metrics = summarize(rows)
        objective = (
            metrics["utility_per_step"]
            - 0.05 * metrics["unnecessary_handoff_rate"]
        )
        scored.append((objective, config))
    return max(
        scored,
        key=lambda item: (
            item[0],
            -item[1].release_streak,
            -item[1].challenger_confidence,
        ),
    )[1]


def config_key(config):
    return (
        f"confidence={config.challenger_confidence:.2f},"
        f"margin={config.switch_margin:.2f},"
        f"streak={config.release_streak},"
        f"failure={config.failure_threshold:.2f}"
    )


def run_benchmark(
    seed_count=12,
    train_steps=14000,
    train_blocks=90,
    eval_blocks=180,
    steps_per_block=10,
):
    per_seed = {condition: [] for condition in CONDITIONS}
    all_rows = {condition: [] for condition in CONDITIONS}
    selected_configs = []

    for seed in range(seed_count):
        conductor = train_conductor(4100 + seed, steps=train_steps)
        training_blocks = make_gnw_blocks(
            5100 + seed,
            train_blocks,
            steps_per_block,
        )
        config = train_governor(
            conductor,
            training_blocks,
            6100 + seed,
        )
        selected_configs.append(config)
        evaluation_blocks = make_gnw_blocks(
            7100 + seed,
            eval_blocks,
            steps_per_block,
        )
        framewise = evaluate_fixed(
            "framewise_conductor",
            conductor,
            evaluation_blocks,
            8100 + seed,
        )
        fixed = evaluate_fixed(
            "gnw_ignition",
            conductor,
            evaluation_blocks,
            8100 + seed,
        )
        condition_rows = {
            "framewise_conductor": framewise,
            "fixed_gnw": fixed,
        }
        for condition in CONDITIONS[2:]:
            condition_rows[condition] = evaluate_adaptive(
                condition,
                conductor,
                evaluation_blocks,
                8100 + seed,
                config,
            )
        for condition, rows in condition_rows.items():
            all_rows[condition].extend(rows)
            per_seed[condition].append(summarize(rows))

    summary = {
        condition: summarize(rows)
        for condition, rows in all_rows.items()
    }

    def contrast(metric, left, right):
        return paired_summary(
            [
                per_seed[left][index][metric]
                - per_seed[right][index][metric]
                for index in range(seed_count)
            ]
        )

    contrasts = {
        "adaptive_minus_framewise_utility": contrast(
            "utility_per_step", "adaptive_gnw", "framewise_conductor"
        ),
        "adaptive_minus_fixed_utility": contrast(
            "utility_per_step", "adaptive_gnw", "fixed_gnw"
        ),
        "adaptive_minus_framewise_distractor_routing": contrast(
            "distractor_optimal_routing",
            "adaptive_gnw",
            "framewise_conductor",
        ),
        "adaptive_minus_fixed_boundary_routing": contrast(
            "boundary_optimal_routing", "adaptive_gnw", "fixed_gnw"
        ),
        "adaptive_minus_feedback_lesion_utility": contrast(
            "utility_per_step",
            "adaptive_gnw",
            "adaptive_feedback_lesion",
        ),
        "adaptive_minus_broadcast_lesion_coordination": contrast(
            "global_coordination_rate",
            "adaptive_gnw",
            "adaptive_broadcast_lesion",
        ),
        "adaptive_minus_scramble_utility": contrast(
            "utility_per_step",
            "adaptive_gnw",
            "adaptive_broadcast_scramble",
        ),
    }
    criteria = {
        "adaptive_utility_beats_fixed_gnw": (
            contrasts["adaptive_minus_fixed_utility"]["ci95_low"] > 0.0
        ),
        "adaptive_utility_not_below_framewise": (
            contrasts["adaptive_minus_framewise_utility"]["ci95_low"] >= -0.01
        ),
        "adaptive_preserves_distractor_advantage": (
            contrasts[
                "adaptive_minus_framewise_distractor_routing"
            ]["ci95_low"]
            > 0.25
        ),
        "adaptive_recovers_boundary_responsiveness": (
            contrasts[
                "adaptive_minus_fixed_boundary_routing"
            ]["ci95_low"]
            > 0.20
        ),
        "reward_feedback_has_positive_utility": (
            contrasts[
                "adaptive_minus_feedback_lesion_utility"
            ]["ci95_low"]
            > 0.0
        ),
        "broadcast_required_for_global_coordination": (
            contrasts[
                "adaptive_minus_broadcast_lesion_coordination"
            ]["ci95_low"]
            > 0.50
        ),
        "correct_broadcast_beats_scramble": (
            contrasts["adaptive_minus_scramble_utility"]["ci95_low"] > 0.20
        ),
    }
    return {
        "experiment": "reward-trained adaptive GNW ignition and release",
        "seed_count": seed_count,
        "held_out_steps_per_condition": (
            seed_count * eval_blocks * steps_per_block
        ),
        "selected_config_counts": dict(
            Counter(config_key(config) for config in selected_configs)
        ),
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "claim_boundary": (
            "The governor is selected from reward on separate synthetic "
            "episodes and evaluated on held-out seeds. Passing supports adaptive "
            "control of global access under this benchmark, not a clinical model "
            "of ADHD, biological GNW equivalence, or phenomenal consciousness."
        ),
    }


def plot(payload, path):
    summary = payload["summary"]
    conditions = list(CONDITIONS)
    labels = [
        "framewise",
        "fixed\nGNW",
        "adaptive\nGNW",
        "feedback\nlesion",
        "broadcast\nlesion",
        "broadcast\nscramble",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = (
        ("utility_per_step", "Utility"),
        ("distractor_optimal_routing", "Distractor resistance"),
        ("boundary_optimal_routing", "Context-shift response"),
        ("global_coordination_rate", "Action-memory-report coordination"),
    )
    colors = ("#168aad", "#2a9d8f", "#e9c46a", "#e76f51")
    for axis, (metric, title), color in zip(
        axes.flat, metrics, colors
    ):
        axis.bar(
            labels,
            [summary[condition][metric] for condition in conditions],
            color=color,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Adaptive GNW ignition governor")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=14000)
    parser.add_argument("--train-blocks", type=int, default=90)
    parser.add_argument("--eval-blocks", type=int, default=180)
    parser.add_argument("--steps-per-block", type=int, default=10)
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=args.seeds,
        train_steps=args.train_steps,
        train_blocks=args.train_blocks,
        eval_blocks=args.eval_blocks,
        steps_per_block=args.steps_per_block,
    )
    OUT.mkdir(exist_ok=True)
    metrics_path = OUT / "adaptive_gnw_ignition_metrics.json"
    plot_path = OUT / "adaptive_gnw_ignition_summary.png"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    plot(payload, plot_path)
    print(
        f"adaptive={payload['summary']['adaptive_gnw']['utility_per_step']:.3f} "
        f"fixed={payload['summary']['fixed_gnw']['utility_per_step']:.3f} "
        f"framewise={payload['summary']['framewise_conductor']['utility_per_step']:.3f}"
    )
    for name, passed in payload["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {payload['all_criteria_pass']}")
    print(f"wrote {metrics_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
