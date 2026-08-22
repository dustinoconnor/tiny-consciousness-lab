#!/usr/bin/env python3
"""Test a predictive attention schema beside the adaptive GNW governor.

The schema is a compact learned model of the router's next local focus. It is
trained on separate trajectories without context labels and evaluated through
correctly bound, shuffled-role, and disconnected lesion conditions.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from adaptive_gnw_ignition_lab import (
    AdaptiveIgnitionGovernor,
    train_governor,
)
from bunge_systemic_emergence_lab import paired_summary
from four_context_conductor_lab import (
    CONTEXTS,
    OPTIMAL_SPECIALIST,
    SPECIALISTS,
    sampled_outcome,
    train_conductor,
)
from gnw_ignition_lab import entropy_bits, make_gnw_blocks, summarize
from tiny_lab import OUT


CONDITIONS = (
    "adaptive_gnw",
    "ast_schema",
    "ast_schema_shuffled",
    "ast_schema_lesion",
)
SCHEMA_BLEND = 0.30


def one_hot(index, size=len(SPECIALISTS)):
    values = np.zeros(size, dtype=np.float64)
    if index is not None:
        values[int(index)] = 1.0
    return values


def schema_features(probabilities, previous_probabilities, active):
    """Build the schema input without a hidden context or outcome label."""
    current = np.asarray(probabilities, dtype=np.float64)
    previous = (
        current
        if previous_probabilities is None
        else np.asarray(previous_probabilities, dtype=np.float64)
    )
    ranked = np.sort(current)
    margin = float(ranked[-1] - ranked[-2])
    return np.concatenate(
        [
            current,
            previous,
            one_hot(active),
            np.array([entropy_bits(current) / 2.0, margin, 1.0]),
        ]
    )


class LinearAttentionSchema:
    """Small softmax model predicting the next specialist focus."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.feature_count = len(
            schema_features(
                np.full(len(SPECIALISTS), 1.0 / len(SPECIALISTS)),
                None,
                None,
            )
        )
        self.weights = self.rng.normal(
            0.0,
            0.015,
            (self.feature_count, len(SPECIALISTS)),
        )

    @property
    def parameter_count(self):
        return int(self.weights.size)

    def probabilities(self, features):
        logits = np.asarray(features, dtype=np.float64) @ self.weights
        logits -= np.max(logits)
        values = np.exp(logits)
        return values / np.sum(values)

    def fit(self, features, targets, epochs=320, learning_rate=0.24, l2=1e-3):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        if len(x) == 0:
            raise ValueError("empty_attention_schema_training_set")
        encoded = np.eye(len(SPECIALISTS), dtype=np.float64)[y]
        for _ in range(int(epochs)):
            logits = x @ self.weights
            logits -= np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            gradient = x.T @ (probabilities - encoded) / len(x)
            gradient += float(l2) * self.weights
            self.weights -= float(learning_rate) * gradient
        return self


def conductor_probabilities(conductor, blocks):
    """Return smoothed conductor bids in their original temporal order."""
    attended = None
    sequence = []
    for block_index, (context, rows) in enumerate(blocks):
        for step, (features, draws, distractor) in enumerate(rows):
            attended = (
                np.asarray(features, dtype=np.float64)
                if attended is None
                else 0.62 * np.asarray(features) + 0.38 * attended
            )
            sequence.append(
                {
                    "block": block_index,
                    "step": step,
                    "context": context,
                    "features": features,
                    "draws": draws,
                    "distractor": distractor,
                    "probabilities": conductor.probabilities(attended),
                }
            )
    return sequence


def train_attention_schema(conductor, blocks, governor_config, seed):
    sequence = conductor_probabilities(conductor, blocks)
    governor = AdaptiveIgnitionGovernor(len(SPECIALISTS), governor_config)
    examples = []
    previous = None
    for row in sequence:
        probabilities = row["probabilities"]
        governor.select(probabilities)
        examples.append(
            schema_features(probabilities, previous, governor.active)
        )
        previous = probabilities
    features = examples[:-1]
    targets = [
        int(np.argmax(row["probabilities"])) for row in sequence[1:]
    ]
    return LinearAttentionSchema(seed).fit(features, targets)


def normalized_blend(observed, predicted, weight=SCHEMA_BLEND):
    values = (
        (1.0 - float(weight)) * np.asarray(observed, dtype=np.float64)
        + float(weight) * np.asarray(predicted, dtype=np.float64)
    )
    return values / np.sum(values)


def shuffled_schema(probabilities, seed, block, step):
    offset = 1 + ((int(seed) + int(block) + int(step)) % (len(SPECIALISTS) - 1))
    return np.roll(np.asarray(probabilities, dtype=np.float64), offset)


def evaluate(condition, conductor, blocks, seed, governor_config, schema):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown_attention_schema_condition:{condition}")
    sequence = conductor_probabilities(conductor, blocks)
    governor = AdaptiveIgnitionGovernor(len(SPECIALISTS), governor_config)
    previous_probabilities = None
    pending_prediction = np.full(
        len(SPECIALISTS), 1.0 / len(SPECIALISTS), dtype=np.float64
    )
    prediction_valid = False
    previous_action = None
    rows = []

    for row in sequence:
        probabilities = row["probabilities"]
        applied_prediction = pending_prediction
        if condition == "ast_schema_shuffled":
            applied_prediction = shuffled_schema(
                pending_prediction,
                seed,
                row["block"],
                row["step"],
            )
        gate_probabilities = probabilities
        if condition in {"ast_schema", "ast_schema_shuffled"} and prediction_valid:
            gate_probabilities = normalized_blend(
                probabilities,
                applied_prediction,
            )

        workspace_winner, ignition = governor.select(gate_probabilities)
        action = int(workspace_winner)
        success, time_cost, collision, utility = sampled_outcome(
            row["context"], action, row["draws"]
        )
        governor.observe_utility(utility)
        handoff = previous_action is not None and action != previous_action
        local_winner = int(np.argmax(probabilities))
        raw_prediction = int(np.argmax(pending_prediction))
        applied_winner = int(np.argmax(applied_prediction))

        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "block": row["block"],
                "step": row["step"],
                "context": CONTEXTS[row["context"]],
                "optimal": int(OPTIMAL_SPECIALIST[row["context"]]),
                "local_winner": local_winner,
                "workspace_winner": action,
                "action_specialist": action,
                "report_specialist": action,
                "memory_specialist": action,
                "success": float(success),
                "utility": utility,
                "time_cost": time_cost,
                "collision": collision,
                "routing_optimal": float(
                    action == OPTIMAL_SPECIALIST[row["context"]]
                ),
                "report_accurate": 1.0,
                "globally_coordinated": 1.0,
                "handoff": float(handoff),
                "unnecessary_handoff": float(handoff and row["step"] > 1),
                "boundary": float(row["step"] <= 1),
                "distractor": float(row["distractor"]),
                "ignition": float(ignition),
                "gate_entropy": entropy_bits(gate_probabilities),
                "bus_overload": 0.0,
                "schema_prediction_valid": float(prediction_valid),
                "schema_prediction_correct": float(
                    prediction_valid and raw_prediction == local_winner
                ),
                "applied_schema_prediction_correct": float(
                    prediction_valid and applied_winner == local_winner
                ),
            }
        )

        next_features = schema_features(
            probabilities,
            previous_probabilities,
            governor.active,
        )
        pending_prediction = schema.probabilities(next_features)
        prediction_valid = True
        previous_probabilities = probabilities
        previous_action = action
    return rows


def schema_summary(rows):
    result = summarize(rows)
    valid = [row for row in rows if row["schema_prediction_valid"]]
    boundary = [row for row in valid if row["boundary"]]
    distractors = [row for row in valid if row["distractor"]]

    def mean(chunk, key):
        return float(np.mean([row[key] for row in chunk])) if chunk else 0.0

    result.update(
        {
            "schema_next_focus_accuracy": mean(
                valid, "schema_prediction_correct"
            ),
            "applied_schema_next_focus_accuracy": mean(
                valid, "applied_schema_prediction_correct"
            ),
            "schema_boundary_accuracy": mean(
                boundary, "schema_prediction_correct"
            ),
            "schema_distractor_accuracy": mean(
                distractors, "schema_prediction_correct"
            ),
        }
    )
    return result


def run_benchmark(
    seed_count=12,
    train_steps=12000,
    governor_train_blocks=70,
    schema_train_blocks=100,
    eval_blocks=180,
    steps_per_block=8,
):
    per_seed = {condition: [] for condition in CONDITIONS}
    all_rows = {condition: [] for condition in CONDITIONS}
    parameter_counts = []

    for seed in range(seed_count):
        conductor = train_conductor(11_000 + seed, steps=train_steps)
        governor_blocks = make_gnw_blocks(
            12_000 + seed, governor_train_blocks, steps_per_block
        )
        governor_config = train_governor(
            conductor, governor_blocks, 13_000 + seed
        )
        schema_blocks = make_gnw_blocks(
            14_000 + seed, schema_train_blocks, steps_per_block
        )
        schema = train_attention_schema(
            conductor,
            schema_blocks,
            governor_config,
            15_000 + seed,
        )
        parameter_counts.append(schema.parameter_count)
        evaluation_blocks = make_gnw_blocks(
            16_000 + seed, eval_blocks, steps_per_block
        )
        for condition in CONDITIONS:
            rows = evaluate(
                condition,
                conductor,
                evaluation_blocks,
                17_000 + seed,
                governor_config,
                schema,
            )
            all_rows[condition].extend(rows)
            per_seed[condition].append(schema_summary(rows))

    summary = {
        condition: schema_summary(rows)
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
        "schema_minus_adaptive_utility": contrast(
            "utility_per_step", "ast_schema", "adaptive_gnw"
        ),
        "schema_minus_shuffled_utility": contrast(
            "utility_per_step", "ast_schema", "ast_schema_shuffled"
        ),
        "schema_minus_adaptive_boundary_routing": contrast(
            "boundary_optimal_routing", "ast_schema", "adaptive_gnw"
        ),
        "schema_minus_adaptive_distractor_routing": contrast(
            "distractor_optimal_routing", "ast_schema", "adaptive_gnw"
        ),
        "schema_minus_adaptive_unnecessary_handoffs": contrast(
            "unnecessary_handoff_rate", "ast_schema", "adaptive_gnw"
        ),
        "lesion_minus_adaptive_utility": contrast(
            "utility_per_step", "ast_schema_lesion", "adaptive_gnw"
        ),
    }
    criteria = {
        "schema_utility_beats_adaptive": (
            contrasts["schema_minus_adaptive_utility"]["ci95_low"] > 0.0
        ),
        "schema_utility_beats_shuffled": (
            contrasts["schema_minus_shuffled_utility"]["ci95_low"] > 0.0
        ),
        "schema_preserves_boundary_routing": (
            contrasts["schema_minus_adaptive_boundary_routing"]["ci95_low"]
            >= -0.01
        ),
        "schema_preserves_distractor_routing": (
            contrasts["schema_minus_adaptive_distractor_routing"]["ci95_low"]
            >= -0.01
        ),
        "lesion_matches_no_schema": (
            abs(contrasts["lesion_minus_adaptive_utility"]["mean"]) < 1e-12
        ),
    }
    return {
        "experiment": "predictive attention schema for adaptive GNW routing",
        "seed_count": seed_count,
        "held_out_steps_per_condition": (
            seed_count * eval_blocks * steps_per_block
        ),
        "schema_parameter_count": int(max(parameter_counts, default=0)),
        "schema_blend": SCHEMA_BLEND,
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "claim_boundary": (
            "Passing supports a functional benefit from a learned prediction "
            "of router focus in this synthetic benchmark. It does not establish "
            "biological Attention Schema Theory, phenomenal consciousness, or "
            "benefit in Unity without separate embodied replication."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--governor-train-blocks", type=int, default=70)
    parser.add_argument("--schema-train-blocks", type=int, default=100)
    parser.add_argument("--eval-blocks", type=int, default=180)
    parser.add_argument("--steps-per-block", type=int, default=8)
    parser.add_argument(
        "--output",
        default="outputs/attention_schema_metrics_20260809.json",
    )
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=args.seeds,
        train_steps=args.train_steps,
        governor_train_blocks=args.governor_train_blocks,
        schema_train_blocks=args.schema_train_blocks,
        eval_blocks=args.eval_blocks,
        steps_per_block=args.steps_per_block,
    )
    path = OUT.parent / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "adaptive="
        f"{payload['summary']['adaptive_gnw']['utility_per_step']:.4f} "
        "schema="
        f"{payload['summary']['ast_schema']['utility_per_step']:.4f} "
        "shuffled="
        f"{payload['summary']['ast_schema_shuffled']['utility_per_step']:.4f}"
    )
    print(
        "schema_prediction_accuracy="
        f"{payload['summary']['ast_schema']['schema_next_focus_accuracy']:.4f}"
    )
    for name, passed in payload["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {payload['all_criteria_pass']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
