#!/usr/bin/env python3
"""Reward-train an executive gate across four embodied control contexts.

The conductor receives noisy state features rather than context labels. Four
frozen specialists differ in competence, latency, collision exposure, and
compute cost:

- recurrent: efficient clear-terrain exploration
- episodic: familiar hidden-goal precedent playback
- predictive: visible-target model-predictive interception
- fallback: physical-wedge recovery

Evaluation compares the learned gate with every fixed specialist, an oracle,
a scrambled gate, and targeted single-specialist lesions. This is a synthetic
contextual-bandit benchmark for adaptive arbitration, not a consciousness test
or a substitute for Unity validation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, ROOT


CONTEXTS = (
    "clear_terrain",
    "familiar_hidden_goal",
    "visible_target",
    "genuine_wedge",
)
SPECIALISTS = ("recurrent", "episodic", "predictive", "fallback")
FEATURE_NAMES = (
    "food_visibility",
    "art_resonance",
    "obstacle_pressure",
    "body_contact",
    "immobility",
    "spatial_novelty",
    "free_space",
)

# No feature is a context label. The distributions overlap and are perturbed
# independently in training and evaluation.
FEATURE_MEANS = np.array(
    [
        [0.03, 0.10, 0.12, 0.03, 0.03, 0.76, 0.90],
        [0.04, 0.88, 0.62, 0.20, 0.27, 0.42, 0.30],
        [0.94, 0.32, 0.26, 0.05, 0.04, 0.34, 0.72],
        [0.07, 0.24, 0.94, 0.93, 0.94, 0.18, 0.07],
    ],
    dtype=np.float64,
)
FEATURE_STD = np.array([0.09, 0.11, 0.12, 0.10, 0.10, 0.12, 0.10])

# Rows are contexts and columns are specialists.
SUCCESS_PROBABILITY = np.array(
    [
        [0.96, 0.72, 0.84, 0.62],
        [0.58, 0.97, 0.66, 0.54],
        [0.78, 0.64, 0.98, 0.56],
        [0.42, 0.46, 0.58, 0.97],
    ],
    dtype=np.float64,
)
TIME_COST = np.array(
    [
        [0.12, 0.34, 0.29, 0.52],
        [0.82, 0.20, 0.66, 0.56],
        [0.50, 0.60, 0.16, 0.76],
        [0.92, 0.87, 0.74, 0.19],
    ],
    dtype=np.float64,
)
COLLISION_EXPOSURE = np.array(
    [
        [0.02, 0.04, 0.03, 0.09],
        [0.52, 0.10, 0.38, 0.30],
        [0.16, 0.22, 0.04, 0.32],
        [0.86, 0.77, 0.62, 0.16],
    ],
    dtype=np.float64,
)
COMPUTE_COST = np.array([0.015, 0.055, 0.125, 0.085], dtype=np.float64)
OPTIMAL_SPECIALIST = np.array([0, 1, 2, 3], dtype=np.int64)


def clamp01(values):
    return np.clip(values, 0.0, 1.0)


def softmax(values, temperature=0.18):
    scaled = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / np.sum(weights)


def gate_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    active = probabilities[probabilities > 1e-12]
    return float(-np.sum(active * np.log2(active)))


def expand_features(features):
    """Add grounded interactions without supplying a categorical context."""
    food, resonance, obstacle, contact, immobility, novelty, free_space = features
    return np.concatenate(
        [
            np.asarray(features, dtype=np.float64),
            np.array(
                [
                    food * free_space,
                    resonance * (1.0 - food),
                    contact * immobility,
                    obstacle * (1.0 - free_space),
                    novelty * free_space,
                    1.0,
                ]
            ),
        ]
    )


def sample_features(context, rng, domain_shift=0.0):
    mean = FEATURE_MEANS[int(context)].copy()
    if domain_shift:
        mean += rng.normal(0.0, domain_shift, len(mean))
    return clamp01(rng.normal(mean, FEATURE_STD))


def expected_utility(context, specialist):
    success = SUCCESS_PROBABILITY[context, specialist]
    return float(
        success * 1.0
        + (1.0 - success) * -1.20
        - 0.24 * TIME_COST[context, specialist]
        - 0.34 * COLLISION_EXPOSURE[context, specialist]
        - COMPUTE_COST[specialist]
    )


def sampled_outcome(context, specialist, draws):
    """Use condition-independent random draws for matched policy evaluation."""
    success = bool(draws["success"][specialist] < SUCCESS_PROBABILITY[context, specialist])
    time_cost = float(
        max(
            0.0,
            TIME_COST[context, specialist] + 0.035 * draws["time"][specialist],
        )
    )
    collision = float(
        max(
            0.0,
            COLLISION_EXPOSURE[context, specialist]
            + 0.025 * draws["collision"][specialist],
        )
    )
    utility = (
        (1.0 if success else -1.20)
        - 0.24 * time_cost
        - 0.34 * collision
        - COMPUTE_COST[specialist]
    )
    return success, time_cost, collision, float(utility)


class ContextualConductor:
    """Linear contextual bandit trained only from selected-specialist reward."""

    def __init__(self, seed, learning_rate=0.055):
        self.rng = np.random.default_rng(seed)
        self.feature_dim = len(expand_features(np.zeros(len(FEATURE_NAMES))))
        self.weights = np.zeros((len(SPECIALISTS), self.feature_dim), dtype=np.float64)
        self.visits = np.zeros(len(SPECIALISTS), dtype=np.int64)
        self.learning_rate = float(learning_rate)

    def scores(self, features):
        return self.weights @ expand_features(features)

    def probabilities(self, features, allowed=None, temperature=0.18):
        scores = self.scores(features)
        if allowed is not None:
            mask = np.ones(len(SPECIALISTS), dtype=bool)
            mask[np.asarray(allowed, dtype=np.int64)] = False
            scores[mask] = -1e9
        return softmax(scores, temperature=temperature)

    def choose(self, features, epsilon=0.0, allowed=None):
        allowed = (
            np.arange(len(SPECIALISTS), dtype=np.int64)
            if allowed is None
            else np.asarray(allowed, dtype=np.int64)
        )
        if self.rng.random() < epsilon:
            return int(self.rng.choice(allowed))
        scores = self.scores(features)
        return int(allowed[np.argmax(scores[allowed])])

    def update(self, features, specialist, reward):
        vector = expand_features(features)
        prediction = float(self.weights[specialist] @ vector)
        error = float(np.clip(reward - prediction, -2.5, 2.5))
        self.visits[specialist] += 1
        rate = self.learning_rate / math.sqrt(
            1.0 + 0.00035 * self.visits[specialist]
        )
        self.weights[specialist] += rate * error * vector


def train_conductor(seed, steps=24000):
    rng = np.random.default_rng(seed + 1)
    conductor = ContextualConductor(seed)
    for step in range(steps):
        context = int(rng.integers(0, len(CONTEXTS)))
        features = sample_features(context, rng, domain_shift=0.025)
        fraction = step / max(steps - 1, 1)
        epsilon = 0.30 * (1.0 - fraction) + 0.035
        specialist = conductor.choose(features, epsilon=epsilon)
        success = rng.random() < SUCCESS_PROBABILITY[context, specialist]
        time_cost = max(
            0.0, TIME_COST[context, specialist] + rng.normal(0.0, 0.035)
        )
        collision = max(
            0.0, COLLISION_EXPOSURE[context, specialist] + rng.normal(0.0, 0.025)
        )
        reward = (
            (1.0 if success else -1.20)
            - 0.24 * time_cost
            - 0.34 * collision
            - COMPUTE_COST[specialist]
        )
        conductor.update(features, specialist, reward)
    return conductor


@dataclass
class EvaluationRow:
    condition: str
    seed: int
    block: int
    step: int
    context: str
    specialist: str
    optimal_specialist: str
    success: float
    utility: float
    time_cost: float
    collision_exposure: float
    routing_optimal: float
    gate_entropy: float
    boundary_step: float
    handoff: float
    unnecessary_handoff: float


def make_evaluation_blocks(seed, blocks=240, steps_per_block=10):
    rng = np.random.default_rng(seed)
    result = []
    previous_context = -1
    for block in range(blocks):
        choices = [index for index in range(len(CONTEXTS)) if index != previous_context]
        context = int(rng.choice(choices))
        previous_context = context
        block_rows = []
        for step in range(steps_per_block):
            features = sample_features(context, rng, domain_shift=0.035)
            draws = {
                "success": rng.random(len(SPECIALISTS)),
                "time": rng.normal(size=len(SPECIALISTS)),
                "collision": rng.normal(size=len(SPECIALISTS)),
            }
            block_rows.append((features, draws))
        result.append((context, block_rows))
    return result


def select_specialist(condition, conductor, features, lesion=None):
    allowed = [
        index for index in range(len(SPECIALISTS)) if index != lesion
    ]
    if condition.startswith("always_"):
        specialist = SPECIALISTS.index(condition.removeprefix("always_"))
        probabilities = np.zeros(len(SPECIALISTS), dtype=np.float64)
        probabilities[specialist] = 1.0
        return specialist, probabilities
    if condition == "oracle":
        raise RuntimeError("oracle selection requires context")
    if condition == "scrambled_gate":
        scrambled = np.asarray(features)[[5, 0, 6, 1, 3, 2, 4]]
        probabilities = conductor.probabilities(scrambled, allowed=allowed)
        return int(allowed[np.argmax(probabilities[allowed])]), probabilities
    probabilities = conductor.probabilities(features, allowed=allowed)
    return int(allowed[np.argmax(probabilities[allowed])]), probabilities


def evaluate_condition(
    condition,
    conductor,
    blocks,
    seed,
    lesion=None,
    smoothing=0.62,
):
    rows = []
    attended = None
    previous_specialist = None
    for block_index, (context, block_rows) in enumerate(blocks):
        for step, (features, draws) in enumerate(block_rows):
            attended = (
                np.asarray(features, dtype=np.float64)
                if attended is None
                else smoothing * np.asarray(features) + (1.0 - smoothing) * attended
            )
            if condition == "oracle":
                specialist = int(OPTIMAL_SPECIALIST[context])
                probabilities = np.zeros(len(SPECIALISTS))
                probabilities[specialist] = 1.0
            else:
                specialist, probabilities = select_specialist(
                    condition, conductor, attended, lesion=lesion
                )
            success, time_cost, collision, utility = sampled_outcome(
                context, specialist, draws
            )
            handoff = previous_specialist is not None and specialist != previous_specialist
            unnecessary = handoff and step > 1
            rows.append(
                EvaluationRow(
                    condition=condition,
                    seed=seed,
                    block=block_index,
                    step=step,
                    context=CONTEXTS[context],
                    specialist=SPECIALISTS[specialist],
                    optimal_specialist=SPECIALISTS[OPTIMAL_SPECIALIST[context]],
                    success=float(success),
                    utility=utility,
                    time_cost=time_cost,
                    collision_exposure=collision,
                    routing_optimal=float(specialist == OPTIMAL_SPECIALIST[context]),
                    gate_entropy=gate_entropy(probabilities),
                    boundary_step=float(step <= 1),
                    handoff=float(handoff),
                    unnecessary_handoff=float(unnecessary),
                )
            )
            previous_specialist = specialist
    return rows


def summarize(rows):
    return {
        "success_rate": float(np.mean([row.success for row in rows])),
        "utility_per_step": float(np.mean([row.utility for row in rows])),
        "time_cost_per_step": float(np.mean([row.time_cost for row in rows])),
        "collision_exposure_per_step": float(
            np.mean([row.collision_exposure for row in rows])
        ),
        "optimal_routing_rate": float(
            np.mean([row.routing_optimal for row in rows])
        ),
        "gate_entropy_bits": float(np.mean([row.gate_entropy for row in rows])),
        "boundary_entropy_bits": float(
            np.mean([row.gate_entropy for row in rows if row.boundary_step])
        ),
        "steady_entropy_bits": float(
            np.mean([row.gate_entropy for row in rows if not row.boundary_step])
        ),
        "handoffs_per_100_steps": float(
            100.0 * np.mean([row.handoff for row in rows])
        ),
        "unnecessary_handoffs_per_100_steps": float(
            100.0 * np.mean([row.unnecessary_handoff for row in rows])
        ),
    }


def context_summary(rows):
    return {
        context: summarize([row for row in rows if row.context == context])
        for context in CONTEXTS
    }


def routing_confusion(rows):
    matrix = np.zeros((len(CONTEXTS), len(SPECIALISTS)), dtype=np.float64)
    for context_index, context in enumerate(CONTEXTS):
        selected = [row.specialist for row in rows if row.context == context]
        for specialist_index, specialist in enumerate(SPECIALISTS):
            matrix[context_index, specialist_index] = (
                selected.count(specialist) / len(selected) if selected else 0.0
            )
    return matrix


def sign_test_p(deltas):
    values = np.asarray(deltas)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    positives = int(np.sum(values > 0.0))
    tail = sum(math.comb(len(values), index) for index in range(positives, len(values) + 1))
    return float(tail / (2 ** len(values)))


def plot_results(summary, confusion, lesion_drops, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    baseline_names = [f"always_{name}" for name in SPECIALISTS]
    names = baseline_names + ["learned_conductor", "oracle"]
    axes[0, 0].bar(
        range(len(names)),
        [summary[name]["utility_per_step"] for name in names],
        color=["#8d99ae"] * 4 + ["#168aad", "#52b788"],
    )
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(
        ["rec", "ART", "MPC", "fallback", "conductor", "oracle"],
        rotation=20,
    )
    axes[0, 0].set_ylabel("utility / step")
    axes[0, 0].set_title("Adaptive conductor vs fixed specialists")

    image = axes[0, 1].imshow(confusion, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(range(len(SPECIALISTS)))
    axes[0, 1].set_xticklabels(["rec", "ART", "MPC", "fallback"])
    axes[0, 1].set_yticks(range(len(CONTEXTS)))
    axes[0, 1].set_yticklabels(
        ["clear", "hidden", "visible", "wedge"]
    )
    axes[0, 1].set_title("Learned routing confusion")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    drop_matrix = np.array(
        [
            [lesion_drops[specialist][context] for specialist in SPECIALISTS]
            for context in CONTEXTS
        ]
    )
    image = axes[1, 0].imshow(drop_matrix, cmap="magma")
    axes[1, 0].set_xticks(range(len(SPECIALISTS)))
    axes[1, 0].set_xticklabels(["block rec", "block ART", "block MPC", "block fallback"])
    axes[1, 0].set_yticks(range(len(CONTEXTS)))
    axes[1, 0].set_yticklabels(["clear", "hidden", "visible", "wedge"])
    axes[1, 0].set_title("Utility loss from targeted branch lesion")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    learned = summary["learned_conductor"]
    axes[1, 1].bar(
        ["boundary", "steady"],
        [learned["boundary_entropy_bits"], learned["steady_entropy_bits"]],
        color=["#f4a261", "#168aad"],
    )
    axes[1, 1].set_ylabel("gate entropy (bits)")
    axes[1, 1].set_title(
        "Boundary uncertainty\n"
        f"unnecessary toggles: {learned['unnecessary_handoffs_per_100_steps']:.2f}/100"
    )

    fig.suptitle("Four-Context Adaptive Conductor Benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(seed_count=20, train_steps=24000, blocks=240, steps_per_block=10):
    condition_names = (
        "always_recurrent",
        "always_episodic",
        "always_predictive",
        "always_fallback",
        "scrambled_gate",
        "learned_conductor",
        "oracle",
    )
    all_rows = {name: [] for name in condition_names}
    lesion_rows = {specialist: [] for specialist in SPECIALISTS}
    per_seed = []
    checkpoints = []

    for seed_index in range(seed_count):
        seed = 24071 + 97 * seed_index
        conductor = train_conductor(seed, steps=train_steps)
        evaluation_blocks = make_evaluation_blocks(
            seed + 10000, blocks=blocks, steps_per_block=steps_per_block
        )
        seed_summary = {}
        for condition in condition_names:
            rows = evaluate_condition(
                condition,
                conductor,
                evaluation_blocks,
                seed=seed,
            )
            all_rows[condition].extend(rows)
            seed_summary[condition] = summarize(rows)
        for specialist_index, specialist in enumerate(SPECIALISTS):
            rows = evaluate_condition(
                "learned_conductor",
                conductor,
                evaluation_blocks,
                seed=seed,
                lesion=specialist_index,
            )
            lesion_rows[specialist].extend(rows)
        per_seed.append(seed_summary)
        checkpoints.append(
            {
                "seed": seed,
                "weights": conductor.weights.tolist(),
                "visits": conductor.visits.tolist(),
                "held_out_utility": seed_summary["learned_conductor"][
                    "utility_per_step"
                ],
            }
        )

    summary = {name: summarize(rows) for name, rows in all_rows.items()}
    contexts = {
        name: context_summary(rows) for name, rows in all_rows.items()
    }
    learned_rows = all_rows["learned_conductor"]
    confusion = routing_confusion(learned_rows)
    lesion_summary = {
        specialist: context_summary(rows)
        for specialist, rows in lesion_rows.items()
    }
    lesion_drops = {
        specialist: {
            context: (
                contexts["learned_conductor"][context]["utility_per_step"]
                - lesion_summary[specialist][context]["utility_per_step"]
            )
            for context in CONTEXTS
        }
        for specialist in SPECIALISTS
    }

    best_static = max(
        (f"always_{name}" for name in SPECIALISTS),
        key=lambda name: summary[name]["utility_per_step"],
    )
    utility_deltas = [
        row["learned_conductor"]["utility_per_step"]
        - row[best_static]["utility_per_step"]
        for row in per_seed
    ]
    best_checkpoint = max(checkpoints, key=lambda item: item["held_out_utility"])
    criteria = {
        "learned_beats_every_static_utility": all(
            summary["learned_conductor"]["utility_per_step"]
            > summary[f"always_{specialist}"]["utility_per_step"]
            for specialist in SPECIALISTS
        ),
        "all_context_routing_at_least_90_percent": bool(
            np.all(np.diag(confusion) >= 0.90)
        ),
        "each_targeted_lesion_largest_in_its_matching_context": all(
            max(lesion_drops[specialist], key=lesion_drops[specialist].get)
            == CONTEXTS[index]
            for index, specialist in enumerate(SPECIALISTS)
        ),
        "boundary_entropy_exceeds_steady_entropy": (
            summary["learned_conductor"]["boundary_entropy_bits"]
            > summary["learned_conductor"]["steady_entropy_bits"]
        ),
        "unnecessary_handoffs_below_2_per_100_steps": (
            summary["learned_conductor"]["unnecessary_handoffs_per_100_steps"] < 2.0
        ),
    }
    contrasts = {
        "best_static": best_static,
        "learned_minus_best_static_utility": float(np.mean(utility_deltas)),
        "learned_beats_best_static_seeds": int(np.sum(np.asarray(utility_deltas) > 0.0)),
        "seed_count": seed_count,
        "one_sided_sign_test_p": sign_test_p(utility_deltas),
        "oracle_minus_learned_utility": (
            summary["oracle"]["utility_per_step"]
            - summary["learned_conductor"]["utility_per_step"]
        ),
    }
    payload = {
        "experiment": "four-context adaptive conductor benchmark",
        "contexts": list(CONTEXTS),
        "specialists": list(SPECIALISTS),
        "feature_names": list(FEATURE_NAMES),
        "optimal_specialist_by_context": {
            context: SPECIALISTS[OPTIMAL_SPECIALIST[index]]
            for index, context in enumerate(CONTEXTS)
        },
        "protocol": {
            "seed_count": seed_count,
            "train_steps_per_seed": train_steps,
            "evaluation_blocks_per_seed": blocks,
            "steps_per_block": steps_per_block,
            "true_context_label_visible_to_conductor": False,
            "training_and_evaluation_random_streams_separate": True,
        },
        "summary": summary,
        "context_summary": contexts,
        "routing_confusion": confusion.tolist(),
        "targeted_lesion_summary": lesion_summary,
        "targeted_lesion_utility_drops": lesion_drops,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "best_checkpoint": best_checkpoint,
        "claim_boundary": (
            "A reward-trained contextual gate learned to allocate four frozen "
            "specialists from noisy embodied-style signals and was evaluated "
            "against fixed policies, an oracle, and targeted branch lesions. "
            "This synthetic result tests adaptive executive routing; it does "
            "not establish Unity transfer, open-ended intelligence, or "
            "phenomenal consciousness."
        ),
    }
    return payload


def write_checkpoint(payload, path):
    best = payload["best_checkpoint"]
    checkpoint = {
        "checkpoint_type": "four_context_conductor",
        "version": 1,
        "contexts": payload["contexts"],
        "specialists": payload["specialists"],
        "feature_names": payload["feature_names"],
        "weights": best["weights"],
        "visits": best["visits"],
        "training_seed": best["seed"],
        "held_out_utility": best["held_out_utility"],
        "control_status": "python_only_not_unity_calibrated",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=5 if args.quick else 20,
        train_steps=6000 if args.quick else 24000,
        blocks=80 if args.quick else 240,
        steps_per_block=10,
    )
    OUT.mkdir(exist_ok=True)
    metrics_path = OUT / "four_context_conductor_metrics.json"
    figure_path = OUT / "four_context_conductor_summary.png"
    checkpoint_path = (
        ROOT / "checkpoints" / "four_context_conductor" / "best.json"
    )
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_results(
        payload["summary"],
        np.asarray(payload["routing_confusion"]),
        payload["targeted_lesion_utility_drops"],
        figure_path,
    )
    write_checkpoint(payload, checkpoint_path)
    print("Four-context conductor benchmark complete")
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "paired_seed_contrasts": payload["paired_seed_contrasts"],
                "criteria": payload["criteria"],
            },
            indent=2,
        )
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
