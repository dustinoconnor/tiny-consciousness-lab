#!/usr/bin/env python3
"""Test Bunge-style systemic emergence in the four-specialist control stack.

Mario Bunge's systemic account treats an emergent property as a property of an
organized system that is absent from its components in isolation and depends
on their causal couplings. This lab operationalizes that narrow claim:

1. Frozen recurrent, episodic, predictive, and fallback specialists are each
   evaluated alone.
2. A reward-trained conductor routes among those same specialists from noisy
   embodied-style signals.
3. Equal-component controls retain all specialists but remove adaptive
   organization.
4. Input and output connections are scrambled without deleting components.
5. Targeted specialist lesions test whether losses are context-specific.

The benchmark measures functional systemic synergy. It is not evidence of
phenomenal consciousness or of biological CNS equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from four_context_conductor_lab import (
    CONTEXTS,
    FEATURE_NAMES,
    OPTIMAL_SPECIALIST,
    SPECIALISTS,
    gate_entropy,
    make_evaluation_blocks,
    sampled_outcome,
    summarize,
    train_conductor,
)
from tiny_lab import OUT, ROOT


@dataclass
class SystemRow:
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


def fixed_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation with no identity connections."""
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != np.arange(size)):
            return permutation


def paired_summary(values) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if not len(samples):
        return {
            "mean": 0.0,
            "standard_deviation": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "positive_seeds": 0,
            "seed_count": 0,
            "one_sided_sign_test_p": 1.0,
            "paired_effect_dz": 0.0,
        }
    mean = float(np.mean(samples))
    standard_deviation = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(len(samples))
    nonzero = samples[np.abs(samples) > 1e-12]
    positives = int(np.sum(nonzero > 0.0))
    sign_tail = (
        sum(
            math.comb(len(nonzero), index)
            for index in range(positives, len(nonzero) + 1)
        )
        / (2 ** len(nonzero))
        if len(nonzero)
        else 1.0
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "positive_seeds": positives,
        "seed_count": int(len(samples)),
        "one_sided_sign_test_p": float(sign_tail),
        "paired_effect_dz": (
            float(mean / standard_deviation)
            if standard_deviation > 1e-12
            else 0.0
        ),
    }


def select_system_specialist(
    condition,
    conductor,
    attended,
    flat_step,
    input_permutation,
    output_permutation,
    lesion=None,
):
    allowed = np.array(
        [
            index
            for index in range(len(SPECIALISTS))
            if lesion is None or index != lesion
        ],
        dtype=np.int64,
    )
    probabilities = np.zeros(len(SPECIALISTS), dtype=np.float64)

    if condition.startswith("isolated_"):
        specialist = SPECIALISTS.index(condition.removeprefix("isolated_"))
        probabilities[specialist] = 1.0
        return specialist, probabilities

    if condition == "static_balanced":
        specialist = int(flat_step % len(SPECIALISTS))
        probabilities[:] = 1.0 / len(SPECIALISTS)
        return specialist, probabilities

    gate_input = np.asarray(attended, dtype=np.float64)
    if condition in {"input_scrambled", "fully_scrambled"}:
        gate_input = gate_input[input_permutation]

    probabilities = conductor.probabilities(gate_input, allowed=allowed)
    selected = int(allowed[np.argmax(probabilities[allowed])])
    if condition in {"output_scrambled", "fully_scrambled"}:
        selected = int(output_permutation[selected])
        if lesion is not None and selected == lesion:
            remaining = [index for index in allowed if index != selected]
            selected = int(remaining[np.argmax(probabilities[remaining])])
    return selected, probabilities


def evaluate_system(
    condition,
    conductor,
    blocks,
    seed,
    input_permutation,
    output_permutation,
    lesion=None,
    smoothing=0.62,
):
    rows = []
    attended = None
    previous_specialist = None
    flat_step = 0
    for block_index, (context, block_rows) in enumerate(blocks):
        for step, (features, draws) in enumerate(block_rows):
            attended = (
                np.asarray(features, dtype=np.float64)
                if attended is None
                else smoothing * np.asarray(features)
                + (1.0 - smoothing) * attended
            )
            specialist, probabilities = select_system_specialist(
                condition,
                conductor,
                attended,
                flat_step,
                input_permutation,
                output_permutation,
                lesion=lesion,
            )
            success, time_cost, collision, utility = sampled_outcome(
                context, specialist, draws
            )
            handoff = previous_specialist is not None and specialist != previous_specialist
            rows.append(
                SystemRow(
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
                    unnecessary_handoff=float(handoff and step > 1),
                )
            )
            previous_specialist = specialist
            flat_step += 1
    return rows


def summarize_rows(rows):
    # Reuse the conductor summary contract through structural duck typing.
    return summarize(rows)


def summarize_contexts(rows):
    return {
        context: summarize_rows([row for row in rows if row.context == context])
        for context in CONTEXTS
    }


def routing_matrix(rows):
    result = np.zeros((len(CONTEXTS), len(SPECIALISTS)), dtype=np.float64)
    for context_index, context in enumerate(CONTEXTS):
        selected = [row.specialist for row in rows if row.context == context]
        for specialist_index, specialist in enumerate(SPECIALISTS):
            result[context_index, specialist_index] = (
                selected.count(specialist) / len(selected) if selected else 0.0
            )
    return result


def plot_results(payload, path):
    summary = payload["summary"]
    contrasts = payload["paired_seed_contrasts"]
    lesion_drops = payload["targeted_lesion_utility_drops"]
    routing = np.asarray(payload["intact_routing_matrix"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    conditions = [
        "best_isolated",
        "static_balanced",
        "intact_coupled",
        "input_scrambled",
        "output_scrambled",
        "fully_scrambled",
    ]
    labels = ["best\nisolated", "static\nall-parts", "intact\nsystem", "input\nscramble", "output\nscramble", "full\nscramble"]
    axes[0, 0].bar(
        labels,
        [summary[name]["utility_per_step"] for name in conditions],
        color=["#8d99ae", "#adb5bd", "#168aad", "#e9c46a", "#f4a261", "#e76f51"],
    )
    axes[0, 0].set_ylabel("utility / step")
    axes[0, 0].set_title("Whole, components, and coupling controls")

    image = axes[0, 1].imshow(routing, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0, 1].set_xticks(range(len(SPECIALISTS)))
    axes[0, 1].set_xticklabels(["rec", "ART", "MPC", "fallback"])
    axes[0, 1].set_yticks(range(len(CONTEXTS)))
    axes[0, 1].set_yticklabels(["clear", "hidden", "visible", "wedge"])
    axes[0, 1].set_title("Intact system routing")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)

    matrix = np.array(
        [
            [lesion_drops[specialist][context] for specialist in SPECIALISTS]
            for context in CONTEXTS
        ]
    )
    image = axes[1, 0].imshow(matrix, cmap="magma")
    axes[1, 0].set_xticks(range(len(SPECIALISTS)))
    axes[1, 0].set_xticklabels(["lesion rec", "lesion ART", "lesion MPC", "lesion fallback"])
    axes[1, 0].set_yticks(range(len(CONTEXTS)))
    axes[1, 0].set_yticklabels(["clear", "hidden", "visible", "wedge"])
    axes[1, 0].set_title("Context-specific utility loss")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    contrast_names = [
        "intact_minus_best_isolated",
        "intact_minus_static_balanced",
        "intact_minus_input_scrambled",
        "intact_minus_output_scrambled",
        "intact_minus_fully_scrambled",
    ]
    axes[1, 1].bar(
        ["best\nisolated", "static\nall-parts", "input\nscramble", "output\nscramble", "full\nscramble"],
        [contrasts[name]["mean"] for name in contrast_names],
        color=["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    )
    axes[1, 1].axhline(0.0, color="#333333", linewidth=1)
    axes[1, 1].set_ylabel("paired utility advantage")
    axes[1, 1].set_title("System-level organizational advantage")

    fig.suptitle("Bunge-Inspired Systemic Emergence Benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(seed_count=20, train_steps=24000, blocks=240, steps_per_block=10):
    isolated = [f"isolated_{specialist}" for specialist in SPECIALISTS]
    conditions = isolated + [
        "static_balanced",
        "intact_coupled",
        "input_scrambled",
        "output_scrambled",
        "fully_scrambled",
    ]
    all_rows = {condition: [] for condition in conditions}
    lesion_rows = {specialist: [] for specialist in SPECIALISTS}
    per_seed = []

    for seed_index in range(seed_count):
        seed = 31091 + 101 * seed_index
        conductor = train_conductor(seed, steps=train_steps)
        blocks_for_seed = make_evaluation_blocks(
            seed + 50000, blocks=blocks, steps_per_block=steps_per_block
        )
        permutation_rng = np.random.default_rng(seed + 70000)
        input_permutation = permutation_rng.permutation(len(FEATURE_NAMES))
        output_permutation = fixed_derangement(len(SPECIALISTS), permutation_rng)

        seed_rows = {}
        seed_summary = {}
        for condition in conditions:
            rows = evaluate_system(
                condition,
                conductor,
                blocks_for_seed,
                seed,
                input_permutation,
                output_permutation,
            )
            seed_rows[condition] = rows
            all_rows[condition].extend(rows)
            seed_summary[condition] = summarize_rows(rows)

        for specialist_index, specialist in enumerate(SPECIALISTS):
            rows = evaluate_system(
                "intact_coupled",
                conductor,
                blocks_for_seed,
                seed,
                input_permutation,
                output_permutation,
                lesion=specialist_index,
            )
            lesion_rows[specialist].extend(rows)

        best_isolated_name = max(
            isolated,
            key=lambda condition: seed_summary[condition]["utility_per_step"],
        )
        seed_summary["best_isolated"] = seed_summary[best_isolated_name]
        seed_summary["best_isolated_name"] = best_isolated_name
        per_seed.append(seed_summary)

    summary = {condition: summarize_rows(rows) for condition, rows in all_rows.items()}
    best_isolated_name = max(
        isolated, key=lambda condition: summary[condition]["utility_per_step"]
    )
    summary["best_isolated"] = summary[best_isolated_name]
    context_summary = {
        condition: summarize_contexts(rows)
        for condition, rows in all_rows.items()
    }
    lesion_summary = {
        specialist: summarize_contexts(rows)
        for specialist, rows in lesion_rows.items()
    }
    lesion_drops = {
        specialist: {
            context: (
                context_summary["intact_coupled"][context]["utility_per_step"]
                - lesion_summary[specialist][context]["utility_per_step"]
            )
            for context in CONTEXTS
        }
        for specialist in SPECIALISTS
    }

    comparison_conditions = [
        "best_isolated",
        "static_balanced",
        "input_scrambled",
        "output_scrambled",
        "fully_scrambled",
    ]
    contrasts = {}
    for comparison in comparison_conditions:
        deltas = [
            seed_result["intact_coupled"]["utility_per_step"]
            - seed_result[comparison]["utility_per_step"]
            for seed_result in per_seed
        ]
        contrasts[f"intact_minus_{comparison}"] = paired_summary(deltas)

    intact_routing = routing_matrix(all_rows["intact_coupled"])
    criteria = {
        "whole_beats_every_isolated_component": all(
            summary["intact_coupled"]["utility_per_step"]
            > summary[condition]["utility_per_step"]
            for condition in isolated
        ),
        "whole_beats_equal_component_static_control": (
            summary["intact_coupled"]["utility_per_step"]
            > summary["static_balanced"]["utility_per_step"]
        ),
        "whole_beats_all_connection_scrambles": all(
            summary["intact_coupled"]["utility_per_step"]
            > summary[condition]["utility_per_step"]
            for condition in ("input_scrambled", "output_scrambled", "fully_scrambled")
        ),
        "whole_beats_best_isolated_in_every_seed": (
            contrasts["intact_minus_best_isolated"]["positive_seeds"] == seed_count
        ),
        "whole_beats_full_scramble_in_every_seed": (
            contrasts["intact_minus_fully_scrambled"]["positive_seeds"] == seed_count
        ),
        "all_context_routing_at_least_90_percent": bool(
            np.all(np.diag(intact_routing) >= 0.90)
        ),
        "each_targeted_lesion_largest_in_matching_context": all(
            max(lesion_drops[specialist], key=lesion_drops[specialist].get)
            == CONTEXTS[index]
            for index, specialist in enumerate(SPECIALISTS)
        ),
    }
    criteria["operational_systemic_emergence_supported"] = all(criteria.values())

    return {
        "experiment": "Bunge-inspired systemic emergence benchmark",
        "operational_definition": (
            "A functional capability counts as system-level in this lab when the "
            "causally coupled whole exceeds every isolated component and an "
            "equal-component nonadaptive control, degrades when connections are "
            "scrambled without removing parts, and exhibits context-specific losses "
            "under targeted component lesions."
        ),
        "contexts": list(CONTEXTS),
        "specialists": list(SPECIALISTS),
        "feature_names": list(FEATURE_NAMES),
        "protocol": {
            "seed_count": seed_count,
            "train_steps_per_seed": train_steps,
            "evaluation_blocks_per_seed": blocks,
            "steps_per_block": steps_per_block,
            "matched_outcome_draws_across_conditions": True,
            "true_context_label_visible_to_conductor": False,
            "component_count_preserved_in_static_and_scrambled_controls": True,
            "training_and_evaluation_random_streams_separate": True,
        },
        "best_isolated_component": best_isolated_name,
        "summary": summary,
        "context_summary": context_summary,
        "intact_routing_matrix": intact_routing.tolist(),
        "targeted_lesion_summary": lesion_summary,
        "targeted_lesion_utility_drops": lesion_drops,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "claim_boundary": (
            "Passing the criteria supports a narrow claim of functional systemic "
            "synergy and causal organizational dependence in this synthetic "
            "benchmark. It does not establish phenomenal consciousness, biological "
            "equivalence, open-ended emergence, or Unity transfer."
        ),
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "bunge_systemic_emergence_metrics.json",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=OUT / "bunge_systemic_emergence_summary.png",
    )
    args = parser.parse_args()

    payload = run_benchmark(
        seed_count=4 if args.quick else 20,
        train_steps=7000 if args.quick else 24000,
        blocks=50 if args.quick else 240,
        steps_per_block=8 if args.quick else 10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(payload, args.plot)

    print("Bunge-inspired systemic emergence benchmark")
    print("=" * 48)
    for condition in (
        "best_isolated",
        "static_balanced",
        "intact_coupled",
        "input_scrambled",
        "output_scrambled",
        "fully_scrambled",
    ):
        metrics = payload["summary"][condition]
        print(
            f"{condition:18s} success={metrics['success_rate']:.3f} "
            f"utility={metrics['utility_per_step']:+.3f} "
            f"routing={metrics['optimal_routing_rate']:.3f}"
        )
    print()
    for name, metrics in payload["paired_seed_contrasts"].items():
        print(
            f"{name}: {metrics['mean']:+.3f} "
            f"95% CI [{metrics['ci95_low']:+.3f}, {metrics['ci95_high']:+.3f}], "
            f"wins={metrics['positive_seeds']}/{metrics['seed_count']}, "
            f"p_sign={metrics['one_sided_sign_test_p']:.3g}"
        )
    print()
    for criterion, passed in payload["criteria"].items():
        print(f"{criterion}: {'PASS' if passed else 'FAIL'}")
    print(f"\nmetrics: {args.output.relative_to(ROOT)}")
    print(f"plot:    {args.plot.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
