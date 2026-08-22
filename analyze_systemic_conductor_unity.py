#!/usr/bin/env python3
"""Replay passive Unity frames through intact and disrupted systemic gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from embodied_systemic_conductor import (
    CONTEXTS,
    SPECIALISTS,
    PassiveSystemicConductor,
    live_features,
    proxy_context,
)


INPUT_SCRAMBLE = (5, 0, 6, 1, 3, 2, 4)
OUTPUT_SCRAMBLE = (1, 2, 3, 0)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def exact_mcnemar_one_sided(intact_correct, control_correct):
    intact_only = sum(
        first and not second
        for first, second in zip(intact_correct, control_correct)
    )
    control_only = sum(
        second and not first
        for first, second in zip(intact_correct, control_correct)
    )
    discordant = intact_only + control_only
    if not discordant:
        return 1.0
    return sum(
        math.comb(discordant, index)
        for index in range(intact_only, discordant + 1)
    ) / (2 ** discordant)


def load_rows(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            features = row.get("systemic_conductor_features")
            if not (isinstance(features, list) and len(features) == 7):
                if not any(
                    key in row
                    for key in (
                        "directional_body_clearance",
                        "directional_rays",
                        "food_visible",
                        "trap_pressure",
                    )
                ):
                    continue
                features = live_features(row)
                row["systemic_conductor_features"] = features
                row.setdefault("systemic_conductor_action_influence", 0)
            context = proxy_context(row, features)
            row["systemic_conductor_proxy_context"] = context
            if (
                isinstance(features, list)
                and len(features) == 7
                and context in CONTEXTS
            ):
                rows.append(row)
    return rows


def evaluate_condition(rows, checkpoint, condition="intact", lesion=None):
    observer = PassiveSystemicConductor(checkpoint)
    attended = None
    selected = []
    correct = []
    contexts = []
    entropies = []
    confidences = []
    for row in rows:
        features = [float(value) for value in row["systemic_conductor_features"]]
        attended = (
            features
            if attended is None
            else [
                observer.smoothing * value
                + (1.0 - observer.smoothing) * previous
                for value, previous in zip(features, attended)
            ]
        )
        input_permutation = (
            INPUT_SCRAMBLE
            if condition in {"input_scrambled", "fully_scrambled"}
            else None
        )
        output_permutation = (
            OUTPUT_SCRAMBLE
            if condition in {"output_scrambled", "fully_scrambled"}
            else None
        )
        index, _scores, probabilities = observer.predict(
            attended,
            input_permutation=input_permutation,
            output_permutation=output_permutation,
            lesion=lesion,
        )
        recommendation = SPECIALISTS[index]
        context = row["systemic_conductor_proxy_context"]
        optimal = SPECIALISTS[CONTEXTS.index(context)]
        ranking = sorted(probabilities, reverse=True)
        selected.append(recommendation)
        correct.append(recommendation == optimal)
        contexts.append(context)
        entropies.append(
            -sum(
                probability * math.log2(probability)
                for probability in probabilities
                if probability > 1e-12
            )
        )
        confidences.append(ranking[0] - ranking[1])

    by_context = {}
    for context in CONTEXTS:
        indices = [index for index, value in enumerate(contexts) if value == context]
        by_context[context] = {
            "frames": len(indices),
            "accuracy": mean([correct[index] for index in indices]),
            "recommendations": dict(
                Counter(selected[index] for index in indices)
            ),
        }
    return {
        "frames": len(rows),
        "accuracy": mean(correct),
        "mean_confidence": mean(confidences),
        "mean_entropy_bits": mean(entropies),
        "recommendations": dict(Counter(selected)),
        "by_context": by_context,
        "_correct": correct,
    }


def analyze(path, checkpoint):
    rows = load_rows(path)
    conditions = {
        condition: evaluate_condition(rows, checkpoint, condition)
        for condition in (
            "intact",
            "input_scrambled",
            "output_scrambled",
            "fully_scrambled",
        )
    }
    lesions = {
        specialist: evaluate_condition(
            rows, checkpoint, "intact", lesion=index
        )
        for index, specialist in enumerate(SPECIALISTS)
    }
    context_counts = Counter(
        row["systemic_conductor_proxy_context"] for row in rows
    )
    lesion_drops = {
        specialist: {
            context: (
                conditions["intact"]["by_context"][context]["accuracy"]
                - lesions[specialist]["by_context"][context]["accuracy"]
            )
            for context in CONTEXTS
        }
        for specialist in SPECIALISTS
    }
    contrasts = {}
    intact_correct = conditions["intact"]["_correct"]
    for condition in ("input_scrambled", "output_scrambled", "fully_scrambled"):
        control = conditions[condition]
        contrasts[f"intact_minus_{condition}"] = {
            "accuracy_difference": (
                conditions["intact"]["accuracy"] - control["accuracy"]
            ),
            "one_sided_exact_mcnemar_p": exact_mcnemar_one_sided(
                intact_correct, control["_correct"]
            ),
        }

    action_influence = max(
        (
            int(row.get("systemic_conductor_action_influence", 0) or 0)
            for row in rows
        ),
        default=0,
    )
    criteria = {
        "all_four_contexts_have_at_least_25_frames": all(
            context_counts[context] >= 25 for context in CONTEXTS
        ),
        "intact_accuracy_at_least_75_percent": (
            conditions["intact"]["accuracy"] >= 0.75
        ),
        "each_context_accuracy_at_least_75_percent": all(
            conditions["intact"]["by_context"][context]["accuracy"] >= 0.75
            for context in CONTEXTS
        ),
        "intact_beats_each_scramble_by_10_points": all(
            contrasts[f"intact_minus_{condition}"]["accuracy_difference"] >= 0.10
            for condition in ("input_scrambled", "output_scrambled", "fully_scrambled")
        ),
        "each_matching_lesion_has_positive_drop": all(
            lesion_drops[specialist][CONTEXTS[index]] > 0.0
            for index, specialist in enumerate(SPECIALISTS)
        ),
        "observer_had_zero_action_influence": action_influence == 0,
    }
    criteria["passive_unity_calibration_supported"] = all(criteria.values())

    for result in list(conditions.values()) + list(lesions.values()):
        result.pop("_correct", None)
    return {
        "experiment": "passive Unity systemic-conductor transfer",
        "source": str(path),
        "checkpoint": str(checkpoint),
        "frames": len(rows),
        "context_counts": dict(context_counts),
        "conditions": conditions,
        "lesions": lesions,
        "targeted_lesion_accuracy_drops": lesion_drops,
        "paired_contrasts": contrasts,
        "criteria": criteria,
        "claim_boundary": (
            "This replays one passive Unity recording against telemetry-derived "
            "context proxies. Passing supports calibration and organizational "
            "dependence in shadow mode, not motor benefit, context-label validity, "
            "or phenomenal consciousness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/four_context_conductor/best.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/systemic_conductor_unity_metrics.json"),
    )
    args = parser.parse_args()
    result = analyze(args.recording, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"frames={result['frames']} "
        f"intact_accuracy={result['conditions']['intact']['accuracy']:.3f}"
    )
    print(f"contexts={result['context_counts']}")
    for name, contrast in result["paired_contrasts"].items():
        print(
            f"{name}: delta={contrast['accuracy_difference']:+.3f} "
            f"p={contrast['one_sided_exact_mcnemar_p']:.3g}"
        )
    for criterion, passed in result["criteria"].items():
        print(f"{criterion}: {'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
