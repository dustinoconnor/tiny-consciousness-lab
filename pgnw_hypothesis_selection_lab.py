#!/usr/bin/env python3
"""Predictive-workspace selection among explicit causal hypotheses.

This is a synthetic Bayesian experimental-design assay. Candidate hypotheses
remain passive descriptions; selected experiments do not control Unity.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

from bunge_systemic_emergence_lab import paired_summary
from tiny_lab import OUT


HYPOTHESES = (
    "red_causes_probe",
    "blue_causes_probe",
    "either_pickup_causes_probe",
    "probe_is_spontaneous",
    "no_tested_cause",
)
ACTIONS = ("observe_red_pickup", "observe_blue_pickup", "wait_no_pickup")
CONDITIONS = (
    "pgnw_efe",
    "epistemic_only",
    "pragmatic_only",
    "posterior_confirmation",
    "random_selection",
    "pgnw_broadcast_scramble",
)

# P(probe rise | hypothesis, experiment). Rows are hypotheses, columns actions.
LIKELIHOOD_RISE = np.array(
    [
        [0.92, 0.08, 0.05],
        [0.08, 0.92, 0.05],
        [0.92, 0.92, 0.05],
        [0.55, 0.55, 0.65],
        [0.05, 0.05, 0.05],
    ],
    dtype=np.float64,
)
PRAGMATIC_VALUE = np.array([0.15, 0.15, -0.01], dtype=np.float64)
PRAGMATIC_WEIGHT = 0.35


def entropy_bits(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    active = values[values > 1e-15]
    return float(-np.sum(active * np.log2(active)))


def posterior_after(prior, action, rise):
    prior = np.asarray(prior, dtype=np.float64)
    likelihood = LIKELIHOOD_RISE[:, int(action)]
    if not rise:
        likelihood = 1.0 - likelihood
    posterior = prior * likelihood
    total = float(np.sum(posterior))
    if total <= 1e-15:
        return np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))
    return posterior / total


def expected_information_gain(prior, action):
    prior = np.asarray(prior, dtype=np.float64)
    rise_probability = float(
        prior @ LIKELIHOOD_RISE[:, int(action)]
    )
    rise_posterior = posterior_after(prior, action, True)
    flat_posterior = posterior_after(prior, action, False)
    expected_entropy = (
        rise_probability * entropy_bits(rise_posterior)
        + (1.0 - rise_probability) * entropy_bits(flat_posterior)
    )
    return entropy_bits(prior) - expected_entropy


def action_scores(condition, posterior):
    epistemic = np.array(
        [expected_information_gain(posterior, action) for action in range(len(ACTIONS))]
    )
    expected_rise = np.asarray(posterior) @ LIKELIHOOD_RISE
    if condition in {"pgnw_efe", "pgnw_broadcast_scramble"}:
        return epistemic + PRAGMATIC_WEIGHT * PRAGMATIC_VALUE
    if condition == "epistemic_only":
        return epistemic
    if condition == "pragmatic_only":
        return PRAGMATIC_VALUE.copy()
    if condition == "posterior_confirmation":
        return expected_rise
    raise ValueError(f"condition_has_no_deterministic_scores:{condition}")


def select_ranked(scores, tie_order):
    scores = np.asarray(scores, dtype=np.float64)
    order_rank = {int(action): rank for rank, action in enumerate(tie_order)}
    return max(
        range(len(scores)),
        key=lambda action: (float(scores[action]), -order_rank[action]),
    )


def run_episode(condition, true_hypothesis, seed, budget=8):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown_pgnw_condition:{condition}")
    rng = np.random.default_rng(seed)
    outcome_draws = rng.random((budget, len(ACTIONS)))
    random_actions = rng.integers(0, len(ACTIONS), size=budget)
    tie_order = rng.permutation(len(ACTIONS))
    scramble_offsets = rng.integers(1, len(ACTIONS), size=budget)
    posterior = np.full(len(HYPOTHESES), 1.0 / len(HYPOTHESES))
    rows = []
    reached_step = None

    for step in range(budget):
        if condition == "random_selection":
            selected = int(random_actions[step])
            scores = np.zeros(len(ACTIONS), dtype=np.float64)
        else:
            scores = action_scores(condition, posterior)
            selected = select_ranked(scores, tie_order)
        executed = selected
        if condition == "pgnw_broadcast_scramble":
            executed = int(
                (selected + int(scramble_offsets[step])) % len(ACTIONS)
            )

        rise_probability = LIKELIHOOD_RISE[
            int(true_hypothesis), executed
        ]
        rise = bool(outcome_draws[step, executed] < rise_probability)
        prior_entropy = entropy_bits(posterior)
        information_gain = expected_information_gain(posterior, executed)
        posterior = posterior_after(posterior, executed, rise)
        true_probability = float(posterior[int(true_hypothesis)])
        if reached_step is None and true_probability >= 0.90:
            reached_step = step + 1
        rows.append(
            {
                "condition": condition,
                "true_hypothesis": HYPOTHESES[int(true_hypothesis)],
                "step": step + 1,
                "broadcast_experiment": ACTIONS[selected],
                "executed_experiment": ACTIONS[executed],
                "broadcast_execution_agreement": float(selected == executed),
                "probe_rise": float(rise),
                "prior_entropy_bits": prior_entropy,
                "expected_information_gain_bits": information_gain,
                "pragmatic_value": float(PRAGMATIC_VALUE[executed]),
                "true_posterior": true_probability,
                "map_hypothesis": HYPOTHESES[int(np.argmax(posterior))],
                "posterior": {
                    name: float(value)
                    for name, value in zip(HYPOTHESES, posterior)
                },
                "broadcast_scores": {
                    action: float(value)
                    for action, value in zip(ACTIONS, scores)
                },
            }
        )
    return {
        "rows": rows,
        "final_posterior": posterior,
        "reached_step": reached_step,
        "correct": int(np.argmax(posterior)) == int(true_hypothesis),
    }


def summarize_episodes(episodes, budget):
    final_true = []
    correct = []
    reached = []
    steps = []
    pragmatic = []
    agreement = []
    action_counts = Counter()
    for episode in episodes:
        true_name = episode["rows"][0]["true_hypothesis"]
        true_index = HYPOTHESES.index(true_name)
        final_true.append(float(episode["final_posterior"][true_index]))
        correct.append(float(episode["correct"]))
        reached.append(float(episode["reached_step"] is not None))
        steps.append(
            float(
                episode["reached_step"]
                if episode["reached_step"] is not None
                else budget + 1
            )
        )
        pragmatic.append(
            sum(row["pragmatic_value"] for row in episode["rows"])
        )
        agreement.extend(
            row["broadcast_execution_agreement"] for row in episode["rows"]
        )
        action_counts.update(
            row["executed_experiment"] for row in episode["rows"]
        )
    return {
        "identification_accuracy": float(np.mean(correct)),
        "mean_true_posterior": float(np.mean(final_true)),
        "reached_090_fraction": float(np.mean(reached)),
        "mean_steps_to_090_or_censored": float(np.mean(steps)),
        "mean_cumulative_pragmatic_value": float(np.mean(pragmatic)),
        "broadcast_execution_agreement": float(np.mean(agreement)),
        "executed_experiment_counts": dict(action_counts),
    }


def run_benchmark(seed_count=20, budget=8, seed_start=0):
    per_seed = {condition: [] for condition in CONDITIONS}
    all_episodes = {condition: [] for condition in CONDITIONS}

    for local_seed_index in range(seed_count):
        seed_index = int(seed_start) + local_seed_index
        seed_episodes = {condition: [] for condition in CONDITIONS}
        for true_hypothesis in range(len(HYPOTHESES)):
            episode_seed = (
                21_000 + seed_index * 1009 + true_hypothesis * 101
            )
            for condition in CONDITIONS:
                episode = run_episode(
                    condition,
                    true_hypothesis,
                    episode_seed,
                    budget=budget,
                )
                seed_episodes[condition].append(episode)
                all_episodes[condition].append(episode)
        for condition in CONDITIONS:
            per_seed[condition].append(
                summarize_episodes(seed_episodes[condition], budget)
            )

    summary = {
        condition: summarize_episodes(episodes, budget)
        for condition, episodes in all_episodes.items()
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
        "pgnw_minus_pragmatic_true_posterior": contrast(
            "mean_true_posterior", "pgnw_efe", "pragmatic_only"
        ),
        "pgnw_minus_random_true_posterior": contrast(
            "mean_true_posterior", "pgnw_efe", "random_selection"
        ),
        "pgnw_minus_scramble_true_posterior": contrast(
            "mean_true_posterior", "pgnw_efe", "pgnw_broadcast_scramble"
        ),
        "pgnw_minus_epistemic_true_posterior": contrast(
            "mean_true_posterior", "pgnw_efe", "epistemic_only"
        ),
        "pgnw_minus_epistemic_pragmatic_value": contrast(
            "mean_cumulative_pragmatic_value", "pgnw_efe", "epistemic_only"
        ),
        "pgnw_minus_confirmation_true_posterior": contrast(
            "mean_true_posterior", "pgnw_efe", "posterior_confirmation"
        ),
    }
    criteria = {
        "pgnw_beats_pragmatic_only": (
            contrasts["pgnw_minus_pragmatic_true_posterior"]["ci95_low"] > 0.0
        ),
        "pgnw_beats_random": (
            contrasts["pgnw_minus_random_true_posterior"]["ci95_low"] > 0.0
        ),
        "bound_broadcast_beats_scramble": (
            contrasts["pgnw_minus_scramble_true_posterior"]["ci95_low"] > 0.0
        ),
        "pgnw_information_noninferior_to_epistemic": (
            contrasts["pgnw_minus_epistemic_true_posterior"]["ci95_low"]
            >= -0.03
        ),
        "pgnw_pragmatic_value_beats_epistemic": (
            contrasts["pgnw_minus_epistemic_pragmatic_value"]["ci95_low"] > 0.0
        ),
    }
    return {
        "experiment": "predictive GNW multiple-hypothesis selection",
        "seed_start": int(seed_start),
        "seed_count": seed_count,
        "counterbalanced_worlds_per_seed": len(HYPOTHESES),
        "experiment_budget": budget,
        "episode_count_per_condition": seed_count * len(HYPOTHESES),
        "hypotheses": list(HYPOTHESES),
        "experiments": list(ACTIONS),
        "likelihood_rise": LIKELIHOOD_RISE.tolist(),
        "pragmatic_value": PRAGMATIC_VALUE.tolist(),
        "pragmatic_weight": PRAGMATIC_WEIGHT,
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "claim_boundary": (
            "Passing supports formal selection among explicit causal models "
            "in this synthetic assay. It does not establish biological PGNW, "
            "conscious access, autonomous LLM hypothesis invention, Unity "
            "intervention control, or natural-world causal discovery."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument(
        "--output",
        default="outputs/pgnw_hypothesis_selection_20260809.json",
    )
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=args.seeds,
        budget=args.budget,
        seed_start=args.seed_start,
    )
    path = OUT.parent / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "pgnw="
        f"{payload['summary']['pgnw_efe']['mean_true_posterior']:.4f} "
        "epistemic="
        f"{payload['summary']['epistemic_only']['mean_true_posterior']:.4f} "
        "pragmatic="
        f"{payload['summary']['pragmatic_only']['mean_true_posterior']:.4f} "
        "random="
        f"{payload['summary']['random_selection']['mean_true_posterior']:.4f}"
    )
    for name, passed in payload["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {payload['all_criteria_pass']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
