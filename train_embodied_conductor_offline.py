#!/usr/bin/env python3
"""Train the passive embodied conductor from matched Unity controller episodes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from lwall_hidden_goal_adapter_lab import grouped_episodes


def circular_yaw_travel(rows):
    yaws = [float(row.get("yaw", 0.0) or 0.0) for row in rows]
    return sum(
        abs((current - previous + 180.0) % 360.0 - 180.0)
        for previous, current in zip(yaws, yaws[1:])
    )


def episode_record(rows, specialist, seed, occurrence):
    hidden = []
    for row in rows:
        if bool(row.get("food_visible", False)):
            break
        hidden.append(row)
    if not hidden:
        hidden = list(rows)
    success = any(row.get("trap_outcome") == "success" for row in rows)
    collision_frames = sum(bool(row.get("body_collision")) for row in hidden)
    hidden_frames = len(hidden)
    collision_rate = collision_frames / max(hidden_frames, 1)
    yaw_per_frame = circular_yaw_travel(hidden) / max(hidden_frames, 1)
    # Success dominates. Efficiency and smoothness decide among successful runs.
    utility = (
        (1.0 if success else -1.0)
        - 0.60 * min(1.0, hidden_frames / 450.0)
        - 0.30 * min(1.0, collision_rate / 0.20)
        - 0.10 * min(1.0, yaw_per_frame / 45.0)
    )
    variant = str(rows[0].get("trap_course_variant", "unknown"))
    return {
        "seed": int(seed),
        "variant": variant,
        "occurrence": int(occurrence),
        "specialist": specialist,
        "success": success,
        "hidden_frames": hidden_frames,
        "collision_frames": collision_frames,
        "collision_rate": collision_rate,
        "yaw_per_frame": yaw_per_frame,
        "utility": utility,
    }


def load_condition(recording_dir, specialist):
    records = []
    for recording in sorted(Path(recording_dir).glob("seed_*.jsonl")):
        seed = int(recording.stem.split("_")[-1])
        occurrence = defaultdict(int)
        for _episode, rows in grouped_episodes(recording).items():
            variant = str(rows[0].get("trap_course_variant", "unknown"))
            index = occurrence[variant]
            occurrence[variant] += 1
            records.append(episode_record(rows, specialist, seed, index))
    return records


def matched_pairs(recurrent_records, episodic_records):
    def keyed(records):
        return {
            (record["seed"], record["variant"], record["occurrence"]): record
            for record in records
        }

    recurrent = keyed(recurrent_records)
    episodic = keyed(episodic_records)
    shared = sorted(set(recurrent) & set(episodic))
    if len(shared) != len(recurrent) or len(shared) != len(episodic):
        raise RuntimeError(
            f"unmatched_episode_sets recurrent={len(recurrent)} "
            f"episodic={len(episodic)} shared={len(shared)}"
        )
    return [
        {
            "key": {
                "seed": key[0],
                "variant": key[1],
                "occurrence": key[2],
            },
            "recurrent": recurrent[key],
            "episodic": episodic[key],
        }
        for key in shared
    ]


def fit_context_values(pairs, excluded_seed=None):
    values = defaultdict(list)
    for pair in pairs:
        if excluded_seed is not None and pair["key"]["seed"] == excluded_seed:
            continue
        for specialist in ("recurrent", "episodic"):
            values[specialist].append(pair[specialist]["utility"])
    return {
        specialist: sum(samples) / len(samples)
        for specialist, samples in values.items()
        if samples
    }, {specialist: len(samples) for specialist, samples in values.items()}


def evaluate_pairs(pairs, q_values, seed=None):
    selected = max(q_values, key=lambda specialist: (q_values[specialist], specialist))
    evaluated = [
        pair for pair in pairs if seed is None or pair["key"]["seed"] == seed
    ]
    correct = 0
    regret = 0.0
    selected_utility = 0.0
    recurrent_utility = 0.0
    for pair in evaluated:
        utilities = {
            specialist: pair[specialist]["utility"]
            for specialist in ("recurrent", "episodic")
        }
        best = max(utilities, key=lambda specialist: (utilities[specialist], specialist))
        correct += int(selected == best)
        regret += utilities[best] - utilities[selected]
        selected_utility += utilities[selected]
        recurrent_utility += utilities["recurrent"]
    count = max(len(evaluated), 1)
    return {
        "episodes": len(evaluated),
        "selected_specialist": selected,
        "selection_accuracy": correct / count,
        "mean_regret": regret / count,
        "mean_selected_utility": selected_utility / count,
        "mean_recurrent_utility": recurrent_utility / count,
        "utility_gain_over_recurrent": (
            selected_utility - recurrent_utility
        )
        / count,
    }


def train_and_validate(recurrent_dir, episodic_dir):
    recurrent = load_condition(recurrent_dir, "recurrent")
    episodic = load_condition(episodic_dir, "episodic")
    pairs = matched_pairs(recurrent, episodic)
    seeds = sorted({pair["key"]["seed"] for pair in pairs})
    folds = []
    for seed in seeds:
        q_values, visits = fit_context_values(pairs, excluded_seed=seed)
        folds.append(
            {
                "held_out_seed": seed,
                "train_q": q_values,
                "train_visits": visits,
                "evaluation": evaluate_pairs(pairs, q_values, seed=seed),
            }
        )
    full_q, full_visits = fit_context_values(pairs)
    aggregate = {
        "fold_selection_accuracy": sum(
            fold["evaluation"]["selection_accuracy"] for fold in folds
        )
        / len(folds),
        "fold_mean_regret": sum(
            fold["evaluation"]["mean_regret"] for fold in folds
        )
        / len(folds),
        "fold_utility_gain_over_recurrent": sum(
            fold["evaluation"]["utility_gain_over_recurrent"] for fold in folds
        )
        / len(folds),
        "full_data_evaluation": evaluate_pairs(pairs, full_q),
    }
    return pairs, folds, full_q, full_visits, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recurrent-dir", type=Path, required=True)
    parser.add_argument("--episodic-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/embodied_conductor/offline_v1.json"),
    )
    args = parser.parse_args()

    pairs, folds, q_values, visits, aggregate = train_and_validate(
        args.recurrent_dir,
        args.episodic_dir,
    )
    payload = {
        "checkpoint_type": "passive_embodied_conductor",
        "version": 1,
        "mode": "passive_only",
        "context": "familiar_hidden_goal",
        "specialists": ["recurrent", "episodic"],
        "q_values": {"familiar_hidden_goal": q_values},
        "visits": {"familiar_hidden_goal": visits},
        "training_sources": {
            "recurrent": str(args.recurrent_dir),
            "episodic": str(args.episodic_dir),
        },
        "matched_episode_count": len(pairs),
        "leave_one_seed_out": folds,
        "aggregate": aggregate,
        "utility_definition": (
            "success_or_timeout - normalized hidden duration - hidden collision rate "
            "- normalized hidden yaw travel"
        ),
        "claim_boundary": (
            "This checkpoint learns a recurrent-versus-ART preference for matched Unity "
            "L-wall hidden-goal episodes. It contains no evidence for visible-target MPC "
            "handoff timing or stable fallback selection and remains passive."
        ),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.checkpoint), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
