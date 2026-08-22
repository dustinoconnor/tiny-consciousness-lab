#!/usr/bin/env python3
"""Run the frozen no-feedback remote-perception reserve with matched controls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import secrets

from remote_perception_agent import LocalGemmaIntuitionSampler, aggregate_impressions
from remote_perception_lab import (
    DESCRIPTOR_FIELDS,
    descriptor_score,
    rank_candidates,
    sha256_json,
    validate_prediction,
)
from run_remote_perception_training import append_durable, load_records


CONDITIONS = ("active", "ablation", "sham")
SHAM_SHIFT = 37
FORMAT = "remote_perception_evaluation_v1"


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_public_schedule(manifest_path, public_commitment_path, schedule_path):
    private = json.loads(Path(manifest_path).read_text())
    public = json.loads(Path(public_commitment_path).read_text())
    if sha256_json(private) != public.get("private_manifest_sha256"):
        raise ValueError("remote_private_manifest_commitment_mismatch")
    trial_ids = [str(item["trial_id"]) for item in private["reserved_trials"]]
    if len(trial_ids) != public.get("reserved_targets") or len(set(trial_ids)) != len(trial_ids):
        raise ValueError("remote_reserved_schedule_invalid")
    schedule = {
        "format": FORMAT,
        "conditions": list(CONDITIONS),
        "sham_shift": SHAM_SHIFT,
        "trial_ids": trial_ids,
        "private_manifest_sha256": public["private_manifest_sha256"],
    }
    schedule["schedule_sha256"] = sha256_json(schedule)
    path = Path(schedule_path)
    if path.exists():
        if json.loads(path.read_text()) != schedule:
            raise ValueError("remote_public_schedule_already_differs")
    else:
        write_json_atomic(path, schedule)
    return schedule


def load_schedule(path):
    schedule = json.loads(Path(path).read_text())
    claimed = schedule.pop("schedule_sha256", None)
    if schedule.get("format") != FORMAT or tuple(schedule.get("conditions", ())) != CONDITIONS:
        raise ValueError("remote_evaluation_schedule_format")
    if claimed != sha256_json(schedule):
        raise ValueError("remote_evaluation_schedule_hash")
    schedule["schedule_sha256"] = claimed
    if len(set(schedule["trial_ids"])) != len(schedule["trial_ids"]):
        raise ValueError("remote_evaluation_schedule_duplicate")
    return schedule


def load_frozen_weights(path, samples):
    state = json.loads(Path(path).read_text())
    claimed = state.get("state_sha256")
    if claimed != sha256_json({key: value for key, value in state.items() if key != "state_sha256"}):
        raise ValueError("remote_frozen_state_hash")
    if state.get("completed_feedback_trials") != 20:
        raise ValueError("remote_feedback_bank_not_complete")
    weights = [float(value) for value in state.get("slot_weights", [])]
    if len(weights) != samples:
        raise ValueError("remote_frozen_state_slot_count")
    return weights, claimed


def condition_commitment(trial_id, condition, prediction, nonce):
    return sha256_json(
        {
            "trial_id": str(trial_id),
            "condition": str(condition),
            "prediction": validate_prediction(prediction),
            "nonce": str(nonce),
        }
    )


def expected_keys(schedule):
    return {(trial_id, condition) for trial_id in schedule["trial_ids"] for condition in CONDITIONS}


def validate_records(records, schedule):
    expected = expected_keys(schedule)
    commits = {}
    reveals = {}
    for row in records:
        if row.get("event") not in ("commit", "reveal"):
            raise ValueError("remote_evaluation_unknown_event")
        key = (row.get("trial_id"), row.get("condition"))
        if key not in expected:
            raise ValueError("remote_evaluation_unexpected_key")
        bucket = commits if row["event"] == "commit" else reveals
        if key in bucket:
            raise ValueError("remote_evaluation_duplicate_record")
        bucket[key] = row
    if not set(reveals).issubset(commits):
        raise ValueError("remote_evaluation_reveal_without_commit")
    if reveals and set(commits) != expected:
        raise ValueError("remote_evaluation_reveal_before_all_commits")
    return commits, reveals


def condition_order(index):
    offset = index % len(CONDITIONS)
    return CONDITIONS[offset:] + CONDITIONS[:offset]


def commit_all(sampler, schedule, log_path, weights, state_sha256, samples=7, temperature=0.9):
    records = load_records(log_path)
    commits, reveals = validate_records(records, schedule)
    if reveals:
        return commits
    for index, trial_id in enumerate(schedule["trial_ids"]):
        for condition in condition_order(index):
            key = (trial_id, condition)
            if key in commits:
                continue
            impressions, raw, malformed = sampler.sample(
                trial_id, count=samples, temperature=temperature
            )
            if condition == "ablation":
                prediction = validate_prediction(impressions[0])
                support = None
            else:
                prediction, support = aggregate_impressions(impressions, weights)
            nonce = secrets.token_hex(16)
            commitment = condition_commitment(trial_id, condition, prediction, nonce)
            row = {
                "event": "commit",
                "trial_id": trial_id,
                "trial_position": index,
                "condition": condition,
                "condition_order": list(condition_order(index)),
                "prediction": prediction,
                "prediction_sha256": commitment,
                "nonce": nonce,
                "workspace_support": support,
                "impressions": impressions,
                "raw_impressions": raw,
                "malformed": malformed,
                "samples": samples,
                "temperature": temperature,
                "frozen_state_sha256": state_sha256,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            append_durable(Path(log_path), row)
            commits[key] = row
    if set(commits) != expected_keys(schedule):
        raise ValueError("remote_evaluation_commit_bank_incomplete")
    return commits


def reveal_all(manifest_path, public_commitment_path, schedule, log_path):
    records = load_records(log_path)
    commits, reveals = validate_records(records, schedule)
    if set(commits) != expected_keys(schedule):
        raise ValueError("remote_evaluation_reveal_before_all_commits")
    private = json.loads(Path(manifest_path).read_text())
    public = json.loads(Path(public_commitment_path).read_text())
    if sha256_json(private) != public.get("private_manifest_sha256"):
        raise ValueError("remote_private_manifest_commitment_mismatch")
    if schedule["private_manifest_sha256"] != public["private_manifest_sha256"]:
        raise ValueError("remote_schedule_manifest_mismatch")
    trials = private["reserved_trials"]
    by_trial = {item["trial_id"]: item for item in trials}
    cards = {item["target_id"]: item for item in private["reserved_cards"]}
    if [item["trial_id"] for item in trials] != schedule["trial_ids"]:
        raise ValueError("remote_private_schedule_mismatch")
    total = len(trials)
    shift = int(schedule["sham_shift"]) % total
    if not shift:
        raise ValueError("remote_sham_shift_identity")
    for index, trial_id in enumerate(schedule["trial_ids"]):
        for condition in CONDITIONS:
            key = (trial_id, condition)
            if key in reveals:
                continue
            commit = commits[key]
            assigned = trials[(index + shift) % total] if condition == "sham" else by_trial[trial_id]
            target = cards[assigned["target_id"]]
            candidates = [cards[target_id] for target_id in assigned["candidate_ids"]]
            if condition_commitment(
                trial_id, condition, commit["prediction"], commit["nonce"]
            ) != commit["prediction_sha256"]:
                raise ValueError("remote_evaluation_commitment_mismatch")
            ranking = rank_candidates(commit["prediction"], candidates)
            target_rank = next(
                rank for rank, item in enumerate(ranking, start=1)
                if item["target_id"] == assigned["target_id"]
            )
            row = {
                "event": "reveal",
                "trial_id": trial_id,
                "condition": condition,
                "prediction_sha256": commit["prediction_sha256"],
                "assigned_trial_id": assigned["trial_id"],
                "sham_shift": shift if condition == "sham" else 0,
                "target_id": assigned["target_id"],
                "target": {field: target[field] for field in DESCRIPTOR_FIELDS},
                "candidate_ids": assigned["candidate_ids"],
                "ranking": ranking,
                "target_rank": target_rank,
                "top1_correct": target_rank == 1,
                "descriptor_matches": descriptor_score(commit["prediction"], target),
            }
            append_durable(Path(log_path), row)
            reveals[key] = row
    return reveals


def binomial_greater_equal(hits, trials, probability=0.25):
    return sum(
        math.comb(trials, value)
        * probability**value
        * (1.0 - probability) ** (trials - value)
        for value in range(hits, trials + 1)
    )


def evaluation_summary(log_path, schedule):
    commits, reveals = validate_records(load_records(log_path), schedule)
    complete = set(reveals) == expected_keys(schedule)
    result = {
        "format": FORMAT,
        "complete": complete,
        "schedule_sha256": schedule["schedule_sha256"],
        "commits": len(commits),
        "reveals": len(reveals),
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = [row for (_trial, name), row in reveals.items() if name == condition]
        hits = sum(bool(row["top1_correct"]) for row in rows)
        result["conditions"][condition] = {
            "trials": len(rows),
            "top1_hits": hits,
            "top1_accuracy": hits / len(rows) if rows else None,
            "one_sided_binomial_p": binomial_greater_equal(hits, len(rows)) if rows else None,
            "mean_descriptor_matches": (
                sum(row["descriptor_matches"] for row in rows) / len(rows) if rows else None
            ),
            "mean_target_rank": (
                sum(row["target_rank"] for row in rows) / len(rows) if rows else None
            ),
            "malformed_samples": sum(
                len(commits[(row["trial_id"], condition)]["malformed"]) for row in rows
            ),
            "target_rank_counts": dict(sorted(Counter(row["target_rank"] for row in rows).items())),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/remote_perception_deck_20260818/private_manifest.json")
    parser.add_argument("--public", default="outputs/remote_perception_deck_20260818/public_commitment.json")
    parser.add_argument("--schedule", default="outputs/remote_perception_evaluation_schedule_20260818.json")
    parser.add_argument("--state", default="outputs/remote_perception_training_state_20260818.json")
    parser.add_argument("--log", default="outputs/remote_perception_evaluation_20260818.jsonl")
    parser.add_argument("--summary", default="outputs/remote_perception_evaluation_summary_20260818.json")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--prepare-schedule-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_schedule_only:
        schedule = prepare_public_schedule(args.manifest, args.public, args.schedule)
        print(json.dumps({"schedule_sha256": schedule["schedule_sha256"], "trials": len(schedule["trial_ids"])}, indent=2))
        return
    schedule = load_schedule(args.schedule)
    weights, state_sha256 = load_frozen_weights(args.state, args.samples)
    records = load_records(Path(args.log))
    commits, reveals = validate_records(records, schedule)
    if set(commits) != expected_keys(schedule):
        sampler = LocalGemmaIntuitionSampler(args.model)
        commit_all(
            sampler, schedule, Path(args.log), weights, state_sha256,
            samples=args.samples, temperature=args.temperature,
        )
    if set(reveals) != expected_keys(schedule):
        reveal_all(args.manifest, args.public, schedule, Path(args.log))
    summary = evaluation_summary(Path(args.log), schedule)
    write_json_atomic(Path(args.summary), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
