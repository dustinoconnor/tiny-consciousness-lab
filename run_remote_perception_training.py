#!/usr/bin/env python3
"""Run resumable feedback trials with prediction committed before target reveal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets

from remote_perception_agent import (
    LocalGemmaIntuitionSampler,
    aggregate_impressions,
    update_slot_weights,
)
from remote_perception_lab import (
    DESCRIPTOR_FIELDS,
    descriptor_score,
    prediction_commitment,
    rank_candidates,
    sha256_json,
)


def append_durable(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_trials(records):
    committed = {row["trial_id"] for row in records if row.get("event") == "commit"}
    revealed = {row["trial_id"] for row in records if row.get("event") == "reveal"}
    if not revealed.issubset(committed):
        raise ValueError("remote_reveal_without_commit")
    return revealed


def reveal_training_trial(private_manifest_path, trial_id):
    private = json.loads(Path(private_manifest_path).read_text())
    cards = {item["target_id"]: item for item in private["training_cards"]}
    matches = [item for item in private["training_trials"] if item["trial_id"] == trial_id]
    if len(matches) != 1:
        raise ValueError("remote_training_trial_not_unique")
    trial = matches[0]
    return trial, cards[trial["target_id"]], [cards[item] for item in trial["candidate_ids"]]


def public_training_schedule(private_manifest_path):
    private = json.loads(Path(private_manifest_path).read_text())
    return [item["trial_id"] for item in private["training_trials"]]


def load_state(path, slots):
    if not path.exists():
        return {"format": "remote_intuition_training_v1", "slot_weights": [1.0] * slots}
    state = json.loads(path.read_text())
    if state.get("format") != "remote_intuition_training_v1":
        raise ValueError("remote_training_state_format")
    if len(state.get("slot_weights", [])) != slots:
        raise ValueError("remote_training_state_slot_count")
    return state


def save_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_one_training_trial(
    sampler,
    trial_id,
    private_manifest_path,
    log_path,
    state,
    samples=7,
    temperature=0.9,
):
    impressions, raw, malformed = sampler.sample(
        trial_id, count=samples, temperature=temperature
    )
    prediction, support = aggregate_impressions(impressions, state["slot_weights"])
    nonce = secrets.token_hex(16)
    commitment = prediction_commitment(trial_id, prediction, nonce)
    commit_record = {
        "event": "commit",
        "trial_id": trial_id,
        "prediction": prediction,
        "prediction_sha256": commitment,
        "nonce": nonce,
        "workspace_support": support,
        "impressions": impressions,
        "raw_impressions": raw,
        "malformed": malformed,
        "slot_weights_before": list(state["slot_weights"]),
    }
    append_durable(log_path, commit_record)

    trial, target, candidates = reveal_training_trial(private_manifest_path, trial_id)
    if prediction_commitment(trial_id, prediction, nonce) != commitment:
        raise ValueError("remote_prediction_commitment_mismatch")
    ranking = rank_candidates(prediction, candidates)
    updated_weights = update_slot_weights(
        state["slot_weights"], impressions, target
    )
    reveal_record = {
        "event": "reveal",
        "trial_id": trial_id,
        "prediction_sha256": commitment,
        "target_id": trial["target_id"],
        "target": {field: target[field] for field in DESCRIPTOR_FIELDS},
        "candidate_ids": trial["candidate_ids"],
        "ranking": ranking,
        "top1_correct": ranking[0]["target_id"] == trial["target_id"],
        "descriptor_matches": descriptor_score(prediction, target),
        "slot_weights_after": updated_weights,
    }
    append_durable(log_path, reveal_record)
    state["slot_weights"] = updated_weights
    state["completed_feedback_trials"] = int(state.get("completed_feedback_trials", 0)) + 1
    state["last_trial_id"] = trial_id
    state["state_sha256"] = sha256_json(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    return reveal_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/remote_perception_deck_20260818/private_manifest.json",
    )
    parser.add_argument(
        "--log", default="outputs/remote_perception_training_20260818.jsonl"
    )
    parser.add_argument(
        "--state", default="outputs/remote_perception_training_state_20260818.json"
    )
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-trials", type=int, default=1)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    log_path = Path(args.log)
    state_path = Path(args.state)
    schedule = public_training_schedule(manifest)
    records = load_records(log_path)
    done = completed_trials(records)
    dangling = [
        row["trial_id"]
        for row in records
        if row.get("event") == "commit" and row["trial_id"] not in done
    ]
    if dangling:
        raise ValueError(f"remote_training_dangling_commit:{dangling[0]}")
    pending = [trial_id for trial_id in schedule if trial_id not in done]
    state = load_state(state_path, args.samples)
    if not pending:
        print("Training feedback bank complete")
        return
    sampler = LocalGemmaIntuitionSampler(args.model)
    for trial_id in pending[: max(0, args.max_new_trials)]:
        result = run_one_training_trial(
            sampler,
            trial_id,
            manifest,
            log_path,
            state,
            samples=args.samples,
            temperature=args.temperature,
        )
        save_state(state_path, state)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
