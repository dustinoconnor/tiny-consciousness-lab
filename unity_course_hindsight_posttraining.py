#!/usr/bin/env python3
"""Conservatively post-train course steering from successful Unity hindsight."""

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orbit_exit_posttraining_lab import nearest_action, sequence_observations
from unity_geometry_navigation_lab import evaluate as evaluate_geometry
from unity_posttraining_lab import ALL_COURSES, evaluate_continuous
from upgraded_foraging_pipeline import load_checkpoint


BASE = Path("checkpoints/unity_geometry_posttrained/best.pt")
CHECKPOINT = Path("checkpoints/unity_course_hindsight/best.pt")
METRICS = Path("outputs/unity_course_hindsight_metrics.json")
LOG_DIRS = (
    Path("outputs/unity_shadow/fixed_course_geometry_ab_20260720"),
    Path("outputs/unity_shadow/fixed_course_geometry_ab_remaining_20260720"),
)
TARGET_COURSES = {"lwall", "offsetbarriers"}
SEQUENCE = 24
FUTURE = 12
STRIDE = 3


def position(row):
    value = row.get("position") or [0.0, 0.0, 0.0]
    return np.asarray([float(value[0] or 0.0), float(value[2] or 0.0)], dtype=np.float32)


def episode_rows(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            grouped[int(row.get("trap_episode", 0) or 0)].append(row)
    return grouped


def mine(log_dirs, holdout_seed=37):
    train, holdout, summary = [], [], defaultdict(int)
    for directory in log_dirs:
        for path in sorted(directory.glob("seed_*.jsonl")):
            for rows in episode_rows(path).values():
                if not rows or rows[-1].get("trap_outcome") != "success":
                    continue
                course = str(rows[-1].get("trap_course"))
                if course not in TARGET_COURSES:
                    continue
                seed = int(rows[0].get("controller_seed", 0) or 0)
                destination = holdout if seed == holdout_seed else train
                summary[f"{course}_{'holdout' if destination is holdout else 'train'}_episodes"] += 1
                for index in range(SEQUENCE - 1, len(rows) - FUTURE, STRIDE):
                    future = rows[index : index + FUTURE + 1]
                    points = np.stack([position(row) for row in future])
                    deltas = np.diff(points, axis=0)
                    path_length = float(np.linalg.norm(deltas, axis=1).sum())
                    vector = points[-1] - points[0]
                    net = float(np.linalg.norm(vector))
                    contact = float(np.mean([bool(row.get("body_collision")) for row in future]))
                    if net < 1.2 or net / max(path_length, 1e-6) < 0.62 or contact > 0.25:
                        continue
                    context = rows[index - SEQUENCE + 1 : index + 1]
                    destination.append((sequence_observations(context), nearest_action(vector), course, seed))
    if not train or not holdout:
        raise RuntimeError("Need successful training and held-out Unity episodes for both splits.")
    return train, holdout, dict(summary)


def tensors(samples):
    return (
        torch.tensor(np.stack([sample[0] for sample in samples]), dtype=torch.float32),
        torch.tensor([sample[1] for sample in samples], dtype=torch.long),
    )


def logits_for_sequences(policy, sequences):
    hidden = policy.initial_state(len(sequences))
    logits = None
    for step in range(sequences.shape[1]):
        logits, _value, hidden = policy.step(sequences[:, step], hidden)
    return logits


def accuracy(policy, samples):
    sequences, labels = tensors(samples)
    with torch.no_grad():
        logits = logits_for_sequences(policy, sequences)
    return float((torch.argmax(logits, dim=-1) == labels).float().mean())


def train(base, samples, epochs=100, seed=2027):
    torch.manual_seed(seed)
    policy, payload = load_checkpoint(base)
    reference = copy.deepcopy(policy).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.actor.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(policy.actor.parameters(), lr=2.5e-4, weight_decay=0.02)
    sequences, labels = tensors(samples)
    generator = torch.Generator().manual_seed(seed)
    curve = []
    best_state = copy.deepcopy(policy.state_dict())
    best_loss = math.inf
    for epoch in range(epochs):
        order = torch.randperm(len(sequences), generator=generator)
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            candidate_logits = logits_for_sequences(policy, sequences[batch])
            with torch.no_grad():
                reference_logits = logits_for_sequences(reference, sequences[batch])
            imitation = F.cross_entropy(candidate_logits, labels[batch])
            stability = F.kl_div(
                F.log_softmax(candidate_logits, dim=-1),
                F.softmax(reference_logits, dim=-1),
                reduction="batchmean",
            )
            loss = imitation + 0.18 * stability
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.6)
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(policy.state_dict())
        if epoch % 20 == 0 or epoch == epochs - 1:
            curve.append({"epoch": epoch, "loss": mean_loss})
    policy.load_state_dict(best_state)
    return policy.eval(), payload, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--holdout-seed", type=int, default=37)
    args = parser.parse_args()
    train_samples, holdout_samples, extraction = mine(LOG_DIRS, args.holdout_seed)
    baseline, _ = load_checkpoint(args.base)
    baseline_holdout = accuracy(baseline, holdout_samples)
    baseline_courses = evaluate_continuous(baseline, ALL_COURSES, [31, 47], episodes_per_seed=3)
    baseline_geometry = evaluate_geometry(baseline, episodes=12)
    candidate, payload, curve = train(args.base, train_samples, args.epochs)
    candidate_holdout = accuracy(candidate, holdout_samples)
    candidate_courses = evaluate_continuous(candidate, ALL_COURSES, [31, 47], episodes_per_seed=3)
    candidate_geometry = evaluate_geometry(candidate, episodes=12)
    accepted = (
        candidate_holdout >= baseline_holdout
        and candidate_courses["success_rate"] >= 0.90
        and candidate_courses["success_rate"] >= baseline_courses["success_rate"] - 0.02
        and candidate_geometry["success_rate"] >= 0.90
    )
    metrics = {
        "protocol": {
            "base": str(args.base),
            "trainable_parameters": "actor_readout_only",
            "hindsight_future_steps": FUTURE,
            "heldout_seed": args.holdout_seed,
            "unity_failures_used_as_positive_labels": False,
        },
        "extraction": {
            **extraction,
            "train_samples": len(train_samples),
            "holdout_samples": len(holdout_samples),
        },
        "baseline": {
            "holdout_hindsight_accuracy": baseline_holdout,
            "courses": baseline_courses,
            "geometry": baseline_geometry,
        },
        "candidate": {
            "holdout_hindsight_accuracy": candidate_holdout,
            "courses": candidate_courses,
            "geometry": candidate_geometry,
        },
        "curve": curve,
        "accepted": accepted,
        "claim_boundary": (
            "Successful Unity trajectories provide grounded hindsight directions. "
            "Acceptance still requires a fresh active-control Unity benchmark."
        ),
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if accepted:
        exported = dict(payload)
        exported["state_dict"] = candidate.state_dict()
        exported["config"] = dict(payload["config"])
        exported["config"].update({
            "posttraining": "unity_course_hindsight_v1",
            "posttraining_scope": "actor_readout_only",
        })
        exported["unity_course_hindsight_metrics"] = metrics
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(exported, CHECKPOINT)
        print(f"accepted and exported {CHECKPOINT}")
    else:
        print("candidate rejected; checkpoint not exported")
    print(json.dumps({
        "train_samples": len(train_samples),
        "holdout_samples": len(holdout_samples),
        "baseline_holdout": baseline_holdout,
        "candidate_holdout": candidate_holdout,
        "baseline_courses": baseline_courses["success_rate"],
        "candidate_courses": candidate_courses["success_rate"],
        "accepted": accepted,
    }, indent=2))


if __name__ == "__main__":
    main()
