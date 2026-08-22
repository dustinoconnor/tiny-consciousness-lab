#!/usr/bin/env python3
"""Mine Unity orbit exits and conservatively post-train the recurrent policy.

The training target is not a scripted breakout maneuver. For each sustained
low-efficiency orbit, hindsight labels the egocentric direction that is followed
by several seconds of clear net displacement. Whole orbit episodes are held out
so overlapping telemetry frames cannot leak between train and evaluation.
"""

import argparse
import copy
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from starvation_posttraining_lab import evaluate_sparse
from unity_posttraining_lab import ALL_COURSES, evaluate_continuous
from upgraded_foraging_pipeline import ACTION_NAMES, MOVES, load_checkpoint


DEFAULT_RECORDING = Path("outputs/unity_shadow/overnight_or_gate_20260718.jsonl")
DEFAULT_BASE = Path("checkpoints/starvation_posttrained/best.pt")
DATASET_PATH = Path("outputs/unity_shadow/orbit_exit_20260718.npz")
METRICS_PATH = Path("outputs/orbit_exit_posttraining_metrics.json")
CHECKPOINT_PATH = Path("checkpoints/orbit_exit_posttrained/best.pt")

SEQUENCE_STEPS = 32
ORBIT_STEPS = 50
FUTURE_STEPS = 15
SAMPLE_STRIDE = 3


def position(row):
    value = row.get("position") or [0.0, 0.0, 0.0]
    return np.asarray([float(value[0] or 0.0), float(value[2] or 0.0)], dtype=np.float32)


def path_stats(rows):
    points = np.stack([position(row) for row in rows])
    deltas = np.diff(points, axis=0)
    path = float(np.linalg.norm(deltas, axis=1).sum())
    net_vector = points[-1] - points[0]
    net = float(np.linalg.norm(net_vector))
    return path, net, net_vector


def nearest_action(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return 0
    unit = vector / norm
    move_units = MOVES.astype(np.float32)
    move_units /= np.linalg.norm(move_units, axis=1, keepdims=True)
    return int(np.argmax(move_units @ unit))


def observation(row, previous_action, reward):
    rays = row.get("rays") or [1.0] * 8
    rays = [float(np.clip(value if value is not None else 1.0, 0.0, 1.0)) for value in rays]
    visible = bool(row.get("food_visible", False))
    food_world = row.get("food_world") or [0.0, 0.0]
    distance = max(0.0, float(row.get("food_distance") or 0.0))
    food_scale = min(1.0, distance / 14.0) if visible else 0.0
    previous = np.zeros(8, dtype=np.float32)
    previous[previous_action] = 1.0
    return np.asarray(
        rays
        + [
            float(visible),
            float(food_world[0] or 0.0) * food_scale,
            float(food_world[1] or 0.0) * food_scale,
            float(row.get("hunger") or 0.0),
        ]
        + previous.tolist()
        + [float(reward)],
        dtype=np.float32,
    )


def sequence_observations(rows):
    observations = []
    previous_action = 0
    previous_pickups = int(rows[0].get("mushroom_pickups_total") or 0)
    for row in rows:
        pickups = int(row.get("mushroom_pickups_total") or 0)
        reward = 1.0 if pickups > previous_pickups else 0.0
        observations.append(observation(row, previous_action, reward))
        action_name = row.get("shadow_action")
        if action_name in ACTION_NAMES:
            previous_action = ACTION_NAMES.index(action_name)
        previous_pickups = pickups
    return np.stack(observations)


def mine_dataset(recording, dataset_path):
    needed = max(SEQUENCE_STEPS, ORBIT_STEPS) + FUTURE_STEPS + 2
    rows = deque(maxlen=needed)
    samples = []
    event_id = -1
    in_orbit = False
    clear_run = 0
    orbit_frames = 0
    source_counts = Counter()

    with recording.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) < needed:
                continue
            values = list(rows)
            target_index = len(values) - FUTURE_STEPS - 1
            context = values[target_index - ORBIT_STEPS + 1 : target_index + 1]
            context_path, context_net, _ = path_stats(context)
            contact_fraction = float(
                np.mean(
                    [
                        bool(row.get("body_collision"))
                        or bool(row.get("stuck"))
                        or float(row.get("physics_wedge_seconds") or 0.0) > 0.0
                        for row in context
                    ]
                )
            )
            orbiting = context_path >= 14.0 and context_net / max(context_path, 1e-6) <= 0.22
            orbiting = orbiting and (contact_fraction >= 0.06 or bool(context[-1].get("food_visible")))
            if orbiting:
                if not in_orbit:
                    event_id += 1
                    in_orbit = True
                    orbit_frames = 0
                clear_run = 0
                orbit_frames += 1
            elif in_orbit:
                clear_run += 1
                if clear_run >= 20:
                    in_orbit = False
                    orbit_frames = 0
            if not orbiting or orbit_frames % SAMPLE_STRIDE:
                continue

            future = values[target_index : target_index + FUTURE_STEPS + 1]
            future_path, future_net, future_vector = path_stats(future)
            future_contact = float(
                np.mean([bool(row.get("body_collision")) or bool(row.get("stuck")) for row in future[1:]])
            )
            if future_net < 3.0 or future_net / max(future_path, 1e-6) < 0.62 or future_contact > 0.25:
                continue
            sequence = values[target_index - SEQUENCE_STEPS + 1 : target_index + 1]
            fallback = any(bool(row.get("fallback_active")) for row in future)
            source_counts["fallback_assisted" if fallback else "self_recovered"] += 1
            samples.append(
                (
                    sequence_observations(sequence),
                    nearest_action(future_vector),
                    event_id,
                    float(context_path),
                    float(context_net),
                    float(future_net),
                    float(future_net / max(future_path, 1e-6)),
                    int(fallback),
                )
            )

    if not samples:
        raise RuntimeError("No orbit-exit samples met the grounded hindsight criteria.")
    arrays = list(zip(*samples))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        sequences=np.stack(arrays[0]).astype(np.float32),
        labels=np.asarray(arrays[1], dtype=np.int64),
        event_ids=np.asarray(arrays[2], dtype=np.int64),
        context_path=np.asarray(arrays[3], dtype=np.float32),
        context_net=np.asarray(arrays[4], dtype=np.float32),
        future_net=np.asarray(arrays[5], dtype=np.float32),
        future_efficiency=np.asarray(arrays[6], dtype=np.float32),
        fallback_assisted=np.asarray(arrays[7], dtype=np.int8),
    )
    return {
        "samples": len(samples),
        "orbit_events": event_id + 1,
        "source_counts": dict(source_counts),
        "dataset": str(dataset_path),
    }


def split_events(event_ids):
    events = np.unique(event_ids)
    if len(events) < 2:
        raise RuntimeError("At least two independent orbit events are required.")
    rng = np.random.default_rng(1907)
    rng.shuffle(events)
    holdout_count = max(1, int(math.ceil(len(events) * 0.20)))
    heldout = set(events[:holdout_count].tolist())
    test_mask = np.asarray([event in heldout for event in event_ids])
    return ~test_mask, test_mask, sorted(heldout)


def sequence_logits(policy, sequences):
    hidden = policy.initial_state(len(sequences))
    logits = None
    for step in range(sequences.shape[1]):
        logits, _value, hidden = policy.step(sequences[:, step], hidden)
    return logits


def accuracy(policy, sequences, labels, batch_size=128):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            logits = sequence_logits(policy, torch.tensor(sequences[start : start + batch_size]))
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
    return float(np.mean(np.asarray(predictions) == labels))


def posttrain(base_path, sequences, labels, train_mask, epochs=28):
    torch.manual_seed(1907)
    policy, payload = load_checkpoint(base_path)
    reference = copy.deepcopy(policy).eval()
    # Preserve the learned sensorimotor representation and adapt only the
    # action readout. This asks whether exit information is already present in
    # recurrent state and avoids rewriting the navigation core from one run.
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.actor.parameters():
        parameter.requires_grad = True
    policy.train()
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=5e-5,
    )
    train_sequences = torch.tensor(sequences[train_mask])
    train_labels = torch.tensor(labels[train_mask])
    curves = []
    generator = torch.Generator().manual_seed(1907)
    for epoch in range(epochs):
        order = torch.randperm(len(train_sequences), generator=generator)
        losses = []
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            batch = train_sequences[indices]
            target = train_labels[indices]
            logits = sequence_logits(policy, batch)
            with torch.no_grad():
                reference_logits = sequence_logits(reference, batch)
            imitation = F.cross_entropy(logits, target)
            kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(reference_logits, dim=-1),
                reduction="batchmean",
            )
            loss = imitation + 0.35 * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.6)
            optimizer.step()
            losses.append(float(loss.detach()))
        curves.append(float(np.mean(losses)))
    return policy.eval(), payload, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, nargs="?", default=DEFAULT_RECORDING)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=28)
    args = parser.parse_args()

    extraction = mine_dataset(args.recording, args.dataset)
    data = np.load(args.dataset)
    sequences = data["sequences"]
    labels = data["labels"]
    train_mask, test_mask, heldout_events = split_events(data["event_ids"])

    baseline, _ = load_checkpoint(args.base)
    baseline_exit_accuracy = accuracy(baseline, sequences[test_mask], labels[test_mask])
    candidate, source_payload, curves = posttrain(
        args.base, sequences, labels, train_mask, epochs=args.epochs
    )
    candidate_exit_accuracy = accuracy(candidate, sequences[test_mask], labels[test_mask])

    course_seeds = [31, 47, 59]
    baseline_courses = evaluate_continuous(baseline, ALL_COURSES, course_seeds, episodes_per_seed=4)
    candidate_courses = evaluate_continuous(candidate, ALL_COURSES, course_seeds, episodes_per_seed=4)
    sparse_seeds = [81_101 + 977 * index for index in range(8)]
    baseline_sparse = evaluate_sparse(baseline, sparse_seeds)
    candidate_sparse = evaluate_sparse(candidate, sparse_seeds)
    candidate_memory_reset = evaluate_sparse(candidate, sparse_seeds, reset_memory=True)

    criteria = {
        "heldout_exit_accuracy_improves_5_points": candidate_exit_accuracy >= baseline_exit_accuracy + 0.05,
        "course_success_regression_at_most_2_points": candidate_courses["success_rate"] >= baseline_courses["success_rate"] - 0.02,
        "course_collisions_not_increased": candidate_courses["mean_collisions"] <= baseline_courses["mean_collisions"] + 0.25,
        "sparse_pickups_retain_90_percent": candidate_sparse["mean_pickups"] >= baseline_sparse["mean_pickups"] * 0.90,
        "memory_reset_reduces_sparse_pickups": candidate_memory_reset["mean_pickups"] < candidate_sparse["mean_pickups"] * 0.60,
    }
    criteria["all_passed"] = all(criteria.values())
    metrics = {
        "protocol": {
            "recording": str(args.recording),
            "base_checkpoint": str(args.base),
            "sequence_steps": SEQUENCE_STEPS,
            "orbit_context_steps": ORBIT_STEPS,
            "future_hindsight_steps": FUTURE_STEPS,
            "event_level_holdout": True,
            "heldout_event_ids": heldout_events,
            "scripted_breakout_labels": False,
            "trainable_parameters": "actor_readout_only",
        },
        "extraction": extraction,
        "split": {"train_samples": int(train_mask.sum()), "heldout_samples": int(test_mask.sum())},
        "baseline_exit_accuracy": baseline_exit_accuracy,
        "candidate_exit_accuracy": candidate_exit_accuracy,
        "baseline_courses": baseline_courses,
        "candidate_courses": candidate_courses,
        "baseline_sparse": baseline_sparse,
        "candidate_sparse": candidate_sparse,
        "candidate_memory_reset": candidate_memory_reset,
        "criteria": criteria,
        "training_loss": curves,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if criteria["all_passed"]:
        payload = dict(source_payload)
        payload["state_dict"] = candidate.state_dict()
        payload["config"] = dict(source_payload["config"])
        payload["config"].update({"posttraining": "unity_orbit_hindsight_actor_v1"})
        payload["orbit_exit_posttraining_metrics"] = metrics
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.checkpoint)
        print(f"exported {args.checkpoint}")
    else:
        print("candidate rejected; baseline checkpoint remains preferred")
    print(json.dumps({"extraction": extraction, "criteria": criteria, "metrics": str(args.metrics)}, indent=2))


if __name__ == "__main__":
    main()
