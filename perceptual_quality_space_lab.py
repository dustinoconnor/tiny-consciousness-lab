#!/usr/bin/env python3
"""Matched RPT-2/HOT-4 perceptual intelligence benchmark.

The task requires binding early continuous object qualities to identities across
crossing, partial occlusion, and a final side choice. HOT-4 auxiliary targets
contain qualities only and never contain the correct action.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_SEEDS = (3101, 3102, 3103, 3104)
VALIDATION_SEEDS = (3201, 3202)
RESERVED_SEEDS = (3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308)
CONDITIONS = ("flat", "rpt2", "hot4", "rpt2_hot4", "rpt2_hot4_permuted")
STEPS = 6
QUALITY_DIM = 4
SLOT_DIM = 1 + QUALITY_DIM + QUALITY_DIM
INPUT_DIM = 2 * SLOT_DIM


@dataclass
class EpisodeBatch:
    observations: torch.Tensor
    needs: torch.Tensor
    qualities: torch.Tensor
    choices: torch.Tensor
    identity_left: torch.Tensor

    def subset(self, indices):
        return EpisodeBatch(
            self.observations[indices],
            self.needs[indices],
            self.qualities[indices],
            self.choices[indices],
            self.identity_left[indices],
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def quality_vector(angle, roundness, size):
    return np.array(
        [np.sin(angle), np.cos(angle), roundness, size], dtype=np.float32
    )


def sample_quality(rng, reserved):
    if reserved:
        arc = int(rng.integers(0, 4))
        angle = arc * (np.pi / 2.0) + np.pi / 4.0 + rng.normal(0.0, 0.12)
        roundness = rng.uniform(0.12, 0.88)
        size = np.clip(1.0 - roundness + rng.normal(0.0, 0.08), 0.05, 0.95)
    else:
        arc = int(rng.integers(0, 4))
        angle = arc * (np.pi / 2.0) + rng.normal(0.0, 0.12)
        roundness = rng.uniform(0.05, 0.95)
        size = np.clip(roundness + rng.normal(0.0, 0.08), 0.05, 0.95)
    return quality_vector(angle, roundness, size)


def generate_episodes(seed, count, reserved=False):
    rng = np.random.default_rng(seed)
    observations = np.zeros((count, STEPS, INPUT_DIM), dtype=np.float32)
    needs = np.zeros((count, QUALITY_DIM), dtype=np.float32)
    qualities = np.zeros((count, 2 * QUALITY_DIM), dtype=np.float32)
    choices = np.zeros(count, dtype=np.int64)
    identity_left = np.zeros(count, dtype=np.int64)

    for episode in range(count):
        q = np.stack([sample_quality(rng, reserved) for _ in range(2)])
        while np.linalg.norm(q[0] - q[1]) < 0.55:
            q[1] = sample_quality(rng, reserved)
        target = int(rng.integers(0, 2))
        needs[episode] = q[target] + rng.normal(0.0, 0.045, QUALITY_DIM)
        qualities[episode] = q.reshape(-1)

        initial_left_identity = int(rng.integers(0, 2))
        initial = np.zeros(2, dtype=np.float32)
        initial[initial_left_identity] = -rng.uniform(0.75, 1.05)
        initial[1 - initial_left_identity] = rng.uniform(0.75, 1.05)
        crosses = bool(rng.random() < 0.72)
        if crosses:
            final = -initial + rng.normal(0.0, 0.08, 2)
        else:
            final = initial + rng.normal(0.0, 0.12, 2)
        curvature = rng.normal(0.0, 0.10 if not reserved else 0.16, 2)

        final_order = np.argsort(final)
        identity_left[episode] = int(final_order[0])
        choices[episode] = int(np.where(final_order == target)[0][0])

        for step in range(STEPS):
            progress = step / (STEPS - 1)
            positions = (
                (1.0 - progress) * initial
                + progress * final
                + curvature * np.sin(np.pi * progress)
            )
            positions += rng.normal(0.0, 0.018 if not reserved else 0.028, 2)
            slot_order = np.argsort(positions)
            frame = []
            for slot, identity in enumerate(slot_order):
                if step == 0:
                    mask = np.ones(QUALITY_DIM, dtype=np.float32)
                elif step == 1:
                    mask = np.array([1, 1, 0, 0], dtype=np.float32)
                elif step == 2:
                    mask = np.array([0, 0, 1, 1], dtype=np.float32)
                    if rng.random() < (0.30 if not reserved else 0.55):
                        mask[:] = 0
                elif step == 3:
                    mask = (rng.random(QUALITY_DIM) > (0.50 if not reserved else 0.70)).astype(np.float32)
                else:
                    mask = np.zeros(QUALITY_DIM, dtype=np.float32)
                sensed = q[identity] + rng.normal(
                    0.0, 0.025 if not reserved else 0.045, QUALITY_DIM
                )
                frame.extend([positions[identity], *(sensed * mask), *mask])
            observations[episode, step] = np.asarray(frame, dtype=np.float32)

    return EpisodeBatch(
        torch.from_numpy(observations),
        torch.from_numpy(needs),
        torch.from_numpy(qualities),
        torch.from_numpy(choices),
        torch.from_numpy(identity_left),
    )


def concatenate(batches):
    return EpisodeBatch(*(
        torch.cat([getattr(batch, field) for batch in batches], dim=0)
        for field in EpisodeBatch.__dataclass_fields__
    ))


class PerceptualModel(nn.Module):
    def __init__(self, recurrent, hot4, hidden=40, code_dim=12):
        super().__init__()
        self.recurrent = recurrent
        self.hot4 = hot4
        self.frame = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.Tanh())
        if recurrent:
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden)
            )
        self.code = nn.Linear(hidden, code_dim)
        nn.init.constant_(self.code.bias, 0.10)
        self.decoder = nn.Linear(code_dim, 2 * QUALITY_DIM)
        self.choice = nn.Sequential(
            nn.Linear(hidden + code_dim + QUALITY_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, observations, needs):
        frames = self.frame(observations)
        if self.recurrent:
            _, hidden = self.temporal(frames)
            state = hidden[-1]
        else:
            state = self.temporal(frames.mean(dim=1))
        code = F.relu(self.code(state))
        return (
            self.choice(torch.cat([state, code, needs], dim=1)),
            code,
            self.decoder(code),
            state,
        )


def geometry_loss(code, qualities):
    shifted_code = torch.roll(code, 1, 0)
    shifted_quality = torch.roll(qualities, 1, 0)
    code_distance = torch.linalg.vector_norm(code - shifted_code, dim=1)
    quality_distance = torch.linalg.vector_norm(qualities - shifted_quality, dim=1)
    # A detached floor prevents a collapsed code from creating an unbounded
    # normalization gradient during development.
    code_distance = code_distance / code_distance.mean().detach().clamp_min(0.25)
    quality_distance = quality_distance / (quality_distance.mean().detach() + 1e-6)
    return F.smooth_l1_loss(code_distance, quality_distance)


def train_model(condition, train, validation, initialization_seed, epochs=40):
    seed_everything(initialization_seed)
    recurrent = condition.startswith("rpt2")
    hot4 = "hot4" in condition
    permuted = condition.endswith("permuted")
    # 59 flat units (13,629 parameters) match 40 recurrent units (13,558)
    # within 0.6%, preventing recurrence from buying capacity by parameter count.
    model = PerceptualModel(recurrent, hot4, hidden=40 if recurrent else 59)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(initialization_seed + 71)
    best_state = None
    best_accuracy = -1.0
    hot4_start_epoch = epochs // 2
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train.choices), generator=generator)
        for start in range(0, len(order), 128):
            batch = train.subset(order[start : start + 128])
            logits, code, reconstruction, _ = model(batch.observations, batch.needs)
            loss = F.cross_entropy(logits, batch.choices)
            if hot4 and epoch >= hot4_start_epoch:
                targets = batch.qualities
                if permuted:
                    targets = targets[torch.randperm(len(targets), generator=generator)]
                loss = (
                    loss
                    + 0.08 * F.mse_loss(reconstruction, targets)
                    + 0.03 * geometry_loss(code, targets)
                    + 0.0015 * code.abs().mean()
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        accuracy = evaluate_batch(model, validation)["accuracy"]
        if accuracy > best_accuracy and (not hot4 or epoch >= hot4_start_epoch):
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def distance_correlation(code, qualities, seed=11, pairs=4096):
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(code), pairs)
    second = rng.integers(0, len(code), pairs)
    code_distance = np.linalg.norm(code[first] - code[second], axis=1)
    quality_distance = np.linalg.norm(qualities[first] - qualities[second], axis=1)
    return float(np.corrcoef(rankdata(code_distance), rankdata(quality_distance))[0, 1])


def ridge_probe(train_state, train_labels, test_state, test_labels, ridge=1e-2):
    train_x = np.column_stack([train_state, np.ones(len(train_state))])
    test_x = np.column_stack([test_state, np.ones(len(test_state))])
    target = np.eye(2)[train_labels]
    weights = np.linalg.solve(train_x.T @ train_x + ridge * np.eye(train_x.shape[1]), train_x.T @ target)
    return float(np.mean(np.argmax(test_x @ weights, axis=1) == test_labels))


@torch.no_grad()
def raw_outputs(model, batch, shuffle_sequence=False, shuffle_seed=99):
    observations = batch.observations
    if shuffle_sequence:
        generator = torch.Generator().manual_seed(shuffle_seed)
        orders = torch.stack([
            torch.randperm(STEPS, generator=generator) for _ in range(len(observations))
        ])
        observations = torch.gather(
            observations, 1, orders[:, :, None].expand(-1, -1, INPUT_DIM)
        )
    logits, code, reconstruction, state = model(observations, batch.needs)
    return (
        logits.numpy(), code.numpy(), reconstruction.numpy(), state.numpy()
    )


def evaluate_batch(model, batch, shuffle_sequence=False):
    logits, code, reconstruction, state = raw_outputs(model, batch, shuffle_sequence)
    return {
        "accuracy": float(np.mean(np.argmax(logits, axis=1) == batch.choices.numpy())),
        "reconstruction_mse": float(np.mean((reconstruction - batch.qualities.numpy()) ** 2)),
        "quality_distance_spearman": distance_correlation(code, batch.qualities.numpy()),
        "active_coordinate_fraction": float(np.mean(code > 0.05)),
        "state": state,
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run(include_reserved=False, train_count=900, validation_count=500, reserved_count=450):
    train = concatenate([generate_episodes(seed, train_count) for seed in TRAIN_SEEDS])
    validation = concatenate([
        generate_episodes(seed, validation_count) for seed in VALIDATION_SEEDS
    ])
    reserved_batches = (
        {seed: generate_episodes(seed, reserved_count, reserved=True) for seed in RESERVED_SEEDS}
        if include_reserved else {}
    )
    rows = []
    models = {}
    for condition in CONDITIONS:
        for initialization_seed in TRAIN_SEEDS:
            model = train_model(condition, train, validation, initialization_seed)
            models[(condition, initialization_seed)] = model
            validation_metrics = evaluate_batch(model, validation)
            train_outputs = raw_outputs(model, train)
            for evaluation_seed, batch in reserved_batches.items():
                metrics = evaluate_batch(model, batch)
                probe = ridge_probe(
                    train_outputs[3],
                    train.identity_left.numpy(),
                    metrics.pop("state"),
                    batch.identity_left.numpy(),
                )
                shuffled = evaluate_batch(model, batch, shuffle_sequence=True)
                rows.append({
                    "condition": condition,
                    "initialization_seed": initialization_seed,
                    "evaluation_seed": evaluation_seed,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "reserved_accuracy": metrics["accuracy"],
                    "identity_binding_probe": probe,
                    "reconstruction_mse": metrics["reconstruction_mse"],
                    "quality_distance_spearman": metrics["quality_distance_spearman"],
                    "active_coordinate_fraction": metrics["active_coordinate_fraction"],
                    "shuffled_sequence_accuracy": shuffled["accuracy"],
                })
    if not include_reserved:
        return {
            "mode": "development_only",
            "validation": {
                condition: summarize([
                    evaluate_batch(models[(condition, seed)], validation)["accuracy"]
                    for seed in TRAIN_SEEDS
                ])
                for condition in CONDITIONS
            },
        }

    aggregate = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {
            key: summarize([row[key] for row in condition_rows])
            for key in (
                "validation_accuracy",
                "reserved_accuracy",
                "identity_binding_probe",
                "reconstruction_mse",
                "quality_distance_spearman",
                "active_coordinate_fraction",
                "shuffled_sequence_accuracy",
            )
        }
    rpt_delta = aggregate["rpt2"]["reserved_accuracy"]["mean"] - aggregate["flat"]["reserved_accuracy"]["mean"]
    rpt_shuffle_delta = aggregate["rpt2"]["reserved_accuracy"]["mean"] - aggregate["rpt2"]["shuffled_sequence_accuracy"]["mean"]
    hot_delta = aggregate["rpt2_hot4"]["reserved_accuracy"]["mean"] - aggregate["rpt2"]["reserved_accuracy"]["mean"]
    geometry_control_delta = aggregate["rpt2_hot4"]["quality_distance_spearman"]["mean"] - aggregate["rpt2_hot4_permuted"]["quality_distance_spearman"]["mean"]
    combined_advantage = aggregate["rpt2_hot4"]["reserved_accuracy"]["mean"] - max(
        aggregate["rpt2"]["reserved_accuracy"]["mean"],
        aggregate["hot4"]["reserved_accuracy"]["mean"],
    )
    validation_tax = max(
        aggregate["rpt2"]["validation_accuracy"]["mean"],
        aggregate["hot4"]["validation_accuracy"]["mean"],
    ) - aggregate["rpt2_hot4"]["validation_accuracy"]["mean"]
    decisions = {
        "rpt2_supported": bool(rpt_delta >= 0.05 and rpt_shuffle_delta >= 0.03),
        "hot4_supported": bool(
            aggregate["rpt2_hot4"]["quality_distance_spearman"]["mean"] >= 0.70
            and aggregate["rpt2_hot4"]["active_coordinate_fraction"]["mean"] <= 0.60
            and hot_delta >= 0.03
            and geometry_control_delta >= 0.20
        ),
        "combined_adds_intelligence": bool(combined_advantage >= 0.03 and validation_tax <= 0.03),
    }
    return {
        "mode": "frozen_reserved",
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "reserved_seeds": RESERVED_SEEDS,
        "rows": rows,
        "aggregate": aggregate,
        "contrasts": {
            "rpt2_minus_flat_reserved_accuracy": rpt_delta,
            "rpt2_sequence_shuffle_loss": rpt_shuffle_delta,
            "hot4_increment_over_rpt2": hot_delta,
            "hot4_geometry_over_permuted_control": geometry_control_delta,
            "combined_over_best_single": combined_advantage,
            "combined_validation_tax": validation_tax,
        },
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserved", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-count", type=int, default=900)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--reserved-count", type=int, default=450)
    args = parser.parse_args()
    result = run(args.reserved, args.train_count, args.validation_count, args.reserved_count)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
