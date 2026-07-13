#!/usr/bin/env python3
"""Calibrate recurrent forward heads on recorded Unity terrain transitions."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from upgraded_foraging_pipeline import ACTION_NAMES, CORE_DIM, MOVES, load_checkpoint


OUTPUT_PATH = Path("outputs/unity_mpc_calibration_metrics.json")
CHECKPOINT_PATH = Path("checkpoints/unity_mpc/best.pt")


def latest_recording():
    recordings = sorted(Path("outputs/unity_shadow").glob("shadow_*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not recordings:
        raise FileNotFoundError("No Unity shadow recordings found")
    return recordings[-1]


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("rays"), list) and len(row["rays"]) == 8:
                rows.append(row)
    if len(rows) < 100:
        raise ValueError("Recording is too short for calibration")
    return rows


def action_index(name):
    try:
        return ACTION_NAMES.index(name)
    except ValueError:
        return 0


def core_from_row(row):
    rays = [float(np.clip(value, 0.0, 1.0)) for value in row["rays"]]
    visible = 1.0 if row.get("food_visible", False) else 0.0
    distance = max(0.0, float(row.get("food_distance", 0.0)))
    scale = min(1.0, distance / 14.0) if visible else 0.0
    world = row.get("food_world") or [0.0, 0.0]
    food = [float(world[0]) * scale, float(world[1]) * scale]
    hunger = float(np.clip(row.get("hunger", 0.0), 0.0, 1.0))
    return np.asarray(rays + [visible] + food + [hunger], dtype=np.float32)


def observation_from_row(row, previous_action):
    previous = np.zeros(8, dtype=np.float32)
    previous[int(previous_action)] = 1.0
    return np.concatenate([core_from_row(row), previous, np.zeros(1, dtype=np.float32)])


def build_features(policy, rows):
    hidden = policy.initial_state(1)
    previous_action = 0
    features = []
    with torch.no_grad():
        for index in range(len(rows) - 1):
            row = rows[index]
            obs = torch.tensor(observation_from_row(row, previous_action)).unsqueeze(0)
            _logits, _value, hidden = policy.step(obs, hidden)
            proposed = action_index(row.get("shadow_action", "up"))
            if row.get("shadow_takeover", False):
                target = core_from_row(rows[index + 1])
                body = row.get("body_clearance") or [1.0] * 8
                constrained = bool(row.get("body_collision", False) or row.get("stuck", False) or sum(v >= 0.5 for v in body) < 8)
                features.append(
                    {
                        "row_index": index,
                        "hidden": hidden[0].clone(),
                        "action": proposed,
                        "target": torch.tensor(target),
                        "constrained": constrained,
                    }
                )
                previous_action = proposed
            else:
                previous_action = action_index(row.get("active_action", "up"))
    return features


def split_features(features, row_count):
    train_edge = int(row_count * 0.70)
    validation_edge = int(row_count * 0.85)
    return (
        [item for item in features if item["row_index"] < train_edge],
        [item for item in features if train_edge <= item["row_index"] < validation_edge],
        [item for item in features if item["row_index"] >= validation_edge],
    )


def tensors(items):
    return (
        torch.stack([item["hidden"] for item in items]),
        torch.tensor([item["action"] for item in items], dtype=torch.long),
        torch.stack([item["target"] for item in items]),
        torch.tensor([item["constrained"] for item in items], dtype=torch.bool),
    )


def predict(policy, hidden, actions):
    with torch.no_grad():
        ensemble = policy.predict_core(hidden, actions)
    return ensemble


def metrics(policy, items):
    hidden, actions, target, constrained = tensors(items)
    ensemble = predict(policy, hidden, actions)
    mean = ensemble.mean(dim=0)
    absolute = torch.abs(mean - target)
    per_sample = absolute.mean(dim=-1)
    uncertainty = torch.var(ensemble, dim=0).mean(dim=-1)
    if len(items) > 1 and float(torch.std(uncertainty)) > 1e-9 and float(torch.std(per_sample)) > 1e-9:
        correlation = float(torch.corrcoef(torch.stack([uncertainty, per_sample]))[0, 1])
    else:
        correlation = 0.0
    constrained_values = per_sample[constrained]
    open_values = per_sample[~constrained]
    return {
        "samples": len(items),
        "mae": float(absolute.mean()),
        "ray_mae": float(absolute[:, :8].mean()),
        "food_visible_mae": float(absolute[:, 8].mean()),
        "food_vector_mae": float(absolute[:, 9:11].mean()),
        "hunger_mae": float(absolute[:, 11].mean()),
        "constrained_mae": float(constrained_values.mean()) if len(constrained_values) else 0.0,
        "open_mae": float(open_values.mean()) if len(open_values) else 0.0,
        "mean_uncertainty": float(uncertainty.mean()),
        "uncertainty_error_correlation": correlation,
    }


def calibrate(policy, train, validation, epochs=80, batch_size=256, seed=913):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for parameter in policy.forward_heads.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(policy.forward_heads.parameters(), lr=8e-4, weight_decay=1e-5)
    train_hidden, train_actions, train_target, train_constrained = tensors(train)
    validation_hidden, validation_actions, validation_target, validation_constrained = tensors(validation)
    best_state = copy.deepcopy(policy.forward_heads.state_dict())
    best_validation = math.inf
    history = []

    for epoch in range(epochs):
        order = rng.permutation(len(train))
        losses = []
        policy.train()
        for start in range(0, len(order), batch_size):
            indices = torch.tensor(order[start : start + batch_size], dtype=torch.long)
            hidden = train_hidden[indices]
            actions = train_actions[indices]
            target = train_target[indices]
            weights = 1.0 + 2.0 * train_constrained[indices].float()
            ensemble = policy.predict_core(hidden, actions)
            head_losses = []
            for head_index, prediction in enumerate(ensemble):
                bootstrap = torch.rand(len(indices)) > 0.20
                if not torch.any(bootstrap):
                    bootstrap[head_index % len(indices)] = True
                element = F.smooth_l1_loss(prediction[bootstrap], target[bootstrap], reduction="none").mean(dim=-1)
                head_losses.append((element * weights[bootstrap]).mean())
            loss = torch.stack(head_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.forward_heads.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        policy.eval()
        with torch.no_grad():
            prediction = policy.predict_core(validation_hidden, validation_actions).mean(dim=0)
            per_sample = F.l1_loss(prediction, validation_target, reduction="none").mean(dim=-1)
            validation_loss = float((per_sample * (1.0 + 2.0 * validation_constrained.float())).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_weighted_mae": validation_loss})
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(policy.forward_heads.state_dict())
        if epoch - min(range(len(history)), key=lambda i: history[i]["validation_weighted_mae"]) >= 15:
            break

    policy.forward_heads.load_state_dict(best_state)
    policy.eval()
    return history


def save_checkpoint(policy, source_payload, recording, results):
    payload = dict(source_payload)
    payload["state_dict"] = policy.state_dict()
    payload["config"] = dict(source_payload["config"])
    payload["config"].update({"unity_mpc_calibration": "terrain_transitions_v1"})
    payload["unity_mpc_calibration"] = {"recording": str(recording), "metrics": results}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", type=Path, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/unity_posttrained/best.pt")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    recording = args.recording or latest_recording()
    rows = load_rows(recording)
    policy, source_payload = load_checkpoint(args.checkpoint)
    features = build_features(policy, rows)
    train, validation, test = split_features(features, len(rows))
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError("Not enough chronological transitions in every split")

    baseline = {"validation": metrics(policy, validation), "test": metrics(policy, test)}
    history = calibrate(policy, train, validation, epochs=args.epochs)
    calibrated = {"validation": metrics(policy, validation), "test": metrics(policy, test)}
    improvement = 1.0 - calibrated["test"]["mae"] / baseline["test"]["mae"]
    passed = improvement >= 0.05 and calibrated["test"]["constrained_mae"] < baseline["test"]["constrained_mae"]
    results = {
        "recording": str(recording),
        "frames": len(rows),
        "transitions": len(features),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "baseline": baseline,
        "calibrated": calibrated,
        "test_mae_improvement_fraction": improvement,
        "criteria": {
            "test_mae_improves_at_least_5_percent": improvement >= 0.05,
            "constrained_test_mae_improves": calibrated["test"]["constrained_mae"] < baseline["test"]["constrained_mae"],
            "all_passed": passed,
        },
        "history": history,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if passed:
        save_checkpoint(policy, source_payload, recording, results)
        print(f"exported {CHECKPOINT_PATH}")
    else:
        print("calibration rejected; no checkpoint exported")
    print(json.dumps({key: results[key] for key in ("split", "baseline", "calibrated", "test_mae_improvement_fraction", "criteria")}, indent=2))


if __name__ == "__main__":
    main()
