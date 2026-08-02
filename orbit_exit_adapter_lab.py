#!/usr/bin/env python3
"""Train a gated orbit-exit readout without modifying the base GRU policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orbit_exit_posttraining_lab import accuracy, sequence_logits, split_events
from upgraded_foraging_pipeline import ACTION_NAMES, load_checkpoint


DATASET = Path("outputs/unity_shadow/orbit_exit_20260718.npz")
BASE = Path("checkpoints/starvation_posttrained/best.pt")
OUTPUT = Path("outputs/orbit_exit_adapter_metrics.json")
CHECKPOINT = Path("checkpoints/orbit_exit_adapter/best.pt")


def hidden_features(policy, sequences, batch_size=128):
    features = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[start : start + batch_size])
            hidden = policy.initial_state(len(batch))
            for step in range(batch.shape[1]):
                _logits, _value, hidden = policy.step(batch[:, step], hidden)
            features.append(hidden)
    return torch.cat(features)


def adapter_accuracy(adapter, features, labels):
    with torch.no_grad():
        predictions = torch.argmax(adapter(features), dim=-1).cpu().numpy()
    return float(np.mean(predictions == labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args()

    data = np.load(args.dataset)
    sequences = data["sequences"]
    labels = data["labels"]
    train_mask, test_mask, heldout_events = split_events(data["event_ids"])
    policy, _payload = load_checkpoint(args.base)
    features = hidden_features(policy, sequences)
    train_features = features[torch.tensor(train_mask)]
    test_features = features[torch.tensor(test_mask)]
    train_labels = torch.tensor(labels[train_mask])
    test_labels = labels[test_mask]

    torch.manual_seed(1911)
    adapter = torch.nn.Sequential(
        torch.nn.LayerNorm(policy.hidden_dim),
        torch.nn.Linear(policy.hidden_dim, len(ACTION_NAMES)),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4, weight_decay=0.02)
    generator = torch.Generator().manual_seed(1911)
    losses = []
    for _epoch in range(args.epochs):
        order = torch.randperm(len(train_features), generator=generator)
        epoch_losses = []
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            logits = adapter(train_features[indices])
            loss = F.cross_entropy(logits, train_labels[indices], label_smoothing=0.04)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.8)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    baseline_accuracy = accuracy(policy, sequences[test_mask], labels[test_mask])
    learned_accuracy = adapter_accuracy(adapter, test_features, test_labels)
    fallback_mask = data["fallback_assisted"][test_mask].astype(bool)
    self_mask = ~fallback_mask
    breakdown = {
        "self_recovered_accuracy": adapter_accuracy(adapter, test_features[self_mask], test_labels[self_mask])
        if self_mask.any()
        else None,
        "fallback_assisted_accuracy": adapter_accuracy(adapter, test_features[fallback_mask], test_labels[fallback_mask])
        if fallback_mask.any()
        else None,
    }
    criteria = {
        "event_heldout_accuracy_improves_5_points": learned_accuracy >= baseline_accuracy + 0.05,
        "heldout_accuracy_at_least_75_percent": learned_accuracy >= 0.75,
        "base_policy_frozen": True,
        "runtime_activation_not_yet_enabled": True,
    }
    criteria["adapter_exported"] = criteria["event_heldout_accuracy_improves_5_points"] and criteria[
        "heldout_accuracy_at_least_75_percent"
    ]
    metrics = {
        "protocol": {
            "dataset": str(args.dataset),
            "base_checkpoint": str(args.base),
            "whole_event_holdout": True,
            "heldout_event_ids": heldout_events,
            "base_policy_parameters_updated": 0,
            "activation_scope": "orbit_monitor_only",
        },
        "samples": {"train": int(train_mask.sum()), "heldout": int(test_mask.sum())},
        "baseline_policy_accuracy": baseline_accuracy,
        "adapter_accuracy": learned_accuracy,
        "heldout_breakdown": breakdown,
        "criteria": criteria,
        "training_loss": losses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if criteria["adapter_exported"]:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": adapter.state_dict(),
                "hidden_dim": policy.hidden_dim,
                "action_names": ACTION_NAMES,
                "base_checkpoint": str(args.base),
                "metrics": metrics,
            },
            args.checkpoint,
        )
        print(f"exported {args.checkpoint}")
    else:
        print("adapter rejected")
    print(json.dumps({"baseline": baseline_accuracy, "adapter": learned_accuracy, "criteria": criteria}, indent=2))


if __name__ == "__main__":
    main()
