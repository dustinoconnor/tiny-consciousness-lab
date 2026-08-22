#!/usr/bin/env python3
"""Train a frozen-GRU readout from successful Unity L-wall trajectories.

The adapter is deliberately narrow: it learns only the direction of efficient
future motion while the L-wall goal is still hidden. Entire recording groups
(normally controller seeds) are held out during validation, and the base
recurrent policy is never updated.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from orbit_exit_posttraining_lab import nearest_action, sequence_observations
from upgraded_foraging_pipeline import ACTION_NAMES, load_checkpoint


DEFAULT_RECORDING = Path("outputs/unity_shadow/lwall_adapter_collection_20260721.jsonl")
DEFAULT_BASE = Path("checkpoints/unity_geometry_posttrained/best.pt")
DEFAULT_CHECKPOINT = Path("checkpoints/lwall_hidden_goal_adapter/best.pt")
DEFAULT_METRICS = Path("outputs/lwall_hidden_goal_adapter_metrics.json")
SEQUENCE_STEPS = 24
FUTURE_STEPS = 12
STRIDE = 3


def position(row):
    values = row.get("position") or [0.0, 0.0, 0.0]
    return np.asarray([float(values[0] or 0.0), float(values[2] or 0.0)], dtype=np.float32)


def grouped_episodes(recording):
    episodes = defaultdict(list)
    with recording.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("trap_course", "")) != "lwall":
                continue
            episodes[int(row.get("trap_episode", 0) or 0)].append(row)
    return dict(episodes)


def mine(recordings):
    samples = []
    summary = Counter()
    for recording in recordings:
        group = recording.stem
        for episode, rows in sorted(grouped_episodes(recording).items()):
            outcome = "success" if any(row.get("trap_outcome") == "success" for row in rows) else "timeout"
            summary[f"episodes_{outcome}"] += 1
            if outcome != "success":
                continue
            for index in range(SEQUENCE_STEPS - 1, len(rows) - FUTURE_STEPS, STRIDE):
                if bool(rows[index].get("food_visible", False)):
                    continue
                future = rows[index : index + FUTURE_STEPS + 1]
                if any(bool(row.get("food_visible", False)) for row in future):
                    continue
                points = np.stack([position(row) for row in future])
                deltas = np.diff(points, axis=0)
                path = float(np.linalg.norm(deltas, axis=1).sum())
                vector = points[-1] - points[0]
                net = float(np.linalg.norm(vector))
                contact = float(np.mean([bool(row.get("body_collision")) for row in future]))
                if net < 1.5 or net / max(path, 1e-6) < 0.62 or contact > 0.25:
                    continue
                context = rows[index - SEQUENCE_STEPS + 1 : index + 1]
                samples.append(
                    {
                        "episode": f"{group}:{episode}",
                        "group": group,
                        "sequence": sequence_observations(context),
                        "label": nearest_action(vector),
                    }
                )
    summary["samples"] = len(samples)
    summary["successful_episodes_with_samples"] = len({sample["episode"] for sample in samples})
    summary["recording_groups_with_samples"] = len({sample["group"] for sample in samples})
    return samples, dict(summary)


def hidden_features(policy, samples, batch_size=128):
    sequences = np.stack([sample["sequence"] for sample in samples])
    features = []
    baseline_logits = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[start : start + batch_size], dtype=torch.float32)
            hidden = policy.initial_state(len(batch))
            logits = None
            for step in range(batch.shape[1]):
                logits, _value, hidden = policy.step(batch[:, step], hidden)
            features.append(hidden)
            baseline_logits.append(logits)
    return torch.cat(features), torch.cat(baseline_logits)


def make_adapter(hidden_dim):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.Linear(hidden_dim, len(ACTION_NAMES)),
    )


def accuracy(logits, labels):
    return float((torch.argmax(logits, dim=-1) == labels).float().mean())


def train_adapter(features, labels, train_mask, epochs, seed):
    torch.manual_seed(seed)
    adapter = make_adapter(features.shape[1])
    train_features = features[train_mask]
    train_labels = labels[train_mask]
    counts = torch.bincount(train_labels, minlength=len(ACTION_NAMES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = torch.sqrt(counts[present].sum() / counts[present])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4, weight_decay=0.025)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_loss = math.inf
    for _epoch in range(epochs):
        order = torch.randperm(len(train_features), generator=generator)
        losses = []
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            logits = adapter(train_features[indices])
            loss = F.cross_entropy(
                logits,
                train_labels[indices],
                weight=weights,
                label_smoothing=0.03,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.8)
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
    adapter.load_state_dict(best_state)
    return adapter.eval(), best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        nargs="+",
        default=[DEFAULT_RECORDING],
        help="One or more Unity JSONL recordings; each file is a held-out validation group.",
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    samples, extraction = mine(args.recording)
    episodes = sorted({sample["episode"] for sample in samples})
    if len(episodes) < 3:
        raise RuntimeError("Need at least three successful L-wall episodes with usable samples")
    groups = sorted({sample["group"] for sample in samples})
    if len(groups) < 2:
        raise RuntimeError("Need at least two recording groups for held-out-group validation")
    policy, _payload = load_checkpoint(args.base)
    for parameter in policy.parameters():
        parameter.requires_grad = False
    features, baseline_logits = hidden_features(policy, samples)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    group_ids = np.asarray([sample["group"] for sample in samples])

    folds = []
    for fold, heldout in enumerate(groups):
        train_mask = torch.tensor(group_ids != heldout)
        test_mask = torch.tensor(group_ids == heldout)
        adapter, loss = train_adapter(features, labels, train_mask, args.epochs, 3101 + fold)
        with torch.no_grad():
            candidate_logits = adapter(features[test_mask])
        folds.append(
            {
                "heldout_group": heldout,
                "train_samples": int(train_mask.sum()),
                "heldout_samples": int(test_mask.sum()),
                "baseline_accuracy": accuracy(baseline_logits[test_mask], labels[test_mask]),
                "adapter_accuracy": accuracy(candidate_logits, labels[test_mask]),
                "loss": loss,
            }
        )

    baseline_accuracy = float(np.mean([fold["baseline_accuracy"] for fold in folds]))
    adapter_accuracy = float(np.mean([fold["adapter_accuracy"] for fold in folds]))
    criteria = {
        "whole_recording_group_cross_validation": True,
        "adapter_improves_at_least_5_points": adapter_accuracy >= baseline_accuracy + 0.05,
        "adapter_accuracy_at_least_45_percent": adapter_accuracy >= 0.45,
        "base_policy_frozen": True,
        "activation_scope_lwall_food_hidden_only": True,
    }
    accepted = all(criteria.values())
    final_adapter, final_loss = train_adapter(
        features,
        labels,
        torch.ones(len(samples), dtype=torch.bool),
        args.epochs,
        3199,
    )
    metrics = {
        "protocol": {
            "recordings": [str(path) for path in args.recording],
            "base_checkpoint": str(args.base),
            "sequence_steps": SEQUENCE_STEPS,
            "future_steps": FUTURE_STEPS,
            "whole_recording_group_holdout": True,
            "labels": "efficient future displacement from successful trajectories",
        },
        "extraction": extraction,
        "action_counts": dict(Counter(ACTION_NAMES[int(label)] for label in labels)),
        "folds": folds,
        "baseline_accuracy": baseline_accuracy,
        "adapter_accuracy": adapter_accuracy,
        "criteria": criteria,
        "accepted": accepted,
        "final_training_loss": final_loss,
        "claim_boundary": (
            "The adapter is trained on repeated instances of one Unity L-wall across "
            "multiple controller seeds. It is a narrow behavioral correction, not "
            "evidence of topology generalization."
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if accepted:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": final_adapter.state_dict(),
                "hidden_dim": policy.hidden_dim,
                "action_names": ACTION_NAMES,
                "base_checkpoint": str(args.base),
                "activation_scope": "lwall_food_hidden_only",
                "metrics": metrics,
            },
            args.checkpoint,
        )
        print(f"accepted and exported {args.checkpoint}")
    else:
        print("adapter rejected; checkpoint not exported")
    print(json.dumps({
        "extraction": extraction,
        "baseline_accuracy": baseline_accuracy,
        "adapter_accuracy": adapter_accuracy,
        "accepted": accepted,
        "metrics": str(args.metrics),
    }, indent=2))


if __name__ == "__main__":
    main()
