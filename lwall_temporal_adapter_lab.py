#!/usr/bin/env python3
"""Test a temporal L-wall adapter over a frozen recurrent navigation policy.

V2 reads one frozen GRU hidden state with a linear classifier. V3 receives a
short causal history of those same hidden states and may learn when an efficient
detour direction should persist or change. Entire controller-seed recordings
are held out, and no Unity or base-policy weights are modified.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lwall_hidden_goal_adapter_lab import (
    FUTURE_STEPS,
    STRIDE,
    grouped_episodes,
    make_adapter,
    position,
)
from orbit_exit_posttraining_lab import nearest_action, sequence_observations
from upgraded_foraging_pipeline import ACTION_NAMES, load_checkpoint


DEFAULT_RECORDINGS = sorted(
    Path("outputs/unity_shadow/lwall_adapter_v2_multiseed_20260721").glob("seed_*.jsonl")
)
DEFAULT_BASE = Path("checkpoints/unity_geometry_posttrained/best.pt")
DEFAULT_V2 = Path("checkpoints/lwall_hidden_goal_adapter/v2.pt")
DEFAULT_CHECKPOINT = Path("checkpoints/lwall_hidden_goal_adapter/v3_temporal.pt")
DEFAULT_METRICS = Path("outputs/lwall_temporal_adapter_v3_metrics.json")
TEMPORAL_STEPS = 12


class TemporalAdapter(torch.nn.Module):
    """Small causal sequence readout; the upstream navigation GRU stays frozen."""

    def __init__(self, input_dim, temporal_dim=32):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(input_dim)
        self.memory = torch.nn.GRU(input_dim, temporal_dim, batch_first=True)
        self.output_norm = torch.nn.LayerNorm(temporal_dim)
        self.actor = torch.nn.Linear(temporal_dim, len(ACTION_NAMES))
        self.temporal_dim = temporal_dim

    def forward(self, sequence):
        values, _hidden = self.memory(self.input_norm(sequence))
        return self.actor(self.output_norm(values[:, -1]))


def episode_hidden_states(policy, rows):
    observations = sequence_observations(rows)
    hidden = policy.initial_state(1)
    hidden_states = []
    base_logits = []
    with torch.no_grad():
        for observation in observations:
            tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits, _value, hidden = policy.step(tensor, hidden)
            hidden_states.append(hidden.squeeze(0).clone())
            base_logits.append(logits.squeeze(0).clone())
    return torch.stack(hidden_states), torch.stack(base_logits)


def mine(recordings, policy):
    samples = []
    summary = Counter()
    for recording in recordings:
        group = recording.stem
        for episode, rows in sorted(grouped_episodes(recording).items()):
            success = any(row.get("trap_outcome") == "success" for row in rows)
            summary["episodes_success" if success else "episodes_timeout"] += 1
            if not success or len(rows) < TEMPORAL_STEPS + FUTURE_STEPS:
                continue
            hidden_states, base_logits = episode_hidden_states(policy, rows)
            for index in range(TEMPORAL_STEPS - 1, len(rows) - FUTURE_STEPS, STRIDE):
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
                samples.append(
                    {
                        "group": group,
                        "episode": f"{group}:{episode}",
                        "sequence": hidden_states[index - TEMPORAL_STEPS + 1 : index + 1],
                        "base_logits": base_logits[index],
                        "final_hidden": hidden_states[index],
                        "label": nearest_action(vector),
                    }
                )
    summary["samples"] = len(samples)
    summary["successful_episodes_with_samples"] = len({item["episode"] for item in samples})
    summary["recording_groups_with_samples"] = len({item["group"] for item in samples})
    return samples, dict(summary)


def accuracy(logits, labels):
    return float((torch.argmax(logits, dim=-1) == labels).float().mean())


def class_weights(labels):
    counts = torch.bincount(labels, minlength=len(ACTION_NAMES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = torch.sqrt(counts[present].sum() / counts[present])
    return weights


def train_temporal(sequences, labels, train_mask, epochs, seed):
    torch.manual_seed(seed)
    adapter = TemporalAdapter(sequences.shape[-1])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=7e-4, weight_decay=0.03)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.where(train_mask)[0]
    weights = class_weights(labels[train_mask])
    best_state = None
    best_loss = math.inf
    for _epoch in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator)]
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            logits = adapter(sequences[batch])
            loss = F.cross_entropy(logits, labels[batch], weight=weights, label_smoothing=0.03)
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


def load_v2(path, hidden_dim):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adapter = make_adapter(hidden_dim)
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, nargs="+", default=DEFAULT_RECORDINGS)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    if len(args.recording) < 2:
        raise RuntimeError("V3 requires at least two recording groups for seed-held-out validation")
    policy, _payload = load_checkpoint(args.base)
    for parameter in policy.parameters():
        parameter.requires_grad = False
    samples, extraction = mine(args.recording, policy)
    if not samples:
        raise RuntimeError("No efficient food-hidden L-wall samples were found")

    sequences = torch.stack([item["sequence"] for item in samples])
    final_hidden = torch.stack([item["final_hidden"] for item in samples])
    base_logits = torch.stack([item["base_logits"] for item in samples])
    labels = torch.tensor([item["label"] for item in samples], dtype=torch.long)
    group_ids = np.asarray([item["group"] for item in samples])
    groups = sorted(set(group_ids))
    v2 = load_v2(args.v2, policy.hidden_dim)

    folds = []
    for fold, heldout in enumerate(groups):
        train_mask = torch.tensor(group_ids != heldout)
        test_mask = torch.tensor(group_ids == heldout)
        candidate, loss = train_temporal(sequences, labels, train_mask, args.epochs, 4301 + fold)
        with torch.no_grad():
            v3_logits = candidate(sequences[test_mask])
            v2_logits = v2(final_hidden[test_mask])
        folds.append(
            {
                "heldout_group": heldout,
                "train_samples": int(train_mask.sum()),
                "heldout_samples": int(test_mask.sum()),
                "base_policy_accuracy": accuracy(base_logits[test_mask], labels[test_mask]),
                "v2_linear_accuracy": accuracy(v2_logits, labels[test_mask]),
                "v3_temporal_accuracy": accuracy(v3_logits, labels[test_mask]),
                "training_loss": loss,
            }
        )

    v2_accuracy = float(np.mean([fold["v2_linear_accuracy"] for fold in folds]))
    v3_accuracy = float(np.mean([fold["v3_temporal_accuracy"] for fold in folds]))
    minimum_fold = min(fold["v3_temporal_accuracy"] for fold in folds)
    criteria = {
        "whole_seed_recording_holdout": True,
        "base_policy_frozen": True,
        "v3_accuracy_at_least_90_percent": v3_accuracy >= 0.90,
        "v3_beats_v2_by_two_points": v3_accuracy >= v2_accuracy + 0.02,
        "no_v3_fold_below_80_percent": minimum_fold >= 0.80,
    }
    accepted = all(criteria.values())
    final_adapter, final_loss = train_temporal(
        sequences,
        labels,
        torch.ones(len(samples), dtype=torch.bool),
        args.epochs,
        4399,
    )
    metrics = {
        "protocol": {
            "recordings": [str(path) for path in args.recording],
            "base_checkpoint": str(args.base),
            "v2_checkpoint": str(args.v2),
            "temporal_steps": TEMPORAL_STEPS,
            "future_hindsight_steps": FUTURE_STEPS,
            "whole_recording_seed_holdout": True,
            "base_policy_frozen": True,
        },
        "extraction": extraction,
        "action_counts": dict(Counter(ACTION_NAMES[int(label)] for label in labels)),
        "folds": folds,
        "base_policy_accuracy": float(np.mean([fold["base_policy_accuracy"] for fold in folds])),
        "v2_linear_accuracy": v2_accuracy,
        "v3_temporal_accuracy": v3_accuracy,
        "v3_improvement_over_v2": v3_accuracy - v2_accuracy,
        "criteria": criteria,
        "accepted": accepted,
        "final_training_loss": final_loss,
        "claim_boundary": (
            "This tests a narrow temporal readout on repeated instances of one fixed Unity "
            "L-wall. Acceptance would support temporal sequencing beyond a linear hidden-state "
            "readout, not general topology learning or consciousness."
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if accepted:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter_type": "temporal_gru",
                "state_dict": final_adapter.state_dict(),
                "input_dim": policy.hidden_dim,
                "temporal_dim": final_adapter.temporal_dim,
                "temporal_steps": TEMPORAL_STEPS,
                "action_names": ACTION_NAMES,
                "base_checkpoint": str(args.base),
                "activation_scope": "lwall_food_hidden_only",
                "metrics": metrics,
            },
            args.checkpoint,
        )
        print(f"accepted and exported {args.checkpoint}")
    else:
        if args.checkpoint.exists():
            args.checkpoint.unlink()
        print("V3 rejected; checkpoint not exported")
    print(json.dumps({"v2": v2_accuracy, "v3": v3_accuracy, "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()
