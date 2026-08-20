#!/usr/bin/env python3
"""Test whether passive ART dynamics add prospective timeout information.

The lab aggregates only an early, food-hidden window from each completed L-wall
episode. It compares raw controller telemetry, ART dynamics alone, and their
combination under leave-one-controller-seed-out validation. This is a
reliability prediction test, not an ART motor controller.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_RUNS = Path("outputs/unity_shadow")
DEFAULT_METRICS = Path("outputs/art_reliability_adapter_metrics.json")
DEFAULT_CHECKPOINT = Path("checkpoints/art_reliability_adapter/best.pt")
RUN_PATTERNS = (
    "lwall_adapter_multiseed_20260721/seed_*.jsonl",
    "lwall_adapter_v2_multiseed_20260721/seed_*.jsonl",
    "lwall_adapter_v2_commit16_20260721/seed_*.jsonl",
    "lwall_adapter_v2_softbias_restart_20260721/seed_*.jsonl",
)
WINDOW_START = 10
WINDOW_FRAMES = 100


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_episodes(root):
    episodes = []
    for pattern in RUN_PATTERNS:
        condition = pattern.split("/", 1)[0]
        for path in sorted(root.glob(pattern)):
            grouped = defaultdict(list)
            for line in path.open(encoding="utf-8"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("trap_course", "")) == "lwall":
                    grouped[int(row.get("trap_episode", 0) or 0)].append(row)
            for episode, rows in grouped.items():
                outcomes = {str(row.get("trap_outcome", "")) for row in rows}
                if "timeout" in outcomes:
                    outcome = "timeout"
                elif "success" in outcomes:
                    outcome = "success"
                else:
                    continue
                seed = int(rows[0].get("controller_seed", 0) or 0)
                eligible = [
                    row
                    for row in rows
                    if str(row.get("trap_outcome", "")) == "running"
                    and not bool(row.get("food_visible", False))
                ]
                window = eligible[WINDOW_START : WINDOW_START + WINDOW_FRAMES]
                if len(window) < 40:
                    continue
                episodes.append(
                    {
                        "id": f"{condition}:{path.stem}:{episode}",
                        "condition": condition,
                        "seed": seed,
                        "outcome": outcome,
                        "rows": window,
                    }
                )
    return episodes


def mean_std(values):
    array = np.asarray(values, dtype=np.float32)
    return [float(array.mean()), float(array.std())]


def feature_vector(rows):
    ray_means = [np.mean([safe_float(row.get("rays", [0] * 8)[i]) for row in rows]) for i in range(8)]
    clearance_means = [
        np.mean([safe_float(row.get("body_clearance", [0] * 8)[i]) for row in rows])
        for i in range(8)
    ]
    raw = list(map(float, ray_means + clearance_means))
    for field in ("shadow_confidence", "shadow_entropy", "orbit_efficiency"):
        raw.extend(mean_std([safe_float(row.get(field)) for row in rows]))
    raw.extend(
        [
            float(np.mean([bool(row.get("blocked", False)) for row in rows])),
            float(np.mean([bool(row.get("body_collision", False)) for row in rows])),
            float(np.mean([bool(row.get("shadow_mpc_engaged", False)) for row in rows])),
        ]
    )

    matches = [safe_float(row.get("art_match")) for row in rows]
    categories = [str(row.get("art_category", "none")) for row in rows]
    switches = sum(a != b for a, b in zip(categories, categories[1:]))
    mismatch = [int(row.get("art_mismatch_resets_total", 0) or 0) for row in rows]
    counts = [int(row.get("art_category_count", 0) or 0) for row in rows]
    art = mean_std(matches)
    art.extend(
        [
            float(np.min(matches)),
            float(np.mean([bool(row.get("art_resonance", False)) for row in rows])),
            float(np.mean([bool(row.get("art_novel", False)) for row in rows])),
            float(np.mean([bool(row.get("art_unknown", False)) for row in rows])),
            switches / max(len(rows) - 1, 1),
            (mismatch[-1] - mismatch[0]) / max(len(rows) - 1, 1),
            float(max(counts) - min(counts)),
        ]
    )
    return np.asarray(raw, dtype=np.float32), np.asarray(art, dtype=np.float32)


def feature_names():
    raw = [f"ray_{i}_mean" for i in range(8)] + [f"clearance_{i}_mean" for i in range(8)]
    raw += [
        "shadow_confidence_mean", "shadow_confidence_std",
        "shadow_entropy_mean", "shadow_entropy_std",
        "orbit_efficiency_mean", "orbit_efficiency_std",
        "blocked_rate", "collision_rate", "mpc_rate",
    ]
    art = [
        "art_match_mean", "art_match_std", "art_match_min", "art_resonance_rate",
        "art_novelty_rate", "art_unknown_rate", "art_switch_rate",
        "art_mismatch_reset_rate", "art_category_growth",
    ]
    return raw, art


def fit_logistic(x, y, epochs, seed):
    torch.manual_seed(seed)
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-5)
    normalized = (x - mean) / std
    model = torch.nn.Linear(x.shape[1], 1)
    positives = y.sum().clamp_min(1.0)
    negatives = (1.0 - y).sum().clamp_min(1.0)
    pos_weight = negatives / positives
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.04)
    for _ in range(epochs):
        logits = model(normalized).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = [(p > n) + 0.5 * (p == n) for p in positive for n in negative]
    return float(np.mean(comparisons))


def cross_validate(x, y, seeds, epochs, seed_offset):
    predictions = np.zeros(len(y), dtype=np.float32)
    folds = []
    for fold, heldout in enumerate(sorted(set(seeds.tolist()))):
        train = torch.tensor(seeds != heldout)
        test = torch.tensor(seeds == heldout)
        model, mean, std = fit_logistic(x[train], y[train], epochs, seed_offset + fold)
        with torch.no_grad():
            score = torch.sigmoid(model((x[test] - mean) / std).squeeze(-1))
        predictions[np.where(test.numpy())[0]] = score.numpy()
        folds.append({
            "heldout_seed": int(heldout),
            "episodes": int(test.sum()),
            "timeouts": int(y[test].sum()),
        })
    return predictions, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    episodes = load_episodes(args.runs)
    raw, art = zip(*(feature_vector(episode["rows"]) for episode in episodes))
    raw = torch.tensor(np.stack(raw), dtype=torch.float32)
    art = torch.tensor(np.stack(art), dtype=torch.float32)
    combined = torch.cat([raw, art], dim=1)
    labels = torch.tensor([episode["outcome"] == "timeout" for episode in episodes], dtype=torch.float32)
    seeds = np.asarray([episode["seed"] for episode in episodes], dtype=np.int64)

    conditions = {
        "raw": raw,
        "art_only": art,
        "raw_plus_art": combined,
    }
    results = {}
    for index, (name, features) in enumerate(conditions.items()):
        predictions, folds = cross_validate(features, labels, seeds, args.epochs, 4200 + index * 100)
        results[name] = {
            "heldout_seed_auc": auc(labels.numpy(), predictions),
            "folds": folds,
        }

    raw_auc = results["raw"]["heldout_seed_auc"]
    combined_auc = results["raw_plus_art"]["heldout_seed_auc"]
    criteria = {
        "complete_seed_holdout": len(set(seeds.tolist())) == 5,
        "combined_auc_at_least_0_70": combined_auc >= 0.70,
        "art_adds_at_least_0_03_auc": combined_auc >= raw_auc + 0.03,
        "prospective_window_excludes_outcome": True,
    }
    accepted = all(criteria.values())
    raw_names, art_names = feature_names()
    metrics = {
        "protocol": {
            "run_patterns": list(RUN_PATTERNS),
            "window_start": WINDOW_START,
            "window_frames": WINDOW_FRAMES,
            "validation": "leave_one_controller_seed_out",
            "target": "episode_timeout",
        },
        "episodes": len(episodes),
        "timeouts": int(labels.sum()),
        "successes": int(len(labels) - labels.sum()),
        "results": results,
        "criteria": criteria,
        "accepted": accepted,
        "claim_boundary": (
            "This tests prospective timeout prediction. ART feature utility is correlational; "
            "it does not establish that ART dynamics cause recovery or consciousness."
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if accepted:
        model, mean, std = fit_logistic(combined, labels, args.epochs, 4999)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "feature_names": raw_names + art_names,
                "metrics": metrics,
                "activation_scope": "prospective_reliability_observer_only",
            },
            args.checkpoint,
        )
        print(f"accepted and exported {args.checkpoint}")
    else:
        print("adapter rejected; checkpoint not exported")
    print(json.dumps({
        "episodes": len(episodes),
        "timeouts": int(labels.sum()),
        "raw_auc": raw_auc,
        "art_only_auc": results["art_only"]["heldout_seed_auc"],
        "raw_plus_art_auc": combined_auc,
        "accepted": accepted,
        "metrics": str(args.metrics),
    }, indent=2))


if __name__ == "__main__":
    main()
