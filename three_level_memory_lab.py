#!/usr/bin/env python3
"""Arbitrate reflex, episodic playback, and predictive simulation by need."""

import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from continuous_reality_engine_lab import (
    PredictorTelemetry,
    prospective_action,
    sensor_action,
    trigger_imagination,
)
from embodied_world_model_lab import MAX_STEPS, collect_training_data, make_world, train_model_ensemble
from episodic_playback_lab import EpisodicMemory, RetrievalTelemetry, paired_contrast
from tiny_lab import OUT, set_seed


CONDITIONS = (
    "reactive_controller",
    "triggered_mpc",
    "three_level_bound",
    "three_level_shuffled",
)


def run_episode(models, memory, condition, kind, seed, max_steps=MAX_STEPS):
    env = make_world(kind, seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 94000)
    prediction = PredictorTelemetry()
    retrieval = RetrievalTelemetry()
    tier_counts = {"reflex": 0, "playback": 0, "mpc": 0}
    collisions = 0
    decision_seconds = 0.0

    for step in range(max_steps):
        started = time.perf_counter()
        if condition == "reactive_controller":
            action = sensor_action(obs, rng)
            tier_counts["reflex"] += 1
        elif condition == "triggered_mpc":
            if trigger_imagination(obs):
                action = prospective_action(models, obs, rng, prediction)
                tier_counts["mpc"] += 1
            else:
                action = sensor_action(obs, rng)
                tier_counts["reflex"] += 1
        elif condition in {"three_level_bound", "three_level_shuffled"}:
            if not trigger_imagination(obs):
                action = sensor_action(obs, rng)
                tier_counts["reflex"] += 1
            else:
                action, value, margin = memory.choose_action_with_confidence(
                    obs, rng, retrieval, use_valence=True
                )
                if value > 0.035 and margin > 0.018:
                    tier_counts["playback"] += 1
                else:
                    action = prospective_action(models, obs, rng, prediction)
                    tier_counts["mpc"] += 1
        else:
            raise ValueError(condition)
        decision_seconds += time.perf_counter() - started

        obs, collision, done = env.step(action)
        collisions += int(collision)
        if done:
            break

    success = env.pos == env.goal
    steps = step + 1
    return {
        "condition": condition,
        "geometry": kind,
        "seed": seed,
        "success": float(success),
        "steps": steps,
        "collisions": collisions,
        "path_efficiency": math.dist(env.start, env.goal) / steps if success else 0.0,
        "decision_milliseconds_per_step": 1000.0 * decision_seconds / steps,
        "model_calls": prediction.model_calls,
        "memory_queries": retrieval.queries,
        "reflex_fraction": tier_counts["reflex"] / steps,
        "playback_fraction": tier_counts["playback"] / steps,
        "mpc_fraction": tier_counts["mpc"] / steps,
    }


def evaluate(models, bound_memory, shuffled_memory, seeds=12, max_steps=MAX_STEPS):
    episodes = []
    for geometry_index, kind in enumerate(("circles", "rectangles", "diagonal_bars", "u_detour")):
        run_count = max(4, seeds // 2) if kind == "u_detour" else seeds
        for condition in CONDITIONS:
            memory = shuffled_memory if condition == "three_level_shuffled" else bound_memory
            for run in range(run_count):
                episodes.append(
                    run_episode(
                        models, memory, condition, kind,
                        64000 + geometry_index * 1000 + run, max_steps,
                    )
                )
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in episodes if row["condition"] == condition]
        summary[condition] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "success", "steps", "collisions", "path_efficiency",
                "decision_milliseconds_per_step", "model_calls", "memory_queries",
                "reflex_fraction", "playback_fraction", "mpc_fraction",
            )
        }
    return summary, episodes


def plot_summary(summary, path):
    labels = ["reactive", "triggered\nMPC", "three-level\nbound", "three-level\nshuffled"]
    success = [summary[name]["success"] for name in CONDITIONS]
    calls = [summary[name]["model_calls"] for name in CONDITIONS]
    reflex = [summary[name]["reflex_fraction"] for name in CONDITIONS]
    playback = [summary[name]["playback_fraction"] for name in CONDITIONS]
    mpc = [summary[name]["mpc_fraction"] for name in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(labels, success, color="#2a9d8f")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Task success")
    axes[1].bar(labels, calls, color="#457b9d")
    axes[1].set_title("Forward-model calls")
    axes[2].bar(labels, reflex, label="reflex", color="#2a9d8f")
    axes[2].bar(labels, playback, bottom=reflex, label="playback", color="#e9c46a")
    lower = np.asarray(reflex) + np.asarray(playback)
    axes[2].bar(labels, mpc, bottom=lower, label="MPC", color="#e76f51")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Control allocation")
    axes[2].legend()
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Conditional reflex, experience playback, and prospective simulation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    set_seed(149)
    rows = collect_training_data(
        worlds_per_kind=6 if args.quick else 24,
        steps_per_world=70 if args.quick else 140,
    )
    models, histories = train_model_ensemble(
        rows, members=2 if args.quick else 3, epochs=10 if args.quick else 42
    )
    bound = EpisodicMemory.from_transitions(rows, k=32 if args.quick else 48)
    shuffled = EpisodicMemory.from_transitions(
        rows, k=32 if args.quick else 48, shuffle_valence_seed=991
    )
    summary, episodes = evaluate(
        models, bound, shuffled,
        seeds=args.seeds or (3 if args.quick else 12),
        max_steps=55 if args.quick else MAX_STEPS,
    )
    contrasts = {
        "bound_vs_triggered": paired_contrast(episodes, "three_level_bound", "triggered_mpc"),
        "bound_vs_reactive": paired_contrast(episodes, "three_level_bound", "reactive_controller"),
        "bound_vs_shuffled": paired_contrast(episodes, "three_level_bound", "three_level_shuffled"),
    }
    metrics_path = OUT / "three_level_memory_metrics.json"
    figure_path = OUT / "three_level_memory_summary.png"
    metrics_path.write_text(json.dumps({
        "summary": summary,
        "paired_contrasts": contrasts,
        "episodes": episodes,
        "memory_packets": len(rows),
        "training_final_losses": [history[-1] for history in histories],
        "claim_boundary": "The arbitration thresholds are designed and the modules are separately trained.",
    }, indent=2))
    plot_summary(summary, figure_path)

    print("\nThree-level memory architecture")
    print("condition                 success  model calls  reflex  playback  mpc")
    for condition in CONDITIONS:
        row = summary[condition]
        print(
            f"{condition:25s} {row['success']:.3f}  {row['model_calls']:11.1f}  "
            f"{row['reflex_fraction']:.3f}   {row['playback_fraction']:.3f}    "
            f"{row['mpc_fraction']:.3f}"
        )
    print("\nPaired success contrasts")
    for name, result in contrasts.items():
        print(f"{name:22s} {result['wins']}-{result['losses']} p={result['exact_mcnemar_p']:.4g}")
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
