#!/usr/bin/env python3
"""Scaling test for progress-shaped valence.

Exact Phi gets expensive quickly, so this script scales hidden size and measures
behavior instead: goal completion, completion speed, reward, and wireheading.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed
from valence_shaping_lab import train_condition


def run_scaling():
    sizes = [8, 16, 32, 64, 96]
    modes = ["goal_only", "small_progress_reward", "large_progress_reward", "direct_positive"]
    results = {}
    for mode in modes:
        results[mode] = {}
        for hidden_dim in sizes:
            data = train_condition(mode, episodes=650, hidden_dim=hidden_dim, eval_runs=64)
            results[mode][str(hidden_dim)] = {
                "eval_goal_rate": data["eval_goal_rate"],
                "eval_hazard_rate": data["eval_hazard_rate"],
                "eval_button_rate": data["eval_button_rate"],
                "eval_mean_steps": data["eval_mean_steps"],
                "eval_mean_reward": data["eval_mean_reward"],
                "task_efficiency": data["eval_goal_rate"] * (1.0 - data["eval_mean_steps"] / 32.0),
            }
    results["note"] = (
        "No Phi is calculated here. This scaling test asks whether progress-shaped valence improves behavior "
        "as hidden size grows: goal completion, speed, reward, and button/wireheading rate."
    )
    return results


def plot_scaling(results, path):
    modes = [m for m in results if m != "note"]
    sizes = [int(s) for s in results[modes[0]].keys()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    metrics = [
        ("eval_goal_rate", "Goal Rate"),
        ("task_efficiency", "Task Efficiency"),
        ("eval_button_rate", "Good-Button Rate"),
        ("eval_mean_reward", "Mean Reward"),
    ]
    colors = {
        "goal_only": "#7a7a7a",
        "small_progress_reward": "#16a3a6",
        "large_progress_reward": "#4b6cff",
        "direct_positive": "#ff8a00",
    }
    for ax, (metric, title) in zip(axes.flat, metrics):
        for mode in modes:
            y = [results[mode][str(size)][metric] for size in sizes]
            ax.plot(sizes, y, marker="o", lw=2, label=mode, color=colors.get(mode))
        ax.set_title(title)
        ax.set_xlabel("hidden units")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("rate")
    axes[0, 1].set_ylabel("goal_rate * speed_bonus")
    axes[1, 0].set_ylabel("rate")
    axes[1, 1].set_ylabel("reward")
    axes[0, 0].legend()
    fig.suptitle("Valence Scaling Without Phi")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(31)
    OUT.mkdir(exist_ok=True)
    results = run_scaling()
    (OUT / "valence_scaling_metrics.json").write_text(json.dumps(results, indent=2))
    plot_scaling(results, OUT / "valence_scaling_behavior.png")
    print("Valence scaling lab complete")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
