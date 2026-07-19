#!/usr/bin/env python3
"""Doyle-inspired experience recording and associative reproduction.

Experience packets bind observation, action, and externally derived outcome
valence. At evaluation time a k-nearest-neighbor reproducer retrieves similar
packets and uses their bound outcomes to select an action. Shuffling valence
across otherwise unchanged packets tests whether the original binding matters.

This is case-based control inspired by ERR. It does not establish phenomenal
consciousness, felt emotion, or Doyle's broader information metaphysics.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from continuous_reality_engine_lab import (
    PredictorTelemetry,
    prospective_action,
    sensor_action,
    trigger_imagination,
)
from embodied_world_model_lab import (
    MAX_STEPS,
    collect_training_data,
    make_world,
    train_model_ensemble,
)
from tiny_lab import OUT, set_seed


CONDITIONS = (
    "reactive_controller",
    "triggered_mpc",
    "bound_valence_playback",
    "action_only_playback",
    "shuffled_valence_playback",
)


@dataclass
class RetrievalTelemetry:
    queries: int = 0
    distance_evaluations: int = 0


class EpisodicMemory:
    def __init__(self, observations, actions, valences, k=48):
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.int64)
        self.valences = np.asarray(valences, dtype=np.float32)
        self.k = min(int(k), len(self.observations))
        # Goal and collision channels matter, but rays dominate local matching.
        self.feature_weights = np.asarray(
            [0.5, 0.5, 1.1, 1.1, 1.2] + [1.4] * 8 + [1.3],
            dtype=np.float32,
        )

    @classmethod
    def from_transitions(cls, rows, k=48, shuffle_valence_seed=None):
        observations = np.stack([row.obs for row in rows])
        actions = np.asarray([row.action for row in rows], dtype=np.int64)
        valences = []
        for row in rows:
            progress = float(row.obs[4] - row.next_obs[4])
            terminal_bonus = float(row.next_obs[4] < 0.045)
            valences.append(9.0 * progress + 1.6 * terminal_bonus - 2.5 * row.collision)
        valences = np.asarray(valences, dtype=np.float32)
        if shuffle_valence_seed is not None:
            rng = np.random.default_rng(shuffle_valence_seed)
            valences = valences[rng.permutation(len(valences))]
        return cls(observations, actions, valences, k=k)

    def neighbors(self, observation, telemetry):
        delta = (self.observations - observation) * self.feature_weights
        distances = np.mean(delta * delta, axis=1)
        indices = np.argpartition(distances, self.k - 1)[: self.k]
        weights = np.exp(-distances[indices] / 0.055) + 1e-6
        telemetry.queries += 1
        telemetry.distance_evaluations += len(self.observations)
        return indices, weights

    def action_scores(self, observation, telemetry, use_valence=True):
        indices, weights = self.neighbors(observation, telemetry)
        scores = np.full(8, -float("inf"), dtype=np.float32)
        for action in range(8):
            mask = self.actions[indices] == action
            if not np.any(mask) or observation[5 + action] <= 0.01:
                continue
            action_weights = weights[mask]
            if use_valence:
                values = self.valences[indices][mask]
                scores[action] = float(np.sum(action_weights * values) / np.sum(action_weights))
            else:
                # Reproduction without the bound outcome can recover only how
                # frequently an action occurred near this cue.
                scores[action] = float(np.sum(action_weights))
        return scores

    def choose_action_with_confidence(self, observation, rng, telemetry, use_valence=True):
        scores = self.action_scores(observation, telemetry, use_valence=use_valence)
        scores += rng.normal(0.0, 1e-5, len(scores))
        if not np.any(np.isfinite(scores)):
            return sensor_action(observation, rng), -float("inf"), 0.0
        finite = np.sort(scores[np.isfinite(scores)])
        best = float(finite[-1])
        margin = best - float(finite[-2]) if len(finite) > 1 else float("inf")
        return int(np.argmax(scores)), best, margin

    def choose_action(self, observation, rng, telemetry, use_valence=True):
        action, _, _ = self.choose_action_with_confidence(
            observation, rng, telemetry, use_valence=use_valence
        )
        return action


def run_episode(models, memory, condition, kind, seed, max_steps=MAX_STEPS):
    env = make_world(kind, seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 83000)
    retrieval = RetrievalTelemetry()
    prediction = PredictorTelemetry()
    collisions = 0
    decision_seconds = 0.0

    for step in range(max_steps):
        started = time.perf_counter()
        if condition == "reactive_controller":
            action = sensor_action(obs, rng)
        elif condition == "triggered_mpc":
            action = (
                prospective_action(models, obs, rng, prediction)
                if trigger_imagination(obs)
                else sensor_action(obs, rng)
            )
        elif condition == "bound_valence_playback":
            action = memory.choose_action(obs, rng, retrieval, use_valence=True)
        elif condition == "action_only_playback":
            action = memory.choose_action(obs, rng, retrieval, use_valence=False)
        elif condition == "shuffled_valence_playback":
            action = memory.choose_action(obs, rng, retrieval, use_valence=True)
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
        "memory_queries": retrieval.queries,
        "distance_evaluations": retrieval.distance_evaluations,
        "model_calls": prediction.model_calls,
    }


def evaluate(models, bound_memory, shuffled_memory, seeds=12, max_steps=MAX_STEPS):
    episodes = []
    geometries = ("circles", "rectangles", "diagonal_bars", "u_detour")
    for geometry_index, kind in enumerate(geometries):
        run_count = max(4, seeds // 2) if kind == "u_detour" else seeds
        for condition in CONDITIONS:
            memory = shuffled_memory if condition == "shuffled_valence_playback" else bound_memory
            for run in range(run_count):
                episodes.append(
                    run_episode(
                        models,
                        memory,
                        condition,
                        kind,
                        53000 + geometry_index * 1000 + run,
                        max_steps,
                    )
                )

    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in episodes if row["condition"] == condition]
        summary[condition] = {
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
            "mean_path_efficiency": float(np.mean([row["path_efficiency"] for row in rows])),
            "mean_decision_milliseconds_per_step": float(
                np.mean([row["decision_milliseconds_per_step"] for row in rows])
            ),
            "mean_distance_evaluations": float(
                np.mean([row["distance_evaluations"] for row in rows])
            ),
            "mean_model_calls": float(np.mean([row["model_calls"] for row in rows])),
        }
    return summary, episodes


def paired_contrast(episodes, treatment, reference):
    index = {
        (row["geometry"], row["seed"], row["condition"]): row
        for row in episodes
    }
    keys = sorted({(row["geometry"], row["seed"]) for row in episodes})
    wins = losses = ties = 0
    for geometry, seed in keys:
        treated = index[(geometry, seed, treatment)]["success"]
        baseline = index[(geometry, seed, reference)]["success"]
        wins += int(treated > baseline)
        losses += int(treated < baseline)
        ties += int(treated == baseline)
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {"wins": wins, "losses": losses, "ties": ties, "exact_mcnemar_p": p_value}


def plot_summary(summary, path):
    labels = ["reactive", "triggered\nMPC", "ERR bound", "ERR action\nonly", "ERR shuffled\nvalence"]
    success = [summary[name]["success_rate"] for name in CONDITIONS]
    efficiency = [summary[name]["mean_path_efficiency"] for name in CONDITIONS]
    latency = [summary[name]["mean_decision_milliseconds_per_step"] for name in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(labels, success, color="#2a9d8f")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Task success")
    axes[1].bar(labels, efficiency, color="#e9c46a")
    axes[1].set_title("Mean path efficiency")
    axes[2].bar(labels, latency, color="#457b9d")
    axes[2].set_title("Measured decision time")
    axes[2].set_ylabel("milliseconds / step")
    for axis in axes[:2]:
        axis.set_ylabel("rate / count")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Experience playback and the valence-binding lesion")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    set_seed(131)
    rows = collect_training_data(
        worlds_per_kind=6 if args.quick else 24,
        steps_per_world=70 if args.quick else 140,
    )
    models, histories = train_model_ensemble(
        rows,
        members=2 if args.quick else 3,
        epochs=10 if args.quick else 42,
    )
    bound_memory = EpisodicMemory.from_transitions(rows, k=32 if args.quick else 48)
    shuffled_memory = EpisodicMemory.from_transitions(
        rows, k=32 if args.quick else 48, shuffle_valence_seed=991
    )
    summary, episodes = evaluate(
        models,
        bound_memory,
        shuffled_memory,
        seeds=args.seeds or (3 if args.quick else 12),
        max_steps=55 if args.quick else MAX_STEPS,
    )
    contrasts = {
        "bound_vs_action_only": paired_contrast(
            episodes, "bound_valence_playback", "action_only_playback"
        ),
        "bound_vs_shuffled": paired_contrast(
            episodes, "bound_valence_playback", "shuffled_valence_playback"
        ),
        "bound_vs_reactive": paired_contrast(
            episodes, "bound_valence_playback", "reactive_controller"
        ),
        "bound_vs_triggered_mpc": paired_contrast(
            episodes, "bound_valence_playback", "triggered_mpc"
        ),
    }

    metrics_path = OUT / "episodic_playback_metrics.json"
    figure_path = OUT / "episodic_playback_summary.png"
    metrics_path.write_text(
        json.dumps(
            {
                "question": "Does similarity-triggered playback use bound outcome valence?",
                "summary": summary,
                "paired_contrasts": contrasts,
                "episodes": episodes,
                "memory_packets": len(rows),
                "training_final_losses": [history[-1] for history in histories],
                "claim_boundary": (
                    "This tests case-based action selection and bound external outcome values. "
                    "It does not demonstrate felt emotion, consciousness, or that biological "
                    "memory uses k-nearest-neighbor search."
                ),
            },
            indent=2,
        )
    )
    plot_summary(summary, figure_path)

    print("\nExperience Recorder and Reproducer analogue")
    print("condition                    success  collisions  ms/step  model calls")
    for condition in CONDITIONS:
        row = summary[condition]
        print(
            f"{condition:28s} {row['success_rate']:7.3f}  "
            f"{row['mean_collisions']:10.2f}  "
            f"{row['mean_decision_milliseconds_per_step']:7.3f}  "
            f"{row['mean_model_calls']:11.1f}"
        )
    print("\nPaired success contrasts")
    for name, result in contrasts.items():
        print(
            f"{name:24s} {result['wins']:2d}-{result['losses']:2d}  "
            f"p={result['exact_mcnemar_p']:.4g}"
        )
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
