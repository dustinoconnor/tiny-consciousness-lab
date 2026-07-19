#!/usr/bin/env python3
"""Compare reactive, queried, continuous, and grounded world-model control.

This lab asks a narrow Llinas-inspired engineering question: does continuously
maintaining a predicted sensorimotor state help an embodied controller, and
what happens when that internally generated state is no longer corrected by
observation?

All predictive conditions use the same frozen forward-model ensemble and the
same short-horizon planner. Only prediction schedule and sensory correction
differ. The experiment therefore tests continuous generation and grounding;
it does not test phenomenal consciousness or electromagnetic field theories.
"""

import argparse
import json
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from embodied_world_model_lab import (
    ACTION_NAMES,
    MAX_STEPS,
    MOVES,
    collect_training_data,
    make_world,
    predict_ensemble,
    score_predicted_state,
    train_model_ensemble,
)
from tiny_lab import OUT, set_seed


CONDITIONS = (
    "reactive_controller",
    "triggered_mpc",
    "continuous_generative",
    "continuous_grounded",
    "ungrounded_generative",
)


@dataclass
class PredictorTelemetry:
    model_calls: int = 0
    planning_steps: int = 0
    corrections: int = 0
    prediction_error_sum: float = 0.0
    prediction_error_samples: int = 0

    @property
    def mean_prediction_error(self):
        if not self.prediction_error_samples:
            return None
        return self.prediction_error_sum / self.prediction_error_samples


def sanitize_observation(obs):
    """Keep recursively predicted sensor packets inside their valid ranges."""
    clean = np.asarray(obs, dtype=np.float32).copy()
    clean[0:2] = np.clip(clean[0:2], 0.0, 1.0)
    clean[2:4] = np.clip(clean[2:4], -1.0, 1.0)
    clean[4] = np.clip(clean[4], 0.0, 1.0)
    clean[5:13] = np.clip(clean[5:13], 0.0, 1.0)
    clean[-1] = np.clip(clean[-1], 0.0, 1.0)
    return clean


def correct_latent(predicted, observed, gain):
    """Prediction-error correction analogous to a simple observer update."""
    gain = float(np.clip(gain, 0.0, 1.0))
    return sanitize_observation((1.0 - gain) * predicted + gain * observed)


def sensor_scores(obs):
    """Memoryless goal attraction plus current-ray collision avoidance."""
    goal = obs[2:4]
    norm = float(np.linalg.norm(goal))
    directions = MOVES.astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    alignment = np.zeros(len(MOVES), dtype=np.float32)
    if norm > 1e-8:
        alignment = directions @ (goal / norm)
    clearance = obs[5:13]
    scores = 1.25 * alignment + 0.72 * clearance
    scores -= 2.4 * (clearance <= 0.01)
    return scores


def sensor_action(obs, rng):
    scores = sensor_scores(obs)
    scores += rng.normal(0.0, 0.008, len(MOVES))
    return int(np.argmax(scores))


def predict_counted(models, obs, action, telemetry):
    telemetry.model_calls += len(models)
    predicted, collision, uncertainty = predict_ensemble(models, obs, action)
    return sanitize_observation(predicted), collision, uncertainty


def prospective_action(models, latent, rng, telemetry, horizon=4):
    """Evaluate all eight roots, then greedily extend each imagined future."""
    telemetry.planning_steps += 1
    best_action = 0
    best_value = -float("inf")
    for root_action in range(len(MOVES)):
        # Prediction cannot vote a currently blocked action back into existence.
        # This keeps the comparison about temporal modeling rather than whether
        # learned imagination is allowed to contradict a ground-truth ray.
        if latent[5 + root_action] <= 0.01:
            continue
        imagined = latent.copy()
        action = root_action
        value = 2.0 * float(sensor_scores(latent)[root_action])
        discount = 1.0
        for _ in range(horizon):
            previous = imagined
            imagined, collision, uncertainty = predict_counted(
                models, imagined, action, telemetry
            )
            value += discount * (
                score_predicted_state(previous, imagined, collision)
                - 1.6 * uncertainty
            )
            discount *= 0.76
            action = sensor_action(imagined, rng)
        value += float(rng.normal(0.0, 0.0005))
        if value > best_value:
            best_value = value
            best_action = root_action
    return int(best_action)


def trigger_imagination(obs):
    """Query imagination only under local risk or failed forward progress."""
    forward_clearance = float(np.max(obs[5:8]))
    crowded = float(np.mean(obs[5:13] < 0.5)) >= 0.375
    return bool(obs[-1] > 0.5 or forward_clearance < 0.5 or crowded)


def latent_error(latent, observed):
    # Position, goal relation, goal distance, and rays are the state channels;
    # the one-step collision bit is excluded because it is an event flag.
    return float(np.mean(np.abs(latent[:-1] - observed[:-1])))


def run_episode(models, condition, kind, seed, max_steps=MAX_STEPS):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    env = make_world(kind, seed)
    obs = env.reset()
    latent = obs.copy()
    rng = np.random.default_rng(seed + 71000)
    telemetry = PredictorTelemetry()
    path = [env.pos]
    collisions = 0
    latent_errors = []
    predicted_next = None

    for step in range(max_steps):
        planned_this_step = False
        if predicted_next is not None:
            error = latent_error(predicted_next, obs)
            latent_errors.append(error)
            telemetry.prediction_error_sum += error
            telemetry.prediction_error_samples += 1

        if condition == "reactive_controller":
            action = sensor_action(obs, rng)
        elif condition == "triggered_mpc":
            if trigger_imagination(obs):
                action = prospective_action(models, obs, rng, telemetry)
                planned_this_step = True
            else:
                action = sensor_action(obs, rng)
        else:
            if condition == "continuous_grounded":
                latent = correct_latent(latent, obs, gain=1.0)
                telemetry.corrections += 1
            elif condition == "continuous_generative" and step % 8 == 0:
                # A continuously running inner trajectory with sparse reality
                # taps, separating generation from full observation locking.
                latent = correct_latent(latent, obs, gain=0.68)
                telemetry.corrections += 1
            # ungrounded_generative receives only the initial state.
            action = prospective_action(models, latent, rng, telemetry)

        next_obs, collision, done = env.step(action)
        collisions += int(collision)
        path.append(env.pos)

        if condition in {
            "continuous_generative",
            "continuous_grounded",
            "ungrounded_generative",
        }:
            predicted_next, _, _ = predict_counted(
                models, latent, action, telemetry
            )
            latent = predicted_next
        elif condition == "triggered_mpc" and planned_this_step:
            # Error of a freshly grounded one-step query, without persisting it.
            predicted_next, _, _ = predict_counted(models, obs, action, telemetry)
        else:
            predicted_next = None

        obs = next_obs
        if done:
            break

    success = env.pos == env.goal
    direct_distance = math.dist(env.start, env.goal)
    return {
        "condition": condition,
        "geometry": kind,
        "seed": seed,
        "success": float(success),
        "steps": len(path) - 1,
        "collisions": collisions,
        "path_efficiency": direct_distance / max(1, len(path) - 1) if success else 0.0,
        "model_calls": telemetry.model_calls,
        "planning_steps": telemetry.planning_steps,
        "corrections": telemetry.corrections,
        "mean_prediction_error": telemetry.mean_prediction_error,
        "final_latent_error": latent_errors[-1] if latent_errors else None,
        "max_latent_error": max(latent_errors) if latent_errors else None,
        "path": path,
    }


def evaluate(models, seeds=12, max_steps=MAX_STEPS):
    geometries = ("circles", "rectangles", "diagonal_bars", "u_detour")
    episodes = []
    for geometry_index, kind in enumerate(geometries):
        run_count = max(4, seeds // 2) if kind == "u_detour" else seeds
        for condition in CONDITIONS:
            for run in range(run_count):
                seed = 42000 + geometry_index * 1000 + run
                episodes.append(run_episode(models, condition, kind, seed, max_steps))

    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in episodes if row["condition"] == condition]
        errors = [row["mean_prediction_error"] for row in rows if row["mean_prediction_error"] is not None]
        final_errors = [row["final_latent_error"] for row in rows if row["final_latent_error"] is not None]
        summary[condition] = {
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
            "mean_path_efficiency": float(np.mean([row["path_efficiency"] for row in rows])),
            "mean_model_calls": float(np.mean([row["model_calls"] for row in rows])),
            "mean_planning_steps": float(np.mean([row["planning_steps"] for row in rows])),
            "mean_corrections": float(np.mean([row["corrections"] for row in rows])),
            "mean_prediction_error": float(np.mean(errors)) if errors else None,
            "mean_final_latent_error": float(np.mean(final_errors)) if final_errors else None,
        }
    return summary, episodes


def paired_contrast(episodes, treatment, reference):
    indexed = {
        (row["geometry"], row["seed"], row["condition"]): row
        for row in episodes
    }
    episode_keys = sorted({(row["geometry"], row["seed"]) for row in episodes})
    wins = losses = ties = 0
    for geometry, seed in episode_keys:
        treated = indexed[(geometry, seed, treatment)]["success"]
        baseline = indexed[(geometry, seed, reference)]["success"]
        wins += int(treated > baseline)
        losses += int(treated < baseline)
        ties += int(treated == baseline)

    discordant = wins + losses
    if not discordant:
        exact_p = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(wins, losses) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * tail)
    return {
        "treatment": treatment,
        "reference": reference,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "discordant_pairs": discordant,
        "exact_mcnemar_p": exact_p,
    }


def plot_results(summary, path):
    labels = list(CONDITIONS)
    short = ["reactive", "triggered\nMPC", "continuous\ngenerative", "continuous\ngrounded", "ungrounded\ngenerative"]
    success = [summary[label]["success_rate"] for label in labels]
    collisions = [summary[label]["mean_collisions"] for label in labels]
    errors = [summary[label]["mean_final_latent_error"] or 0.0 for label in labels]
    calls = [summary[label]["mean_model_calls"] for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(short, success, color="#2a9d8f")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Task success")
    axes[0, 1].bar(short, collisions, color="#e76f51")
    axes[0, 1].set_title("Mean collisions")
    axes[1, 0].bar(short, errors, color="#e9c46a")
    axes[1, 0].set_title("Final internal-state drift")
    axes[1, 1].bar(short, calls, color="#457b9d")
    axes[1, 1].set_title("Forward-model calls")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=16)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Continuous generation requires recurrent sensory grounding")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller smoke-test run")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    set_seed(91)
    quick = args.quick
    rows = collect_training_data(
        worlds_per_kind=6 if quick else 24,
        steps_per_world=70 if quick else 140,
    )
    models, histories = train_model_ensemble(
        rows,
        members=2 if quick else 3,
        epochs=10 if quick else 42,
    )
    seeds = args.seeds or (3 if quick else 12)
    summary, episodes = evaluate(models, seeds=seeds, max_steps=55 if quick else MAX_STEPS)
    contrasts = {
        "grounded_vs_reactive": paired_contrast(
            episodes, "continuous_grounded", "reactive_controller"
        ),
        "grounded_vs_triggered": paired_contrast(
            episodes, "continuous_grounded", "triggered_mpc"
        ),
        "grounded_vs_sparse": paired_contrast(
            episodes, "continuous_grounded", "continuous_generative"
        ),
        "grounded_vs_ungrounded": paired_contrast(
            episodes, "continuous_grounded", "ungrounded_generative"
        ),
    }

    metrics_path = OUT / "continuous_reality_engine_metrics.json"
    figure_path = OUT / "continuous_reality_engine_summary.png"
    payload = {
        "question": "Does continuous internal generation help, and is sensory correction necessary?",
        "definitions": {
            "reactive_controller": "current observation only; no forward prediction",
            "triggered_mpc": "freshly grounded prediction only when local risk triggers it",
            "continuous_generative": "persistent prediction every step with a reality tap every eight steps",
            "continuous_grounded": "persistent prediction corrected by observation every step",
            "ungrounded_generative": "persistent prediction after initialization with correction removed",
        },
        "summary": summary,
        "paired_contrasts": contrasts,
        "episodes": [{key: value for key, value in row.items() if key != "path"} for row in episodes],
        "training": {
            "transitions": len(rows),
            "ensemble_members": len(models),
            "final_losses": [history[-1] for history in histories],
        },
        "claim_boundary": (
            "The lab compares prediction schedules and sensory grounding in a bounded learned "
            "sensorimotor model. It does not establish phenomenal consciousness or an "
            "electromagnetic mechanism."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    plot_results(summary, figure_path)

    print("\nContinuous reality-engine comparison")
    print("condition                    success  collisions  drift    model calls")
    for condition in CONDITIONS:
        row = summary[condition]
        drift = row["mean_final_latent_error"]
        drift_text = "n/a" if drift is None else f"{drift:.3f}"
        print(
            f"{condition:28s} {row['success_rate']:7.3f}  "
            f"{row['mean_collisions']:10.2f}  {drift_text:>7s}  "
            f"{row['mean_model_calls']:11.1f}"
        )
    print("\nPaired success contrasts (wins-losses, exact McNemar p)")
    for name, contrast in contrasts.items():
        print(
            f"{name:24s} {contrast['wins']:2d}-{contrast['losses']:2d}  "
            f"p={contrast['exact_mcnemar_p']:.4g}"
        )
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
