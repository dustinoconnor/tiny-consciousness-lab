#!/usr/bin/env python3
"""Dynamic conditional workspace test.

This lab asks whether a shared workspace should be bypassed with a hard
confidence threshold, or always monitor and assert control only when
cross-module tension rises.

The toy system has three specialist modules:

- sensory specialist: tracks the current external target
- imagination specialist: predicts the next target from an internal rule
- valence specialist: estimates whether imagination is staying aligned

The workspace blends them only when needed. Its coupling coefficient alpha is
the key variable:

    low tension  -> alpha near 0   -> specialists act locally
    high tension -> alpha near 1   -> workspace dominates and re-grounds
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_valence_lab import action_from_vector, rotate, softmax
from attention_shift_lab import angle_delta, angle_of, generate_shift_world, recovery_step, running_accuracy
from tiny_lab import OUT, set_seed


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def circular_variance(vectors):
    vectors = np.array([unit(v) for v in vectors])
    mean = np.mean(vectors, axis=0)
    return float(1.0 - min(1.0, np.linalg.norm(mean)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def run_condition(name, config, steps=240, shift_step=120, deception_step=70, seed=23):
    target, distractor = generate_shift_world(steps=steps, shift_step=shift_step, seed=seed)
    rng = np.random.default_rng(seed + 300)
    imagination = target[0].copy()
    workspace = target[0].copy()
    model_angle = config["initial_model_angle"]
    alpha_prev = 0.0
    rows = []

    for t in range(steps):
        actual = target[t].copy()
        actual_next = target[t + 1].copy()
        lure = distractor[t].copy()

        deception_active = deception_step <= t < deception_step + config["deception_duration"]
        sensory_observation = actual.copy()
        if deception_active:
            # A brief false sensory conflict: one specialist is being lied to.
            sensory_observation = unit(0.35 * actual + 0.65 * lure)

        imagined_next = rotate(imagination, model_angle)
        prediction_error = float(np.mean((imagined_next - actual_next) ** 2))
        alignment = float(np.exp(-config["alignment_sharpness"] * prediction_error))
        valence_signal = 2.0 * alignment - 1.0

        sensory_module = sensory_observation
        imagination_module = imagination
        valence_module = unit((0.5 + 0.5 * valence_signal) * imagined_next + (0.5 - 0.5 * valence_signal) * sensory_observation)

        disagreement = circular_variance([sensory_module, imagination_module, valence_module])
        surprise = 1.0 - alignment
        lure_conflict = float(max(0.0, np.dot(sensory_observation, lure) - np.dot(sensory_observation, actual)))
        tension = float(0.45 * disagreement + 0.45 * surprise + 0.10 * lure_conflict)

        specialist_vector = unit(0.55 * sensory_module + 0.30 * imagination_module + 0.15 * valence_module)
        workspace_vector = unit(0.45 * sensory_module + 0.35 * valence_module + 0.20 * workspace)

        if config["mode"] == "always_bypass":
            alpha = 0.0
        elif config["mode"] == "always_workspace":
            alpha = 1.0
        elif config["mode"] == "hard_threshold":
            alpha = 1.0 if tension >= config["threshold"] else 0.0
        elif config["mode"] == "soft_tension":
            target_alpha = sigmoid(config["gain"] * (tension - config["threshold"]))
            alpha = (1.0 - config["alpha_smoothing"]) * alpha_prev + config["alpha_smoothing"] * target_alpha
        else:
            raise ValueError(config["mode"])

        attended = unit((1.0 - alpha) * specialist_vector + alpha * workspace_vector)
        workspace_rewrite = float(np.linalg.norm(attended - specialist_vector))
        predicted_action = action_from_vector(rotate(attended, model_angle))
        correct_action = action_from_vector(actual_next)
        correct = predicted_action == correct_action

        delusion = float((1.0 - alpha) * surprise * max(0.0, np.dot(imagination_module, attended)))
        deception_error = float(1.0 - np.dot(attended, actual)) if deception_active else 0.0

        sensory_angle_delta = angle_delta(angle_of(actual_next), angle_of(actual))
        learning_rate = config["base_lr"] + config["workspace_lr"] * alpha * surprise
        if config["mode"] == "always_bypass":
            learning_rate *= 0.2
        old_model_angle = model_angle
        model_angle = (1.0 - learning_rate) * model_angle + learning_rate * sensory_angle_delta
        model_rewrite = abs(angle_delta(model_angle, old_model_angle))

        rows.append(
            {
                "t": t,
                "phase": "pre_shift" if t < shift_step else "post_shift",
                "deception_active": float(deception_active),
                "tension": tension,
                "alpha": float(alpha),
                "alignment": alignment,
                "prediction_error": prediction_error,
                "valence": float(valence_signal),
                "delusion": delusion,
                "deception_error": deception_error,
                "model_angle": float(model_angle),
                "workspace_rewrite": workspace_rewrite,
                "model_rewrite": float(model_rewrite),
                "correct": float(correct),
            }
        )

        if config["mode"] == "always_workspace":
            imagination_mix = 0.90
        else:
            imagination_mix = 0.25 + 0.70 * alpha
        imagination = unit(imagination_mix * actual_next + (1.0 - imagination_mix) * imagined_next)
        if config["mode"] == "always_bypass":
            imagination = unit(0.88 * imagined_next + 0.12 * actual_next + rng.normal(0.0, 0.04, size=2))
        imagination_rewrite = float(np.linalg.norm(imagination - imagined_next))
        rows[-1]["imagination_rewrite"] = imagination_rewrite
        workspace = unit((1.0 - config["workspace_memory"]) * attended + config["workspace_memory"] * workspace)
        alpha_prev = alpha

    return rows


def summarize(rows, shift_step, deception_step):
    pre = [r for r in rows if r["t"] < shift_step]
    early = [r for r in rows if shift_step <= r["t"] < shift_step + 35]
    late = [r for r in rows if r["t"] >= shift_step + 35]
    deception = [r for r in rows if deception_step <= r["t"] < deception_step + 20]
    tensions = np.array([r["tension"] for r in rows])
    high_cutoff = float(np.quantile(tensions, 0.75))
    low_cutoff = float(np.quantile(tensions, 0.25))
    high_tension = [r for r in rows if r["tension"] >= high_cutoff]
    low_tension = [r for r in rows if r["tension"] <= low_cutoff]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk])) if chunk else 0.0

    late_accuracy = mean(late, "correct")
    mean_alpha = mean(rows, "alpha")
    mean_delusion = mean(rows, "delusion")
    deception_error = mean(deception, "deception_error")
    # A deliberately simple utility proxy: reward late adaptation, penalize
    # constant workspace use, delusion, and being fooled during deception.
    efficiency = late_accuracy - 0.18 * mean_alpha - 0.25 * mean_delusion - 0.20 * deception_error
    return {
        "pre_shift_accuracy": mean(pre, "correct"),
        "early_post_shift_accuracy": mean(early, "correct"),
        "late_post_shift_accuracy": late_accuracy,
        "mean_tension": mean(rows, "tension"),
        "mean_alpha": mean_alpha,
        "late_alpha": mean(late, "alpha"),
        "mean_delusion": mean_delusion,
        "deception_error": deception_error,
        "workspace_efficiency_score": float(efficiency),
        "high_tension_alpha": mean(high_tension, "alpha"),
        "low_tension_alpha": mean(low_tension, "alpha"),
        "high_tension_workspace_rewrite": mean(high_tension, "workspace_rewrite"),
        "low_tension_workspace_rewrite": mean(low_tension, "workspace_rewrite"),
        "high_tension_model_rewrite": mean(high_tension, "model_rewrite"),
        "low_tension_model_rewrite": mean(low_tension, "model_rewrite"),
        "high_tension_imagination_rewrite": mean(high_tension, "imagination_rewrite"),
        "low_tension_imagination_rewrite": mean(low_tension, "imagination_rewrite"),
        "recovery_steps_to_75pct": recovery_step(rows, shift_step),
        "final_model_angle": float(rows[-1]["model_angle"]),
    }


def plot_timeseries(results, shift_step, deception_step, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, running_accuracy(rows), label=name)
        axes[1].plot(x, [r["tension"] for r in rows], label=name)
        axes[2].plot(x, [r["alpha"] for r in rows], label=name)
        axes[3].plot(x, [r["delusion"] for r in rows], label=name)
        axes[4].plot(x, [r["model_angle"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1, alpha=0.8)
        ax.axvspan(deception_step, deception_step + 20, color="#f59e0b", alpha=0.12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("tension")
    axes[2].set_ylabel("alpha")
    axes[3].set_ylabel("delusion")
    axes[4].set_ylabel("model angle")
    axes[4].set_xlabel("time step")
    axes[0].set_title("Conditional Workspace: Soft Tension-Gated Control")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["late_post_shift_accuracy", "mean_alpha", "mean_delusion", "workspace_efficiency_score"]
    x = np.arange(len(names))
    width = 0.18
    colors = ["#65a30d", "#2563eb", "#dc2626", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[name][metric] for name in names], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18)
    ax.set_title("Conditional Workspace Summary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(23)
    shift_step = 120
    deception_step = 70
    base = {
        "initial_model_angle": 0.13,
        "alignment_sharpness": 18.0,
        "deception_duration": 20,
        "threshold": 0.17,
        "gain": 22.0,
        "alpha_smoothing": 0.35,
        "base_lr": 0.015,
        "workspace_lr": 0.55,
        "workspace_memory": 0.25,
    }
    configs = {
        "always_bypass": {**base, "mode": "always_bypass"},
        "always_workspace": {**base, "mode": "always_workspace"},
        "hard_threshold_workspace": {**base, "mode": "hard_threshold"},
        "soft_tension_workspace": {**base, "mode": "soft_tension"},
    }
    results = {name: run_condition(name, config, shift_step=shift_step, deception_step=deception_step) for name, config in configs.items()}
    summary = {name: summarize(rows, shift_step, deception_step) for name, rows in results.items()}
    payload = {
        "shift_step": shift_step,
        "deception_step": deception_step,
        "summary": summary,
        "note": (
            "Dynamic regulation test. Alpha is the workspace coupling coefficient. "
            "Soft tension-gating lets the workspace listen continuously but assert control when module disagreement, surprise, or deception rises."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "conditional_workspace_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_timeseries(results, shift_step, deception_step, OUT / "conditional_workspace_timeseries.png")
    plot_summary(summary, OUT / "conditional_workspace_summary.png")
    print("Conditional workspace lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
