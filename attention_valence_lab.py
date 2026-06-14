#!/usr/bin/env python3
"""Attention-valence filter experiment.

This is the "Ritalin circuit" toy test:

Can a valence-shaped attention gate keep an internal imagination loop grounded
to sensory reality while ignoring tempting distractors?

The setup is intentionally small and deterministic enough to inspect. A target
signal follows a smooth hidden path. A distractor signal has high novelty but no
task value. An imagination stream predicts the next target state, but can drift
if it is trusted without reality-checking.

The attention-valence filter rewards imagination when it predicts the sensory
target and starves it when it becomes detached.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def generate_world(steps=180, seed=7):
    rng = np.random.default_rng(seed)
    target = []
    distractor = []
    phase = 0.0
    noise_phase = rng.uniform(0, 2 * np.pi)
    for t in range(steps + 1):
        phase += 0.13 + 0.025 * math.sin(t / 17.0)
        true = np.array([math.cos(phase), math.sin(phase)])
        target.append(true)

        noise_phase += rng.normal(0.0, 0.9)
        amp = 1.0 + (0.9 if t % 23 in {0, 1, 2} else 0.0)
        lure = amp * np.array([math.cos(noise_phase), math.sin(noise_phase)])
        distractor.append(lure)
    return np.array(target), np.array(distractor)


def action_from_vector(v):
    # Four coarse action bins: right, up, left, down.
    angle = math.atan2(v[1], v[0])
    if -math.pi / 4 <= angle < math.pi / 4:
        return 0
    if math.pi / 4 <= angle < 3 * math.pi / 4:
        return 1
    if angle >= 3 * math.pi / 4 or angle < -3 * math.pi / 4:
        return 2
    return 3


def run_condition(name, config, steps=180, seed=7):
    target, distractor = generate_world(steps=steps, seed=seed)
    rng = np.random.default_rng(seed + 100)

    imagination = target[0].copy()
    last_target = target[0].copy()
    last_distractor = distractor[0].copy()
    rows = []

    for t in range(steps):
        actual = target[t]
        actual_next = target[t + 1]
        lure = distractor[t]

        imagined_next = rotate(imagination, config["model_angle"])
        prediction_error = float(np.mean((imagined_next - actual_next) ** 2))
        alignment = float(np.exp(-config["alignment_sharpness"] * prediction_error))
        novelty = float(np.linalg.norm(lure - last_distractor))

        query = imagination
        keys = np.array([actual, lure, imagination])
        logits = keys @ query / math.sqrt(2.0)
        logits[1] += config["distractor_novelty_bias"] * novelty
        logits[2] += config["imagination_prior"]
        base_attention = softmax(logits)

        if config["valence_filter"]:
            task_valence = alignment + config["task_stay_reward"] * base_attention[0]
            channel_valence = np.array(
                [
                    1.0 + config["task_stay_reward"],
                    config["distractor_penalty"],
                    max(config["floor"], task_valence),
                ]
            )
            attention = base_attention * channel_valence
            attention = attention / np.sum(attention)
            valence = float(task_valence - config["delusion_penalty"] * base_attention[2] * (1.0 - alignment))
        else:
            attention = base_attention
            valence = float(0.35 + 0.65 * base_attention[2])

        attended = attention[0] * actual + attention[1] * lure + attention[2] * imagination
        predicted_action = action_from_vector(rotate(attended, config["model_angle"]))
        correct_action = action_from_vector(actual_next)
        correct = predicted_action == correct_action

        delusion = float(attention[2] * (1.0 - alignment))
        task_attention = float(attention[0] + attention[2] * alignment)
        distractor_fixation = float(attention[1])

        rows.append(
            {
                "t": t,
                "alignment": alignment,
                "prediction_error": prediction_error,
                "valence": valence,
                "target_attention": float(attention[0]),
                "distractor_attention": float(attention[1]),
                "imagination_attention": float(attention[2]),
                "task_attention": task_attention,
                "distractor_fixation": distractor_fixation,
                "delusion": delusion,
                "correct": float(correct),
            }
        )

        sensory_update = actual_next
        imagined_update = imagined_next
        if config["valence_filter"]:
            sensory_mix = min(1.0, 0.25 + 0.75 * max(0.0, 1.0 - delusion))
            imagination = sensory_mix * sensory_update + (1.0 - sensory_mix) * imagined_update
        else:
            self_mix = config["self_loop_strength"]
            imagination = self_mix * imagined_update + (1.0 - self_mix) * sensory_update
            imagination += rng.normal(0.0, config["imagination_noise"], size=2)

        norm = np.linalg.norm(imagination)
        if norm > 1e-6:
            imagination = imagination / norm
        last_target = actual
        last_distractor = lure

    return rows


def summarize(rows):
    keys = [
        "correct",
        "task_attention",
        "distractor_fixation",
        "delusion",
        "valence",
        "alignment",
        "prediction_error",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def plot_attention(results, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for name, rows in results.items():
        x = [row["t"] for row in rows]
        axes[0].plot(x, [row["task_attention"] for row in rows], label=name)
        axes[1].plot(x, [row["distractor_fixation"] for row in rows], label=name)
        axes[2].plot(x, [row["delusion"] for row in rows], label=name)
        window = 12
        acc = np.convolve([row["correct"] for row in rows], np.ones(window) / window, mode="same")
        axes[3].plot(x, acc, label=name)

    axes[0].set_ylabel("task attention")
    axes[1].set_ylabel("distractor fixation")
    axes[2].set_ylabel("delusion index")
    axes[3].set_ylabel("rolling accuracy")
    axes[3].set_xlabel("time step")
    axes[0].set_title("Attention-Valence Filter: Grounding Imagination to Sense")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["correct", "task_attention", "distractor_fixation", "delusion"]
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#16a3a6", "#ff8a00", "#dc2626", "#8b5cf6"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[name][metric] for name in names], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18)
    ax.set_ylim(0, 1.05)
    ax.set_title("Attention-Valence Filter Summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(7)
    configs = {
        "ungated_attention": {
            "model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 1.2,
            "imagination_prior": 0.55,
            "valence_filter": False,
            "task_stay_reward": 0.0,
            "distractor_penalty": 1.0,
            "delusion_penalty": 0.0,
            "floor": 0.05,
            "self_loop_strength": 0.82,
            "imagination_noise": 0.035,
        },
        "self_amplified_imagination": {
            "model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 0.7,
            "imagination_prior": 1.25,
            "valence_filter": False,
            "task_stay_reward": 0.0,
            "distractor_penalty": 1.0,
            "delusion_penalty": 0.0,
            "floor": 0.05,
            "self_loop_strength": 0.94,
            "imagination_noise": 0.06,
        },
        "attention_valence_filter": {
            "model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 1.2,
            "imagination_prior": 0.9,
            "valence_filter": True,
            "task_stay_reward": 0.35,
            "distractor_penalty": 0.22,
            "delusion_penalty": 0.8,
            "floor": 0.03,
            "self_loop_strength": 0.0,
            "imagination_noise": 0.0,
        },
        "overconstrained_filter": {
            "model_angle": 0.13,
            "alignment_sharpness": 32.0,
            "distractor_novelty_bias": 1.2,
            "imagination_prior": 0.45,
            "valence_filter": True,
            "task_stay_reward": 0.55,
            "distractor_penalty": 0.06,
            "delusion_penalty": 1.8,
            "floor": 0.005,
            "self_loop_strength": 0.0,
            "imagination_noise": 0.0,
        },
    }
    results = {name: run_condition(name, config) for name, config in configs.items()}
    summary = {name: summarize(rows) for name, rows in results.items()}
    payload = {
        "summary": summary,
        "note": (
            "Toy attention-valence filter. The valence-gated condition multiplies attention by a prediction-alignment "
            "signal so imagination is trusted only when it matches sensory reality. This is not a biological ADHD model."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "attention_valence_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_attention(results, OUT / "attention_valence_timeseries.png")
    plot_summary(summary, OUT / "attention_valence_summary.png")
    print("Attention-valence lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
