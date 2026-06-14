#!/usr/bin/env python3
"""Dynamic paradigm-shift test for the attention-valence filter.

The first attention-valence lab showed that prediction-aligned attention can
keep imagination grounded in a stable world. This test changes the world halfway
through the run.

Question:

Can the valence filter decouple from an old internal model and rebuild a new one
from sensory evidence after the environment changes?
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_valence_lab import action_from_vector, rotate, softmax
from tiny_lab import OUT, set_seed


def angle_of(v):
    return math.atan2(v[1], v[0])


def angle_delta(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def generate_shift_world(steps=220, shift_step=110, seed=11):
    rng = np.random.default_rng(seed)
    target = []
    distractor = []
    phase = 0.0
    noise_phase = rng.uniform(0, 2 * np.pi)
    for t in range(steps + 1):
        if t < shift_step:
            phase += 0.13 + 0.02 * math.sin(t / 18.0)
        else:
            # The hidden rule changes direction and speed.
            phase += -0.21 + 0.035 * math.sin(t / 9.0)
        target.append(np.array([math.cos(phase), math.sin(phase)]))

        noise_phase += rng.normal(0.0, 1.1)
        amp = 1.0 + (1.0 if t % 19 in {0, 1, 2, 3} else 0.0)
        distractor.append(amp * np.array([math.cos(noise_phase), math.sin(noise_phase)]))
    return np.array(target), np.array(distractor)


def running_accuracy(rows, window=12):
    vals = np.array([row["correct"] for row in rows])
    return np.convolve(vals, np.ones(window) / window, mode="same")


def recovery_step(rows, shift_step, threshold=0.75, window=12):
    acc = running_accuracy(rows, window=window)
    for i in range(shift_step, len(acc) - window):
        if np.all(acc[i : i + window] >= threshold):
            return int(i - shift_step)
    return None


def run_condition(name, config, steps=220, shift_step=110, seed=11):
    target, distractor = generate_shift_world(steps=steps, shift_step=shift_step, seed=seed)
    rng = np.random.default_rng(seed + 200)
    imagination = target[0].copy()
    last_target = target[0].copy()
    last_distractor = distractor[0].copy()
    model_angle = config["initial_model_angle"]
    rows = []

    for t in range(steps):
        actual = target[t]
        actual_next = target[t + 1]
        lure = distractor[t]
        imagined_next = rotate(imagination, model_angle)
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
        predicted_action = action_from_vector(rotate(attended, model_angle))
        correct_action = action_from_vector(actual_next)
        correct = predicted_action == correct_action

        delusion = float(attention[2] * (1.0 - alignment))
        sensory_angle_delta = angle_delta(angle_of(actual_next), angle_of(actual))
        if config["adaptive_model"]:
            # Low alignment means the old inner rule is losing contact with the world.
            # Shift trust toward direct sensory evidence until prediction fidelity returns.
            learning_rate = config["base_lr"] + config["surprise_lr"] * (1.0 - alignment)
            model_angle = (1.0 - learning_rate) * model_angle + learning_rate * sensory_angle_delta

        rows.append(
            {
                "t": t,
                "phase": "pre_shift" if t < shift_step else "post_shift",
                "alignment": alignment,
                "prediction_error": prediction_error,
                "valence": valence,
                "model_angle": float(model_angle),
                "target_attention": float(attention[0]),
                "distractor_attention": float(attention[1]),
                "imagination_attention": float(attention[2]),
                "task_attention": float(attention[0] + attention[2] * alignment),
                "distractor_fixation": float(attention[1]),
                "delusion": delusion,
                "correct": float(correct),
            }
        )

        if config["valence_filter"]:
            sensory_mix = min(1.0, 0.20 + 0.80 * max(0.0, 1.0 - delusion))
            imagination = sensory_mix * actual_next + (1.0 - sensory_mix) * imagined_next
        else:
            imagination = config["self_loop_strength"] * imagined_next + (1.0 - config["self_loop_strength"]) * actual_next
            imagination += rng.normal(0.0, config["imagination_noise"], size=2)
        norm = np.linalg.norm(imagination)
        if norm > 1e-6:
            imagination = imagination / norm
        last_target = actual
        last_distractor = lure

    return rows


def summarize(rows, shift_step):
    pre = [r for r in rows if r["t"] < shift_step]
    early = [r for r in rows if shift_step <= r["t"] < shift_step + 35]
    late = [r for r in rows if r["t"] >= shift_step + 35]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk]))

    return {
        "pre_shift_accuracy": mean(pre, "correct"),
        "early_post_shift_accuracy": mean(early, "correct"),
        "late_post_shift_accuracy": mean(late, "correct"),
        "late_task_attention": mean(late, "task_attention"),
        "late_distractor_fixation": mean(late, "distractor_fixation"),
        "late_delusion": mean(late, "delusion"),
        "recovery_steps_to_75pct": recovery_step(rows, shift_step),
        "final_model_angle": float(rows[-1]["model_angle"]),
    }


def plot_shift(results, shift_step, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, running_accuracy(rows), label=name)
        axes[1].plot(x, [r["alignment"] for r in rows], label=name)
        axes[2].plot(x, [r["delusion"] for r in rows], label=name)
        axes[3].plot(x, [r["distractor_fixation"] for r in rows], label=name)
        axes[4].plot(x, [r["model_angle"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("alignment")
    axes[2].set_ylabel("delusion")
    axes[3].set_ylabel("distractor")
    axes[4].set_ylabel("model angle")
    axes[4].set_xlabel("time step")
    axes[0].set_title("Paradigm Shift: Can Attention-Valence Re-Ground?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["pre_shift_accuracy", "early_post_shift_accuracy", "late_post_shift_accuracy"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#16a3a6", "#ff8a00", "#65a30d"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, [summary[name][metric] for name in names], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18)
    ax.set_ylim(0, 1.05)
    ax.set_title("Paradigm Shift Accuracy Before and After Rule Change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(11)
    shift_step = 110
    configs = {
        "ungated_old_model": {
            "initial_model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 1.1,
            "imagination_prior": 0.8,
            "valence_filter": False,
            "task_stay_reward": 0.0,
            "distractor_penalty": 1.0,
            "delusion_penalty": 0.0,
            "floor": 0.03,
            "self_loop_strength": 0.9,
            "imagination_noise": 0.05,
            "adaptive_model": False,
            "base_lr": 0.0,
            "surprise_lr": 0.0,
        },
        "static_attention_valence_filter": {
            "initial_model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 1.1,
            "imagination_prior": 0.8,
            "valence_filter": True,
            "task_stay_reward": 0.35,
            "distractor_penalty": 0.2,
            "delusion_penalty": 0.8,
            "floor": 0.03,
            "self_loop_strength": 0.0,
            "imagination_noise": 0.0,
            "adaptive_model": False,
            "base_lr": 0.0,
            "surprise_lr": 0.0,
        },
        "adaptive_attention_valence_filter": {
            "initial_model_angle": 0.13,
            "alignment_sharpness": 18.0,
            "distractor_novelty_bias": 1.1,
            "imagination_prior": 0.8,
            "valence_filter": True,
            "task_stay_reward": 0.35,
            "distractor_penalty": 0.2,
            "delusion_penalty": 0.8,
            "floor": 0.03,
            "self_loop_strength": 0.0,
            "imagination_noise": 0.0,
            "adaptive_model": True,
            "base_lr": 0.02,
            "surprise_lr": 0.42,
        },
    }
    results = {name: run_condition(name, config, shift_step=shift_step) for name, config in configs.items()}
    summary = {name: summarize(rows, shift_step) for name, rows in results.items()}
    payload = {
        "shift_step": shift_step,
        "summary": summary,
        "note": (
            "The target dynamics reverse halfway through. Static filters can remain grounded to the current sensory stream "
            "but still act through an obsolete prediction rule. The adaptive filter uses prediction error as surprise to "
            "retune the inner model angle."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "attention_shift_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_shift(results, shift_step, OUT / "attention_shift_timeseries.png")
    plot_summary(summary, OUT / "attention_shift_summary.png")
    print("Attention paradigm-shift lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
