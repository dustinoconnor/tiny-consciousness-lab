#!/usr/bin/env python3
"""Counterfactual depth vs breadth for the attention-valence filter.

This lab asks a small Friston-ish question:

Does an attention-valence system behave more like an agent when it can project
the consequences of an attention shift into its private future?

Breadth is how many candidate attention shifts the system compares right now.
Depth is how many recursive future sensorium updates it rolls out for each
candidate before acting.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_valence_lab import action_from_vector, rotate, softmax
from tiny_lab import OUT, set_seed


def generate_counterfactual_world(steps=240, seed=23):
    """A smooth target with tempting high-novelty distractor bursts."""
    rng = np.random.default_rng(seed)
    target = []
    distractor = []
    phase = 0.0
    lure_phase = rng.uniform(0, 2 * np.pi)
    for t in range(steps + 8):
        phase += 0.115 + 0.03 * math.sin(t / 16.0) + 0.018 * math.sin(t / 7.0)
        target.append(np.array([math.cos(phase), math.sin(phase)]))

        burst = 1.15 if t % 31 in {0, 1, 2, 3, 4} else 0.0
        lure_phase += rng.normal(0.0, 0.82 + burst)
        amp = 1.0 + burst
        distractor.append(amp * np.array([math.cos(lure_phase), math.sin(lure_phase)]))
    return np.array(target), np.array(distractor)


def normalized(v):
    norm = np.linalg.norm(v)
    if norm <= 1e-8:
        return v
    return v / norm


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rolling_mean(values, window=14):
    arr = np.asarray(values, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode="same")


def candidate_nudges(breadth, rng):
    """Attention-logit offsets for sensory, distractor, imagination channels."""
    library = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.1, -0.4, -0.2]),
        np.array([-0.2, 1.1, -0.2]),
        np.array([-0.2, -0.4, 1.1]),
        np.array([0.65, -0.25, 0.65]),
        np.array([0.65, 0.65, -0.35]),
        np.array([-0.35, 0.55, 0.75]),
        np.array([1.4, -0.8, 0.15]),
        np.array([0.2, -0.8, 1.35]),
    ]
    nudges = library[: max(1, min(breadth, len(library)))]
    while len(nudges) < breadth:
        nudges.append(rng.normal(0.0, 0.75, size=3))
    return nudges


def base_attention_logits(actual, lure, imagination, config, last_lure):
    novelty = float(np.linalg.norm(lure - last_lure))
    query = imagination
    keys = np.array([actual, lure, imagination])
    logits = keys @ query / math.sqrt(2.0)
    logits[1] += config["distractor_novelty_bias"] * novelty
    logits[2] += config["imagination_prior"]
    return logits


def apply_valence_gate(base_attention, alignment, config):
    if not config["valence_filter"]:
        return base_attention

    channel_valence = np.array(
        [
            1.0 + config["sensory_reward"],
            config["distractor_penalty"],
            max(config["floor"], alignment),
        ]
    )
    attention = base_attention * channel_valence
    return attention / np.sum(attention)


def next_imagination(sim_imagination, actual_next, attended, alignment, attention, config):
    imagined_next = rotate(sim_imagination, config["model_angle"])
    if config["valence_filter"]:
        sensory_mix = config["sensory_anchor"] + config["valence_anchor"] * alignment
        sensory_mix -= config["private_future_leak"] * attention[2]
        sensory_mix = float(np.clip(sensory_mix, 0.04, 1.0))
        update = sensory_mix * actual_next + (1.0 - sensory_mix) * imagined_next
    else:
        update = (
            config["self_loop_strength"] * imagined_next
            + config["attended_loop_strength"] * attended
            + config["sensory_anchor"] * actual_next
        )
    return normalized(update)


def evaluate_attention_shift(target, distractor, t, imagination, nudge, config, last_lure):
    """Score one current attention shift by recursively rolling it forward."""
    horizon = max(1, int(config["depth"]))
    sim_imagination = imagination.copy()
    total_value = 0.0
    total_error = 0.0
    first_attention = None
    first_alignment = None
    discount = 1.0

    for d in range(horizon):
        idx = t + d
        actual = target[idx]
        actual_next = target[idx + 1]
        lure = distractor[idx]
        previous_lure = last_lure if d == 0 else distractor[idx - 1]

        logits = base_attention_logits(actual, lure, sim_imagination, config, previous_lure)
        base_attention = softmax(logits + nudge)
        predicted_next = rotate(sim_imagination, config["model_angle"])
        prediction_error = mse(predicted_next, actual_next)
        alignment = float(np.exp(-config["alignment_sharpness"] * prediction_error))
        attention = apply_valence_gate(base_attention, alignment, config)

        attended = attention[0] * actual + attention[1] * lure + attention[2] * sim_imagination
        predicted_action = action_from_vector(rotate(attended, config["model_angle"]))
        correct_action = action_from_vector(actual_next)
        correct = float(predicted_action == correct_action)
        delusion = float(attention[2] * (1.0 - alignment))
        future_error = mse(rotate(attended, config["model_angle"]), actual_next)

        value = (
            config["alignment_reward"] * alignment
            + config["action_reward"] * correct
            - config["future_error_penalty"] * future_error
            - config["distractor_cost"] * attention[1]
            - config["delusion_cost"] * delusion
        )
        total_value += discount * value
        total_error += discount * future_error

        if first_attention is None:
            first_attention = attention
            first_alignment = alignment

        sim_imagination = next_imagination(sim_imagination, actual_next, attended, alignment, attention, config)
        discount *= config["discount"]

    return {
        "value": float(total_value),
        "rollout_error": float(total_error / horizon),
        "attention": first_attention,
        "alignment": float(first_alignment),
    }


def choose_attention(target, distractor, t, imagination, config, last_lure, rng):
    nudges = candidate_nudges(config["breadth"], rng)
    scored = [
        evaluate_attention_shift(target, distractor, t, imagination, nudge, config, last_lure)
        for nudge in nudges
    ]
    best = max(scored, key=lambda item: item["value"])
    return best


def run_condition(name, config, steps=240, seed=23):
    target, distractor = generate_counterfactual_world(steps=steps, seed=seed)
    rng = np.random.default_rng(seed + config["seed_offset"])
    imagination = normalized(target[0] + rng.normal(0.0, 0.05, size=2))
    last_lure = distractor[0]
    rows = []

    for t in range(steps):
        actual = target[t]
        actual_next = target[t + 1]
        lure = distractor[t]

        chosen = choose_attention(target, distractor, t, imagination, config, last_lure, rng)
        attention = chosen["attention"]
        alignment = chosen["alignment"]
        attended = attention[0] * actual + attention[1] * lure + attention[2] * imagination

        predicted_action = action_from_vector(rotate(attended, config["model_angle"]))
        correct_action = action_from_vector(actual_next)
        correct = float(predicted_action == correct_action)
        delusion = float(attention[2] * (1.0 - alignment))
        future_error = mse(rotate(attended, config["model_angle"]), actual_next)
        task_attention = float(attention[0] + attention[2] * alignment)
        valence = float(alignment - config["distractor_cost"] * attention[1] - config["delusion_cost"] * delusion)

        rows.append(
            {
                "t": t,
                "breadth": int(config["breadth"]),
                "depth": int(config["depth"]),
                "target_attention": float(attention[0]),
                "distractor_attention": float(attention[1]),
                "imagination_attention": float(attention[2]),
                "task_attention": task_attention,
                "distractor_fixation": float(attention[1]),
                "delusion": delusion,
                "alignment": alignment,
                "future_error": future_error,
                "rollout_error": chosen["rollout_error"],
                "counterfactual_value": chosen["value"],
                "valence": valence,
                "correct": correct,
            }
        )

        imagination = next_imagination(imagination, actual_next, attended, alignment, attention, config)
        if not config["valence_filter"]:
            imagination = normalized(imagination + rng.normal(0.0, config["imagination_noise"], size=2))
        last_lure = lure

    return rows


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    return {
        "accuracy": mean("correct"),
        "task_attention": mean("task_attention"),
        "distractor_fixation": mean("distractor_fixation"),
        "delusion": mean("delusion"),
        "future_error": mean("future_error"),
        "rollout_error": mean("rollout_error"),
        "counterfactual_value": mean("counterfactual_value"),
        "breadth": int(rows[0]["breadth"]),
        "depth": int(rows[0]["depth"]),
    }


def plot_timeseries(results, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, rolling_mean([r["correct"] for r in rows]), label=name)
        axes[1].plot(x, [r["task_attention"] for r in rows], label=name)
        axes[2].plot(x, [r["distractor_fixation"] for r in rows], label=name)
        axes[3].plot(x, [r["delusion"] for r in rows], label=name)
        axes[4].plot(x, rolling_mean([r["future_error"] for r in rows]), label=name)

    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("task attention")
    axes[2].set_ylabel("distractor fixation")
    axes[3].set_ylabel("delusion")
    axes[4].set_ylabel("future error")
    axes[4].set_xlabel("time step")
    axes[0].set_title("Counterfactual Depth vs Breadth")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["accuracy", "task_attention", "distractor_fixation", "delusion", "future_error"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#0f9f6e", "#2563eb", "#f59e0b", "#dc2626", "#6b7280"]
    for i, metric in enumerate(metrics):
        vals = [summary[name][metric] for name in names]
        ax.bar(x + (i - 2) * width, vals, width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Depth helps only when future imagination stays valence-grounded")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(23)
    base = {
        "model_angle": 0.12,
        "alignment_sharpness": 20.0,
        "distractor_novelty_bias": 1.05,
        "imagination_prior": 0.55,
        "sensory_reward": 0.25,
        "distractor_penalty": 0.18,
        "floor": 0.025,
        "alignment_reward": 0.9,
        "action_reward": 0.55,
        "future_error_penalty": 0.75,
        "distractor_cost": 0.45,
        "delusion_cost": 1.05,
        "discount": 0.82,
        "sensory_anchor": 0.25,
        "valence_anchor": 0.68,
        "private_future_leak": 0.25,
        "self_loop_strength": 0.70,
        "attended_loop_strength": 0.18,
        "imagination_noise": 0.035,
    }
    configs = {
        "reactive_thermostat": {
            **base,
            "breadth": 1,
            "depth": 1,
            "valence_filter": True,
            "imagination_prior": 0.15,
            "seed_offset": 1,
        },
        "wide_next_step": {
            **base,
            "breadth": 9,
            "depth": 1,
            "valence_filter": True,
            "seed_offset": 2,
        },
        "deep_counterfactual": {
            **base,
            "breadth": 5,
            "depth": 5,
            "valence_filter": True,
            "seed_offset": 3,
        },
        "deep_ungrounded": {
            **base,
            "breadth": 5,
            "depth": 5,
            "valence_filter": False,
            "imagination_prior": 1.25,
            "sensory_anchor": 0.08,
            "self_loop_strength": 0.84,
            "attended_loop_strength": 0.23,
            "distractor_cost": 0.18,
            "delusion_cost": 0.12,
            "imagination_noise": 0.06,
            "seed_offset": 4,
        },
        "deep_valence_filter": {
            **base,
            "breadth": 7,
            "depth": 6,
            "valence_filter": True,
            "imagination_prior": 0.82,
            "distractor_penalty": 0.10,
            "delusion_cost": 1.45,
            "valence_anchor": 0.78,
            "seed_offset": 5,
        },
    }

    results = {name: run_condition(name, config) for name, config in configs.items()}
    summary = {name: summarize(rows) for name, rows in results.items()}
    payload = {
        "summary": summary,
        "note": (
            "Breadth compares more immediate attention shifts. Depth recursively rolls each shift forward into a "
            "private future sensorium before acting. In this toy lab, deeper rollouts help when the valence filter "
            "keeps imagination prediction-aligned; ungrounded depth becomes a self-reinforcing private future."
        ),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "counterfactual_depth_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_timeseries(results, OUT / "counterfactual_depth_timeseries.png")
    plot_summary(summary, OUT / "counterfactual_depth_summary.png")
    print("Counterfactual depth lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
