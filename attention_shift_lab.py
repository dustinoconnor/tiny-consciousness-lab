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
    """Convert a 2D direction vector into an angle in radians."""
    return math.atan2(v[1], v[0])


def angle_delta(a, b):
    """Smallest signed angular difference between two directions."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def generate_shift_world(steps=220, shift_step=110, seed=11):
    """Create the toy environment.

    The "real world" is a moving 2D target direction. Halfway through, the
    hidden rule reverses. The distractor is a noisy second signal that can grab
    attention if the model is too novelty-seeking.
    """
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
    """Smooth the 0/1 correctness trace so the plot shows trends."""
    vals = np.array([row["correct"] for row in rows])
    return np.convolve(vals, np.ones(window) / window, mode="same")


def recovery_step(rows, shift_step, threshold=0.75, window=12):
    """How many steps after the rule change until accuracy is stable again?"""
    acc = running_accuracy(rows, window=window)
    for i in range(shift_step, len(acc) - window):
        if np.all(acc[i : i + window] >= threshold):
            return int(i - shift_step)
    return None


def run_condition(name, config, steps=220, shift_step=110, seed=11):
    """Run one architecture through the same shifting world.

    This is the core experiment. Each condition has the same sensory data but a
    different internal control scheme: ungated imagination, static
    attention-valence filtering, or adaptive attention-valence filtering.
    """
    target, distractor = generate_shift_world(steps=steps, shift_step=shift_step, seed=seed)
    rng = np.random.default_rng(seed + 200)

    # "Imagination" is the model's current internal guess about the world.
    # It starts aligned with reality, then can drift if the loop is not gated.
    imagination = target[0].copy()
    last_target = target[0].copy()
    last_distractor = distractor[0].copy()

    # model_angle is the inner rule: "how much should the world rotate next?"
    # Before the shift, 0.13 is correct. After the shift, the correct rule is
    # roughly negative. The adaptive model is allowed to retune this angle.
    model_angle = config["initial_model_angle"]
    rows = []

    for t in range(steps):
        actual = target[t]
        actual_next = target[t + 1]
        lure = distractor[t]

        # Predict one step ahead using the current inner model. If this diverges
        # from the next sensory truth, prediction_error rises.
        imagined_next = rotate(imagination, model_angle)
        prediction_error = float(np.mean((imagined_next - actual_next) ** 2))
        alignment = float(np.exp(-config["alignment_sharpness"] * prediction_error))
        novelty = float(np.linalg.norm(lure - last_distractor))

        # A tiny attention mechanism: compare the current internal query against
        # three possible channels: real sensory target, noisy distractor, and
        # imagination itself.
        query = imagination
        keys = np.array([actual, lure, imagination])
        logits = keys @ query / math.sqrt(2.0)
        logits[1] += config["distractor_novelty_bias"] * novelty
        logits[2] += config["imagination_prior"]
        base_attention = softmax(logits)

        if config["valence_filter"]:
            # Valence is used as a grounding gate. Channels that keep the model
            # aligned with reality get boosted; distractors and delusional
            # self-looping get penalized.
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
            # Ungated condition: imagination can reward itself just for being
            # attended to, which is the toy version of a closed delusional loop.
            attention = base_attention
            valence = float(0.35 + 0.65 * base_attention[2])

        # The attended vector is the combined "current belief" after attention
        # chooses how much to trust sensation, distraction, and imagination.
        attended = attention[0] * actual + attention[1] * lure + attention[2] * imagination
        predicted_action = action_from_vector(rotate(attended, model_angle))
        correct_action = action_from_vector(actual_next)
        correct = predicted_action == correct_action

        # Delusion here means: the model is attending to imagination while that
        # imagination is poorly aligned with what actually happens next.
        delusion = float(attention[2] * (1.0 - alignment))
        sensory_angle_delta = angle_delta(angle_of(actual_next), angle_of(actual))
        model_rule_alignment = float(0.5 + 0.5 * math.cos(angle_delta(model_angle, sensory_angle_delta)))
        if config["adaptive_model"]:
            # Low alignment means the old inner rule is losing contact with the
            # world. Surprise increases learning_rate, so the system updates its
            # inner rule faster exactly when reality proves it wrong.
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
                "sensory_action_influence": float(attention[0] * model_rule_alignment),
                "distractor_action_influence": float(attention[1] * model_rule_alignment),
                "imagination_action_influence": float(attention[2] * alignment * model_rule_alignment),
                "model_rule_alignment": model_rule_alignment,
                "task_attention": float(attention[0] + attention[2] * alignment),
                "distractor_fixation": float(attention[1]),
                "delusion": delusion,
                "correct": float(correct),
            }
        )

        if config["valence_filter"]:
            # If delusion is low, preserve the sensory truth strongly. If
            # delusion rises, the next imagination state is forced back toward
            # the next real target instead of free-running.
            sensory_mix = min(1.0, 0.20 + 0.80 * max(0.0, 1.0 - delusion))
            imagination = sensory_mix * actual_next + (1.0 - sensory_mix) * imagined_next
        else:
            # The ungated self-loop lets imagination recursively influence
            # itself, plus a bit of noise, which makes it fragile after the
            # world rule changes.
            imagination = config["self_loop_strength"] * imagined_next + (1.0 - config["self_loop_strength"]) * actual_next
            imagination += rng.normal(0.0, config["imagination_noise"], size=2)

        # Keep the imagination vector on the unit circle. Otherwise length would
        # grow/shrink and confuse angle-based comparisons.
        norm = np.linalg.norm(imagination)
        if norm > 1e-6:
            imagination = imagination / norm
        last_target = actual
        last_distractor = lure

    return rows


def summarize(rows, shift_step):
    """Collapse a full run into the metrics shown in the README."""
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
        "late_sensory_action_influence": mean(late, "sensory_action_influence"),
        "late_imagination_action_influence": mean(late, "imagination_action_influence"),
        "late_model_rule_alignment": mean(late, "model_rule_alignment"),
        "recovery_steps_to_75pct": recovery_step(rows, shift_step),
        "final_model_angle": float(rows[-1]["model_angle"]),
    }


def plot_shift(results, shift_step, path):
    """Plot the main time-series view around the sudden rule change."""
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
    """Plot the before/after accuracy bars for each condition."""
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


def plot_action_influence(results, shift_step, path):
    """Show which channel is actually steering action after attention."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    keep = ["static_attention_valence_filter", "adaptive_attention_valence_filter"]
    for name in keep:
        rows = results[name]
        x = [r["t"] for r in rows]
        axes[0].plot(x, [r["sensory_action_influence"] for r in rows], label=name)
        axes[1].plot(x, [r["imagination_action_influence"] for r in rows], label=name)
        axes[2].plot(x, [r["distractor_action_influence"] for r in rows], label=name)
        axes[3].plot(x, [r["model_rule_alignment"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("sensory influence")
    axes[1].set_ylabel("imagination influence")
    axes[2].set_ylabel("distractor influence")
    axes[3].set_ylabel("model rule alignment")
    axes[3].set_xlabel("time step")
    axes[0].set_title("Action Influence Around Paradigm Shift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(11)
    shift_step = 110

    # Three toy "minds" face the same world:
    # 1. ungated_old_model: imagination can loop on itself and gets stale.
    # 2. static_attention_valence_filter: grounding helps, but the inner rule
    #    itself cannot adapt after the world changes.
    # 3. adaptive_attention_valence_filter: prediction error updates the inner
    #    rule, so surprise becomes useful instead of destabilizing.
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
    plot_action_influence(results, shift_step, OUT / "attention_shift_action_influence.png")
    print("Attention paradigm-shift lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
