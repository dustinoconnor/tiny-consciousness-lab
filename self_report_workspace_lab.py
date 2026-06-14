#!/usr/bin/env python3
"""Self-report workspace test.

This lab adds a tiny persistent self-model register to the conditional
workspace system.

The self-model tracks recent internal dynamics:

- executive dominance: alpha
- system friction: tension
- grounding delta: imagination rewrite
- valence/alignment state
- recent delusion

Two questions:

1. Can the system produce a symbolic chronicle of its own internal regulation?
2. Does feeding that self-model back into control improve behavior, or is it
   only a dashboard?
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_valence_lab import action_from_vector, rotate
from attention_shift_lab import angle_delta, angle_of, generate_shift_world, recovery_step, running_accuracy
from conditional_workspace_lab import circular_variance, sigmoid, unit
from tiny_lab import OUT, set_seed


def classify_report(row, previous_report):
    if row["alignment_restored"]:
        return "alignment_restored"
    if row["workspace_asserted"]:
        return "executive_attention_asserted"
    if row["conflict_detected"]:
        return "conflict_detected"
    if row["stable_autonomy"]:
        return "autonomous_stable"
    return previous_report or "initializing"


REPORT_TEXT = {
    "initializing": "Initializing self-model. Awaiting stable module alignment.",
    "autonomous_stable": "System operating autonomously. Specialists aligned. Environment highly predictable.",
    "conflict_detected": "Conflict detected between sensory, imagination, and valence modules. Internal surprise rising.",
    "executive_attention_asserted": "Executive attention asserted. Re-grounding imagination and updating the internal rule.",
    "alignment_restored": "Alignment restored. Model updated. Relinquishing executive control back to specialists.",
}


def run_condition(name, config, steps=260, shift_step=130, deception_step=75, seed=31):
    target, distractor = generate_shift_world(steps=steps, shift_step=shift_step, seed=seed)
    rng = np.random.default_rng(seed + 400)
    imagination = target[0].copy()
    workspace = target[0].copy()
    model_angle = config["initial_model_angle"]
    alpha_prev = 0.0
    previous_report = "initializing"
    self_state = {
        "dominance": 0.0,
        "friction": 0.0,
        "grounding": 0.0,
        "valence": 1.0,
        "delusion": 0.0,
        "vigilance": 0.0,
    }
    rows = []
    reports = []

    for t in range(steps):
        actual = target[t].copy()
        actual_next = target[t + 1].copy()
        lure = distractor[t].copy()

        deception_active = deception_step <= t < deception_step + config["deception_duration"]
        sensory_observation = actual.copy()
        if deception_active:
            sensory_observation = unit(0.28 * actual + 0.72 * lure)

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
        raw_tension = float(0.45 * disagreement + 0.45 * surprise + 0.10 * lure_conflict)

        # Report-only mode records the self-model but does not let it regulate.
        # Feedback mode uses remembered delusion/conflict as vigilance.
        if config["self_feedback"]:
            vigilance = self_state["vigilance"]
            threshold = config["threshold"] - config["vigilance_threshold_drop"] * vigilance
            gain = config["gain"] + config["vigilance_gain_boost"] * vigilance
            tension = min(1.0, raw_tension + config["vigilance_tension_boost"] * vigilance)
        else:
            vigilance = 0.0
            threshold = config["threshold"]
            gain = config["gain"]
            tension = raw_tension

        specialist_vector = unit(0.55 * sensory_module + 0.30 * imagination_module + 0.15 * valence_module)
        workspace_vector = unit(0.45 * sensory_module + 0.35 * valence_module + 0.20 * workspace)
        target_alpha = sigmoid(gain * (tension - threshold))
        alpha = (1.0 - config["alpha_smoothing"]) * alpha_prev + config["alpha_smoothing"] * target_alpha

        attended = unit((1.0 - alpha) * specialist_vector + alpha * workspace_vector)
        workspace_rewrite = float(np.linalg.norm(attended - specialist_vector))
        predicted_action = action_from_vector(rotate(attended, model_angle))
        correct_action = action_from_vector(actual_next)
        correct = predicted_action == correct_action

        delusion = float((1.0 - alpha) * surprise * max(0.0, np.dot(imagination_module, attended)))
        deception_error = float(1.0 - np.dot(attended, actual)) if deception_active else 0.0

        sensory_angle_delta = angle_delta(angle_of(actual_next), angle_of(actual))
        learning_rate = config["base_lr"] + config["workspace_lr"] * alpha * surprise
        if config["self_feedback"]:
            learning_rate += config["vigilance_lr_boost"] * vigilance * surprise
        old_model_angle = model_angle
        model_angle = (1.0 - learning_rate) * model_angle + learning_rate * sensory_angle_delta
        model_rewrite = abs(angle_delta(model_angle, old_model_angle))

        imagination_mix = min(0.96, 0.25 + 0.70 * alpha + config["vigilance_imagination_boost"] * vigilance)
        imagination = unit(imagination_mix * actual_next + (1.0 - imagination_mix) * imagined_next)
        imagination_rewrite = float(np.linalg.norm(imagination - imagined_next))
        workspace = unit((1.0 - config["workspace_memory"]) * attended + config["workspace_memory"] * workspace)

        self_state = {
            "dominance": 0.90 * self_state["dominance"] + 0.10 * alpha,
            "friction": 0.86 * self_state["friction"] + 0.14 * tension,
            "grounding": 0.82 * self_state["grounding"] + 0.18 * imagination_rewrite,
            "valence": 0.88 * self_state["valence"] + 0.12 * valence_signal,
            "delusion": 0.84 * self_state["delusion"] + 0.16 * delusion,
            "vigilance": min(
                1.0,
                0.78 * self_state["vigilance"]
                + 0.22 * max(tension, delusion, deception_error, 1.0 - alignment),
            ),
        }

        row = {
            "t": t,
            "phase": "pre_shift" if t < shift_step else "post_shift",
            "deception_active": float(deception_active),
            "raw_tension": raw_tension,
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
            "imagination_rewrite": imagination_rewrite,
            "self_dominance": self_state["dominance"],
            "self_friction": self_state["friction"],
            "self_grounding": self_state["grounding"],
            "self_valence": self_state["valence"],
            "self_delusion": self_state["delusion"],
            "self_vigilance": self_state["vigilance"],
            "correct": float(correct),
        }
        row["stable_autonomy"] = row["tension"] < 0.045 and row["alpha"] < 0.16 and row["alignment"] > 0.92
        row["conflict_detected"] = row["tension"] > 0.16 or row["delusion"] > 0.12 or row["deception_error"] > 0.18
        row["workspace_asserted"] = row["alpha"] > 0.45 and row["workspace_rewrite"] > 0.015
        row["alignment_restored"] = previous_report in {"conflict_detected", "executive_attention_asserted"} and row["alignment"] > 0.88 and row["alpha"] < 0.28
        report = classify_report(row, previous_report)
        row["report"] = report
        if not reports or reports[-1]["report"] != report:
            reports.append({"t": t, "report": report, "text": REPORT_TEXT[report]})
        previous_report = report
        rows.append(row)
        alpha_prev = alpha

    return rows, reports


def summarize(rows, reports, shift_step, deception_step):
    pre = [r for r in rows if r["t"] < shift_step]
    early = [r for r in rows if shift_step <= r["t"] < shift_step + 35]
    late = [r for r in rows if r["t"] >= shift_step + 35]
    deception = [r for r in rows if deception_step <= r["t"] < deception_step + 20]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk])) if chunk else 0.0

    late_accuracy = mean(late, "correct")
    report_sequence = [item["report"] for item in reports]
    expected = ["autonomous_stable", "conflict_detected", "executive_attention_asserted", "alignment_restored"]
    report_coverage = sum(1 for label in expected if label in report_sequence) / len(expected)
    efficiency = (
        late_accuracy
        - 0.16 * mean(rows, "alpha")
        - 0.25 * mean(rows, "delusion")
        - 0.20 * mean(deception, "deception_error")
        + 0.08 * report_coverage
    )
    return {
        "pre_shift_accuracy": mean(pre, "correct"),
        "early_post_shift_accuracy": mean(early, "correct"),
        "late_post_shift_accuracy": late_accuracy,
        "mean_alpha": mean(rows, "alpha"),
        "mean_delusion": mean(rows, "delusion"),
        "mean_self_vigilance": mean(rows, "self_vigilance"),
        "deception_error": mean(deception, "deception_error"),
        "report_coverage": float(report_coverage),
        "workspace_efficiency_score": float(efficiency),
        "recovery_steps_to_75pct": recovery_step(rows, shift_step),
        "final_model_angle": float(rows[-1]["model_angle"]),
        "report_count": len(reports),
    }


def plot_timeseries(results, shift_step, deception_step, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for name, bundle in results.items():
        rows = bundle["rows"]
        x = [r["t"] for r in rows]
        axes[0].plot(x, running_accuracy(rows), label=name)
        axes[1].plot(x, [r["alpha"] for r in rows], label=name)
        axes[2].plot(x, [r["self_vigilance"] for r in rows], label=name)
        axes[3].plot(x, [r["delusion"] for r in rows], label=name)
        axes[4].plot(x, [r["model_angle"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1, alpha=0.8)
        ax.axvspan(deception_step, deception_step + 20, color="#f59e0b", alpha=0.12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("alpha")
    axes[2].set_ylabel("self vigilance")
    axes[3].set_ylabel("delusion")
    axes[4].set_ylabel("model angle")
    axes[4].set_xlabel("time step")
    axes[0].set_title("Self-Report Workspace: Introspection as Control Signal")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["late_post_shift_accuracy", "mean_self_vigilance", "report_coverage", "workspace_efficiency_score"]
    x = np.arange(len(names))
    width = 0.18
    colors = ["#65a30d", "#2563eb", "#8b5cf6", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[name][metric] for name in names], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_title("Self-Report Workspace Summary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(31)
    shift_step = 130
    deception_step = 75
    base = {
        "initial_model_angle": 0.13,
        "alignment_sharpness": 18.0,
        "deception_duration": 20,
        "threshold": 0.17,
        "gain": 22.0,
        "alpha_smoothing": 0.35,
        "base_lr": 0.015,
        "workspace_lr": 0.52,
        "workspace_memory": 0.25,
        "vigilance_threshold_drop": 0.065,
        "vigilance_gain_boost": 7.0,
        "vigilance_tension_boost": 0.055,
        "vigilance_lr_boost": 0.20,
        "vigilance_imagination_boost": 0.12,
    }
    configs = {
        "report_only": {**base, "self_feedback": False},
        "self_feedback": {**base, "self_feedback": True},
    }
    results = {}
    summary = {}
    for name, config in configs.items():
        rows, reports = run_condition(name, config, shift_step=shift_step, deception_step=deception_step)
        results[name] = {"rows": rows, "reports": reports}
        summary[name] = summarize(rows, reports, shift_step, deception_step)

    payload = {
        "shift_step": shift_step,
        "deception_step": deception_step,
        "summary": summary,
        "reports": {name: bundle["reports"] for name, bundle in results.items()},
        "note": (
            "The self-model is a rolling register of alpha, tension, grounding, valence, and delusion. "
            "report_only logs symbolic introspection; self_feedback lets the self-model alter vigilance and control."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "self_report_workspace_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_timeseries(results, shift_step, deception_step, OUT / "self_report_workspace_timeseries.png")
    plot_summary(summary, OUT / "self_report_workspace_summary.png")
    print("Self-report workspace lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
