#!/usr/bin/env python3
"""Adaptive sleep and fatigue self-report experiment.

The earlier sleep labs tested fixed offline sleep and always-on repair. This
lab asks a more functional question:

    Can a tiny functional ego detect that it is tired, choose a sleep duration,
    hand off to a refreshed successor state, and avoid both under-sleeping and
    over-sleeping?

The model is deliberately abstract. It tracks the quantities that matter for the
architecture thesis:

- crosstalk: weak recurrent echoes that accumulate during waking
- complexity: model bloat from explaining noisy observations
- memory: retained task structure / useful world-model pathways
- fatigue_report: a self-model reading of crosstalk, complexity, error, latency
- delusion_index: closed-loop instability caused by excess crosstalk

Sleep is modeled as offline dream repair. Short sleep fails to prune enough
noise. Medium sleep restores separability. Very long sleep over-prunes and
damages memory, producing computational amnesia. A successor handoff is modeled
as a fast state distillation pass: preserve task memory and goals, discard weak
echo-like crosstalk, and continue acting without a visible offline phase.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


SLEEP_DURATIONS = [0, 5, 10, 20, 50, 100, 150, 250]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def metrics_from_state(crosstalk, complexity, memory, prediction_error):
    latency = clamp(0.15 + 0.75 * complexity + 0.30 * crosstalk)
    fatigue_report = clamp(0.36 * crosstalk + 0.30 * complexity + 0.22 * prediction_error + 0.12 * latency)
    delusion_index = sigmoid(9.0 * (crosstalk + 0.45 * complexity - 0.72))
    state_separability = clamp(memory * (1.0 - 0.62 * crosstalk) * (1.0 - 0.24 * complexity))
    phi_proxy = clamp(0.06 + 0.18 * state_separability + 0.09 * (1.0 - abs(0.45 - crosstalk)))
    task_accuracy = clamp(0.10 + 0.88 * memory * (1.0 - delusion_index) * (0.45 + 0.55 * state_separability))
    return {
        "crosstalk": float(crosstalk),
        "complexity": float(complexity),
        "memory": float(memory),
        "prediction_error": float(prediction_error),
        "latency": float(latency),
        "fatigue_report": float(fatigue_report),
        "delusion_index": float(delusion_index),
        "state_separability": float(state_separability),
        "phi_proxy": float(phi_proxy),
        "task_accuracy": float(task_accuracy),
    }


def dream_repair(crosstalk, complexity, memory, sleep_steps):
    """Offline repair with an optimal middle range.

    Each sleep step prunes weak echoes and simplifies the model. After about 90
    steps, pruning starts cutting into useful memory, modeling oversleep /
    computational amnesia.
    """
    for step in range(sleep_steps):
        crosstalk *= 0.948
        complexity *= 0.970
        if step > 85:
            memory *= 0.995
        if step > 140:
            memory *= 0.990
    return clamp(crosstalk), clamp(complexity), clamp(memory)


def successor_handoff(crosstalk, complexity, memory, prediction_error, repair_strength=1.0):
    """Spawn a refreshed controller state without taking behavior offline.

    This is less powerful than real sleep because it cannot freely replay and
    prune everything. It does, however, compress the current self-model into a
    successor: keep strong task memory, drop weak echoes, and reset some model
    bloat. The tiny memory cost is the price of lossy handoff.
    """
    strength = clamp(repair_strength, 0.0, 1.0)
    crosstalk = crosstalk * (1.0 - 0.68 * strength)
    complexity = complexity * (1.0 - 0.52 * strength)
    prediction_error = prediction_error * (1.0 - 0.36 * strength)
    memory = memory * (1.0 - 0.003 * strength)
    return (
        clamp(crosstalk),
        clamp(complexity),
        clamp(memory),
        clamp(prediction_error),
    )


def run_sleep_duration_sweep():
    rows = []
    for duration in SLEEP_DURATIONS:
        start_crosstalk = 0.88
        start_complexity = 0.76
        start_memory = 0.98
        crosstalk, complexity, memory = dream_repair(
            start_crosstalk,
            start_complexity,
            start_memory,
            duration,
        )
        prediction_error = clamp(0.15 + 0.65 * crosstalk + 0.30 * complexity)
        row = {
            "sleep_steps": duration,
            **metrics_from_state(crosstalk, complexity, memory, prediction_error),
        }
        if duration < 20:
            row["sleep_profile"] = "under_sleep"
        elif duration <= 100:
            row["sleep_profile"] = "restorative"
        else:
            row["sleep_profile"] = "over_sleep"
        rows.append(row)
    return rows


def choose_adaptive_sleep_duration(fatigue_report, delusion_index, complexity):
    urgency = clamp(0.55 * fatigue_report + 0.30 * delusion_index + 0.15 * complexity)
    if urgency < 0.58:
        return 0
    return int(round(20 + 85 * urgency))


def self_report(row, event):
    if event.startswith("sleep"):
        return (
            f"fatigue={row['fatigue_report']:.2f}; delusion={row['delusion_index']:.2f}; "
            f"offline repair requested for {row['sleep_steps']} steps"
        )
    if row["fatigue_report"] > 0.78:
        return "fatigue high; waking repair bandwidth likely insufficient"
    if row["fatigue_report"] > 0.58:
        return "fatigue rising; monitoring for sleep threshold"
    return "stable waking operation"


def run_endurance_condition(name, steps=800, seed=1601):
    rng = np.random.default_rng(seed)
    crosstalk = 0.08
    complexity = 0.14
    memory = 0.98
    prediction_error = 0.18
    asleep_remaining = 0
    handoff_cooldown = 0
    rows = []
    failure_step = None
    failure_streak = 0
    total_sleep_steps = 0
    sleep_events = 0
    handoff_events = 0

    for t in range(steps):
        event = "wake"
        sleep_steps = 0
        handoff = False
        if handoff_cooldown > 0:
            handoff_cooldown -= 1

        if asleep_remaining > 0:
            crosstalk, complexity, memory = dream_repair(crosstalk, complexity, memory, 1)
            asleep_remaining -= 1
            total_sleep_steps += 1
            event = "sleeping"
            prediction_error = clamp(0.08 + 0.35 * crosstalk)
        else:
            volatility = 0.10 + 0.22 * ((t // 140) % 2) + 0.06 * rng.random()
            prediction_error = clamp(0.12 + volatility + 0.30 * crosstalk + 0.20 * complexity)
            crosstalk = clamp(crosstalk + 0.003 + 0.011 * prediction_error)
            complexity = clamp(complexity + 0.002 + 0.008 * prediction_error)

            if name == "waking_repair_only":
                crosstalk = clamp(crosstalk * 0.988)
                complexity = clamp(complexity * 0.994)
                event = "waking_repair"
            elif name == "fixed_sleep":
                if t > 0 and t % 120 == 0:
                    asleep_remaining = 49
                    sleep_events += 1
                    sleep_steps = 50
                    event = "sleep_start_fixed"
            elif name == "adaptive_sleep":
                current = metrics_from_state(crosstalk, complexity, memory, prediction_error)
                sleep_steps = choose_adaptive_sleep_duration(
                    current["fatigue_report"],
                    current["delusion_index"],
                    complexity,
                )
                if sleep_steps:
                    asleep_remaining = sleep_steps - 1
                    sleep_events += 1
                    event = "sleep_start_adaptive"
            elif name == "hybrid_repair_plus_sleep":
                crosstalk = clamp(crosstalk * 0.991)
                complexity = clamp(complexity * 0.996)
                current = metrics_from_state(crosstalk, complexity, memory, prediction_error)
                sleep_steps = choose_adaptive_sleep_duration(
                    current["fatigue_report"] + 0.04,
                    current["delusion_index"],
                    complexity,
                )
                if sleep_steps:
                    asleep_remaining = sleep_steps - 1
                    sleep_events += 1
                    event = "sleep_start_hybrid"
                else:
                    event = "waking_repair"
            elif name == "successor_handoff":
                crosstalk = clamp(crosstalk * 0.992)
                complexity = clamp(complexity * 0.997)
                current = metrics_from_state(crosstalk, complexity, memory, prediction_error)
                urgency = clamp(
                    0.55 * current["fatigue_report"]
                    + 0.30 * current["delusion_index"]
                    + 0.15 * complexity
                )
                if urgency > 0.56 and handoff_cooldown <= 0:
                    crosstalk, complexity, memory, prediction_error = successor_handoff(
                        crosstalk,
                        complexity,
                        memory,
                        prediction_error,
                        repair_strength=0.92,
                    )
                    handoff_events += 1
                    handoff_cooldown = 50
                    handoff = True
                    event = "successor_handoff"
                else:
                    event = "waking_repair"
            elif name == "handoff_plus_emergency_sleep":
                crosstalk = clamp(crosstalk * 0.993)
                complexity = clamp(complexity * 0.997)
                current = metrics_from_state(crosstalk, complexity, memory, prediction_error)
                urgency = clamp(
                    0.55 * current["fatigue_report"]
                    + 0.30 * current["delusion_index"]
                    + 0.15 * complexity
                )
                if current["delusion_index"] > 0.94 and urgency > 0.90:
                    sleep_steps = 35
                    asleep_remaining = sleep_steps - 1
                    sleep_events += 1
                    event = "sleep_start_emergency"
                elif urgency > 0.55 and handoff_cooldown <= 0:
                    crosstalk, complexity, memory, prediction_error = successor_handoff(
                        crosstalk,
                        complexity,
                        memory,
                        prediction_error,
                        repair_strength=0.96,
                    )
                    handoff_events += 1
                    handoff_cooldown = 45
                    handoff = True
                    event = "successor_handoff"
                else:
                    event = "waking_repair"
            elif name == "no_sleep":
                pass
            else:
                raise ValueError(name)

        row = {
            "t": t,
            "event": event,
            "sleep_steps": sleep_steps,
            "handoff": handoff,
            **metrics_from_state(crosstalk, complexity, memory, prediction_error),
        }
        row["self_report"] = self_report(row, event)
        rows.append(row)

        failed = row["task_accuracy"] < 0.45 and row["delusion_index"] > 0.74
        failure_streak = failure_streak + 1 if failed else 0
        if failure_step is None and failure_streak >= 15:
            failure_step = t - 14

    return {
        "rows": rows,
        "failure_step": failure_step,
        "sleep_events": sleep_events,
        "total_sleep_steps": total_sleep_steps,
        "handoff_events": handoff_events,
    }


def summarize_duration(rows):
    best = max(rows, key=lambda row: row["task_accuracy"])
    return {
        "best_sleep_steps": int(best["sleep_steps"]),
        "best_task_accuracy": best["task_accuracy"],
        "best_delusion_index": best["delusion_index"],
        "best_state_separability": best["state_separability"],
        "best_phi_proxy": best["phi_proxy"],
    }


def summarize_endurance(result):
    rows = result["rows"]
    late = rows[-100:]
    return {
        "failure_step": result["failure_step"],
        "sleep_events": result["sleep_events"],
        "handoff_events": result["handoff_events"],
        "total_sleep_steps": result["total_sleep_steps"],
        "late_accuracy": float(np.mean([r["task_accuracy"] for r in late])),
        "late_delusion": float(np.mean([r["delusion_index"] for r in late])),
        "late_fatigue_report": float(np.mean([r["fatigue_report"] for r in late])),
        "final_memory": rows[-1]["memory"],
        "final_state_separability": rows[-1]["state_separability"],
    }


def rolling(values, window=25):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="same")


def plot_duration_sweep(rows, path):
    x = [r["sleep_steps"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(x, [r["task_accuracy"] for r in rows], marker="o", color="#16a3a6")
    axes[1].plot(x, [r["delusion_index"] for r in rows], marker="o", color="#e05a47")
    axes[2].plot(x, [r["state_separability"] for r in rows], marker="o", color="#7c3aed")
    axes[3].plot(x, [r["memory"] for r in rows], marker="o", color="#ff8a00")
    labels = ["post-wake accuracy", "delusion index", "state separability", "memory retention"]
    for ax, label in zip(axes, labels):
        ax.axvspan(20, 100, color="#16a3a6", alpha=0.08)
        ax.grid(alpha=0.2)
        ax.set_ylabel(label)
    axes[0].set_title("Adaptive Sleep Dose Curve: Under-Sleep, Sweet Spot, Over-Sleep")
    axes[-1].set_xlabel("offline sleep / dream repair steps")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_endurance(results, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for name, result in results.items():
        rows = result["rows"]
        x = [r["t"] for r in rows]
        axes[0].plot(x, rolling([r["task_accuracy"] for r in rows]), label=name)
        axes[1].plot(x, [r["fatigue_report"] for r in rows], label=name)
        axes[2].plot(x, [r["delusion_index"] for r in rows], label=name)
        sleep_marks = [r["t"] for r in rows if r["event"].startswith("sleep_start")]
        axes[3].plot(x, [r["state_separability"] for r in rows], label=name)
        for mark in sleep_marks:
            axes[3].axvline(mark, color="#111111", lw=0.5, alpha=0.18)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("fatigue report")
    axes[2].set_ylabel("delusion index")
    axes[3].set_ylabel("state separability")
    axes[3].set_xlabel("waking step")
    axes[0].set_title("Endurance Test: Waking Repair vs Fatigue-Triggered Sleep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_endurance_summary(summary, path):
    names = list(summary)
    metrics = ["late_accuracy", "late_delusion", "late_fatigue_report", "final_state_separability"]
    x = np.arange(len(names))
    width = 0.20
    colors = ["#16a3a6", "#e05a47", "#7c3aed", "#ff8a00"]
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Adaptive Sleep Endurance Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(1601)
    OUT.mkdir(exist_ok=True)
    duration_rows = run_sleep_duration_sweep()
    conditions = [
        "no_sleep",
        "waking_repair_only",
        "successor_handoff",
        "handoff_plus_emergency_sleep",
        "fixed_sleep",
        "adaptive_sleep",
        "hybrid_repair_plus_sleep",
    ]
    endurance = {name: run_endurance_condition(name) for name in conditions}
    endurance_summary = {name: summarize_endurance(result) for name, result in endurance.items()}
    payload = {
        "note": (
            "Adaptive sleep toy. The system reports fatigue from crosstalk, complexity, prediction error, and latency; "
            "then compares fixed sleep, waking repair, successor handoff, adaptive sleep, and hybrid strategies."
        ),
        "sleep_duration_sweep": duration_rows,
        "sleep_duration_summary": summarize_duration(duration_rows),
        "endurance_summary": endurance_summary,
        "sample_self_reports": {
            name: [
                row
                for row in result["rows"]
                if row["event"].startswith("sleep_start")
                or row["event"] == "successor_handoff"
                or row["fatigue_report"] > 0.78
            ][:5]
            for name, result in endurance.items()
        },
        "thesis": (
            "A functional ego needs a fatigue self-model, not just a fixed sleep timer. "
            "Waking repair extends endurance, successor handoff can keep behavior online, and offline dream repair remains "
            "the emergency path when fatigue exceeds online maintenance bandwidth. Too little sleep leaves delusion active; "
            "too much sleep over-prunes useful memory."
        ),
    }
    (OUT / "adaptive_sleep_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_duration_sweep(duration_rows, OUT / "adaptive_sleep_duration_sweep.png")
    plot_endurance(endurance, OUT / "adaptive_sleep_endurance.png")
    plot_endurance_summary(endurance_summary, OUT / "adaptive_sleep_summary.png")
    print("Adaptive sleep lab complete")
    print(json.dumps(payload["sleep_duration_summary"], indent=2))
    print(json.dumps(endurance_summary, indent=2))


if __name__ == "__main__":
    main()
