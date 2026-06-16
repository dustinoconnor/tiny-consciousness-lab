#!/usr/bin/env python3
"""Long-run sleep cycle agent experiment.

This extends `sleep_homeostasis_lab.py` from a one-shot maintenance test into a
500-step behavioral run.

Conditions:

- `no_sleep`: continuous online operation; dense recurrent crosstalk accumulates
- `offline_sleep`: every 100 steps, sensory/action processing is paused and the
  recurrent matrix is down-selected
- `active_dreaming`: the agent never goes offline, but a small background
  cleanup process dampens weak recurrent loops every step

The goal is not to claim biological sleep was recreated. The goal is to test a
functional maintenance question:

Can always-on background repair match the restoration provided by an explicit
offline pruning phase?
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, systems
from sleep_homeostasis_lab import fatigue_step, sleep_downselect, transition_metrics


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def generate_endless_world(steps=500, shift_period=85, seed=409):
    """Generate a 4-action endless rule-shift world."""
    rng = np.random.default_rng(seed)
    sense = []
    target = []
    current = int(rng.integers(0, 4))
    for t in range(steps):
        if rng.random() < 0.75:
            current = int((current + rng.choice([-1, 0, 1])) % 4)
        else:
            current = int(rng.integers(0, 4))
        rule = (t // shift_period) % 2
        sense.append(current)
        target.append(current if rule == 0 else (current + 2) % 4)
    return np.array(sense), np.array(target)


def active_background_repair(weights, bias):
    """Small always-on cleanup pass.

    This is the non-offline alternative: every active step, slightly damp all
    weights, then damp the weaker recurrent edges a bit more. It preserves
    operation, but cannot perform as aggressive a reset as offline sleep.
    """
    repaired = 0.997 * weights.copy()
    repaired_bias = 0.996 * bias.copy()
    for row in range(repaired.shape[0]):
        weak = np.argsort(np.abs(repaired[row]))[:-2]
        repaired[row, weak] *= 0.985
    return repaired, repaired_bias


def run_condition(name, steps=500, shift_period=85, seed=409):
    weights, bias = systems(n=4)["recurrent_with_valence_feedback"]
    baseline = transition_metrics(weights, bias)
    base_phi = baseline["phi_proxy"]
    base_sep = baseline["state_separability"]

    sense, target = generate_endless_world(steps=steps, shift_period=shift_period, seed=seed)
    rule_belief = np.array([0.90, 0.10], dtype=float)
    stale_rule = 0
    reward_ema = 0.90
    crosstalk = 0.0
    latest_metrics = baseline
    rows = []
    samples = []

    for t in range(steps):
        if t % 5 == 0:
            latest_metrics = transition_metrics(weights, bias)
        sep_ratio = latest_metrics["state_separability"] / base_sep
        crosstalk = max(0.0, min(1.2, crosstalk + 0.003 + 0.012 * (1.0 - sep_ratio)))
        delusion = sigmoid(10.0 * (crosstalk - 0.52))

        effective_belief = (1.0 - delusion) * rule_belief + delusion * np.eye(2)[stale_rule]
        predicted_rule = int(np.argmax(effective_belief))
        action = int(sense[t] if predicted_rule == 0 else (sense[t] + 2) % 4)
        correct = action == int(target[t])
        reward = 1.0 if correct else 0.0
        reward_ema = 0.96 * reward_ema + 0.04 * reward

        true_rule = 0 if target[t] == sense[t] else 1
        evidence = np.array([1.0 if true_rule == 0 else 0.05, 1.0 if true_rule == 1 else 0.05])
        evidence = evidence / evidence.sum()
        learning_rate = 0.04 + 0.12 * (1.0 - reward_ema)
        if delusion > 0.60:
            learning_rate *= 0.25
        rule_belief = (1.0 - learning_rate) * rule_belief + learning_rate * evidence
        rule_belief = rule_belief / rule_belief.sum()
        if correct and reward_ema > 0.75:
            stale_rule = predicted_rule

        weights, bias = fatigue_step(weights, bias, echo_gain=0.006, scale=1.002, bias_drift=0.004)
        maintenance_event = "none"
        if name == "offline_sleep" and (t + 1) % 100 == 0:
            weights, bias = sleep_downselect(weights, bias)
            crosstalk *= 0.25
            reward_ema = 0.75 * reward_ema + 0.25 * 0.90
            maintenance_event = "offline_sleep"
        elif name == "active_dreaming":
            weights, bias = active_background_repair(weights, bias)
            crosstalk *= 0.985
            maintenance_event = "background_repair"
        elif name not in {"no_sleep", "offline_sleep"}:
            raise ValueError(name)

        rows.append(
            {
                "t": t,
                "shift_epoch": int(t // shift_period),
                "true_rule": int(true_rule),
                "predicted_rule": int(predicted_rule),
                "correct": float(correct),
                "reward_ema": float(reward_ema),
                "delusion": float(delusion),
                "crosstalk": float(crosstalk),
                "rule_0_belief": float(rule_belief[0]),
                "rule_1_belief": float(rule_belief[1]),
                "maintenance_event": maintenance_event,
            }
        )

        if (t + 1) % 50 == 0:
            sample = transition_metrics(weights, bias)
            samples.append({"t": t + 1, **sample})

    return {"rows": rows, "samples": samples, "baseline": baseline}


def summarize(result):
    rows = result["rows"]
    early = rows[:100]
    late = rows[-100:]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk]))

    return {
        "early_accuracy": mean(early, "correct"),
        "late_accuracy": mean(late, "correct"),
        "late_delusion": mean(late, "delusion"),
        "late_crosstalk": mean(late, "crosstalk"),
        "final_phi_proxy": float(result["samples"][-1]["phi_proxy"]),
        "final_state_separability": float(result["samples"][-1]["state_separability"]),
        "maintenance_events": int(sum(1 for r in rows if r["maintenance_event"] != "none")),
    }


def rolling(values, window=25):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="same")


def plot_behavior(results, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    for name, result in results.items():
        rows = result["rows"]
        x = [r["t"] for r in rows]
        axes[0].plot(x, rolling([r["correct"] for r in rows]), label=name)
        axes[1].plot(x, [r["delusion"] for r in rows], label=name)
        axes[2].plot(x, [r["crosstalk"] for r in rows], label=name)
        axes[3].plot(x, [r["reward_ema"] for r in rows], label=name)
    for ax in axes:
        for step in range(100, 500, 100):
            ax.axvline(step, color="#111111", lw=0.8, ls="--", alpha=0.35)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("delusion")
    axes[2].set_ylabel("crosstalk")
    axes[3].set_ylabel("reward EMA")
    axes[3].set_xlabel("step")
    axes[0].set_title("Long-Run Cognitive Maintenance: No Sleep vs Offline Sleep vs Active Dreaming")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phi_samples(results, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, result in results.items():
        samples = result["samples"]
        x = [s["t"] for s in samples]
        axes[0].plot(x, [s["phi_proxy"] for s in samples], marker="o", label=name)
        axes[1].plot(x, [s["state_separability"] for s in samples], marker="o", label=name)
    for ax in axes:
        for step in range(100, 500, 100):
            ax.axvline(step, color="#111111", lw=0.8, ls="--", alpha=0.35)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Phi proxy")
    axes[1].set_ylabel("state separability")
    axes[1].set_xlabel("step")
    axes[0].set_title("Sampled Integration Metrics During Long Run")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["late_accuracy", "late_delusion", "final_phi_proxy", "final_state_separability"]
    x = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#16a3a6", "#e05a47", "#7c3aed", "#ff8a00"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Maintenance Strategy Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    conditions = ["no_sleep", "offline_sleep", "active_dreaming"]
    results = {name: run_condition(name) for name in conditions}
    summary = {name: summarize(result) for name, result in results.items()}
    payload = {
        "note": (
            "500-step behavioral sleep-cycle toy. No-sleep accumulates dense recurrent crosstalk; "
            "offline sleep pauses every 100 steps for down-selection; active dreaming performs small always-on repair."
        ),
        "summary": summary,
        "samples": {name: result["samples"] for name, result in results.items()},
        "thesis": (
            "Always-on repair helps, but in this toy it does not fully match offline sleep. "
            "The explicit offline phase gives the system permission to prune more aggressively without corrupting active behavior."
        ),
    }
    (OUT / "sleep_cycle_agent_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_behavior(results, OUT / "sleep_cycle_agent_behavior.png")
    plot_phi_samples(results, OUT / "sleep_cycle_agent_phi_samples.png")
    plot_summary(summary, OUT / "sleep_cycle_agent_summary.png")
    print("Sleep cycle agent lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
