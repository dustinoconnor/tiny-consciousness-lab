#!/usr/bin/env python3
"""Biological control motifs as toy AI architecture tests.

This file adds three neuroscience-inspired control mechanisms to the lab:

1. Low-road threat hijack:
   A fast feedforward veto bypasses the slow workspace when an instant hazard
   appears.

2. Inhibitory action gate:
   A basal-ganglia-like winner-take-all gate turns blended workspace proposals
   into crisp action.

3. Neuromodulation fluid:
   Dopamine/norepinephrine/acetylcholine-like global scalars alter learning,
   attention width, and update speed when the world becomes chaotic.

These are not biological models. They are small architecture probes that ask
whether a brain-like control motif solves a failure exposed by earlier labs.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float) / max(temperature, 1e-6)
    x -= np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def run_low_road_hijack(steps=300, seed=701):
    """Test whether a fast somatic veto saves the agent from instant hazards."""
    rng = np.random.default_rng(seed)
    conditions = ["slow_workspace_only", "low_road_veto", "overactive_veto"]
    results = {}
    for condition in conditions:
        rows = []
        pending_master = []
        for t in range(steps):
            hazard = rng.random() < 0.18
            false_alarm = (not hazard) and rng.random() < 0.08
            goal_signal = rng.normal(0.75, 0.25)
            threat_signature = 1.0 if hazard else 0.0
            if false_alarm:
                threat_signature = 0.65

            # Slow workspace needs two frames to serialize the threat summary.
            pending_master.append(threat_signature)
            workspace_threat = pending_master.pop(0) if len(pending_master) > 2 else 0.0
            workspace_policy = "evade" if workspace_threat > 0.5 else "advance"

            low_road_fired = False
            if condition == "low_road_veto":
                low_road_fired = threat_signature > 0.82
            elif condition == "overactive_veto":
                low_road_fired = threat_signature > 0.45
            elif condition != "slow_workspace_only":
                raise ValueError(condition)

            action = "evade" if low_road_fired else workspace_policy
            survived = not (hazard and action != "evade")
            progress = (not hazard) and action == "advance" and goal_signal > 0.15
            cost = 0.06 if low_road_fired else 0.02
            if action == "evade" and not hazard:
                cost += 0.16

            rows.append(
                {
                    "t": t,
                    "hazard": float(hazard),
                    "false_alarm": float(false_alarm),
                    "low_road_fired": float(low_road_fired),
                    "survived": float(survived),
                    "progress": float(progress),
                    "cost": float(cost),
                    "utility": float(survived) + 0.35 * float(progress) - cost,
                }
            )
        results[condition] = rows
    return results


def summarize_low_road(results):
    summary = {}
    for name, rows in results.items():
        hazards = [r for r in rows if r["hazard"]]
        nonhazards = [r for r in rows if not r["hazard"]]
        summary[name] = {
            "survival_rate_on_hazards": float(np.mean([r["survived"] for r in hazards])),
            "false_veto_rate": float(np.mean([r["low_road_fired"] for r in nonhazards])),
            "progress_rate": float(np.mean([r["progress"] for r in rows])),
            "mean_utility": float(np.mean([r["utility"] for r in rows])),
        }
    return summary


def run_inhibitory_gate(trials=360, seed=802):
    """Test whether winner-take-all action gating reduces jitter."""
    rng = np.random.default_rng(seed)
    conditions = ["blended_policy", "inhibitory_gate", "overclamped_gate"]
    actions = np.arange(4)
    results = {}
    for condition in conditions:
        rows = []
        previous_action = None
        for t in range(trials):
            desired = int((t // 45) % 4)
            ambiguity = 0.35 + 0.50 * (rng.random() < 0.35)
            workspace_a = np.full(4, -0.35)
            workspace_b = np.full(4, -0.35)
            workspace_a[desired] = 1.0
            workspace_b[(desired + 1) % 4] = 0.85 * ambiguity
            workspace_b[desired] += 0.40 * (1.0 - ambiguity)
            proposal = 0.55 * workspace_a + 0.45 * workspace_b + rng.normal(0.0, 0.10, size=4)

            if condition == "blended_policy":
                probs = softmax(proposal, temperature=1.15)
                action = int(rng.choice(actions, p=probs))
                decisiveness = float(np.max(probs) - np.partition(probs, -2)[-2])
            elif condition == "inhibitory_gate":
                gate = softmax(5.0 * proposal, temperature=1.0)
                action = int(np.argmax(gate))
                decisiveness = float(np.max(gate) - np.partition(gate, -2)[-2])
            elif condition == "overclamped_gate":
                # Too much inhibition creates freezing unless one proposal is
                # overwhelmingly stronger than the others.
                gate = softmax(8.0 * proposal, temperature=1.0)
                margin = float(np.max(proposal) - np.partition(proposal, -2)[-2])
                action = int(np.argmax(gate)) if margin > 0.55 else -1
                decisiveness = margin
            else:
                raise ValueError(condition)

            correct = action == desired
            jitter = previous_action is not None and action != previous_action and t % 45 != 0
            previous_action = action
            rows.append(
                {
                    "t": t,
                    "desired": desired,
                    "action": action,
                    "correct": float(correct),
                    "jitter": float(jitter),
                    "freeze": float(action == -1),
                    "decisiveness": float(decisiveness),
                }
            )
        results[condition] = rows
    return results


def summarize_inhibitory(results):
    summary = {}
    for name, rows in results.items():
        summary[name] = {
            "accuracy": float(np.mean([r["correct"] for r in rows])),
            "jitter_rate": float(np.mean([r["jitter"] for r in rows])),
            "freeze_rate": float(np.mean([r["freeze"] for r in rows])),
            "mean_decisiveness": float(np.mean([r["decisiveness"] for r in rows])),
        }
    return summary


def run_neuromodulation(steps=360, rewrite_step=150, chaos_step=240, seed=903):
    """Test dynamic global chemistry under predictable-to-chaotic shifts."""
    rng = np.random.default_rng(seed)
    conditions = ["static_params", "fluid_chemistry"]
    results = {}
    for condition in conditions:
        belief = np.array([0.84, 0.16], dtype=float)
        reward_ema = 0.82
        prediction_error = 0.20
        rows = []
        for t in range(steps):
            if t < rewrite_step:
                true_rule = 0
                noise = 0.05
            elif t < chaos_step:
                true_rule = 1
                noise = 0.15
            else:
                true_rule = int((t // 14) % 2)
                noise = 0.34

            surprise = abs(float(belief[true_rule]) - 1.0) + noise
            reward_prediction_error = surprise * (1.0 - reward_ema)

            if condition == "fluid_chemistry":
                dopamine = sigmoid(8.0 * (reward_prediction_error - 0.08))
                norepinephrine = sigmoid(7.0 * (surprise - 0.38))
                acetylcholine = sigmoid(6.0 * (prediction_error - 0.22))
                learning_rate = 0.025 + 0.28 * dopamine + 0.12 * acetylcholine
                attention_temperature = 0.65 + 1.10 * norepinephrine
            elif condition == "static_params":
                dopamine = 0.35
                norepinephrine = 0.35
                acetylcholine = 0.35
                learning_rate = 0.07
                attention_temperature = 1.0
            else:
                raise ValueError(condition)

            evidence = np.array([1.0 if true_rule == 0 else 0.05, 1.0 if true_rule == 1 else 0.05])
            evidence += rng.normal(0.0, noise, size=2)
            evidence = np.clip(evidence, 0.01, None)
            evidence = evidence / evidence.sum()
            attended = softmax(np.log(np.clip(evidence, 1e-6, 1.0)), temperature=attention_temperature)
            belief = (1.0 - learning_rate) * belief + learning_rate * attended
            belief = belief / belief.sum()

            action_rule = int(np.argmax(belief))
            correct = action_rule == true_rule
            reward = 1.0 if correct else 0.0
            reward_ema = 0.93 * reward_ema + 0.07 * reward
            prediction_error = 0.88 * prediction_error + 0.12 * (1.0 - belief[true_rule])

            rows.append(
                {
                    "t": t,
                    "phase": "stable" if t < rewrite_step else "rewrite" if t < chaos_step else "chaotic",
                    "true_rule": true_rule,
                    "action_rule": action_rule,
                    "correct": float(correct),
                    "belief_true_rule": float(belief[true_rule]),
                    "reward_ema": float(reward_ema),
                    "dopamine": float(dopamine),
                    "norepinephrine": float(norepinephrine),
                    "acetylcholine": float(acetylcholine),
                    "learning_rate": float(learning_rate),
                    "attention_temperature": float(attention_temperature),
                    "prediction_error": float(prediction_error),
                }
            )
        results[condition] = rows
    return results


def summarize_neuromod(results):
    summary = {}
    for name, rows in results.items():
        stable = [r for r in rows if r["phase"] == "stable"]
        rewrite = [r for r in rows if r["phase"] == "rewrite"]
        chaotic = [r for r in rows if r["phase"] == "chaotic"]
        summary[name] = {
            "stable_accuracy": float(np.mean([r["correct"] for r in stable])),
            "rewrite_accuracy": float(np.mean([r["correct"] for r in rewrite])),
            "chaotic_accuracy": float(np.mean([r["correct"] for r in chaotic])),
            "mean_learning_rate": float(np.mean([r["learning_rate"] for r in rows])),
            "mean_norepinephrine": float(np.mean([r["norepinephrine"] for r in rows])),
        }
    return summary


def plot_bar_group(summary, metrics, title, path):
    names = list(summary)
    x = np.arange(len(names))
    width = 0.78 / len(metrics)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#16a3a6", "#ff8a00", "#7c3aed", "#e05a47", "#64748b"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i % len(colors)])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_neuromod_timeseries(results, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, [r["correct"] for r in rows], alpha=0.35, label=f"{name} raw")
        axes[0].plot(x, np.convolve([r["correct"] for r in rows], np.ones(20) / 20, mode="same"), lw=2, label=name)
        axes[1].plot(x, [r["learning_rate"] for r in rows], label=name)
        axes[2].plot(x, [r["norepinephrine"] for r in rows], label=name)
        axes[3].plot(x, [r["belief_true_rule"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(150, color="#111111", ls="--", lw=1)
        ax.axvline(240, color="#111111", ls=":", lw=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("learning rate")
    axes[2].set_ylabel("alert/widening")
    axes[3].set_ylabel("true-rule belief")
    axes[3].set_xlabel("step")
    axes[0].set_title("Neuromodulation Fluid: Dynamic Internal Physics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(701)
    OUT.mkdir(exist_ok=True)

    low_road = run_low_road_hijack()
    inhibitory = run_inhibitory_gate()
    neuromod = run_neuromodulation()

    low_road_summary = summarize_low_road(low_road)
    inhibitory_summary = summarize_inhibitory(inhibitory)
    neuromod_summary = summarize_neuromod(neuromod)

    plot_bar_group(
        low_road_summary,
        ["survival_rate_on_hazards", "false_veto_rate", "progress_rate", "mean_utility"],
        "Low-Road Threat Hijack",
        OUT / "biological_low_road_summary.png",
    )
    plot_bar_group(
        inhibitory_summary,
        ["accuracy", "jitter_rate", "freeze_rate", "mean_decisiveness"],
        "Inhibitory Action Gate",
        OUT / "biological_inhibitory_gate_summary.png",
    )
    plot_bar_group(
        neuromod_summary,
        ["stable_accuracy", "rewrite_accuracy", "chaotic_accuracy", "mean_learning_rate"],
        "Neuromodulation Fluid",
        OUT / "biological_neuromod_summary.png",
    )
    plot_neuromod_timeseries(neuromod, OUT / "biological_neuromod_timeseries.png")

    payload = {
        "note": (
            "Three neuroscience-inspired toy control motifs: low-road threat veto, inhibitory winner-take-all action gate, "
            "and neuromodulator-like dynamic global variables."
        ),
        "low_road_threat_hijack": low_road_summary,
        "inhibitory_action_gate": inhibitory_summary,
        "neuromodulation_fluid": neuromod_summary,
        "thesis": (
            "Biological control motifs solve distinct architecture problems: fast vetoes protect slow awareness, "
            "inhibitory gates sharpen action selection, and fluid global variables retune learning and attention when the world changes."
        ),
    }
    (OUT / "biological_control_metrics.json").write_text(json.dumps(payload, indent=2))
    print("Biological control lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
