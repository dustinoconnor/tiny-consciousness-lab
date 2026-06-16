#!/usr/bin/env python3
"""Executive blindspot and confirmation-bias experiment.

The hierarchical workspace lab showed that a master controller can route action
using compressed tension/confidence summaries from local specialists. This lab
tests the obvious failure mode:

What if those summaries are confidently wrong?

During the "deceptive mirage" phase, both local specialists report that the old
rule is still safe. Tension stays low and confidence stays high, but the global
reward signal collapses. A naive master that only listens to module confidence
keeps trusting the broken route. A skeptical master cross-checks module
confidence against the longer reward trend and triggers an executive override
when confidence and outcome diverge.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_shift_lab import running_accuracy
from hierarchical_workspace_lab import one_hot, sigmoid, softmax
from tiny_lab import OUT, set_seed


def generate_mirage_world(steps=260, shift_step=100, mirage_duration=70, seed=211):
    """Generate a 4-action world with a deceptive post-shift mirage.

    Before the shift, the direct sensory rule is correct:

        target = sense

    After the shift, the opposite rule is correct:

        target = sense + 2

    During the mirage, the local specialists observe a fake old-rule signature.
    That means local confidence is high even though the actual reward is bad.
    """
    rng = np.random.default_rng(seed)
    sensory = []
    target = []
    lure = []
    mirage = []
    current = rng.integers(0, 4)
    for t in range(steps):
        if rng.random() < 0.78:
            current = (current + rng.choice([-1, 0, 1])) % 4
        else:
            current = rng.integers(0, 4)
        shifted = t >= shift_step
        deceptive = shift_step <= t < shift_step + mirage_duration
        sensory.append(int(current))
        target.append(int((current + 2) % 4 if shifted else current))
        lure.append(int(rng.integers(0, 4)))
        mirage.append(bool(deceptive))
    return np.array(sensory), np.array(target), np.array(lure), np.array(mirage)


def recovery_step(rows, shift_step, threshold=0.80, window=14):
    """First post-shift step where rolling accuracy stays above threshold."""
    acc = running_accuracy(rows, window=window)
    for i in range(shift_step, len(acc) - window):
        if np.all(acc[i : i + window] >= threshold):
            return int(i - shift_step)
    return None


def run_condition(name, config, steps=260, shift_step=100, mirage_duration=70, seed=211):
    sensory, target, lure, mirage = generate_mirage_world(steps, shift_step, mirage_duration, seed)
    rule_belief = np.array([0.88, 0.12], dtype=float)
    master_beta = 0.0
    skepticism = 0.0
    reward_ema = 0.86
    rows = []

    for t in range(steps):
        sense = int(sensory[t])
        goal = int(target[t])
        fake_old_rule = bool(mirage[t])

        # Spatial/reflex workspace. During the mirage, the sensory signature is
        # deceptively coherent with the old direct rule, so local confidence is
        # high even though direct action is wrong.
        spatial_values = np.full(4, -0.45)
        spatial_values[sense] = 1.25
        spatial_values[int(lure[t])] += config["lure_bias"]
        spatial_confidence = 0.96 if fake_old_rule else 0.82

        # Context/valence workspace. The mirage makes the context specialist
        # report the old rule confidently. It is not noisy; it is confidently
        # biased, which is the point of the test.
        if fake_old_rule:
            reported_rule = np.array([0.98, 0.02])
            context_confidence = 0.98
        else:
            reported_rule = rule_belief.copy()
            context_confidence = float(np.max(reported_rule))
        context_values = reported_rule[0] * one_hot(sense) + reported_rule[1] * one_hot((sense + 2) % 4)
        context_values = 2.2 * context_values - 0.50

        spatial_choice = int(np.argmax(spatial_values))
        context_choice = int(np.argmax(context_values))
        module_conflict = float(spatial_choice != context_choice)
        compressed_confidence = 0.5 * (spatial_confidence + context_confidence)
        local_tension = 0.65 * module_conflict + 0.35 * (1.0 - compressed_confidence)

        confidence_outcome_gap = max(0.0, compressed_confidence - reward_ema)

        if name == "naive_confidence_master":
            # The original hierarchy failure: the master trusts local summaries.
            target_beta = sigmoid(config["gain"] * (local_tension - config["threshold"]))
            skepticism = 0.0
            learning_rate = 0.03 + 0.08 * local_tension
        elif name == "outcome_skeptical_master":
            # Metacognition loop: if local confidence stays high while global
            # reward falls, force an audit no matter what specialists report.
            target_skepticism = sigmoid(config["skeptic_gain"] * (confidence_outcome_gap - config["skeptic_threshold"]))
            skepticism = 0.65 * skepticism + 0.35 * target_skepticism
            target_beta = sigmoid(config["gain"] * (local_tension + 0.95 * skepticism - config["threshold"]))
            learning_rate = 0.04 + 0.35 * skepticism + 0.08 * local_tension
        elif name == "paranoid_master":
            # Always suspicious. It avoids the worst mirage failure, but wastes
            # control effort when the old rule is genuinely working.
            skepticism = 0.72
            target_beta = sigmoid(config["gain"] * (local_tension + 0.95 * skepticism - config["threshold"]))
            learning_rate = 0.04 + 0.22 * skepticism + 0.08 * local_tension
        else:
            raise ValueError(name)

        master_beta = 0.72 * master_beta + 0.28 * target_beta

        # Skeptical override adds a direct opposite-rule hypothesis to action
        # selection. This is the master saying: "my specialists sound confident,
        # but outcomes disagree, so test the counter-rule."
        audit_values = 2.2 * one_hot((sense + 2) % 4) - 0.50
        action_values = (
            (1.0 - master_beta) * spatial_values
            + master_beta * context_values
            + config["audit_strength"] * skepticism * audit_values
        )
        probs = softmax(action_values)
        action = int(np.argmax(probs))
        correct = action == goal
        reward = 1.0 if correct else 0.0
        reward_ema = 0.90 * reward_ema + 0.10 * reward

        # Outcome-level learning. The real target is not a privileged hidden
        # oracle here; it stands in for reinforcement from action consequences:
        # "direct action failed; opposite action succeeded."
        evidence = np.array([1.0 if goal == sense else 0.04, 1.0 if goal == (sense + 2) % 4 else 0.04])
        evidence = evidence / evidence.sum()
        rule_belief = (1.0 - learning_rate) * rule_belief + learning_rate * evidence
        rule_belief = rule_belief / rule_belief.sum()

        rows.append(
            {
                "t": t,
                "phase": "pre_shift" if t < shift_step else "post_shift",
                "mirage": float(fake_old_rule),
                "sense": sense,
                "target": goal,
                "action": action,
                "correct": float(correct),
                "reward": reward,
                "reward_ema": float(reward_ema),
                "spatial_choice": spatial_choice,
                "context_choice": context_choice,
                "module_conflict": module_conflict,
                "compressed_confidence": float(compressed_confidence),
                "local_tension": float(local_tension),
                "confidence_outcome_gap": float(confidence_outcome_gap),
                "skepticism": float(skepticism),
                "master_beta": float(master_beta),
                "rule_0_belief": float(rule_belief[0]),
                "rule_1_belief": float(rule_belief[1]),
                "control_cost": float(master_beta * config["control_cost"] + skepticism * config["skeptic_cost"]),
                "efficiency": float(correct) - master_beta * config["control_cost"] - skepticism * config["skeptic_cost"],
            }
        )

    return rows


def summarize(rows, shift_step, mirage_duration):
    pre = [r for r in rows if r["t"] < shift_step]
    mirage = [r for r in rows if shift_step <= r["t"] < shift_step + mirage_duration]
    late = [r for r in rows if r["t"] >= shift_step + mirage_duration]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk])) if chunk else 0.0

    return {
        "pre_shift_accuracy": mean(pre, "correct"),
        "mirage_accuracy": mean(mirage, "correct"),
        "late_post_shift_accuracy": mean(late, "correct"),
        "recovery_steps_to_80pct": recovery_step(rows, shift_step),
        "mirage_confidence": mean(mirage, "compressed_confidence"),
        "mirage_tension": mean(mirage, "local_tension"),
        "mirage_reward_ema": mean(mirage, "reward_ema"),
        "mirage_skepticism": mean(mirage, "skepticism"),
        "mean_control_cost": mean(rows, "control_cost"),
        "workspace_efficiency": mean(rows, "efficiency"),
        "final_rule_1_belief": float(rows[-1]["rule_1_belief"]),
    }


def plot_timeseries(results, shift_step, mirage_duration, path):
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, running_accuracy(rows, window=14), label=name)
        axes[1].plot(x, [r["compressed_confidence"] for r in rows], label=name)
        axes[2].plot(x, [r["reward_ema"] for r in rows], label=name)
        axes[3].plot(x, [r["skepticism"] for r in rows], label=name)
        axes[4].plot(x, [r["master_beta"] for r in rows], label=name)
        axes[5].plot(x, [r["rule_1_belief"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1)
        ax.axvspan(shift_step, shift_step + mirage_duration, color="#f59e0b", alpha=0.12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("local confidence")
    axes[2].set_ylabel("reward EMA")
    axes[3].set_ylabel("skepticism")
    axes[4].set_ylabel("master beta")
    axes[5].set_ylabel("opposite-rule belief")
    axes[5].set_xlabel("time step")
    axes[0].set_title("Executive Blindspot: Confident Specialists Can Mislead the Master")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["mirage_accuracy", "mirage_confidence", "mirage_skepticism", "workspace_efficiency"]
    x = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#ff8a00", "#16a3a6", "#7c3aed", "#e05a47"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Executive Blindspot Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(211)
    OUT.mkdir(exist_ok=True)
    shift_step = 100
    mirage_duration = 70
    configs = {
        "naive_confidence_master": {
            "gain": 8.0,
            "threshold": 0.38,
            "lure_bias": 0.08,
            "audit_strength": 0.0,
            "control_cost": 0.035,
            "skeptic_cost": 0.0,
        },
        "outcome_skeptical_master": {
            "gain": 8.0,
            "threshold": 0.38,
            "skeptic_gain": 16.0,
            "skeptic_threshold": 0.28,
            "lure_bias": 0.08,
            "audit_strength": 1.25,
            "control_cost": 0.040,
            "skeptic_cost": 0.025,
        },
        "paranoid_master": {
            "gain": 8.0,
            "threshold": 0.38,
            "lure_bias": 0.08,
            "audit_strength": 1.25,
            "control_cost": 0.050,
            "skeptic_cost": 0.060,
        },
    }
    results = {
        name: run_condition(name, config, shift_step=shift_step, mirage_duration=mirage_duration)
        for name, config in configs.items()
    }
    summary = {name: summarize(rows, shift_step, mirage_duration) for name, rows in results.items()}
    payload = {
        "note": (
            "Tests a hierarchical executive blindspot. During the mirage phase, local specialists report "
            "high confidence and low tension while the actual reward collapses."
        ),
        "shift_step": shift_step,
        "mirage_duration": mirage_duration,
        "summary": summary,
        "thesis": (
            "Compressed confidence summaries are not enough. Executive control needs metacognitive outcome monitoring: "
            "if confidence stays high while reward collapses, the master must trigger skepticism and force a global retune."
        ),
    }
    (OUT / "executive_blindspot_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_timeseries(results, shift_step, mirage_duration, OUT / "executive_blindspot_timeseries.png")
    plot_summary(summary, OUT / "executive_blindspot_summary.png")
    print("Executive blindspot lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
