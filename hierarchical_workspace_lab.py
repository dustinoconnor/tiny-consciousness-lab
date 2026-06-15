#!/usr/bin/env python3
"""Hierarchical workspace experiment.

This lab asks whether multiple local workspaces plus a higher-level controller
help, or whether they create latency and architectural noise.

Brain analogy:

- spatial/reflex workspace: like local sensorimotor cortex, fast but myopic
- context/valence workspace: like a rule/context system tracking which policy
  is currently rewarded
- master workspace: a prefrontal-like coordinator that does not read all raw
  sensory detail; it reads compressed tension/confidence summaries and decides
  which local workspace should control action

The world is deliberately simple: the correct action rule flips halfway through.
Before the shift, the current sensory direction predicts the next action. After
the shift, the opposite context rule is correct. The diagnostic is recovery
latency after the context flip.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from attention_shift_lab import running_accuracy
from tiny_lab import OUT, set_seed


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def generate_context_world(steps=240, shift_step=120, seed=151):
    """Generate a 4-action world with a hidden context-rule flip."""
    rng = np.random.default_rng(seed)
    sensory = []
    correct = []
    distractor = []
    current = rng.integers(0, 4)
    for t in range(steps):
        if rng.random() < 0.72:
            current = (current + rng.choice([-1, 0, 1])) % 4
        else:
            current = rng.integers(0, 4)
        context = 0 if t < shift_step else 1
        target = current if context == 0 else (current + 2) % 4
        sensory.append(current)
        correct.append(target)
        distractor.append(rng.integers(0, 4))
    return np.array(sensory), np.array(correct), np.array(distractor)


def one_hot(i, n=4):
    v = np.zeros(n)
    v[int(i)] = 1.0
    return v


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def recovery_step(rows, shift_step, threshold=0.82, window=14):
    acc = running_accuracy(rows, window=window)
    for i in range(shift_step, len(acc) - window):
        if np.all(acc[i : i + window] >= threshold):
            return int(i - shift_step)
    return None


def run_condition(name, config, steps=240, shift_step=120, seed=151):
    sensory, correct, distractor = generate_context_world(steps=steps, shift_step=shift_step, seed=seed)
    context_belief = np.array([0.85, 0.15])
    context_workspace = np.array([0.85, 0.15])
    spatial_confidence = 0.78
    context_confidence = 0.55
    master_beta = 0.0
    rows = []

    for t in range(steps):
        sense = int(sensory[t])
        target = int(correct[t])
        lure = int(distractor[t])

        spatial_values = np.full(4, -0.35)
        spatial_values[sense] = 1.0
        spatial_values[lure] += config["sensory_noise_bias"]

        context_values = context_workspace[0] * one_hot(sense) + context_workspace[1] * one_hot((sense + 2) % 4)
        context_values = 2.0 * context_values - 0.45

        spatial_choice = int(np.argmax(spatial_values))
        context_choice = int(np.argmax(context_values))
        spatial_correct = spatial_choice == target
        context_correct = context_choice == target

        spatial_error = 0.0 if spatial_correct else 1.0
        context_error = 0.0 if context_correct else 1.0
        module_conflict = 1.0 if spatial_choice != context_choice else 0.0

        if name == "monolithic_workspace":
            # One big controller blends everything directly. It adapts, but the
            # context update is diluted by raw sensory pull.
            beta = sigmoid(4.0 * (context_error + module_conflict - 0.75))
            action_values = (1.0 - beta) * spatial_values + beta * context_values
            learning_rate = 0.05 + 0.16 * context_error
        elif name == "flat_multi_workspace":
            # Two local workspaces vote without a master. Disagreement creates
            # indecision and can slow recovery.
            beta = 0.50
            action_values = 0.50 * spatial_values + 0.50 * context_values
            learning_rate = 0.04 + 0.06 * context_error
            if module_conflict:
                action_values *= 0.75
        elif name == "hierarchical_master_workspace":
            # The master reads compressed uncertainty, not raw sensory detail.
            # It raises context control when spatial confidence fails or module
            # conflict rises after the rule shift.
            tension = 0.45 * module_conflict + 0.35 * spatial_error + 0.20 * (1.0 - context_confidence)
            target_beta = sigmoid(8.0 * (tension - 0.38))
            master_beta = 0.70 * master_beta + 0.30 * target_beta
            beta = master_beta
            action_values = (1.0 - beta) * spatial_values + beta * context_values
            learning_rate = 0.05 + 0.38 * spatial_error * beta + 0.10 * module_conflict
        elif name == "bad_hierarchy_bureaucracy":
            # Same hierarchy, but with a slow master. This models propagation
            # delay / committee latency.
            tension = 0.45 * module_conflict + 0.35 * spatial_error + 0.20 * (1.0 - context_confidence)
            target_beta = sigmoid(8.0 * (tension - 0.38))
            master_beta = 0.94 * master_beta + 0.06 * target_beta
            beta = master_beta
            action_values = (1.0 - beta) * spatial_values + beta * context_values
            learning_rate = 0.03 + 0.12 * spatial_error * beta
        else:
            raise ValueError(name)

        action = int(np.argmax(action_values))
        correct_action = action == target

        # Update context belief from outcome. If sensory action fails, evidence
        # shifts toward the opposite rule; if it succeeds, evidence supports the
        # direct rule.
        evidence = np.array([1.0 if target == sense else 0.05, 1.0 if target == (sense + 2) % 4 else 0.05])
        evidence = evidence / evidence.sum()
        context_belief = (1.0 - learning_rate) * context_belief + learning_rate * evidence
        context_belief = context_belief / context_belief.sum()
        context_workspace = 0.84 * context_workspace + 0.16 * context_belief
        context_workspace = context_workspace / context_workspace.sum()

        spatial_confidence = 0.88 * spatial_confidence + 0.12 * float(spatial_correct)
        context_confidence = 0.86 * context_confidence + 0.14 * float(context_correct)
        tension = 0.45 * module_conflict + 0.35 * spatial_error + 0.20 * (1.0 - context_confidence)
        latency_cost = float(beta * config["latency_cost"])
        efficiency = float(correct_action) - latency_cost - 0.18 * module_conflict

        rows.append(
            {
                "t": t,
                "phase": "pre_shift" if t < shift_step else "post_shift",
                "sensory": sense,
                "target": target,
                "action": action,
                "correct": float(correct_action),
                "spatial_choice": spatial_choice,
                "context_choice": context_choice,
                "spatial_error": float(spatial_error),
                "context_error": float(context_error),
                "module_conflict": float(module_conflict),
                "context_rule_0_belief": float(context_workspace[0]),
                "context_rule_1_belief": float(context_workspace[1]),
                "master_beta": float(beta),
                "tension": float(tension),
                "latency_cost": latency_cost,
                "efficiency": efficiency,
            }
        )

    return rows


def summarize(rows, shift_step):
    pre = [r for r in rows if r["t"] < shift_step]
    early = [r for r in rows if shift_step <= r["t"] < shift_step + 35]
    late = [r for r in rows if r["t"] >= shift_step + 35]

    def mean(chunk, key):
        return float(np.mean([r[key] for r in chunk])) if chunk else 0.0

    return {
        "pre_shift_accuracy": mean(pre, "correct"),
        "early_post_shift_accuracy": mean(early, "correct"),
        "late_post_shift_accuracy": mean(late, "correct"),
        "recovery_steps_to_82pct": recovery_step(rows, shift_step),
        "mean_master_beta": mean(rows, "master_beta"),
        "early_master_beta": mean(early, "master_beta"),
        "late_master_beta": mean(late, "master_beta"),
        "mean_module_conflict": mean(rows, "module_conflict"),
        "mean_tension": mean(rows, "tension"),
        "mean_latency_cost": mean(rows, "latency_cost"),
        "workspace_efficiency": mean(rows, "efficiency"),
        "final_context_rule_1_belief": float(rows[-1]["context_rule_1_belief"]),
    }


def plot_timeseries(results, shift_step, path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for name, rows in results.items():
        x = [r["t"] for r in rows]
        axes[0].plot(x, running_accuracy(rows, window=14), label=name)
        axes[1].plot(x, [r["master_beta"] for r in rows], label=name)
        axes[2].plot(x, [r["context_rule_1_belief"] for r in rows], label=name)
        axes[3].plot(x, [r["module_conflict"] for r in rows], label=name)
        axes[4].plot(x, [r["efficiency"] for r in rows], label=name)
    for ax in axes:
        ax.axvline(shift_step, color="#111111", ls="--", lw=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rolling accuracy")
    axes[1].set_ylabel("master beta")
    axes[2].set_ylabel("context belief")
    axes[3].set_ylabel("module conflict")
    axes[4].set_ylabel("efficiency")
    axes[-1].set_xlabel("time step")
    axes[0].set_title("Hierarchical Workspace: Rule-Shift Recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["early_post_shift_accuracy", "late_post_shift_accuracy", "mean_latency_cost", "workspace_efficiency"]
    x = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#ff8a00", "#16a3a6", "#e05a47", "#7c3aed"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Hierarchical Workspace Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=14)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(151)
    OUT.mkdir(exist_ok=True)
    shift_step = 120
    configs = {
        "monolithic_workspace": {"latency_cost": 0.05, "sensory_noise_bias": 0.12},
        "flat_multi_workspace": {"latency_cost": 0.03, "sensory_noise_bias": 0.12},
        "hierarchical_master_workspace": {"latency_cost": 0.035, "sensory_noise_bias": 0.12},
        "bad_hierarchy_bureaucracy": {"latency_cost": 0.10, "sensory_noise_bias": 0.12},
    }
    results = {name: run_condition(name, config, shift_step=shift_step) for name, config in configs.items()}
    summary = {name: summarize(rows, shift_step) for name, rows in results.items()}
    payload = {
        "note": (
            "Compares monolithic, flat multi-workspace, fast hierarchical master, and slow bureaucratic hierarchy "
            "on a rule-shift task. The hierarchy is brain-inspired: local workspaces compress sensorimotor/context "
            "signals and a master controller gates which workspace controls action."
        ),
        "shift_step": shift_step,
        "summary": summary,
        "thesis": (
            "Multiple workspaces help only when a master controller receives compressed tension/confidence signals "
            "and can re-route control faster than a monolithic system. A slow hierarchy becomes bureaucracy."
        ),
    }
    (OUT / "hierarchical_workspace_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_timeseries(results, shift_step, OUT / "hierarchical_workspace_timeseries.png")
    plot_summary(summary, OUT / "hierarchical_workspace_summary.png")
    print("Hierarchical workspace lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
