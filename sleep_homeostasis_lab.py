#!/usr/bin/env python3
"""Sleep / synaptic homeostasis toy experiment.

This lab asks whether an integrated recurrent system can degrade when feedback
gets too dense and echo-like, and whether an offline down-selection pass can
restore useful separability.

The model is deliberately tiny and transparent:

1. Start with the recurrent + valence-feedback binary system from
   `exact_phi_lab.py`.
2. Add repeated "fatigue" cycles: dense common-mode coupling and bias drift.
   This mimics runaway synchrony / echo-chamber pressure, not biological sleep.
3. Measure exact Phi proxy, state separability, node variance, and saturation.
4. Apply a "sleep" intervention: disconnect from input, downscale global
   coupling, keep only the strongest structured edges in each row, and recenter
   bias.

This is not a model of dreams or subjective experience. It is a toy maintenance
test for the claim that integrated systems may need offline pruning to prevent
unstructured feedback from becoming architectural sludge.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, phi_proxy, systems, transition_distribution


def transition_metrics(weights, bias):
    """Measure integration and state separability for a tiny binary system."""
    n = weights.shape[0]
    dists = []
    node_probs = []
    for state in all_states(n):
        dists.append(transition_distribution(state, weights, bias, noise=0.04))
        logits = weights @ state + bias
        probs = 0.04 + (1.0 - 0.08) / (1.0 + np.exp(-logits))
        node_probs.append(probs)
    dists = np.array(dists)
    node_probs = np.array(node_probs)

    separations = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            separations.append(np.mean(np.abs(dists[i] - dists[j])))

    entropy = -float(np.mean(np.sum(dists * np.log2(np.clip(dists, 1e-12, 1.0)), axis=1)))
    node_variance = float(np.mean(np.var(node_probs, axis=0)))
    saturation = float(2.0 * np.mean(np.abs(node_probs - 0.5)))
    phi = phi_proxy(weights, bias, noise=0.04)["phi_proxy"]
    return {
        "phi_proxy": float(phi),
        "state_separability": float(np.mean(separations)),
        "transition_entropy": entropy,
        "node_variance": node_variance,
        "saturation": saturation,
    }


def fatigue_step(weights, bias, echo_gain=0.10, scale=1.04, bias_drift=0.08):
    """Add dense common-mode feedback and bias drift."""
    common = np.ones_like(weights)
    np.fill_diagonal(common, 0.7)
    return scale * weights + echo_gain * common, bias + bias_drift


def sleep_downselect(weights, bias, keep_per_row=2):
    """Offline pruning/down-selection.

    Strong structured edges survive. Weak dense crosstalk is damped. Bias is
    recentered so the system is not stuck in an always-on attractor.
    """
    slept = 0.78 * weights.copy()
    slept_bias = 0.40 * bias.copy()
    for row in range(slept.shape[0]):
        weak = np.argsort(np.abs(slept[row]))[:-keep_per_row]
        slept[row, weak] *= 0.15
    return slept, slept_bias


def run_experiment(fatigue_cycles=4):
    weights, bias = systems(n=4)["recurrent_with_valence_feedback"]
    rows = []
    rows.append({"phase": "baseline", "cycle": 0, **transition_metrics(weights, bias)})

    fatigued_w = weights.copy()
    fatigued_b = bias.copy()
    for cycle in range(1, fatigue_cycles + 1):
        fatigued_w, fatigued_b = fatigue_step(fatigued_w, fatigued_b)
        rows.append({"phase": "fatigue", "cycle": cycle, **transition_metrics(fatigued_w, fatigued_b)})

    slept_w, slept_b = sleep_downselect(fatigued_w, fatigued_b)
    rows.append({"phase": "sleep_downselection", "cycle": fatigue_cycles + 1, **transition_metrics(slept_w, slept_b)})
    return rows


def plot_metrics(rows, path):
    x = np.arange(len(rows))
    labels = [f"{r['phase']}\n{r['cycle']}" for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    metrics = [
        ("phi_proxy", "Phi proxy"),
        ("state_separability", "state separability"),
        ("node_variance", "node variance"),
        ("saturation", "saturation"),
    ]
    colors = ["#7c3aed", "#16a3a6", "#ff8a00", "#e05a47"]
    for ax, (metric, label), color in zip(axes, metrics, colors):
        ax.plot(x, [r[metric] for r in rows], marker="o", lw=2.5, color=color)
        ax.axvline(len(rows) - 1, color="#111111", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20)
    axes[0].set_title("Sleep Homeostasis Toy: Fatigue vs Offline Down-Selection")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(rows, path):
    baseline = rows[0]
    fatigued = rows[-2]
    slept = rows[-1]
    names = ["baseline", "fatigued", "after_sleep"]
    metrics = ["phi_proxy", "state_separability", "node_variance", "saturation"]
    x = np.arange(len(names))
    width = 0.20
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#7c3aed", "#16a3a6", "#ff8a00", "#e05a47"]
    for i, metric in enumerate(metrics):
        ax.bar(
            x + (i - 1.5) * width,
            [baseline[metric], fatigued[metric], slept[metric]],
            width,
            label=metric,
            color=colors[i],
        )
    ax.set_title("Sleep Down-Selection Restores Useful Dynamics")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows = run_experiment()
    baseline = rows[0]
    fatigued = rows[-2]
    slept = rows[-1]
    payload = {
        "note": (
            "Toy sleep/homeostasis experiment on a 4-node recurrent valence-feedback system. "
            "Fatigue adds dense common-mode feedback; sleep down-selects weak crosstalk and recenters bias."
        ),
        "rows": rows,
        "summary": {
            "baseline_phi": baseline["phi_proxy"],
            "fatigued_phi": fatigued["phi_proxy"],
            "after_sleep_phi": slept["phi_proxy"],
            "baseline_state_separability": baseline["state_separability"],
            "fatigued_state_separability": fatigued["state_separability"],
            "after_sleep_state_separability": slept["state_separability"],
            "fatigue_phi_drop_pct": 100.0 * (baseline["phi_proxy"] - fatigued["phi_proxy"]) / baseline["phi_proxy"],
            "sleep_phi_recovery_pct_of_baseline": 100.0 * slept["phi_proxy"] / baseline["phi_proxy"],
        },
        "thesis": (
            "Integrated recurrence is not maintenance-free. Dense echo-like feedback can reduce state separability "
            "and Phi proxy; offline down-selection can restore structured integration by pruning ungrounded crosstalk."
        ),
    }
    (OUT / "sleep_homeostasis_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_metrics(rows, OUT / "sleep_homeostasis_timeseries.png")
    plot_summary(rows, OUT / "sleep_homeostasis_summary.png")
    print("Sleep homeostasis lab complete")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
