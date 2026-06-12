#!/usr/bin/env python3
"""Intervention tests for the tiny Phi proxy systems.

These tests ask whether the valence-feedback architecture behaves differently
under damage, noise, and scale.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, phi_proxy, sigmoid, state_index, systems, transition_distribution


def expected_next_state(state, weights, bias, node_noise=0.0):
    logits = weights @ state + bias
    if node_noise:
        logits = logits + np.random.normal(0.0, node_noise, size=logits.shape)
    return sigmoid(logits)


def rollout(weights, bias, start, steps=32, ablate_node=None, node_noise=0.0):
    state = np.array(start, dtype=np.float64)
    states = [state.copy()]
    for _ in range(steps):
        next_state = expected_next_state(state, weights, bias, node_noise=node_noise)
        if ablate_node is not None:
            next_state[ablate_node] = 0.0
        state = next_state
        states.append(state.copy())
    return np.array(states)


def trajectory_divergence(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def ablation_shock_test():
    start = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
    results = {}
    traces = {}
    for name, (weights, bias) in systems().items():
        base = rollout(weights, bias, start)
        node_scores = []
        for node in range(weights.shape[0]):
            damaged = rollout(weights, bias, start, ablate_node=node)
            node_scores.append(trajectory_divergence(base, damaged))
        results[name] = {
            "valence_or_node0_shock": float(node_scores[0]),
            "mean_other_node_shock": float(np.mean(node_scores[1:])),
            "max_node_shock": float(np.max(node_scores)),
            "most_disruptive_node": int(np.argmax(node_scores)),
        }
        traces[name] = (base, rollout(weights, bias, start, ablate_node=0))
    return results, traces


def noise_tolerance_test(levels=(0.0, 0.05, 0.1, 0.2, 0.35), trials=64):
    np.random.seed(11)
    start = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
    results = {}
    for name, (weights, bias) in systems().items():
        clean = rollout(weights, bias, start)
        errors = []
        for level in levels:
            trial_errors = []
            for _ in range(trials):
                noisy = rollout(weights, bias, start, node_noise=level)
                trial_errors.append(trajectory_divergence(clean, noisy))
            errors.append(float(np.mean(trial_errors)))
        results[name] = {"noise_levels": list(levels), "mean_trajectory_error": errors}
    return results


def scale_vs_integration_test():
    results = {}
    for n in range(3, 8):
        bias = np.zeros(n)
        weights = np.zeros((n, n))
        for i in range(1, n):
            weights[i, i - 1] = 2.0
        results[f"feedforward_chain_{n}_nodes"] = phi_proxy(weights, bias)["phi_proxy"]

    valence_weights, valence_bias = systems(6)["recurrent_with_valence_feedback"]
    results["tiny_valence_feedback_6_nodes"] = phi_proxy(valence_weights, valence_bias)["phi_proxy"]
    return results


def plot_ablation(results, path):
    names = list(results)
    node0 = [results[name]["valence_or_node0_shock"] for name in names]
    others = [results[name]["mean_other_node_shock"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, node0, width, label="node 0 ablated", color="#ff8a00")
    ax.bar(x + width / 2, others, width, label="mean other node ablated", color="#16a3a6")
    ax.set_title("Ablation Shock Test")
    ax.set_ylabel("mean trajectory divergence")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise(results, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, data in results.items():
        ax.plot(data["noise_levels"], data["mean_trajectory_error"], marker="o", lw=2, label=name)
    ax.set_title("Noise Tolerance Test")
    ax.set_xlabel("Gaussian noise added to node logits")
    ax.set_ylabel("mean trajectory divergence from clean rollout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scale(results, path):
    chain_items = [(k, v) for k, v in results.items() if k.startswith("feedforward")]
    chain_x = [int(k.split("_")[-2]) for k, _ in chain_items]
    chain_y = [v for _, v in chain_items]
    val = results["tiny_valence_feedback_6_nodes"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(chain_x, chain_y, marker="o", lw=2, color="#7a7a7a", label="feedforward chain")
    ax.axhline(val, color="#ff8a00", lw=2, label="6-node valence feedback")
    ax.scatter([6], [val], color="#ff8a00", s=80)
    ax.set_title("Scale vs. Integration Proxy")
    ax.set_xlabel("feedforward chain node count")
    ax.set_ylabel("exact tiny Phi proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    ablation_results, _ = ablation_shock_test()
    noise_results = noise_tolerance_test()
    scale_results = scale_vs_integration_test()
    all_results = {
        "ablation_shock": ablation_results,
        "noise_tolerance": noise_results,
        "scale_vs_integration": scale_results,
        "note": "These are intervention tests on the toy Phi proxy systems, not official IIT results. The exact scale sweep is capped at 7 nodes because naive exact partition enumeration gets slow quickly.",
    }
    (OUT / "intervention_metrics.json").write_text(json.dumps(all_results, indent=2))
    plot_ablation(ablation_results, OUT / "ablation_shock_test.png")
    plot_noise(noise_results, OUT / "noise_tolerance_test.png")
    plot_scale(scale_results, OUT / "scale_vs_integration_test.png")

    print("Intervention lab complete")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
