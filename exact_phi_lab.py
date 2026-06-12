#!/usr/bin/env python3
"""Exact tiny binary integration experiment.

This is an exact calculation for a deliberately small Phi-like definition:

For every binary state, compare the full system's next-state distribution against
the best bipartitioned approximation. Phi_proxy is the minimum KL divergence
over all bipartitions, averaged across all states.

This is not official IIT Phi. It is an exact, transparent proxy that lets us
visualize how recurrent coupling and a valence feedback node change integration.
"""

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EPS = 1e-12


def all_states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int8)


def state_index(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def bipartitions(n):
    nodes = set(range(n))
    parts = []
    for r in range(1, n // 2 + 1):
        for combo in itertools.combinations(range(n), r):
            a = set(combo)
            b = nodes - a
            if min(a) != 0:
                continue
            parts.append((tuple(sorted(a)), tuple(sorted(b))))
    return parts


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transition_distribution(state, weights, bias, noise=0.04):
    logits = weights @ state + bias
    probs = noise + (1 - 2 * noise) * sigmoid(logits)
    dist = np.ones(1, dtype=np.float64)
    for p in probs:
        dist = np.kron(dist, np.array([1 - p, p], dtype=np.float64))
    return dist / dist.sum()


def subset_transition(subset, subset_state, weights, bias, noise=0.04):
    n = weights.shape[0]
    outside = [i for i in range(n) if i not in subset]
    out_states = all_states(len(outside))
    accum = np.zeros(2 ** len(subset), dtype=np.float64)
    for outside_state in out_states:
        full = np.zeros(n)
        for i, bit in zip(subset, subset_state):
            full[i] = bit
        for i, bit in zip(outside, outside_state):
            full[i] = bit
        full_dist = transition_distribution(full, weights, bias, noise)
        for next_bits in all_states(n):
            prob = full_dist[state_index(next_bits)]
            sub_next = [next_bits[i] for i in subset]
            accum[state_index(sub_next)] += prob
    accum /= max(len(out_states), 1)
    return accum / accum.sum()


def product_partition_distribution(state, partition, weights, bias, noise=0.04):
    a, b = partition
    pa = subset_transition(a, [state[i] for i in a], weights, bias, noise)
    pb = subset_transition(b, [state[i] for i in b], weights, bias, noise)
    full = np.zeros(2 ** len(state), dtype=np.float64)
    for next_a in all_states(len(a)):
        for next_b in all_states(len(b)):
            bits = np.zeros(len(state), dtype=np.int8)
            for i, bit in zip(a, next_a):
                bits[i] = bit
            for i, bit in zip(b, next_b):
                bits[i] = bit
            full[state_index(bits)] += pa[state_index(next_a)] * pb[state_index(next_b)]
    return full / full.sum()


def kl(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def phi_proxy(weights, bias, noise=0.04):
    n = weights.shape[0]
    states = all_states(n)
    partitions = bipartitions(n)
    state_phi = []
    best_parts = []
    for state in states:
        p = transition_distribution(state, weights, bias, noise)
        scores = []
        for part in partitions:
            q = product_partition_distribution(state, part, weights, bias, noise)
            scores.append(kl(p, q))
        best = int(np.argmin(scores))
        state_phi.append(scores[best])
        best_parts.append(best)
    return {
        "phi_proxy": float(np.mean(state_phi)),
        "state_phi": np.array(state_phi),
        "partitions": partitions,
        "best_partition_index_by_state": best_parts,
    }


def systems(n=6):
    bias = np.zeros(n)

    feedforward = np.zeros((n, n))
    for i in range(1, n):
        feedforward[i, i - 1] = 2.0

    recurrent = np.zeros((n, n))
    for i in range(n):
        recurrent[i, (i - 1) % n] = 1.6
        recurrent[i, (i + 1) % n] = 1.2
        recurrent[i, i] = 0.5

    valence_feedback = recurrent.copy()
    valence_node = 0
    for i in range(1, n):
        valence_feedback[i, valence_node] += 1.1
        valence_feedback[valence_node, i] += 0.55
    valence_bias = bias.copy()
    valence_bias[valence_node] = -0.25

    return {
        "feedforward_chain": (feedforward, bias),
        "recurrent_ring": (recurrent, bias),
        "recurrent_with_valence_feedback": (valence_feedback, valence_bias),
    }


def plot_phi_bars(results, path):
    names = list(results)
    vals = [results[name]["phi_proxy"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, vals, color=["#7a7a7a", "#16a3a6", "#ff8a00"])
    ax.set_title("Exact Tiny Phi Proxy by Architecture")
    ax.set_ylabel("mean minimum partition KL divergence, bits")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_phi(results, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, result in results.items():
        ax.plot(result["state_phi"], lw=2, label=name)
    ax.set_title("State-by-State Phi Proxy")
    ax.set_xlabel("binary state index")
    ax.set_ylabel("minimum partition KL divergence, bits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_network(weights, title, path):
    n = weights.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[np.cos(angles), np.sin(angles)]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    max_w = max(float(np.max(np.abs(weights))), 1e-6)
    for src in range(n):
        for dst in range(n):
            w = weights[dst, src]
            if abs(w) < 0.05:
                continue
            start, end = pts[src], pts[dst]
            color = "#ff8a00" if w > 0 else "#4b6cff"
            alpha = 0.25 + 0.65 * abs(w) / max_w
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1 + 2 * abs(w) / max_w, alpha=alpha),
            )
    ax.scatter(pts[:, 0], pts[:, 1], s=900, c="#111111")
    for i, (x, y) in enumerate(pts):
        label = f"V{i}" if i == 0 and "valence" in title.lower() else str(i)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    weights_for_plot = {}
    for name, (weights, bias) in systems().items():
        results[name] = phi_proxy(weights, bias)
        weights_for_plot[name] = weights

    serializable = {
        name: {
            "phi_proxy": result["phi_proxy"],
            "best_partition_counts": {
                str(i): int(result["best_partition_index_by_state"].count(i))
                for i in sorted(set(result["best_partition_index_by_state"]))
            },
            "note": "Exact for this toy Phi proxy, not official IIT Phi.",
        }
        for name, result in results.items()
    }
    (OUT / "exact_phi_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_phi_bars(results, OUT / "exact_phi_bar_graph.png")
    plot_state_phi(results, OUT / "exact_phi_by_state.png")
    for name, weights in weights_for_plot.items():
        plot_network(weights, name.replace("_", " "), OUT / f"{name}_network.png")

    print("Exact tiny Phi proxy lab complete")
    print(f"outputs: {OUT}")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
