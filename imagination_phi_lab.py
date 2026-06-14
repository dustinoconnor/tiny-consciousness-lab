#!/usr/bin/env python3
"""Exact tiny Phi-proxy test for valence, imagination, and confidence gating.

This is not official IIT Phi. It reuses the exact toy proxy from
`exact_phi_lab.py`: for every binary state, compare the full next-state
distribution against the best bipartitioned approximation.

The question is narrower than "is this conscious?":

Does adding an imagination/world-model channel increase irreducible causal
structure, or does it only help behavior in planning tasks?
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, bipartitions, kl, state_index, transition_distribution


NODES = [
    "sense",
    "memory",
    "valence",
    "imagination",
    "confidence",
    "self",
    "imagined_valence",
    "action",
]


def blank_system():
    return np.zeros((len(NODES), len(NODES))), np.zeros(len(NODES))


def add(weights, dst, src, amount):
    weights[NODES.index(dst), NODES.index(src)] += amount


def phi_proxy_cached(weights, bias, noise=0.04):
    """Same toy Phi proxy as exact_phi_lab.py, with memoized subset distributions."""
    n = weights.shape[0]
    states = all_states(n)
    state_tuples = [tuple(int(x) for x in state) for state in states]
    full_by_state = {
        state_tuple: transition_distribution(np.array(state_tuple), weights, bias, noise)
        for state_tuple in state_tuples
    }
    next_states = all_states(n)
    next_tuples = [tuple(int(x) for x in state) for state in next_states]
    partitions = bipartitions(n)
    subset_cache = {}

    def subset_dist(subset, subset_state):
        subset = tuple(subset)
        subset_state = tuple(int(x) for x in subset_state)
        key = (subset, subset_state)
        if key in subset_cache:
            return subset_cache[key]

        outside = [i for i in range(n) if i not in subset]
        out_states = all_states(len(outside))
        accum = np.zeros(2 ** len(subset), dtype=np.float64)
        for outside_state in out_states:
            full = [0] * n
            for i, bit in zip(subset, subset_state):
                full[i] = int(bit)
            for i, bit in zip(outside, outside_state):
                full[i] = int(bit)
            full_dist = full_by_state[tuple(full)]
            for next_bits in next_tuples:
                sub_next = [next_bits[i] for i in subset]
                accum[state_index(sub_next)] += full_dist[state_index(next_bits)]
        accum /= max(len(out_states), 1)
        subset_cache[key] = accum / accum.sum()
        return subset_cache[key]

    def partition_dist(state_tuple, partition):
        a, b = partition
        pa = subset_dist(a, [state_tuple[i] for i in a])
        pb = subset_dist(b, [state_tuple[i] for i in b])
        full = np.zeros(2 ** n, dtype=np.float64)
        for next_a in all_states(len(a)):
            for next_b in all_states(len(b)):
                bits = np.zeros(n, dtype=np.int8)
                for i, bit in zip(a, next_a):
                    bits[i] = bit
                for i, bit in zip(b, next_b):
                    bits[i] = bit
                full[state_index(bits)] += pa[state_index(next_a)] * pb[state_index(next_b)]
        return full / full.sum()

    state_phi = []
    best_parts = []
    for state_tuple in state_tuples:
        p = full_by_state[state_tuple]
        scores = [kl(p, partition_dist(state_tuple, part)) for part in partitions]
        best = int(np.argmin(scores))
        state_phi.append(scores[best])
        best_parts.append(best)
    return {
        "phi_proxy": float(np.mean(state_phi)),
        "state_phi": np.array(state_phi),
        "partitions": partitions,
        "best_partition_index_by_state": best_parts,
    }


def systems():
    """Build small binary circuits with increasingly rich feedback loops."""
    variants = {}

    weights, bias = blank_system()
    add(weights, "action", "sense", 2.0)
    add(weights, "memory", "sense", 1.0)
    add(weights, "memory", "memory", 0.7)
    variants["reflex_only"] = (weights.copy(), bias.copy())

    weights, bias = blank_system()
    add(weights, "action", "sense", 1.6)
    add(weights, "memory", "sense", 1.0)
    add(weights, "memory", "memory", 0.7)
    add(weights, "valence", "sense", 1.0)
    add(weights, "valence", "action", 0.9)
    add(weights, "action", "valence", 1.2)
    add(weights, "memory", "valence", 0.7)
    bias[NODES.index("valence")] = -0.2
    variants["reflex_valence"] = (weights.copy(), bias.copy())

    weights, bias = blank_system()
    add(weights, "action", "sense", 1.2)
    add(weights, "memory", "sense", 0.9)
    add(weights, "memory", "memory", 1.0)
    add(weights, "valence", "sense", 1.0)
    add(weights, "valence", "action", 0.8)
    add(weights, "action", "valence", 1.1)
    add(weights, "memory", "valence", 0.9)
    add(weights, "valence", "memory", 0.8)
    add(weights, "action", "memory", 0.8)
    variants["valence_memory"] = (weights.copy(), bias.copy())

    weights, bias = blank_system()
    add(weights, "action", "sense", 0.9)
    add(weights, "memory", "sense", 0.9)
    add(weights, "memory", "memory", 1.0)
    add(weights, "valence", "sense", 0.9)
    add(weights, "valence", "action", 0.8)
    add(weights, "action", "valence", 1.0)
    add(weights, "memory", "valence", 0.8)
    add(weights, "valence", "memory", 0.7)
    add(weights, "imagination", "sense", 0.8)
    add(weights, "imagination", "memory", 1.1)
    add(weights, "imagination", "action", 0.7)
    add(weights, "action", "imagination", 1.2)
    add(weights, "valence", "imagination", 0.8)
    add(weights, "memory", "imagination", 0.6)
    variants["valence_imagination"] = (weights.copy(), bias.copy())

    weights, bias = blank_system()
    add(weights, "action", "sense", 0.8)
    add(weights, "memory", "sense", 0.9)
    add(weights, "memory", "memory", 1.0)
    add(weights, "valence", "sense", 0.8)
    add(weights, "valence", "action", 0.8)
    add(weights, "action", "valence", 0.9)
    add(weights, "memory", "valence", 0.8)
    add(weights, "valence", "memory", 0.7)
    add(weights, "imagination", "sense", 0.7)
    add(weights, "imagination", "memory", 1.0)
    add(weights, "imagination", "action", 0.6)
    add(weights, "confidence", "imagination", 1.0)
    add(weights, "confidence", "valence", 0.7)
    add(weights, "confidence", "memory", 0.5)
    add(weights, "action", "imagination", 1.0)
    add(weights, "action", "confidence", 1.1)
    add(weights, "valence", "confidence", 0.8)
    add(weights, "imagination", "confidence", 0.7)
    add(weights, "memory", "confidence", 0.5)
    variants["gated_imagination"] = (weights.copy(), bias.copy())

    weights, bias = variants["gated_imagination"]
    weights = weights.copy()
    bias = bias.copy()
    add(weights, "imagination", "imagination", 0.9)
    add(weights, "confidence", "confidence", 0.9)
    add(weights, "valence", "valence", 0.7)
    add(weights, "action", "action", 0.5)
    variants["recurrent_gated_imagination"] = (weights.copy(), bias.copy())

    weights, bias = variants["gated_imagination"]
    weights = weights.copy()
    bias = bias.copy()
    add(weights, "self", "sense", 0.6)
    add(weights, "self", "memory", 0.9)
    add(weights, "self", "action", 0.8)
    add(weights, "self", "valence", 0.7)
    add(weights, "memory", "self", 0.7)
    add(weights, "action", "self", 0.7)
    add(weights, "valence", "self", 0.6)
    variants["self_model_loop"] = (weights.copy(), bias.copy())

    weights, bias = variants["self_model_loop"]
    weights = weights.copy()
    bias = bias.copy()
    add(weights, "imagination", "self", 0.8)
    add(weights, "imagination", "action", 0.8)
    add(weights, "imagination", "memory", 0.8)
    add(weights, "confidence", "imagination", 0.7)
    add(weights, "self", "imagination", 0.7)
    variants["counterfactual_self_imagination"] = (weights.copy(), bias.copy())

    weights, bias = variants["counterfactual_self_imagination"]
    weights = weights.copy()
    bias = bias.copy()
    add(weights, "imagined_valence", "imagination", 1.0)
    add(weights, "imagined_valence", "self", 0.8)
    add(weights, "imagined_valence", "memory", 0.6)
    add(weights, "confidence", "imagined_valence", 0.8)
    add(weights, "action", "imagined_valence", 1.0)
    add(weights, "valence", "imagined_valence", 0.8)
    add(weights, "self", "imagined_valence", 0.6)
    variants["counterfactual_imagined_valence"] = (weights.copy(), bias.copy())

    weights, bias = variants["counterfactual_imagined_valence"]
    weights = weights.copy()
    bias = bias.copy()
    add(weights, "self", "self", 0.65)
    add(weights, "imagination", "imagination", 0.65)
    add(weights, "imagined_valence", "imagined_valence", 0.55)
    add(weights, "confidence", "confidence", 0.55)
    add(weights, "memory", "memory", 0.35)
    variants["recursive_inner_world"] = (weights.copy(), bias.copy())

    weights, bias = variants["counterfactual_imagined_valence"]
    weights = weights.copy()
    bias = bias.copy()
    # Tight routing: confidence reconciles real valence, imagined valence, and self-state.
    add(weights, "confidence", "valence", 1.2)
    add(weights, "confidence", "imagined_valence", 1.2)
    add(weights, "confidence", "self", 0.9)
    add(weights, "confidence", "sense", 0.8)
    add(weights, "valence", "confidence", 0.9)
    add(weights, "imagined_valence", "confidence", 0.9)
    add(weights, "self", "confidence", 0.8)
    add(weights, "imagination", "confidence", 0.8)
    # Action is no longer allowed to lean mostly on raw sense; it needs the reconciled loop.
    weights[NODES.index("action"), NODES.index("sense")] *= 0.25
    add(weights, "action", "confidence", 1.4)
    add(weights, "action", "valence", 0.9)
    add(weights, "action", "imagined_valence", 1.1)
    add(weights, "action", "self", 0.8)
    variants["attention_reconciled_inner_world"] = (weights.copy(), bias.copy())

    return variants


def ablation_damage(weights, bias):
    rows = []
    states = all_states(weights.shape[0])
    baseline = [transition_distribution(state, weights, bias) for state in states]
    for i, node in enumerate(NODES):
        damaged = weights.copy()
        damaged[i, :] = 0.0
        damaged[:, i] = 0.0
        damaged_dists = [transition_distribution(state, damaged, bias) for state in states]
        damage = float(np.mean([kl(p, q) for p, q in zip(baseline, damaged_dists)]))
        rows.append(
            {
                "node": node,
                "causal_distribution_damage": damage,
            }
        )
    return rows


def plot_phi_bars(results, path):
    names = list(results)
    vals = [results[name]["phi_proxy"] for name in names]
    colors = [
        "#7a7a7a",
        "#16a3a6",
        "#ff8a00",
        "#8b5cf6",
        "#d946ef",
        "#0f766e",
        "#2563eb",
        "#dc2626",
        "#65a30d",
        "#111827",
        "#0891b2",
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(names, vals, color=colors[: len(names)])
    ax.set_title("Exact Tiny Phi Proxy: Valence, Self-Model, and Inner-World Circuits")
    ax.set_ylabel("mean minimum partition KL divergence, bits")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_phi(results, path):
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, result in results.items():
        ax.plot(result["state_phi"], lw=2, label=name)
    ax.set_title("State-by-State Phi Proxy for Imagination Circuits")
    ax.set_xlabel("binary state index")
    ax.set_ylabel("minimum partition KL divergence, bits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_named_network(weights, title, path):
    n = weights.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[np.cos(angles), np.sin(angles)]
    activity = np.abs(weights).sum(axis=0) + np.abs(weights).sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))
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

    node_colors = ["#111111" if a > 0.05 else "#d5d5d5" for a in activity]
    text_colors = ["white" if a > 0.05 else "#555555" for a in activity]
    ax.scatter(pts[:, 0], pts[:, 1], s=1100, c=node_colors, edgecolors="#222222")
    for i, (x, y) in enumerate(pts):
        label = f"{i}\n{NODES[i]}"
        ax.text(x, y, label, ha="center", va="center", color=text_colors[i], fontsize=9, fontweight="bold")
    ax.text(
        0,
        -1.28,
        "Gray node = present for comparison but inactive in this variant",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation_damage(ablation_results, path):
    labels = list(ablation_results)
    x = np.arange(len(NODES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, label in enumerate(labels):
        values = [row["causal_distribution_damage"] for row in ablation_results[label]]
        ax.bar(x + (offset - (len(labels) - 1) / 2) * width, values, width, label=label)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_title("Node Ablation Damage: Which Nodes Change The System Most?")
    ax.set_ylabel("mean KL divergence after node removal")
    ax.set_xticks(x)
    ax.set_xticklabels(NODES, rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    weights_for_plot = {}
    for name, (weights, bias) in systems().items():
        results[name] = phi_proxy_cached(weights, bias)
        weights_for_plot[name] = weights

    ablation_targets = [
        "self_model_loop",
        "counterfactual_imagined_valence",
        "attention_reconciled_inner_world",
    ]
    ablations = {
        name: ablation_damage(weights_for_plot[name], systems()[name][1])
        for name in ablation_targets
        if name in results
    }

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
    serializable["ablation_damage"] = ablations
    serializable["node_order"] = NODES
    serializable["interpretation"] = (
        "If imagination raises the proxy, it suggests richer irreducible causal routing in this toy definition. "
        "It does not prove subjective experience or official IIT Phi."
    )
    (OUT / "imagination_phi_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_phi_bars(results, OUT / "imagination_phi_bar_graph.png")
    plot_state_phi(results, OUT / "imagination_phi_by_state.png")
    plot_ablation_damage(ablations, OUT / "imagination_phi_ablation_damage.png")
    for name, weights in weights_for_plot.items():
        plot_named_network(weights, name.replace("_", " "), OUT / f"{name}_network.png")

    print("Imagination Phi proxy lab complete")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
