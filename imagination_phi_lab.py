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

from exact_phi_lab import OUT, phi_proxy


NODES = ["sense", "memory", "valence", "imagination", "confidence", "action"]


def blank_system():
    return np.zeros((len(NODES), len(NODES))), np.zeros(len(NODES))


def add(weights, dst, src, amount):
    weights[NODES.index(dst), NODES.index(src)] += amount


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

    return variants


def plot_phi_bars(results, path):
    names = list(results)
    vals = [results[name]["phi_proxy"] for name in names]
    colors = ["#7a7a7a", "#16a3a6", "#ff8a00", "#8b5cf6", "#d946ef", "#0f766e"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(names, vals, color=colors[: len(names)])
    ax.set_title("Exact Tiny Phi Proxy: Valence and Imagination Circuits")
    ax.set_ylabel("mean minimum partition KL divergence, bits")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_phi(results, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
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
    serializable["node_order"] = NODES
    serializable["interpretation"] = (
        "If imagination raises the proxy, it suggests richer irreducible causal routing in this toy definition. "
        "It does not prove subjective experience or official IIT Phi."
    )
    (OUT / "imagination_phi_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_phi_bars(results, OUT / "imagination_phi_bar_graph.png")
    plot_state_phi(results, OUT / "imagination_phi_by_state.png")
    for name, weights in weights_for_plot.items():
        plot_named_network(weights, name.replace("_", " "), OUT / f"{name}_network.png")

    print("Imagination Phi proxy lab complete")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
