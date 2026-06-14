#!/usr/bin/env python3
"""Segregation-plus-integration toy test.

This lab tests the next architectural lesson:

Do not simply add loops everywhere. Useful mind-like routing needs specialized
subsystems plus a shared integrated workspace.

We compare small exact binary circuits:

- feedforward_specialists: clean but weakly integrated
- random_feedback_soup: many loops with poor organization
- segregated_modules: specialized clusters with little shared binding
- modular_workspace: specialists feeding a central workspace/action core
- overconnected_workspace: the same system with too much recurrent cross-talk

The exact Phi proxy is calculated for all variants, but we also measure
grounding, action sensitivity, and a simple "useful integration" score so Phi is
not treated as a scoreboard by itself.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, phi_proxy, transition_distribution


NODES = ["sense", "memory", "imagination", "valence", "workspace", "action"]
SENSE = NODES.index("sense")
ACTION = NODES.index("action")
WORKSPACE = NODES.index("workspace")
INTERNAL = [NODES.index(name) for name in ["memory", "imagination", "valence", "workspace"]]


def add(weights, dst, src, amount):
    weights[NODES.index(dst), NODES.index(src)] += amount


def make_systems():
    bias = np.zeros(len(NODES))

    feedforward = np.zeros((len(NODES), len(NODES)))
    add(feedforward, "memory", "sense", 1.8)
    add(feedforward, "imagination", "memory", 1.3)
    add(feedforward, "valence", "sense", 1.2)
    add(feedforward, "workspace", "imagination", 1.0)
    add(feedforward, "action", "workspace", 1.4)
    add(feedforward, "action", "valence", 1.0)

    rng = np.random.default_rng(17)
    soup = rng.normal(0.55, 0.55, size=(len(NODES), len(NODES)))
    soup[np.abs(soup) < 0.35] = 0.0
    soup[SENSE, :] = 0.0
    soup[:, ACTION] *= 0.8
    soup[ACTION, SENSE] += 0.35

    modules = np.zeros((len(NODES), len(NODES)))
    # Sensory-memory specialist.
    add(modules, "memory", "sense", 1.8)
    add(modules, "memory", "memory", 0.7)
    add(modules, "sense", "memory", 0.2)
    # Imagination specialist.
    add(modules, "imagination", "memory", 1.2)
    add(modules, "imagination", "imagination", 0.9)
    # Valence specialist.
    add(modules, "valence", "sense", 1.2)
    add(modules, "valence", "action", 0.8)
    add(modules, "valence", "valence", 0.55)
    # Weak output path.
    add(modules, "action", "valence", 0.7)
    add(modules, "action", "imagination", 0.45)

    workspace = modules.copy()
    # Central binding core: specialists broadcast into workspace; workspace
    # broadcasts back only enough to coordinate action and update specialists.
    add(workspace, "workspace", "sense", 0.8)
    add(workspace, "workspace", "memory", 1.0)
    add(workspace, "workspace", "imagination", 1.0)
    add(workspace, "workspace", "valence", 1.0)
    add(workspace, "memory", "workspace", 0.55)
    add(workspace, "imagination", "workspace", 0.55)
    add(workspace, "valence", "workspace", 0.55)
    add(workspace, "action", "workspace", 1.4)
    add(workspace, "workspace", "action", 0.55)
    add(workspace, "action", "sense", 0.6)

    overconnected = workspace.copy()
    for dst in ["memory", "imagination", "valence", "workspace", "action"]:
        for src in ["memory", "imagination", "valence", "workspace", "action"]:
            if dst != src:
                add(overconnected, dst, src, 0.75)
    add(overconnected, "imagination", "imagination", 1.1)
    add(overconnected, "workspace", "workspace", 1.0)
    add(overconnected, "action", "imagination", 1.0)
    add(overconnected, "action", "workspace", 0.8)

    return {
        "feedforward_specialists": (feedforward, bias.copy()),
        "random_feedback_soup": (soup, bias.copy()),
        "segregated_modules": (modules, bias.copy()),
        "modular_workspace": (workspace, bias.copy()),
        "overconnected_workspace": (overconnected, bias.copy()),
    }


def node_one_probability(dist, node):
    prob = 0.0
    for idx, bits in enumerate(all_states(len(NODES))):
        if bits[node]:
            prob += dist[idx]
    return float(prob)


def sensitivity(weights, bias, source, target=ACTION):
    diffs = []
    for state in all_states(len(NODES)):
        if state[source] == 1:
            continue
        flipped = state.copy()
        flipped[source] = 1
        p0 = node_one_probability(transition_distribution(state, weights, bias), target)
        p1 = node_one_probability(transition_distribution(flipped, weights, bias), target)
        diffs.append(abs(p1 - p0))
    return float(np.mean(diffs))


def grounding_ratio(weights):
    targets = INTERNAL + [ACTION]
    external = sum(abs(weights[dst, SENSE]) for dst in targets)
    internal = 0.0
    for dst in targets:
        for src in INTERNAL + [ACTION]:
            internal += abs(weights[dst, src])
    return float(external / max(external + internal, 1e-9))


def module_segregation(weights):
    groups = [
        [NODES.index("sense"), NODES.index("memory")],
        [NODES.index("imagination")],
        [NODES.index("valence")],
        [NODES.index("workspace"), NODES.index("action")],
    ]
    within = 0.0
    between = 0.0
    for dst in range(len(NODES)):
        for src in range(len(NODES)):
            if dst == src:
                continue
            same = any(dst in group and src in group for group in groups)
            if same:
                within += abs(weights[dst, src])
            else:
                between += abs(weights[dst, src])
    return float(within / max(within + between, 1e-9))


def workspace_binding(weights):
    incoming = sum(abs(weights[WORKSPACE, src]) for src in range(len(NODES)) if src != WORKSPACE)
    outgoing = sum(abs(weights[dst, WORKSPACE]) for dst in range(len(NODES)) if dst != WORKSPACE)
    return float((incoming + outgoing) / max(np.sum(np.abs(weights)), 1e-9))


def summarize_system(name, weights, bias):
    phi = phi_proxy(weights, bias)["phi_proxy"]
    sense_action = sensitivity(weights, bias, SENSE, ACTION)
    workspace_action = sensitivity(weights, bias, WORKSPACE, ACTION)
    specialist_nodes = [NODES.index(name) for name in ["memory", "imagination", "valence"]]
    specialist_action = float(np.mean([sensitivity(weights, bias, node, ACTION) for node in specialist_nodes]))
    internal_action = float(np.mean([sensitivity(weights, bias, node, ACTION) for node in INTERNAL]))
    grounding = grounding_ratio(weights)
    segregation = module_segregation(weights)
    binding = workspace_binding(weights)
    grounded_control = 0.5 * sense_action + workspace_action
    delusion_risk = float(specialist_action / max(grounded_control, 1e-9))
    useful = float(phi * (0.5 + grounding) * (0.5 + segregation) * binding * (0.5 + grounded_control) / (1.0 + delusion_risk))
    return {
        "phi_proxy": float(phi),
        "grounding_ratio": grounding,
        "module_segregation": segregation,
        "workspace_binding": binding,
        "action_sensitivity_sense": sense_action,
        "action_sensitivity_workspace": workspace_action,
        "grounded_control_sensitivity": float(grounded_control),
        "action_sensitivity_specialist_mean": specialist_action,
        "action_sensitivity_internal_mean": internal_action,
        "delusion_risk_index": delusion_risk,
        "useful_integration_score": useful,
    }


def plot_summary(rows, path):
    names = list(rows)
    metrics = ["phi_proxy", "grounding_ratio", "module_segregation", "workspace_binding", "useful_integration_score"]
    x = np.arange(len(names))
    width = 0.15
    colors = ["#8b5cf6", "#16a3a6", "#ff8a00", "#2563eb", "#65a30d"]
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, [rows[name][metric] for name in names], width, label=metric, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18)
    ax.set_title("Structured Routing Beats Raw Feedback Loops")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_networks(systems, path):
    fig, axes = plt.subplots(1, len(systems), figsize=(18, 4))
    angles = np.linspace(0, 2 * np.pi, len(NODES), endpoint=False)
    pts = np.c_[np.cos(angles), np.sin(angles)]
    for ax, (name, (weights, _bias)) in zip(axes, systems.items()):
        ax.set_title(name, fontsize=10)
        max_w = max(float(np.max(np.abs(weights))), 1e-6)
        for src in range(len(NODES)):
            for dst in range(len(NODES)):
                w = weights[dst, src]
                if abs(w) < 0.08:
                    continue
                color = "#ff8a00" if w > 0 else "#4b6cff"
                alpha = 0.2 + 0.7 * abs(w) / max_w
                ax.annotate(
                    "",
                    xy=pts[dst],
                    xytext=pts[src],
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8 + 1.8 * abs(w) / max_w, alpha=alpha),
                )
        node_colors = ["#16a3a6", "#16a3a6", "#8b5cf6", "#ff8a00", "#2563eb", "#111111"]
        ax.scatter(pts[:, 0], pts[:, 1], s=600, c=node_colors, edgecolors="#222222")
        for i, (x, y) in enumerate(pts):
            ax.text(x, y, NODES[i][:3], ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    systems = make_systems()
    rows = {name: summarize_system(name, weights, bias) for name, (weights, bias) in systems.items()}
    payload = {
        "summary": rows,
        "node_order": NODES,
        "note": (
            "Exact tiny circuit comparison. The useful_integration_score is a heuristic combining Phi proxy, "
            "external grounding, module segregation, workspace binding, grounded control, and delusion penalty. "
            "It is not official IIT Phi."
        ),
    }
    (OUT / "modular_workspace_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(rows, OUT / "modular_workspace_summary.png")
    plot_networks(systems, OUT / "modular_workspace_networks.png")
    print("Modular workspace lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
