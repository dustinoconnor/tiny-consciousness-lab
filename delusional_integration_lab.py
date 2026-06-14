#!/usr/bin/env python3
"""Sweep internal loop strength against external grounding.

This is a toy test for the "delusional integration" warning:

If self/imagination feedback becomes too strong relative to sensory input, a
system can become more internally integrated while becoming less grounded in the
outside world.

We keep this exact and small with six binary nodes:

    sense, memory, valence, imagination, self, action

For each internal-loop scale, we calculate:

- exact tiny Phi proxy
- external grounding ratio from weights
- action sensitivity to flipping the sensory node
- action sensitivity to flipping internal nodes
- a simple delusion-risk index: internal sensitivity / external sensitivity

This is not official IIT Phi and not a consciousness test.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, phi_proxy, transition_distribution


NODES = ["sense", "memory", "valence", "imagination", "self", "action"]
SENSE = NODES.index("sense")
ACTION = NODES.index("action")
INTERNAL = [NODES.index(name) for name in ["memory", "valence", "imagination", "self"]]


def add(weights, dst, src, amount):
    weights[NODES.index(dst), NODES.index(src)] += amount


def make_system(internal_scale, external_scale=1.0):
    weights = np.zeros((len(NODES), len(NODES)))
    bias = np.zeros(len(NODES))

    # External grounding.
    add(weights, "memory", "sense", 1.0 * external_scale)
    add(weights, "valence", "sense", 0.8 * external_scale)
    add(weights, "imagination", "sense", 0.7 * external_scale)
    add(weights, "self", "sense", 0.6 * external_scale)
    add(weights, "action", "sense", 0.8 * external_scale)

    # Useful integrated loop.
    add(weights, "valence", "action", 0.8)
    add(weights, "action", "valence", 0.9)
    add(weights, "imagination", "memory", 0.9)
    add(weights, "action", "imagination", 0.9)
    add(weights, "self", "memory", 0.8)
    add(weights, "memory", "self", 0.7)
    add(weights, "self", "action", 0.6)
    add(weights, "valence", "self", 0.5)

    # Potentially delusional internal recurrence.
    add(weights, "memory", "memory", 0.45 * internal_scale)
    add(weights, "valence", "valence", 0.45 * internal_scale)
    add(weights, "imagination", "imagination", 0.75 * internal_scale)
    add(weights, "self", "self", 0.75 * internal_scale)
    add(weights, "imagination", "self", 0.8 * internal_scale)
    add(weights, "self", "imagination", 0.8 * internal_scale)
    add(weights, "valence", "imagination", 0.55 * internal_scale)
    add(weights, "action", "self", 0.55 * internal_scale)

    return weights, bias


def action_one_probability(dist):
    prob = 0.0
    for idx, bits in enumerate(all_states(len(NODES))):
        if bits[ACTION]:
            prob += dist[idx]
    return float(prob)


def action_sensitivity(weights, bias, node):
    states = all_states(len(NODES))
    diffs = []
    for state in states:
        if state[node] == 1:
            continue
        flipped = state.copy()
        flipped[node] = 1
        p0 = action_one_probability(transition_distribution(state, weights, bias))
        p1 = action_one_probability(transition_distribution(flipped, weights, bias))
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


def run_sweep():
    rows = []
    for internal_scale in np.linspace(0.0, 2.4, 9):
        weights, bias = make_system(internal_scale)
        phi = phi_proxy(weights, bias)["phi_proxy"]
        external_sense = action_sensitivity(weights, bias, SENSE)
        internal_sense = float(np.mean([action_sensitivity(weights, bias, node) for node in INTERNAL]))
        rows.append(
            {
                "internal_scale": float(internal_scale),
                "phi_proxy": float(phi),
                "external_grounding_ratio": grounding_ratio(weights),
                "action_sensitivity_external": external_sense,
                "action_sensitivity_internal": internal_sense,
                "delusion_risk_index": internal_sense / max(external_sense, 1e-9),
            }
        )
    return rows


def plot(rows):
    x = [r["internal_scale"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x, [r["phi_proxy"] for r in rows], marker="o", color="#8b5cf6")
    axes[0].set_title("Phi Proxy Does Not Automatically Rise As Internal Loops Dominate")
    axes[0].set_ylabel("Phi proxy")

    axes[1].plot(x, [r["external_grounding_ratio"] for r in rows], marker="o", label="external grounding", color="#16a3a6")
    axes[1].plot(x, [r["delusion_risk_index"] for r in rows], marker="o", label="internal/external action influence", color="#dc2626")
    axes[1].set_title("Grounding Falls While Internal Influence Rises")
    axes[1].set_ylabel("ratio")
    axes[1].legend()

    axes[2].plot(x, [r["action_sensitivity_external"] for r in rows], marker="o", label="external sense", color="#16a3a6")
    axes[2].plot(x, [r["action_sensitivity_internal"] for r in rows], marker="o", label="internal nodes", color="#ff8a00")
    axes[2].set_title("Action Becomes More Sensitive To Inner State Than Outer Sense")
    axes[2].set_xlabel("internal self/imagination loop scale")
    axes[2].set_ylabel("mean action sensitivity")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "delusional_integration_sweep.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows = run_sweep()
    (OUT / "delusional_integration_metrics.json").write_text(json.dumps(rows, indent=2))
    plot(rows)
    print("Delusional integration sweep complete")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
