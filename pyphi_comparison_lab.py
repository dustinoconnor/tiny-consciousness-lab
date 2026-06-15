#!/usr/bin/env python3
"""Compare the repo's Phi proxy against PyPhi on tiny systems.

This is the first bridge from our transparent partition-KL proxy toward a
standard IIT toolchain.

Important caveats:

- PyPhi 1.2.0 is an IIT 3.x-era library, not a full IIT 4.0 implementation.
- PyPhi's subsystem phi is state-specific; our proxy averages over all states.
- The comparison is restricted to tiny 3-node systems because exact IIT-style
  calculations grow combinatorially.

The point is not to claim "official consciousness math." The point is to ask
whether our proxy ranks tiny architectures in a way that roughly tracks a
standard integrated-information implementation.
"""

import collections
import collections.abc
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from exact_phi_lab import OUT, all_states, phi_proxy, state_index, systems, transition_distribution


def import_pyphi():
    """Import PyPhi with small compatibility patches for modern Python."""
    os.environ["PYPHI_WELCOME_OFF"] = "yes"
    for name in ["Iterable", "Mapping", "MutableMapping", "Sequence"]:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    import pyphi

    # Keep this small script single-process. Multiprocessing is unnecessary for
    # 3-node systems and causes trouble when launched from some app contexts.
    for key, value in {
        "NUMBER_OF_CORES": 1,
        "PARALLEL_CONCEPT_EVALUATION": False,
        "PARALLEL_CUT_EVALUATION": False,
        "PROGRESS_BARS": False,
    }.items():
        if hasattr(pyphi.config, key):
            setattr(pyphi.config, key, value)
    return pyphi


def weights_to_state_by_node_tpm(weights, bias, noise=0.04):
    """Convert our transition distribution into PyPhi's state-by-node TPM.

    Shape is (2, 2, ..., n). For each current binary state, the last dimension
    stores each node's probability of being ON at t+1.
    """
    n = weights.shape[0]
    tpm = np.zeros((*(2 for _ in range(n)), n), dtype=float)
    for state in all_states(n):
        dist = transition_distribution(state, weights, bias, noise=noise)
        probs = np.zeros(n, dtype=float)
        for next_state in all_states(n):
            probs += dist[state_index(next_state)] * next_state
        tpm[tuple(state)] = probs
    return tpm


def pyphi_for_system(pyphi, name, weights, bias, states_to_sample):
    """Compute PyPhi subsystem phi for selected states."""
    tpm = weights_to_state_by_node_tpm(weights, bias)
    cm = (np.abs(weights) > 0.05).astype(int)
    network = pyphi.Network(tpm, cm=cm)
    state_results = {}
    errors = {}

    for state in states_to_sample:
        state_tuple = tuple(int(x) for x in state)
        try:
            subsystem = pyphi.Subsystem(network, state_tuple, network.node_indices)
            value = pyphi.compute.phi(subsystem)
            state_results["".join(map(str, state_tuple))] = float(value)
        except Exception as exc:
            errors["".join(map(str, state_tuple))] = f"{type(exc).__name__}: {exc}"

    return {
        "pyphi_state_phi": state_results,
        "pyphi_mean_sampled_phi": float(np.mean(list(state_results.values()))) if state_results else None,
        "pyphi_errors": errors,
    }


def plot_comparison(results, path):
    names = list(results)
    proxy = [results[name]["proxy_phi_mean_all_states"] for name in names]
    pyphi_vals = [
        0.0 if results[name]["pyphi_mean_sampled_phi"] is None else results[name]["pyphi_mean_sampled_phi"]
        for name in names
    ]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, proxy, width, label="repo proxy: all-state mean", color="#7c3aed")
    ax.bar(x + width / 2, pyphi_vals, width, label="PyPhi: sampled-state mean", color="#16a3a6")
    ax.set_title("Tiny-System Integrated Information: Proxy vs PyPhi")
    ax.set_ylabel("bits / Phi-like units")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    pyphi = import_pyphi()
    states_to_sample = [(0, 0, 0), (1, 0, 1), (1, 1, 1)]
    results = {}

    for name, (weights, bias) in systems(n=3).items():
        proxy = phi_proxy(weights, bias)
        pyphi_result = pyphi_for_system(pyphi, name, weights, bias, states_to_sample)
        results[name] = {
            "proxy_phi_mean_all_states": proxy["phi_proxy"],
            "proxy_state_phi": {
                "".join(map(str, state)): float(proxy["state_phi"][state_index(state)])
                for state in all_states(3)
            },
            **pyphi_result,
        }

    payload = {
        "note": (
            "PyPhi comparison on 3-node systems only. PyPhi is state-specific and IIT 3.x-era; "
            "the repo proxy is an all-state partition-KL mean. This compares ranking behavior, not identical definitions."
        ),
        "pyphi_version": getattr(pyphi, "__version__", "unknown"),
        "sampled_states": ["".join(map(str, s)) for s in states_to_sample],
        "results": results,
    }
    (OUT / "pyphi_comparison_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_comparison(results, OUT / "pyphi_comparison.png")
    print("PyPhi comparison lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
