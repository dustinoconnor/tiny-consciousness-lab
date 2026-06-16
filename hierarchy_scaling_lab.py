#!/usr/bin/env python3
"""Hierarchy scaling sweep.

The first hierarchical workspace lab showed that compressed summaries can make
executive routing more efficient than one monolithic controller. This lab asks
where that advantage breaks:

As the number of specialists grows, does one master become a bottleneck, and can
regional sub-masters fix the problem by compressing signals a second time?

Architectures:

- flat_monolith: every specialist sends raw noisy state into one global pool
- single_level_master: every specialist sends one compressed summary to master
- multi_level_deep_hierarchy: regional sub-masters compress summaries again

This is a deterministic toy routing model. The numbers are not meant as a claim
about biological scale. The point is to make the tradeoff visible: raw bandwidth
vs compression vs propagation delay.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def recovery_delay(load, depth, architecture):
    """Synthetic delay from raw shock to action adjustment."""
    if architecture == "flat_monolith":
        return 1.0 + 0.060 * load
    if architecture == "single_level_master":
        return 1.2 + 0.030 * load
    if architecture == "multi_level_deep_hierarchy":
        return 2.8 + 0.006 * load + 0.30 * max(0, depth - 2)
    raise ValueError(architecture)


def simulate_architecture(architecture, n, trials=96, seed=512):
    """Run routing shocks for one architecture at one specialist count."""
    rng = np.random.default_rng(seed + n * 17 + len(architecture))
    if architecture == "flat_monolith":
        channel_load = float(n * 4)
        depth = 1
        compression_loss = 0.0
        local_noise_gain = 0.018
    elif architecture == "single_level_master":
        channel_load = float(n)
        depth = 2
        compression_loss = 0.035
        local_noise_gain = 0.010
    elif architecture == "multi_level_deep_hierarchy":
        group_size = 8
        regional_masters = int(math.ceil(n / group_size))
        channel_load = float(regional_masters)
        depth = 3
        compression_loss = 0.075
        local_noise_gain = 0.004
    else:
        raise ValueError(architecture)

    routing_delays = []
    delusions = []
    successes = []
    costs = []
    for _ in range(trials):
        shock_strength = rng.uniform(0.45, 1.0)
        local_conflict = rng.beta(2.0, 5.0, size=n) * shock_strength
        false_confidence = rng.beta(1.4, 7.5, size=n)

        if architecture == "flat_monolith":
            conflict_signal = float(np.mean(local_conflict) + 0.25 * np.std(local_conflict))
            false_signal = float(np.mean(false_confidence) + local_noise_gain * math.sqrt(n))
        elif architecture == "single_level_master":
            conflict_signal = float(np.mean(local_conflict))
            # One master gets every summary. Past a moderate count, competing
            # summaries become an attention/routing load even though compressed.
            false_signal = float(np.mean(false_confidence) + local_noise_gain * math.log2(n))
        else:
            group_size = 8
            group_conflicts = []
            group_false = []
            for start in range(0, n, group_size):
                sl = slice(start, min(n, start + group_size))
                group_conflicts.append(float(np.mean(local_conflict[sl]) + 0.10 * np.max(local_conflict[sl])))
                group_false.append(float(np.mean(false_confidence[sl])))
            conflict_signal = float(np.mean(group_conflicts) - compression_loss * 0.25)
            false_signal = float(np.mean(group_false) + local_noise_gain * math.log2(max(2, len(group_conflicts))))

        load_pressure = channel_load / 24.0
        if architecture == "multi_level_deep_hierarchy":
            load_pressure *= 0.42
        elif architecture == "single_level_master":
            load_pressure *= 0.78

        delay = recovery_delay(channel_load, depth, architecture) * (0.75 + shock_strength)
        delusion = sigmoid(2.2 * false_signal + 1.8 * load_pressure + 0.22 * delay - 2.25)
        routing_quality = sigmoid(4.0 * conflict_signal - 0.55 * delay - compression_loss + 1.8)
        success_prob = np.clip(routing_quality * (1.0 - 0.75 * delusion), 0.0, 1.0)
        success = float(rng.random() < success_prob)

        compute_cost = 0.0018 * channel_load + 0.018 * depth + 0.010 * delay
        if architecture == "multi_level_deep_hierarchy":
            compute_cost += 0.012  # regional coordination overhead
        costs.append(compute_cost)
        routing_delays.append(delay)
        delusions.append(delusion)
        successes.append(success)

    accuracy = float(np.mean(successes))
    delay = float(np.mean(routing_delays))
    delusion = float(np.mean(delusions))
    cost = float(np.mean(costs))
    efficiency = float(accuracy - 0.18 * cost - 0.35 * delusion - 0.015 * delay)
    return {
        "num_specialists": int(n),
        "architecture": architecture,
        "master_channel_load": channel_load,
        "routing_propagation_delay": delay,
        "cross_module_delusion_rate": delusion,
        "global_accuracy": accuracy,
        "compute_cost": cost,
        "global_efficiency": efficiency,
    }


def run_sweep(sizes=(4, 8, 16, 32, 64, 128)):
    architectures = ["flat_monolith", "single_level_master", "multi_level_deep_hierarchy"]
    return {
        arch: [simulate_architecture(arch, n) for n in sizes]
        for arch in architectures
    }


def plot_metric(results, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for arch, rows in results.items():
        ax.plot([r["num_specialists"] for r in rows], [r[metric] for r in rows], marker="o", lw=2.5, label=arch)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("number of specialists")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Hierarchy Scaling Sweep: {ylabel}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(results, path):
    metrics = [
        ("global_efficiency", "efficiency"),
        ("global_accuracy", "accuracy"),
        ("routing_propagation_delay", "delay"),
        ("cross_module_delusion_rate", "delusion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for arch, rows in results.items():
            ax.plot([r["num_specialists"] for r in rows], [r[metric] for r in rows], marker="o", lw=2.3, label=arch)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64, 128])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Hierarchy Scaling: When a Single Master Becomes a Bottleneck", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def find_crossover(results):
    sizes = [r["num_specialists"] for r in results["single_level_master"]]
    for i, n in enumerate(sizes):
        single = results["single_level_master"][i]["global_efficiency"]
        deep = results["multi_level_deep_hierarchy"][i]["global_efficiency"]
        if deep > single:
            return int(n)
    return None


def main():
    set_seed(512)
    OUT.mkdir(exist_ok=True)
    results = run_sweep()
    payload = {
        "note": (
            "Synthetic routing-load sweep. Tests when monolithic control, single-level master routing, "
            "and multi-level regional hierarchy win as specialist count increases."
        ),
        "results": results,
        "crossover_specialist_count": find_crossover(results),
        "thesis": (
            "Hierarchy helps scaling, but one global master does not scale forever. "
            "At larger specialist counts, regional compression protects the master from routing overload."
        ),
    }
    (OUT / "hierarchy_scaling_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(results, OUT / "hierarchy_scaling_summary.png")
    plot_metric(results, "master_channel_load", "master channel load", OUT / "hierarchy_scaling_channel_load.png")
    print("Hierarchy scaling lab complete")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    for arch, rows in results.items():
        print(arch)
        for r in rows:
            print(
                f"  N={r['num_specialists']:3d} load={r['master_channel_load']:5.1f} "
                f"delay={r['routing_propagation_delay']:.2f} delusion={r['cross_module_delusion_rate']:.3f} "
                f"acc={r['global_accuracy']:.3f} eff={r['global_efficiency']:.3f}"
            )


if __name__ == "__main__":
    main()
