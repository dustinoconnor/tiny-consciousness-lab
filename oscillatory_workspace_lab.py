#!/usr/bin/env python3
"""Oscillatory workspace binding and phase-intervention lab.

This is a software-level test inspired by Llinas's thalamocortical synchrony
hypothesis. It does not simulate neurons, biological gamma, or phenomenal
consciousness. It asks a narrower engineering question:

    Can shared phase organize distributed features into a more accurate,
    reportable, and causally useful workspace packet?

Every condition receives the same feature values and number of events. Only
their timing differs. Four modules independently emit color, shape, motion,
and valence observations for two objects. Object identity is hidden from the
workspace; temporal coincidence is its only binding cue.

The conditions separate frequency from phase coherence:

- coherent_40hz: shared 40 Hz carrier with low phase jitter.
- coherent_20hz_control: a slower but equally coherent carrier.
- coherent_40hz_jitter: shared carrier with degraded temporal precision.
- same_40hz_unlocked: every module runs at 40 Hz with a stable private phase.
- mixed_frequency: module frequencies drift relative to one another.
- phase_shift_intervention: coherent timing except the valence stream is
  shifted by half a cycle, a targeted causal disruption.
- asynchronous: event phases are independent on every cycle.

The numeric frequency is an experimental label here. Biological claims about
40 Hz require neuroscience; this lab tests phase-based routing in software.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT


MODULES = ("color", "shape", "motion", "valence")
VALUES = {
    "color": ("amber", "cyan"),
    "shape": ("round", "angular"),
    "motion": ("approach", "recede"),
    "valence": ("positive", "negative"),
}

CONDITIONS = (
    "coherent_40hz",
    "coherent_20hz_control",
    "coherent_40hz_jitter",
    "same_40hz_unlocked",
    "mixed_frequency",
    "phase_shift_intervention",
    "asynchronous",
)


@dataclass(frozen=True)
class Emission:
    module: str
    value: str
    phase: float


def circular_distance(a: float, b: float) -> float:
    delta = abs(a - b) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def circular_mean(phases: list[float]) -> float:
    return float(math.atan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))) % (2.0 * math.pi))


def action_for(packet: dict[str, str]) -> str:
    """A conjunction forces correct binding instead of single-feature guessing."""
    if packet["shape"] == "round":
        return "approach_left" if packet["motion"] == "approach" else "hold"
    return "approach_right" if packet["motion"] == "approach" else "retreat"


def sample_scene(rng: np.random.Generator) -> tuple[list[dict[str, str]], int]:
    positive_index = int(rng.integers(0, 2))
    objects: list[dict[str, str]] = []
    for object_index in range(2):
        packet = {
            module: str(rng.choice(VALUES[module]))
            for module in MODULES
            if module != "valence"
        }
        packet["valence"] = "positive" if object_index == positive_index else "negative"
        objects.append(packet)

    # Avoid scenes in which a false bind accidentally yields the same report.
    while action_for(objects[0]) == action_for(objects[1]):
        objects[1]["shape"] = str(rng.choice(VALUES["shape"]))
        objects[1]["motion"] = str(rng.choice(VALUES["motion"]))
    return objects, positive_index


def condition_parameters(condition: str, rng: np.random.Generator):
    base_frequency = 20.0 if condition == "coherent_20hz_control" else 40.0
    if condition == "mixed_frequency":
        frequencies = dict(zip(MODULES, (31.0, 37.0, 43.0, 49.0)))
    else:
        frequencies = {module: base_frequency for module in MODULES}

    if condition == "same_40hz_unlocked":
        module_offsets = {module: float(rng.uniform(0.0, 2.0 * math.pi)) for module in MODULES}
    else:
        module_offsets = {module: 0.0 for module in MODULES}

    jitter = 0.07
    if condition == "coherent_40hz_jitter":
        jitter = 0.62
    elif condition == "asynchronous":
        jitter = math.pi

    if condition == "phase_shift_intervention":
        module_offsets["valence"] = math.pi

    return frequencies, module_offsets, jitter


def emit_scene(
    objects: list[dict[str, str]],
    condition: str,
    cycle: int,
    rng: np.random.Generator,
    parameters,
) -> list[Emission]:
    frequencies, offsets, jitter = parameters
    emissions: list[Emission] = []
    object_phases = (0.0, math.pi)

    for module in MODULES:
        for object_index, packet in enumerate(objects):
            if condition == "asynchronous":
                phase = float(rng.uniform(0.0, 2.0 * math.pi))
            else:
                drift = 2.0 * math.pi * cycle * (frequencies[module] / 40.0)
                phase = object_phases[object_index] + offsets[module] + drift
                phase += float(rng.normal(0.0, jitter))
            emissions.append(Emission(module, packet[module], phase % (2.0 * math.pi)))
    return emissions


def bind_workspace(emissions: list[Emission], window: float = 0.46):
    """Find the phase-centered packet with maximum temporal coherence."""
    candidates = [emission.phase for emission in emissions]
    best_packet = None
    best_score = -math.inf
    best_coherence = 0.0

    for center in candidates:
        packet: dict[str, str] = {}
        selected_phases: list[float] = []
        distances: list[float] = []
        for module in MODULES:
            choices = [emission for emission in emissions if emission.module == module]
            selected = min(choices, key=lambda item: circular_distance(item.phase, center))
            distance = circular_distance(selected.phase, center)
            packet[module] = selected.value
            selected_phases.append(selected.phase)
            distances.append(distance)

        coverage = sum(distance <= window for distance in distances) / len(MODULES)
        coherence = abs(np.mean(np.exp(1j * np.asarray(selected_phases))))
        positive_bonus = 0.18 if packet["valence"] == "positive" else 0.0
        score = 0.64 * coverage + 0.36 * coherence + positive_bonus
        if score > best_score:
            best_score = score
            best_packet = packet
            best_coherence = float(coherence)

    assert best_packet is not None
    confidence = float(np.clip((best_score - 0.35) / 0.83, 0.0, 1.0))
    return best_packet, confidence, best_coherence


def run_trial(condition: str, seed: int, cycles: int = 5):
    rng = np.random.default_rng(seed)
    objects, target_index = sample_scene(rng)
    target = objects[target_index]
    parameters = condition_parameters(condition, rng)
    cycle_rows = []

    for cycle in range(cycles):
        emissions = emit_scene(objects, condition, cycle, rng, parameters)
        packet, confidence, coherence = bind_workspace(emissions)
        cycle_rows.append(
            {
                "cycle": cycle,
                "confidence": confidence,
                "coherence": coherence,
                "binding_correct": packet == target,
                "report_correct": packet["valence"] == "positive" and packet == target,
                "action_correct": action_for(packet) == action_for(target),
                "report_action_consistent": (
                    (packet["valence"] == "positive")
                    == (action_for(packet) == action_for(target))
                ),
                "false_binding": packet != objects[0] and packet != objects[1],
            }
        )

    # Sustained coordination matters. Scoring every cycle prevents a drifting
    # system from hiding behind one initially aligned sample.
    return {
        "binding_correct": float(np.mean([row["binding_correct"] for row in cycle_rows])),
        "report_correct": float(np.mean([row["report_correct"] for row in cycle_rows])),
        "action_correct": float(np.mean([row["action_correct"] for row in cycle_rows])),
        "report_action_consistent": float(
            np.mean([row["report_action_consistent"] for row in cycle_rows])
        ),
        "false_binding": float(np.mean([row["false_binding"] for row in cycle_rows])),
        "confidence": float(np.mean([row["confidence"] for row in cycle_rows])),
        "coherence": float(np.mean([row["coherence"] for row in cycle_rows])),
        "cycles": cycle_rows,
    }


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int = 1500):
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run_experiment(trials: int = 1600, seed: int = 19):
    metrics = {}
    raw = {}
    bootstrap_rng = np.random.default_rng(seed + 100_000)
    metric_names = (
        "binding_correct",
        "report_correct",
        "action_correct",
        "report_action_consistent",
        "false_binding",
        "confidence",
        "coherence",
    )

    for condition_index, condition in enumerate(CONDITIONS):
        rows = [
            run_trial(condition, seed + condition_index * 100_000 + trial)
            for trial in range(trials)
        ]
        raw[condition] = rows
        metrics[condition] = {}
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in rows], dtype=float)
            metrics[condition][metric_name] = float(np.mean(values))
            metrics[condition][f"{metric_name}_95ci"] = bootstrap_interval(values, bootstrap_rng)

    coherent = metrics["coherent_40hz"]["binding_correct"]
    unlocked = metrics["same_40hz_unlocked"]["binding_correct"]
    intervention = metrics["phase_shift_intervention"]["binding_correct"]
    coherent_20 = metrics["coherent_20hz_control"]["binding_correct"]
    summary = {
        "trials_per_condition": trials,
        "seed": seed,
        "conditions": metrics,
        "contrasts": {
            "phase_lock_effect_vs_same_frequency_unlocked": coherent - unlocked,
            "valence_phase_intervention_effect": coherent - intervention,
            "coherent_40_minus_coherent_20": coherent - coherent_20,
        },
        "interpretation": {
            "supported_if_positive": "Shared phase can serve as a routing/binding coordinate in this explicit software mechanism.",
            "not_tested": "Biological gamma, phenomenal consciousness, or a privileged causal role for exactly 40 Hz.",
        },
    }
    return summary, raw


def plot_results(summary: dict, raw: dict, output: Path):
    labels = [condition.replace("_", "\n") for condition in CONDITIONS]
    binding = [summary["conditions"][condition]["binding_correct"] for condition in CONDITIONS]
    action = [summary["conditions"][condition]["action_correct"] for condition in CONDITIONS]
    false_binding = [summary["conditions"][condition]["false_binding"] for condition in CONDITIONS]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    x = np.arange(len(CONDITIONS))
    width = 0.27
    axes[0].bar(x - width, binding, width, label="bound packet correct", color="#247ba0")
    axes[0].bar(x, action, width, label="action correct", color="#70c1b3")
    axes[0].bar(x + width, false_binding, width, label="false binding", color="#e76f51")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Timing changes binding and downstream control")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    for condition, color in (
        ("coherent_40hz", "#247ba0"),
        ("coherent_40hz_jitter", "#f4a261"),
        ("mixed_frequency", "#e76f51"),
    ):
        cycle_accuracy = []
        cycles = len(raw[condition][0]["cycles"])
        for cycle in range(cycles):
            cycle_accuracy.append(np.mean([row["cycles"][cycle]["binding_correct"] for row in raw[condition]]))
        axes[1].plot(range(cycles), cycle_accuracy, marker="o", linewidth=2.2, label=condition, color=color)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Oscillatory cycle")
    axes[1].set_ylabel("Binding accuracy")
    axes[1].set_title("Frequency mismatch accumulates phase drift")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle("Oscillatory Workspace Lab", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: dict):
    print("Oscillatory workspace binding lab")
    print("condition                      bind    action  false-bind  coherence")
    for condition in CONDITIONS:
        values = summary["conditions"][condition]
        print(
            f"{condition:30s} "
            f"{values['binding_correct']:6.3f}  "
            f"{values['action_correct']:6.3f}  "
            f"{values['false_binding']:10.3f}  "
            f"{values['coherence']:9.3f}"
        )
    print("\nCausal contrasts")
    for name, value in summary["contrasts"].items():
        print(f"  {name}: {value:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary, raw = run_experiment(trials=args.trials, seed=args.seed)
    metrics_path = OUT / "oscillatory_workspace_metrics.json"
    figure_path = OUT / "oscillatory_workspace_summary.png"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(summary, raw, figure_path)
    print_summary(summary)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
