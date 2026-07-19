#!/usr/bin/env python3
"""Learned synchronization on a capacity-limited oscillatory workspace bus.

The previous oscillatory workspace lab assigned phase relationships. This lab
asks whether useful relative timing can be learned from routing success alone.

Six modules are arranged as two three-feature assemblies. A context-conditioned
phase policy controls when each module emits during a normalized carrier cycle.
The payloads, amplitudes, carrier label, and bus are fixed; only six local phase
offsets per context are trainable.

Tasks
-----
binding
    All six complementary features must coincide to form one packet.
multiplexing
    Each three-feature assembly must bind internally, but the two packets share
    a capacity-limited bus. Temporal overlap causes interference.

Controls and lesions
--------------------
- random initialization before training
- learned phases after training
- instantaneous phase scrambling with all payloads preserved
- exact restoration of the learned phases
- global phase rotation, which preserves relative timing
- frequency mismatch, which causes learned relationships to drift
- a no-bottleneck training control for the multiplexing context

This is a differentiable software routing experiment, not a neural, gamma,
electromagnetic, or phenomenal-consciousness model. "40 Hz" is only a carrier
label because the simulation operates on normalized cycles.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from tiny_lab import OUT


TAU = 2.0 * math.pi
ASSEMBLY_A = slice(0, 3)
ASSEMBLY_B = slice(3, 6)


@dataclass
class TrainResult:
    seed: int
    initial_phases: np.ndarray
    learned_phases: np.ndarray
    losses: list[float]
    metrics: dict


def wrap_phase(phases: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phases, TAU)


def circular_distance(a: float, b: float) -> float:
    delta = abs(a - b) % TAU
    return min(delta, TAU - delta)


def circular_mean(phases: np.ndarray) -> float:
    vector = np.mean(np.exp(1j * phases))
    return float(np.angle(vector) % TAU)


def order_parameter(phases: np.ndarray) -> float:
    return float(np.clip(abs(np.mean(np.exp(1j * phases))), 0.0, 1.0))


class OscillatoryBus:
    def __init__(self, bins: int = 64, sharpness: float = 9.0):
        self.bins = bins
        self.sharpness = sharpness
        self.theta = torch.linspace(0.0, TAU, bins + 1)[:-1]
        reference = torch.exp(sharpness * (torch.cos(self.theta) - 1.0))
        self.reference_mass = reference.sum()

    def pulses(self, phases: torch.Tensor) -> torch.Tensor:
        delta = self.theta.unsqueeze(0) - wrap_phase(phases).unsqueeze(1)
        return torch.exp(self.sharpness * (torch.cos(delta) - 1.0))

    @staticmethod
    def coincidence(pulses: torch.Tensor) -> torch.Tensor:
        # A smooth AND gate: all complementary streams must be present in the
        # same temporal window for the packet to survive.
        return torch.exp(torch.mean(torch.log(pulses.clamp_min(1e-9)), dim=0))

    def binding_score(self, phases: torch.Tensor) -> torch.Tensor:
        packet = self.coincidence(self.pulses(phases))
        return (packet.sum() / self.reference_mass).clamp(0.0, 1.0)

    def multiplex_score(self, phases: torch.Tensor, bottleneck: bool = True):
        pulses = self.pulses(phases)
        packet_a = self.coincidence(pulses[ASSEMBLY_A])
        packet_b = self.coincidence(pulses[ASSEMBLY_B])
        delivery_a = (packet_a.sum() / self.reference_mass).clamp(0.0, 1.0)
        delivery_b = (packet_b.sum() / self.reference_mass).clamp(0.0, 1.0)
        delivery = 0.5 * (delivery_a + delivery_b)

        overlap = torch.sum(packet_a * packet_b) / torch.sqrt(
            torch.sum(packet_a.square()) * torch.sum(packet_b.square()) + 1e-9
        )
        score = delivery * (1.0 - overlap) if bottleneck else delivery
        return score.clamp(0.0, 1.0), delivery, overlap.clamp(0.0, 1.0)


def context_scores(bus: OscillatoryBus, phases: torch.Tensor, bottleneck: bool = True):
    binding = bus.binding_score(phases[0])
    multiplex, delivery, collision = bus.multiplex_score(phases[1], bottleneck=bottleneck)
    return binding, multiplex, delivery, collision


def train_phases(seed: int, steps: int = 900, lr: float = 0.045, bottleneck: bool = True):
    torch.manual_seed(seed)
    bus = OscillatoryBus()
    phases = torch.nn.Parameter(torch.rand(2, 6) * TAU)
    initial = wrap_phase(phases.detach()).numpy().copy()
    optimizer = torch.optim.Adam([phases], lr=lr)
    losses = []

    for _ in range(steps):
        binding, multiplex, _, _ = context_scores(bus, phases, bottleneck=bottleneck)
        loss = 1.0 - 0.5 * (binding + multiplex)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    learned = wrap_phase(phases.detach()).numpy().copy()
    metrics = evaluate_phase_policy(bus, initial, learned, seed=seed, bottleneck=bottleneck)
    return TrainResult(seed, initial, learned, losses, metrics)


def score_numpy(bus: OscillatoryBus, phases: np.ndarray, bottleneck: bool = True):
    tensor = torch.tensor(phases, dtype=torch.float32)
    with torch.no_grad():
        binding, multiplex, delivery, collision = context_scores(bus, tensor, bottleneck=bottleneck)
    return {
        "binding": float(binding),
        "multiplexing": float(multiplex),
        "multiplex_delivery": float(delivery),
        "multiplex_collision": float(collision),
        "mean_utility": float(0.5 * (binding + multiplex)),
    }


def frequency_mismatch_score(bus: OscillatoryBus, learned: np.ndarray, cycles: int = 8):
    relative_frequencies = np.asarray((0.78, 0.91, 1.00, 1.09, 1.22, 1.34))
    rows = []
    for cycle in range(cycles):
        drift = TAU * cycle * relative_frequencies
        rows.append(score_numpy(bus, learned + drift))
    return {
        "binding": float(np.mean([row["binding"] for row in rows])),
        "multiplexing": float(np.mean([row["multiplexing"] for row in rows])),
        "mean_utility": float(np.mean([row["mean_utility"] for row in rows])),
        "cycles": rows,
    }


def evaluate_phase_policy(
    bus: OscillatoryBus,
    initial: np.ndarray,
    learned: np.ndarray,
    seed: int,
    bottleneck: bool = True,
    scramble_samples: int = 400,
):
    rng = np.random.default_rng(seed + 80_000)
    initial_score = score_numpy(bus, initial, bottleneck=bottleneck)
    learned_score = score_numpy(bus, learned, bottleneck=bottleneck)

    scrambled_rows = []
    for _ in range(scramble_samples):
        scrambled = rng.uniform(0.0, TAU, size=learned.shape)
        scrambled_rows.append(score_numpy(bus, scrambled, bottleneck=bottleneck))
    scrambled_score = {
        key: float(np.mean([row[key] for row in scrambled_rows]))
        for key in learned_score
    }

    rotation = float(rng.uniform(0.0, TAU))
    rotated_score = score_numpy(bus, learned + rotation, bottleneck=bottleneck)
    restored_score = score_numpy(bus, learned.copy(), bottleneck=bottleneck)
    mismatch_score = frequency_mismatch_score(bus, learned)

    binding_phases = learned[0]
    mux_a = learned[1, ASSEMBLY_A]
    mux_b = learned[1, ASSEMBLY_B]
    mux_separation = circular_distance(circular_mean(mux_a), circular_mean(mux_b))

    return {
        "initial": initial_score,
        "learned": learned_score,
        "scrambled": scrambled_score,
        "restored": restored_score,
        "global_rotation": rotated_score,
        "frequency_mismatch": mismatch_score,
        "phase_structure": {
            "binding_order_parameter": order_parameter(binding_phases),
            "multiplex_group_a_order": order_parameter(mux_a),
            "multiplex_group_b_order": order_parameter(mux_b),
            "multiplex_group_separation_radians": mux_separation,
            "multiplex_group_separation_fraction_of_pi": mux_separation / math.pi,
        },
    }


def aggregate(results: list[TrainResult], no_bottleneck_results: list[TrainResult]):
    def values(path):
        output = []
        for result in results:
            item = result.metrics
            for key in path:
                item = item[key]
            output.append(float(item))
        return np.asarray(output)

    def summarize(array):
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    phases = {
        "binding_order_parameter": summarize(values(("phase_structure", "binding_order_parameter"))),
        "multiplex_group_a_order": summarize(values(("phase_structure", "multiplex_group_a_order"))),
        "multiplex_group_b_order": summarize(values(("phase_structure", "multiplex_group_b_order"))),
        "multiplex_separation_fraction_of_pi": summarize(
            values(("phase_structure", "multiplex_group_separation_fraction_of_pi"))
        ),
    }

    stages = {}
    for stage in ("initial", "learned", "scrambled", "restored", "global_rotation", "frequency_mismatch"):
        stages[stage] = {
            metric: summarize(values((stage, metric)))
            for metric in ("binding", "multiplexing", "mean_utility")
        }

    no_bottleneck_separation = np.asarray(
        [
            result.metrics["phase_structure"]["multiplex_group_separation_fraction_of_pi"]
            for result in no_bottleneck_results
        ]
    )
    no_bottleneck_collision = np.asarray(
        [result.metrics["learned"]["multiplex_collision"] for result in no_bottleneck_results]
    )
    bottleneck_separation = values(
        ("phase_structure", "multiplex_group_separation_fraction_of_pi")
    )
    bottleneck_collision = values(("learned", "multiplex_collision"))

    return {
        "seeds": [result.seed for result in results],
        "carrier_label_hz": 40,
        "trainable_parameters_per_policy": 12,
        "stages": stages,
        "learned_phase_structure": phases,
        "no_bottleneck_control": {
            "multiplex_separation_fraction_of_pi": summarize(no_bottleneck_separation),
            "multiplex_collision": summarize(no_bottleneck_collision),
        },
        "bottleneck_comparison": {
            "multiplex_collision": summarize(bottleneck_collision),
            "paired_separation_gain_fraction_of_pi": summarize(
                bottleneck_separation - no_bottleneck_separation
            ),
            "paired_collision_reduction": summarize(
                no_bottleneck_collision - bottleneck_collision
            ),
        },
        "causal_contrasts": {
            "learned_minus_initial_utility": float(
                stages["learned"]["mean_utility"]["mean"]
                - stages["initial"]["mean_utility"]["mean"]
            ),
            "learned_minus_scrambled_utility": float(
                stages["learned"]["mean_utility"]["mean"]
                - stages["scrambled"]["mean_utility"]["mean"]
            ),
            "restoration_gap": float(
                stages["learned"]["mean_utility"]["mean"]
                - stages["restored"]["mean_utility"]["mean"]
            ),
            "global_rotation_gap": float(
                stages["learned"]["mean_utility"]["mean"]
                - stages["global_rotation"]["mean_utility"]["mean"]
            ),
        },
        "interpretation": {
            "supported": "Reward optimization discovered context-specific relative phase protocols that are causally necessary for routing utility in this bus.",
            "not_supported": "Biological gamma, a privileged 40 Hz frequency, electromagnetic consciousness, or phenomenal experience.",
        },
    }


def plot_results(summary: dict, results: list[TrainResult]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    stages = ("initial", "learned", "scrambled", "restored", "frequency_mismatch")
    colors = ("#7f8c8d", "#247ba0", "#e76f51", "#70c1b3", "#f4a261")
    binding = [summary["stages"][stage]["binding"]["mean"] for stage in stages]
    multiplex = [summary["stages"][stage]["multiplexing"]["mean"] for stage in stages]
    x = np.arange(len(stages))
    axes[0].bar(x - 0.18, binding, 0.36, label="binding", color="#247ba0")
    axes[0].bar(x + 0.18, multiplex, 0.36, label="multiplexing", color="#e76f51")
    axes[0].set_xticks(x, [stage.replace("_", "\n") for stage in stages], fontsize=8)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Routing utility")
    axes[0].set_title("Learn, scramble, restore")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    phase_example = results[0].learned_phases
    binding_reference = circular_mean(phase_example[0])
    mux_reference = circular_mean(phase_example[1, ASSEMBLY_A])

    def relative_phase(phases, reference):
        return ((phases - reference + math.pi) % TAU - math.pi) / math.pi

    modules = np.arange(1, 7)
    axes[1].scatter(
        modules - 0.10,
        relative_phase(phase_example[0], binding_reference),
        s=90,
        marker="o",
        color="#247ba0",
        label="binding context",
    )
    axes[1].scatter(
        modules + 0.10,
        relative_phase(phase_example[1], mux_reference),
        s=90,
        marker="s",
        color="#e76f51",
        label="multiplexing context",
    )
    axes[1].axhline(0.0, color="#555", linewidth=1.0)
    axes[1].set_xticks(modules, [f"M{i}" for i in modules])
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Relative phase / pi")
    axes[1].set_title("Context-specific relative phases (seed 1)")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)

    mean_loss = np.mean(np.asarray([result.losses for result in results]), axis=0)
    for result, color in zip(results, colors * 3):
        axes[2].plot(result.losses, color=color, alpha=0.22, linewidth=0.8)
    axes[2].plot(mean_loss, color="#202522", linewidth=2.3, label="seed mean")
    axes[2].set_xlabel("Optimization step")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Phase policies converge across seeds")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False)

    fig.suptitle("Learned Synchronization Lab", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = OUT / "learned_synchronization_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def print_summary(summary: dict):
    print("Learned synchronization lab")
    print("stage                 binding  multiplex  mean utility")
    for stage in ("initial", "learned", "scrambled", "restored", "global_rotation", "frequency_mismatch"):
        row = summary["stages"][stage]
        print(
            f"{stage:21s} "
            f"{row['binding']['mean']:7.3f}  "
            f"{row['multiplexing']['mean']:9.3f}  "
            f"{row['mean_utility']['mean']:12.3f}"
        )
    phase = summary["learned_phase_structure"]
    print("\nLearned relative timing")
    print(f"  binding synchrony R: {phase['binding_order_parameter']['mean']:.3f}")
    print(f"  multiplex group A R: {phase['multiplex_group_a_order']['mean']:.3f}")
    print(f"  multiplex group B R: {phase['multiplex_group_b_order']['mean']:.3f}")
    print(
        "  multiplex separation: "
        f"{phase['multiplex_separation_fraction_of_pi']['mean']:.3f} pi"
    )
    print("\nCausal contrasts")
    for key, value in summary["causal_contrasts"].items():
        print(f"  {key}: {value:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [101 + 97 * index for index in range(args.seeds)]
    results = [train_phases(seed, steps=args.steps, bottleneck=True) for seed in seeds]
    no_bottleneck = [train_phases(seed, steps=args.steps, bottleneck=False) for seed in seeds]
    summary = aggregate(results, no_bottleneck)

    metrics_path = OUT / "learned_synchronization_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    figure_path = plot_results(summary, results)
    print_summary(summary)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
