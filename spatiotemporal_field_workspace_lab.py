#!/usr/bin/env python3
"""Pockett-inspired spatial superposition coupled to temporal coordination.

The task binds a spatial payload identity to a relative temporal phase context.
It compares compact discrete messages, an equally distributed random code,
spatial-only fields, temporal-only pulses, and a coupled 2D+time field.

This is a software representation experiment. A NumPy field is not a physical
electromagnetic field and cannot test Pockett's identity claim about qualia.
"""

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


MODULES = 6
SPATIAL_IDENTITIES = 4
TEMPORAL_CONTEXTS = 2
CLASSES = SPATIAL_IDENTITIES * TEMPORAL_CONTEXTS
GRID_SIZE = 16
TIME_BINS = 8
ARCHITECTURES = (
    "discrete_message",
    "random_distributed_code",
    "spatial_field_only",
    "temporal_phase_only",
    "spatiotemporal_field",
)
LESIONS = (
    "clean",
    "source_drop_50",
    "readout_mask_50",
    "spatial_scramble",
    "phase_scramble",
    "space_phase_scramble",
)


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def make_spatial_basis():
    coordinates = np.linspace(-1.0, 1.0, GRID_SIZE)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    angles = np.linspace(0.0, 2.0 * np.pi, MODULES, endpoint=False)
    centers = np.stack([0.62 * np.cos(angles), 0.62 * np.sin(angles)], axis=1)
    basis = []
    for center_x, center_y in centers:
        radius_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        field = np.exp(-radius_sq / (2.0 * 0.27**2))
        basis.append(field / np.linalg.norm(field))
    return np.asarray(basis, dtype=np.float32).reshape(MODULES, -1)


def make_codebook(seed):
    rng = np.random.default_rng(seed)
    codes = []
    while len(codes) < SPATIAL_IDENTITIES:
        candidate = rng.choice([-1.0, 1.0], size=MODULES).astype(np.float32)
        if abs(float(np.sum(candidate))) > 2.0:
            continue
        if any(np.array_equal(candidate, existing) for existing in codes):
            continue
        codes.append(candidate)
    # Context is encoded by relative phase, not a global clock rotation.
    phase_patterns = np.asarray(
        [
            [0, 0, 0, np.pi, np.pi, np.pi],
            [0, np.pi, 0, np.pi, 0, np.pi],
        ],
        dtype=np.float32,
    )
    return np.stack(codes), phase_patterns


def sample_packets(labels, codes, phase_patterns, rng, payload_noise=0.16, phase_noise=0.10):
    identities = labels // TEMPORAL_CONTEXTS
    contexts = labels % TEMPORAL_CONTEXTS
    payloads = codes[identities] + rng.normal(0.0, payload_noise, (len(labels), MODULES))
    phases = phase_patterns[contexts] + rng.normal(0.0, phase_noise, (len(labels), MODULES))
    active = np.ones((len(labels), MODULES), dtype=np.float32)
    return payloads.astype(np.float32), phases.astype(np.float32), active


def temporal_waves(phases):
    clock = np.linspace(0.0, 2.0 * np.pi, TIME_BINS, endpoint=False)
    waves = np.exp(2.4 * np.cos(clock[None, None, :] - phases[:, :, None]))
    waves /= np.linalg.norm(waves, axis=2, keepdims=True)
    return waves.astype(np.float32)


def encode(architecture, payloads, phases, active, spatial_basis, random_projection):
    signed_payloads = payloads * active
    phase_messages = np.concatenate(
        [signed_payloads, active * np.cos(phases), active * np.sin(phases)], axis=1
    )
    if architecture == "discrete_message":
        representation = phase_messages
    elif architecture == "random_distributed_code":
        representation = phase_messages @ random_projection
    elif architecture == "spatial_field_only":
        representation = signed_payloads @ spatial_basis
    elif architecture == "temporal_phase_only":
        representation = (active[:, :, None] * temporal_waves(phases)).reshape(len(payloads), -1)
    elif architecture == "spatiotemporal_field":
        waves = temporal_waves(phases)
        field = np.einsum("nm,nmt,ms->nts", signed_payloads, waves, spatial_basis)
        representation = field.reshape(len(payloads), -1)
    else:
        raise ValueError(architecture)
    return normalize_rows(representation.astype(np.float32))


def lesion_packets(payloads, phases, active, lesion, rng):
    payloads = payloads.copy()
    phases = phases.copy()
    active = active.copy()
    if lesion == "source_drop_50":
        active *= (rng.random(active.shape) >= 0.5)
    if lesion in {"phase_scramble", "space_phase_scramble"}:
        phases = rng.uniform(0.0, 2.0 * np.pi, phases.shape).astype(np.float32)
    return payloads, phases, active


def lesion_representation(representation, architecture, lesion, spatial_permutation, rng):
    damaged = representation.copy()
    if lesion == "readout_mask_50":
        damaged *= rng.random(damaged.shape) >= 0.5

    if lesion in {"spatial_scramble", "space_phase_scramble"}:
        if architecture == "spatial_field_only":
            damaged = damaged[:, spatial_permutation]
        elif architecture == "spatiotemporal_field":
            shaped = damaged.reshape(len(damaged), TIME_BINS, GRID_SIZE * GRID_SIZE)
            damaged = shaped[:, :, spatial_permutation].reshape(len(damaged), -1)
        else:
            # The lesion is undefined for representations without spatial axes.
            return None
    if lesion == "phase_scramble" and architecture not in {
        "temporal_phase_only",
        "spatiotemporal_field",
        "discrete_message",
        "random_distributed_code",
    }:
        return None
    if lesion == "space_phase_scramble" and architecture != "spatiotemporal_field":
        return None
    return normalize_rows(damaged)


def fit_prototypes(representations, labels):
    prototypes = np.stack(
        [np.mean(representations[labels == label], axis=0) for label in range(CLASSES)]
    )
    return normalize_rows(prototypes)


def accuracy(representations, labels, prototypes):
    predictions = np.argmax(representations @ prototypes.T, axis=1)
    return float(np.mean(predictions == labels))


def run_seed(seed, train_per_class=100, test_per_class=200):
    rng = np.random.default_rng(seed)
    codes, phase_patterns = make_codebook(7021)
    spatial_basis = make_spatial_basis()
    full_dimension = TIME_BINS * GRID_SIZE * GRID_SIZE
    random_projection = rng.normal(0.0, 1.0, (MODULES * 3, full_dimension)).astype(np.float32)
    random_projection /= np.linalg.norm(random_projection, axis=1, keepdims=True)
    spatial_permutation = rng.permutation(GRID_SIZE * GRID_SIZE)

    train_labels = np.repeat(np.arange(CLASSES), train_per_class)
    test_labels = np.repeat(np.arange(CLASSES), test_per_class)
    rng.shuffle(train_labels)
    rng.shuffle(test_labels)
    train_packets = sample_packets(train_labels, codes, phase_patterns, rng)
    test_packets = sample_packets(test_labels, codes, phase_patterns, rng)

    rows = []
    for architecture in ARCHITECTURES:
        train_representation = encode(
            architecture, *train_packets, spatial_basis, random_projection
        )
        prototypes = fit_prototypes(train_representation, train_labels)
        for lesion in LESIONS:
            packet_rng = np.random.default_rng(seed * 100 + LESIONS.index(lesion))
            packets = lesion_packets(*test_packets, lesion, packet_rng)
            representation = encode(
                architecture, *packets, spatial_basis, random_projection
            )
            representation = lesion_representation(
                representation,
                architecture,
                lesion,
                spatial_permutation,
                packet_rng,
            )
            if representation is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "lesion": lesion,
                    "accuracy": accuracy(representation, test_labels, prototypes),
                }
            )
    return rows


def summarize(rows):
    summary = {}
    for architecture in ARCHITECTURES:
        for lesion in LESIONS:
            values = [
                row["accuracy"]
                for row in rows
                if row["architecture"] == architecture and row["lesion"] == lesion
            ]
            if values:
                summary[f"{architecture}:{lesion}"] = {
                    "architecture": architecture,
                    "lesion": lesion,
                    "mean_accuracy": float(np.mean(values)),
                    "std_accuracy": float(np.std(values)),
                    "min_accuracy": float(np.min(values)),
                    "max_accuracy": float(np.max(values)),
                    "seeds": len(values),
                }
    return summary


def plot_summary(summary, path):
    clean = [summary[f"{name}:clean"]["mean_accuracy"] for name in ARCHITECTURES]
    source = [summary[f"{name}:source_drop_50"]["mean_accuracy"] for name in ARCHITECTURES]
    readout = [summary[f"{name}:readout_mask_50"]["mean_accuracy"] for name in ARCHITECTURES]
    field_lesions = [
        summary["spatiotemporal_field:clean"]["mean_accuracy"],
        summary["spatiotemporal_field:spatial_scramble"]["mean_accuracy"],
        summary["spatiotemporal_field:phase_scramble"]["mean_accuracy"],
        summary["spatiotemporal_field:space_phase_scramble"]["mean_accuracy"],
    ]
    labels = ["discrete", "random\ncode", "spatial", "temporal", "space+time"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, clean, width, label="clean", color="#2a9d8f")
    axes[0].bar(x, source, width, label="50% source loss", color="#e9c46a")
    axes[0].bar(x + width, readout, width, label="50% readout mask", color="#457b9d")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Architecture and damage location")
    axes[0].legend()

    lesion_labels = ["clean", "space\nscramble", "phase\nscramble", "both"]
    axes[1].bar(lesion_labels, field_lesions, color=["#2a9d8f", "#e76f51", "#f4a261", "#9b2226"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Causal coordinates of the coupled field")
    for axis in axes:
        axis.set_ylabel("Binding accuracy")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Spatial field superposition coupled to relative temporal phase")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    set_seed(117)
    seed_count = args.seeds or (4 if args.quick else 24)
    train_per_class = 35 if args.quick else 100
    test_per_class = 60 if args.quick else 200
    rows = []
    for seed in range(seed_count):
        rows.extend(run_seed(9100 + seed, train_per_class, test_per_class))
    summary = summarize(rows)

    metrics_path = OUT / "spatiotemporal_field_workspace_metrics.json"
    figure_path = OUT / "spatiotemporal_field_workspace_summary.png"
    payload = {
        "question": "Can spatial superposition and relative phase jointly encode a bound workspace state?",
        "task": {
            "spatial_identities": SPATIAL_IDENTITIES,
            "temporal_contexts": TEMPORAL_CONTEXTS,
            "classes": CLASSES,
            "grid": [GRID_SIZE, GRID_SIZE],
            "time_bins": TIME_BINS,
            "train_samples_per_class": train_per_class,
            "test_samples_per_class": test_per_class,
        },
        "summary": summary,
        "per_seed": rows,
        "claim_boundary": (
            "This tests software spatial coding, temporal coding, redundancy, and causal "
            "coordinate lesions. It does not instantiate a physical electromagnetic field, "
            "test qualia, or establish Pockett's identity theory."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    plot_summary(summary, figure_path)

    print("\nSpatiotemporal field workspace")
    print("architecture                 clean  source50  readout50")
    for architecture in ARCHITECTURES:
        print(
            f"{architecture:28s} "
            f"{summary[f'{architecture}:clean']['mean_accuracy']:.3f}  "
            f"{summary[f'{architecture}:source_drop_50']['mean_accuracy']:.3f}     "
            f"{summary[f'{architecture}:readout_mask_50']['mean_accuracy']:.3f}"
        )
    print("\nCoupled-field coordinate lesions")
    for lesion in ("clean", "spatial_scramble", "phase_scramble", "space_phase_scramble"):
        print(f"{lesion:24s} {summary[f'spatiotemporal_field:{lesion}']['mean_accuracy']:.3f}")
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
