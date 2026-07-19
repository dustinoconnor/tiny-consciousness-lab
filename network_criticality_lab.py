#!/usr/bin/env python3
"""Test whether useful recurrent dynamics peak near an independently measured edge.

One frozen reservoir per seed is evaluated across a global recurrent-gain sweep.
Criticality is estimated from the driven largest Lyapunov exponent and activity
branching behavior before inspecting delayed-memory task performance.

This tests computational consequences of dynamical regime. Criticality is
common in non-conscious physical systems, so no result here is sufficient
evidence for phenomenal consciousness.
"""

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


def normalize_spectral_radius(matrix):
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return (matrix / max(radius, 1e-8)).astype(np.float32)


def make_reservoir(seed, units=48):
    rng = np.random.default_rng(seed)
    weights = normalize_spectral_radius(
        rng.normal(0.0, 1.0 / math.sqrt(units), (units, units))
    )
    inputs = rng.normal(0.0, 0.42, (units, 2)).astype(np.float32)
    return weights, inputs


def step_state(state, drive, weights, inputs, gain):
    return np.tanh(gain * (state @ weights.T) + drive @ inputs.T)


def driven_lyapunov(weights, inputs, gain, seed, steps=180, warmup=40, epsilon=1e-6):
    rng = np.random.default_rng(seed)
    units = weights.shape[0]
    state = rng.normal(0.0, 0.03, units).astype(np.float32)
    direction = rng.normal(0.0, 1.0, units).astype(np.float32)
    direction /= np.linalg.norm(direction)
    perturbed = state + epsilon * direction
    logs = []
    for tick in range(steps + warmup):
        drive = np.asarray([0.0, rng.normal(0.0, 0.10)], dtype=np.float32)
        state = step_state(state[None, :], drive[None, :], weights, inputs, gain)[0]
        perturbed = step_state(
            perturbed[None, :], drive[None, :], weights, inputs, gain
        )[0]
        delta = perturbed - state
        distance = float(np.linalg.norm(delta))
        growth = max(distance / epsilon, 1e-8)
        if tick >= warmup:
            logs.append(math.log(growth))
        if distance < 1e-12:
            direction = rng.normal(0.0, 1.0, units).astype(np.float32)
            direction /= np.linalg.norm(direction)
        else:
            direction = delta / distance
        perturbed = state + epsilon * direction
    return float(np.mean(logs))


def collect_driven_states(weights, inputs, gain, seed, steps=1050):
    rng = np.random.default_rng(seed)
    drive = rng.uniform(-1.0, 1.0, steps).astype(np.float32)
    states = np.zeros((steps, weights.shape[0]), dtype=np.float32)
    state = np.zeros(weights.shape[0], dtype=np.float32)
    for tick in range(steps):
        packet = np.asarray([[drive[tick], 0.0]], dtype=np.float32)
        state = step_state(state[None, :], packet, weights, inputs, gain)[0]
        states[tick] = state
    return drive, states


def ridge_fit(features, targets, ridge=1e-3):
    augmented = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    gram = augmented.T @ augmented + ridge * np.eye(augmented.shape[1])
    return np.linalg.solve(gram, augmented.T @ targets)


def ridge_predict(features, weights):
    augmented = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    return augmented @ weights


def memory_capacity(weights, inputs, gain, seed, max_lag=25):
    drive, states = collect_driven_states(weights, inputs, gain, seed)
    start = 100 + max_lag
    features = states[start:]
    targets = np.stack(
        [drive[start - lag : len(drive) - lag] for lag in range(1, max_lag + 1)],
        axis=1,
    )
    split = 650
    readout = ridge_fit(features[:split], targets[:split])
    predicted = ridge_predict(features[split:], readout)
    scores = []
    for lag in range(max_lag):
        truth = targets[split:, lag]
        residual = np.sum((truth - predicted[:, lag]) ** 2)
        total = np.sum((truth - np.mean(truth)) ** 2)
        scores.append(max(0.0, 1.0 - float(residual / max(total, 1e-8))))
    covariance = np.cov(features[split:].T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    effective_dimension = float(
        np.sum(eigenvalues) ** 2 / max(np.sum(eigenvalues**2), 1e-8)
    )
    saturation = float(np.mean(np.abs(features[split:]) > 0.95))
    return float(np.sum(scores)), effective_dimension, saturation


def delayed_cue_accuracy(weights, inputs, gain, seed, delay=55):
    rng = np.random.default_rng(seed)

    def make_batch(samples):
        labels = rng.choice([-1.0, 1.0], samples).astype(np.float32)
        state = np.zeros((samples, weights.shape[0]), dtype=np.float32)
        for tick in range(delay):
            packet = np.zeros((samples, 2), dtype=np.float32)
            if tick == 0:
                packet[:, 0] = labels
            packet[:, 1] = rng.normal(0.0, 0.24, samples)
            state = step_state(state, packet, weights, inputs, gain)
        return state, labels

    train_states, train_labels = make_batch(360)
    test_states, test_labels = make_batch(220)
    readout = ridge_fit(train_states, train_labels)
    predictions = ridge_predict(test_states, readout)
    return float(np.mean(np.sign(predictions) == test_labels))


def dynamic_range_db(weights, inputs, gain):
    amplitudes = np.logspace(-4, 0, 28).astype(np.float32)
    states = np.zeros((len(amplitudes), weights.shape[0]), dtype=np.float32)
    for tick in range(8):
        packets = np.zeros((len(amplitudes), 2), dtype=np.float32)
        if tick == 0:
            packets[:, 0] = amplitudes
        states = step_state(states, packets, weights, inputs, gain)
    responses = np.linalg.norm(states, axis=1)
    responses /= max(float(np.max(responses)), 1e-8)
    low_index = int(np.argmax(responses >= 0.1))
    high_index = int(np.argmax(responses >= 0.9))
    if high_index <= low_index:
        return 0.0
    return float(20.0 * np.log10(amplitudes[high_index] / amplitudes[low_index]))


def avalanche_metrics(weights, gain, threshold=0.05, trials=72, horizon=80):
    sizes = []
    durations = []
    ratios = []
    runaway = 0
    units = weights.shape[0]
    for trial in range(trials):
        state = np.zeros(units, dtype=np.float32)
        state[trial % units] = 0.55 if trial % 2 == 0 else -0.55
        size = 0
        duration = 0
        previous_activity = float(np.sum(np.abs(state)))
        for tick in range(horizon):
            state = np.tanh(gain * (weights @ state))
            activity = float(np.sum(np.abs(state)))
            count = int(np.sum(np.abs(state) > threshold))
            if count == 0:
                break
            size += count
            duration = tick + 1
            if previous_activity > 1e-8:
                ratios.append(activity / previous_activity)
            previous_activity = activity
        runaway += int(duration == horizon)
        sizes.append(size)
        durations.append(duration)
    mean_size = float(np.mean(sizes))
    return {
        "mean_avalanche_size": mean_size,
        "avalanche_size_cv": float(np.std(sizes) / max(mean_size, 1e-8)),
        "mean_avalanche_duration": float(np.mean(durations)),
        "branching_ratio": float(np.median(ratios)) if ratios else 0.0,
        "runaway_fraction": runaway / trials,
    }


def evaluate_gain(weights, inputs, gain, seed):
    lyapunov = driven_lyapunov(weights, inputs, gain, seed + 1)
    capacity, dimension, saturation = memory_capacity(weights, inputs, gain, seed + 2)
    delayed = delayed_cue_accuracy(weights, inputs, gain, seed + 3)
    avalanche = avalanche_metrics(weights, gain)
    return {
        "gain": float(gain),
        "lyapunov": lyapunov,
        "memory_capacity": capacity,
        "delayed_accuracy": delayed,
        "dynamic_range_db": dynamic_range_db(weights, inputs, gain),
        "effective_dimension": dimension,
        "saturation_fraction": saturation,
        **avalanche,
    }


def tune_gain_to_edge(weights, inputs, start_gain, seed, iterations=18):
    gain = float(start_gain)
    history = []
    for iteration in range(iterations):
        exponent = driven_lyapunov(
            weights, inputs, gain, seed + iteration, steps=80, warmup=25
        )
        history.append({"iteration": iteration, "gain": gain, "lyapunov": exponent})
        gain = float(np.clip(gain * math.exp(-0.32 * exponent), 0.25, 2.2))
    return gain, history


def summarize(rows, gains):
    summary = []
    for gain in gains:
        selected = [row for row in rows if abs(row["gain"] - gain) < 1e-7]
        entry = {"gain": float(gain), "seeds": len(selected)}
        for key in selected[0]:
            if key != "gain":
                entry[f"mean_{key}"] = float(np.mean([row[key] for row in selected]))
                entry[f"std_{key}"] = float(np.std([row[key] for row in selected]))
        summary.append(entry)
    return summary


def plot_summary(summary, path):
    gains = [row["gain"] for row in summary]
    lyapunov = [row["mean_lyapunov"] for row in summary]
    memory = [row["mean_memory_capacity"] for row in summary]
    accuracy = [row["mean_delayed_accuracy"] for row in summary]
    runaway = [row["mean_runaway_fraction"] for row in summary]
    dimension = [row["mean_effective_dimension"] for row in summary]
    critical_index = int(np.argmin(np.abs(lyapunov)))
    critical_gain = gains[critical_index]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(gains, lyapunov, marker="o", color="#e76f51")
    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set_title("Driven Lyapunov exponent")
    normalized_memory = np.asarray(memory) / max(memory)
    axes[0, 1].plot(gains, normalized_memory, marker="o", label="normalized memory", color="#457b9d")
    axes[0, 1].plot(gains, accuracy, marker="s", label="delayed accuracy", color="#2a9d8f")
    axes[0, 1].set_title("Functional performance")
    axes[0, 1].legend()
    axes[1, 0].plot(gains, runaway, marker="o", color="#f4a261")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_title("Persistent-avalanche fraction")
    axes[1, 1].plot(gains, dimension, marker="o", color="#9b2226")
    axes[1, 1].set_title("Effective state dimension")
    for axis in axes.flat:
        axis.axvline(critical_gain, color="#6a4c93", linestyle="--", alpha=0.8)
        axis.set_xlabel("Recurrent gain")
        axis.grid(alpha=0.2)
    fig.suptitle(f"Frozen-network gain sweep; empirical edge near g={critical_gain:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()

    set_seed(173)
    gains = np.linspace(0.50, 1.70, 5 if args.quick else 9)
    seed_count = args.seeds or (3 if args.quick else 8)
    rows = []
    tuning = []
    for seed_index in range(seed_count):
        seed = 7300 + seed_index
        weights, inputs = make_reservoir(seed)
        for gain in gains:
            rows.append(evaluate_gain(weights, inputs, float(gain), seed * 100))
            rows[-1]["seed"] = seed
        for start in (0.55, 1.55):
            final_gain, history = tune_gain_to_edge(
                weights, inputs, start, seed * 1000 + int(start * 100)
            )
            tuning.append({
                "seed": seed,
                "start_gain": start,
                "final_gain": final_gain,
                "history": history,
            })

    summary = summarize(rows, gains)
    mean_lyapunov = np.asarray([row["mean_lyapunov"] for row in summary])
    critical_index = int(np.argmin(np.abs(mean_lyapunov)))
    memory_index = int(np.argmax([row["mean_memory_capacity"] for row in summary]))
    accuracy_index = int(np.argmax([row["mean_delayed_accuracy"] for row in summary]))
    seed_alignment = []
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = sorted(
            [row for row in rows if row["seed"] == seed], key=lambda row: row["gain"]
        )
        critical = min(seed_rows, key=lambda row: abs(row["lyapunov"]))["gain"]
        memory_peak = max(seed_rows, key=lambda row: row["memory_capacity"])["gain"]
        accuracy_peak = max(seed_rows, key=lambda row: row["delayed_accuracy"])["gain"]
        seed_alignment.append({
            "seed": seed,
            "critical_gain": critical,
            "memory_peak_gain": memory_peak,
            "accuracy_peak_gain": accuracy_peak,
            "memory_distance_from_edge": abs(memory_peak - critical),
            "accuracy_distance_from_edge": abs(accuracy_peak - critical),
        })
    diagnostics = {
        "empirical_critical_gain": float(gains[critical_index]),
        "memory_peak_gain": float(gains[memory_index]),
        "delayed_accuracy_peak_gain": float(gains[accuracy_index]),
        "memory_peak_distance_from_edge": float(abs(gains[memory_index] - gains[critical_index])),
        "accuracy_peak_distance_from_edge": float(abs(gains[accuracy_index] - gains[critical_index])),
        "mean_tuned_gain": float(np.mean([item["final_gain"] for item in tuning])),
        "std_tuned_gain": float(np.std([item["final_gain"] for item in tuning])),
        "mean_seed_memory_distance_from_edge": float(
            np.mean([item["memory_distance_from_edge"] for item in seed_alignment])
        ),
        "mean_seed_accuracy_distance_from_edge": float(
            np.mean([item["accuracy_distance_from_edge"] for item in seed_alignment])
        ),
        "seed_memory_peak_at_edge_fraction": float(
            np.mean([item["memory_distance_from_edge"] < 1e-8 for item in seed_alignment])
        ),
        "seed_accuracy_peak_at_edge_fraction": float(
            np.mean([item["accuracy_distance_from_edge"] < 1e-8 for item in seed_alignment])
        ),
    }

    metrics_path = OUT / "network_criticality_metrics.json"
    figure_path = OUT / "network_criticality_summary.png"
    metrics_path.write_text(json.dumps({
        "summary": summary,
        "per_seed": rows,
        "gain_tuning": tuning,
        "per_seed_alignment": seed_alignment,
        "diagnostics": diagnostics,
        "claim_boundary": (
            "Near-critical dynamics can optimize selected computational metrics, but "
            "criticality is neither unique to conscious systems nor sufficient evidence "
            "for phenomenal consciousness."
        ),
    }, indent=2))
    plot_summary(summary, figure_path)

    print("\nNetwork criticality sweep")
    print("gain   lyapunov  branching  memory  delayed  dimension  runaway")
    for row in summary:
        print(
            f"{row['gain']:.2f}  {row['mean_lyapunov']:+.4f}    "
            f"{row['mean_branching_ratio']:.3f}      "
            f"{row['mean_memory_capacity']:.2f}    "
            f"{row['mean_delayed_accuracy']:.3f}    "
            f"{row['mean_effective_dimension']:.2f}      "
            f"{row['mean_runaway_fraction']:.3f}"
        )
    print("\nDiagnostics")
    for key, value in diagnostics.items():
        print(f"{key:34s} {value:.4f}")
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
