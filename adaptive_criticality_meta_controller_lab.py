#!/usr/bin/env python3
"""Test reward-driven recurrent-gain regulation under changing conditions.

The frozen reservoir is exposed to alternating quiet and noisy delayed-memory
regimes. A non-stationary bandit adjusts recurrent gain using only recent task
reward. Fixed-gain, random, and phase-aware oracle controls separate online
adaptation from selecting a favorable gain in advance.

This is a functional homeostasis experiment. Dynamical criticality occurs in
many non-conscious systems and is not evidence of phenomenal consciousness.
"""

import argparse
import json
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from network_criticality_lab import (
    driven_lyapunov,
    dynamic_range_db,
    make_reservoir,
    memory_capacity,
    ridge_fit,
    ridge_predict,
    step_state,
)
from tiny_lab import OUT, set_seed


GAINS = np.asarray([0.65, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70])
PHASES = (
    {"name": "quiet_memory", "noise": 0.10},
    {"name": "noisy_memory", "noise": 0.42},
    {"name": "quiet_memory", "noise": 0.10},
    {"name": "noisy_memory", "noise": 0.42},
)


def delayed_cue_accuracy_at_noise(weights, inputs, gain, seed, noise, delay=55):
    """Train and test a linear readout after a delayed cue under given noise."""
    rng = np.random.default_rng(seed)

    def make_batch(samples):
        labels = rng.choice([-1.0, 1.0], samples).astype(np.float32)
        state = np.zeros((samples, weights.shape[0]), dtype=np.float32)
        for tick in range(delay):
            packet = np.zeros((samples, 2), dtype=np.float32)
            if tick == 0:
                packet[:, 0] = labels
            packet[:, 1] = rng.normal(0.0, noise, samples)
            state = step_state(state, packet, weights, inputs, gain)
        return state, labels

    train_states, train_labels = make_batch(360)
    test_states, test_labels = make_batch(240)
    readout = ridge_fit(train_states, train_labels)
    prediction = ridge_predict(test_states, readout)
    return float(np.mean(np.sign(prediction) == test_labels))


def minmax(values):
    values = np.asarray(values, dtype=float)
    spread = float(np.max(values) - np.min(values))
    if spread < 1e-9:
        return np.ones_like(values)
    return (values - np.min(values)) / spread


def build_gain_table(weights, inputs, seed):
    """Measure each gain before the online controller sees any phase rewards."""
    rows = []
    for index, gain in enumerate(GAINS):
        capacity, dimension, saturation = memory_capacity(
            weights, inputs, float(gain), seed + 100 + index
        )
        rows.append({
            "gain": float(gain),
            "lyapunov": driven_lyapunov(
                weights, inputs, float(gain), seed + 200 + index
            ),
            "memory_capacity": capacity,
            "effective_dimension": dimension,
            "saturation_fraction": saturation,
            "dynamic_range_db": dynamic_range_db(weights, inputs, float(gain)),
            "quiet_accuracy": delayed_cue_accuracy_at_noise(
                weights, inputs, float(gain), seed + 300 + index, 0.10
            ),
            "noisy_accuracy": delayed_cue_accuracy_at_noise(
                weights, inputs, float(gain), seed + 400 + index, 0.42
            ),
        })

    memory = minmax([row["memory_capacity"] for row in rows])
    dynamic = minmax([row["dynamic_range_db"] for row in rows])
    dimension = minmax([row["effective_dimension"] for row in rows])
    saturation = minmax([row["saturation_fraction"] for row in rows])
    quiet = minmax([row["quiet_accuracy"] for row in rows])
    noisy = minmax([row["noisy_accuracy"] for row in rows])
    for index, row in enumerate(rows):
        # Quiet conditions reward detailed fading memory. Under strong noise,
        # persistence and representational spread matter more, but saturation
        # remains costly in both regimes.
        row["quiet_memory_utility"] = float(np.clip(
            0.50 * memory[index] + 0.35 * quiet[index]
            + 0.15 * dynamic[index] - 0.18 * saturation[index], 0.0, 1.0
        ))
        row["noisy_memory_utility"] = float(np.clip(
            0.70 * noisy[index] + 0.20 * dimension[index]
            + 0.10 * memory[index] - 0.12 * saturation[index], 0.0, 1.0
        ))
    return rows


class AdaptiveGainController:
    """Contextual sliding-window UCB using an observed low/high noise estimate."""

    def __init__(self, arms, rng, window=28):
        self.arms = arms
        self.rng = rng
        self.window = window
        self.history = {
            context: [deque(maxlen=window) for _ in range(arms)]
            for context in ("low_noise", "high_noise")
        }
        self.relearn_queue = {
            context: deque(range(arms)) for context in self.history
        }
        self.last_expected = None
        self.change_detections = 0

    def choose(self, tick, context="low_noise"):
        queue = self.relearn_queue[context]
        history = self.history[context]
        if queue:
            return queue.popleft()
        if self.rng.random() < 0.08:
            return int(self.rng.integers(self.arms))
        means = np.asarray([
            np.mean(values) if values else 0.5 for values in history
        ])
        counts = np.asarray([max(len(values), 1) for values in history])
        bonus = 0.20 * np.sqrt(np.log(tick + 2.0) / counts)
        return int(np.argmax(means + bonus))

    def observe(self, arm, reward, context="low_noise"):
        history = self.history[context]
        expected = (
            float(np.mean(history[arm])) if history[arm] else reward
        )
        history[arm].append(reward)
        if len(history[arm]) >= 4 and reward < expected - 0.24:
            self.change_detections += 1
            # Old values become weak priors rather than being erased entirely.
            for values in history:
                while len(values) > 3:
                    values.popleft()
            self.relearn_queue[context] = deque(
                self.rng.permutation(self.arms).tolist()
            )
        self.last_expected = expected


def policy_arm(
    condition, tick, table, controller, rng, fixed_arms, phase_name, noise_context
):
    if condition == "adaptive_meta_controller":
        return controller.choose(tick, noise_context)
    if condition == "random_gain":
        return int(rng.integers(len(table)))
    if condition == "oracle":
        return int(np.argmax([row[f"{phase_name}_utility"] for row in table]))
    return fixed_arms[condition]


def run_condition(condition, table, seed, phase_length):
    rng = np.random.default_rng(seed)
    controller = AdaptiveGainController(len(table), rng)
    fixed_arms = {
        "fixed_memory_peak": int(np.argmax([row["memory_capacity"] for row in table])),
        "fixed_empirical_edge": int(np.argmin([abs(row["lyapunov"]) for row in table])),
        "fixed_noisy_peak": int(np.argmax([row["noisy_accuracy"] for row in table])),
        "fixed_global_best": int(np.argmax([
            0.5 * (row["quiet_memory_utility"] + row["noisy_memory_utility"])
            for row in table
        ])),
    }
    trace = []
    tick = 0
    for phase_index, phase in enumerate(PHASES):
        utility_key = f"{phase['name']}_utility"
        oracle_reward = max(row[utility_key] for row in table)
        oracle_arm = int(np.argmax([row[utility_key] for row in table]))
        for phase_tick in range(phase_length):
            noise_estimate = float(np.clip(phase["noise"] + rng.normal(0.0, 0.025), 0, 1))
            noise_context = "high_noise" if noise_estimate >= 0.25 else "low_noise"
            arm = policy_arm(
                condition, tick, table, controller, rng, fixed_arms, phase["name"],
                noise_context,
            )
            expected = table[arm][utility_key]
            reward = float(np.clip(expected + rng.normal(0.0, 0.025), 0.0, 1.0))
            if condition == "adaptive_meta_controller":
                controller.observe(arm, reward, noise_context)
            trace.append({
                "tick": tick,
                "phase_index": phase_index,
                "phase_tick": phase_tick,
                "phase": phase["name"],
                "noise": phase["noise"],
                "observed_noise": noise_estimate,
                "noise_context": noise_context,
                "gain": table[arm]["gain"],
                "arm": arm,
                "reward": reward,
                "expected_reward": expected,
                "oracle_reward": oracle_reward,
                "oracle_gain": table[oracle_arm]["gain"],
                "regret": oracle_reward - expected,
            })
            tick += 1
    return trace, controller.change_detections


def adaptation_latency(trace, phase_length, tolerance=0.08, hold=5):
    latencies = []
    for start in range(phase_length, len(trace), phase_length):
        latency = phase_length
        for offset in range(0, phase_length - hold + 1):
            window = trace[start + offset : start + offset + hold]
            if all(row["regret"] <= tolerance for row in window):
                latency = offset
                break
        latencies.append(latency)
    return latencies


def summarize(runs, phase_length):
    summary = []
    for condition in sorted({run["condition"] for run in runs}):
        selected = [run for run in runs if run["condition"] == condition]
        rewards = [np.mean([row["reward"] for row in run["trace"]]) for run in selected]
        regrets = [np.mean([row["regret"] for row in run["trace"]]) for run in selected]
        latencies = [
            value for run in selected
            for value in adaptation_latency(run["trace"], phase_length)
        ]
        switches = []
        for run in selected:
            gains = [row["gain"] for row in run["trace"]]
            switches.append(np.mean(np.diff(gains) != 0.0))
        summary.append({
            "condition": condition,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_regret": float(np.mean(regrets)),
            "mean_adaptation_latency": float(np.mean(latencies)),
            "gain_switch_fraction": float(np.mean(switches)),
            "mean_change_detections": float(np.mean([
                run["change_detections"] for run in selected
            ])),
        })
    return summary


def exact_two_sided_sign_p(wins, losses):
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(max(wins, losses), total + 1))
    return min(1.0, 2.0 * tail / (2**total))


def diagnostics(runs):
    adaptive = [run for run in runs if run["condition"] == "adaptive_meta_controller"]
    phase_summary = []
    for phase_index, phase in enumerate(PHASES):
        rows = [
            row for run in adaptive for row in run["trace"]
            if row["phase_index"] == phase_index
        ]
        phase_summary.append({
            "phase_index": phase_index,
            "phase": phase["name"],
            "mean_gain": float(np.mean([row["gain"] for row in rows])),
            "mean_oracle_gain": float(np.mean([row["oracle_gain"] for row in rows])),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
            "mean_regret": float(np.mean([row["regret"] for row in rows])),
        })

    comparisons = []
    seeds = sorted({run["seed"] for run in runs})
    for control in (
        "fixed_memory_peak", "fixed_empirical_edge", "fixed_noisy_peak",
        "fixed_global_best", "random_gain"
    ):
        differences = []
        for seed in seeds:
            selected = {
                run["condition"]: np.mean([row["reward"] for row in run["trace"]])
                for run in runs if run["seed"] == seed
            }
            differences.append(
                selected["adaptive_meta_controller"] - selected[control]
            )
        wins = int(sum(value > 0 for value in differences))
        losses = int(sum(value < 0 for value in differences))
        comparisons.append({
            "control": control,
            "adaptive_wins": wins,
            "adaptive_losses": losses,
            "ties": int(len(differences) - wins - losses),
            "mean_reward_difference": float(np.mean(differences)),
            "exact_two_sided_sign_p": exact_two_sided_sign_p(wins, losses),
        })
    return {"adaptive_phase_summary": phase_summary, "paired_comparisons": comparisons}


def plot_results(runs, summary, phase_length, path):
    conditions = [row["condition"] for row in summary]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(
        conditions, [row["mean_reward"] for row in summary], color="#2a9d8f"
    )
    axes[0].set_ylabel("Mean normalized task reward")
    axes[0].tick_params(axis="x", rotation=18)
    adaptive = [run for run in runs if run["condition"] == "adaptive_meta_controller"]
    length = len(adaptive[0]["trace"])
    gain_matrix = np.asarray([[row["gain"] for row in run["trace"]] for run in adaptive])
    oracle_matrix = np.asarray([
        [row["oracle_gain"] for row in run["trace"]] for run in adaptive
    ])
    axes[1].plot(np.mean(gain_matrix, axis=0), label="adaptive mean gain", color="#e76f51")
    axes[1].plot(np.mean(oracle_matrix, axis=0), label="oracle gain", color="#264653", linestyle="--")
    for boundary in range(phase_length, length, phase_length):
        axes[1].axvline(boundary, color="black", alpha=0.25)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Recurrent gain")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Online gain regulation under alternating sensory noise")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--phase-length", type=int, default=None)
    args = parser.parse_args()

    set_seed(211)
    seed_count = args.seeds or (3 if args.quick else 10)
    phase_length = args.phase_length or (28 if args.quick else 56)
    conditions = (
        "fixed_memory_peak",
        "fixed_empirical_edge",
        "fixed_noisy_peak",
        "fixed_global_best",
        "random_gain",
        "adaptive_meta_controller",
        "oracle",
    )
    runs = []
    gain_tables = []
    for seed_index in range(seed_count):
        seed = 9100 + seed_index
        weights, inputs = make_reservoir(seed)
        table = build_gain_table(weights, inputs, seed * 100)
        gain_tables.append({"seed": seed, "rows": table})
        for condition_index, condition in enumerate(conditions):
            trace, detections = run_condition(
                condition,
                table,
                seed * 1000 + condition_index * 97,
                phase_length,
            )
            runs.append({
                "seed": seed,
                "condition": condition,
                "change_detections": detections,
                "trace": trace,
            })

    summary = summarize(runs, phase_length)
    result_diagnostics = diagnostics(runs)
    metrics_path = OUT / "adaptive_criticality_meta_controller_metrics.json"
    figure_path = OUT / "adaptive_criticality_meta_controller_summary.png"
    metrics_path.write_text(json.dumps({
        "phase_schedule": list(PHASES),
        "phase_length": phase_length,
        "gain_tables": gain_tables,
        "summary": summary,
        "diagnostics": result_diagnostics,
        "runs": runs,
        "claim_boundary": (
            "This tests online regulation of a frozen recurrent reservoir under "
            "engineered task shifts. It does not establish that criticality is "
            "necessary for consciousness or that the controller has awareness."
        ),
    }, indent=2))
    plot_results(runs, summary, phase_length, figure_path)

    print("\nAdaptive criticality meta-controller")
    print("condition                  reward   regret  latency  switches")
    for row in summary:
        print(
            f"{row['condition']:26s} {row['mean_reward']:.3f}    "
            f"{row['mean_regret']:.3f}   {row['mean_adaptation_latency']:.1f}     "
            f"{row['gain_switch_fraction']:.3f}"
        )
    print("\nAdaptive paired comparisons")
    for row in result_diagnostics["paired_comparisons"]:
        print(
            f"vs {row['control']:22s} "
            f"{row['adaptive_wins']}-{row['adaptive_losses']}  "
            f"delta={row['mean_reward_difference']:+.3f}  "
            f"p={row['exact_two_sided_sign_p']:.4f}"
        )
    print(f"\nSaved {metrics_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
