#!/usr/bin/env python3
"""AIR-inspired attention gating for fixed-capacity episodic terrain memory.

Each synthetic terrain episode contains many intermediate-level perceptual
packets but only a few action-relevant decision packets. Conditions receive the
same storage budget. A learned gate predicts later retrieval utility from local
geometry stability, prediction error, signed valence, novelty, target evidence,
and motion change. It never receives the terrain class or correct action.

The experiment tests whether attention improves what becomes available for
episodic encoding. It does not implement Prinz's biological AIR theory, prove
consciousness, or establish transfer to Unity terrain.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT


CONDITIONS = (
    "uniform_downsample",
    "random_capacity_match",
    "hand_salience",
    "learned_attention",
    "attention_scrambled",
    "learned_without_valence",
    "learned_without_prediction_error",
)
ATTENTION_FEATURES = (
    "geometry_stability",
    "prediction_error",
    "signed_valence",
    "novelty",
    "target_evidence",
    "motion_change",
)
PACKETS_PER_EPISODE = 32
MEMORY_BUDGET = 4
RAY_COUNT = 8
ACTION_COUNT = 8
# Calibrated on the quick pilot split, before the full held-out evaluation.
# Known queries clustered near 0.04 and unknown geometry near 0.29.
RETRIEVAL_REJECTION_THRESHOLD = 0.18


@dataclass
class Packet:
    rays: np.ndarray
    attention_features: np.ndarray
    action: int
    useful: int
    packet_kind: str


@dataclass
class Episode:
    packets: list[Packet]
    query_rays: np.ndarray
    correct_action: int
    unknown_query: np.ndarray


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    weights = np.exp(values)
    return weights / np.sum(weights)


def sigmoid(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def ray_distance(first, second):
    return float(np.mean(np.abs(np.asarray(first) - np.asarray(second))))


def make_prototypes(seed=381):
    rng = np.random.default_rng(seed)
    prototypes = []
    for action in range(ACTION_COUNT):
        rays = rng.uniform(0.12, 0.95, RAY_COUNT)
        # Give each action a grounded opening in its egocentric direction.
        rays[action] = rng.uniform(0.88, 1.0)
        rays[(action - 1) % RAY_COUNT] = rng.uniform(0.62, 0.90)
        rays[(action + 1) % RAY_COUNT] = rng.uniform(0.62, 0.90)
        rays[(action + 4) % RAY_COUNT] = rng.uniform(0.05, 0.28)
        prototypes.append(rays)
    return np.asarray(prototypes)


PROTOTYPES = make_prototypes()


def packet_features(kind, rng):
    if kind == "useful":
        values = [
            rng.normal(0.90, 0.05),
            rng.normal(0.86, 0.08),
            rng.normal(0.88, 0.08),
            rng.normal(0.82, 0.08),
            rng.normal(0.78, 0.10),
            rng.normal(0.72, 0.11),
        ]
    elif kind == "misleading":
        values = [
            rng.normal(0.88, 0.06),
            rng.normal(0.89, 0.07),
            rng.normal(-0.86, 0.09),
            rng.normal(0.84, 0.08),
            rng.normal(0.72, 0.12),
            rng.normal(0.75, 0.10),
        ]
    else:
        values = [
            rng.normal(0.28, 0.14),
            rng.normal(0.15, 0.10),
            rng.normal(0.00, 0.16),
            rng.normal(0.24, 0.13),
            rng.normal(0.12, 0.10),
            rng.normal(0.20, 0.12),
        ]
    values[2] = np.clip(values[2], -1.0, 1.0)
    other = np.clip([values[index] for index in (0, 1, 3, 4, 5)], 0.0, 1.0)
    return np.array([other[0], other[1], values[2], *other[2:]], dtype=np.float64)


def make_episode(rng, transformed=False):
    correct_action = int(rng.integers(0, ACTION_COUNT))
    prototype = PROTOTYPES[correct_action].copy()
    if transformed:
        prototype = np.clip(
            prototype * rng.uniform(0.88, 1.12) + rng.normal(0.0, 0.025, RAY_COUNT),
            0.0,
            1.0,
        )

    packets = []
    useful_slots = set(rng.choice(PACKETS_PER_EPISODE, 2, replace=False).tolist())
    remaining = [index for index in range(PACKETS_PER_EPISODE) if index not in useful_slots]
    misleading_slots = set(rng.choice(remaining, 2, replace=False).tolist())
    for index in range(PACKETS_PER_EPISODE):
        if index in useful_slots:
            kind = "useful"
            rays = np.clip(prototype + rng.normal(0.0, 0.035, RAY_COUNT), 0.0, 1.0)
            action = correct_action
            useful = 1
        elif index in misleading_slots:
            kind = "misleading"
            rays = np.clip(prototype + rng.normal(0.0, 0.030, RAY_COUNT), 0.0, 1.0)
            wrong = [action for action in range(ACTION_COUNT) if action != correct_action]
            action = int(rng.choice(wrong))
            useful = 0
        else:
            kind = "background"
            rays = rng.uniform(0.0, 1.0, RAY_COUNT)
            action = int(rng.integers(0, ACTION_COUNT))
            useful = 0
        packets.append(
            Packet(
                rays=rays,
                attention_features=packet_features(kind, rng),
                action=action,
                useful=useful,
                packet_kind=kind,
            )
        )

    query = np.clip(prototype + rng.normal(0.0, 0.045, RAY_COUNT), 0.0, 1.0)
    # Unknown geometry is deliberately far from the known decision prototype.
    for _ in range(50):
        unknown = rng.uniform(0.0, 1.0, RAY_COUNT)
        if ray_distance(unknown, prototype) > 0.28:
            break
    return Episode(
        packets=packets,
        query_rays=query,
        correct_action=correct_action,
        unknown_query=unknown,
    )


class LearnedAttentionGate:
    """Weighted logistic gate trained from delayed retrieval-utility labels."""

    def __init__(self, disabled_features=()):
        self.disabled = {
            ATTENTION_FEATURES.index(name) for name in disabled_features
        }
        self.weights = np.zeros(len(ATTENTION_FEATURES) + 1, dtype=np.float64)

    def vectors(self, packets):
        matrix = np.array([packet.attention_features for packet in packets])
        if self.disabled:
            matrix[:, list(self.disabled)] = 0.0
        return np.column_stack([matrix, np.ones(len(matrix))])

    def scores(self, packets):
        return self.vectors(packets) @ self.weights

    def fit(self, packets, epochs=260, learning_rate=0.12):
        x = self.vectors(packets)
        y = np.array([packet.useful for packet in packets], dtype=np.float64)
        positive_weight = (len(y) - np.sum(y)) / max(np.sum(y), 1.0)
        for _ in range(epochs):
            probability = sigmoid(x @ self.weights)
            weights = np.where(y > 0.5, positive_weight, 1.0)
            gradient = x.T @ (weights * (probability - y)) / np.sum(weights)
            gradient += 0.001 * np.r_[self.weights[:-1], 0.0]
            self.weights -= learning_rate * gradient


def train_gates(seed, episodes=900):
    rng = np.random.default_rng(seed)
    packets = []
    for _ in range(episodes):
        packets.extend(make_episode(rng, transformed=False).packets)
    gates = {
        "learned_attention": LearnedAttentionGate(),
        "learned_without_valence": LearnedAttentionGate(
            disabled_features=("signed_valence",)
        ),
        "learned_without_prediction_error": LearnedAttentionGate(
            disabled_features=("prediction_error",)
        ),
    }
    for gate in gates.values():
        gate.fit(packets)
    return gates


def select_packets(condition, episode, gates, rng):
    packets = episode.packets
    if condition == "uniform_downsample":
        indices = np.linspace(
            0, len(packets) - 1, MEMORY_BUDGET, dtype=np.int64
        )
    elif condition == "random_capacity_match":
        indices = rng.choice(len(packets), MEMORY_BUDGET, replace=False)
    elif condition == "hand_salience":
        scores = np.array(
            [
                0.35 * packet.attention_features[1]
                + 0.30 * abs(packet.attention_features[2])
                + 0.20 * packet.attention_features[3]
                + 0.15 * packet.attention_features[4]
                for packet in packets
            ]
        )
        indices = np.argsort(scores)[-MEMORY_BUDGET:]
    elif condition == "attention_scrambled":
        scores = gates["learned_attention"].scores(packets)
        scores = rng.permutation(scores)
        indices = np.argsort(scores)[-MEMORY_BUDGET:]
    else:
        scores = gates[condition].scores(packets)
        indices = np.argsort(scores)[-MEMORY_BUDGET:]
    return [packets[int(index)] for index in indices]


def retrieve(memory, query, rejection_threshold=RETRIEVAL_REJECTION_THRESHOLD):
    distances = np.array([ray_distance(packet.rays, query) for packet in memory])
    index = int(np.argmin(distances))
    if distances[index] > rejection_threshold:
        return None, float(distances[index])
    return memory[index], float(distances[index])


def evaluate_condition(condition, gates, episodes, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for episode_index, episode in enumerate(episodes):
        memory = select_packets(condition, episode, gates, rng)
        recalled, distance = retrieve(memory, episode.query_rays)
        unknown, unknown_distance = retrieve(memory, episode.unknown_query)
        useful_count = sum(packet.useful for packet in memory)
        rows.append(
            {
                "condition": condition,
                "episode": episode_index,
                "retrieval_success": float(
                    recalled is not None
                    and recalled.action == episode.correct_action
                    and recalled.useful
                ),
                "retrieval_accepted": float(recalled is not None),
                "false_recall": float(unknown is not None),
                "selected_useful_precision": useful_count / len(memory),
                "selected_useful_recall": useful_count / 2.0,
                "query_distance": distance,
                "unknown_distance": unknown_distance,
                "stored_packets": len(memory),
                "compression_ratio": len(memory) / len(episode.packets),
            }
        )
    return rows


def summarize(rows):
    keys = (
        "retrieval_success",
        "retrieval_accepted",
        "false_recall",
        "selected_useful_precision",
        "selected_useful_recall",
        "query_distance",
        "unknown_distance",
        "stored_packets",
        "compression_ratio",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def sign_test_p(deltas):
    values = np.asarray(deltas)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    positives = int(np.sum(values > 0.0))
    return float(
        sum(
            math.comb(len(values), index)
            for index in range(positives, len(values) + 1)
        )
        / (2 ** len(values))
    )


def plot_summary(summary, path):
    labels = [
        "uniform",
        "random",
        "hand\nsalience",
        "learned\nattention",
        "scrambled",
        "no\nvalence",
        "no pred.\nerror",
    ]
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(
        x,
        [summary[name]["retrieval_success"] for name in CONDITIONS],
        color=["#8d99ae", "#8d99ae", "#f4a261", "#168aad", "#9b5de5", "#d1495b", "#52b788"],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("retrieval success")
    axes[0].set_title("Equal-capacity episodic retrieval")

    width = 0.36
    axes[1].bar(
        x - width / 2,
        [summary[name]["selected_useful_precision"] for name in CONDITIONS],
        width,
        label="useful-packet precision",
        color="#168aad",
    )
    axes[1].bar(
        x + width / 2,
        [summary[name]["false_recall"] for name in CONDITIONS],
        width,
        label="unknown false recall",
        color="#d1495b",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Memory quality and rejection")
    axes[1].legend()
    fig.suptitle("AIR-Inspired Attention-Gated Episodic Encoding")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_benchmark(seed_count=20, train_episodes=900, evaluation_episodes=400):
    all_rows = {condition: [] for condition in CONDITIONS}
    per_seed = []
    gate_weights = []
    for seed_index in range(seed_count):
        seed = 51031 + 109 * seed_index
        gates = train_gates(seed, episodes=train_episodes)
        rng = np.random.default_rng(seed + 10000)
        episodes = [
            make_episode(rng, transformed=True)
            for _ in range(evaluation_episodes)
        ]
        seed_summary = {}
        for condition in CONDITIONS:
            rows = evaluate_condition(
                condition, gates, episodes, seed=seed + 20000
            )
            all_rows[condition].extend(rows)
            seed_summary[condition] = summarize(rows)
        per_seed.append(seed_summary)
        gate_weights.append(
            {
                "seed": seed,
                "learned_attention": gates["learned_attention"].weights.tolist(),
                "without_valence": gates["learned_without_valence"].weights.tolist(),
                "without_prediction_error": gates[
                    "learned_without_prediction_error"
                ].weights.tolist(),
            }
        )

    summary = {
        condition: summarize(rows)
        for condition, rows in all_rows.items()
    }
    random_delta = [
        row["learned_attention"]["retrieval_success"]
        - row["random_capacity_match"]["retrieval_success"]
        for row in per_seed
    ]
    hand_delta = [
        row["learned_attention"]["retrieval_success"]
        - row["hand_salience"]["retrieval_success"]
        for row in per_seed
    ]
    scramble_delta = [
        row["learned_attention"]["retrieval_success"]
        - row["attention_scrambled"]["retrieval_success"]
        for row in per_seed
    ]
    criteria = {
        "equal_memory_capacity": all(
            abs(summary[condition]["stored_packets"] - MEMORY_BUDGET) < 1e-12
            for condition in CONDITIONS
        ),
        "learned_beats_random_all_seeds": all(delta > 0.0 for delta in random_delta),
        "learned_beats_hand_salience_all_seeds": all(
            delta > 0.0 for delta in hand_delta
        ),
        "scrambling_reduces_retrieval": float(np.mean(scramble_delta)) > 0.25,
        "learned_retrieval_at_least_90_percent": (
            summary["learned_attention"]["retrieval_success"] >= 0.90
        ),
        "learned_false_recall_below_5_percent": (
            summary["learned_attention"]["false_recall"] <= 0.05
        ),
    }
    contrasts = {
        "learned_minus_random_retrieval": float(np.mean(random_delta)),
        "learned_minus_hand_salience_retrieval": float(np.mean(hand_delta)),
        "learned_minus_scrambled_retrieval": float(np.mean(scramble_delta)),
        "learned_vs_random_one_sided_sign_test_p": sign_test_p(random_delta),
        "learned_vs_hand_one_sided_sign_test_p": sign_test_p(hand_delta),
        "learned_vs_scrambled_one_sided_sign_test_p": sign_test_p(scramble_delta),
        "seed_count": seed_count,
    }
    return {
        "experiment": "AIR-inspired fixed-capacity episodic encoding",
        "conditions": list(CONDITIONS),
        "attention_features": list(ATTENTION_FEATURES),
        "protocol": {
            "seed_count": seed_count,
            "training_episodes_per_seed": train_episodes,
            "evaluation_episodes_per_seed": evaluation_episodes,
            "packets_per_episode": PACKETS_PER_EPISODE,
            "memory_budget": MEMORY_BUDGET,
            "compression_ratio": MEMORY_BUDGET / PACKETS_PER_EPISODE,
            "terrain_class_visible_to_attention_gate": False,
            "correct_action_visible_to_attention_gate": False,
            "training_and_evaluation_streams_separate": True,
            "retrieval_rejection_threshold": RETRIEVAL_REJECTION_THRESHOLD,
            "rejection_threshold_calibrated_before_full_evaluation": True,
        },
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "criteria": criteria,
        "gate_weights": gate_weights,
        "claim_boundary": (
            "Under equal memory capacity, a learned attention score selected "
            "intermediate packets that supported later associative action "
            "retrieval. Scrambling and feature lesions test causal dependence. "
            "This synthetic software analogue does not implement biological AIR, "
            "demonstrate Unity terrain transfer, or establish consciousness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    payload = run_benchmark(
        seed_count=5 if args.quick else 20,
        train_episodes=250 if args.quick else 900,
        evaluation_episodes=120 if args.quick else 400,
    )
    OUT.mkdir(exist_ok=True)
    metrics_path = OUT / "attended_episodic_encoding_metrics.json"
    figure_path = OUT / "attended_episodic_encoding_summary.png"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_summary(payload["summary"], figure_path)
    print("AIR-inspired episodic encoding benchmark complete")
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "paired_seed_contrasts": payload["paired_seed_contrasts"],
                "criteria": payload["criteria"],
            },
            indent=2,
        )
    )
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
