#!/usr/bin/env python3
"""Build and evaluate passive AIR-inspired memory from Unity terrain telemetry.

The encoder only sees information available at the current frame. Delayed
outcomes label useful moments for offline attention training. The exported
library is observational and cannot alter Unity motor commands.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "geometry_stability",
    "prediction_error",
    "signed_valence",
    "novelty",
    "target_evidence",
    "motion_change",
)
ACTIONS = (
    "up",
    "up_right",
    "right",
    "down_right",
    "down",
    "down_left",
    "left",
    "up_left",
)
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


@dataclass
class TerrainPacket:
    time: float
    step: int
    rays: np.ndarray
    body_clearance: np.ndarray
    hunger: float
    action: str
    features: np.ndarray
    utility: float
    useful: int
    source_index: int

    def retrieval_vector(self):
        return np.r_[self.rays, self.body_clearance, self.hunger]

    def as_json(self):
        return {
            "time": self.time,
            "step": self.step,
            "rays": self.rays.tolist(),
            "body_clearance": self.body_clearance.tolist(),
            "hunger": self.hunger,
            "action": self.action,
            "attention_features": self.features.tolist(),
            "utility": self.utility,
            "useful": self.useful,
            "source_index": self.source_index,
        }


class LogisticAttentionGate:
    def __init__(self):
        self.weights = np.zeros(len(FEATURE_NAMES) + 1, dtype=np.float64)
        self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)

    def _matrix(self, packets, fit=False):
        values = np.array([packet.features for packet in packets], dtype=np.float64)
        if fit:
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale[self.scale < 1e-6] = 1.0
        values = (values - self.mean) / self.scale
        return np.column_stack([values, np.ones(len(values))])

    def fit(self, packets, epochs=500, learning_rate=0.08):
        x = self._matrix(packets, fit=True)
        y = np.array([packet.useful for packet in packets], dtype=np.float64)
        positives = max(float(y.sum()), 1.0)
        positive_weight = (len(y) - positives) / positives
        for _ in range(epochs):
            logits = np.clip(x @ self.weights, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            sample_weight = np.where(y > 0.5, positive_weight, 1.0)
            gradient = x.T @ (sample_weight * (probability - y))
            gradient /= max(float(sample_weight.sum()), 1.0)
            gradient += 0.001 * np.r_[self.weights[:-1], 0.0]
            self.weights -= learning_rate * gradient

    def scores(self, packets):
        return self._matrix(packets) @ self.weights

    def as_json(self):
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if row.get("rays") and row.get("body_clearance"):
                rows.append(row)
    if len(rows) < 20:
        raise ValueError("Recording contains too few complete telemetry rows")
    return rows


def vector(row, key, size=8, default=1.0):
    values = np.asarray(row.get(key, []), dtype=np.float64)
    if values.size != size:
        return np.full(size, default, dtype=np.float64)
    return np.clip(values, 0.0, 1.0)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def future_slice(rows, index, horizon_seconds):
    end_time = float(rows[index].get("time", 0.0)) + horizon_seconds
    end = index + 1
    while end < len(rows) and float(rows[end].get("time", 0.0)) <= end_time:
        end += 1
    return rows[index + 1 : end]


def delayed_utility(row, future):
    if not future:
        return 0.0
    pickup_now = int(row.get("mushroom_pickups_total", 0) or 0)
    pickup_rows = [
        item
        for item in future
        if int(item.get("mushroom_pickups_total", pickup_now) or pickup_now)
        > pickup_now
    ]
    pickup_gain = int(bool(pickup_rows))
    pickup_delay = (
        float(pickup_rows[0].get("time", 0.0))
        - float(row.get("time", 0.0))
        if pickup_rows
        else math.inf
    )
    # Nearby decisions receive more credit than every frame in a broad
    # pre-pickup window.
    pickup_credit = pickup_gain * math.exp(-pickup_delay / 1.5)
    failure_now = int(row.get("survival_failures", 0) or 0)
    failure_gain = max(
        int(item.get("survival_failures", failure_now) or failure_now)
        for item in future
    ) - failure_now
    collision_rate = np.mean(
        [bool(item.get("body_collision", False)) for item in future]
    )
    current_wedge = float(row.get("physics_wedge_seconds", 0.0) or 0.0)
    future_wedge = min(
        float(item.get("physics_wedge_seconds", 0.0) or 0.0)
        for item in future
    )
    wedge_resolution = current_wedge > 0.5 and future_wedge < 0.1

    visible = bool(row.get("food_visible", False))
    distance = float(row.get("food_distance", 0.0) or 0.0)
    future_distances = [
        float(item.get("food_distance", 0.0) or 0.0)
        for item in future
        if bool(item.get("food_visible", False))
        and float(item.get("food_distance", 0.0) or 0.0) > 0.0
    ]
    progress = 0.0
    if visible and distance > 0.0 and future_distances:
        progress = clamp((distance - min(future_distances)) / max(distance, 1.0))
    return float(
        1.4 * pickup_credit
        + 0.65 * progress
        + 0.55 * float(wedge_resolution)
        - 1.5 * min(failure_gain, 1)
        - 0.35 * collision_rate
    )


def build_packets(rows, sample_seconds=0.5, horizon_seconds=6.0):
    packets = []
    last_sample_time = -math.inf
    recent_rays = []
    previous_rays = None
    previous_action = None
    previous_reward = 0.0
    for index, row in enumerate(rows):
        now = float(row.get("time", index))
        if now - last_sample_time < sample_seconds:
            continue
        rays = vector(row, "rays")
        clearance = vector(row, "body_clearance")
        if previous_rays is None:
            prediction_error = 0.0
            geometry_stability = 1.0
        else:
            prediction_error = float(np.mean(np.abs(rays - previous_rays)))
            geometry_stability = 1.0 - prediction_error
        novelty = (
            min(float(np.mean(np.abs(rays - prior))) for prior in recent_rays)
            if recent_rays
            else 1.0
        )
        reward = float(row.get("mushroom_reward_total", 0.0) or 0.0)
        reward_delta = max(0.0, reward - previous_reward)
        body_cost = float(bool(row.get("body_collision", False)))
        wedge_cost = clamp(float(row.get("physics_wedge_seconds", 0.0) or 0.0) / 8.0)
        signed_valence = clamp(
            reward_delta - 0.45 * body_cost - 0.35 * wedge_cost,
            -1.0,
            1.0,
        )
        visible = float(bool(row.get("food_visible", False)))
        food_distance = float(row.get("food_distance", 0.0) or 0.0)
        target_evidence = visible * (1.0 - clamp(food_distance / 28.0))
        active_action = str(row.get("active_action", "idle"))
        action = (
            active_action
            if active_action in ACTION_INDEX
            else str(row.get("shadow_action", "idle"))
        )
        motion_change = float(previous_action is not None and action != previous_action)
        future = future_slice(rows, index, horizon_seconds)
        utility = delayed_utility(row, future)
        packets.append(
            TerrainPacket(
                time=now,
                step=int(row.get("step", index) or index),
                rays=rays,
                body_clearance=clearance,
                hunger=clamp(row.get("hunger", 0.0)),
                action=action,
                features=np.array(
                    [
                        geometry_stability,
                        prediction_error,
                        signed_valence,
                        novelty,
                        target_evidence,
                        motion_change,
                    ],
                    dtype=np.float64,
                ),
                utility=utility,
                useful=int(utility >= 0.35),
                source_index=index,
            )
        )
        last_sample_time = now
        previous_rays = rays
        previous_action = action
        previous_reward = reward
        recent_rays.append(rays)
        recent_rays = recent_rays[-24:]
    return packets


def angular_action_error(first, second):
    if first not in ACTION_INDEX or second not in ACTION_INDEX:
        return 8
    delta = abs(ACTION_INDEX[first] - ACTION_INDEX[second])
    return min(delta, len(ACTIONS) - delta)


def retrieve(memory, query, threshold):
    if not memory:
        return None, math.inf
    query_vector = query.retrieval_vector()
    distances = [
        float(np.mean(np.abs(packet.retrieval_vector() - query_vector)))
        for packet in memory
    ]
    index = int(np.argmin(distances))
    return (memory[index] if distances[index] <= threshold else None), distances[index]


def select_memory(packets, gate, capacity):
    scores = gate.scores(packets)
    indices = np.argsort(scores)[-min(capacity, len(packets)) :]
    return [packets[int(index)] for index in indices]


def evaluate(memory, queries, threshold):
    useful_queries = [packet for packet in queries if packet.useful]
    negative_queries = [packet for packet in queries if not packet.useful]
    correct = accepted = 0
    distances = []
    for query in useful_queries:
        recalled, distance = retrieve(memory, query, threshold)
        distances.append(distance)
        accepted += recalled is not None
        correct += (
            recalled is not None
            and angular_action_error(recalled.action, query.action) <= 1
        )
    nonutility_recall = 0
    for query in negative_queries:
        recalled, _ = retrieve(memory, query, threshold)
        nonutility_recall += recalled is not None
    return {
        "useful_queries": len(useful_queries),
        "negative_queries": len(negative_queries),
        "retrieval_acceptance": accepted / max(len(useful_queries), 1),
        "directional_retrieval_agreement": correct / max(len(useful_queries), 1),
        # Low delayed utility does not imply novel geometry. This is an
        # acceptance diagnostic, not an unknown-scene false-recall measure.
        "nonutility_query_recall": nonutility_recall / max(len(negative_queries), 1),
        "mean_useful_query_distance": float(np.mean(distances)) if distances else None,
    }


def run(recording, output, capacity=256, train_fraction=0.6, threshold=0.12):
    rows = load_rows(recording)
    packets = build_packets(rows)
    split = max(10, min(len(packets) - 10, int(len(packets) * train_fraction)))
    train = packets[:split]
    evaluation = packets[split:]
    if not any(packet.useful for packet in train):
        raise ValueError(
            "Training segment has no delayed-success events; record pickups or "
            "wedge recoveries before building terrain memory"
        )
    gate = LogisticAttentionGate()
    gate.fit(train)
    memory = select_memory(train, gate, capacity)
    random_rng = np.random.default_rng(7103)
    random_indices = random_rng.choice(
        len(train), min(capacity, len(train)), replace=False
    )
    random_memory = [train[int(index)] for index in random_indices]
    payload = {
        "format": "terrain_air_memory_v1",
        "mode": "passive_observer_only",
        "source_recording": str(Path(recording).resolve()),
        "protocol": {
            "telemetry_rows": len(rows),
            "packets": len(packets),
            "training_packets": len(train),
            "evaluation_packets": len(evaluation),
            "sample_seconds": 0.5,
            "delayed_utility_horizon_seconds": 6.0,
            "memory_capacity": len(memory),
            "retrieval_threshold": threshold,
            "future_outcome_present_in_packet": False,
            "motor_control_enabled": False,
        },
        "attention_gate": gate.as_json(),
        "selected_useful_fraction": float(
            np.mean([packet.useful for packet in memory])
        ),
        "random_selected_useful_fraction": float(
            np.mean([packet.useful for packet in random_memory])
        ),
        "learned_attention_retrieval": evaluate(memory, evaluation, threshold),
        "random_capacity_retrieval": evaluate(
            random_memory, evaluation, threshold
        ),
        "memory": [packet.as_json() for packet in memory],
        "claim_boundary": (
            "This library is selected from Unity telemetry using delayed "
            "behavioral utility. Retrieval is passive and action agreement is "
            "an observational proxy, not evidence that recall improves control."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/terrain_air/passive_memory.json"),
    )
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--retrieval-threshold", type=float, default=0.12)
    args = parser.parse_args()
    payload = run(
        args.recording,
        args.output,
        capacity=args.capacity,
        train_fraction=args.train_fraction,
        threshold=args.retrieval_threshold,
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "selected_useful_fraction": payload["selected_useful_fraction"],
                "learned_attention_retrieval": payload[
                    "learned_attention_retrieval"
                ],
                "random_capacity_retrieval": payload[
                    "random_capacity_retrieval"
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
