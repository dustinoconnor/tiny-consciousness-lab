#!/usr/bin/env python3
"""Bach-inspired learned conductor and reflective protocol-memory experiment.

Three frozen specialists propose actions:

- reflex: cheap and strongest in urgent/reactive contexts
- episodic: ART-like precedent playback, strongest in familiar contexts
- predictive: expensive MPC-like imagination, strongest in novel contexts

Only the conductor learns. It allocates control from reward and writes an
integrated protocol containing attended context, selected specialist,
prediction, action, outcome, and valence. The protocol preserves an early
context cue after that cue disappears from immediate observation.

Conditions compare a static confidence router, a learned conductor without
protocol memory, an intact protocol conductor, lesions/scrambling of the same
trained conductor, a forced false-context intervention, and an oracle bound.

This tests functional executive coordination and reflective access. It does
not establish phenomenal consciousness or implement the full cortical theory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT


SPECIALISTS = ("reflex", "episodic", "predictive")
CONTEXTS = ("reactive", "familiar", "novel")
UNKNOWN_CONTEXT = len(CONTEXTS)
ACTIONS = 4

COMPETENCE = np.array(
    [
        [0.92, 0.58, 0.72],
        [0.62, 0.93, 0.76],
        [0.59, 0.42, 0.91],
    ],
    dtype=np.float64,
)
COMPUTE_COST = np.array([0.015, 0.065, 0.18], dtype=np.float64)
CONFIDENCE_BASE = np.array([0.72, 0.80, 0.70], dtype=np.float64)


@dataclass
class ProtocolEntry:
    step: int
    attended_context: str
    selected_specialist: str
    specialist_confidence: float
    predicted_action: int
    executed_action: int
    correct_action: int
    outcome: str
    reward: float
    valence: float


def observed_context(true_context: int, accuracy: float, rng: np.random.Generator) -> int:
    if rng.random() < accuracy:
        return true_context
    alternatives = [index for index in range(len(CONTEXTS)) if index != true_context]
    return int(rng.choice(alternatives))


def frozen_specialist_proposals(
    context: int,
    correct_action: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    proposals = np.empty(len(SPECIALISTS), dtype=np.int64)
    confidences = np.empty(len(SPECIALISTS), dtype=np.float64)
    for specialist in range(len(SPECIALISTS)):
        if rng.random() < COMPETENCE[context, specialist]:
            proposals[specialist] = correct_action
        else:
            wrong = [action for action in range(ACTIONS) if action != correct_action]
            proposals[specialist] = int(rng.choice(wrong))
        # Confidence is intentionally imperfect. A conductor is useful only if
        # it learns outcomes instead of treating specialist certainty as truth.
        confidences[specialist] = float(
            np.clip(CONFIDENCE_BASE[specialist] + rng.normal(0.0, 0.09), 0.05, 0.99)
        )
    return proposals, confidences


class LearnedConductor:
    """Small reward-trained executive over frozen specialist proposals."""

    def __init__(self, seed: int, learning_rate: float = 0.16):
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.q = np.zeros((len(CONTEXTS) + 1, len(SPECIALISTS)), dtype=np.float64)
        self.visits = np.zeros_like(self.q)

    def choose(self, context: int, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, len(SPECIALISTS)))
        jitter = self.rng.normal(0.0, 1e-7, len(SPECIALISTS))
        return int(np.argmax(self.q[context] + jitter))

    def update(self, context: int, specialist: int, reward: float) -> None:
        self.visits[context, specialist] += 1.0
        rate = self.learning_rate / np.sqrt(1.0 + 0.025 * self.visits[context, specialist])
        self.q[context, specialist] += rate * (reward - self.q[context, specialist])


def train_conductor(
    conductor: LearnedConductor,
    protocol_enabled: bool,
    episodes: int,
    steps: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        true_context = int(rng.integers(0, len(CONTEXTS)))
        cue = (
            UNKNOWN_CONTEXT
            if rng.random() < 0.04
            else observed_context(true_context, 0.91, rng)
        )
        for step in range(steps):
            immediate = observed_context(true_context, 0.48, rng)
            routing_context = cue if protocol_enabled else immediate
            correct_action = int(rng.integers(0, ACTIONS))
            proposals, _confidences = frozen_specialist_proposals(
                true_context, correct_action, rng
            )
            specialist = conductor.choose(
                routing_context,
                epsilon=max(0.035, 0.24 * (1.0 - episode / max(episodes, 1))),
            )
            correct = int(proposals[specialist]) == correct_action
            reward = (1.0 if correct else -1.0) - COMPUTE_COST[specialist]
            conductor.update(routing_context, specialist, reward)


def protocol_context(
    condition: str,
    cue: int,
    immediate: int,
    true_context: int,
    step: int = 0,
) -> int:
    if condition == "learned_no_protocol":
        return immediate
    if condition == "protocol_intact":
        return cue
    if condition == "protocol_lesion":
        return UNKNOWN_CONTEXT
    if condition == "protocol_scrambled":
        return (cue + step + 1) % len(CONTEXTS)
    if condition == "false_protocol_intervention":
        return (cue + 1) % len(CONTEXTS)
    if condition == "oracle_conductor":
        return true_context
    raise ValueError(condition)


def delayed_report(
    condition: str,
    protocol: list[ProtocolEntry],
    immediate_context: int,
) -> tuple[int, str, float]:
    if condition in {"protocol_intact", "oracle_conductor"} and protocol:
        report_context = CONTEXTS.index(protocol[0].attended_context)
        selected = Counter(entry.selected_specialist for entry in protocol).most_common(1)[0][0]
        mean_valence = float(np.mean([entry.valence for entry in protocol]))
        return report_context, selected, mean_valence
    if condition in {"protocol_scrambled", "false_protocol_intervention"} and protocol:
        report_context = CONTEXTS.index(protocol[0].attended_context)
        selected = Counter(entry.selected_specialist for entry in protocol).most_common(1)[0][0]
        mean_valence = float(np.mean([entry.valence for entry in protocol]))
        return report_context, selected, mean_valence
    if condition == "learned_no_protocol":
        return immediate_context, "unavailable", 0.0
    return UNKNOWN_CONTEXT, "unavailable", 0.0


def run_evaluation(
    condition: str,
    protocol_conductor: LearnedConductor,
    no_protocol_conductor: LearnedConductor,
    episodes: int,
    steps: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    episode_rows = []
    sample_protocol = []
    for episode in range(episodes):
        true_context = int(rng.integers(0, len(CONTEXTS)))
        cue = observed_context(true_context, 0.91, rng)
        protocol: list[ProtocolEntry] = []
        correct_count = 0
        total_reward = 0.0
        total_compute = 0.0
        selections = Counter()
        final_immediate = UNKNOWN_CONTEXT

        for step in range(steps):
            final_immediate = observed_context(true_context, 0.48, rng)
            correct_action = int(rng.integers(0, ACTIONS))
            proposals, confidences = frozen_specialist_proposals(
                true_context, correct_action, rng
            )

            if condition == "static_confidence_router":
                specialist = int(np.argmax(confidences))
                attended_context = final_immediate
            else:
                routing_context = protocol_context(
                    condition, cue, final_immediate, true_context, step
                )
                conductor = (
                    no_protocol_conductor
                    if condition == "learned_no_protocol"
                    else protocol_conductor
                )
                specialist = conductor.choose(routing_context)
                attended_context = routing_context

            action = int(proposals[specialist])
            correct = action == correct_action
            reward = (1.0 if correct else -1.0) - COMPUTE_COST[specialist]
            valence = 1.0 if correct else -1.0
            correct_count += int(correct)
            total_reward += reward
            total_compute += COMPUTE_COST[specialist]
            selections[SPECIALISTS[specialist]] += 1

            if condition not in {
                "static_confidence_router",
                "learned_no_protocol",
                "protocol_lesion",
            }:
                protocol.append(
                    ProtocolEntry(
                        step=step,
                        attended_context=CONTEXTS[attended_context],
                        selected_specialist=SPECIALISTS[specialist],
                        specialist_confidence=float(confidences[specialist]),
                        predicted_action=action,
                        executed_action=action,
                        correct_action=correct_action,
                        outcome="success" if correct else "error",
                        reward=float(reward),
                        valence=valence,
                    )
                )

        report_context, reported_specialist, reported_valence = delayed_report(
            condition, protocol, final_immediate
        )
        dominant = selections.most_common(1)[0][0]
        best_specialist = SPECIALISTS[int(np.argmax(COMPETENCE[true_context] - COMPUTE_COST))]
        report_context_correct = report_context == true_context
        report_control_aligned = (
            reported_specialist == dominant and reported_specialist != "unavailable"
        )
        episode_rows.append(
            {
                "condition": condition,
                "episode": episode,
                "true_context": CONTEXTS[true_context],
                "cue_context": CONTEXTS[cue],
                "accuracy": correct_count / steps,
                "reward_per_step": total_reward / steps,
                "compute_cost_per_step": total_compute / steps,
                "dominant_specialist": dominant,
                "best_specialist": best_specialist,
                "routing_optimal": float(dominant == best_specialist),
                "report_context_correct": float(report_context_correct),
                "report_control_alignment": float(report_control_aligned),
                "reported_valence": reported_valence,
                "actual_valence": 2.0 * correct_count / steps - 1.0,
                "valence_report_error": abs(
                    reported_valence - (2.0 * correct_count / steps - 1.0)
                )
                if protocol
                else 1.0,
                "protocol_entries": len(protocol),
            }
        )
        if protocol and len(sample_protocol) < 2:
            sample_protocol.append(
                {
                    "condition": condition,
                    "true_context": CONTEXTS[true_context],
                    "delayed_report": {
                        "context": CONTEXTS[report_context],
                        "dominant_specialist": reported_specialist,
                        "mean_valence": reported_valence,
                    },
                    "entries": [asdict(entry) for entry in protocol[:4]],
                }
            )
    return episode_rows, sample_protocol


def summarize(rows: list[dict]) -> dict[str, float]:
    return {
        "action_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "reward_per_step": float(np.mean([row["reward_per_step"] for row in rows])),
        "compute_cost_per_step": float(
            np.mean([row["compute_cost_per_step"] for row in rows])
        ),
        "optimal_routing_rate": float(
            np.mean([row["routing_optimal"] for row in rows])
        ),
        "delayed_context_report_accuracy": float(
            np.mean([row["report_context_correct"] for row in rows])
        ),
        "report_control_alignment": float(
            np.mean([row["report_control_alignment"] for row in rows])
        ),
        "valence_report_error": float(
            np.mean([row["valence_report_error"] for row in rows])
        ),
        "mean_protocol_entries": float(
            np.mean([row["protocol_entries"] for row in rows])
        ),
    }


def paired_seed_summary(seed_results: list[dict]) -> dict[str, float]:
    intact = np.array(
        [result["protocol_intact"]["reward_per_step"] for result in seed_results]
    )
    lesion = np.array(
        [result["protocol_lesion"]["reward_per_step"] for result in seed_results]
    )
    scrambled = np.array(
        [result["protocol_scrambled"]["reward_per_step"] for result in seed_results]
    )
    no_protocol = np.array(
        [result["learned_no_protocol"]["reward_per_step"] for result in seed_results]
    )
    lesion_delta = intact - lesion
    scrambled_delta = intact - scrambled
    no_protocol_delta = intact - no_protocol

    def sign_test_p(delta: np.ndarray) -> float:
        nonzero = delta[np.abs(delta) > 1e-12]
        if not len(nonzero):
            return 1.0
        positives = int(np.sum(nonzero > 0.0))
        tail_count = min(positives, len(nonzero) - positives)
        tail = sum(math.comb(len(nonzero), index) for index in range(tail_count + 1))
        return float(min(1.0, 2.0 * tail / (2 ** len(nonzero))))

    def paired_effect(delta: np.ndarray) -> float:
        spread = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        return float(np.mean(delta) / spread) if spread > 1e-12 else float("inf")

    return {
        "intact_minus_lesion_reward": float(np.mean(lesion_delta)),
        "intact_minus_scrambled_reward": float(np.mean(scrambled_delta)),
        "intact_minus_no_protocol_reward": float(np.mean(no_protocol_delta)),
        "intact_vs_lesion_paired_effect_dz": paired_effect(lesion_delta),
        "intact_vs_lesion_sign_test_p": sign_test_p(lesion_delta),
        "intact_vs_scrambled_sign_test_p": sign_test_p(scrambled_delta),
        "intact_beats_lesion_seeds": int(np.sum(intact > lesion)),
        "intact_beats_scrambled_seeds": int(np.sum(intact > scrambled)),
        "seed_count": len(seed_results),
    }


def plot_summary(summary: dict[str, dict[str, float]], path) -> None:
    names = list(summary)
    labels = [
        "static",
        "learned\nno protocol",
        "protocol\nintact",
        "protocol\nlesion",
        "protocol\nscrambled",
        "false\nprotocol",
        "oracle",
    ]
    metrics = (
        "action_accuracy",
        "reward_per_step",
        "delayed_context_report_accuracy",
        "report_control_alignment",
    )
    colors = ("#168aad", "#52b788", "#f4a261", "#9b5de5")
    x = np.arange(len(names))
    width = 0.19
    fig, ax = plt.subplots(figsize=(14, 6))
    for index, metric in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            [summary[name][metric] for name in names],
            width,
            label=metric,
            color=colors[index],
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Cortical Conductor: Learned Coordination and Protocol Lesions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = 5 if args.quick else 20
    train_episodes = 500 if args.quick else 1400
    evaluation_episodes = 120 if args.quick else 400
    steps = 10
    conditions = (
        "static_confidence_router",
        "learned_no_protocol",
        "protocol_intact",
        "protocol_lesion",
        "protocol_scrambled",
        "false_protocol_intervention",
        "oracle_conductor",
    )

    seed_results = []
    all_rows = {condition: [] for condition in conditions}
    samples = {}
    q_tables = []
    for seed_index in range(seeds):
        seed = 8701 + 101 * seed_index
        protocol_conductor = LearnedConductor(seed)
        no_protocol_conductor = LearnedConductor(seed + 1)
        train_conductor(
            protocol_conductor,
            protocol_enabled=True,
            episodes=train_episodes,
            steps=steps,
            seed=seed + 2,
        )
        train_conductor(
            no_protocol_conductor,
            protocol_enabled=False,
            episodes=train_episodes,
            steps=steps,
            seed=seed + 3,
        )
        per_seed = {}
        for condition in conditions:
            rows, protocols = run_evaluation(
                condition,
                protocol_conductor,
                no_protocol_conductor,
                episodes=evaluation_episodes,
                steps=steps,
                seed=seed + 1000,
            )
            all_rows[condition].extend(rows)
            per_seed[condition] = summarize(rows)
            if protocols and condition not in samples:
                samples[condition] = protocols
        seed_results.append(per_seed)
        q_tables.append(
            {
                "seed": seed,
                "protocol_q": protocol_conductor.q.tolist(),
                "no_protocol_q": no_protocol_conductor.q.tolist(),
            }
        )

    summary = {condition: summarize(rows) for condition, rows in all_rows.items()}
    contrasts = paired_seed_summary(seed_results)
    payload = {
        "theory": "Bach cortical conductor theory software analogue",
        "conditions": list(conditions),
        "frozen_specialists": list(SPECIALISTS),
        "competence_matrix": COMPETENCE.tolist(),
        "summary": summary,
        "paired_seed_contrasts": contrasts,
        "learned_q_tables": q_tables,
        "sample_protocols": samples,
        "claim_boundary": (
            "A reward-trained executive used an integrated attention protocol to allocate "
            "control among frozen specialists and support delayed self-report. Lesion and "
            "false-protocol interventions test causal dependence. This is evidence for "
            "functional executive access, not phenomenal consciousness."
        ),
    }
    OUT.mkdir(exist_ok=True)
    metrics_path = OUT / "cortical_conductor_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_summary(summary, OUT / "cortical_conductor_summary.png")
    print("Cortical conductor lab complete")
    print(json.dumps({"summary": summary, "paired_seed_contrasts": contrasts}, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
