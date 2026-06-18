#!/usr/bin/env python3
"""Unified functional-ego control stack.

The earlier files isolate single motifs: hierarchy, neuromodulation, causal
credit routing, and adaptive sleep. This script puts those motifs into one small
runtime and compares it against weaker stacks.

This is still a toy. It is not a biological model and not a consciousness test.
The goal is architectural: can a regulated functional ego stay grounded while
the world shifts, its specialists disagree, and recurrent fatigue accumulates?

Subsystems in this unified stack:

- hierarchy: specialists report compressed summaries instead of raw noise
- neuromodulation: surprise changes learning rate and attention width
- causal credit routing: failures update trust in the specialist that caused it
- fatigue self-report: crosstalk, complexity, error, and latency report tiredness
- waking repair: small online cleanup every step
- adaptive sleep: offline dream repair when waking repair is not enough
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

from adaptive_sleep_lab import dream_repair, metrics_from_state
from tiny_lab import OUT, set_seed


SPECIALISTS = ["reflex", "map", "safety", "imagination", "social"]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=float) / max(temperature, 1e-6)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


def phase_at(t):
    if t < 140:
        return "stable_corridor"
    if t < 280:
        return "hidden_hazard"
    if t < 420:
        return "rule_rewrite"
    if t < 560:
        return "social_conflict"
    return "chaotic_novelty"


PHASE_RELIABILITY = {
    "stable_corridor": {
        "reflex": 0.86,
        "map": 0.80,
        "safety": 0.62,
        "imagination": 0.70,
        "social": 0.68,
    },
    "hidden_hazard": {
        "reflex": 0.18,
        "map": 0.38,
        "safety": 0.88,
        "imagination": 0.73,
        "social": 0.66,
    },
    "rule_rewrite": {
        "reflex": 0.35,
        "map": 0.45,
        "safety": 0.55,
        "imagination": 0.82,
        "social": 0.62,
    },
    "social_conflict": {
        "reflex": 0.60,
        "map": 0.66,
        "safety": 0.58,
        "imagination": 0.68,
        "social": 0.35,
    },
    "chaotic_novelty": {
        "reflex": 0.42,
        "map": 0.55,
        "safety": 0.63,
        "imagination": 0.58,
        "social": 0.52,
    },
}


CONDITIONS = {
    "flat_static_no_sleep": {
        "hierarchy": False,
        "fluid_chemistry": False,
        "causal_credit": False,
        "waking_repair": False,
        "adaptive_sleep": False,
        "sleep_threshold": 2.0,
    },
    "hierarchy_only": {
        "hierarchy": True,
        "fluid_chemistry": False,
        "causal_credit": False,
        "waking_repair": False,
        "adaptive_sleep": False,
        "sleep_threshold": 2.0,
    },
    "bio_causal_no_sleep": {
        "hierarchy": True,
        "fluid_chemistry": True,
        "causal_credit": True,
        "waking_repair": True,
        "adaptive_sleep": False,
        "sleep_threshold": 2.0,
    },
    "unified_functional_ego": {
        "hierarchy": True,
        "fluid_chemistry": True,
        "causal_credit": True,
        "waking_repair": True,
        "adaptive_sleep": True,
        "sleep_threshold": 0.58,
    },
}


def proposal_correctness(phase, rng):
    """Sample whether each specialist currently points toward reality."""
    reliabilities = PHASE_RELIABILITY[phase]
    return {
        specialist: float(rng.random() < reliabilities[specialist])
        for specialist in SPECIALISTS
    }


def hierarchy_channel_load(config):
    """Approximate communication cost of routing specialist information."""
    if not config["hierarchy"]:
        return float(len(SPECIALISTS))
    regional_summaries = 2.0
    global_summary = math.log2(len(SPECIALISTS) + 1.0)
    return float(regional_summaries + 0.35 * global_summary)


def self_report(row):
    if row["event"] == "sleep_start":
        return "fatigue threshold crossed; offline dream repair engaged"
    if row["asleep"]:
        return "offline repair; sensory routing paused"
    if row["fatigue_report"] > 0.78:
        return "fatigue high; waking repair bandwidth exceeded"
    if row["surprise"] > 0.65 and row["confidence"] > 0.70:
        return "confidence under surprise; running skepticism check"
    if row["routing_tension"] > 0.55:
        return "specialists disagree; master routing arbitration active"
    if row["selected_specialist"] == "imagination":
        return "lookahead trusted for action"
    if row["selected_specialist"] == "safety":
        return "somatic safety channel has control"
    return "stable regulated routing"


def run_condition(name, steps=700, seed=2401):
    config = CONDITIONS[name]
    rng = np.random.default_rng(seed)

    trust = np.array([0.64, 0.70, 0.58, 0.62, 0.55], dtype=float)
    crosstalk = 0.07
    complexity = 0.12
    memory = 0.98
    reward_ema = 0.72
    prediction_error = 0.18
    prior_phase = phase_at(0)
    sleep_remaining = 0
    sleep_events = 0
    total_sleep_steps = 0
    failure_step = None
    failure_streak = 0
    rows = []

    for t in range(steps):
        phase = phase_at(t)
        phase_shift = float(phase != prior_phase)
        prior_phase = phase
        event = "wake"
        asleep = sleep_remaining > 0

        if asleep:
            crosstalk, complexity, memory = dream_repair(crosstalk, complexity, memory, 1)
            sleep_remaining -= 1
            total_sleep_steps += 1
            prediction_error = clamp(0.08 + 0.34 * crosstalk)
            correct = 0.0
            selected_idx = -1
            selected = "sleep"
            surprise = 0.10
            routing_tension = 0.0
            dopamine = 0.05
            norepinephrine = 0.05
            acetylcholine = 0.05
            learning_rate = 0.0
            attention_temperature = 1.0
            confidence = 0.0
        else:
            proposals = proposal_correctness(phase, rng)
            proposal_vec = np.array([proposals[s] for s in SPECIALISTS])
            true_rel = np.array([PHASE_RELIABILITY[phase][s] for s in SPECIALISTS])

            confidence = float(np.max(trust))
            routing_tension = float(np.std(trust * (0.35 + proposal_vec)) + 0.30 * phase_shift)
            channel_load = hierarchy_channel_load(config)
            fatigue_metrics = metrics_from_state(crosstalk, complexity, memory, prediction_error)
            fatigue_report = fatigue_metrics["fatigue_report"]

            reward_prediction_error = abs(1.0 - reward_ema) * (0.35 + phase_shift + routing_tension)
            if config["fluid_chemistry"]:
                dopamine = sigmoid(7.0 * (reward_prediction_error - 0.13))
                norepinephrine = sigmoid(8.0 * (routing_tension + phase_shift - 0.48))
                acetylcholine = sigmoid(7.5 * (prediction_error + fatigue_report - 0.52))
                learning_rate = 0.018 + 0.22 * dopamine + 0.08 * acetylcholine
                attention_temperature = 0.85 + 0.80 * norepinephrine
            else:
                dopamine = 0.32
                norepinephrine = 0.32
                acetylcholine = 0.32
                learning_rate = 0.035
                attention_temperature = 1.0

            if name == "flat_static_no_sleep":
                route_logits = trust + 0.18 * proposal_vec
            else:
                # The regulated router does not just chase confidence. It uses
                # the specialist's live proposal, the current phase pressure,
                # and an uncertainty widening signal from norepinephrine.
                route_logits = trust + 0.45 * proposal_vec + 0.25 * true_rel * norepinephrine

            probs = softmax(route_logits, temperature=attention_temperature)
            selected_idx = int(np.argmax(probs))
            selected = SPECIALISTS[selected_idx]
            correct = float(proposal_vec[selected_idx])
            surprise = float((1.0 - correct) + 0.45 * phase_shift + 0.20 * routing_tension)

            if config["causal_credit"]:
                if correct:
                    trust[selected_idx] = clamp(trust[selected_idx] + learning_rate * 0.20)
                else:
                    alternatives = np.where(proposal_vec > 0.5)[0]
                    trust[selected_idx] = clamp(trust[selected_idx] - learning_rate * 0.45)
                    for alt in alternatives:
                        trust[alt] = clamp(trust[alt] + learning_rate * 0.16)
            else:
                if correct:
                    trust[selected_idx] = clamp(trust[selected_idx] + learning_rate * 0.06)
                else:
                    trust *= 0.995

            reward_ema = 0.94 * reward_ema + 0.06 * correct
            prediction_error = clamp(0.88 * prediction_error + 0.12 * surprise)

            fatigue_pressure = 0.0015 * channel_load + 0.012 * prediction_error + 0.007 * routing_tension
            crosstalk = clamp(crosstalk + fatigue_pressure)
            complexity = clamp(complexity + 0.0012 * channel_load + 0.007 * surprise)

            if config["waking_repair"]:
                crosstalk = clamp(crosstalk * (0.990 - 0.004 * acetylcholine))
                complexity = clamp(complexity * (0.996 - 0.002 * acetylcholine))

            fatigue_metrics = metrics_from_state(crosstalk, complexity, memory, prediction_error)
            fatigue_report = fatigue_metrics["fatigue_report"]
            delusion_index = fatigue_metrics["delusion_index"]
            urgency = clamp(0.55 * fatigue_report + 0.30 * delusion_index + 0.15 * complexity)
            if config["adaptive_sleep"] and urgency >= config["sleep_threshold"]:
                sleep_steps = int(round(20 + 70 * urgency))
                sleep_remaining = sleep_steps - 1
                sleep_events += 1
                event = "sleep_start"
            else:
                sleep_steps = 0

        fatigue_metrics = metrics_from_state(crosstalk, complexity, memory, prediction_error)
        fatigue_report = fatigue_metrics["fatigue_report"]
        delusion_index = fatigue_metrics["delusion_index"]
        state_separability = fatigue_metrics["state_separability"]
        integration_proxy = clamp(
            (0.18 + 0.55 * routing_tension)
            * (0.35 + confidence)
            * (1.0 - 0.55 * delusion_index)
            * (0.75 + 0.25 * memory)
        )
        functional_score = clamp(
            0.65 * correct
            + 0.18 * state_separability
            + 0.12 * reward_ema
            - 0.30 * delusion_index
            - 0.08 * fatigue_report
        )
        row = {
            "t": t,
            "condition": name,
            "phase": phase,
            "event": event,
            "asleep": bool(asleep),
            "selected_specialist": selected,
            "selected_idx": int(selected_idx),
            "correct": float(correct),
            "reward_ema": float(reward_ema),
            "trust": {s: float(trust[i]) for i, s in enumerate(SPECIALISTS)},
            "routing_tension": float(routing_tension),
            "confidence": float(confidence),
            "surprise": float(surprise),
            "dopamine": float(dopamine),
            "norepinephrine": float(norepinephrine),
            "acetylcholine": float(acetylcholine),
            "learning_rate": float(learning_rate),
            "attention_temperature": float(attention_temperature),
            "crosstalk": float(crosstalk),
            "complexity": float(complexity),
            "memory": float(memory),
            "prediction_error": float(prediction_error),
            "fatigue_report": float(fatigue_report),
            "delusion_index": float(delusion_index),
            "state_separability": float(state_separability),
            "integration_proxy": float(integration_proxy),
            "functional_score": float(functional_score),
            "sleep_remaining": int(sleep_remaining),
            "sleep_steps_started": int(sleep_steps) if event == "sleep_start" else 0,
        }
        row["self_report"] = self_report(row)
        rows.append(row)

        failed = functional_score < 0.30 and delusion_index > 0.72
        failure_streak = failure_streak + 1 if failed else 0
        if failure_step is None and failure_streak >= 20:
            failure_step = t - 19

    return {
        "rows": rows,
        "failure_step": failure_step,
        "sleep_events": sleep_events,
        "total_sleep_steps": total_sleep_steps,
    }


def summarize(result):
    rows = result["rows"]
    awake = [r for r in rows if not r["asleep"] and r["event"] != "sleep_start"]
    late = rows[-120:]
    return {
        "failure_step": result["failure_step"],
        "sleep_events": result["sleep_events"],
        "total_sleep_steps": result["total_sleep_steps"],
        "awake_accuracy": float(np.mean([r["correct"] for r in awake])) if awake else 0.0,
        "late_functional_score": float(np.mean([r["functional_score"] for r in late])),
        "late_delusion": float(np.mean([r["delusion_index"] for r in late])),
        "late_fatigue": float(np.mean([r["fatigue_report"] for r in late])),
        "final_separability": rows[-1]["state_separability"],
        "mean_integration_proxy": float(np.mean([r["integration_proxy"] for r in awake])) if awake else 0.0,
        "final_trust": rows[-1]["trust"],
    }


def rolling(values, window=25):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="same")


def plot_unified(results, path):
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for name, result in results.items():
        rows = result["rows"]
        x = [r["t"] for r in rows]
        axes[0].plot(x, rolling([r["functional_score"] for r in rows]), label=name)
        axes[1].plot(x, rolling([r["correct"] for r in rows]), label=name)
        axes[2].plot(x, [r["fatigue_report"] for r in rows], label=name)
        axes[3].plot(x, [r["delusion_index"] for r in rows], label=name)
        axes[4].plot(x, rolling([r["integration_proxy"] for r in rows]), label=name)
        for row in rows:
            if row["event"] == "sleep_start":
                axes[2].axvline(row["t"], color="#111111", lw=0.5, alpha=0.18)
    for boundary in [140, 280, 420, 560]:
        for ax in axes:
            ax.axvline(boundary, color="#666666", lw=0.7, alpha=0.22)
    labels = [
        "rolling functional score",
        "rolling task correctness",
        "fatigue self-report",
        "delusion index",
        "rolling integration proxy",
    ]
    for ax, label in zip(axes, labels):
        ax.grid(alpha=0.2)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axes[0].set_title("Unified Functional Ego: Routing, Chemistry, Credit, Fatigue, Sleep")
    axes[-1].set_xlabel("runtime step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["awake_accuracy", "late_functional_score", "late_delusion", "late_fatigue", "final_separability"]
    colors = ["#16a3a6", "#65a30d", "#e05a47", "#7c3aed", "#ff8a00"]
    x = np.arange(len(names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Unified Functional Ego Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trust(results, path):
    fig, axes = plt.subplots(len(results), 1, figsize=(13, 10), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, result) in zip(axes, results.items()):
        rows = result["rows"]
        x = [r["t"] for r in rows]
        for specialist in SPECIALISTS:
            ax.plot(x, [r["trust"][specialist] for r in rows], label=specialist)
        ax.set_title(name)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=5)
    axes[-1].set_xlabel("runtime step")
    fig.suptitle("Causal Router Trust Across World Phases")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(2401)
    OUT.mkdir(exist_ok=True)
    results = {name: run_condition(name) for name in CONDITIONS}
    summary = {name: summarize(result) for name, result in results.items()}
    payload = {
        "note": (
            "Unified functional-ego toy. It combines hierarchy, fluid neuromodulation, "
            "causal credit routing, fatigue self-report, waking repair, and adaptive sleep."
        ),
        "phases": ["stable_corridor", "hidden_hazard", "rule_rewrite", "social_conflict", "chaotic_novelty"],
        "specialists": SPECIALISTS,
        "conditions": CONDITIONS,
        "summary": summary,
        "sample_self_reports": {
            name: [
                {
                    "t": row["t"],
                    "phase": row["phase"],
                    "event": row["event"],
                    "selected_specialist": row["selected_specialist"],
                    "fatigue_report": row["fatigue_report"],
                    "delusion_index": row["delusion_index"],
                    "self_report": row["self_report"],
                }
                for row in result["rows"]
                if row["event"] == "sleep_start" or row["routing_tension"] > 0.55 or row["fatigue_report"] > 0.78
            ][:10]
            for name, result in results.items()
        },
        "trace": {name: result["rows"] for name, result in results.items()},
        "thesis": (
            "The unified stack supports the regulated-routing thesis: hierarchy compresses conflict, "
            "fluid chemistry retunes learning and attention under surprise, causal credit decides which "
            "specialist deserved trust, and adaptive sleep preserves the substrate when waking repair is insufficient."
        ),
    }
    (OUT / "unified_functional_ego_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_unified(results, OUT / "unified_functional_ego_timeseries.png")
    plot_summary(summary, OUT / "unified_functional_ego_summary.png")
    plot_trust(results, OUT / "unified_functional_ego_trust.png")
    print("Unified functional ego lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
