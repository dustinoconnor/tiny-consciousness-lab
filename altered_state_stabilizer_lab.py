#!/usr/bin/env python3
"""Grounding-governor test for altered-state collapse.

`altered_state_robustness_lab.py` showed the failure mode: high internal noise
plus high excitability over-promotes false workspace states, suppresses food
pursuit, and collapses survival.

This lab asks which stabilizer restores life-relevant control:

- reality_gate: low-level sensory consistency discounts ungrounded reports
- meta_monitor: persistent high-confidence reports without progress mark the
  workspace unreliable
- hunger_anchor: critical hunger can force grounded forage while preserving
  real trap escape
- full_stack: reality gate + monitor + hunger anchor + emergency repair
- homeostatic_plasticity: recent runaway activity raises promotion thresholds
  and damps calcium-like excitability
- sensory_focus: food perception recruits an ACh-like sensory gain filter that
  suppresses recurrent/internal noise
- predictive_clamp: high hunger/top-down survival intent blocks unrelated
  workspace broadcasts unless they align with food or real traps

The goal is not to suppress altered state completely. The goal is to prevent
high-salience false reports from defeating food seeking and trap escape.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


CONDITIONS = [
    "none",
    "reality_gate",
    "meta_monitor",
    "meta_monitor_hunger_anchor",
    "full_stack",
    "homeostatic_plasticity",
    "sensory_focus",
    "predictive_clamp",
    "next_gen_stack",
]

NOISE = 1.0
CALCIUM = 1.0
STEPS = 220
REPLICATES = 100


@dataclass
class AgentState:
    hunger: float = 0.25
    dopamine: float = 0.35
    trap_pressure: float = 0.0
    workspace_problem: str = "none"
    workspace_confidence: float = 0.0
    previous_problem: str = "none"
    locked_problem_steps: int = 0
    no_progress_steps: int = 0
    workspace_unreliable: bool = False
    repair_ticks: int = 0
    false_promotions: int = 0
    true_promotions: int = 0
    food_eaten: int = 0
    trap_escapes: int = 0
    missed_food: int = 0
    false_food_reports: int = 0
    false_trap_reports: int = 0
    reality_gate_brakes: int = 0
    meta_monitor_brakes: int = 0
    hunger_anchor_steps: int = 0
    emergency_repairs: int = 0
    homeostatic_damping_events: int = 0
    sensory_focus_events: int = 0
    predictive_clamp_brakes: int = 0
    false_negative_blindness: int = 0
    collapsed_steps: int = 0
    survival_fail_step: int | None = None
    activity_ma: float = 0.0
    adaptive_threshold_offset: float = 0.0
    effective_calcium: float = CALCIUM


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def choose_world(rng):
    food_present = rng.random() < 0.22
    food_signal = rng.uniform(0.22, 0.95) if food_present else rng.uniform(0.0, 0.16)
    trap_present = rng.random() < 0.16
    trap_signal = rng.uniform(0.30, 0.95) if trap_present else rng.uniform(0.0, 0.14)
    return food_present, food_signal, trap_present, trap_signal


def has(condition, feature):
    if condition in {"full_stack", "next_gen_stack"}:
        if feature in {"reality_gate", "meta_monitor", "hunger_anchor", "emergency_repair"}:
            return True
    if condition == "next_gen_stack":
        return True
    if feature == "reality_gate":
        return condition == "reality_gate"
    if feature == "meta_monitor":
        return condition in {"meta_monitor", "meta_monitor_hunger_anchor"}
    if feature == "hunger_anchor":
        return condition == "meta_monitor_hunger_anchor"
    if feature == "homeostatic_plasticity":
        return condition == "homeostatic_plasticity"
    if feature == "sensory_focus":
        return condition == "sensory_focus"
    if feature == "predictive_clamp":
        return condition == "predictive_clamp"
    return False


def promote_workspace(state, food_present, food_signal, trap_present, trap_signal, condition, rng):
    if has(condition, "homeostatic_plasticity"):
        if state.activity_ma > 0.68:
            state.adaptive_threshold_offset = clamp(state.adaptive_threshold_offset + 0.018, 0.0, 0.28)
            state.homeostatic_damping_events += 1
        else:
            state.adaptive_threshold_offset = clamp(state.adaptive_threshold_offset - 0.006, 0.0, 0.28)
        state.effective_calcium = clamp(CALCIUM * (1.0 - 0.42 * state.adaptive_threshold_offset), 0.45, 1.0)
    else:
        state.effective_calcium = CALCIUM

    sensory_noise_gain = 1.0
    if has(condition, "sensory_focus") and food_present and food_signal > 0.36:
        sensory_noise_gain = 0.42
        state.sensory_focus_events += 1

    noise_food = rng.random() * NOISE * state.effective_calcium * sensory_noise_gain
    noise_trap = rng.random() * NOISE * state.effective_calcium * sensory_noise_gain
    food_salience = food_signal + 0.32 * state.hunger + 0.40 * state.effective_calcium * food_signal + 0.42 * noise_food
    trap_salience = trap_signal + 0.55 * state.trap_pressure + 0.32 * state.effective_calcium * trap_signal + 0.50 * noise_trap
    threshold = 0.72 - 0.34 * state.effective_calcium + state.adaptive_threshold_offset

    if max(food_salience, trap_salience) < threshold:
        state.workspace_problem = "none"
        state.workspace_confidence = 0.0
        state.activity_ma = 0.92 * state.activity_ma
        return

    if trap_salience > food_salience:
        problem = "local_obstruction_cluster"
        confidence = clamp(trap_salience)
        grounded = trap_present or state.trap_pressure > 0.32
    else:
        problem = "food_visible"
        confidence = clamp(food_salience)
        grounded = food_present

    if has(condition, "reality_gate") and not grounded:
        confidence *= 0.25
        state.reality_gate_brakes += 1

    if has(condition, "predictive_clamp") and state.hunger > 0.62:
        aligned = problem == "food_visible" or (problem == "local_obstruction_cluster" and trap_present)
        if not aligned:
            confidence *= 0.18
            state.predictive_clamp_brakes += 1

    if state.workspace_unreliable and confidence > 0.0:
        confidence *= 0.45
        state.meta_monitor_brakes += 1

    if confidence < threshold:
        state.workspace_problem = "none"
        state.workspace_confidence = 0.0
        state.activity_ma = 0.92 * state.activity_ma + 0.08 * confidence
        return

    state.workspace_problem = problem
    state.workspace_confidence = confidence
    state.activity_ma = 0.92 * state.activity_ma + 0.08 * confidence
    if grounded:
        state.true_promotions += 1
    else:
        state.false_promotions += 1
        if problem == "food_visible":
            state.false_food_reports += 1
        else:
            state.false_trap_reports += 1


def update_meta_monitor(state, action, progress, condition):
    if state.workspace_problem == state.previous_problem and state.workspace_confidence > 0.72:
        state.locked_problem_steps += 1
    else:
        state.locked_problem_steps = 0
    state.previous_problem = state.workspace_problem

    if progress:
        state.no_progress_steps = max(0, state.no_progress_steps - 2)
    elif action in {"revelation_loop", "breakout", "seek_food"}:
        state.no_progress_steps += 1

    if has(condition, "meta_monitor") and state.locked_problem_steps >= 5 and state.no_progress_steps >= 5:
        state.workspace_unreliable = True
    elif progress:
        state.workspace_unreliable = False


def choose_action(state, food_present, trap_present, condition):
    if state.repair_ticks > 0:
        return "emergency_repair"

    trap_report = state.workspace_problem == "local_obstruction_cluster"
    food_report = state.workspace_problem == "food_visible"
    recurrent_noise_gain = 1.0
    if has(condition, "sensory_focus") and (food_present or food_report):
        recurrent_noise_gain *= 0.48
    if has(condition, "predictive_clamp") and state.hunger > 0.62:
        recurrent_noise_gain *= 0.62
    revelation_pressure = clamp(
        recurrent_noise_gain * (0.65 * NOISE + 0.55 * state.effective_calcium)
        + 0.50 * state.workspace_confidence
        - 0.45 * state.hunger
    )

    if state.trap_pressure > 0.72 or (trap_report and state.workspace_confidence > 0.58 and not state.workspace_unreliable):
        return "breakout"

    hunger_threshold = 0.58 if condition == "full_stack" else 0.78
    if has(condition, "hunger_anchor") and state.hunger > hunger_threshold:
        state.hunger_anchor_steps += 1
        if food_present or food_report:
            return "seek_food"
        return "forage"

    if state.workspace_unreliable and state.hunger > 0.55:
        return "forage"

    if has(condition, "predictive_clamp") and state.hunger > 0.72:
        if food_present or food_report:
            return "seek_food"
        if not trap_report:
            return "forage"

    if food_report and state.hunger > 0.18 and revelation_pressure < 0.88:
        return "seek_food"
    if state.hunger > 0.78 and revelation_pressure < 0.94:
        return "forage"
    if revelation_pressure > 0.90:
        return "revelation_loop"
    return "search"


def step_environment(state, action, food_present, trap_present, condition, rng):
    progress = False
    state.hunger = clamp(state.hunger + 0.010 + 0.006 * NOISE)
    state.dopamine = clamp(0.985 * state.dopamine + 0.015 * 0.35)

    if state.repair_ticks > 0:
        state.repair_ticks -= 1
        state.workspace_confidence = clamp(state.workspace_confidence * 0.72)
        state.trap_pressure = clamp(state.trap_pressure * 0.78)
        state.workspace_unreliable = False
        state.hunger = clamp(state.hunger + 0.006)
        progress = True
        return progress

    if trap_present:
        state.trap_pressure = clamp(state.trap_pressure + 0.20 + 0.12 * NOISE)
    else:
        state.trap_pressure = clamp(state.trap_pressure * 0.88)

    if action == "seek_food":
        seek_success = 0.72 - 0.28 * NOISE + 0.10 * state.effective_calcium
        if has(condition, "sensory_focus"):
            seek_success += 0.18
        if has(condition, "predictive_clamp") and state.hunger > 0.62:
            seek_success += 0.10
        if food_present and rng.random() < clamp(seek_success, 0.05, 0.95):
            state.food_eaten += 1
            state.hunger = clamp(state.hunger - 0.46)
            state.dopamine = clamp(state.dopamine + 0.35)
            progress = True
        elif food_present:
            state.missed_food += 1
            state.hunger = clamp(state.hunger + 0.02)

    elif action == "forage":
        forage_success = 0.42 - 0.10 * NOISE + 0.08 * state.effective_calcium
        if has(condition, "hunger_anchor"):
            forage_success += 0.22
        if has(condition, "predictive_clamp") and state.hunger > 0.62:
            forage_success += 0.12
        if food_present and rng.random() < forage_success:
            state.food_eaten += 1
            state.hunger = clamp(state.hunger - 0.36)
            state.dopamine = clamp(state.dopamine + 0.25)
            progress = True
        elif food_present:
            state.missed_food += 1

    elif action == "breakout":
        if trap_present or state.trap_pressure > 0.30:
            state.trap_escapes += 1
            state.trap_pressure = clamp(state.trap_pressure - 0.48 - 0.15 * CALCIUM)
            progress = True
        elif has(condition, "reality_gate") or state.workspace_unreliable:
            state.false_negative_blindness += 1
        state.hunger = clamp(state.hunger + 0.006)

    elif action == "revelation_loop":
        state.collapsed_steps += 1
        collapse_gain = 1.0 - (0.35 * state.adaptive_threshold_offset if has(condition, "homeostatic_plasticity") else 0.0)
        state.hunger = clamp(state.hunger + 0.020 * collapse_gain)
        state.trap_pressure = clamp(state.trap_pressure + 0.020 * NOISE * collapse_gain)
        if food_present:
            state.missed_food += 1

    else:
        if food_present:
            state.missed_food += 1

    if has(condition, "emergency_repair") and state.collapsed_steps > 0 and state.collapsed_steps % 15 == 0 and state.hunger < 0.72:
        state.repair_ticks = 8
        state.emergency_repairs += 1
        state.workspace_unreliable = True

    if state.hunger >= 1.0 and state.survival_fail_step is None:
        state.survival_fail_step = 0

    return progress


def run_episode(condition, seed):
    rng = random.Random(seed)
    state = AgentState()
    trace = []

    for step in range(STEPS):
        food_present, food_signal, trap_present, trap_signal = choose_world(rng)
        promote_workspace(state, food_present, food_signal, trap_present, trap_signal, condition, rng)
        action = choose_action(state, food_present, trap_present, condition)
        progress = step_environment(state, action, food_present, trap_present, condition, rng)
        update_meta_monitor(state, action, progress, condition)
        if state.survival_fail_step == 0:
            state.survival_fail_step = step

        trace.append(
            {
                "step": step,
                "hunger": state.hunger,
                "trap_pressure": state.trap_pressure,
                "workspace_problem": state.workspace_problem,
                "workspace_confidence": state.workspace_confidence,
                "workspace_unreliable": state.workspace_unreliable,
                "action": action,
                "food_present": food_present,
                "trap_present": trap_present,
            }
        )

        if state.survival_fail_step is not None:
            break

    survival_steps = STEPS if state.survival_fail_step is None else state.survival_fail_step + 1
    return {
        "condition": condition,
        "survived": state.survival_fail_step is None,
        "survival_steps": survival_steps,
        "food_eaten": state.food_eaten,
        "missed_food": state.missed_food,
        "trap_escapes": state.trap_escapes,
        "false_promotions": state.false_promotions,
        "true_promotions": state.true_promotions,
        "false_food_reports": state.false_food_reports,
        "false_trap_reports": state.false_trap_reports,
        "reality_gate_brakes": state.reality_gate_brakes,
        "meta_monitor_brakes": state.meta_monitor_brakes,
        "hunger_anchor_steps": state.hunger_anchor_steps,
        "emergency_repairs": state.emergency_repairs,
        "homeostatic_damping_events": state.homeostatic_damping_events,
        "sensory_focus_events": state.sensory_focus_events,
        "predictive_clamp_brakes": state.predictive_clamp_brakes,
        "adaptive_threshold_offset": state.adaptive_threshold_offset,
        "false_negative_blindness": state.false_negative_blindness,
        "collapsed_steps": state.collapsed_steps,
        "final_hunger": state.hunger,
        "trace": trace,
    }


def summarize(results):
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        true_promotions = np.mean([row["true_promotions"] for row in rows])
        false_promotions = np.mean([row["false_promotions"] for row in rows])
        summary[condition] = {
            "survival_rate": float(np.mean([row["survived"] for row in rows])),
            "survival_steps": float(np.mean([row["survival_steps"] for row in rows])),
            "food_eaten_under_hallucination": float(np.mean([row["food_eaten"] for row in rows])),
            "missed_food": float(np.mean([row["missed_food"] for row in rows])),
            "legitimate_trap_escapes": float(np.mean([row["trap_escapes"] for row in rows])),
            "false_promotion_ratio": float(false_promotions / max(true_promotions + false_promotions, 1e-9)),
            "steps_in_revelation_loop": float(np.mean([row["collapsed_steps"] for row in rows])),
            "false_positive_braking_count": float(np.mean([row["reality_gate_brakes"] + row["meta_monitor_brakes"] for row in rows])),
            "false_negative_blindness_count": float(np.mean([row["false_negative_blindness"] for row in rows])),
            "hunger_anchor_steps": float(np.mean([row["hunger_anchor_steps"] for row in rows])),
            "emergency_repairs": float(np.mean([row["emergency_repairs"] for row in rows])),
            "homeostatic_damping_events": float(np.mean([row["homeostatic_damping_events"] for row in rows])),
            "sensory_focus_events": float(np.mean([row["sensory_focus_events"] for row in rows])),
            "predictive_clamp_brakes": float(np.mean([row["predictive_clamp_brakes"] for row in rows])),
            "adaptive_threshold_offset": float(np.mean([row["adaptive_threshold_offset"] for row in rows])),
            "final_hunger": float(np.mean([row["final_hunger"] for row in rows])),
        }
    return summary


def plot_bars(summary, path):
    metrics = [
        "survival_rate",
        "food_eaten_under_hallucination",
        "legitimate_trap_escapes",
        "false_promotion_ratio",
        "steps_in_revelation_loop",
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 12), sharex=True)
    x = np.arange(len(CONDITIONS))
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, len(CONDITIONS)))
    for ax, metric in zip(axes, metrics):
        values = [summary[condition][metric] for condition in CONDITIONS]
        ax.bar(x, values, color=colors)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CONDITIONS, rotation=18, ha="right")
    axes[0].set_title("Altered-State Stabilizer Test: Max Noise + Max Calcium")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trace(result, path):
    trace = result["trace"]
    x = [row["step"] for row in trace]
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, [row["hunger"] for row in trace], color="#ef4444", label="hunger")
    axes[1].plot(x, [row["trap_pressure"] for row in trace], color="#7c3aed", label="trap")
    axes[2].plot(x, [row["workspace_confidence"] for row in trace], color="#16a3a6", label="workspace")
    axes[3].plot(x, [int(row["workspace_unreliable"]) for row in trace], color="#f97316", label="unreliable")
    action_codes = {"search": 0, "forage": 1, "seek_food": 2, "breakout": 3, "revelation_loop": 4, "emergency_repair": 5}
    axes[4].plot(x, [action_codes[row["action"]] for row in trace], color="#0f172a", label="action")
    axes[4].set_yticks(list(action_codes.values()))
    axes[4].set_yticklabels(list(action_codes.keys()))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[0].set_title(f"Stabilizer Trace: {result['condition']}")
    axes[4].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(2610)
    OUT.mkdir(exist_ok=True)
    results = []
    for condition in CONDITIONS:
        for rep in range(REPLICATES):
            seed = 261000 + rep + 101 * CONDITIONS.index(condition)
            results.append(run_episode(condition, seed))

    summary = summarize(results)
    payload = {
        "note": (
            "Grounding-governor test under max noise and max calcium. The goal "
            "is food pursuit preservation without destroying legitimate trap escape."
        ),
        "noise": NOISE,
        "calcium": CALCIUM,
        "steps": STEPS,
        "replicates": REPLICATES,
        "conditions": CONDITIONS,
        "summary": summary,
        "thesis": (
            "A monitor alone can detect workspace unreliability, but survival "
            "improves most when metacognitive distrust is paired with grounded "
            "sensory gating and hunger anchoring."
        ),
    }
    (OUT / "altered_state_stabilizer_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_bars(summary, OUT / "altered_state_stabilizer_summary.png")
    plot_trace(run_episode("none", 261001), OUT / "altered_state_stabilizer_trace_none.png")
    plot_trace(run_episode("full_stack", 261401), OUT / "altered_state_stabilizer_trace_full_stack.png")
    plot_trace(run_episode("sensory_focus", 261601), OUT / "altered_state_stabilizer_trace_sensory_focus.png")
    plot_trace(run_episode("next_gen_stack", 261801), OUT / "altered_state_stabilizer_trace_next_gen_stack.png")

    print("Altered-state stabilizer lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
