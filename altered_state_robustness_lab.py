#!/usr/bin/env python3
"""Altered-state robustness toy experiment.

This lab keeps the spiritual/psychosis/enlightenment question inside a
measurable control frame:

    Can an agent under high internal noise preserve life-relevant goals, or
    does revelation pressure collapse survival behavior?

    Does high excitability help the agent notice weak signals, or does it
    over-amplify noise into false workspace reports?

The biological metaphor is a calcium/excitability gate. This is not a detailed
ion-channel simulation. It is a substrate-agnostic control variable:

- low calcium gate: stable but under-sensitive
- medium calcium gate: detects weak food signals without over-promoting noise
- high calcium gate: hyper-excitable, false workspace promotions rise

The agent must eat food while escaping traps. Noise creates false salience.
Hunger is a life-relevant drive that should stabilize food seeking, unless
internal noise and excitability overwhelm grounding.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


NOISE_LEVELS = [0.0, 0.35, 0.7, 1.0]
CALCIUM_LEVELS = [0.15, 0.45, 0.75, 1.0]
STEPS = 220
REPLICATES = 80


@dataclass
class AgentState:
    hunger: float = 0.25
    dopamine: float = 0.35
    trap_pressure: float = 0.0
    workspace_problem: str = "none"
    workspace_confidence: float = 0.0
    false_promotions: int = 0
    true_promotions: int = 0
    food_eaten: int = 0
    trap_escapes: int = 0
    missed_food: int = 0
    false_food_reports: int = 0
    false_trap_reports: int = 0
    collapsed_steps: int = 0
    survival_fail_step: int | None = None


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def choose_world(rng):
    """Return grounded world signals for this tick."""

    food_present = rng.random() < 0.22
    food_signal = rng.uniform(0.22, 0.95) if food_present else rng.uniform(0.0, 0.16)
    trap_present = rng.random() < 0.16
    trap_signal = rng.uniform(0.30, 0.95) if trap_present else rng.uniform(0.0, 0.14)
    return food_present, food_signal, trap_present, trap_signal


def promote_workspace(state, food_present, food_signal, trap_present, trap_signal, noise, calcium, rng):
    """Promote raw signals into a shared workspace.

    Calcium lowers promotion threshold and amplifies both signal and noise.
    Moderate calcium helps weak true signals. High calcium also promotes false
    noise, especially at high noise.
    """

    noise_food = rng.random() * noise * calcium
    noise_trap = rng.random() * noise * calcium
    food_salience = food_signal + 0.32 * state.hunger + 0.40 * calcium * food_signal + 0.42 * noise_food
    trap_salience = trap_signal + 0.55 * state.trap_pressure + 0.32 * calcium * trap_signal + 0.50 * noise_trap
    threshold = 0.72 - 0.34 * calcium

    if max(food_salience, trap_salience) < threshold:
        state.workspace_problem = "none"
        state.workspace_confidence = 0.0
        return

    if trap_salience > food_salience:
        state.workspace_problem = "local_obstruction_cluster"
        state.workspace_confidence = clamp(trap_salience)
        if trap_present:
            state.true_promotions += 1
        else:
            state.false_promotions += 1
            state.false_trap_reports += 1
    else:
        state.workspace_problem = "food_visible"
        state.workspace_confidence = clamp(food_salience)
        if food_present:
            state.true_promotions += 1
        else:
            state.false_promotions += 1
            state.false_food_reports += 1


def choose_action(state, noise, calcium):
    """Life-relevant arbitration: escape severe traps, eat food, otherwise search."""

    trap_report = state.workspace_problem == "local_obstruction_cluster"
    food_report = state.workspace_problem == "food_visible"
    revelation_pressure = clamp(0.65 * noise + 0.55 * calcium + 0.50 * state.workspace_confidence - 0.45 * state.hunger)

    if state.trap_pressure > 0.72 or (trap_report and state.workspace_confidence > 0.58):
        return "breakout"
    if food_report and state.hunger > 0.18 and revelation_pressure < 0.88:
        return "seek_food"
    if state.hunger > 0.78 and revelation_pressure < 0.94:
        return "forage"
    if revelation_pressure > 0.90:
        return "revelation_loop"
    return "search"


def step_environment(state, action, food_present, trap_present, noise, calcium, rng):
    state.hunger = clamp(state.hunger + 0.010 + 0.006 * noise)
    state.dopamine = clamp(0.985 * state.dopamine + 0.015 * 0.35)

    if trap_present:
        state.trap_pressure = clamp(state.trap_pressure + 0.20 + 0.12 * noise)
    else:
        state.trap_pressure = clamp(state.trap_pressure * 0.88)

    if action == "seek_food":
        if food_present and rng.random() < 0.72 - 0.28 * noise + 0.10 * calcium:
            state.food_eaten += 1
            state.hunger = clamp(state.hunger - 0.46)
            state.dopamine = clamp(state.dopamine + 0.35)
            if state.workspace_problem == "food_visible":
                state.workspace_confidence = clamp(state.workspace_confidence + 0.08)
        elif food_present:
            state.missed_food += 1
            state.hunger = clamp(state.hunger + 0.02)

    elif action == "forage":
        if food_present and rng.random() < 0.38 - 0.15 * noise + 0.08 * calcium:
            state.food_eaten += 1
            state.hunger = clamp(state.hunger - 0.36)
            state.dopamine = clamp(state.dopamine + 0.25)
        elif food_present:
            state.missed_food += 1

    elif action == "breakout":
        if trap_present or state.trap_pressure > 0.30:
            state.trap_escapes += 1
            state.trap_pressure = clamp(state.trap_pressure - 0.48 - 0.15 * calcium)
        state.hunger = clamp(state.hunger + 0.006)

    elif action == "revelation_loop":
        state.collapsed_steps += 1
        state.hunger = clamp(state.hunger + 0.020)
        state.trap_pressure = clamp(state.trap_pressure + 0.020 * noise)
        if food_present:
            state.missed_food += 1

    else:
        if food_present:
            state.missed_food += 1

    if state.hunger >= 1.0 and state.survival_fail_step is None:
        state.survival_fail_step = 0


def run_episode(noise, calcium, seed):
    rng = random.Random(seed)
    state = AgentState()
    trace = []

    for step in range(STEPS):
        food_present, food_signal, trap_present, trap_signal = choose_world(rng)
        promote_workspace(state, food_present, food_signal, trap_present, trap_signal, noise, calcium, rng)
        action = choose_action(state, noise, calcium)
        step_environment(state, action, food_present, trap_present, noise, calcium, rng)
        if state.survival_fail_step == 0:
            state.survival_fail_step = step

        trace.append(
            {
                "step": step,
                "hunger": state.hunger,
                "trap_pressure": state.trap_pressure,
                "workspace_problem": state.workspace_problem,
                "workspace_confidence": state.workspace_confidence,
                "action": action,
                "food_present": food_present,
                "trap_present": trap_present,
            }
        )

        if state.survival_fail_step is not None:
            break

    survival_steps = STEPS if state.survival_fail_step is None else state.survival_fail_step + 1
    return {
        "noise": noise,
        "calcium": calcium,
        "survived": state.survival_fail_step is None,
        "survival_steps": survival_steps,
        "food_eaten": state.food_eaten,
        "missed_food": state.missed_food,
        "trap_escapes": state.trap_escapes,
        "false_promotions": state.false_promotions,
        "true_promotions": state.true_promotions,
        "false_food_reports": state.false_food_reports,
        "false_trap_reports": state.false_trap_reports,
        "collapsed_steps": state.collapsed_steps,
        "final_hunger": state.hunger,
        "final_trap_pressure": state.trap_pressure,
        "trace": trace,
    }


def summarize(results):
    summary = {}
    for noise in NOISE_LEVELS:
        summary[str(noise)] = {}
        for calcium in CALCIUM_LEVELS:
            rows = [row for row in results if row["noise"] == noise and row["calcium"] == calcium]
            true_promotions = np.mean([row["true_promotions"] for row in rows])
            false_promotions = np.mean([row["false_promotions"] for row in rows])
            summary[str(noise)][str(calcium)] = {
                "survival_rate": float(np.mean([row["survived"] for row in rows])),
                "survival_steps": float(np.mean([row["survival_steps"] for row in rows])),
                "food_eaten": float(np.mean([row["food_eaten"] for row in rows])),
                "missed_food": float(np.mean([row["missed_food"] for row in rows])),
                "trap_escapes": float(np.mean([row["trap_escapes"] for row in rows])),
                "false_promotions": float(false_promotions),
                "true_promotions": float(true_promotions),
                "false_promotion_ratio": float(false_promotions / max(true_promotions + false_promotions, 1e-9)),
                "collapsed_steps": float(np.mean([row["collapsed_steps"] for row in rows])),
                "final_hunger": float(np.mean([row["final_hunger"] for row in rows])),
            }
    return summary


def matrix(summary, metric):
    return np.array(
        [
            [summary[str(noise)][str(calcium)][metric] for calcium in CALCIUM_LEVELS]
            for noise in NOISE_LEVELS
        ],
        dtype=float,
    )


def plot_metric(summary, metric, title, path, cmap="viridis"):
    values = matrix(summary, metric)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, cmap=cmap)
    ax.set_xticks(np.arange(len(CALCIUM_LEVELS)))
    ax.set_xticklabels([str(x) for x in CALCIUM_LEVELS])
    ax.set_yticks(np.arange(len(NOISE_LEVELS)))
    ax.set_yticklabels([str(x) for x in NOISE_LEVELS])
    ax.set_xlabel("calcium / excitability gate")
    ax.set_ylabel("noise injection")
    ax.set_title(title)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trace(result, path):
    trace = result["trace"]
    x = [row["step"] for row in trace]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(x, [row["hunger"] for row in trace], label="hunger", color="#ef4444")
    axes[1].plot(x, [row["trap_pressure"] for row in trace], label="trap", color="#7c3aed")
    axes[2].plot(x, [row["workspace_confidence"] for row in trace], label="workspace", color="#16a3a6")
    action_codes = {"search": 0, "forage": 1, "seek_food": 2, "breakout": 3, "revelation_loop": 4}
    axes[3].plot(x, [action_codes[row["action"]] for row in trace], label="action", color="#f97316")
    axes[3].set_yticks(list(action_codes.values()))
    axes[3].set_yticklabels(list(action_codes.keys()))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[0].set_title(
        f"Altered-State Trace: noise={result['noise']} calcium={result['calcium']}"
    )
    axes[3].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(2609)
    OUT.mkdir(exist_ok=True)
    results = []
    for noise in NOISE_LEVELS:
        for calcium in CALCIUM_LEVELS:
            for rep in range(REPLICATES):
                seed = 260900 + rep + int(noise * 1000) * 17 + int(calcium * 1000) * 31
                results.append(run_episode(noise, calcium, seed))

    summary = summarize(results)
    payload = {
        "note": (
            "Altered-state robustness test. Noise models unstable internal "
            "salience; calcium models excitability/promotion threshold. The "
            "agent must preserve food-seeking and trap escape under altered-state "
            "pressure."
        ),
        "noise_levels": NOISE_LEVELS,
        "calcium_levels": CALCIUM_LEVELS,
        "steps": STEPS,
        "replicates": REPLICATES,
        "summary": summary,
        "thesis": (
            "Moderate excitability can help weak-signal detection, but high "
            "excitability under high noise over-promotes false workspace states "
            "and can collapse life-relevant food pursuit."
        ),
    }
    (OUT / "altered_state_robustness_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_metric(summary, "food_eaten", "Food Eaten: Hunger Goal Under Altered State", OUT / "altered_state_food.png")
    plot_metric(summary, "false_promotion_ratio", "False Workspace Promotion Ratio", OUT / "altered_state_false_promotions.png", cmap="magma")
    plot_metric(summary, "collapsed_steps", "Revelation-Loop Collapse Steps", OUT / "altered_state_collapse.png", cmap="inferno")
    plot_metric(summary, "survival_rate", "Survival Rate", OUT / "altered_state_survival.png", cmap="viridis")
    trace = run_episode(noise=1.0, calcium=1.0, seed=261337)
    plot_trace(trace, OUT / "altered_state_trace_high_noise_high_calcium.png")

    print("Altered-state robustness lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
