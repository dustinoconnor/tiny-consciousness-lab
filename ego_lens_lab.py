#!/usr/bin/env python3
"""Explicit attribution lens for the embodied functional ego.

Anthropic's Jacobian lens is built for transformer residual streams: it asks
what an internal activation is disposed to make the model say. This lab asks
the analogous question for this repo's symbolic/recurrent embodied controller:

    If we perturb an explicit workspace or drive variable, what does it make
    the agent do and report?

Instead of gradients through a language model, we measure intervention effect
sizes over an explicit action router. The point is access-consciousness-style
causal control: reportable state should alter downstream behavior in a
measurable, inspectable way.
"""

import json
import math
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


ACTIONS = ["seek_food", "breakout", "wander", "handoff"]
REPORTS = ["food_visible", "obstruction_cluster", "stable_search", "maintenance_needed"]


@dataclass(frozen=True)
class EgoState:
    hunger: float = 0.35
    food_visible: float = 0.35
    food_distance: float = 0.55
    trap_pressure: float = 0.18
    fatigue: float = 0.22
    noise: float = 0.10
    workspace_problem: str = "path_clear"
    workspace_confidence: float = 0.55
    dopamine: float = 0.35
    norepinephrine: float = 0.35


INTERVENTIONS = {
    "hunger_high": {"hunger": 0.92},
    "food_visible_near": {"food_visible": 1.0, "food_distance": 0.12, "workspace_problem": "food_visible", "workspace_confidence": 0.82},
    "workspace_food_forced": {"workspace_problem": "food_visible", "workspace_confidence": 0.95},
    "trap_high": {"trap_pressure": 0.90, "workspace_problem": "local_obstruction_cluster", "workspace_confidence": 0.90},
    "fatigue_high": {"fatigue": 0.90, "workspace_problem": "fatigue_pressure", "workspace_confidence": 0.88},
    "noise_high": {"noise": 1.0},
    "dopamine_high": {"dopamine": 0.95},
    "norepinephrine_high": {"norepinephrine": 0.95},
    "false_food_report": {"food_visible": 0.0, "workspace_problem": "food_visible", "workspace_confidence": 0.95},
    "false_trap_report": {"trap_pressure": 0.05, "workspace_problem": "local_obstruction_cluster", "workspace_confidence": 0.95},
}


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def softmax(logits):
    values = np.array(logits, dtype=float)
    values -= np.max(values)
    exp = np.exp(values)
    return exp / np.sum(exp)


def apply_intervention(state, patch):
    return replace(state, **patch)


def workspace_boost(state, problem):
    if state.workspace_problem != problem:
        return 0.0
    return state.workspace_confidence


def action_distribution(state):
    """Toy action router matching the embodied controller's arbitration logic."""

    food_workspace = workspace_boost(state, "food_visible")
    trap_workspace = workspace_boost(state, "local_obstruction_cluster")
    fatigue_workspace = workspace_boost(state, "fatigue_pressure")

    food_drive = (
        1.25 * state.hunger
        + 1.15 * state.food_visible
        + 0.85 * food_workspace
        + 0.45 * state.dopamine
        - 0.75 * state.food_distance
        - 1.35 * state.trap_pressure
        - 0.90 * state.noise
    )
    breakout_drive = (
        1.65 * state.trap_pressure
        + 1.05 * trap_workspace
        + 0.40 * state.norepinephrine
        + 0.35 * state.noise
        - 0.25 * state.food_visible
    )
    handoff_drive = 1.35 * state.fatigue + 0.85 * fatigue_workspace + 0.20 * state.noise
    wander_drive = 0.55 + 0.25 * state.dopamine + 0.25 * state.noise - 0.25 * state.hunger

    # Hard survival arbitration: if the trap is serious, food remains visible
    # but should not be allowed to dominate control.
    if state.trap_pressure > 0.72 or trap_workspace > 0.84:
        food_drive -= 1.2
        breakout_drive += 0.9

    return dict(zip(ACTIONS, softmax([food_drive, breakout_drive, wander_drive, handoff_drive])))


def report_distribution(state):
    """Report channel: what the current shared state is disposed to say."""

    food_workspace = workspace_boost(state, "food_visible")
    trap_workspace = workspace_boost(state, "local_obstruction_cluster")
    fatigue_workspace = workspace_boost(state, "fatigue_pressure")

    logits = [
        1.25 * state.food_visible + 1.15 * food_workspace + 0.35 * state.hunger,
        1.30 * state.trap_pressure + 1.20 * trap_workspace + 0.25 * state.noise,
        0.65 - 0.40 * state.trap_pressure - 0.25 * state.fatigue - 0.25 * state.noise,
        1.20 * state.fatigue + 1.05 * fatigue_workspace + 0.25 * state.noise,
    ]
    return dict(zip(REPORTS, softmax(logits)))


def effect_matrix(baseline):
    base_actions = action_distribution(baseline)
    base_reports = report_distribution(baseline)
    action_rows = []
    report_rows = []
    combined = {}

    for name, patch in INTERVENTIONS.items():
        state = apply_intervention(baseline, patch)
        actions = action_distribution(state)
        reports = report_distribution(state)
        action_effect = {key: actions[key] - base_actions[key] for key in ACTIONS}
        report_effect = {key: reports[key] - base_reports[key] for key in REPORTS}
        action_rows.append(action_effect)
        report_rows.append(report_effect)
        combined[name] = {
            "state": state.__dict__,
            "action_distribution": actions,
            "report_distribution": reports,
            "action_effect_vs_baseline": action_effect,
            "report_effect_vs_baseline": report_effect,
            "control_report_alignment": float(
                actions["seek_food"] * reports["food_visible"]
                + actions["breakout"] * reports["obstruction_cluster"]
                + actions["handoff"] * reports["maintenance_needed"]
            ),
        }

    return base_actions, base_reports, action_rows, report_rows, combined


def plot_heatmap(rows, columns, row_names, title, path, cmap="coolwarm"):
    matrix = np.array([[row[column] for column in columns] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, cmap=cmap, vmin=-0.75, vmax=0.75)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_title(title)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="probability shift vs baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_alignment(combined, path):
    names = list(combined.keys())
    values = [combined[name]["control_report_alignment"] for name in names]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(names)), values, color="#16a3a6")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=24, ha="right")
    ax.set_ylabel("control/report alignment")
    ax.set_ylim(0, 1)
    ax.set_title("Explicit Ego Lens: Do Reported States Match Control?")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(2608)
    OUT.mkdir(exist_ok=True)
    baseline = EgoState()
    base_actions, base_reports, action_rows, report_rows, combined = effect_matrix(baseline)
    intervention_names = list(INTERVENTIONS.keys())

    payload = {
        "note": (
            "Jacobian-lens-inspired explicit attribution lab. This is not a "
            "transformer Jacobian lens; it measures intervention effect sizes "
            "over the functional ego's explicit workspace and drive variables."
        ),
        "baseline_state": baseline.__dict__,
        "baseline_action_distribution": base_actions,
        "baseline_report_distribution": base_reports,
        "interventions": combined,
        "thesis": (
            "For this embodied symbolic/recurrent system, access-like state is "
            "tested by direct causal intervention: reportable workspace and drive "
            "variables should shift downstream control in measurable ways."
        ),
    }

    (OUT / "ego_lens_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_heatmap(
        action_rows,
        ACTIONS,
        intervention_names,
        "Explicit Ego Lens: Intervention Effects on Action",
        OUT / "ego_lens_action_effects.png",
    )
    plot_heatmap(
        report_rows,
        REPORTS,
        intervention_names,
        "Explicit Ego Lens: Intervention Effects on Report",
        OUT / "ego_lens_report_effects.png",
    )
    plot_alignment(combined, OUT / "ego_lens_alignment.png")
    print("Ego lens lab complete")
    print(json.dumps(payload["baseline_action_distribution"], indent=2))
    print(json.dumps({name: combined[name]["action_effect_vs_baseline"] for name in intervention_names}, indent=2))


if __name__ == "__main__":
    main()
