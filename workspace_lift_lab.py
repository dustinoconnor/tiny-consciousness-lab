#!/usr/bin/env python3
"""Workspace lifting and intervention toy experiment.

This lab translates the Anthropic "J-space" conversation into this repo's
explicit architecture style. We are not reproducing a Jacobian lens. Instead,
we test the functional claim:

    Does a compressed, reportable workspace packet improve flexible control
    and transfer across superficially different obstruction pockets?

Conditions:

- reflex_only: raw blocked/stuck signals drive local escape only.
- private_modules: movement, valence, and memory compute internal states, but
  nothing is lifted into a shared reportable packet.
- global_workspace: surprise/tension promotes a generalized packet into a
  shared workspace that movement, memory, valence, and report modules can read.
- workspace_intervention: the workspace packet is forced from the beginning,
  testing whether the representation is causally active.

The result is meant to be an access-consciousness-style test: reportable,
reusable, causally active state, not phenomenal consciousness.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


CONDITIONS = [
    "reflex_only",
    "private_modules",
    "global_workspace",
    "workspace_intervention",
]


SCENARIOS = {
    "tree_pocket": {
        "kind": "enclosure",
        "density": 0.82,
        "novelty": 0.20,
        "label_hint": "trees",
    },
    "rock_pocket": {
        "kind": "enclosure",
        "density": 0.88,
        "novelty": 0.38,
        "label_hint": "rocks",
    },
    "dense_mushrooms": {
        "kind": "enclosure",
        "density": 0.68,
        "novelty": 0.52,
        "label_hint": "mushrooms",
    },
    "false_alarm": {
        "kind": "open",
        "density": 0.18,
        "novelty": 0.30,
        "label_hint": "open_path",
    },
}


TASK_SEQUENCE = ["tree_pocket", "rock_pocket", "dense_mushrooms", "false_alarm"]


@dataclass
class ModuleState:
    tension: float = 0.0
    valence: float = 0.25
    arousal: float = 0.15
    trap_memory: float = 0.0
    local_attempts: int = 0
    generalized_cluster_memory: float = 0.0


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def empty_workspace():
    return {
        "intent": "continue_heading",
        "problem": "none",
        "strategy": "local_probe",
        "feeling": "calm_positive_valence",
        "confidence": 0.0,
    }


def obstruction_probability(scenario, action, state, workspace, condition):
    density = scenario["density"]
    base = density

    if action == "forward":
        base += 0.05
    elif action == "local_escape":
        base -= 0.20
    elif action == "breakout_arc":
        base -= 0.55
    elif action == "wander":
        base -= 0.12

    if workspace["problem"] == "local_obstruction_cluster":
        base -= 0.12 * workspace["confidence"]

    if condition == "global_workspace":
        base -= 0.16 * state.generalized_cluster_memory
    elif condition == "private_modules":
        # Private memory helps only inside the current pocket. It does not
        # generalize through a shared label.
        base -= 0.05 * state.trap_memory

    if scenario["kind"] == "open":
        base -= 0.30

    return clamp(base, 0.03, 0.97)


def choose_action(condition, state, workspace, scenario, rng):
    if condition in {"global_workspace", "workspace_intervention"}:
        if workspace["strategy"] == "breakout_arc":
            return "breakout_arc"
        if workspace["problem"] == "false_alarm":
            return "forward"

    if condition == "private_modules" and state.trap_memory > 0.72:
        return "breakout_arc" if rng.random() < 0.45 else "local_escape"

    if state.local_attempts >= 4:
        return "wander" if condition == "reflex_only" else "local_escape"
    if state.tension > 0.45:
        return "local_escape"
    return "forward"


def promote_workspace(condition, state, scenario, workspace, forced_packet=None):
    if condition == "workspace_intervention" and forced_packet is not None:
        return dict(forced_packet), True

    if condition != "global_workspace":
        return workspace, False

    surprise = 0.55 * state.tension + 0.35 * state.trap_memory + 0.10 * scenario["novelty"]
    learned_bonus = 0.18 * state.generalized_cluster_memory
    threshold = 0.56 - learned_bonus

    if scenario["kind"] == "open" and state.tension < 0.35:
        return {
            "intent": "continue_heading",
            "problem": "false_alarm",
            "strategy": "continue_forward",
            "feeling": "low_arousal_positive_valence",
            "confidence": clamp(0.75 - state.tension),
        }, True

    if surprise >= threshold:
        confidence = clamp(surprise + 0.20 * state.generalized_cluster_memory)
        return {
            "intent": "continue_heading",
            "problem": "local_obstruction_cluster",
            "strategy": "breakout_arc",
            "feeling": "high_arousal_negative_valence",
            "confidence": confidence,
        }, True

    return workspace, False


def report_matches_state(workspace, state, scenario):
    if workspace["confidence"] <= 0.0:
        return 0.0
    if scenario["kind"] == "open":
        return float(workspace["problem"] == "false_alarm")
    return float(workspace["problem"] == "local_obstruction_cluster")


def run_task(condition, scenario_name, carried_memory=0.0, seed=0, max_steps=80, forced_packet=None):
    rng = random.Random(seed)
    scenario = SCENARIOS[scenario_name]
    state = ModuleState(generalized_cluster_memory=carried_memory)
    workspace = empty_workspace()
    rows = []
    collisions = 0
    repeated_collisions = 0
    promotions = 0
    breakout_steps = 0
    progress = 0.0
    escape_step = None
    last_blocked = False

    for t in range(max_steps):
        workspace, promoted = promote_workspace(condition, state, scenario, workspace, forced_packet)
        promotions += int(promoted)
        action = choose_action(condition, state, workspace, scenario, rng)
        blocked = rng.random() < obstruction_probability(scenario, action, state, workspace, condition)

        if blocked:
            collisions += 1
            repeated_collisions += int(last_blocked)
            state.local_attempts += 1
            state.tension = clamp(0.72 * state.tension + 0.30 + 0.18 * scenario["density"])
            state.trap_memory = clamp(state.trap_memory + 0.18 + 0.10 * scenario["density"])
            state.valence = clamp(state.valence - 0.20, -1.0, 1.0)
            state.arousal = clamp(state.arousal + 0.22)
        else:
            state.local_attempts = max(0, state.local_attempts - 1)
            state.tension = clamp(0.82 * state.tension - 0.06)
            state.trap_memory = clamp(0.92 * state.trap_memory)
            state.valence = clamp(state.valence + 0.08, -1.0, 1.0)
            state.arousal = clamp(0.86 * state.arousal)

            if action == "breakout_arc":
                progress += 0.13
                breakout_steps += 1
            elif action == "local_escape":
                progress += 0.07
            elif action == "forward":
                progress += 0.09 if scenario["kind"] == "open" else 0.04
            else:
                progress += 0.03

        if workspace["problem"] == "local_obstruction_cluster" and not blocked:
            state.generalized_cluster_memory = clamp(state.generalized_cluster_memory + 0.08)

        rows.append(
            {
                "t": t,
                "scenario": scenario_name,
                "condition": condition,
                "action": action,
                "blocked": blocked,
                "progress": progress,
                "tension": state.tension,
                "trap_memory": state.trap_memory,
                "valence": state.valence,
                "arousal": state.arousal,
                "workspace_problem": workspace["problem"],
                "workspace_strategy": workspace["strategy"],
                "workspace_confidence": workspace["confidence"],
                "report_match": report_matches_state(workspace, state, scenario),
            }
        )

        last_blocked = blocked
        if progress >= 1.0:
            escape_step = t + 1
            break

    success = escape_step is not None
    reportable_rows = [row for row in rows if row["workspace_confidence"] > 0.0]
    self_report_accuracy = float(np.mean([row["report_match"] for row in reportable_rows])) if reportable_rows else 0.0

    return {
        "condition": condition,
        "scenario": scenario_name,
        "success": success,
        "steps_to_escape": escape_step if success else max_steps,
        "collisions": collisions,
        "repeated_collisions": repeated_collisions,
        "workspace_promotions": promotions,
        "breakout_steps": breakout_steps,
        "self_report_accuracy": self_report_accuracy,
        "generalized_cluster_memory": state.generalized_cluster_memory,
        "rows": rows,
    }


def run_condition(condition, replicates=40):
    forced_packet = {
        "intent": "continue_heading",
        "problem": "local_obstruction_cluster",
        "strategy": "breakout_arc",
        "feeling": "high_arousal_negative_valence",
        "confidence": 0.85,
    }
    results = []
    for rep in range(replicates):
        carried_memory = 0.0
        for index, scenario_name in enumerate(TASK_SEQUENCE):
            result = run_task(
                condition,
                scenario_name,
                carried_memory=carried_memory,
                seed=1000 + 97 * rep + 13 * index + CONDITIONS.index(condition),
                forced_packet=forced_packet,
            )
            carried_memory = result["generalized_cluster_memory"]
            results.append(result)
    return results


def summarize(results):
    summary = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in results if row["condition"] == condition]
        summary[condition] = {}
        for scenario_name in TASK_SEQUENCE:
            rows = [row for row in condition_rows if row["scenario"] == scenario_name]
            summary[condition][scenario_name] = {
                "escape_success": float(np.mean([row["success"] for row in rows])),
                "steps_to_escape": float(np.mean([row["steps_to_escape"] for row in rows])),
                "repeated_collision_count": float(np.mean([row["repeated_collisions"] for row in rows])),
                "workspace_promotions": float(np.mean([row["workspace_promotions"] for row in rows])),
                "self_report_accuracy": float(np.mean([row["self_report_accuracy"] for row in rows])),
                "breakout_steps": float(np.mean([row["breakout_steps"] for row in rows])),
            }

    for condition in CONDITIONS:
        tree_steps = summary[condition]["tree_pocket"]["steps_to_escape"]
        transfer_steps = np.mean(
            [
                summary[condition]["rock_pocket"]["steps_to_escape"],
                summary[condition]["dense_mushrooms"]["steps_to_escape"],
            ]
        )
        summary[condition]["generalization"] = {
            "tree_steps": float(tree_steps),
            "transfer_steps": float(transfer_steps),
            "transfer_gain_vs_tree": float(tree_steps - transfer_steps),
        }

    baseline = summary["reflex_only"]["tree_pocket"]["steps_to_escape"]
    intervention = summary["workspace_intervention"]["tree_pocket"]["steps_to_escape"]
    summary["workspace_intervention"]["intervention_effect_size"] = {
        "tree_steps_saved_vs_reflex": float(baseline - intervention)
    }
    return summary


def plot_summary(summary, path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    x = np.arange(len(CONDITIONS))
    width = 0.18
    colors = {
        "tree_pocket": "#16a3a6",
        "rock_pocket": "#7c3aed",
        "dense_mushrooms": "#ff8a00",
        "false_alarm": "#64748b",
    }

    for i, scenario_name in enumerate(TASK_SEQUENCE):
        offset = (i - 1.5) * width
        axes[0].bar(
            x + offset,
            [summary[c][scenario_name]["steps_to_escape"] for c in CONDITIONS],
            width,
            label=scenario_name,
            color=colors[scenario_name],
        )
        axes[1].bar(
            x + offset,
            [summary[c][scenario_name]["repeated_collision_count"] for c in CONDITIONS],
            width,
            color=colors[scenario_name],
        )
        axes[2].bar(
            x + offset,
            [summary[c][scenario_name]["self_report_accuracy"] for c in CONDITIONS],
            width,
            color=colors[scenario_name],
        )

    axes[0].set_ylabel("steps to escape")
    axes[1].set_ylabel("repeated collisions")
    axes[2].set_ylabel("self-report accuracy")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(CONDITIONS, rotation=12)
    axes[0].legend(fontsize=8, ncol=4)
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].set_title("Workspace Lift Test: Reportable Shared State Improves Transfer")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trace(results, path):
    chosen = []
    for condition in CONDITIONS:
        matches = [
            result
            for result in results
            if result["condition"] == condition and result["scenario"] == "rock_pocket"
        ]
        chosen.append(matches[0])

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for result in chosen:
        rows = result["rows"]
        x = [row["t"] for row in rows]
        label = result["condition"]
        axes[0].plot(x, [row["progress"] for row in rows], label=label)
        axes[1].plot(x, [row["tension"] for row in rows], label=label)
        axes[2].plot(x, [row["workspace_confidence"] for row in rows], label=label)
        axes[3].plot(x, [int(row["blocked"]) for row in rows], label=label)
    axes[0].set_ylabel("progress")
    axes[1].set_ylabel("tension")
    axes[2].set_ylabel("workspace confidence")
    axes[3].set_ylabel("blocked")
    axes[3].set_xlabel("step")
    axes[0].set_title("Rock Pocket Transfer Trace")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(2607)
    OUT.mkdir(exist_ok=True)
    results = []
    for condition in CONDITIONS:
        results.extend(run_condition(condition))
    summary = summarize(results)

    payload = {
        "note": (
            "J-space-inspired workspace lift test. The lab compares reflex-only, "
            "private modules, global workspace promotion, and forced workspace "
            "intervention across tree, rock, mushroom, and false-alarm scenarios."
        ),
        "conditions": CONDITIONS,
        "task_sequence": TASK_SEQUENCE,
        "summary": summary,
        "thesis": (
            "A workspace-like representation should be reportable, reusable by "
            "multiple downstream modules, causally active under intervention, and "
            "able to generalize from one obstruction type to another."
        ),
    }

    (OUT / "workspace_lift_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(summary, OUT / "workspace_lift_summary.png")
    plot_trace(results, OUT / "workspace_lift_trace.png")
    print("Workspace lift lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
