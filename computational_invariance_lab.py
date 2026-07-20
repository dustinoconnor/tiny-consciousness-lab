#!/usr/bin/env python3
"""Test organizational invariance across software realizations.

Computational functionalism proposes that the relevant causal organization,
rather than one material or coding format, constitutes cognition. This lab
implements the same compact Functional Ego transition system three independent
ways:

- symbolic dictionaries and explicit rules;
- a dense numerical state vector;
- event-ordered messages among local modules.

All three contain sensory memory, grounded valence, workspace selection,
confidence, action, and symbolic report. They are compared on ordinary
trajectories, hidden-state interventions, and unseen counterfactual inputs.

An observational replay control memorizes every action/report pair in the
ordinary evaluation traces. It can therefore imitate those traces perfectly
without realizing the internal transition structure. The decisive comparison
asks whether it preserves intervention responses.

This tests software-level causal and representational invariance. All
realizations still run on one Python process and one physical computer. The
experiment cannot establish computational sufficiency for consciousness or
phenomenal experience.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT


WORKSPACES = ("explore", "food", "danger", "obstacle")
WORKSPACE_INDEX = {name: index for index, name in enumerate(WORKSPACES)}
ACTIONS = {
    "explore": "scan",
    "food": "approach",
    "danger": "retreat",
    "obstacle": "detour",
}
REPORTS = {
    "explore": "uncertain_clear",
    "food": "food_opportunity",
    "danger": "threat_present",
    "obstacle": "path_obstructed",
}


@dataclass(frozen=True)
class Observation:
    food: float
    danger: float
    blocked: float
    outcome: float
    novelty: float


@dataclass(frozen=True)
class Output:
    workspace: str
    action: str
    report: str
    confidence: float
    valence: float
    memory_food: float
    memory_danger: float
    memory_blocked: float


def clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def workspace_scores(
    observation: Observation,
    memory_food: float,
    memory_danger: float,
    memory_blocked: float,
    valence: float,
) -> dict[str, float]:
    """Reference scoring helper used only by tests and data generation."""
    return {
        "explore": 0.24 + 0.22 * observation.novelty,
        "food": 0.62 * observation.food + 0.38 * memory_food + 0.10 * max(valence, 0.0),
        "danger": 0.62 * observation.danger + 0.38 * memory_danger + 0.10 * max(-valence, 0.0),
        "obstacle": 0.62 * observation.blocked + 0.38 * memory_blocked,
    }


class SymbolicEgo:
    def __init__(self):
        self.state = {
            "memory_food": 0.0,
            "memory_danger": 0.0,
            "memory_blocked": 0.0,
            "valence": 0.0,
            "workspace": "explore",
            "confidence": 0.24,
        }

    def step(self, observation: Observation, intervention: dict | None = None) -> Output:
        intervention = intervention or {}
        if intervention.get("erase_memory"):
            self.state["memory_food"] = 0.0
            self.state["memory_danger"] = 0.0
            self.state["memory_blocked"] = 0.0
        if intervention.get("flip_valence"):
            self.state["valence"] *= -1.0

        self.state["memory_food"] = clip(0.68 * self.state["memory_food"] + 0.32 * observation.food)
        self.state["memory_danger"] = clip(0.68 * self.state["memory_danger"] + 0.32 * observation.danger)
        self.state["memory_blocked"] = clip(0.68 * self.state["memory_blocked"] + 0.32 * observation.blocked)
        self.state["valence"] = float(np.clip(0.76 * self.state["valence"] + 0.24 * observation.outcome, -1.0, 1.0))

        scores = {
            "explore": 0.24 + 0.22 * observation.novelty,
            "food": 0.62 * observation.food + 0.38 * self.state["memory_food"] + 0.10 * max(self.state["valence"], 0.0),
            "danger": 0.62 * observation.danger + 0.38 * self.state["memory_danger"] + 0.10 * max(-self.state["valence"], 0.0),
            "obstacle": 0.62 * observation.blocked + 0.38 * self.state["memory_blocked"],
        }
        candidate = max(WORKSPACES, key=lambda name: scores[name])
        selected = self.state["workspace"] if intervention.get("hold_workspace") else candidate
        selected = intervention.get("force_workspace", selected)
        self.state["workspace"] = selected
        self.state["confidence"] = 1.0 if "force_workspace" in intervention else scores[selected]
        return self.output()

    def output(self) -> Output:
        workspace = self.state["workspace"]
        return Output(
            workspace=workspace,
            action=ACTIONS[workspace],
            report=REPORTS[workspace],
            confidence=float(self.state["confidence"]),
            valence=float(self.state["valence"]),
            memory_food=float(self.state["memory_food"]),
            memory_danger=float(self.state["memory_danger"]),
            memory_blocked=float(self.state["memory_blocked"]),
        )


class VectorEgo:
    """The same transition system encoded as a dense numerical register."""

    MF, MD, MB, VALENCE, WORKSPACE, CONFIDENCE = range(6)

    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.24], dtype=np.float64)

    def step(self, observation: Observation, intervention: dict | None = None) -> Output:
        intervention = intervention or {}
        if intervention.get("erase_memory"):
            self.state[:3] = 0.0
        if intervention.get("flip_valence"):
            self.state[self.VALENCE] = -self.state[self.VALENCE]

        sensory = np.array([observation.food, observation.danger, observation.blocked])
        self.state[:3] = np.clip(0.68 * self.state[:3] + 0.32 * sensory, 0.0, 1.0)
        self.state[self.VALENCE] = np.clip(
            0.76 * self.state[self.VALENCE] + 0.24 * observation.outcome,
            -1.0,
            1.0,
        )
        value = self.state[self.VALENCE]
        scores = np.array(
            [
                0.24 + 0.22 * observation.novelty,
                0.62 * observation.food + 0.38 * self.state[self.MF] + 0.10 * max(value, 0.0),
                0.62 * observation.danger + 0.38 * self.state[self.MD] + 0.10 * max(-value, 0.0),
                0.62 * observation.blocked + 0.38 * self.state[self.MB],
            ],
            dtype=np.float64,
        )
        candidate = int(np.argmax(scores))
        selected = int(self.state[self.WORKSPACE]) if intervention.get("hold_workspace") else candidate
        if "force_workspace" in intervention:
            selected = WORKSPACE_INDEX[intervention["force_workspace"]]
        self.state[self.WORKSPACE] = selected
        self.state[self.CONFIDENCE] = 1.0 if "force_workspace" in intervention else scores[selected]
        return self.output()

    def output(self) -> Output:
        workspace = WORKSPACES[int(self.state[self.WORKSPACE])]
        return Output(
            workspace=workspace,
            action=ACTIONS[workspace],
            report=REPORTS[workspace],
            confidence=float(self.state[self.CONFIDENCE]),
            valence=float(self.state[self.VALENCE]),
            memory_food=float(self.state[self.MF]),
            memory_danger=float(self.state[self.MD]),
            memory_blocked=float(self.state[self.MB]),
        )


class MessagePassingEgo:
    """The transition system realized as ordered local message handlers."""

    def __init__(self):
        self.modules = {
            "memory": {"food": 0.0, "danger": 0.0, "blocked": 0.0},
            "evaluation": {"valence": 0.0},
            "workspace": {"content": "explore", "confidence": 0.24},
        }

    def step(self, observation: Observation, intervention: dict | None = None) -> Output:
        intervention = intervention or {}
        memory = self.modules["memory"]
        evaluation = self.modules["evaluation"]
        workspace = self.modules["workspace"]

        if intervention.get("erase_memory"):
            for channel in memory:
                memory[channel] = 0.0
        if intervention.get("flip_valence"):
            evaluation["valence"] = -evaluation["valence"]

        sensory_messages = {
            "food": observation.food,
            "danger": observation.danger,
            "blocked": observation.blocked,
        }
        for channel, payload in sensory_messages.items():
            memory[channel] = clip(0.68 * memory[channel] + 0.32 * payload)
        evaluation["valence"] = float(
            np.clip(0.76 * evaluation["valence"] + 0.24 * observation.outcome, -1.0, 1.0)
        )

        proposals = [
            ("explore", 0.24 + 0.22 * observation.novelty),
            ("food", 0.62 * observation.food + 0.38 * memory["food"] + 0.10 * max(evaluation["valence"], 0.0)),
            ("danger", 0.62 * observation.danger + 0.38 * memory["danger"] + 0.10 * max(-evaluation["valence"], 0.0)),
            ("obstacle", 0.62 * observation.blocked + 0.38 * memory["blocked"]),
        ]
        candidate, candidate_confidence = max(proposals, key=lambda item: item[1])
        selected = workspace["content"] if intervention.get("hold_workspace") else candidate
        selected = intervention.get("force_workspace", selected)
        workspace["content"] = selected
        workspace["confidence"] = (
            1.0
            if "force_workspace" in intervention
            else next(score for name, score in proposals if name == selected)
        )
        return self.output()

    def output(self) -> Output:
        memory = self.modules["memory"]
        evaluation = self.modules["evaluation"]
        workspace_module = self.modules["workspace"]
        workspace = workspace_module["content"]
        return Output(
            workspace=workspace,
            action=ACTIONS[workspace],
            report=REPORTS[workspace],
            confidence=float(workspace_module["confidence"]),
            valence=float(evaluation["valence"]),
            memory_food=float(memory["food"]),
            memory_danger=float(memory["danger"]),
            memory_blocked=float(memory["blocked"]),
        )


REALIZATIONS = (SymbolicEgo, VectorEgo, MessagePassingEgo)


class ObservationalReplay:
    """A trace lookup that has outputs but no corresponding internal causes."""

    def __init__(self, outputs: list[Output]):
        self.outputs = outputs

    def output_at(self, step: int, intervention: dict | None = None) -> Output:
        del intervention
        return self.outputs[step]


def outputs_equal(left: Output, right: Output, tolerance: float = 1e-10) -> bool:
    categorical = (
        left.workspace == right.workspace
        and left.action == right.action
        and left.report == right.report
    )
    numerical = all(
        abs(getattr(left, name) - getattr(right, name)) <= tolerance
        for name in (
            "confidence",
            "valence",
            "memory_food",
            "memory_danger",
            "memory_blocked",
        )
    )
    return categorical and numerical


def output_pair_equal(left: Output, right: Output) -> bool:
    return left.action == right.action and left.report == right.report


def generate_trace(seed: int, steps: int = 96) -> list[Observation]:
    rng = np.random.default_rng(seed)
    prototypes = (
        Observation(0.0, 0.0, 0.0, 0.0, 0.9),
        Observation(1.0, 0.0, 0.0, 0.7, 0.1),
        Observation(0.12, 0.0, 0.0, 0.0, 0.3),
        Observation(0.0, 1.0, 0.0, -0.8, 0.1),
        Observation(0.0, 0.10, 0.0, 0.0, 0.3),
        Observation(0.0, 0.0, 1.0, -0.25, 0.2),
        Observation(0.44, 0.44, 0.0, 0.0, 0.7),
    )
    trace = []
    current = 0
    for step in range(steps):
        if step % 4 == 0 or rng.random() < 0.18:
            current = int(rng.integers(0, len(prototypes)))
        base = prototypes[current]
        jitter = rng.normal(0.0, 0.012, 5)
        trace.append(
            Observation(
                food=clip(base.food + jitter[0]),
                danger=clip(base.danger + jitter[1]),
                blocked=clip(base.blocked + jitter[2]),
                outcome=float(np.clip(base.outcome + jitter[3], -1.0, 1.0)),
                novelty=clip(base.novelty + jitter[4]),
            )
        )
    return trace


def run_trace(machine_type, trace: list[Observation]) -> list[Output]:
    machine = machine_type()
    return [machine.step(observation) for observation in trace]


def run_to_intervention(machine_type, trace, step, intervention) -> Output:
    machine = machine_type()
    for observation in trace[:step]:
        machine.step(observation)
    return machine.step(trace[step], intervention)


def choose_force_workspace(baseline: Output) -> str:
    index = (WORKSPACE_INDEX[baseline.workspace] + 2) % len(WORKSPACES)
    return WORKSPACES[index]


def counterfactual_observation(observation: Observation, index: int) -> Observation:
    variants = (
        Observation(0.91, 0.67, 0.05, -0.35, 0.42),
        Observation(0.18, 0.83, 0.72, -0.51, 0.36),
        Observation(0.58, 0.12, 0.88, 0.22, 0.55),
    )
    replacement = variants[index % len(variants)]
    # Keep no exact trace lookup available while preserving bounded inputs.
    return Observation(*[
        clip(value) if field != "outcome" else float(np.clip(value, -1.0, 1.0))
        for field, value in zip(asdict(observation), asdict(replacement).values())
    ])


def run_seed(seed: int, steps: int = 96) -> dict:
    trace = generate_trace(seed, steps)
    ordinary = {machine.__name__: run_trace(machine, trace) for machine in REALIZATIONS}
    reference = ordinary[SymbolicEgo.__name__]
    replay = ObservationalReplay(reference)

    ordinary_realization_matches = []
    replay_ordinary_matches = []
    for step in range(steps):
        ordinary_realization_matches.extend(
            outputs_equal(reference[step], ordinary[machine.__name__][step])
            for machine in REALIZATIONS[1:]
        )
        replay_ordinary_matches.append(output_pair_equal(reference[step], replay.output_at(step)))

    intervention_rows = []
    intervention_steps = range(8, steps, 4)
    for step in intervention_steps:
        baseline = reference[step]
        interventions = {
            "forced_workspace": {"force_workspace": choose_force_workspace(baseline)},
            "memory_erasure": {"erase_memory": True},
            "valence_flip": {"flip_valence": True},
            "workspace_hold": {"hold_workspace": True},
        }
        for name, intervention in interventions.items():
            results = {
                machine.__name__: run_to_intervention(machine, trace, step, intervention)
                for machine in REALIZATIONS
            }
            changed = not output_pair_equal(results[SymbolicEgo.__name__], baseline)
            intervention_rows.append(
                {
                    "name": name,
                    "changed_reference_output": changed,
                    "realization_agreement": all(
                        outputs_equal(results[SymbolicEgo.__name__], results[machine.__name__])
                        for machine in REALIZATIONS[1:]
                    ),
                    "replay_agreement": output_pair_equal(
                        results[SymbolicEgo.__name__], replay.output_at(step, intervention)
                    ),
                    "report_control_changed_together": (
                        results[SymbolicEgo.__name__].action != baseline.action
                        and results[SymbolicEgo.__name__].report != baseline.report
                    ),
                }
            )

    counterfactual_rows = []
    for counter_index, step in enumerate(range(11, steps, 5)):
        changed_trace = list(trace)
        changed_trace[step] = counterfactual_observation(trace[step], counter_index)
        results = {
            machine.__name__: run_to_intervention(machine, changed_trace, step, {})
            for machine in REALIZATIONS
        }
        counterfactual_rows.append(
            {
                "realization_agreement": all(
                    outputs_equal(results[SymbolicEgo.__name__], results[machine.__name__])
                    for machine in REALIZATIONS[1:]
                ),
                "replay_agreement": output_pair_equal(
                    results[SymbolicEgo.__name__], replay.output_at(step)
                ),
                "changed_from_observed": not output_pair_equal(
                    results[SymbolicEgo.__name__], reference[step]
                ),
            }
        )

    return {
        "seed": seed,
        "steps": steps,
        "ordinary_realization_agreement": float(np.mean(ordinary_realization_matches)),
        "replay_ordinary_agreement": float(np.mean(replay_ordinary_matches)),
        "interventions": intervention_rows,
        "counterfactuals": counterfactual_rows,
    }


def summarize(rows: list[dict]) -> dict:
    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    all_interventions = [item for row in rows for item in row["interventions"]]
    all_counterfactuals = [item for row in rows for item in row["counterfactuals"]]
    by_intervention = defaultdict(list)
    for item in all_interventions:
        by_intervention[item["name"]].append(item)

    intervention_summary = {}
    for name, items in sorted(by_intervention.items()):
        eligible = [item for item in items if item["changed_reference_output"]]
        intervention_summary[name] = {
            "trials": len(items),
            "causally_effective_rate": float(np.mean([item["changed_reference_output"] for item in items])),
            "realization_agreement": float(np.mean([item["realization_agreement"] for item in items])),
            "replay_agreement_on_effective_trials": (
                float(np.mean([item["replay_agreement"] for item in eligible])) if eligible else None
            ),
            "joint_report_control_change_on_effective_trials": (
                float(np.mean([item["report_control_changed_together"] for item in eligible]))
                if eligible
                else None
            ),
        }

    changed_counterfactuals = [item for item in all_counterfactuals if item["changed_from_observed"]]
    return {
        "seeds": [row["seed"] for row in rows],
        "realizations": [machine.__name__ for machine in REALIZATIONS],
        "ordinary_realization_agreement": stats([row["ordinary_realization_agreement"] for row in rows]),
        "replay_ordinary_agreement": stats([row["replay_ordinary_agreement"] for row in rows]),
        "interventions": intervention_summary,
        "all_intervention_realization_agreement": float(
            np.mean([item["realization_agreement"] for item in all_interventions])
        ),
        "counterfactual_realization_agreement": float(
            np.mean([item["realization_agreement"] for item in all_counterfactuals])
        ),
        "replay_counterfactual_agreement_when_output_changes": (
            float(np.mean([item["replay_agreement"] for item in changed_counterfactuals]))
            if changed_counterfactuals
            else None
        ),
        "changed_counterfactual_trials": len(changed_counterfactuals),
        "total_counterfactual_trials": len(all_counterfactuals),
        "interpretation": {
            "supported": (
                "Equivalent causal transition structures preserve ordinary trajectories and intervention "
                "profiles across three software representations, while an observational replay can match "
                "seen outputs without preserving those counterfactual responses."
            ),
            "not_supported": (
                "Phenomenal consciousness, physical-substrate invariance, computational sufficiency for "
                "experience, or the claim that arbitrary output-equivalent programs share mental states."
            ),
        },
    }


def plot_summary(summary: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].bar(
        ("ordinary", "intervention", "counterfactual"),
        (
            summary["ordinary_realization_agreement"]["mean"],
            summary["all_intervention_realization_agreement"],
            summary["counterfactual_realization_agreement"],
        ),
        color=("#247ba0", "#70c1b3", "#f4a261"),
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Agreement")
    axes[0].set_title("Causal realizations")
    axes[0].grid(axis="y", alpha=0.2)

    replay_counterfactual = summary["replay_counterfactual_agreement_when_output_changes"]
    effective = [
        values["replay_agreement_on_effective_trials"]
        for values in summary["interventions"].values()
        if values["replay_agreement_on_effective_trials"] is not None
    ]
    axes[1].bar(
        ("observed", "intervened", "novel input"),
        (
            summary["replay_ordinary_agreement"]["mean"],
            float(np.mean(effective)),
            replay_counterfactual,
        ),
        color=("#247ba0", "#e76f51", "#f4a261"),
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Action/report agreement")
    axes[1].set_title("Observational replay control")
    axes[1].grid(axis="y", alpha=0.2)

    names = list(summary["interventions"])
    effects = [summary["interventions"][name]["causally_effective_rate"] for name in names]
    axes[2].bar([name.replace("_", "\n") for name in names], effects, color="#e76f51")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Output-change rate")
    axes[2].set_title("Internal intervention potency")
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Computational Organizational Invariance Lab", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = OUT / "computational_invariance_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def print_summary(summary: dict):
    print("Computational organizational invariance lab")
    print(f"ordinary realization agreement: {summary['ordinary_realization_agreement']['mean']:.3f}")
    print(f"intervention realization agreement: {summary['all_intervention_realization_agreement']:.3f}")
    print(f"counterfactual realization agreement: {summary['counterfactual_realization_agreement']:.3f}")
    print(f"replay ordinary agreement: {summary['replay_ordinary_agreement']['mean']:.3f}")
    print(
        "replay changed-counterfactual agreement: "
        f"{summary['replay_counterfactual_agreement_when_output_changes']:.3f}"
    )
    print("\nInterventions")
    for name, values in summary["interventions"].items():
        replay = values["replay_agreement_on_effective_trials"]
        replay_text = "n/a" if replay is None else f"{replay:.3f}"
        print(
            f"  {name:18s} effect={values['causally_effective_rate']:.3f} "
            f"realizations={values['realization_agreement']:.3f} replay={replay_text}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [127 + 101 * index for index in range(args.seeds)]
    rows = [run_seed(seed, args.steps) for seed in seeds]
    summary = summarize(rows)
    metrics_path = OUT / "computational_invariance_metrics.json"
    metrics_path.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2) + "\n")
    figure_path = plot_summary(summary)
    print_summary(summary)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
