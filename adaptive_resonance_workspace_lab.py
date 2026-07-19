#!/usr/bin/env python3
"""Adaptive resonance, stable learning, and report/control routing.

This lab translates the computational core of Grossberg's Adaptive Resonance
Theory (ART) into a small workspace experiment. Bottom-up sensory evidence
selects a learned category; its top-down template must pass a vigilance match
before a resonant workspace packet can stabilize learning and route both an
action and a symbolic report. Mismatch resets the candidate and continues the
category search.

The non-stationary curriculum first teaches familiar food and obstacle
categories, then introduces overlapping novel categories, and finally tests
whether the familiar categories were retained. Controls lesion mismatch reset,
stable resonant learning, or the entire resonant workspace. A forced-category
intervention tests whether one internal category causally changes report and
control together.

This is a functional software test of ART-like mechanisms. It does not model
the full biological ART architecture and cannot establish phenomenal
consciousness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT


ACTIONS = ("approach", "avoid", "detour_left", "detour_right")


@dataclass(frozen=True)
class Concept:
    name: str
    prototype: np.ndarray
    action: str
    valence: float


@dataclass(frozen=True)
class WorkspacePacket:
    category: int
    match: float
    action: str
    report: str
    valence: float
    resonant: bool = True


CONCEPTS = (
    Concept("ripe_food", np.array([0.86, 0.18, 0.82, 0.14, 0.24, 0.30]), "approach", 1.0),
    Concept("toxic_food", np.array([0.82, 0.22, 0.20, 0.84, 0.24, 0.30]), "avoid", -1.0),
    Concept("left_opening", np.array([0.20, 0.88, 0.30, 0.18, 0.84, 0.20]), "detour_left", -0.2),
    Concept("right_opening", np.array([0.24, 0.86, 0.30, 0.18, 0.20, 0.86]), "detour_right", -0.2),
    # These overlap familiar categories in several dimensions but require the
    # learner to form distinct categories during the second curriculum phase.
    Concept("luminous_food", np.array([0.64, 0.12, 0.90, 0.22, 0.54, 0.22]), "approach", 0.8),
    Concept("deceptive_hazard", np.array([0.66, 0.16, 0.22, 0.86, 0.54, 0.22]), "avoid", -0.9),
)

FAMILIAR = CONCEPTS[:4]
NOVEL = CONCEPTS[4:]


def complement_code(vector: np.ndarray) -> np.ndarray:
    vector = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)
    return np.concatenate((vector, 1.0 - vector))


def sample_concept(concept: Concept, rng: np.random.Generator, noise: float) -> np.ndarray:
    return np.clip(concept.prototype + rng.normal(0.0, noise, concept.prototype.shape), 0.0, 1.0)


class AdaptiveResonanceWorkspace:
    """Compact fuzzy-ART category memory with an explicit workspace packet."""

    def __init__(
        self,
        vigilance: float = 0.82,
        choice_alpha: float = 0.001,
        learning_rate: float = 0.55,
        use_reset: bool = True,
        stable_learning: bool = True,
        workspace_enabled: bool = True,
    ):
        self.vigilance = vigilance
        self.choice_alpha = choice_alpha
        self.learning_rate = learning_rate
        self.use_reset = use_reset
        self.stable_learning = stable_learning
        self.workspace_enabled = workspace_enabled
        self.templates: list[np.ndarray] = []
        self.action_counts: list[np.ndarray] = []
        self.report_counts: list[dict[str, float]] = []
        self.valence_sum: list[float] = []
        self.visits: list[float] = []
        self.reset_count = 0
        self.resonance_count = 0

    @staticmethod
    def _match(pattern: np.ndarray, template: np.ndarray) -> float:
        return float(np.minimum(pattern, template).sum() / max(pattern.sum(), 1e-12))

    def _choice(self, pattern: np.ndarray, template: np.ndarray) -> float:
        return float(
            np.minimum(pattern, template).sum()
            / (self.choice_alpha + template.sum())
        )

    def _search(self, pattern: np.ndarray) -> tuple[int | None, float]:
        if not self.templates:
            return None, 0.0
        available = set(range(len(self.templates)))
        while available:
            category = max(available, key=lambda index: self._choice(pattern, self.templates[index]))
            match = self._match(pattern, self.templates[category])
            if match >= self.vigilance or not self.use_reset:
                return category, match
            available.remove(category)
            self.reset_count += 1
        return None, 0.0

    def _new_category(self, pattern: np.ndarray) -> int:
        category = len(self.templates)
        self.templates.append(pattern.copy())
        self.action_counts.append(np.full(len(ACTIONS), 0.05, dtype=np.float64))
        self.report_counts.append({})
        self.valence_sum.append(0.0)
        self.visits.append(0.0)
        return category

    def learn(self, observation: np.ndarray, action: str, report: str, valence: float) -> WorkspacePacket | None:
        if not self.workspace_enabled:
            return None
        pattern = complement_code(observation)
        category, match = self._search(pattern)
        if category is None:
            category = self._new_category(pattern)
            match = 1.0

        if self.stable_learning:
            beta = self.learning_rate
            self.templates[category] = (
                beta * np.minimum(pattern, self.templates[category])
                + (1.0 - beta) * self.templates[category]
            )
            self.action_counts[category][ACTIONS.index(action)] += 1.0
            self.report_counts[category][report] = self.report_counts[category].get(report, 0.0) + 1.0
            self.valence_sum[category] += valence
            self.visits[category] += 1.0
        else:
            # The stability lesion retains selection but replaces its memory
            # with the latest sample and consequence, maximizing plasticity at
            # the expense of category retention.
            self.templates[category] = pattern.copy()
            self.action_counts[category] = np.full(len(ACTIONS), 0.05, dtype=np.float64)
            self.action_counts[category][ACTIONS.index(action)] = 1.0
            self.report_counts[category] = {report: 1.0}
            self.valence_sum[category] = valence
            self.visits[category] = 1.0

        self.resonance_count += 1
        return self._packet(category, match)

    def infer(
        self,
        observation: np.ndarray,
        forced_category: int | None = None,
    ) -> WorkspacePacket | None:
        if not self.workspace_enabled or not self.templates:
            return None
        pattern = complement_code(observation)
        if forced_category is None:
            category, match = self._search(pattern)
            if category is None:
                return None
        else:
            category = int(forced_category)
            match = self._match(pattern, self.templates[category])
        self.resonance_count += 1
        return self._packet(category, match)

    def _packet(self, category: int, match: float) -> WorkspacePacket:
        action = ACTIONS[int(np.argmax(self.action_counts[category]))]
        reports = self.report_counts[category]
        report = max(reports, key=reports.get) if reports else "unclassified"
        visits = max(self.visits[category], 1.0)
        return WorkspacePacket(
            category=category,
            match=match,
            action=action,
            report=report,
            valence=float(self.valence_sum[category] / visits),
        )


def curriculum(agent: AdaptiveResonanceWorkspace, rng: np.random.Generator) -> None:
    phases = ((FAMILIAR, 36, 0.035), (NOVEL, 12, 0.035))
    for concepts, repeats, noise in phases:
        rows = [(concept, sample_concept(concept, rng, noise)) for concept in concepts for _ in range(repeats)]
        rng.shuffle(rows)
        for concept, observation in rows:
            agent.learn(observation, concept.action, concept.name, concept.valence)


def evaluate_concepts(
    agent: AdaptiveResonanceWorkspace,
    concepts: tuple[Concept, ...],
    rng: np.random.Generator,
    samples: int = 80,
    noise: float = 0.045,
) -> dict[str, float]:
    action_correct = 0
    report_correct = 0
    promoted = 0
    aligned = 0
    total = samples * len(concepts)
    for concept in concepts:
        for _ in range(samples):
            packet = agent.infer(sample_concept(concept, rng, noise))
            if packet is None:
                continue
            promoted += 1
            action_correct += int(packet.action == concept.action)
            report_correct += int(packet.report == concept.name)
            # Both outputs are aligned when they describe the same learned
            # concept/action association, independent of external correctness.
            report_concept = next((item for item in CONCEPTS if item.name == packet.report), None)
            aligned += int(report_concept is not None and report_concept.action == packet.action)
    return {
        "action_accuracy": action_correct / total,
        "report_accuracy": report_correct / total,
        "promotion_rate": promoted / total,
        "report_control_alignment": aligned / max(promoted, 1),
    }


def unknown_rejection(agent: AdaptiveResonanceWorkspace, rng: np.random.Generator, samples: int = 200) -> float:
    rejected = 0
    for _ in range(samples):
        # Mid-range feature conjunctions were absent from the curriculum and
        # should fail a sufficiently vigilant top-down template match.
        observation = np.clip(rng.normal(0.50, 0.035, 6), 0.0, 1.0)
        rejected += int(agent.infer(observation) is None)
    return rejected / samples


def forced_false_resonance(agent: AdaptiveResonanceWorkspace, rng: np.random.Generator, samples: int = 160) -> dict:
    eligible = 0
    action_switches = 0
    report_switches = 0
    joint_switches = 0
    false_match = []
    for _ in range(samples):
        concept = FAMILIAR[int(rng.integers(0, len(FAMILIAR)))]
        observation = sample_concept(concept, rng, 0.03)
        baseline = agent.infer(observation)
        if baseline is None:
            continue
        candidates = [
            index
            for index in range(len(agent.templates))
            if agent._packet(index, 0.0).action != baseline.action
        ]
        if not candidates:
            continue
        forced_category = min(
            candidates,
            key=lambda index: agent._match(complement_code(observation), agent.templates[index]),
        )
        forced = agent.infer(observation, forced_category=forced_category)
        eligible += 1
        changed_action = forced.action != baseline.action
        changed_report = forced.report != baseline.report
        action_switches += int(changed_action)
        report_switches += int(changed_report)
        joint_switches += int(changed_action and changed_report)
        false_match.append(forced.match)
    return {
        "eligible_trials": eligible,
        "action_switch_rate": action_switches / max(eligible, 1),
        "report_switch_rate": report_switches / max(eligible, 1),
        "joint_report_control_switch_rate": joint_switches / max(eligible, 1),
        "forced_category_mean_sensory_match": float(np.mean(false_match)) if false_match else 0.0,
    }


def run_condition(seed: int, condition: str) -> dict:
    rng = np.random.default_rng(seed)
    settings = {
        "full_art": {},
        "reset_lesion": {"use_reset": False},
        "stability_lesion": {"stable_learning": False},
        "resonance_lesion": {"workspace_enabled": False},
    }[condition]
    agent = AdaptiveResonanceWorkspace(**settings)
    curriculum(agent, rng)
    return {
        "condition": condition,
        "seed": seed,
        "categories": len(agent.templates),
        "resets": agent.reset_count,
        "resonances": agent.resonance_count,
        "familiar_retention": evaluate_concepts(agent, FAMILIAR, rng),
        "novel_adaptation": evaluate_concepts(agent, NOVEL, rng),
        "unknown_rejection": unknown_rejection(agent, rng),
        "forced_false_resonance": (
            forced_false_resonance(agent, rng) if condition == "full_art" else None
        ),
    }


def summarize(rows: list[dict]) -> dict:
    conditions = sorted({row["condition"] for row in rows})
    summary = {"seeds": sorted({row["seed"] for row in rows}), "conditions": {}}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]

        def stats(values):
            values = np.asarray(values, dtype=np.float64)
            return {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }

        result = {
            "categories": stats([row["categories"] for row in selected]),
            "resets": stats([row["resets"] for row in selected]),
            "unknown_rejection": stats([row["unknown_rejection"] for row in selected]),
        }
        for phase in ("familiar_retention", "novel_adaptation"):
            result[phase] = {
                metric: stats([row[phase][metric] for row in selected])
                for metric in selected[0][phase]
            }
        if condition == "full_art":
            result["forced_false_resonance"] = {
                metric: stats([row["forced_false_resonance"][metric] for row in selected])
                for metric in selected[0]["forced_false_resonance"]
            }
        summary["conditions"][condition] = result

    full = summary["conditions"]["full_art"]
    reset = summary["conditions"]["reset_lesion"]
    stability = summary["conditions"]["stability_lesion"]
    summary["causal_contrasts"] = {
        "reset_contribution_to_novel_action_accuracy": (
            full["novel_adaptation"]["action_accuracy"]["mean"]
            - reset["novel_adaptation"]["action_accuracy"]["mean"]
        ),
        "stable_resonance_contribution_to_familiar_category_report_retention": (
            full["familiar_retention"]["report_accuracy"]["mean"]
            - stability["familiar_retention"]["report_accuracy"]["mean"]
        ),
        "forced_false_joint_switch_rate": full["forced_false_resonance"][
            "joint_report_control_switch_rate"
        ]["mean"],
    }
    summary["interpretation"] = {
        "supported": (
            "ART-like match, vigilance, reset, and stable category learning preserve old action/report "
            "associations while admitting novel ones; an internal category intervention jointly redirects "
            "report and control through the shared workspace packet."
        ),
        "not_supported": (
            "Phenomenal consciousness, qualia, a complete biological ART implementation, or the claim "
            "that every computational resonance is conscious."
        ),
    }
    return summary


def plot_summary(summary: dict):
    conditions = ("full_art", "reset_lesion", "stability_lesion", "resonance_lesion")
    labels = ("full ART", "reset\nlesion", "stability\nlesion", "resonance\nlesion")
    colors = ("#247ba0", "#e76f51", "#f4a261", "#7f8c8d")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    familiar = [summary["conditions"][c]["familiar_retention"]["action_accuracy"]["mean"] for c in conditions]
    novel = [summary["conditions"][c]["novel_adaptation"]["action_accuracy"]["mean"] for c in conditions]
    x = np.arange(len(conditions))
    axes[0].bar(x - 0.18, familiar, 0.36, color="#247ba0", label="familiar retention")
    axes[0].bar(x + 0.18, novel, 0.36, color="#e76f51", label="novel adaptation")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Action accuracy")
    axes[0].set_title("Stable learning under change")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    report_retention = [
        summary["conditions"][c]["familiar_retention"]["report_accuracy"]["mean"]
        for c in conditions
    ]
    rejection = [summary["conditions"][c]["unknown_rejection"]["mean"] for c in conditions]
    axes[1].bar(x - 0.18, report_retention, 0.36, color="#247ba0", label="familiar report")
    axes[1].bar(x + 0.18, rejection, 0.36, color="#f4a261", label="unknown rejection")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Category stability and mismatch reset")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    forced = summary["conditions"]["full_art"]["forced_false_resonance"]
    intervention_values = [
        forced["action_switch_rate"]["mean"],
        forced["report_switch_rate"]["mean"],
        forced["joint_report_control_switch_rate"]["mean"],
    ]
    axes[2].bar(("action", "report", "joint"), intervention_values, color=("#e76f51", "#247ba0", "#70c1b3"))
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Switch rate")
    axes[2].set_title("Forced false-resonance intervention")
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Adaptive Resonance Workspace Lab", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = OUT / "adaptive_resonance_workspace_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def print_summary(summary: dict) -> None:
    print("Adaptive resonance workspace lab")
    print("condition            action familiar/novel  report retention  reject unknown  categories")
    for condition in ("full_art", "reset_lesion", "stability_lesion", "resonance_lesion"):
        row = summary["conditions"][condition]
        print(
            f"{condition:20s} "
            f"{row['familiar_retention']['action_accuracy']['mean']:6.3f}/"
            f"{row['novel_adaptation']['action_accuracy']['mean']:5.3f}  "
            f"{row['familiar_retention']['report_accuracy']['mean']:16.3f}  "
            f"{row['unknown_rejection']['mean']:14.3f}  "
            f"{row['categories']['mean']:10.2f}"
        )
    forced = summary["conditions"]["full_art"]["forced_false_resonance"]
    print("\nForced false resonance")
    print(f"  action switch: {forced['action_switch_rate']['mean']:.3f}")
    print(f"  report switch: {forced['report_switch_rate']['mean']:.3f}")
    print(f"  joint switch:  {forced['joint_report_control_switch_rate']['mean']:.3f}")
    print("\nCausal contrasts")
    for key, value in summary["causal_contrasts"].items():
        print(f"  {key}: {value:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [113 + 97 * index for index in range(args.seeds)]
    conditions = ("full_art", "reset_lesion", "stability_lesion", "resonance_lesion")
    rows = [run_condition(seed, condition) for seed in seeds for condition in conditions]
    summary = summarize(rows)

    metrics_path = OUT / "adaptive_resonance_workspace_metrics.json"
    metrics_path.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2) + "\n")
    figure_path = plot_summary(summary)
    print_summary(summary)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
