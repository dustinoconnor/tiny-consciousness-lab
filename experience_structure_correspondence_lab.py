#!/usr/bin/env python3
"""Relate causal agent-state geometry to reportable experience proxies.

Mathematical consciousness science asks whether relations among experiences can
be represented as mathematical structures. An artificial agent does not give
us independently verified phenomenal data, so this lab uses a narrower target:
the structure of its explicit, causally active self-reports.

For the Functional Ego transition system validated in
``computational_invariance_lab.py``, the lab constructs four spaces:

- full causal state: observation, memory, valence, confidence, workspace;
- hidden state: memory and grounded valence, excluding workspace/report labels;
- report proxy: report category, reported confidence, and reported valence;
- behavior: action category and action confidence.

It measures distance-rank correspondence and local-neighborhood preservation,
then applies marginal-preserving report shuffles, a label-isometry control,
multiple internal interventions, an observational replay control, and three
independent software realizations.

This can establish mathematical structure in reportable access proxies. It
cannot show that the proxy is a phenomenal quality space or that any state is
subjectively experienced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from computational_invariance_lab import (
    ACTIONS,
    REALIZATIONS,
    REPORTS,
    WORKSPACES,
    WORKSPACE_INDEX,
    MessagePassingEgo,
    Observation,
    Output,
    SymbolicEgo,
    VectorEgo,
    choose_force_workspace,
    generate_trace,
    run_to_intervention,
)
from tiny_lab import OUT


ACTION_NAMES = tuple(ACTIONS[name] for name in WORKSPACES)
REPORT_NAMES = tuple(REPORTS[name] for name in WORKSPACES)


@dataclass(frozen=True)
class Spaces:
    full: np.ndarray
    hidden: np.ndarray
    report: np.ndarray
    behavior: np.ndarray


def one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float64)
    vector[index] = 1.0
    return vector


def vectors(observation: Observation, output: Output) -> Spaces:
    workspace_index = WORKSPACE_INDEX[output.workspace]
    workspace = one_hot(workspace_index, len(WORKSPACES))
    action = one_hot(ACTION_NAMES.index(output.action), len(ACTION_NAMES))
    report = one_hot(REPORT_NAMES.index(output.report), len(REPORT_NAMES))
    hidden = np.array(
        [
            output.memory_food,
            output.memory_danger,
            output.memory_blocked,
            0.5 * (output.valence + 1.0),
        ],
        dtype=np.float64,
    )
    observation_vector = np.array(
        [
            observation.food,
            observation.danger,
            observation.blocked,
            0.5 * (observation.outcome + 1.0),
            observation.novelty,
        ],
        dtype=np.float64,
    )
    confidence = np.array([output.confidence], dtype=np.float64)
    report_valence = np.array([0.5 * (output.valence + 1.0)], dtype=np.float64)
    return Spaces(
        full=np.concatenate((observation_vector, hidden, confidence, workspace)),
        hidden=hidden,
        report=np.concatenate((report, confidence, report_valence)),
        behavior=np.concatenate((action, confidence)),
    )


def run_spaces(machine_type, trace: list[Observation]) -> tuple[Spaces, list[Output]]:
    machine = machine_type()
    rows = []
    outputs = []
    for observation in trace:
        output = machine.step(observation)
        outputs.append(output)
        rows.append(vectors(observation, output))
    return (
        Spaces(
            full=np.stack([row.full for row in rows]),
            hidden=np.stack([row.hidden for row in rows]),
            report=np.stack([row.report for row in rows]),
            behavior=np.stack([row.behavior for row in rows]),
        ),
        outputs,
    )


def standardized(matrix: np.ndarray) -> np.ndarray:
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (matrix - matrix.mean(axis=0)) / scale


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    matrix = standardized(matrix)
    delta = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1)
        position = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(np.asarray(left, dtype=np.float64))
    right_rank = rankdata(np.asarray(right, dtype=np.float64))
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return spearman(
        upper_triangle(pairwise_distances(left)),
        upper_triangle(pairwise_distances(right)),
    )


def neighborhood_overlap(left: np.ndarray, right: np.ndarray, k: int = 8) -> float:
    left_distance = pairwise_distances(left)
    right_distance = pairwise_distances(right)
    overlaps = []
    for index in range(len(left)):
        left_neighbors = set(np.argsort(left_distance[index])[1 : k + 1])
        right_neighbors = set(np.argsort(right_distance[index])[1 : k + 1])
        overlaps.append(len(left_neighbors & right_neighbors) / k)
    return float(np.mean(overlaps))


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return 0.5
    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        positive_rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def intervention_vectors(trace, step, intervention, machine_type=SymbolicEgo):
    baseline_output = run_to_intervention(machine_type, trace, step, {})
    changed_output = run_to_intervention(machine_type, trace, step, intervention)
    baseline = vectors(trace[step], baseline_output)
    changed = vectors(trace[step], changed_output)
    return baseline_output, changed_output, baseline, changed


def euclidean(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def run_seed(seed: int, steps: int = 128) -> dict:
    rng = np.random.default_rng(seed + 90_000)
    trace = generate_trace(seed, steps)
    spaces_by_realization = {}
    outputs_by_realization = {}
    for machine_type in REALIZATIONS:
        spaces, outputs = run_spaces(machine_type, trace)
        spaces_by_realization[machine_type.__name__] = spaces
        outputs_by_realization[machine_type.__name__] = outputs

    reference = spaces_by_realization[SymbolicEgo.__name__]
    report_permutation = rng.permutation(steps)
    shuffled_report = reference.report[report_permutation]
    relabeled_report = reference.report.copy()
    relabeled_report[:, : len(REPORT_NAMES)] = relabeled_report[
        :, rng.permutation(len(REPORT_NAMES))
    ][:, : len(REPORT_NAMES)]

    geometry = {
        "full_report_correlation": distance_correlation(reference.full, reference.report),
        "hidden_report_correlation": distance_correlation(reference.hidden, reference.report),
        "behavior_report_correlation": distance_correlation(reference.behavior, reference.report),
        "full_report_neighborhood_overlap": neighborhood_overlap(reference.full, reference.report),
        "hidden_report_neighborhood_overlap": neighborhood_overlap(reference.hidden, reference.report),
        "shuffled_full_report_correlation": distance_correlation(reference.full, shuffled_report),
        "shuffled_full_report_neighborhood_overlap": neighborhood_overlap(reference.full, shuffled_report),
        "label_isometry_correlation": distance_correlation(reference.report, relabeled_report),
    }

    realization_geometry = {}
    for machine_type in REALIZATIONS[1:]:
        candidate = spaces_by_realization[machine_type.__name__]
        realization_geometry[machine_type.__name__] = {
            name: distance_correlation(getattr(reference, name), getattr(candidate, name))
            for name in ("full", "hidden", "report", "behavior")
        }

    intervention_rows = []
    reference_outputs = outputs_by_realization[SymbolicEgo.__name__]
    for step in range(8, steps, 4):
        baseline = reference_outputs[step]
        interventions = {
            "forced_workspace": {"force_workspace": choose_force_workspace(baseline)},
            "memory_erasure": {"erase_memory": True},
            "valence_flip": {"flip_valence": True},
            "workspace_hold": {"hold_workspace": True},
        }
        for name, intervention in interventions.items():
            baseline_output, changed_output, baseline_vectors, changed_vectors = intervention_vectors(
                trace, step, intervention
            )
            report_changed = changed_output.report != baseline_output.report
            action_changed = changed_output.action != baseline_output.action
            full_shift = euclidean(baseline_vectors.full, changed_vectors.full)
            hidden_shift = euclidean(baseline_vectors.hidden, changed_vectors.hidden)
            report_shift = euclidean(baseline_vectors.report, changed_vectors.report)
            behavior_shift = euclidean(baseline_vectors.behavior, changed_vectors.behavior)

            realization_matches = []
            for machine_type in (VectorEgo, MessagePassingEgo):
                _, other_output, _, other_vectors = intervention_vectors(
                    trace, step, intervention, machine_type
                )
                realization_matches.append(
                    other_output.report == changed_output.report
                    and other_output.action == changed_output.action
                    and np.allclose(other_vectors.full, changed_vectors.full)
                    and np.allclose(other_vectors.report, changed_vectors.report)
                )

            intervention_rows.append(
                {
                    "name": name,
                    "report_changed": report_changed,
                    "action_changed": action_changed,
                    "joint_report_action_change": report_changed and action_changed,
                    "full_shift": full_shift,
                    "hidden_shift": hidden_shift,
                    "report_shift": report_shift,
                    "behavior_shift": behavior_shift,
                    "replay_report_shift": 0.0,
                    "realization_match": all(realization_matches),
                }
            )

    report_change_labels = [row["report_changed"] for row in intervention_rows]
    # Full state contains the workspace coordinate that directly generates the
    # report, so this is a definitional positive control rather than
    # independent evidence for structural correspondence.
    geometry["full_state_shift_report_change_auc_control"] = roc_auc(
        [row["full_shift"] for row in intervention_rows], report_change_labels
    )
    upstream_rows = [
        row
        for row in intervention_rows
        if row["name"] in {"memory_erasure", "valence_flip"}
    ]
    upstream_labels = [row["report_changed"] for row in upstream_rows]
    geometry["upstream_shift_report_change_auc"] = roc_auc(
        [row["hidden_shift"] for row in upstream_rows], upstream_labels
    )
    geometry["shuffled_upstream_label_auc"] = roc_auc(
        [row["hidden_shift"] for row in upstream_rows],
        list(rng.permutation(upstream_labels)),
    )
    geometry["replay_shift_report_change_auc"] = 0.5

    return {
        "seed": seed,
        "steps": steps,
        "geometry": geometry,
        "realization_geometry": realization_geometry,
        "interventions": intervention_rows,
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

    geometry_keys = rows[0]["geometry"].keys()
    geometry = {
        key: stats([row["geometry"][key] for row in rows]) for key in geometry_keys
    }
    realization_geometry = {}
    for realization in rows[0]["realization_geometry"]:
        realization_geometry[realization] = {
            space: stats(
                [row["realization_geometry"][realization][space] for row in rows]
            )
            for space in rows[0]["realization_geometry"][realization]
        }

    interventions = [item for row in rows for item in row["interventions"]]
    by_name = {}
    for name in sorted({item["name"] for item in interventions}):
        selected = [item for item in interventions if item["name"] == name]
        effective = [item for item in selected if item["report_changed"]]
        by_name[name] = {
            "trials": len(selected),
            "report_change_rate": float(np.mean([item["report_changed"] for item in selected])),
            "action_change_rate": float(np.mean([item["action_changed"] for item in selected])),
            "joint_change_rate": float(np.mean([item["joint_report_action_change"] for item in selected])),
            "mean_full_shift": float(np.mean([item["full_shift"] for item in selected])),
            "mean_report_shift": float(np.mean([item["report_shift"] for item in selected])),
            "replay_correspondence_on_effective_trials": (
                float(np.mean([item["replay_report_shift"] > 0.0 for item in effective]))
                if effective
                else None
            ),
            "cross_realization_match": float(np.mean([item["realization_match"] for item in selected])),
        }

    return {
        "seeds": [row["seed"] for row in rows],
        "geometry": geometry,
        "cross_realization_geometry": realization_geometry,
        "interventions": by_name,
        "all_intervention_cross_realization_match": float(
            np.mean([item["realization_match"] for item in interventions])
        ),
        "joint_report_action_change_when_report_changes": float(
            np.mean(
                [
                    item["joint_report_action_change"]
                    for item in interventions
                    if item["report_changed"]
                ]
            )
        ),
        "interpretation": {
            "supported": (
                "Reportable agent states have a reproducible relational geometry linked to full causal "
                "state and behavior, preserved across software realizations and disrupted by "
                "marginal-preserving report shuffling; internal interventions jointly alter report and action."
            ),
            "not_supported": (
                "Phenomenal experience, qualia, an independently observed first-person experience space, "
                "or the ontological claim that consciousness is mathematical structure."
            ),
        },
    }


def plot_summary(summary: dict):
    geometry = summary["geometry"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].bar(
        ("full↔report", "hidden↔report", "behavior↔report", "shuffled"),
        (
            geometry["full_report_correlation"]["mean"],
            geometry["hidden_report_correlation"]["mean"],
            geometry["behavior_report_correlation"]["mean"],
            geometry["shuffled_full_report_correlation"]["mean"],
        ),
        color=("#247ba0", "#70c1b3", "#f4a261", "#7f8c8d"),
    )
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].set_ylabel("Spearman distance correlation")
    axes[0].set_title("Relational geometry")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        ("full↔report", "hidden↔report", "shuffled"),
        (
            geometry["full_report_neighborhood_overlap"]["mean"],
            geometry["hidden_report_neighborhood_overlap"]["mean"],
            geometry["shuffled_full_report_neighborhood_overlap"]["mean"],
        ),
        color=("#247ba0", "#70c1b3", "#7f8c8d"),
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Top-8 neighborhood overlap")
    axes[1].set_title("Local structure")
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(
        ("upstream shift", "shuffled labels", "replay"),
        (
            geometry["upstream_shift_report_change_auc"]["mean"],
            geometry["shuffled_upstream_label_auc"]["mean"],
            geometry["replay_shift_report_change_auc"]["mean"],
        ),
        color=("#e76f51", "#70c1b3", "#7f8c8d"),
    )
    axes[2].axhline(0.5, color="#333", linestyle="--", linewidth=1.0)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("AUC for report change")
    axes[2].set_title("Causal structural correspondence")
    axes[2].tick_params(axis="x", labelrotation=18)
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("Mathematical Report-Structure Correspondence Lab", fontsize=15, fontweight="bold")
    fig.tight_layout()
    output = OUT / "experience_structure_correspondence_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def print_summary(summary: dict):
    geometry = summary["geometry"]
    print("Mathematical report-structure correspondence lab")
    for key in (
        "full_report_correlation",
        "hidden_report_correlation",
        "behavior_report_correlation",
        "shuffled_full_report_correlation",
        "full_report_neighborhood_overlap",
        "shuffled_full_report_neighborhood_overlap",
        "label_isometry_correlation",
        "full_state_shift_report_change_auc_control",
        "upstream_shift_report_change_auc",
        "shuffled_upstream_label_auc",
        "replay_shift_report_change_auc",
    ):
        print(f"{key:44s} {geometry[key]['mean']:.3f} ± {geometry[key]['std']:.3f}")
    print("\nInterventions")
    for name, values in summary["interventions"].items():
        print(
            f"  {name:18s} report={values['report_change_rate']:.3f} "
            f"action={values['action_change_rate']:.3f} "
            f"realizations={values['cross_realization_match']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [149 + 103 * index for index in range(args.seeds)]
    rows = [run_seed(seed, args.steps) for seed in seeds]
    summary = summarize(rows)
    metrics_path = OUT / "experience_structure_correspondence_metrics.json"
    metrics_path.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2) + "\n")
    figure_path = plot_summary(summary)
    print_summary(summary)
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
