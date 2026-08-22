#!/usr/bin/env python3
"""Answer-blind PGNW arbitration among actionable typed causal hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from terrain_resource_memory import PassiveTerrainResourceMemory
from typed_causal_domain import TypedInteractionPool


FEATURE_ACTION = {"yellow": 1, "blue": 2}


@dataclass(frozen=True)
class CandidateRoute:
    feature: str
    distance: float
    memory_confidence: float
    safety_allowed: bool = True


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def candidate_score(
    pool,
    candidate,
    *,
    hazard_cost=0.25,
    deadline_remaining=240.0,
    conservative_speed=2.0,
    epistemic_weight=0.10,
    route_weight=0.05,
    metabolic_pool=None,
    hunger_urgency=0.0,
):
    """Return transparent negative-EFE terms without a correct-answer input."""
    if candidate.feature not in FEATURE_ACTION:
        raise ValueError("unsupported_arbitration_feature")
    if not candidate.safety_allowed:
        return {
            "eligible": False,
            "score": -math.inf,
            "reason": "safety_gate",
        }
    action = FEATURE_ACTION[candidate.feature]
    distance = max(0.0, float(candidate.distance))
    confidence = clamp(candidate.memory_confidence)
    deadline = max(1e-6, float(deadline_remaining))
    travel_seconds = distance / max(1e-6, float(conservative_speed))
    on_time_probability = clamp(1.0 - travel_seconds / deadline)
    suppression_probability = float(
        np.asarray(pool.posterior, dtype=np.float64)
        @ pool.likelihood_matrix[:, action]
    )
    information_gain = float(pool.expected_information_gain(action))
    normalized_information = information_gain / math.log2(
        len(pool.hypotheses)
    )
    pragmatic_value = (
        clamp(hazard_cost)
        * suppression_probability
        * confidence
        * on_time_probability
    )
    metabolic_relief_probability = (
        float(metabolic_pool.relief_probability(candidate.feature))
        if metabolic_pool is not None
        else 0.0
    )
    metabolic_value = (
        clamp(hunger_urgency)
        * metabolic_relief_probability
        * confidence
        * on_time_probability
    )
    epistemic_value = max(0.0, float(epistemic_weight)) * normalized_information
    route_cost = max(0.0, float(route_weight)) * clamp(
        travel_seconds / deadline
    )
    score = pragmatic_value + metabolic_value + epistemic_value - route_cost
    return {
        "eligible": True,
        "score": score,
        "reason": "candidate",
        "action_index": action,
        "suppression_probability": suppression_probability,
        "information_gain_bits": information_gain,
        "normalized_information": normalized_information,
        "travel_seconds": travel_seconds,
        "on_time_probability": on_time_probability,
        "pragmatic_value": pragmatic_value,
        "metabolic_relief_probability": metabolic_relief_probability,
        "metabolic_value": metabolic_value,
        "epistemic_value": epistemic_value,
        "route_cost": route_cost,
    }


def classify_candidate_record(record, mode="meaningful"):
    """Attach a report label without changing evidence, score, or authority."""
    if mode not in {"meaningful", "anonymous", "shuffled"}:
        raise ValueError("unsupported_language_classification_mode")
    protective = float(record.get("suppression_probability", 0.0))
    metabolic = float(record.get("metabolic_relief_probability", 0.0))
    causal_role = "antidote" if protective >= metabolic else "food"
    if mode == "anonymous":
        label = "role_1" if causal_role == "antidote" else "role_2"
    elif mode == "shuffled":
        label = "food" if causal_role == "antidote" else "antidote"
    else:
        label = causal_role
    return {**record, "causal_role": causal_role, "language_label": label}


def arbitrate(pool, candidates, classification_mode="meaningful", **score_options):
    """Select a candidate invariant to caller presentation order."""
    if not candidates:
        raise ValueError("arbitration_requires_candidates")
    if len({candidate.feature for candidate in candidates}) != len(candidates):
        raise ValueError("arbitration_requires_unique_features")
    records = []
    for candidate in candidates:
        metrics = candidate_score(pool, candidate, **score_options)
        records.append(classify_candidate_record(
            {"candidate": asdict(candidate), **metrics},
            mode=classification_mode,
        ))
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        return {
            "selected_feature": "none",
            "selected_score": -math.inf,
            "records": sorted(
                records, key=lambda record: record["candidate"]["feature"]
            ),
        }
    selected = max(
        eligible,
        key=lambda record: (
            record["score"],
            record["suppression_probability"],
            -record["travel_seconds"],
            record["candidate"]["feature"],
        ),
    )
    return {
        "selected_feature": selected["candidate"]["feature"],
        "selected_score": selected["score"],
        "records": sorted(
            records, key=lambda record: record["candidate"]["feature"]
        ),
    }


def pool_with_posterior(values):
    pool = TypedInteractionPool()
    posterior = np.asarray(values, dtype=np.float64)
    if posterior.shape != (len(pool.hypotheses),):
        raise ValueError("posterior_shape_mismatch")
    if np.any(posterior < 0.0) or not math.isclose(
        float(np.sum(posterior)), 1.0, abs_tol=1e-9
    ):
        raise ValueError("invalid_arbitration_posterior")
    pool.posterior = posterior.copy()
    return pool


def counterbalanced_matrix():
    cases = []
    posteriors = {
        "yellow": (0.70, 0.05, 0.15, 0.10),
        "blue": (0.05, 0.70, 0.15, 0.10),
    }
    for dominant, posterior in posteriors.items():
        other = "blue" if dominant == "yellow" else "yellow"
        for dominant_distance, other_distance, distance_order in (
            (36.0, 12.0, "dominant_far"),
            (12.0, 36.0, "dominant_near"),
        ):
            routes = {
                dominant: CandidateRoute(dominant, dominant_distance, 0.60),
                other: CandidateRoute(other, other_distance, 0.60),
            }
            for presentation in (("yellow", "blue"), ("blue", "yellow")):
                result = arbitrate(
                    pool_with_posterior(posterior),
                    [routes[feature] for feature in presentation],
                )
                cases.append(
                    {
                        "dominant_feature": dominant,
                        "distance_order": distance_order,
                        "presentation_order": list(presentation),
                        "selected_feature": result["selected_feature"],
                        "passed": result["selected_feature"] == dominant,
                        "records": result["records"],
                    }
                )
    return cases


def registered_embodied_case(typed_path, resource_path):
    typed_payload = json.loads(Path(typed_path).read_text(encoding="utf-8"))
    pool = TypedInteractionPool()
    names = [hypothesis.name for hypothesis in pool.hypotheses]
    pool.posterior = np.asarray(
        [float(typed_payload["posterior"][name]) for name in names],
        dtype=np.float64,
    )
    memory = PassiveTerrainResourceMemory(resource_path)
    x, z = 79.98, -176.18
    candidates = []
    targets = {}
    for feature in ("yellow", "blue"):
        selected = memory.best_typed_region(feature, x, z)
        if selected is None:
            continue
        _score, negative_distance, _key, entry = selected
        distance = -negative_distance
        candidates.append(
            CandidateRoute(feature, distance, entry.confidence)
        )
        targets[feature] = [entry.x, entry.z]
    result = arbitrate(pool, candidates)
    return {
        "position": [x, z],
        "posterior": {
            name: float(value) for name, value in zip(names, pool.posterior)
        },
        "targets": targets,
        **result,
    }


def run_audit(typed_path, resource_path):
    cases = counterbalanced_matrix()
    embodied = registered_embodied_case(typed_path, resource_path)
    safety_pool = pool_with_posterior((0.70, 0.05, 0.15, 0.10))
    safety = arbitrate(
        safety_pool,
        [
            CandidateRoute("yellow", 12.0, 0.9, safety_allowed=False),
            CandidateRoute("blue", 20.0, 0.6, safety_allowed=True),
        ],
    )
    criteria = {
        "counterbalanced_identity_order_and_distance": all(
            case["passed"] for case in cases
        ),
        "registered_embodied_case_selects_yellow": (
            embodied["selected_feature"] == "yellow"
        ),
        "safety_gate_is_absolute": safety["selected_feature"] == "blue",
        "all_candidate_records_are_answer_blind": all(
            "dominant_feature" not in record
            for case in cases
            for record in case["records"]
        ),
    }
    return {
        "experiment": "PGNW actionable multi-hypothesis arbitration",
        "scoring": {
            "hazard_cost": 0.25,
            "deadline_remaining_seconds": 240.0,
            "conservative_speed": 2.0,
            "epistemic_weight": 0.10,
            "route_weight": 0.05,
        },
        "counterbalanced_cases": cases,
        "registered_embodied_case": embodied,
        "safety_ablation": safety,
        "criteria": criteria,
        "all_criteria_pass": all(criteria.values()),
        "claim_boundary": (
            "Passing establishes deterministic answer-blind arbitration among "
            "typed causal candidates in an offline and replay-grounded assay. "
            "It does not yet grant the arbiter Unity motor authority or prove "
            "online adaptation under novel causal rules."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--typed-memory",
        default="outputs/typed_interaction_stale_completion_seed168_20260820.json",
    )
    parser.add_argument(
        "--resource-memory",
        default="outputs/resource_memory_typed_acquisition_seed163_20260819.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/pgnw_multi_hypothesis_arbitration_20260820.json",
    )
    args = parser.parse_args()
    payload = run_audit(args.typed_memory, args.resource_memory)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"counterbalanced={sum(c['passed'] for c in payload['counterbalanced_cases'])}/"
        f"{len(payload['counterbalanced_cases'])} "
        f"embodied={payload['registered_embodied_case']['selected_feature']} "
        f"safety={payload['safety_ablation']['selected_feature']}"
    )
    for name, passed in payload["criteria"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"all_criteria_pass: {payload['all_criteria_pass']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
