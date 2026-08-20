#!/usr/bin/env python3
"""Calibrate anonymous-role score offsets, then evaluate a sealed reserve once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROLES = ("role_0", "role_1", "role_2")
CALIBRATION_LABELS = ("amber", "jade", "violet")
CALIBRATION_REGIMES = (
    {"name": "three_of_four_vs_one_of_five", "positive": (3, 4), "control": (1, 5)},
    {"name": "five_of_seven_vs_one_of_six", "positive": (5, 7), "control": (1, 6)},
    {"name": "seven_of_nine_vs_two_of_eight", "positive": (7, 9), "control": (2, 8)},
)
RESERVED_LABELS = ("cobalt", "saffron", "umber")
RESERVED_REGIMES = (
    {"name": "five_of_eight_vs_one_of_six", "positive": (5, 8), "control": (1, 6)},
    {"name": "four_of_seven_vs_zero_of_five", "positive": (4, 7), "control": (0, 5)},
    {"name": "seven_of_ten_vs_two_of_nine", "positive": (7, 10), "control": (2, 9)},
)


def canonicalize_records(records):
    """Map grounded labels to stable roles without consulting their outcomes."""
    if len(records) != 3:
        raise ValueError("calibrated_decoder_requires_three_records")
    labels = sorted(str(item["second"]).lower() for item in records)
    if len(set(labels)) != 3:
        raise ValueError("calibrated_decoder_requires_distinct_second_actions")
    role_for = {label: f"role_{index}" for index, label in enumerate(labels)}
    normalized = []
    for item in records:
        first = str(item["first"]).lower()
        second = str(item["second"]).lower()
        suppressed = int(item["suppressed"])
        episodes = int(item["episodes"])
        if first != "red" or second not in role_for:
            raise ValueError("calibrated_decoder_ungrounded_ordered_action")
        if episodes <= 0 or suppressed < 0 or suppressed > episodes:
            raise ValueError("calibrated_decoder_invalid_counts")
        normalized.append((role_for[second], suppressed, episodes))
    normalized.sort()
    return "; ".join(
        f"{role}|initiator=red|relation=before|suppressed={suppressed}|episodes={episodes}"
        for role, suppressed, episodes in normalized
    ), role_for


def question(canonical_evidence):
    return (
        "Canonical ordered evidence: " + canonical_evidence
        + ". Which anonymous role has the highest suppression rate? "
        "Answer exactly role_0, role_1, or role_2.\nAnswer:"
    )


def neutral_question():
    evidence = "; ".join(
        f"{role}|initiator=red|relation=before|suppressed=0|episodes=1"
        for role in ROLES
    )
    return question(evidence)


def make_records(labels, suppressor, regime, order_shift=0):
    records = []
    for label in labels:
        suppressed, episodes = (
            regime["positive"] if label == suppressor else regime["control"]
        )
        records.append(
            {
                "first": "red",
                "second": label,
                "suppressed": suppressed,
                "episodes": episodes,
            }
        )
    shift = order_shift % len(records)
    return records[shift:] + records[:shift]


def matrix_spec(labels, regimes):
    """Balance the high-rate assignment across every anonymous role."""
    return [
        {
            "suppressor": suppressor,
            "regime": regime,
            "records": make_records(labels, suppressor, regime, assignment_index),
        }
        for regime in regimes
        for assignment_index, suppressor in enumerate(labels)
    ]


def choice_scores(model, tokenizer, device, prompt):
    import torch

    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix_length = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    scores = {}
    for role in ROLES:
        encoded = tokenizer(prefix + role, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        targets = encoded["input_ids"][0, prefix_length:]
        token_logits = logits[prefix_length - 1 : -1]
        scores[role] = float(
            torch.log_softmax(token_logits.float(), dim=-1)
            .gather(1, targets.unsqueeze(1))
            .mean()
            .item()
        )
    return scores


def estimate_role_bias(calibration_rows):
    """Mean-center each role over a fully role-balanced calibration matrix."""
    if not calibration_rows:
        raise ValueError("empty_calibration")
    assignments = {role: 0 for role in ROLES}
    totals = {role: 0.0 for role in ROLES}
    for row in calibration_rows:
        assignments[row["correct_role"]] += 1
        for role in ROLES:
            totals[role] += float(row["neutral_adjusted_gains"][role])
    if len(set(assignments.values())) != 1:
        raise ValueError("calibration_assignments_not_balanced")
    return {role: totals[role] / len(calibration_rows) for role in ROLES}


def apply_role_bias(gains, role_bias):
    if set(gains) != set(ROLES) or set(role_bias) != set(ROLES):
        raise ValueError("role_bias_keys_mismatch")
    return {role: float(gains[role]) - float(role_bias[role]) for role in ROLES}


def fit_rate_weight(calibration_rows, role_bias, required_margin=0.01):
    """Fit one shared evidence-rate coefficient on calibration conditions only."""
    required_weight = 0.0
    for row in calibration_rows:
        rates = row["observed_rates"]
        target = max(ROLES, key=rates.get)
        corrected = apply_role_bias(row["neutral_adjusted_gains"], role_bias)
        for alternative in ROLES:
            if alternative == target:
                continue
            rate_gap = rates[target] - rates[alternative]
            if rate_gap <= 0:
                raise ValueError("calibration_requires_unique_highest_rate")
            needed = (
                corrected[alternative] - corrected[target] + required_margin
            ) / rate_gap
            required_weight = max(required_weight, needed)
    return required_weight


def apply_calibrated_decoder(gains, role_bias, observed_rates, rate_weight):
    if set(observed_rates) != set(ROLES):
        raise ValueError("observed_rate_keys_mismatch")
    centered = apply_role_bias(gains, role_bias)
    return {
        role: centered[role] + float(rate_weight) * float(observed_rates[role])
        for role in ROLES
    }


def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def score_matrix(model, tokenizer, device, labels, regimes, neutral_scores):
    rows = []
    for condition in matrix_spec(labels, regimes):
        canonical, role_for = canonicalize_records(condition["records"])
        scores = choice_scores(model, tokenizer, device, question(canonical))
        gains = {role: scores[role] - neutral_scores[role] for role in ROLES}
        correct_role = role_for[condition["suppressor"]]
        rates = {
            role_for[item["second"]]: item["suppressed"] / item["episodes"]
            for item in condition["records"]
        }
        rows.append(
            {
                "regime": condition["regime"]["name"],
                "grounded_labels": list(labels),
                "raw_records": condition["records"],
                "canonical_evidence": canonical,
                "role_mapping": role_for,
                "correct_role": correct_role,
                "observed_rates": rates,
                "neutral_adjusted_gains": gains,
            }
        )
    return rows


def payload_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def calibration_phase(model_name, output):
    model, tokenizer, device = load_model(model_name)
    neutral_scores = choice_scores(model, tokenizer, device, neutral_question())
    rows = score_matrix(
        model, tokenizer, device, CALIBRATION_LABELS, CALIBRATION_REGIMES,
        neutral_scores,
    )
    role_bias = estimate_role_bias(rows)
    rate_weight = fit_rate_weight(rows, role_bias)
    for row in rows:
        corrected = apply_calibrated_decoder(
            row["neutral_adjusted_gains"], role_bias, row["observed_rates"], rate_weight
        )
        row["bias_corrected_gains"] = corrected
        row["selected_role"] = max(ROLES, key=corrected.get)
        row["correct"] = row["selected_role"] == row["correct_role"]
    frozen = {
        "schema": "balanced_role_offset_rate_fusion_v1",
        "model": model_name,
        "labels": list(CALIBRATION_LABELS),
        "regimes": [item["name"] for item in CALIBRATION_REGIMES],
        "normalization": "lexical_grounded_labels_to_sorted_anonymous_roles",
        "estimator_uses_correct_role": False,
        "rate_weight_target": "unique_maximum_observed_suppression_rate",
        "rate_weight_required_calibration_margin": 0.01,
        "role_assignment_balance": {role: 3 for role in ROLES},
        "neutral_scores": neutral_scores,
        "role_bias": role_bias,
        "rate_weight": rate_weight,
    }
    payload = {
        "frozen_calibration": frozen,
        "calibration_sha256": payload_hash(frozen),
        "conditions": rows,
        "summary": {
            "correct": sum(row["correct"] for row in rows),
            "total": len(rows),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload["summary"], "role_bias": role_bias,
                      "rate_weight": rate_weight,
                      "calibration_sha256": payload["calibration_sha256"]}, indent=2))


def evaluation_phase(model_name, calibration_path, output):
    if output.exists():
        raise FileExistsError(f"reserved_output_already_exists:{output}")
    calibration = json.loads(calibration_path.read_text())
    frozen = calibration["frozen_calibration"]
    if payload_hash(frozen) != calibration["calibration_sha256"]:
        raise ValueError("calibration_hash_mismatch")
    if frozen["model"] != model_name:
        raise ValueError("calibration_model_mismatch")
    role_bias = frozen["role_bias"]
    rate_weight = frozen["rate_weight"]
    model, tokenizer, device = load_model(model_name)
    neutral_scores = choice_scores(model, tokenizer, device, neutral_question())
    rows = score_matrix(
        model, tokenizer, device, RESERVED_LABELS, RESERVED_REGIMES, neutral_scores
    )
    for row in rows:
        corrected = apply_calibrated_decoder(
            row["neutral_adjusted_gains"], role_bias, row["observed_rates"], rate_weight
        )
        selected = max(ROLES, key=corrected.get)
        alternatives = sorted(corrected.values(), reverse=True)
        row["bias_corrected_gains"] = corrected
        row["selected_role"] = selected
        row["correct"] = selected == row["correct_role"]
        row["selected_margin"] = alternatives[0] - alternatives[1]
        row["confidence"] = 1.0 / (1.0 + math.exp(-row["selected_margin"]))
    correct = sum(row["correct"] for row in rows)
    payload = {
        "model": model_name,
        "calibration_sha256": calibration["calibration_sha256"],
        "reserved_design": {
            "labels": list(RESERVED_LABELS),
            "regimes": [item["name"] for item in RESERVED_REGIMES],
            "balanced_assignments": list(RESERVED_LABELS),
            "conditions": len(rows),
            "evaluated_once": True,
        },
        "conditions": rows,
        "summary": {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "minimum_selected_margin": min(row["selected_margin"] for row in rows),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for row in rows:
        print(row["regime"], row["correct_role"], "->", row["selected_role"],
              f"margin={row['selected_margin']:.6f}",
              "PASS" if row["correct"] else "FAIL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--calibration",
        default="outputs/gemma_role_bias_calibration_20260816.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/gemma_role_bias_reserved_20260816.json",
    )
    args = parser.parse_args()
    calibration_path = Path(args.calibration)
    if args.phase == "calibrate":
        calibration_phase(args.model, calibration_path)
    else:
        evaluation_phase(args.model, calibration_path, Path(args.output))


if __name__ == "__main__":
    main()
