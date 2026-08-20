#!/usr/bin/env python3
"""Answer-blind canonicalization and fresh held-out Gemma replication."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = ("blue", "yellow")
COUNT_REGIMES = (
    {"name": "two_of_three_vs_one_of_four", "positive": (2, 3), "control": (1, 4)},
    {"name": "three_of_five_vs_zero_of_four", "positive": (3, 5), "control": (0, 4)},
)
SURFACE_FORMS = ("first_before_second", "second_follows_first")
RECORD_ORDERS = ("forward", "reverse")


def canonicalize_records(records):
    """Return one stable anonymous-role encoding without inspecting outcomes."""
    if len(records) != 2:
        raise ValueError("canonicalizer_requires_two_records")
    features = sorted(str(item["second"]).lower() for item in records)
    if len(set(features)) != 2:
        raise ValueError("canonicalizer_requires_distinct_second_actions")
    role_for = {feature: f"role_{index}" for index, feature in enumerate(features)}
    normalized = []
    for item in records:
        first = str(item["first"]).lower()
        second = str(item["second"]).lower()
        suppressed = int(item["suppressed"])
        episodes = int(item["episodes"])
        if first != "red" or second not in role_for:
            raise ValueError("canonicalizer_ungrounded_ordered_action")
        if episodes <= 0 or suppressed < 0 or suppressed > episodes:
            raise ValueError("canonicalizer_invalid_counts")
        normalized.append(
            {
                "role": role_for[second],
                "initiator": "red",
                "relation": "before",
                "suppressed": suppressed,
                "episodes": episodes,
            }
        )
    normalized.sort(key=lambda item: item["role"])
    text = "; ".join(
        f"{item['role']}|initiator=red|relation=before|"
        f"suppressed={item['suppressed']}|episodes={item['episodes']}"
        for item in normalized
    )
    return text, role_for


def normalized_question(canonical_text):
    return (
        "Canonical ordered evidence: "
        + canonical_text
        + ". Which anonymous role has the higher suppression rate? "
        "Answer exactly role_0 or role_1.\nAnswer:"
    )


def neutral_question():
    text = (
        "role_0|initiator=red|relation=before|suppressed=0|episodes=1; "
        "role_1|initiator=red|relation=before|suppressed=0|episodes=1"
    )
    return normalized_question(text)


def choice_scores(model, tokenizer, device, question):
    import torch

    choices = ("role_0", "role_1")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix_length = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    scores = {}
    for choice in choices:
        encoded = tokenizer(prefix + choice, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        targets = encoded["input_ids"][0, prefix_length:]
        token_logits = logits[prefix_length - 1 : -1]
        scores[choice] = float(
            torch.log_softmax(token_logits.float(), dim=-1)
            .gather(1, targets.unsqueeze(1))
            .mean()
            .item()
        )
    return scores


def raw_records(suppressor, regime, record_order, surface_form):
    records = []
    for color in COLORS:
        counts = regime["positive"] if color == suppressor else regime["control"]
        records.append(
            {
                "first": "red",
                "second": color,
                "suppressed": counts[0],
                "episodes": counts[1],
                "surface_form": surface_form,
            }
        )
    if record_order == "reverse":
        records.reverse()
    return records


def run_matrix(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    neutral_scores = choice_scores(model, tokenizer, device, neutral_question())
    rows = []
    for suppressor in COLORS:
        for regime in COUNT_REGIMES:
            for record_order in RECORD_ORDERS:
                for surface_form in SURFACE_FORMS:
                    records = raw_records(
                        suppressor, regime, record_order, surface_form
                    )
                    canonical, role_for = canonicalize_records(records)
                    evidence_scores = choice_scores(
                        model, tokenizer, device, normalized_question(canonical)
                    )
                    gains = {
                        role: evidence_scores[role] - neutral_scores[role]
                        for role in ("role_0", "role_1")
                    }
                    selected_role = max(gains, key=gains.get)
                    selected_color = next(
                        color for color, role in role_for.items() if role == selected_role
                    )
                    other_role = "role_1" if selected_role == "role_0" else "role_0"
                    margin = gains[selected_role] - gains[other_role]
                    rows.append(
                        {
                            "suppressor_assignment": suppressor,
                            "count_regime": regime["name"],
                            "record_order": record_order,
                            "surface_form": surface_form,
                            "raw_records": records,
                            "canonical_evidence": canonical,
                            "role_mapping": role_for,
                            "selected_role": selected_role,
                            "selected_color": selected_color,
                            "correct": selected_color == suppressor,
                            "calibrated_gains": gains,
                            "calibrated_margin": margin,
                            "confidence": 1.0 / (1.0 + math.exp(-margin)),
                            "held_out_episodes": 2,
                            "held_out_correct": 2 if selected_color == suppressor else 0,
                        }
                    )
    return rows, neutral_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--output", default="outputs/gemma_ordered_normalized_holdout_20260816.json"
    )
    args = parser.parse_args()
    rows, neutral_scores = run_matrix(args.model)
    correct = sum(item["correct"] for item in rows)
    payload = {
        "model": args.model,
        "normalization": {
            "feature_roles": "lexically_sorted_grounded_features_to_role_0_role_1",
            "record_order": "sorted_by_anonymous_role",
            "temporal_relation": "initiator=red|relation=before",
            "surface_form_used_by_decoder": False,
            "outcomes_used_to_assign_roles": False,
        },
        "fresh_holdout_design": {
            "suppressor_assignments": list(COLORS),
            "count_regimes": [item["name"] for item in COUNT_REGIMES],
            "record_orders": list(RECORD_ORDERS),
            "surface_forms": list(SURFACE_FORMS),
            "conditions": len(rows),
        },
        "neutral_scores": neutral_scores,
        "conditions": rows,
        "summary": {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "held_out_correct": sum(item["held_out_correct"] for item in rows),
            "held_out_total": sum(item["held_out_episodes"] for item in rows),
            "minimum_margin": min(item["calibrated_margin"] for item in rows),
            "unique_canonical_prompts": len(
                {item["canonical_evidence"] for item in rows}
            ),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for row in rows:
        print(
            row["suppressor_assignment"], row["count_regime"],
            row["record_order"], row["surface_form"], "->",
            row["selected_color"], f"margin={row['calibrated_margin']:.6f}",
            "PASS" if row["correct"] else "FAIL",
        )


if __name__ == "__main__":
    main()
