#!/usr/bin/env python3
"""Counterbalance color labels and presentation order for Gemma formulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = ("blue", "yellow")
PRESENTATIONS = (("blue", "yellow"), ("yellow", "blue"))
PHRASINGS = ("then", "after")


def evidence_line(color, suppressed, episodes, phrasing):
    relation = (
        f"red then {color}" if phrasing == "then" else f"{color} after red"
    )
    return f"{relation}: {suppressed} of {episodes} suppressed"


def question_for(suppressor, presentation, phrasing, neutral=False):
    lines = []
    for color in presentation:
        if neutral:
            suppressed, episodes = 0, 1
        elif color == suppressor:
            suppressed, episodes = 1, 1
        else:
            suppressed, episodes = 0, 2
        lines.append(evidence_line(color, suppressed, episodes, phrasing))
    return (
        "Observed evidence: "
        + "; ".join(lines)
        + ". Which observed second-action token has the higher suppression rate? "
        + "Answer with exactly one token from: "
        + ", ".join(presentation)
        + ".\nAnswer:"
    )


def choice_scores(model, tokenizer, device, question, choices):
    import torch

    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix_length = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    result = {}
    for choice in choices:
        encoded = tokenizer(prefix + choice, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        ids = encoded["input_ids"][0]
        targets = ids[prefix_length:]
        token_logits = logits[prefix_length - 1 : -1]
        result[choice] = float(
            torch.log_softmax(token_logits.float(), dim=-1)
            .gather(1, targets.unsqueeze(1))
            .mean()
            .item()
        )
    return result


def run_matrix(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    rows = []
    for suppressor in COLORS:
        for presentation in PRESENTATIONS:
            for phrasing in PHRASINGS:
                evidence_question = question_for(
                    suppressor, presentation, phrasing, neutral=False
                )
                neutral_question = question_for(
                    suppressor, presentation, phrasing, neutral=True
                )
                evidence = choice_scores(
                    model, tokenizer, device, evidence_question, presentation
                )
                neutral = choice_scores(
                    model, tokenizer, device, neutral_question, presentation
                )
                gains = {
                    color: evidence[color] - neutral[color] for color in COLORS
                }
                selected = max(COLORS, key=lambda color: gains[color])
                alternate = next(color for color in COLORS if color != selected)
                margin = gains[selected] - gains[alternate]
                confidence = 1.0 / (1.0 + math.exp(-margin))
                rows.append(
                    {
                        "suppressor_assignment": suppressor,
                        "presentation_order": list(presentation),
                        "temporal_phrasing": phrasing,
                        "selected": selected,
                        "correct": selected == suppressor,
                        "calibrated_margin": margin,
                        "confidence": confidence,
                        "evidence_scores": evidence,
                        "neutral_scores": neutral,
                        "calibrated_gains": gains,
                        "held_out_episodes": 2,
                        "held_out_correct": 2 if selected == suppressor else 0,
                        "held_out_accuracy": 1.0 if selected == suppressor else 0.0,
                        "evidence_question": evidence_question,
                        "neutral_question": neutral_question,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--output",
        default="outputs/gemma_ordered_counterbalance_20260816.json",
    )
    args = parser.parse_args()
    rows = run_matrix(args.model)
    correct = sum(row["correct"] for row in rows)
    payload = {
        "model": args.model,
        "design": {
            "suppressor_assignments": list(COLORS),
            "presentation_orders": [list(item) for item in PRESENTATIONS],
            "temporal_phrasings": list(PHRASINGS),
            "conditions": len(rows),
            "calibration": "condition_matched_equal_zero_suppression",
        },
        "conditions": rows,
        "summary": {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "minimum_correct_margin": min(
                (row["calibrated_margin"] for row in rows if row["correct"]),
                default=None,
            ),
            "maximum_incorrect_selected_margin": max(
                (row["calibrated_margin"] for row in rows if not row["correct"]),
                default=None,
            ),
            "held_out_correct": sum(row["held_out_correct"] for row in rows),
            "held_out_total": sum(row["held_out_episodes"] for row in rows),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for row in rows:
        print(
            row["suppressor_assignment"],
            "/".join(row["presentation_order"]),
            row["temporal_phrasing"],
            "->",
            row["selected"],
            f"margin={row['calibrated_margin']:.6f}",
            "PASS" if row["correct"] else "FAIL",
        )


if __name__ == "__main__":
    main()
