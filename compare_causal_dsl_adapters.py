#!/usr/bin/env python3
"""Frozen comparison of JSON and labeled causal-DSL LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_dsl_lora import (
    BASE_MODEL,
    CausalExample,
    build_curriculum,
    evaluate,
)
from tiny_scientist import analyze, load_rows


def unity_examples(recording):
    summary = analyze(load_rows(recording))["summary"]
    outcomes = summary["feature_outcomes"]
    delay = float(outcomes["red"]["mean_positive_delay_seconds"])
    return [
        CausalExample(
            summary=summary,
            target="red",
            comparison="blue",
            effect="pressure_increase",
            effect_code="+",
            delay=delay,
            evidence_order=order,
        )
        for order in ("canonical", "reversed")
    ]


def paired_summary(json_result, l1_result):
    json_records = json_result["records"]
    l1_records = l1_result["records"]
    if len(json_records) != len(l1_records):
        raise ValueError("paired_evaluation_length_mismatch")
    json_only = 0
    l1_only = 0
    both_correct = 0
    both_wrong = 0
    for json_record, l1_record in zip(json_records, l1_records, strict=True):
        json_ok = bool(json_record["semantic_match"])
        l1_ok = bool(l1_record["semantic_match"])
        both_correct += int(json_ok and l1_ok)
        both_wrong += int(not json_ok and not l1_ok)
        json_only += int(json_ok and not l1_ok)
        l1_only += int(l1_ok and not json_ok)
    json_tokens = json_result["mean_total_tokens"]
    l1_tokens = l1_result["mean_total_tokens"]
    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "json_only_correct": json_only,
        "l1_only_correct": l1_only,
        "accuracy_difference_l1_minus_json": (
            l1_result["semantic_accuracy"] - json_result["semantic_accuracy"]
        ),
        "l1_visible_token_reduction_percent": 100.0
        * (json_tokens - l1_tokens)
        / json_tokens,
    }


def load_and_evaluate(adapter, representation, examples, unity, model_name):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    result = evaluate(model, tokenizer, examples, representation, device)
    unity_result = evaluate(model, tokenizer, unity, representation, device)
    del model
    del base
    if device == "mps":
        torch.mps.empty_cache()
    return result, unity_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--json-adapter", required=True)
    parser.add_argument("--l1-adapter", required=True)
    parser.add_argument(
        "--l1-representation", choices=["l1", "h1"], default="l1"
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--heldout-size", type=int, default=128)
    parser.add_argument(
        "--output",
        default="outputs/causal_dsl_frozen_comparison_20260804.json",
    )
    args = parser.parse_args()
    _unused_train, heldout = build_curriculum(
        seed=args.seed, train_size=1, heldout_size=args.heldout_size
    )
    unity = unity_examples(args.recording)
    json_result, json_unity = load_and_evaluate(
        args.json_adapter, "json", heldout, unity, args.model
    )
    l1_result, l1_unity = load_and_evaluate(
        args.l1_adapter, args.l1_representation, heldout, unity, args.model
    )
    result = {
        "experiment": "frozen_labeled_causal_dsl_comparison_v1",
        "evaluation_seed": args.seed,
        "heldout_examples": args.heldout_size,
        "training_changed_after_evaluation_started": False,
        "json_adapter": args.json_adapter,
        "l1_adapter": args.l1_adapter,
        "compact_representation": args.l1_representation,
        "heldout": {
            "json": json_result,
            "l1": l1_result,
            "paired": paired_summary(json_result, l1_result),
        },
        "unity_transfer": {
            "recording": args.recording,
            "orders": ["canonical", "reversed"],
            "json": json_unity,
            "l1": l1_unity,
            "paired": paired_summary(json_unity, l1_unity),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    concise = {
        "heldout": {
            "json_accuracy": json_result["semantic_accuracy"],
            "l1_accuracy": l1_result["semantic_accuracy"],
            "json_mean_tokens": json_result["mean_total_tokens"],
            "l1_mean_tokens": l1_result["mean_total_tokens"],
            "paired": result["heldout"]["paired"],
        },
        "unity_transfer": {
            "json_accuracy": json_unity["semantic_accuracy"],
            "l1_accuracy": l1_unity["semantic_accuracy"],
            "paired": result["unity_transfer"]["paired"],
        },
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
