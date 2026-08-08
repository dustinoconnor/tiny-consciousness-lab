#!/usr/bin/env python3
"""Frozen new-seed audit of greedy versus symmetric constrained L1 decoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from causal_dsl_lora import (
    build_curriculum,
    evaluate,
    exact_semantic_match,
    parse_prediction,
)
from tiny_scientist import HypothesisGenerationError, generate_hypothesis_with_model


def evaluate_constrained(
    model,
    tokenizer,
    examples,
    device,
    contract="labeled_causal_ir_constrained",
):
    records = []
    started = time.perf_counter()
    for example in examples:
        try:
            _hypothesis, raw, metrics = generate_hypothesis_with_model(
                example.summary,
                tokenizer,
                model,
                device,
                evidence_interface="homeostatic_filtered",
                hypothesis_contract=contract,
                evidence_order=example.evidence_order,
            )
            prediction = parse_prediction(raw, "l1")
            match = exact_semantic_match(prediction, example)
            rejection = None
        except (HypothesisGenerationError, KeyError, TypeError, ValueError) as exc:
            raw = getattr(exc, "raw_output", "")
            metrics = getattr(exc, "metrics", {})
            prediction = None
            match = False
            rejection = str(exc)
        records.append(
            {
                "raw": raw,
                "prediction": prediction,
                "semantic_match": match,
                "rejection": rejection,
                "metrics": metrics,
                "expected": {
                    "target": example.target,
                    "comparison": example.comparison,
                    "effect": example.effect,
                    "delay": example.delay,
                },
                "evidence_order": example.evidence_order,
            }
        )
        print(
            json.dumps(
                {
                    "constrained_case": len(records),
                    "total": len(examples),
                    "correct": sum(record["semantic_match"] for record in records),
                }
            ),
            flush=True,
        )
    return {
        "examples": len(records),
        "semantic_matches": sum(record["semantic_match"] for record in records),
        "semantic_accuracy": sum(record["semantic_match"] for record in records)
        / len(records),
        "mean_visible_tokens": sum(
            record["metrics"].get("total_tokens", 0) for record in records
        )
        / len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "records": records,
    }


def paired_summary(greedy, constrained):
    pairs = list(zip(greedy["records"], constrained["records"], strict=True))
    return {
        "both_correct": sum(a["semantic_match"] and b["semantic_match"] for a, b in pairs),
        "greedy_only_correct": sum(a["semantic_match"] and not b["semantic_match"] for a, b in pairs),
        "constrained_only_correct": sum(not a["semantic_match"] and b["semantic_match"] for a, b in pairs),
        "both_wrong": sum(not a["semantic_match"] and not b["semantic_match"] for a, b in pairs),
        "accuracy_difference_constrained_minus_greedy": (
            constrained["semantic_accuracy"] - greedy["semantic_accuracy"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--adapter", default="checkpoints/causal_dsl_l1_lora_1b_20260804"
    )
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--heldout-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--contract",
        choices=[
            "labeled_causal_ir_constrained",
            "labeled_causal_ir_masked_greedy",
            "labeled_causal_ir_syntax_masked_greedy",
        ],
        default="labeled_causal_ir_constrained",
    )
    parser.add_argument(
        "--output", default="outputs/l1_constraint_repair_seed307_20260805.json"
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _train, examples = build_curriculum(
        seed=args.seed, train_size=1, heldout_size=args.heldout_size
    )
    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    greedy = evaluate(model, tokenizer, examples, "l1", device)
    constrained = evaluate_constrained(
        model, tokenizer, examples, device, contract=args.contract
    )
    result = {
        "experiment": "post_mechanistic_l1_formal_constraint_repair_v1",
        "evaluation_seed": args.seed,
        "heldout_examples": args.heldout_size,
        "model": args.model,
        "adapter": args.adapter,
        "adapter_or_prompt_changed": False,
        "repair_contract": args.contract,
        "repair": (
            "syntax-only finite-state masked greedy; repeated cause/comparison "
            "roles remain permitted"
            if args.contract == "labeled_causal_ir_syntax_masked_greedy"
            else "symmetric candidate-constrained L1 decoding; cause and "
            "comparison must be distinct observed features"
        ),
        "greedy": greedy,
        "constrained": constrained,
        "paired": paired_summary(greedy, constrained),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "greedy_accuracy": greedy["semantic_accuracy"],
                "constrained_accuracy": constrained["semantic_accuracy"],
                "paired": result["paired"],
                "greedy_elapsed_seconds": greedy["elapsed_seconds"],
                "constrained_elapsed_seconds": constrained["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
