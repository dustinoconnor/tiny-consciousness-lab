#!/usr/bin/env python3
"""Frozen three-way audit of ordinary, syntax-only, and role-bound L1 decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_dsl_lora import build_curriculum, evaluate
from compare_l1_constraint_repair import evaluate_constrained


def pairwise_summary(left, right):
    pairs = list(zip(left["records"], right["records"], strict=True))
    return {
        "both_correct": sum(
            a["semantic_match"] and b["semantic_match"] for a, b in pairs
        ),
        "left_only_correct": sum(
            a["semantic_match"] and not b["semantic_match"] for a, b in pairs
        ),
        "right_only_correct": sum(
            not a["semantic_match"] and b["semantic_match"] for a, b in pairs
        ),
        "both_wrong": sum(
            not a["semantic_match"] and not b["semantic_match"] for a, b in pairs
        ),
        "accuracy_difference_right_minus_left": (
            right["semantic_accuracy"] - left["semantic_accuracy"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--adapter", default="checkpoints/causal_dsl_l1_lora_1b_20260804"
    )
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--heldout-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--output", default="outputs/l1_mask_controls_seed313_20260808.json"
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

    ordinary = evaluate(model, tokenizer, examples, "l1", device)
    syntax_only = evaluate_constrained(
        model,
        tokenizer,
        examples,
        device,
        contract="labeled_causal_ir_syntax_masked_greedy",
    )
    role_bound = evaluate_constrained(
        model,
        tokenizer,
        examples,
        device,
        contract="labeled_causal_ir_masked_greedy",
    )
    result = {
        "experiment": "l1_syntax_versus_semantic_role_constraint_v1",
        "protocol": "L1_SYNTAX_CONTROL_PROTOCOL_20260808.md",
        "evaluation_seed": args.seed,
        "heldout_examples": args.heldout_size,
        "model": args.model,
        "adapter": args.adapter,
        "prompt_adapter_validator_changed_after_registration": False,
        "ordinary_greedy": ordinary,
        "syntax_only_masked_greedy": syntax_only,
        "syntax_plus_distinct_role_masked_greedy": role_bound,
        "paired_ordinary_to_syntax": pairwise_summary(ordinary, syntax_only),
        "paired_syntax_to_role": pairwise_summary(syntax_only, role_bound),
        "paired_ordinary_to_role": pairwise_summary(ordinary, role_bound),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "ordinary": ordinary["semantic_matches"],
                "syntax_only": syntax_only["semantic_matches"],
                "syntax_plus_distinct_role": role_bound["semantic_matches"],
                "paired_ordinary_to_syntax": result[
                    "paired_ordinary_to_syntax"
                ],
                "paired_syntax_to_role": result["paired_syntax_to_role"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
