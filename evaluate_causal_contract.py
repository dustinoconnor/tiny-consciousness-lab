#!/usr/bin/env python3
"""Evaluate one frozen causal adapter/contract on a deterministic curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_dsl_lora import build_curriculum, evaluate
from compare_l1_constraint_repair import evaluate_constrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--contract",
        choices=[
            "json",
            "labeled_causal_ir_masked_greedy",
            "labeled_causal_ir_syntax_masked_greedy",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--heldout-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", required=True)
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
    if args.contract == "json":
        evaluation = evaluate(model, tokenizer, examples, "json", device)
    else:
        evaluation = evaluate_constrained(
            model, tokenizer, examples, device, contract=args.contract
        )
    result = {
        "experiment": "frozen_causal_contract_confirmation_v1",
        "evaluation_seed": args.seed,
        "heldout_examples": args.heldout_size,
        "model": args.model,
        "adapter": args.adapter,
        "contract": args.contract,
        "adapter_prompt_or_decoder_changed_after_registration": False,
        "evaluation": evaluation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "contract": args.contract,
                "semantic_matches": evaluation["semantic_matches"],
                "examples": evaluation["examples"],
                "semantic_accuracy": evaluation["semantic_accuracy"],
                "elapsed_seconds": evaluation["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
