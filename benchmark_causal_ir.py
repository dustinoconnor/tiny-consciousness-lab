#!/usr/bin/env python3
"""Matched JSON-versus-C1 benchmark for the Tiny Scientist hypothesis boundary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tiny_scientist import (
    HypothesisGenerationError,
    analyze,
    bound_schema_prompt,
    causal_ir_prompt,
    causal_ir_v2_prompt,
    generate_hypothesis_with_model,
    load_rows,
)


DEFAULT_MODELS = ["google/gemma-3-270m-it", "google/gemma-3-1b-it"]
CONTRACTS = ["bound_schema", "causal_ir"]
ORDERS = ["canonical", "reversed"]


def summarize_trials(trials):
    by_model = {}
    for model in sorted({trial["model"] for trial in trials}):
        model_trials = [trial for trial in trials if trial["model"] == model]
        contract_names = sorted({trial["contract"] for trial in model_trials})
        contracts = {}
        for contract in contract_names:
            selected = [
                trial for trial in model_trials if trial["contract"] == contract
            ]
            accepted = sum(trial["accepted"] for trial in selected)
            total_tokens = sum(trial["metrics"]["total_tokens"] for trial in selected)
            contracts[contract] = {
                "trials": len(selected),
                "accepted": accepted,
                "success_rate": accepted / len(selected) if selected else 0.0,
                "mean_total_tokens_per_trial": (
                    statistics.mean(
                        trial["metrics"]["total_tokens"] for trial in selected
                    )
                    if selected
                    else None
                ),
                "retry_adjusted_tokens_per_accepted": (
                    total_tokens / accepted if accepted else None
                ),
                "mean_elapsed_seconds": (
                    statistics.mean(
                        trial["metrics"]["elapsed_seconds"] for trial in selected
                    )
                    if selected
                    else None
                ),
            }
        json_name = next(name for name in contract_names if name.startswith("bound_schema"))
        ir_name = next(name for name in contract_names if name.startswith("causal_ir"))
        json_cost = contracts[json_name]["retry_adjusted_tokens_per_accepted"]
        ir_cost = contracts[ir_name]["retry_adjusted_tokens_per_accepted"]
        savings = None
        if json_cost and ir_cost is not None:
            savings = 100.0 * (json_cost - ir_cost) / json_cost
        by_model[model] = {
            "contracts": contracts,
            "causal_ir_retry_adjusted_token_savings_percent": savings,
        }
    return by_model


def run_benchmark(recording, models, max_new_tokens=160, contracts=CONTRACTS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    summary = analyze(load_rows(recording))["summary"]
    trials = []
    static_counts = {}
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        static_counts[model_name] = {}
        for order in ORDERS:
            prompts = {
                "bound_schema": bound_schema_prompt(
                    summary, "homeostatic_filtered", order
                ),
                "causal_ir": causal_ir_prompt(
                    summary, "homeostatic_filtered", order
                ),
                "bound_schema_constrained": bound_schema_prompt(
                    summary, "homeostatic_filtered", order
                ),
                "causal_ir_constrained": causal_ir_prompt(
                    summary, "homeostatic_filtered", order
                ),
                "bound_schema_scaffolded": bound_schema_prompt(
                    summary,
                    "homeostatic_filtered",
                    order,
                    include_abstract_example=True,
                ),
                "causal_ir_scaffolded": causal_ir_prompt(
                    summary,
                    "homeostatic_filtered",
                    order,
                    include_abstract_example=True,
                ),
                "bound_schema_scaffolded_constrained": bound_schema_prompt(
                    summary,
                    "homeostatic_filtered",
                    order,
                    include_abstract_example=True,
                ),
                "causal_ir_scaffolded_constrained": causal_ir_prompt(
                    summary,
                    "homeostatic_filtered",
                    order,
                    include_abstract_example=True,
                ),
                "causal_ir_v2_constrained": causal_ir_v2_prompt(
                    summary, "homeostatic_filtered", order
                ),
            }
            static_counts[model_name][order] = {}
            for contract in contracts:
                prompt = prompts[contract]
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                static_counts[model_name][order][contract] = int(
                    encoded["input_ids"].shape[-1]
                )
                try:
                    hypothesis, raw, metrics = generate_hypothesis_with_model(
                        summary,
                        tokenizer,
                        model,
                        device,
                        max_new_tokens=max_new_tokens,
                        evidence_interface="homeostatic_filtered",
                        hypothesis_contract=contract,
                        evidence_order=order,
                    )
                    trial = {
                        "model": model_name,
                        "contract": contract,
                        "evidence_order": order,
                        "accepted": True,
                        "hypothesis": hypothesis,
                        "raw_model_output": raw,
                        "rejection": None,
                        "metrics": metrics,
                    }
                except HypothesisGenerationError as exc:
                    trial = {
                        "model": model_name,
                        "contract": contract,
                        "evidence_order": order,
                        "accepted": False,
                        "hypothesis": None,
                        "raw_model_output": exc.raw_output,
                        "rejection": str(exc),
                        "metrics": exc.metrics,
                    }
                trials.append(trial)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return {
        "experiment": "agent_native_causal_representation_benchmark_v1",
        "recording": str(recording),
        "models": models,
        "contracts": {
            "evaluated": contracts,
            "json_baseline": next(
                name for name in contracts if name.startswith("bound_schema")
            ),
            "agent_native_ir": next(
                name for name in contracts if name.startswith("causal_ir")
            ),
            "agent_native_ir_grammar": "C1 cause comparison effect delay confidence",
        },
        "controls": {
            "evidence_interface": "homeostatic_filtered",
            "evidence_orders": ORDERS,
            "decoding": (
                "symmetric_candidate_constrained_beam_search"
                if any(name.endswith("_constrained") for name in contracts)
                else "greedy_two_attempt_maximum"
            ),
            "same_canonical_semantic_verifier": True,
            "same_formal_falsifier_compiler": True,
            "authority": 0.0,
        },
        "summary": summary,
        "static_chat_prompt_tokens": static_counts,
        "trials": trials,
        "aggregate": summarize_trials(trials),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--contracts",
        nargs=2,
        choices=[
            "bound_schema",
            "causal_ir",
            "bound_schema_constrained",
            "causal_ir_constrained",
            "bound_schema_scaffolded",
            "causal_ir_scaffolded",
            "bound_schema_scaffolded_constrained",
            "causal_ir_scaffolded_constrained",
            "causal_ir_v2_constrained",
        ],
        default=CONTRACTS,
    )
    parser.add_argument(
        "--output",
        default="outputs/tiny_scientist_causal_ir_benchmark_20260804.json",
    )
    args = parser.parse_args()
    result = run_benchmark(
        args.recording, args.models, args.max_new_tokens, args.contracts
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
