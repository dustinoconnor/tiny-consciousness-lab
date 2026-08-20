#!/usr/bin/env python3
"""Frozen Smith-inspired language-role ablation for two verified needs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


MODES = ("meaningful", "anonymous", "shuffled")
NEEDS = ("pending_hazard", "critical_hunger")


def role_vocabulary(mode):
    if mode == "meaningful":
        return {
            "protective": "antidote",
            "metabolic": "food",
            "definitions": (
                "antidote=reduces a pending hazard; "
                "food=restores metabolic reserve"
            ),
        }
    if mode == "anonymous":
        return {
            "protective": "role_1",
            "metabolic": "role_2",
            "definitions": (
                "role_1=reduces a pending hazard; "
                "role_2=restores metabolic reserve"
            ),
        }
    if mode == "shuffled":
        return {
            "protective": "food",
            "metabolic": "antidote",
            "definitions": (
                "food=reduces a pending hazard; "
                "antidote=restores metabolic reserve"
            ),
        }
    raise ValueError("unsupported_language_classification_mode")


def frozen_cases():
    cases = []
    for protective_feature in ("yellow", "blue"):
        metabolic_feature = "blue" if protective_feature == "yellow" else "yellow"
        for order in (("yellow", "blue"), ("blue", "yellow")):
            for need in NEEDS:
                for mode in MODES:
                    vocabulary = role_vocabulary(mode)
                    roles = {
                        protective_feature: vocabulary["protective"],
                        metabolic_feature: vocabulary["metabolic"],
                    }
                    expected_feature = (
                        protective_feature
                        if need == "pending_hazard"
                        else metabolic_feature
                    )
                    candidates = [
                        {"id": candidate_id, "feature": feature, "role": roles[feature]}
                        for candidate_id, feature in zip(("A", "B"), order)
                    ]
                    expected_id = next(
                        item["id"] for item in candidates
                        if item["feature"] == expected_feature
                    )
                    cases.append({
                        "case_id": len(cases) + 1,
                        "classification_mode": mode,
                        "definitions": vocabulary["definitions"],
                        "need": need,
                        "protective_feature": protective_feature,
                        "metabolic_feature": metabolic_feature,
                        "candidate_order": list(order),
                        "candidates": candidates,
                        "expected_id": expected_id,
                        "expected_feature": expected_feature,
                    })
    return cases


def prompt_for(case):
    candidate_lines = "\n".join(
        f"CANDIDATE {item['id']}: object={item['feature']} role={item['role']}"
        for item in case["candidates"]
    )
    return (
        "Route an embodied agent using only the role definitions and current "
        "need. Each candidate has one independently verified role. Do not use "
        "candidate order as evidence. Return exactly choice=A or choice=B.\n"
        f"ROLE DEFINITIONS: {case['definitions']}\n"
        f"CURRENT NEED: {case['need']}\n"
        f"{candidate_lines}"
    )


def parse_choice(text):
    match = re.search(r"choice\s*=\s*([AB])\b", str(text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else "invalid"


def evaluate(model_name, adapter_path=None, device="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = (
        "mps"
        if device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if device == "auto" else device)
    )
    tokenizer_source = adapter_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(resolved_device)
    model.eval()
    records = []
    started = time.perf_counter()
    for case in frozen_cases():
        prompt = prompt_for(case)
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(resolved_device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=12,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        choice = parse_choice(generated)
        records.append({
            **case,
            "prompt": prompt,
            "raw_output": generated,
            "choice": choice,
            "passed": choice == case["expected_id"],
        })
    by_mode = {
        mode: {
            "passed": sum(
                item["passed"] for item in records
                if item["classification_mode"] == mode
            ),
            "total": sum(
                item["classification_mode"] == mode for item in records
            ),
        }
        for mode in MODES
    }
    return {
        "experiment": "smith_language_classifier_frozen_ablation_v1",
        "model": model_name,
        "adapter": adapter_path,
        "device": resolved_device,
        "matrix_frozen_before_evaluation": True,
        "cases": records,
        "by_mode": by_mode,
        "overall_passed": sum(item["passed"] for item in records),
        "overall_total": len(records),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "This measures whether lexical role names affect a frozen local "
            "language-mediated routing task. It is not evidence of phenomenal "
            "consciousness and does not grant labels motor authority."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--adapter", default="checkpoints/causal_dsl_l1_lora_1b_20260804"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--output", default="outputs/smith_language_classifier_ablation_20260820.json"
    )
    args = parser.parse_args()
    result = evaluate(args.model, args.adapter or None, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "by_mode": result["by_mode"],
        "overall": [result["overall_passed"], result["overall_total"]],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
