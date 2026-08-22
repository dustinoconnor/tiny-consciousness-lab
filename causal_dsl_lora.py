#!/usr/bin/env python3
"""Train and evaluate matched JSON/C1 LoRA adapters for causal binding."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from tiny_scientist import (
    bound_schema_prompt,
    causal_ir_prompt,
    extract_causal_ir,
    extract_labeled_causal_ir,
    extract_semantic_causal_ir,
    extract_json_object,
    labeled_causal_ir_prompt,
    semantic_causal_ir_prompt,
)


BASE_MODEL = "google/gemma-3-270m-it"
TRAIN_FEATURES = (
    "amber",
    "cyan",
    "green",
    "violet",
    "orange",
    "white",
    "black",
    "yellow",
)
HELDOUT_PAIRS = (
    ("red", "blue"),
    ("silver", "gold"),
    ("teal", "pink"),
    ("copper", "indigo"),
)
EFFECTS = {
    1: ("pressure_increase", "+"),
    -1: ("pressure_decrease", "-"),
}


@dataclass(frozen=True)
class CausalExample:
    summary: dict
    target: str
    comparison: str
    effect: str
    effect_code: str
    delay: float
    evidence_order: str


def make_example(
    first,
    second,
    target,
    direction,
    magnitude,
    delay,
    count,
    evidence_order,
):
    comparison = second if target == first else first
    target_values = {
        "isolated_episodes": count,
        "mean_pressure_delta": round(direction * magnitude, 3),
        "positive_pressure_fraction": 1.0 if direction > 0 else 0.0,
        "mean_positive_delay_seconds": delay,
    }
    control_values = {
        "isolated_episodes": count,
        "mean_pressure_delta": 0.0,
        "positive_pressure_fraction": 0.0,
        "mean_positive_delay_seconds": None,
    }
    return CausalExample(
        summary={
            "feature_outcomes": {
                first: target_values if target == first else control_values,
                second: target_values if target == second else control_values,
            },
            "gnw_broadcast_features": {},
        },
        target=target,
        comparison=comparison,
        effect=EFFECTS[direction][0],
        effect_code=EFFECTS[direction][1],
        delay=float(delay),
        evidence_order=evidence_order,
    )


def build_curriculum(seed=104, train_size=192, heldout_size=32):
    rng = random.Random(seed)
    magnitudes = (0.12, 0.2, 0.34, 0.47)
    delays = (3.0, 5.0, 7.0, 10.464, 12.5)

    def examples_from_pairs(pairs, size):
        specifications = [
            (first, second, target, direction, order)
            for first, second in pairs
            for target in (first, second)
            for direction in (1, -1)
            for order in ("canonical", "reversed")
        ]
        if size > len(specifications):
            repeats = math.ceil(size / len(specifications))
            specifications *= repeats
        rng.shuffle(specifications)
        examples = []
        for index, (first, second, target, direction, order) in enumerate(
            specifications[:size]
        ):
            examples.append(
                make_example(
                    first,
                    second,
                    target,
                    direction,
                    rng.choice(magnitudes),
                    rng.choice(delays),
                    2 + (index % 7),
                    order,
                )
            )
        rng.shuffle(examples)
        return examples

    train_pairs = list(itertools.combinations(TRAIN_FEATURES, 2))
    return (
        examples_from_pairs(train_pairs, train_size),
        examples_from_pairs(list(HELDOUT_PAIRS), heldout_size),
    )


def prompt_for(example, representation):
    if representation == "json":
        return bound_schema_prompt(
            example.summary,
            evidence_interface="homeostatic_filtered",
            evidence_order=example.evidence_order,
        )
    if representation == "c1":
        return causal_ir_prompt(
            example.summary,
            evidence_interface="homeostatic_filtered",
            evidence_order=example.evidence_order,
        )
    if representation == "l1":
        return labeled_causal_ir_prompt(
            example.summary,
            evidence_interface="homeostatic_filtered",
            evidence_order=example.evidence_order,
        )
    if representation == "h1":
        return semantic_causal_ir_prompt(
            example.summary,
            evidence_interface="homeostatic_filtered",
            evidence_order=example.evidence_order,
        )
    raise ValueError(f"unsupported_representation:{representation}")


def target_for(example, representation):
    if representation == "json":
        return json.dumps(
            {
                "target_cause": example.target,
                "comparison_feature": example.comparison,
                "observed_effect": example.effect,
                "latency_seconds": example.delay,
                "confidence": 0.8,
            },
            separators=(",", ":"),
        )
    if representation == "c1":
        return (
            f"C1 {example.target} {example.comparison} "
            f"{example.effect_code} {example.delay} 0.8"
        )
    if representation == "l1":
        return (
            f"L1 c {example.target} k {example.comparison} "
            f"e {example.effect_code} t {example.delay} q 0.8"
        )
    if representation == "h1":
        direction = "increase" if example.effect_code == "+" else "decrease"
        return (
            f"H1 cause {example.target} control {example.comparison} "
            f"effect {direction} delay {example.delay} confidence 0.8"
        )
    raise ValueError(f"unsupported_representation:{representation}")


def encode_example(tokenizer, example, representation):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(example, representation)}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    answer_ids = tokenizer.encode(
        target_for(example, representation), add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    return {
        "input_ids": list(prompt_ids) + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def collate_batch(items, pad_token_id):
    import torch

    width = max(len(item["input_ids"]) for item in items)
    input_ids = []
    labels = []
    attention_mask = []
    for item in items:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def parse_prediction(text, representation):
    if representation == "json":
        payload = extract_json_object(text)
    elif representation == "c1":
        payload = extract_causal_ir(text)
    elif representation == "l1":
        payload = extract_labeled_causal_ir(text)
    elif representation == "h1":
        payload = extract_semantic_causal_ir(text)
    else:
        raise ValueError(f"unsupported_representation:{representation}")
    return {
        "target": str(payload["target_cause"]).lower(),
        "comparison": str(payload["comparison_feature"]).lower(),
        "effect": str(payload["observed_effect"]).lower(),
        "delay": float(payload["latency_seconds"]),
    }


def exact_semantic_match(prediction, example):
    return (
        prediction["target"] == example.target
        and prediction["comparison"] == example.comparison
        and prediction["effect"] == example.effect
        and math.isclose(prediction["delay"], example.delay, abs_tol=0.01)
    )


def evaluate(model, tokenizer, examples, representation, device, max_new_tokens=96):
    import torch

    records = []
    started = time.perf_counter()
    model.eval()
    for example in examples:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(example, representation)}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_tokens = int(output.shape[-1] - input_tokens)
        raw = tokenizer.decode(output[0][input_tokens:], skip_special_tokens=True)
        parse_error = None
        semantic_match = False
        try:
            prediction = parse_prediction(raw, representation)
            semantic_match = exact_semantic_match(prediction, example)
        except (KeyError, TypeError, ValueError) as exc:
            prediction = None
            parse_error = str(exc)
        records.append(
            {
                "expected": target_for(example, representation),
                "raw": raw,
                "prediction": prediction,
                "syntax_valid": prediction is not None,
                "semantic_match": semantic_match,
                "parse_error": parse_error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "evidence_order": example.evidence_order,
                "heldout_features": [example.target, example.comparison],
            }
        )
    return {
        "examples": len(records),
        "syntax_valid": sum(record["syntax_valid"] for record in records),
        "semantic_matches": sum(record["semantic_match"] for record in records),
        "syntax_rate": sum(record["syntax_valid"] for record in records) / len(records),
        "semantic_accuracy": sum(record["semantic_match"] for record in records) / len(records),
        "mean_total_tokens": sum(record["total_tokens"] for record in records) / len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "records": records,
    }


def train_adapter(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples, heldout_examples = build_curriculum(
        args.seed, args.train_size, args.heldout_size
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.config.use_cache = False
    encoded = [
        encode_example(tokenizer, example, args.representation)
        for example in train_examples
    ]
    loader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: collate_batch(items, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.perf_counter()
    update = 0
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if (batch_index + 1) % args.gradient_accumulation == 0 or (
                batch_index + 1 == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                print(
                    json.dumps(
                        {
                            "representation": args.representation,
                            "epoch": epoch + 1,
                            "update": update,
                            "loss": round(losses[-1], 4),
                        }
                    ),
                    flush=True,
                )
    training_seconds = round(time.perf_counter() - started, 3)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    model.config.use_cache = True
    heldout = evaluate(
        model,
        tokenizer,
        heldout_examples,
        args.representation,
        device,
    )
    report = {
        "experiment": "causal_dsl_lora_v1",
        "base_model": args.model,
        "representation": args.representation,
        "seed": args.seed,
        "train_examples": len(train_examples),
        "heldout_examples": len(heldout_examples),
        "heldout_feature_pairs": HELDOUT_PAIRS,
        "epochs": args.epochs,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_updates": update,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "mean_training_loss": sum(losses) / len(losses),
        "training_seconds": training_seconds,
        "heldout": heldout,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"training_seconds": training_seconds, "heldout": {k: v for k, v in heldout.items() if k != "records"}}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation", choices=["json", "c1", "l1", "h1"], required=True
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--heldout-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    train_adapter(args)


if __name__ == "__main__":
    main()
