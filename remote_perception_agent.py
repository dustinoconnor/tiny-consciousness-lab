#!/usr/bin/env python3
"""Local Gemma first-impression sampler for the sealed target-card pilot."""

from __future__ import annotations

from collections import Counter
import json
import math
import re

from remote_perception_lab import DESCRIPTOR_FIELDS, validate_prediction


DEFAULT_PREDICTION = {
    "dominant_color": "blue",
    "shape": "circle",
    "count": 1,
    "arrangement": "horizontal",
    "texture": "solid",
    "background": "light",
}


def intuition_prompt(trial_id):
    return (
        "You receive only an opaque target identifier. Do not reason about the "
        "identifier or explain. Emit one rapid first-impression JSON object using "
        "exactly these fields and allowed values: "
        "dominant_color=[red,blue,yellow,green,purple,orange]; "
        "shape=[circle,triangle,square,star,cross,wave]; count=[1,2,3,5]; "
        "arrangement=[horizontal,vertical,diagonal,radial]; "
        "texture=[solid,striped,dotted]; background=[light,dark]. "
        f"TARGET_ID={trial_id}"
    )


def parse_impression(text):
    match = re.search(r"\{[^{}]+\}", str(text), flags=re.DOTALL)
    if not match:
        raise ValueError("remote_impression_json_missing")
    return validate_prediction(json.loads(match.group(0)))


def aggregate_impressions(impressions, weights=None):
    if not impressions:
        raise ValueError("remote_impressions_empty")
    normalized = [validate_prediction(item) for item in impressions]
    weights = [1.0] * len(normalized) if weights is None else [float(x) for x in weights]
    if len(weights) != len(normalized) or any(not math.isfinite(x) or x <= 0 for x in weights):
        raise ValueError("remote_impression_weights_invalid")
    result = {}
    support = {}
    for field in DESCRIPTOR_FIELDS:
        totals = Counter()
        for item, weight in zip(normalized, weights):
            totals[item[field]] += weight
        selected = max(totals, key=lambda value: (totals[value], str(value)))
        result[field] = selected
        support[field] = totals[selected] / sum(totals.values())
    return validate_prediction(result), support


def impression_accuracy(impression, target):
    impression = validate_prediction(impression)
    return sum(impression[field] == target[field] for field in DESCRIPTOR_FIELDS) / len(
        DESCRIPTOR_FIELDS
    )


def update_slot_weights(weights, impressions, target, learning_rate=0.25):
    if len(weights) != len(impressions):
        raise ValueError("remote_slot_weight_count_mismatch")
    updated = [
        float(weight)
        * math.exp(float(learning_rate) * (impression_accuracy(item, target) - 0.25))
        for weight, item in zip(weights, impressions)
    ]
    mean = sum(updated) / len(updated)
    return [value / mean for value in updated]


class LocalGemmaIntuitionSampler:
    def __init__(self, model_name="google/gemma-3-1b-it"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", local_files_only=True
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model_name = model_name

    def sample(self, trial_id, count=7, temperature=0.9):
        count = max(1, int(count))
        prompt = intuition_prompt(trial_id)
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=90,
                do_sample=count > 1,
                temperature=float(temperature) if count > 1 else None,
                num_return_sequences=count,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prefix = inputs["input_ids"].shape[-1]
        raw = [
            self.tokenizer.decode(output[prefix:], skip_special_tokens=True)
            for output in outputs
        ]
        parsed = []
        malformed = []
        for index, text in enumerate(raw):
            try:
                parsed.append(parse_impression(text))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                malformed.append({"slot": index, "error": f"{type(exc).__name__}:{exc}", "raw": text})
                parsed.append(dict(DEFAULT_PREDICTION))
        return parsed, raw, malformed
