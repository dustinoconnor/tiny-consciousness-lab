#!/usr/bin/env python3
"""Let local Gemma formulate one typed ordered rule from embodied evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


ALLOWED_FEATURES = {"red", "blue", "yellow"}
ALLOWED_EFFECTS = {"suppresses_probe", "does_not_suppress_probe"}


class LocalCalibratedOrderedProposer:
    """Callable used by the live loop's background formulation worker."""

    def __init__(self, model_name="google/gemma-3-1b-it"):
        self.model_name = str(model_name)

    def __call__(self, discovery_summary, cutoff):
        proposal, raw, diagnostics = score_suppressor_slot(
            self.model_name, discovery_summary
        )
        compiled = compile_proposal(proposal, discovery_summary, cutoff)
        return compiled, raw, diagnostics


def accepted_observations(paths):
    observations = []
    seen_updates = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            state = row.get("typed_interaction_learning") or {}
            updates = int(state.get("updates", 0) or 0)
            if updates <= seen_updates:
                continue
            action = str(state.get("last_action", "none"))
            outcome = str(state.get("last_outcome", "none"))
            observations.append(
                {
                    "index": updates,
                    "ordered_action": action,
                    "probe_suppressed": outcome == "suppressed",
                }
            )
            seen_updates = updates
    return observations


def summarize(observations):
    counts = {}
    for item in observations:
        action = item["ordered_action"]
        values = counts.setdefault(action, Counter())
        values["episodes"] += 1
        values["suppressed"] += int(item["probe_suppressed"])
    return {
        action: {
            "episodes": values["episodes"],
            "suppressed": values["suppressed"],
            "not_suppressed": values["episodes"] - values["suppressed"],
        }
        for action, values in sorted(counts.items())
    }


def prompt_for(summary):
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return (
        "You are an embodied scientist. Compare the observed suppression rates "
        "and infer the most specific positive suppression rule. Do not merely "
        "restate a negative control that failed to suppress. Object tokens are red, "
        "blue, and yellow. A record such as red_then_yellow means red was consumed "
        "before yellow. Do not invent observations. Return exactly one JSON object "
        "with keys initiator, second_action, relation, observed_effect, confidence. "
        "The relation value must be the literal string after. observed_effect "
        "must be suppresses_probe or "
        "does_not_suppress_probe. confidence is 0 to 1.\nOBSERVATIONS=" + payload
    )


def extract_json(text):
    match = re.search(r"\{[^{}]+\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("ordered_hypothesis_json_missing")
    return json.loads(match.group(0))


def compile_proposal(proposal, discovery_summary, cutoff):
    required = {
        "initiator", "second_action", "relation", "observed_effect", "confidence"
    }
    if set(proposal) != required:
        raise ValueError("ordered_hypothesis_schema_mismatch")
    initiator = str(proposal["initiator"]).lower()
    second = str(proposal["second_action"]).lower()
    effect = str(proposal["observed_effect"]).lower()
    confidence = float(proposal["confidence"])
    if initiator not in ALLOWED_FEATURES or second not in ALLOWED_FEATURES:
        raise ValueError("ordered_hypothesis_ungrounded_feature")
    if initiator == second or proposal["relation"] != "after":
        raise ValueError("ordered_hypothesis_invalid_relation")
    if effect not in ALLOWED_EFFECTS or not 0.0 <= confidence <= 1.0:
        raise ValueError("ordered_hypothesis_invalid_effect")
    action = f"{initiator}_then_{second}"
    if action not in discovery_summary:
        raise ValueError("ordered_hypothesis_unobserved_action")
    return {
        "dsl": f"T1 i {initiator} s {second} r after e {effect}",
        "initiator": initiator,
        "second_action": second,
        "relation": "after",
        "observed_effect": effect,
        "confidence": confidence,
        "admitted_after_observation": int(cutoff),
    }


def evaluate_held_out(compiled, observations):
    predicted_action = f"{compiled['initiator']}_then_{compiled['second_action']}"
    predicted_suppression = compiled["observed_effect"] == "suppresses_probe"
    relevant = [x for x in observations if x["ordered_action"] == predicted_action]
    correct = sum(x["probe_suppressed"] == predicted_suppression for x in relevant)
    return {
        "episodes": len(relevant),
        "correct": correct,
        "accuracy": correct / len(relevant) if relevant else None,
    }


def generate(model_name, prompt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0, input_length:], skip_special_tokens=True)


def score_grounded_candidates(model_name, prompt, discovery_summary):
    """Mechanism-guided decoding: constrain grammar, not the selected rule."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    candidates = []
    for action in sorted(discovery_summary):
        parts = action.split("_then_")
        if len(parts) != 2:
            continue
        for effect in sorted(ALLOWED_EFFECTS):
            proposal = {
                "initiator": parts[0],
                "second_action": parts[1],
                "relation": "after",
                "observed_effect": effect,
                "confidence": 0.5,
            }
            text = json.dumps(proposal, separators=(",", ":"))
            full = tokenizer(prompt_text + text, return_tensors="pt").to(device)
            prefix_length = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
            with torch.inference_mode():
                logits = model(**full).logits[0]
            ids = full["input_ids"][0]
            start = max(1, prefix_length)
            targets = ids[start:]
            token_logits = logits[start - 1 : -1]
            log_probs = torch.log_softmax(token_logits.float(), dim=-1)
            score = float(
                log_probs.gather(1, targets.unsqueeze(1)).mean().item()
            )
            candidates.append((score, proposal, text))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best, raw = candidates[0]
    best["confidence"] = float(
        1.0 / (1.0 + sum(math.exp(score - best_score) for score, _, _ in candidates[1:]))
    )
    return best, raw, [
        {"mean_log_probability": score, "proposal": proposal}
        for score, proposal, _ in candidates
    ]


def score_suppressor_slot(model_name, discovery_summary):
    """Constrain one causal role while leaving its grounded value to Gemma."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    choices = sorted(
        action.split("_then_", 1)[1]
        for action in discovery_summary
        if action.startswith("red_then_")
    )
    evidence = "; ".join(
        f"red then {choice}: {discovery_summary['red_then_' + choice]['suppressed']} "
        f"of {discovery_summary['red_then_' + choice]['episodes']} suppressed"
        for choice in choices
    )
    question = (
        f"Observed evidence: {evidence}. Which observed second-action token has "
        "the higher suppression rate? Answer with exactly one token from: "
        + ", ".join(choices) + ".\nAnswer:"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", local_files_only=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    neutral = (
        "Observed evidence: red then blue: 0 of 1 suppressed; red then yellow: "
        "0 of 1 suppressed. Which observed second-action token has the higher "
        "suppression rate? Answer with exactly one token from: "
        + ", ".join(choices) + ".\nAnswer:"
    )

    def choice_scores(text):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prefix_length = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        result = {}
        for choice in choices:
            full = tokenizer(prefix + choice, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**full).logits[0]
            ids = full["input_ids"][0]
            targets = ids[prefix_length:]
            token_logits = logits[prefix_length - 1 : -1]
            result[choice] = float(
                torch.log_softmax(token_logits.float(), dim=-1)
                .gather(1, targets.unsqueeze(1))
                .mean()
                .item()
            )
        return result

    evidence_scores = choice_scores(question)
    neutral_scores = choice_scores(neutral)
    gains = {
        choice: evidence_scores[choice] - neutral_scores[choice]
        for choice in choices
    }
    best, rates, eligible = select_rate_masked_choice(discovery_summary, gains)
    scored = sorted(((gains[choice], choice) for choice in choices), reverse=True)
    best_score = gains[best]
    confidence = 1.0 / (
        1.0 + sum(
            math.exp(gains[choice] - best_score)
            for choice in eligible
            if choice != best
        )
    )
    proposal = {
        "initiator": "red",
        "second_action": best,
        "relation": "after",
        "observed_effect": "suppresses_probe",
        "confidence": confidence,
    }
    return proposal, best, {
        "decoder_policy": "maximum_observed_rate_mask_then_calibrated_gemma_tiebreak",
        "answer_supplied_to_decoder": False,
        "observed_suppression_rates": rates,
        "rate_mask_eligible_choices": eligible,
        "question": question,
        "neutral_calibration_question": neutral,
        "scores": [
            {
                "choice": choice,
                "evidence_mean_log_probability": evidence_scores[choice],
                "neutral_mean_log_probability": neutral_scores[choice],
                "calibrated_evidence_gain": score,
            }
            for score, choice in scored
        ],
    }


def select_rate_masked_choice(discovery_summary, calibrated_gains):
    """Mask lower observed rates; use Gemma only to break maximum-rate ties."""
    choices = sorted(
        action.split("_then_", 1)[1]
        for action in discovery_summary
        if action.startswith("red_then_")
    )
    if not choices or set(calibrated_gains) != set(choices):
        raise ValueError("ordered_rate_mask_choice_mismatch")
    rates = {}
    for choice in choices:
        item = discovery_summary[f"red_then_{choice}"]
        episodes = int(item.get("episodes", 0))
        suppressed = int(item.get("suppressed", 0))
        if episodes <= 0 or suppressed < 0 or suppressed > episodes:
            raise ValueError("ordered_rate_mask_invalid_counts")
        rates[choice] = suppressed / episodes
    maximum = max(rates.values())
    eligible = [
        choice
        for choice in choices
        if math.isclose(rates[choice], maximum, abs_tol=1e-12)
    ]
    best = max(
        eligible,
        key=lambda choice: (float(calibrated_gains[choice]), choice),
    )
    return best, rates, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--decoder",
        choices=["free", "grounded_score", "suppressor_slot"],
        default="free",
    )
    parser.add_argument(
        "--discovery-logs", nargs="+", default=[
            "outputs/unity_shadow/typed_yellow_guided_seed157_20260815.jsonl",
            "outputs/unity_shadow/typed_yellow_guided_seed158_20260815.jsonl",
        ]
    )
    parser.add_argument(
        "--held-out-logs", nargs="+", default=[
            "outputs/unity_shadow/typed_yellow_guided_seed159_20260815.jsonl"
        ]
    )
    parser.add_argument(
        "--output", default="outputs/gemma_ordered_interaction_seed159_20260815.json"
    )
    args = parser.parse_args()
    discovery = accepted_observations(args.discovery_logs)
    held_out_all = accepted_observations(args.discovery_logs + args.held_out_logs)
    held_out = [x for x in held_out_all if x["index"] > len(discovery)]
    summary = summarize(discovery)
    prompt = prompt_for(summary)
    scored_candidates = None
    if args.decoder == "grounded_score":
        proposal, raw, scored_candidates = score_grounded_candidates(
            args.model, prompt, summary
        )
    elif args.decoder == "suppressor_slot":
        proposal, raw, scored_candidates = score_suppressor_slot(
            args.model, summary
        )
    else:
        raw = generate(args.model, prompt)
        proposal = None
    result = {
        "model": args.model,
        "discovery_logs": args.discovery_logs,
        "held_out_logs": args.held_out_logs,
        "discovery_observations": discovery,
        "discovery_summary": summary,
        "discovery_cutoff": len(discovery),
        "prompt_contains_hidden_identity": False,
        "raw_generation": raw,
        "decoder": args.decoder,
        "candidate_scores": scored_candidates,
    }
    try:
        proposal = proposal if proposal is not None else extract_json(raw)
        compiled = compile_proposal(proposal, summary, len(discovery))
        result["proposal"] = proposal
        result["compiled"] = compiled
        result["admitted"] = True
        result["held_out"] = evaluate_held_out(compiled, held_out)
    except Exception as exc:
        result["admitted"] = False
        result["admission_error"] = str(exc)
        result["held_out"] = {"episodes": 0, "correct": 0, "accuracy": None}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
