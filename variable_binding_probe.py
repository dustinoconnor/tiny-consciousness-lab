#!/usr/bin/env python3
"""Mechanistic probe for frozen Gemma causal-role binding.

The primary endpoint is leave-one-feature-pair-out decoding of whether the
correct cause appears in evidence row zero or row one. Attention ranking is
exploratory until a ranked head is causally ablated on the held-out half.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from causal_dsl_lora import (
    CausalExample,
    build_curriculum,
    prompt_for,
    target_for,
)


DEFAULT_ARTIFACT = "outputs/causal_dsl_frozen_comparison_1b_20260804.json"
DEFAULT_ADAPTER = "checkpoints/causal_dsl_l1_lora_1b_20260804"
DEFAULT_MODEL = "google/gemma-3-1b-it"


@dataclass(frozen=True)
class ProbeCase:
    index: int
    example: CausalExample
    generation_correct: bool
    syntax_valid: bool
    raw_generation: str


@dataclass(frozen=True)
class TokenizedProbeCase:
    input_ids: tuple[int, ...]
    cause_prediction_position: int
    comparison_prediction_position: int
    cause_token_id: int
    comparison_token_id: int
    cause_evidence_positions: tuple[int, ...]
    comparison_evidence_positions: tuple[int, ...]
    target_row: int


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(example):
    return "/".join(sorted((example.target, example.comparison)))


def evidence_features(example):
    features = list(example.summary["feature_outcomes"])
    if example.evidence_order == "reversed":
        features.reverse()
    return features


def target_row(example):
    return evidence_features(example).index(example.target)


def load_frozen_cases(path=DEFAULT_ARTIFACT):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    seed = int(artifact["evaluation_seed"])
    size = int(artifact["heldout_examples"])
    _train, examples = build_curriculum(seed=seed, train_size=1, heldout_size=size)
    records = artifact["heldout"]["l1"]["records"]
    if len(records) != len(examples):
        raise ValueError("probe_artifact_example_count_mismatch")
    cases = []
    for index, (example, record) in enumerate(zip(examples, records, strict=True)):
        expected = target_for(example, "l1")
        if record["expected"] != expected:
            raise ValueError(f"probe_artifact_curriculum_mismatch:{index}")
        cases.append(
            ProbeCase(
                index=index,
                example=example,
                generation_correct=bool(record["semantic_match"]),
                syntax_valid=bool(record["syntax_valid"]),
                raw_generation=str(record["raw"]),
            )
        )
    return artifact, cases


def select_balanced_cases(cases, maximum=32):
    """Retain every frozen error, then greedily balance success controls."""
    failures = [case for case in cases if not case.generation_correct]
    if len(failures) > maximum:
        raise ValueError("maximum_smaller_than_frozen_failure_count")
    selected = list(failures)
    remaining = [case for case in cases if case.generation_correct]

    def counts(key):
        result = {}
        for case in selected:
            value = key(case)
            result[value] = result.get(value, 0) + 1
        return result

    while remaining and len(selected) < maximum:
        pair_counts = counts(lambda case: pair_key(case.example))
        row_counts = counts(lambda case: target_row(case.example))
        order_counts = counts(lambda case: case.example.evidence_order)
        effect_counts = counts(lambda case: case.example.effect)
        remaining.sort(
            key=lambda case: (
                pair_counts.get(pair_key(case.example), 0),
                row_counts.get(target_row(case.example), 0),
                order_counts.get(case.example.evidence_order, 0),
                effect_counts.get(case.example.effect, 0),
                case.index,
            )
        )
        selected.append(remaining.pop(0))
    return sorted(selected, key=lambda case: case.index)


def _offsets_and_ids(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    return list(input_ids), [tuple(pair) for pair in offsets]


def _tokens_overlapping(offsets, start, end):
    result = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not result:
        raise ValueError(f"probe_character_span_has_no_token:{start}:{end}")
    return tuple(result)


def tokenize_probe_case(tokenizer, example):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_for(example, "l1")}],
        add_generation_prompt=True,
        tokenize=False,
    )
    answer = target_for(example, "l1")
    full_text = prompt + answer
    input_ids, offsets = _offsets_and_ids(tokenizer, full_text)

    cause_answer_start = answer.index(f"c {example.target}") + 2
    comparison_answer_start = answer.index(f"k {example.comparison}") + 2
    cause_char = len(prompt) + cause_answer_start
    comparison_char = len(prompt) + comparison_answer_start
    cause_tokens = _tokens_overlapping(
        offsets, cause_char, cause_char + len(example.target)
    )
    comparison_tokens = _tokens_overlapping(
        offsets, comparison_char, comparison_char + len(example.comparison)
    )
    if cause_tokens[0] == 0 or comparison_tokens[0] == 0:
        raise ValueError("probe_role_token_has_no_prediction_position")

    cause_evidence_char = prompt.rfind(example.target)
    comparison_evidence_char = prompt.rfind(example.comparison)
    if cause_evidence_char < 0 or comparison_evidence_char < 0:
        raise ValueError("probe_evidence_feature_not_found")
    cause_evidence = _tokens_overlapping(
        offsets,
        cause_evidence_char,
        cause_evidence_char + len(example.target),
    )
    comparison_evidence = _tokens_overlapping(
        offsets,
        comparison_evidence_char,
        comparison_evidence_char + len(example.comparison),
    )
    return TokenizedProbeCase(
        input_ids=tuple(int(value) for value in input_ids),
        cause_prediction_position=cause_tokens[0] - 1,
        comparison_prediction_position=comparison_tokens[0] - 1,
        cause_token_id=int(input_ids[cause_tokens[0]]),
        comparison_token_id=int(input_ids[comparison_tokens[0]]),
        cause_evidence_positions=cause_evidence,
        comparison_evidence_positions=comparison_evidence,
        target_row=target_row(example),
    )


def _ridge_predictions(train_x, train_y, test_x, ridge):
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    labels = train_y.astype(np.float64) * 2.0 - 1.0
    # Dual ridge is stable when hidden width is much larger than case count.
    kernel = train @ train.T
    alpha = np.linalg.solve(
        kernel + ridge * np.eye(kernel.shape[0], dtype=np.float64), labels
    )
    scores = test @ train.T @ alpha
    return (scores >= 0.0).astype(np.int64), scores


def leave_group_out_curve(activations, labels, groups, ridge=1.0):
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if activations.ndim != 3:
        raise ValueError("probe_activations_must_be_cases_layers_hidden")
    curve = []
    predictions = np.zeros((len(labels), activations.shape[1]), dtype=np.int64)
    scores = np.zeros((len(labels), activations.shape[1]), dtype=np.float64)
    for layer in range(activations.shape[1]):
        for group in sorted(set(groups.tolist())):
            test_mask = groups == group
            train_mask = ~test_mask
            prediction, score = _ridge_predictions(
                activations[train_mask, layer],
                labels[train_mask],
                activations[test_mask, layer],
                ridge,
            )
            predictions[test_mask, layer] = prediction
            scores[test_mask, layer] = score
        curve.append(float((predictions[:, layer] == labels).mean()))
    return {
        "accuracy_by_layer": curve,
        "predictions": predictions,
        "scores": scores,
    }


def permutation_peak_pvalue(
    activations, labels, groups, observed_peak, repeats=100, seed=2505, ridge=1.0
):
    rng = random.Random(seed)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    null_peaks = []
    for _ in range(repeats):
        shuffled = labels.copy()
        for group in sorted(set(groups.tolist())):
            indices = np.flatnonzero(groups == group).tolist()
            values = shuffled[indices].tolist()
            rng.shuffle(values)
            shuffled[indices] = values
        curve = leave_group_out_curve(
            activations, shuffled, groups, ridge=ridge
        )["accuracy_by_layer"]
        null_peaks.append(max(curve))
    exceedances = sum(value >= observed_peak for value in null_peaks)
    return {
        "repeats": repeats,
        "pvalue_max_layer_corrected": (exceedances + 1) / (repeats + 1),
        "null_peak_mean": float(np.mean(null_peaks)),
        "null_peak_max": float(np.max(null_peaks)),
    }


def binary_auc(scores, positive):
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    pos = scores[positive]
    neg = scores[~positive]
    if not len(pos) or not len(neg):
        return None
    wins = 0.0
    for left in pos:
        for right in neg:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / (len(pos) * len(neg))


def rank_attention_heads(head_scores, case_indices):
    head_scores = np.asarray(head_scores, dtype=np.float64)
    discovery = np.asarray([index % 2 == 0 for index in case_indices])
    if not discovery.any() or discovery.all():
        raise ValueError("probe_requires_discovery_and_confirmation_cases")
    means = head_scores[discovery].mean(axis=0)
    top_flat = int(np.argmax(means))
    top_layer, top_head = np.unravel_index(top_flat, means.shape)
    same_layer = np.abs(means[top_layer])
    control_head = int(np.argmin(same_layer))
    return {
        "split_rule": "even_original_index_discovery_odd_confirmation",
        "discovery_cases": int(discovery.sum()),
        "confirmation_cases": int((~discovery).sum()),
        "top_binding_head": {
            "layer": int(top_layer),
            "head": int(top_head),
            "mean_target_minus_control_attention": float(means[top_layer, top_head]),
        },
        "same_layer_low_signal_control": {
            "layer": int(top_layer),
            "head": control_head,
            "mean_target_minus_control_attention": float(means[top_layer, control_head]),
        },
        "mean_scores": means.tolist(),
    }


@contextlib.contextmanager
def ablate_attention_head(o_projection, head, head_dim):
    """Zero one pre-output-projection head slice; used only on confirmation."""
    start = int(head) * int(head_dim)
    stop = start + int(head_dim)

    def hook(_module, arguments):
        values = arguments[0].clone()
        if stop > values.shape[-1]:
            raise ValueError("probe_head_slice_exceeds_projection_input")
        values[..., start:stop] = 0
        return (values, *arguments[1:])

    handle = o_projection.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def collect_probe(model, tokenizer, cases, device, include_attentions=True):
    import torch

    activations = []
    comparison_activations = []
    margins = []
    comparison_margins = []
    attention_scores = []
    comparison_attention_scores = []
    model.eval()
    for number, case in enumerate(cases, start=1):
        tokenized = tokenize_probe_case(tokenizer, case.example)
        input_ids = torch.tensor(
            [tokenized.input_ids], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=include_attentions,
                use_cache=False,
                return_dict=True,
            )
        activations.append(
            np.stack(
                [
                    state[0, tokenized.cause_prediction_position]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for state in output.hidden_states
                ]
            )
        )
        comparison_activations.append(
            np.stack(
                [
                    state[0, tokenized.comparison_prediction_position]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for state in output.hidden_states
                ]
            )
        )
        logits = output.logits[0, tokenized.cause_prediction_position]
        margins.append(
            float(
                (
                    logits[tokenized.cause_token_id]
                    - logits[tokenized.comparison_token_id]
                )
                .detach()
                .cpu()
            )
        )
        comparison_logits = output.logits[
            0, tokenized.comparison_prediction_position
        ]
        comparison_margins.append(
            float(
                (
                    comparison_logits[tokenized.comparison_token_id]
                    - comparison_logits[tokenized.cause_token_id]
                )
                .detach()
                .cpu()
            )
        )
        if include_attentions:
            per_layer = []
            comparison_per_layer = []
            for attention in output.attentions:
                query = attention[0, :, tokenized.cause_prediction_position]
                cause_mass = query[:, list(tokenized.cause_evidence_positions)].sum(-1)
                control_mass = query[:, list(tokenized.comparison_evidence_positions)].sum(-1)
                per_layer.append(
                    (cause_mass - control_mass).detach().float().cpu().numpy()
                )
                comparison_query = attention[
                    0, :, tokenized.comparison_prediction_position
                ]
                comparison_mass = comparison_query[
                    :, list(tokenized.comparison_evidence_positions)
                ].sum(-1)
                cause_mass_at_comparison = comparison_query[
                    :, list(tokenized.cause_evidence_positions)
                ].sum(-1)
                comparison_per_layer.append(
                    (comparison_mass - cause_mass_at_comparison)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            attention_scores.append(np.stack(per_layer))
            comparison_attention_scores.append(np.stack(comparison_per_layer))
        print(json.dumps({"probe_case": number, "total": len(cases)}), flush=True)
        del output
    return {
        "activations": np.stack(activations),
        "comparison_activations": np.stack(comparison_activations),
        "logit_margins": np.asarray(margins),
        "comparison_logit_margins": np.asarray(comparison_margins),
        "attention_scores": (
            np.stack(attention_scores) if include_attentions else None
        ),
        "comparison_attention_scores": (
            np.stack(comparison_attention_scores) if include_attentions else None
        ),
    }


def find_attention_output_projection(model, layer):
    suffix = f"layers.{int(layer)}.self_attn.o_proj"
    matches = [module for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"probe_expected_one_attention_output_projection:{layer}:{len(matches)}"
        )
    return matches[0]


def collect_role_logit_margins(model, tokenizer, cases, device, role="cause"):
    import torch

    margins = []
    model.eval()
    for case in cases:
        tokenized = tokenize_probe_case(tokenizer, case.example)
        input_ids = torch.tensor(
            [tokenized.input_ids], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
        if role == "cause":
            position = tokenized.cause_prediction_position
            correct_id = tokenized.cause_token_id
            swapped_id = tokenized.comparison_token_id
        elif role == "comparison":
            position = tokenized.comparison_prediction_position
            correct_id = tokenized.comparison_token_id
            swapped_id = tokenized.cause_token_id
        else:
            raise ValueError(f"unsupported_probe_role:{role}")
        logits = output.logits[0, position]
        margins.append(
            float(
                (logits[correct_id] - logits[swapped_id])
                .detach()
                .cpu()
            )
        )
        del output
    return np.asarray(margins, dtype=np.float64)


def confirm_ranked_head_ablation(
    model,
    tokenizer,
    cases,
    device,
    ranking,
    baseline_margins,
    head_dim,
    role="cause",
):
    """Causally compare the discovery-ranked head with its fixed control."""
    confirmation_mask = np.asarray([case.index % 2 == 1 for case in cases])
    confirmation_cases = [
        case for case, keep in zip(cases, confirmation_mask, strict=True) if keep
    ]
    baseline = np.asarray(baseline_margins)[confirmation_mask]
    top = ranking["top_binding_head"]
    control = ranking["same_layer_low_signal_control"]
    top_projection = find_attention_output_projection(model, top["layer"])
    with ablate_attention_head(top_projection, top["head"], head_dim):
        top_margins = collect_role_logit_margins(
            model, tokenizer, confirmation_cases, device, role=role
        )
    control_projection = find_attention_output_projection(model, control["layer"])
    with ablate_attention_head(control_projection, control["head"], head_dim):
        control_margins = collect_role_logit_margins(
            model, tokenizer, confirmation_cases, device, role=role
        )
    top_drop = float(np.mean(baseline - top_margins))
    control_drop = float(np.mean(baseline - control_margins))
    passed = top_drop > 0.0 and top_drop > control_drop
    return {
        "role": role,
        "endpoint": f"mean_correct_vs_swapped_first_{role}_token_logit_margin",
        "confirmation_cases": len(confirmation_cases),
        "baseline_mean_margin": float(np.mean(baseline)),
        "top_head_ablated_mean_margin": float(np.mean(top_margins)),
        "same_layer_control_ablated_mean_margin": float(np.mean(control_margins)),
        "top_head_margin_drop": top_drop,
        "control_head_margin_drop": control_drop,
        "registered_gate_passed": passed,
        "claim_if_passed": (
            "ranked_head_contributes_to_first_token_causal_role_binding; "
            "not_a_complete_variable_binding_circuit"
        ),
    }


def analyze_collected(collected, cases, permutations=100, ridge=1.0):
    labels = np.asarray([target_row(case.example) for case in cases])
    groups = np.asarray([pair_key(case.example) for case in cases])
    curve = leave_group_out_curve(
        collected["activations"], labels, groups, ridge=ridge
    )
    observed_peak = max(curve["accuracy_by_layer"])
    peak_layer = int(np.argmax(curve["accuracy_by_layer"]))
    failures = np.asarray([not case.generation_correct for case in cases])
    result = {
        "primary_layer_probe": {
            "label": "correct_cause_evidence_row_0_or_1",
            "cross_validation": "leave_one_feature_pair_out",
            "ridge": ridge,
            "accuracy_by_layer": curve["accuracy_by_layer"],
            "peak_layer": peak_layer,
            "peak_accuracy": observed_peak,
            "permutation": permutation_peak_pvalue(
                collected["activations"],
                labels,
                groups,
                observed_peak,
                repeats=permutations,
                ridge=ridge,
            ),
        },
        "generation_failure_diagnostic": {
            "failure_cases": int(failures.sum()),
            "success_cases": int((~failures).sum()),
            "auc_negative_correct_vs_swapped_logit_margin": binary_auc(
                -collected["logit_margins"], failures
            ),
            "mean_margin_failures": float(collected["logit_margins"][failures].mean()),
            "mean_margin_successes": float(collected["logit_margins"][~failures].mean()),
        },
    }
    if collected["attention_scores"] is not None:
        result["attention_head_ranking"] = rank_attention_heads(
            collected["attention_scores"], [case.index for case in cases]
        )
        result["attention_head_ranking"]["claim_status"] = (
            "exploratory_until_top_vs_same_layer_control_ablation_on_confirmation"
        )
    comparison_labels = 1 - labels
    comparison_curve = leave_group_out_curve(
        collected["comparison_activations"],
        comparison_labels,
        groups,
        ridge=ridge,
    )
    comparison_peak = max(comparison_curve["accuracy_by_layer"])
    result["post_registered_comparison_role_audit"] = {
        "status": "secondary_added_after_primary_cause_probe_result",
        "accuracy_by_layer": comparison_curve["accuracy_by_layer"],
        "peak_layer": int(np.argmax(comparison_curve["accuracy_by_layer"])),
        "peak_accuracy": comparison_peak,
        "permutation": permutation_peak_pvalue(
            collected["comparison_activations"],
            comparison_labels,
            groups,
            comparison_peak,
            repeats=permutations,
            ridge=ridge,
        ),
        "failure_auc_negative_correct_vs_swapped_margin": binary_auc(
            -collected["comparison_logit_margins"], failures
        ),
        "mean_margin_failures": float(
            collected["comparison_logit_margins"][failures].mean()
        ),
        "mean_margin_successes": float(
            collected["comparison_logit_margins"][~failures].mean()
        ),
    }
    if collected["comparison_attention_scores"] is not None:
        result["post_registered_comparison_role_audit"]["attention_head_ranking"] = (
            rank_attention_heads(
                collected["comparison_attention_scores"],
                [case.index for case in cases],
            )
        )
    return result


def manifest(artifact_path, cases, maximum):
    return {
        "experiment": "frozen_gemma_1b_variable_binding_probe_v1",
        "paper_motivation": "Wu_Geiger_Milliere_ICML_2025",
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "selection": {
            "maximum_cases": maximum,
            "selected_cases": len(cases),
            "frozen_failures_retained": sum(not case.generation_correct for case in cases),
            "case_indices": [case.index for case in cases],
            "pair_counts": {
                pair: sum(pair_key(case.example) == pair for case in cases)
                for pair in sorted({pair_key(case.example) for case in cases})
            },
            "target_row_counts": {
                str(row): sum(target_row(case.example) == row for case in cases)
                for row in (0, 1)
            },
        },
        "primary_endpoint": (
            "max-layer leave-one-feature-pair-out cause-row decoding accuracy "
            "with within-pair permutation max-layer correction"
        ),
        "attention_split": "even original index discovery; odd index confirmation",
        "causal_claim_gate": (
            "top ranked head must reduce frozen confirmation performance more "
            "than the preregistered same-layer low-signal control when ablated"
        ),
        "training_or_adapter_changes_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--maximum-cases", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output", default="outputs/variable_binding_probe_1b_20260805.json"
    )
    parser.add_argument(
        "--activations",
        default="outputs/variable_binding_probe_1b_20260805.npz",
    )
    args = parser.parse_args()

    _artifact, frozen = load_frozen_cases(args.artifact)
    cases = select_balanced_cases(frozen, args.maximum_cases)
    preregistration = manifest(args.artifact, cases, args.maximum_cases)
    if args.dry_run:
        print(json.dumps(preregistration, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    collected = collect_probe(model, tokenizer, cases, device)
    analysis = analyze_collected(
        collected, cases, permutations=args.permutations, ridge=args.ridge
    )
    if "attention_head_ranking" in analysis:
        head_dim = int(getattr(model.config, "head_dim"))
        analysis["confirmation_ablation"] = confirm_ranked_head_ablation(
            model,
            tokenizer,
            cases,
            device,
            analysis["attention_head_ranking"],
            collected["logit_margins"],
            head_dim,
        )
        comparison = analysis["post_registered_comparison_role_audit"]
        if "attention_head_ranking" in comparison:
            comparison["confirmation_ablation"] = confirm_ranked_head_ablation(
                model,
                tokenizer,
                cases,
                device,
                comparison["attention_head_ranking"],
                collected["comparison_logit_margins"],
                head_dim,
                role="comparison",
            )
    result = {
        **preregistration,
        "model": args.model,
        "adapter": args.adapter,
        "device": device,
        "analysis": analysis,
    }
    activation_path = Path(args.activations)
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        activation_path,
        activations=collected["activations"],
        comparison_activations=collected["comparison_activations"],
        logit_margins=collected["logit_margins"],
        comparison_logit_margins=collected["comparison_logit_margins"],
        attention_scores=collected["attention_scores"],
        comparison_attention_scores=collected["comparison_attention_scores"],
        case_indices=np.asarray([case.index for case in cases]),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
