#!/usr/bin/env python3
"""Passive GNW selection and local-LM hypotheses from embodied telemetry."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MODEL = "google/gemma-3-270m-it"
REQUIRED_HYPOTHESIS_FIELDS = {
    "condition",
    "proposed_cause",
    "predicted_effect",
    "delay_seconds",
    "comparison",
    "falsifier",
    "confidence",
}
REQUIRED_BOUND_FIELDS = {
    "target_cause",
    "comparison_feature",
    "observed_effect",
    "latency_seconds",
    "confidence",
}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class MetabolicEpisode:
    feature: str
    pickup_time: float
    pressure_before: float
    peak_pressure: float
    pressure_delta: float
    peak_delay_seconds: float
    intervening_pickup: bool


@dataclass(frozen=True)
class ScientificBroadcast:
    feature: str
    salience: float
    prediction_error: float
    novelty: float
    episode: MetabolicEpisode


class PassiveScientificGNW:
    """Select anomalous episodes for science without influencing behavior."""

    def __init__(self, threshold=0.58, refractory_episodes=1):
        self.threshold = clamp(threshold)
        self.refractory_episodes = max(0, int(refractory_episodes))
        self.refractory = 0
        self.feature_counts = Counter()
        self.observations = 0
        self.ignitions = 0
        self.held = 0
        self.action_influence = 0

    def observe(self, episode):
        self.observations += 1
        prior_count = self.feature_counts[episode.feature]
        self.feature_counts[episode.feature] += 1
        novelty = 1.0 / math.sqrt(prior_count + 1.0)
        prediction_error = clamp(max(0.0, episode.pressure_delta) / 0.34)
        isolation = 0.45 if episode.intervening_pickup else 1.0
        salience = clamp(
            isolation * (0.68 * prediction_error + 0.32 * novelty)
        )
        if self.refractory > 0:
            self.refractory -= 1
            self.held += 1
            return None
        if salience < self.threshold:
            self.held += 1
            return None
        self.ignitions += 1
        self.refractory = self.refractory_episodes
        return ScientificBroadcast(
            feature=episode.feature,
            salience=salience,
            prediction_error=prediction_error,
            novelty=novelty,
            episode=episode,
        )


def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metabolic_episodes(rows, min_delay=7.0, max_delay=14.0):
    pickups = []
    previous_total = None
    previous_red_total = None
    for index, row in enumerate(rows):
        total = int(row.get("mushroom_pickups_total", 0) or 0)
        has_red_counter = "red_mushroom_pickups_total" in row
        red_total = int(row.get("red_mushroom_pickups_total", 0) or 0)
        if previous_total is None:
            previous_total = total
            previous_red_total = red_total if has_red_counter else None
            continue
        if total > previous_total:
            feature = str(row.get("mushroom_feature", "unknown") or "unknown")
            pickup_delta = total - previous_total
            if has_red_counter and previous_red_total is not None:
                red_delta = min(
                    pickup_delta,
                    max(0, red_total - previous_red_total),
                )
                pickups.extend((index, "red") for _ in range(red_delta))
                pickups.extend(
                    (index, "blue") for _ in range(pickup_delta - red_delta)
                )
            else:
                pickups.extend((index, feature) for _ in range(pickup_delta))
        previous_total = total
        if has_red_counter:
            previous_red_total = red_total

    episodes = []
    for pickup_number, (index, feature) in enumerate(pickups):
        row = rows[index]
        pickup_time = float(row["time"])
        pressure_before = float(row.get("metabolic_pressure", 0.0) or 0.0)
        window = []
        for candidate in rows[index:]:
            delay = float(candidate["time"]) - pickup_time
            if delay > max_delay:
                break
            if delay >= min_delay:
                window.append(candidate)
        if not window:
            continue
        peak = max(
            window,
            key=lambda candidate: float(
                candidate.get("metabolic_pressure", 0.0) or 0.0
            ),
        )
        peak_pressure = float(peak.get("metabolic_pressure", 0.0) or 0.0)
        next_pickup_time = (
            float(rows[pickups[pickup_number + 1][0]]["time"])
            if pickup_number + 1 < len(pickups)
            else math.inf
        )
        episodes.append(
            MetabolicEpisode(
                feature=feature,
                pickup_time=pickup_time,
                pressure_before=pressure_before,
                peak_pressure=peak_pressure,
                pressure_delta=peak_pressure - pressure_before,
                peak_delay_seconds=float(peak["time"]) - pickup_time,
                intervening_pickup=next_pickup_time <= pickup_time + max_delay,
            )
        )
    return episodes


def evidence_summary(episodes, broadcasts):
    features = {}
    for feature in sorted({episode.feature for episode in episodes}):
        selected = [episode for episode in episodes if episode.feature == feature]
        isolated = [episode for episode in selected if not episode.intervening_pickup]
        analysis_set = isolated or selected
        deltas = [episode.pressure_delta for episode in analysis_set]
        delays = [
            episode.peak_delay_seconds
            for episode in analysis_set
            if episode.pressure_delta > 0.10
        ]
        features[feature] = {
            "episodes": len(selected),
            "isolated_episodes": len(isolated),
            "mean_pressure_delta": round(statistics.mean(deltas), 4),
            "positive_pressure_fraction": round(
                sum(delta > 0.10 for delta in deltas) / max(len(deltas), 1), 4
            ),
            "mean_positive_delay_seconds": (
                round(statistics.mean(delays), 3) if delays else None
            ),
        }
    return {
        "feature_outcomes": features,
        "episode_count": len(episodes),
        "gnw_broadcast_count": len(broadcasts),
        "gnw_broadcast_features": dict(
            Counter(broadcast.feature for broadcast in broadcasts)
        ),
        "gnw_action_influence": 0,
    }


def hypothesis_prompt(summary, evidence_interface="original"):
    if evidence_interface not in {"original", "homeostatic_filtered"}:
        raise ValueError(f"unsupported_evidence_interface:{evidence_interface}")
    feature_lines = []
    for feature, outcomes in summary.get("feature_outcomes", {}).items():
        feature_lines.append(
            f"- observed feature {feature}: {outcomes.get('isolated_episodes', 0)} "
            f"isolated episodes; mean internal-pressure change "
            f"{outcomes.get('mean_pressure_delta')}; positive-pressure fraction "
            f"{outcomes.get('positive_pressure_fraction')}; mean delay after "
            f"pickup {outcomes.get('mean_positive_delay_seconds')} seconds."
        )
    evidence = "\n".join(feature_lines)
    if evidence_interface == "original":
        broadcast_lines = ", ".join(
            f"{feature}:{count}"
            for feature, count in summary.get("gnw_broadcast_features", {}).items()
        ) or "none"
        evidence += (
            f"\nPassive GNW surprise broadcasts by feature: {broadcast_lines}."
        )
    else:
        evidence = (
            "Homeostatic relevance gate selected this question without "
            "selecting its answer: which observable pickup feature predicts "
            "a delayed increase in internal pressure? Compare every observed "
            "feature below.\n" + evidence
        )
    return (
        "You are a tiny embodied scientist. Infer one falsifiable causal "
        "hypothesis from the observations below. Color words are observable "
        "features, not semantic labels. Correlation is not verification. "
        "The condition and proposed cause must name an observed feature. "
        "The comparison must name a different observed feature. The predicted "
        "effect must state the direction of change in internal pressure. Use "
        "the observed delay, and make the falsifier describe a held-out result "
        "that would contradict the prediction. Aggregate field names such as "
        "episode_count are evidence, never causes. "
        "Return exactly one JSON object with keys condition, proposed_cause, "
        "predicted_effect, delay_seconds, comparison, falsifier, confidence. "
        "confidence must be between 0 and 1. Do not include markdown.\n"
        + evidence
    )


def bound_schema_prompt(
    summary,
    evidence_interface="homeostatic_filtered",
    evidence_order="canonical",
):
    if evidence_interface not in {"original", "homeostatic_filtered"}:
        raise ValueError(f"unsupported_evidence_interface:{evidence_interface}")
    if evidence_order not in {"canonical", "reversed"}:
        raise ValueError(f"unsupported_evidence_order:{evidence_order}")
    feature_lines = []
    feature_items = list(summary.get("feature_outcomes", {}).items())
    if evidence_order == "reversed":
        feature_items.reverse()
    for feature, outcomes in feature_items:
        feature_lines.append(
            f"- observed feature {feature}: {outcomes.get('isolated_episodes', 0)} "
            f"isolated episodes; mean internal-pressure change "
            f"{outcomes.get('mean_pressure_delta')}; positive-pressure fraction "
            f"{outcomes.get('positive_pressure_fraction')}; mean delay after "
            f"pickup {outcomes.get('mean_positive_delay_seconds')} seconds."
        )
    evidence = "\n".join(feature_lines)
    if evidence_interface == "original":
        broadcast_lines = ", ".join(
            f"{feature}:{count}"
            for feature, count in summary.get("gnw_broadcast_features", {}).items()
        ) or "none"
        evidence += (
            f"\nPassive GNW surprise broadcasts by feature: {broadcast_lines}."
        )
    else:
        evidence = (
            "Homeostatic relevance gate selected this question without "
            "selecting its answer: which observable pickup feature predicts "
            "a delayed change in internal pressure? Compare every observed "
            "feature below.\n" + evidence
        )
    return (
        "You are a tiny embodied scientist. Select one causal hypothesis from "
        "the evidence. Return exactly one JSON object with keys target_cause, "
        "comparison_feature, observed_effect, latency_seconds, confidence. "
        "target_cause and comparison_feature must each be exactly one observed "
        "feature token and must differ. observed_effect must be exactly one of "
        "pressure_increase, no_pressure_change, pressure_decrease. "
        "latency_seconds must be a number and confidence must be between 0 and "
        "1. Do not output a falsifier: a formal deduction layer will bind the "
        "selected cause once and derive its logical contradiction. Do not "
        "include markdown.\n" + evidence
    )


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("tiny_scientist_returned_no_json_object")


def validate_hypothesis(payload, summary=None):
    missing = REQUIRED_HYPOTHESIS_FIELDS - set(payload)
    if missing:
        raise ValueError(f"tiny_scientist_missing_fields:{','.join(sorted(missing))}")
    result = {key: payload[key] for key in sorted(REQUIRED_HYPOTHESIS_FIELDS)}
    result["delay_seconds"] = max(0.0, float(result["delay_seconds"]))
    result["confidence"] = clamp(result["confidence"])
    if summary is not None:
        features = {
            str(feature).lower()
            for feature in summary.get("feature_outcomes", {})
        }
        cause_text = (
            str(result["condition"]) + " " + str(result["proposed_cause"])
        ).lower()
        comparison_text = str(result["comparison"]).lower()
        named_causes = {feature for feature in features if feature in cause_text}
        named_controls = {
            feature for feature in features if feature in comparison_text
        }
        if not named_causes:
            raise ValueError("tiny_scientist_cause_has_no_observed_feature")
        if not named_controls or named_controls <= named_causes:
            raise ValueError("tiny_scientist_comparison_has_no_distinct_feature")
        predicted_text = str(result["predicted_effect"]).lower()
        if "pressure" not in predicted_text:
            raise ValueError("tiny_scientist_prediction_has_no_measured_outcome")
        increase_terms = ("increase", "rise", "rises", "higher", "raise", "elevat")
        if not any(term in predicted_text for term in increase_terms):
            raise ValueError("tiny_scientist_prediction_has_wrong_direction")
        if "episode_count" in cause_text:
            raise ValueError("tiny_scientist_aggregate_used_as_cause")
        outcomes = summary.get("feature_outcomes", {})
        means = {
            str(feature).lower(): float(values.get("mean_pressure_delta") or 0.0)
            for feature, values in outcomes.items()
        }
        high_feature = None
        if len(means) >= 2:
            high_feature = max(means, key=means.get)
            low_feature = min(means, key=means.get)
            if means[high_feature] - means[low_feature] >= 0.10:
                if high_feature not in named_causes:
                    raise ValueError("tiny_scientist_cause_reverses_observed_direction")
                if high_feature in named_controls or low_feature not in named_controls:
                    raise ValueError("tiny_scientist_comparison_reverses_observed_direction")
                observed_delay = outcomes[high_feature].get(
                    "mean_positive_delay_seconds"
                )
                if observed_delay is not None and abs(
                    result["delay_seconds"] - float(observed_delay)
                ) > 4.0:
                    raise ValueError("tiny_scientist_delay_not_grounded_in_evidence")
        falsifier_text = str(result["falsifier"]).lower()
        contradiction_terms = (
            " no ", "not ", "doesn't", "does not", "fails", "absent",
            "equal", "lower", "decrease", "without", "zero",
        )
        padded_falsifier = f" {falsifier_text} "
        if not any(term in padded_falsifier for term in contradiction_terms):
            raise ValueError("tiny_scientist_falsifier_does_not_contradict")
        if high_feature is not None:
            falsifier_clauses = re.split(r"[.!?;]+", falsifier_text)
            grounded_contradiction = any(
                high_feature in clause
                and any(term in f" {clause} " for term in contradiction_terms)
                for clause in falsifier_clauses
            )
            if not grounded_contradiction:
                raise ValueError(
                    "tiny_scientist_falsifier_targets_wrong_feature"
                )
    result["status"] = "proposed_unverified"
    result["authority"] = 0.0
    return result


def validate_bound_hypothesis(payload, summary):
    missing = REQUIRED_BOUND_FIELDS - set(payload)
    if missing:
        raise ValueError(
            f"tiny_scientist_missing_bound_fields:{','.join(sorted(missing))}"
        )
    features = {
        str(feature).lower(): values
        for feature, values in summary.get("feature_outcomes", {}).items()
    }
    target = str(payload["target_cause"]).strip().lower()
    comparison = str(payload["comparison_feature"]).strip().lower()
    effect = str(payload["observed_effect"]).strip().lower()
    latency = max(0.0, float(payload["latency_seconds"]))
    if target not in features:
        raise ValueError("tiny_scientist_bound_cause_has_no_observed_feature")
    if comparison not in features or comparison == target:
        raise ValueError("tiny_scientist_bound_comparison_not_distinct")
    if effect not in {
        "pressure_increase",
        "no_pressure_change",
        "pressure_decrease",
    }:
        raise ValueError("tiny_scientist_bound_effect_not_canonical")
    means = {
        feature: float(values.get("mean_pressure_delta") or 0.0)
        for feature, values in features.items()
    }
    high_feature = max(means, key=means.get)
    low_feature = min(means, key=means.get)
    if means[high_feature] - means[low_feature] >= 0.10:
        if target != high_feature or comparison != low_feature:
            raise ValueError("tiny_scientist_bound_cause_reverses_evidence")
        if effect != "pressure_increase":
            raise ValueError("tiny_scientist_bound_effect_reverses_evidence")
        observed_delay = features[high_feature].get(
            "mean_positive_delay_seconds"
        )
        if observed_delay is not None and abs(
            latency - float(observed_delay)
        ) > 4.0:
            raise ValueError("tiny_scientist_bound_delay_not_grounded")
    contradiction = {
        "pressure_increase": "no_pressure_increase_or_pressure_decrease",
        "no_pressure_change": "pressure_increase_or_pressure_decrease",
        "pressure_decrease": "no_pressure_decrease_or_pressure_increase",
    }[effect]
    return {
        "hypothesis": {
            "target_cause": target,
            "comparison_feature": comparison,
            "observed_effect": effect,
            "latency_seconds": latency,
            "confidence": clamp(payload["confidence"]),
        },
        "falsifier_schema": {
            "action_to_retest": target,
            "expected_contradictory_outcome": contradiction,
        },
        "status": "proposed_unverified",
        "authority": 0.0,
    }


def compile_formal_falsifier(payload, summary):
    model_falsifier = payload.get("falsifier")
    features = {
        str(feature).lower()
        for feature in summary.get("feature_outcomes", {})
    }
    cause_text = (
        str(payload.get("condition", ""))
        + " "
        + str(payload.get("proposed_cause", ""))
    ).lower()
    named_causes = sorted(feature for feature in features if feature in cause_text)
    if len(named_causes) != 1:
        raise ValueError("tiny_scientist_compiler_requires_single_bound_cause")
    target = named_causes[0]
    predicted_text = str(payload.get("predicted_effect", "")).lower()
    if any(
        term in predicted_text
        for term in ("increase", "rise", "rises", "higher", "raise", "elevat")
    ):
        contradiction = "no_pressure_increase_or_pressure_decrease"
        falsifier = (
            f"Matched {target} pickups do not precede an increase in internal "
            "pressure at the predicted delay."
        )
    elif any(term in predicted_text for term in ("decrease", "fall", "lower")):
        contradiction = "no_pressure_decrease_or_pressure_increase"
        falsifier = (
            f"Matched {target} pickups do not precede a decrease in internal "
            "pressure at the predicted delay."
        )
    else:
        raise ValueError("tiny_scientist_compiler_effect_has_no_direction")
    compiled_payload = dict(payload)
    compiled_payload["falsifier"] = falsifier
    result = validate_hypothesis(compiled_payload, summary)
    result["model_falsifier_ignored"] = model_falsifier
    result["falsifier_source"] = "formal_symbolic_compiler"
    result["falsifier_schema"] = {
        "action_to_retest": target,
        "expected_contradictory_outcome": contradiction,
    }
    return result


def bound_candidate_payloads(summary):
    features = sorted(
        str(feature).lower()
        for feature in summary.get("feature_outcomes", {})
    )
    delays = {0.0}
    for values in summary.get("feature_outcomes", {}).values():
        delay = values.get("mean_positive_delay_seconds")
        if delay is not None:
            delays.add(float(delay))
    effects = (
        "pressure_increase",
        "no_pressure_change",
        "pressure_decrease",
    )
    return [
        {
            "target_cause": target,
            "comparison_feature": comparison,
            "observed_effect": effect,
            "latency_seconds": delay,
            "confidence": 0.5,
        }
        for target in features
        for comparison in features
        if comparison != target
        for effect in effects
        for delay in sorted(delays)
    ]


def generate_local_hypothesis(
    summary,
    model_name=DEFAULT_MODEL,
    max_new_tokens=256,
    evidence_interface="original",
    hypothesis_contract="freeform",
    evidence_order="canonical",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    if hypothesis_contract == "freeform":
        prompt = hypothesis_prompt(summary, evidence_interface)
        validator = lambda payload: validate_hypothesis(payload, summary)
    elif hypothesis_contract == "freeform_compiled_falsifier":
        prompt = hypothesis_prompt(summary, evidence_interface)
        validator = lambda payload: compile_formal_falsifier(payload, summary)
    elif hypothesis_contract in {"bound_schema", "bound_schema_constrained"}:
        prompt = bound_schema_prompt(
            summary,
            evidence_interface,
            evidence_order,
        )
        validator = lambda payload: validate_bound_hypothesis(payload, summary)
    else:
        raise ValueError(f"unsupported_hypothesis_contract:{hypothesis_contract}")
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    if hypothesis_contract == "bound_schema_constrained":
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        candidate_texts = [
            json.dumps(payload, separators=(",", ":"))
            for payload in bound_candidate_payloads(summary)
        ]
        candidate_tokens = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in candidate_texts
        ]
        eos_token_id = tokenizer.eos_token_id

        def allowed_tokens(_batch_id, input_ids):
            suffix = input_ids[prompt_length:].tolist()
            matches = [
                tokens for tokens in candidate_tokens
                if tokens[: len(suffix)] == suffix
            ]
            allowed = {
                tokens[len(suffix)]
                for tokens in matches
                if len(tokens) > len(suffix)
            }
            if any(len(tokens) == len(suffix) for tokens in matches):
                allowed.add(eos_token_id)
            return sorted(allowed or {eos_token_id})

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max(len(tokens) for tokens in candidate_tokens) + 1,
                do_sample=False,
                num_beams=min(4, len(candidate_tokens)),
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        )
        try:
            return validator(extract_json_object(generated)), generated
        except ValueError as exc:
            raise ValueError(
                f"tiny_scientist_constrained_hypothesis_rejected:{exc}; "
                f"raw={generated}"
            ) from exc
    generated = ""
    last_error = None
    for _attempt in range(2):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max(32, int(max_new_tokens)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][encoded["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            return (
                validator(extract_json_object(generated)),
                generated,
            )
        except ValueError as exc:
            last_error = exc
            messages.extend(
                [
                    {"role": "assistant", "content": generated},
                    {
                        "role": "user",
                        "content": (
                            f"Verifier rejection: {exc}. Re-read the evidence "
                            "and return one corrected JSON object only."
                        ),
                    },
                ]
            )
    raise ValueError(
        f"tiny_scientist_hypothesis_rejected:{last_error}; raw={generated}"
    )


def analyze(
    rows,
    backend="none",
    model_name=DEFAULT_MODEL,
    evidence_interface="original",
    hypothesis_contract="freeform",
    evidence_order="canonical",
):
    episodes = metabolic_episodes(rows)
    gate = PassiveScientificGNW()
    broadcasts = [
        broadcast
        for episode in episodes
        if (broadcast := gate.observe(episode)) is not None
    ]
    summary = evidence_summary(episodes, broadcasts)
    result = {
        "experiment": "passive GNW-selected embodied metabolic hypothesis",
        "evidence_interface": evidence_interface,
        "hypothesis_contract": hypothesis_contract,
        "evidence_order": evidence_order,
        "summary": summary,
        "gnw": {
            "mode": "passive_scientific_broadcast",
            "observations": gate.observations,
            "ignitions": gate.ignitions,
            "held": gate.held,
            "action_influence": gate.action_influence,
            "broadcasts": [asdict(item) for item in broadcasts],
        },
        "hypothesis": None,
        "raw_model_output": None,
    }
    if backend == "local":
        hypothesis, raw = generate_local_hypothesis(
            summary,
            model_name=model_name,
            evidence_interface=evidence_interface,
            hypothesis_contract=hypothesis_contract,
            evidence_order=evidence_order,
        )
        result["hypothesis"] = hypothesis
        result["raw_model_output"] = raw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--backend", choices=["none", "local"], default="none")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--evidence-interface",
        choices=["original", "homeostatic_filtered"],
        default="original",
    )
    parser.add_argument(
        "--hypothesis-contract",
        choices=[
            "freeform",
            "freeform_compiled_falsifier",
            "bound_schema",
            "bound_schema_constrained",
        ],
        default="freeform",
    )
    parser.add_argument(
        "--evidence-order",
        choices=["canonical", "reversed"],
        default="canonical",
    )
    parser.add_argument(
        "--output", default="outputs/tiny_scientist_hypothesis.json"
    )
    args = parser.parse_args()
    rows = load_rows(args.recording)
    rejection = None
    try:
        result = analyze(
            rows,
            args.backend,
            args.model,
            args.evidence_interface,
            args.hypothesis_contract,
            args.evidence_order,
        )
        result["status"] = (
            "accepted_unverified" if result.get("hypothesis") else "evidence_only"
        )
    except ValueError as exc:
        rejection = str(exc)
        result = analyze(
            rows,
            "none",
            args.model,
            args.evidence_interface,
            args.hypothesis_contract,
            args.evidence_order,
        )
        result["status"] = "rejected"
        result["rejection"] = rejection
    result["model"] = args.model
    result["recording"] = args.recording
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if rejection is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
