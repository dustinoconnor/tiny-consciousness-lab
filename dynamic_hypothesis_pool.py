#!/usr/bin/env python3
"""Formal admission and held-out testing of Tiny Scientist L1 hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from tiny_scientist import extract_labeled_causal_ir, validate_bound_hypothesis


ACTIONS = ("observe_red_pickup", "observe_blue_pickup", "wait_no_pickup")


class LocalL1HypothesisProposer:
    """Lazily load Gemma and produce one masked-greedy L1 proposal."""

    def __init__(self, model_name, adapter_path):
        self.model_name = str(model_name)
        self.adapter_path = str(adapter_path)

    def __call__(self, summary):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tiny_scientist import generate_hypothesis_with_model

        tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto"
        )
        model = PeftModel.from_pretrained(model, self.adapter_path)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        _result, raw, metrics = generate_hypothesis_with_model(
            summary,
            tokenizer,
            model,
            device,
            evidence_interface="homeostatic_filtered",
            hypothesis_contract="labeled_causal_ir_masked_greedy",
        )
        return raw, metrics


@dataclass(frozen=True)
class DynamicCausalHypothesis:
    name: str
    target_cause: str
    comparison_feature: str
    observed_effect: str
    latency_seconds: float
    likelihood_rise: tuple[float, float, float]
    source_l1: str
    admitted_after_observation: int


def _entropy(probabilities):
    active = np.asarray(probabilities, dtype=np.float64)
    active = active[active > 1e-15]
    return float(-np.sum(active * np.log2(active)))


def compile_l1_hypothesis(l1_text, discovery_summary, observation_count):
    """Compile a grounded L1 proposal without granting it authority."""
    validated = validate_bound_hypothesis(
        extract_labeled_causal_ir(l1_text), discovery_summary
    )
    hypothesis = validated["hypothesis"]
    target = hypothesis["target_cause"]
    comparison = hypothesis["comparison_feature"]
    effect = hypothesis["observed_effect"]
    if {target, comparison} != {"red", "blue"}:
        raise ValueError("dynamic_pool_requires_actionable_red_blue_roles")
    if effect not in {"pressure_increase", "pressure_decrease"}:
        raise ValueError("dynamic_pool_requires_directional_probe_effect")

    # The current Unity sensor exposes a binary delayed rise. A positive L1
    # effect therefore predicts a rise after the target; a negative effect
    # predicts its complement. The control and no-pickup rates stay symmetric.
    target_rate = 0.92 if effect == "pressure_increase" else 0.08
    control_rate = 0.08 if effect == "pressure_increase" else 0.92
    rates = {target: target_rate, comparison: control_rate}
    likelihood = (rates["red"], rates["blue"], 0.05)
    effect_atom = "rise" if effect == "pressure_increase" else "fall"
    return DynamicCausalHypothesis(
        name=f"dsl:{target}_causes_probe_{effect_atom}@{hypothesis['latency_seconds']:g}s",
        target_cause=target,
        comparison_feature=comparison,
        observed_effect=effect,
        latency_seconds=float(hypothesis["latency_seconds"]),
        likelihood_rise=likelihood,
        source_l1=str(l1_text).strip(),
        admitted_after_observation=int(observation_count),
    )


class DynamicHypothesisPool:
    """A variable Bayesian pool with strict post-admission evidence accounting."""

    def __init__(self, candidate_prior=0.20):
        self.candidate_prior = float(candidate_prior)
        if not 0.0 < self.candidate_prior < 1.0:
            raise ValueError("dynamic_candidate_prior_must_be_between_zero_and_one")
        self.hypotheses = [
            DynamicCausalHypothesis(
                "probe_is_spontaneous", "none", "none", "probe_rise", 0.0,
                (0.55, 0.55, 0.65), "formal_baseline", 0,
            ),
            DynamicCausalHypothesis(
                "no_tested_cause", "none", "none", "no_probe_change", 0.0,
                (0.05, 0.05, 0.05), "formal_baseline", 0,
            ),
        ]
        self.posterior = np.array([0.5, 0.5], dtype=np.float64)
        self.admission_count = 0
        self.duplicate_rejections = 0
        self.pre_admission_evidence_rejections = 0
        self.held_out_updates = 0

    @property
    def likelihood_matrix(self):
        return np.asarray(
            [hypothesis.likelihood_rise for hypothesis in self.hypotheses],
            dtype=np.float64,
        )

    @property
    def map_hypothesis(self):
        return self.hypotheses[int(np.argmax(self.posterior))].name

    @property
    def map_confidence(self):
        return float(np.max(self.posterior))

    def admit(self, hypothesis):
        if any(item.name == hypothesis.name for item in self.hypotheses):
            self.duplicate_rejections += 1
            raise ValueError("dynamic_hypothesis_duplicate")
        self.posterior *= 1.0 - self.candidate_prior
        self.posterior = np.append(self.posterior, self.candidate_prior)
        self.hypotheses.append(hypothesis)
        self.admission_count += 1

    def update(self, action, rise, observation_index):
        action = int(action)
        observation_index = int(observation_index)
        if action < 0 or action >= len(ACTIONS):
            raise ValueError("dynamic_pool_unknown_action")
        if any(
            observation_index <= hypothesis.admitted_after_observation
            for hypothesis in self.hypotheses
            if hypothesis.admitted_after_observation > 0
        ):
            self.pre_admission_evidence_rejections += 1
            return False
        likelihood = self.likelihood_matrix[:, action]
        if not rise:
            likelihood = 1.0 - likelihood
        updated = self.posterior * likelihood
        total = float(np.sum(updated))
        if total <= 1e-15:
            raise ValueError("dynamic_pool_zero_posterior_mass")
        self.posterior = updated / total
        self.held_out_updates += 1
        return True

    def expected_information_gain(self, action):
        action = int(action)
        likelihood = self.likelihood_matrix[:, action]
        rise_probability = float(self.posterior @ likelihood)
        before = _entropy(self.posterior)
        entropies = []
        for rise in (True, False):
            weights = likelihood if rise else 1.0 - likelihood
            posterior = self.posterior * weights
            posterior /= float(np.sum(posterior))
            entropies.append(_entropy(posterior))
        return before - (
            rise_probability * entropies[0]
            + (1.0 - rise_probability) * entropies[1]
        )

    def select_experiment(self):
        scores = [self.expected_information_gain(i) for i in range(len(ACTIONS))]
        return int(np.argmax(scores)), scores

    def audit(self):
        return {
            "hypotheses": [asdict(hypothesis) for hypothesis in self.hypotheses],
            "posterior": {
                hypothesis.name: float(probability)
                for hypothesis, probability in zip(self.hypotheses, self.posterior)
            },
            "map_hypothesis": self.map_hypothesis,
            "map_confidence": self.map_confidence,
            "admission_count": self.admission_count,
            "duplicate_rejections": self.duplicate_rejections,
            "pre_admission_evidence_rejections": (
                self.pre_admission_evidence_rejections
            ),
            "held_out_updates": self.held_out_updates,
        }
