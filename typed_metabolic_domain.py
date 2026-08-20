#!/usr/bin/env python3
"""Answer-blind Bayesian learner for pickup-specific metabolic relief."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


FEATURES = ("blue", "red", "yellow")


@dataclass(frozen=True)
class MetabolicRoleHypothesis:
    name: str
    nutrient: str
    relief_likelihood: tuple[float, float, float]


class TypedMetabolicRolePool:
    """Symmetric alternatives for which existing pickup relieves hunger."""

    def __init__(self):
        self.hypotheses = (
            MetabolicRoleHypothesis(
                "blue_restores_metabolic_reserve", "blue", (0.92, 0.05, 0.05)
            ),
            MetabolicRoleHypothesis(
                "red_restores_metabolic_reserve", "red", (0.05, 0.92, 0.05)
            ),
            MetabolicRoleHypothesis(
                "yellow_restores_metabolic_reserve", "yellow", (0.05, 0.05, 0.92)
            ),
            MetabolicRoleHypothesis(
                "any_pickup_restores_metabolic_reserve", "any_pickup",
                (0.85, 0.85, 0.85),
            ),
            MetabolicRoleHypothesis(
                "no_pickup_restores_metabolic_reserve", "none", (0.05, 0.05, 0.05)
            ),
        )
        self.posterior = np.full(
            len(self.hypotheses), 1.0 / len(self.hypotheses), dtype=np.float64
        )
        self.updates = 0

    @property
    def likelihood_matrix(self):
        return np.asarray(
            [item.relief_likelihood for item in self.hypotheses],
            dtype=np.float64,
        )

    @property
    def map_hypothesis(self):
        return self.hypotheses[int(np.argmax(self.posterior))]

    @property
    def map_confidence(self):
        return float(np.max(self.posterior))

    def update(self, feature, relieved):
        if feature not in FEATURES:
            raise ValueError("unknown_metabolic_feature")
        index = FEATURES.index(feature)
        likelihood = self.likelihood_matrix[:, index]
        weights = likelihood if bool(relieved) else 1.0 - likelihood
        posterior = self.posterior * weights
        total = float(np.sum(posterior))
        if not math.isfinite(total) or total <= 1e-15:
            raise ValueError("metabolic_role_zero_posterior_mass")
        self.posterior = posterior / total
        self.updates += 1

    def relief_probability(self, feature):
        if feature not in FEATURES:
            raise ValueError("unknown_metabolic_feature")
        index = FEATURES.index(feature)
        return float(self.posterior @ self.likelihood_matrix[:, index])

    def audit(self):
        return {
            "features": list(FEATURES),
            "hypotheses": [asdict(item) for item in self.hypotheses],
            "posterior": {
                item.name: float(value)
                for item, value in zip(self.hypotheses, self.posterior)
            },
            "map_hypothesis": self.map_hypothesis.name,
            "map_nutrient": self.map_hypothesis.nutrient,
            "map_confidence": self.map_confidence,
            "updates": self.updates,
        }


class HeldOutMetabolicRoleLearner:
    """Freeze an admitted rule, then verify it on post-cutoff outcomes."""

    def __init__(self, admission_confidence=0.95):
        self.pool = TypedMetabolicRolePool()
        self.admission_confidence = float(admission_confidence)
        self.observations = []
        self.status = "collecting"
        self.admitted_nutrient = "none"
        self.admission_cutoff = 0
        self.held_out_confirmations = 0
        self.held_out_contradictions = 0
        self.held_out_positive = 0
        self.held_out_negative = 0

    def observe(self, feature, relieved):
        feature = str(feature).strip().lower()
        relieved = bool(relieved)
        self.pool.update(feature, relieved)
        record = {
            "index": len(self.observations) + 1,
            "feature": feature,
            "metabolic_relief": relieved,
        }
        self.observations.append(record)
        if self.status == "collecting":
            hypothesis = self.pool.map_hypothesis
            if (
                hypothesis.nutrient in FEATURES
                and self.pool.map_confidence >= self.admission_confidence
            ):
                self.status = "admitted_unverified"
                self.admitted_nutrient = hypothesis.nutrient
                self.admission_cutoff = record["index"]
            return
        expected = feature == self.admitted_nutrient
        if relieved == expected:
            self.held_out_confirmations += 1
        else:
            self.held_out_contradictions += 1
        if expected and relieved:
            self.held_out_positive += 1
        if not expected and not relieved:
            self.held_out_negative += 1
        if (
            self.held_out_contradictions == 0
            and self.held_out_positive > 0
            and self.held_out_negative > 0
        ):
            self.status = "verified_held_out"

    def audit(self):
        return {
            "status": self.status,
            "admitted_nutrient": self.admitted_nutrient,
            "admission_cutoff": self.admission_cutoff,
            "held_out_confirmations": self.held_out_confirmations,
            "held_out_contradictions": self.held_out_contradictions,
            "held_out_positive": self.held_out_positive,
            "held_out_negative": self.held_out_negative,
            "observations": list(self.observations),
            "pool": self.pool.audit(),
        }
