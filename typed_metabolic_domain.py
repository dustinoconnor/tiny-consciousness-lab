#!/usr/bin/env python3
"""Answer-blind Bayesian learner for pickup-specific metabolic relief."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

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

    def __init__(
        self,
        admission_confidence=0.95,
        memory_path=None,
        discovery_memory_path=None,
    ):
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
        self.memory_path = (
            Path(memory_path).expanduser().resolve() if memory_path else None
        )
        self.discovery_memory_path = (
            Path(discovery_memory_path).expanduser().resolve()
            if discovery_memory_path else None
        )
        if (
            self.memory_path is not None
            and self.discovery_memory_path is not None
            and self.memory_path == self.discovery_memory_path
        ):
            raise ValueError("metabolic_discovery_memory_must_be_read_only")
        self.memory_loaded = False
        self._load_discovery_memory()
        if self.memory_loaded:
            self._save()

    def _load_discovery_memory(self):
        path = self.discovery_memory_path
        if path is None or not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        pool_payload = payload.get("pool", {})
        posterior = pool_payload.get("posterior", {})
        names = [item.name for item in self.pool.hypotheses]
        if set(posterior) != set(names):
            raise ValueError("metabolic_memory_hypothesis_mismatch")
        values = np.asarray([posterior[name] for name in names], dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not math.isclose(float(np.sum(values)), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("metabolic_memory_invalid_posterior")
        if payload.get("status") != "verified_held_out":
            raise ValueError("metabolic_discovery_memory_not_verified")
        self.pool.posterior = values / float(np.sum(values))
        self.pool.updates = max(0, int(pool_payload.get("updates", 0)))
        self.observations = list(payload.get("observations", []))
        self.status = "verified_held_out"
        self.admitted_nutrient = str(payload.get("admitted_nutrient", "none"))
        self.admission_cutoff = max(0, int(payload.get("admission_cutoff", 0)))
        self.held_out_confirmations = max(
            0, int(payload.get("held_out_confirmations", 0))
        )
        self.held_out_contradictions = max(
            0, int(payload.get("held_out_contradictions", 0))
        )
        self.held_out_positive = max(0, int(payload.get("held_out_positive", 0)))
        self.held_out_negative = max(0, int(payload.get("held_out_negative", 0)))
        if self.admitted_nutrient not in FEATURES:
            raise ValueError("metabolic_memory_invalid_nutrient")
        self.memory_loaded = True

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
            self._save()
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
        self._save()

    def _save(self):
        if self.memory_path is None:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.audit(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.memory_path)

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
