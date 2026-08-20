#!/usr/bin/env python3
"""Typed ordered-intervention substrate for the yellow-flower experiment."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np


OBJECT_CONCEPTS = {
    "red": {"type": "pickup", "form": "mushroom"},
    "blue": {"type": "pickup", "form": "mushroom"},
    "yellow": {"type": "pickup", "form": "flower"},
}

ORDERED_ACTIONS = (
    "red_wait",
    "red_then_yellow",
    "red_then_blue",
    "yellow_then_red",
)

SEALED_DISCOVERY_OBSERVATIONS = [
    {"index": 1, "ordered_action": "red_then_blue", "probe_suppressed": False},
    {"index": 2, "ordered_action": "red_then_blue", "probe_suppressed": False},
    {"index": 3, "ordered_action": "red_then_yellow", "probe_suppressed": True},
]


def validate_consume_action(feature):
    """Return a typed action without assigning an unknown causal role."""
    name = str(feature).strip().lower()
    concept = OBJECT_CONCEPTS.get(name)
    if concept is None or concept.get("type") != "pickup":
        raise ValueError("consume_requires_grounded_pickup_concept")
    return {"operator": "consume", "argument": name, "argument_type": "pickup"}


class OrderedCausalProbeWorld:
    """Hidden passive transition dynamics, kept outside the cognitive model."""

    def __init__(self, hz=5.0, delay_seconds=10.0, cancellation_feature="none"):
        self.hz = max(0.1, float(hz))
        self.delay_ticks = max(1, int(round(float(delay_seconds) * self.hz)))
        self.cancellation_feature = str(cancellation_feature).strip().lower()
        self.pending_due_steps = deque()
        self.scheduled_events = 0
        self.completed_events = 0
        self.cancelled_events = 0
        self.ambiguous_pickup_frames = 0

    def register_pickups(self, step, red=0, blue=0, yellow=0):
        """Apply a telemetry frame; same-frame red/yellow order is unknowable."""
        step = int(step)
        red = max(0, int(red))
        blue = max(0, int(blue))
        yellow = max(0, int(yellow))
        cancelled = 0
        if red > 0 and yellow > 0:
            self.ambiguous_pickup_frames += 1
        elif yellow > 0 and self.cancellation_feature == "yellow":
            for _ in range(yellow):
                if not self.pending_due_steps:
                    break
                # The most recent unresolved intervention is the only event
                # eligible for this ordered relation.
                self.pending_due_steps.pop()
                self.cancelled_events += 1
                cancelled += 1
        elif blue > 0 and self.cancellation_feature == "blue":
            for _ in range(blue):
                if not self.pending_due_steps:
                    break
                self.pending_due_steps.pop()
                self.cancelled_events += 1
                cancelled += 1

        for _ in range(red):
            self.pending_due_steps.append(step + self.delay_ticks)
            self.scheduled_events += 1
        return cancelled

    def pop_due(self, step):
        due = 0
        while self.pending_due_steps and self.pending_due_steps[0] <= int(step):
            self.pending_due_steps.popleft()
            due += 1
        self.completed_events += due
        return due

    def reset(self):
        self.pending_due_steps.clear()

    def audit(self):
        return {
            "delay_seconds": self.delay_ticks / self.hz,
            "pending_events": len(self.pending_due_steps),
            "scheduled_events": self.scheduled_events,
            "completed_events": self.completed_events,
            "cancelled_events": self.cancelled_events,
            "ambiguous_pickup_frames": self.ambiguous_pickup_frames,
        }


@dataclass(frozen=True)
class OrderedInteractionHypothesis:
    name: str
    suppressor: str
    relation: str
    suppression_likelihood: tuple[float, float, float, float]


def _entropy(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    values = values[values > 1e-15]
    return float(-np.sum(values * np.log2(values)))


class TypedInteractionPool:
    """Symmetric formal alternatives for ordered causal suppression."""

    def __init__(self):
        self.hypotheses = (
            OrderedInteractionHypothesis(
                "yellow_after_red_suppresses_probe",
                "yellow",
                "after_red",
                (0.05, 0.92, 0.05, 0.05),
            ),
            OrderedInteractionHypothesis(
                "blue_after_red_suppresses_probe",
                "blue",
                "after_red",
                (0.05, 0.05, 0.92, 0.05),
            ),
            OrderedInteractionHypothesis(
                "any_pickup_after_red_suppresses_probe",
                "any_pickup",
                "after_red",
                (0.05, 0.85, 0.85, 0.05),
            ),
            OrderedInteractionHypothesis(
                "no_ordered_suppression",
                "none",
                "none",
                (0.05, 0.05, 0.05, 0.05),
            ),
        )
        self.posterior = np.full(
            len(self.hypotheses), 1.0 / len(self.hypotheses), dtype=np.float64
        )
        self.updates = 0

    @property
    def likelihood_matrix(self):
        return np.asarray(
            [item.suppression_likelihood for item in self.hypotheses],
            dtype=np.float64,
        )

    @property
    def map_hypothesis(self):
        return self.hypotheses[int(np.argmax(self.posterior))].name

    @property
    def map_confidence(self):
        return float(np.max(self.posterior))

    def update(self, action, suppressed):
        action = int(action)
        if action < 0 or action >= len(ORDERED_ACTIONS):
            raise ValueError("typed_interaction_unknown_action")
        likelihood = self.likelihood_matrix[:, action]
        weights = likelihood if bool(suppressed) else 1.0 - likelihood
        posterior = self.posterior * weights
        total = float(np.sum(posterior))
        if not math.isfinite(total) or total <= 1e-15:
            raise ValueError("typed_interaction_zero_posterior_mass")
        self.posterior = posterior / total
        self.updates += 1

    def expected_information_gain(self, action):
        action = int(action)
        likelihood = self.likelihood_matrix[:, action]
        probability = float(self.posterior @ likelihood)
        expected = 0.0
        for outcome, outcome_probability in (
            (True, probability),
            (False, 1.0 - probability),
        ):
            weights = likelihood if outcome else 1.0 - likelihood
            posterior = self.posterior * weights
            posterior /= float(np.sum(posterior))
            expected += outcome_probability * _entropy(posterior)
        return _entropy(self.posterior) - expected

    def select_experiment(self):
        scores = [
            self.expected_information_gain(index)
            for index in range(len(ORDERED_ACTIONS))
        ]
        return int(np.argmax(scores)), scores

    def audit(self):
        return {
            "typed_objects": OBJECT_CONCEPTS,
            "ordered_actions": list(ORDERED_ACTIONS),
            "hypotheses": [asdict(item) for item in self.hypotheses],
            "posterior": {
                item.name: float(probability)
                for item, probability in zip(self.hypotheses, self.posterior)
            },
            "map_hypothesis": self.map_hypothesis,
            "map_confidence": self.map_confidence,
            "updates": self.updates,
        }


def validate_sealed_discovery_memory(path):
    """Validate the immutable three-observation live-smoke checkpoint."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("observations") != SEALED_DISCOVERY_OBSERVATIONS:
        raise ValueError("sealed_discovery_observations_mismatch")
    if int(payload.get("updates", -1)) != 3:
        raise ValueError("sealed_discovery_update_count_mismatch")
    if int(payload.get("completed_episodes", -1)) != 3:
        raise ValueError("sealed_discovery_completed_count_mismatch")
    pool = TypedInteractionPool()
    pool.update(2, False)
    pool.update(2, False)
    pool.update(1, True)
    expected = pool.audit()["posterior"]
    actual = payload.get("posterior", {})
    if set(actual) != set(expected) or any(
        not math.isclose(float(actual[name]), value, abs_tol=1e-12)
        for name, value in expected.items()
    ):
        raise ValueError("sealed_discovery_posterior_mismatch")
    return payload


class PassiveOrderedEpisodeLearner:
    """Admit one clean red-led episode and update only at its deadline."""

    def __init__(
        self,
        hz=5.0,
        delay_seconds=60.0,
        enabled=False,
        memory_path=None,
        discovery_memory_path=None,
        hypothesis_proposer=None,
        formulation_minimum=3,
    ):
        self.enabled = bool(enabled)
        self.delay_ticks = max(1, int(round(float(delay_seconds) * float(hz))))
        self.pool = TypedInteractionPool()
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
            raise ValueError("typed_interaction_discovery_must_be_read_only")
        self.memory_loaded = False
        self.observations = []
        self.hypothesis_proposer = hypothesis_proposer
        self.formulation_minimum = max(2, int(formulation_minimum))
        self.formulation_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ordered-scientist")
            if hypothesis_proposer is not None
            else None
        )
        self.formulation_future = None
        self.formulation_status = (
            "collecting" if hypothesis_proposer is not None else "disabled"
        )
        self.formulation_cutoff = 0
        self.formulated_hypothesis = None
        self.formulation_raw = ""
        self.formulation_diagnostics = {}
        self.formulation_held_out_evaluations = 0
        self.formulation_held_out_confirmations = 0
        self.active = None
        self.completed_episodes = 0
        self.discarded_episodes = 0
        self.last_outcome = "none"
        self.last_action = "none"
        self._load_memory()
        if (
            self.memory_loaded
            and self.discovery_memory_path is not None
            and self.memory_path is not None
        ):
            self._save_memory()

    def _load_memory(self):
        source = self.discovery_memory_path or self.memory_path
        if source is None or not source.exists():
            return
        payload = json.loads(source.read_text(encoding="utf-8"))
        posterior = payload.get("posterior", {})
        names = [item.name for item in self.pool.hypotheses]
        if set(posterior) != set(names):
            raise ValueError("typed_interaction_memory_hypothesis_mismatch")
        values = np.asarray([posterior[name] for name in names], dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not math.isclose(float(np.sum(values)), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("typed_interaction_memory_invalid_posterior")
        self.pool.posterior = values / float(np.sum(values))
        self.pool.updates = max(0, int(payload.get("updates", 0)))
        self.completed_episodes = max(
            0, int(payload.get("completed_episodes", self.pool.updates))
        )
        self.discarded_episodes = max(
            0, int(payload.get("discarded_episodes", 0))
        )
        self.observations = list(payload.get("observations", []))
        self.memory_loaded = True

    def _save_memory(self):
        if self.memory_path is None:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "posterior": self.pool.audit()["posterior"],
            "updates": self.pool.updates,
            "completed_episodes": self.completed_episodes,
            "discarded_episodes": self.discarded_episodes,
            "last_action": self.last_action,
            "last_outcome": self.last_outcome,
            "observations": self.observations,
        }
        temporary = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.memory_path)

    def reset(self):
        if self.active is not None:
            self.discarded_episodes += 1
            self.last_outcome = "discarded_reset"
            self._save_memory()
        self.active = None

    def observe_pickups(self, step, red=0, blue=0, yellow=0, can_start=True):
        if not self.enabled:
            return
        step = int(step)
        red, blue, yellow = max(0, int(red)), max(0, int(blue)), max(0, int(yellow))
        total = red + blue + yellow
        if self.active is not None and total:
            if red or total != 1 or self.active["second"] != "none":
                self.active["contaminated"] = True
            elif yellow == 1:
                self.active["second"] = "yellow"
            elif blue == 1:
                self.active["second"] = "blue"

        if self.active is None and can_start and red == 1 and total == 1:
            self.active = {
                "start_step": step,
                "deadline_step": step + self.delay_ticks,
                "second": "none",
                "contaminated": False,
            }

    def observe_deadline(self, step, observed_probe_events=0):
        if not self.enabled or self.active is None:
            return False
        if int(step) < self.active["deadline_step"]:
            return False
        episode = self.active
        self.active = None
        if episode["contaminated"]:
            self.discarded_episodes += 1
            self.last_outcome = "discarded_contamination"
            self._save_memory()
            return False
        action = {"none": 0, "yellow": 1, "blue": 2}[episode["second"]]
        suppressed = int(observed_probe_events) == 0
        self.pool.update(action, suppressed)
        self.observations.append(
            {
                "index": self.pool.updates,
                "ordered_action": ORDERED_ACTIONS[action],
                "probe_suppressed": suppressed,
            }
        )
        self.completed_episodes += 1
        self.last_action = ORDERED_ACTIONS[action]
        self.last_outcome = "suppressed" if suppressed else "probe_observed"
        self._refresh_formulation_verification()
        self._save_memory()
        return True

    def _refresh_formulation_verification(self):
        """Promote only from matching observations strictly after admission cutoff."""
        hypothesis = self.formulated_hypothesis
        if not isinstance(hypothesis, dict):
            return
        initiator = str(hypothesis.get("initiator", ""))
        second = str(hypothesis.get("second_action", ""))
        expected_action = f"{initiator}_then_{second}"
        expected_suppression = (
            hypothesis.get("observed_effect") == "suppresses_probe"
        )
        relevant = [
            item
            for item in self.observations
            if int(item.get("index", 0)) > self.formulation_cutoff
            and item.get("ordered_action") == expected_action
        ]
        self.formulation_held_out_evaluations = len(relevant)
        self.formulation_held_out_confirmations = sum(
            bool(item.get("probe_suppressed")) == expected_suppression
            for item in relevant
        )
        if self.formulation_held_out_confirmations > 0:
            self.formulation_status = "verified_held_out"

    def poll_formulation(self):
        if self.hypothesis_proposer is None or self.formulated_hypothesis:
            return
        if self.formulation_future is not None:
            if not self.formulation_future.done():
                return
            try:
                compiled, raw, diagnostics = self.formulation_future.result()
                self.formulated_hypothesis = compiled
                self.formulation_raw = raw
                self.formulation_diagnostics = diagnostics
                self.formulation_status = "admitted_unverified"
                self._refresh_formulation_verification()
            except Exception as exc:
                self.formulation_status = f"rejected:{type(exc).__name__}:{exc}"
            self.formulation_future = None
            return
        if len(self.observations) < self.formulation_minimum:
            return
        counts = {}
        for item in self.observations:
            action = item["ordered_action"]
            values = counts.setdefault(action, {"episodes": 0, "suppressed": 0})
            values["episodes"] += 1
            values["suppressed"] += int(item["probe_suppressed"])
        summary = {
            action: {
                "episodes": values["episodes"],
                "suppressed": values["suppressed"],
                "not_suppressed": values["episodes"] - values["suppressed"],
            }
            for action, values in sorted(counts.items())
        }
        self.formulation_cutoff = len(self.observations)
        self.formulation_status = "generating"
        self.formulation_future = self.formulation_executor.submit(
            self.hypothesis_proposer, summary, self.formulation_cutoff
        )

    def audit(self):
        result = self.pool.audit()
        result.update(
            {
                "enabled": self.enabled,
                "active_episode": dict(self.active) if self.active else None,
                "completed_episodes": self.completed_episodes,
                "discarded_episodes": self.discarded_episodes,
                "last_action": self.last_action,
                "last_outcome": self.last_outcome,
                "motor_authority": 0.0,
                "memory_path": str(self.memory_path) if self.memory_path else None,
                "discovery_memory_path": (
                    str(self.discovery_memory_path)
                    if self.discovery_memory_path else None
                ),
                "discovery_memory_read_only": self.discovery_memory_path is not None,
                "memory_loaded": self.memory_loaded,
                "observations": list(self.observations),
                "formulation_status": self.formulation_status,
                "formulation_cutoff": self.formulation_cutoff,
                "formulated_hypothesis": self.formulated_hypothesis,
                "formulation_raw": self.formulation_raw,
                "formulation_diagnostics": self.formulation_diagnostics,
                "formulation_held_out_evaluations": (
                    self.formulation_held_out_evaluations
                ),
                "formulation_held_out_confirmations": (
                    self.formulation_held_out_confirmations
                ),
            }
        )
        return result
