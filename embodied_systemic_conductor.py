#!/usr/bin/env python3
"""Unity observation and bounded routing for the four-context conductor."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path


CONTEXTS = (
    "clear_terrain",
    "familiar_hidden_goal",
    "visible_target",
    "genuine_wedge",
)
SPECIALISTS = ("recurrent", "episodic", "predictive", "fallback")
FEATURE_NAMES = (
    "food_visibility",
    "art_resonance",
    "obstacle_pressure",
    "body_contact",
    "immobility",
    "spatial_novelty",
    "free_space",
)


def clamp(value, low=0.0, high=1.0):
    return float(max(low, min(high, float(value))))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def expand_features(features):
    food, resonance, obstacle, contact, immobility, novelty, free_space = features
    return list(features) + [
        food * free_space,
        resonance * (1.0 - food),
        contact * immobility,
        obstacle * (1.0 - free_space),
        novelty * free_space,
        1.0,
    ]


def softmax(values, temperature=0.18):
    scaled = [float(value) / max(float(temperature), 1e-6) for value in values]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def entropy_bits(probabilities):
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 1e-12
    )


def normalized_array(snapshot, key):
    values = snapshot.get(key, [])
    if not isinstance(values, (list, tuple)):
        return []
    return [clamp(value) for value in values]


def live_features(snapshot):
    """Map grounded Unity telemetry onto the checkpoint's seven inputs."""
    food = float(bool(snapshot.get("food_visible", False)))
    route_match = clamp(snapshot.get("terrain_air_route_match", 0.0))
    contact = float(
        bool(
            snapshot.get("blocked", False)
            or snapshot.get("body_collision", False)
            or snapshot.get("horizontal_collision", False)
        )
    )
    obstacle = clamp(snapshot.get("trap_pressure", 0.0))
    wedge = clamp(float(snapshot.get("physics_wedge_seconds", 0.0)) / 2.0)
    trap = clamp(float(snapshot.get("trap_accumulation_seconds", 0.0)) / 3.0)
    immobility = max(wedge, trap, float(bool(snapshot.get("stuck", False))))
    # Generic AIR retrieval is common in clear terrain. For this specialist
    # gate, "resonance" means the necessity-gated ART route precedent.
    resonance = (
        route_match
        if (
            snapshot.get("terrain_air_route_active", False)
            or snapshot.get("terrain_air_route_pending", False)
        )
        else 0.0
    )
    novelty = clamp(1.0 - route_match)
    clearance = normalized_array(snapshot, "directional_body_clearance")
    if not clearance:
        clearance = normalized_array(snapshot, "body_clearance")
    if not clearance:
        clearance = normalized_array(snapshot, "directional_rays")
    if not clearance:
        clearance = normalized_array(snapshot, "rays")
    free_space = mean(clearance) if clearance else clamp(
        snapshot.get("free_space", 0.5)
    )
    return [
        food,
        resonance,
        obstacle,
        contact,
        immobility,
        novelty,
        free_space,
    ]


def proxy_context(snapshot, features):
    """Evaluation label derived separately from the conductor's input path."""
    food, resonance, obstacle, contact, immobility, _novelty, free_space = features
    # Fallback can remain active after physical escape. Label the observed
    # condition rather than treating controller bookkeeping as ground truth.
    if snapshot.get("stuck", False) or immobility >= 0.75:
        return "genuine_wedge"
    if food >= 0.5:
        return "visible_target"
    if (
        snapshot.get("hidden_goal_active", False)
        or snapshot.get("terrain_air_route_active", False)
        or (resonance >= 0.72 and obstacle >= 0.35 and free_space <= 0.55)
    ):
        return "familiar_hidden_goal"
    if contact and obstacle >= 0.65 and free_space <= 0.30:
        return "genuine_wedge"
    return "clear_terrain"


class PassiveSystemicConductor:
    """Frozen four-context gate with no action-selection method."""

    def __init__(self, checkpoint=None, enabled=False, smoothing=0.62):
        self.enabled = bool(enabled or checkpoint)
        self.smoothing = clamp(smoothing)
        self.weights = [[0.0] * 13 for _ in SPECIALISTS]
        self.checkpoint = "none"
        self.attended = None
        self.features = [0.0] * len(FEATURE_NAMES)
        self.scores = [0.0] * len(SPECIALISTS)
        self.probabilities = [0.25] * len(SPECIALISTS)
        self.raw_probabilities = [0.25] * len(SPECIALISTS)
        self.raw_recommendation = "unobserved"
        self.recommendation = "unobserved"
        self.confidence = 0.0
        self.entropy = 2.0
        self.proxy_context = "unobserved"
        self.proxy_optimal = "unobserved"
        self.agreement = False
        self.agreements = 0
        self.observations = 0
        self.context_counts = Counter()
        self.recommendation_counts = Counter()
        self.action_influence = 0
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        path = Path(checkpoint).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_type") != "four_context_conductor":
            raise ValueError("unsupported_systemic_conductor_checkpoint")
        if tuple(payload.get("contexts", ())) != CONTEXTS:
            raise ValueError("systemic_conductor_context_mismatch")
        if tuple(payload.get("specialists", ())) != SPECIALISTS:
            raise ValueError("systemic_conductor_specialist_mismatch")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("systemic_conductor_feature_mismatch")
        weights = payload.get("weights", [])
        if len(weights) != len(SPECIALISTS) or any(
            len(row) != 13 for row in weights
        ):
            raise ValueError("systemic_conductor_weight_shape")
        self.weights = [[float(value) for value in row] for row in weights]
        self.checkpoint = str(path)

    def predict(self, features, input_permutation=None, output_permutation=None, lesion=None):
        values = list(features)
        if input_permutation is not None:
            values = [values[index] for index in input_permutation]
        expanded = expand_features(values)
        scores = [
            sum(weight * value for weight, value in zip(row, expanded))
            for row in self.weights
        ]
        allowed = [
            index for index in range(len(SPECIALISTS)) if index != lesion
        ]
        masked = [
            score if index in allowed else -1e9
            for index, score in enumerate(scores)
        ]
        probabilities = softmax(masked)
        selected = max(allowed, key=lambda index: (probabilities[index], -index))
        if output_permutation is not None:
            selected = int(output_permutation[selected])
            if selected == lesion:
                selected = max(
                    allowed, key=lambda index: (probabilities[index], -index)
                )
        return selected, scores, probabilities

    def observe(self, snapshot):
        if not self.enabled:
            return
        current = live_features(snapshot)
        raw_selected, _raw_scores, raw_probabilities = self.predict(current)
        self.attended = (
            current
            if self.attended is None
            else [
                self.smoothing * value + (1.0 - self.smoothing) * previous
                for value, previous in zip(current, self.attended)
            ]
        )
        selected, scores, probabilities = self.predict(self.attended)
        context = proxy_context(snapshot, current)
        optimal = SPECIALISTS[CONTEXTS.index(context)]
        recommendation = SPECIALISTS[selected]

        ranked = sorted(probabilities, reverse=True)
        self.features = list(current)
        self.scores = scores
        self.probabilities = probabilities
        self.raw_probabilities = raw_probabilities
        self.raw_recommendation = SPECIALISTS[raw_selected]
        self.recommendation = recommendation
        self.confidence = clamp(ranked[0] - ranked[1])
        self.entropy = entropy_bits(probabilities)
        self.proxy_context = context
        self.proxy_optimal = optimal
        self.agreement = recommendation == optimal
        self.observations += 1
        self.agreements += int(self.agreement)
        self.context_counts[context] += 1
        self.recommendation_counts[recommendation] += 1

    @property
    def agreement_rate(self):
        return self.agreements / max(self.observations, 1)


class PassiveAdaptiveIgnitionGate:
    """Hold or release a specialist broadcast without influencing actions."""

    def __init__(
        self,
        enabled=False,
        challenger_confidence=0.55,
        switch_margin=0.04,
        release_frames=1,
    ):
        self.enabled = bool(enabled)
        self.challenger_confidence = clamp(challenger_confidence)
        self.switch_margin = max(0.0, float(switch_margin))
        self.release_frames = max(1, int(release_frames))
        self.reset(clear_totals=True)

    def reset(self, clear_totals=False):
        self.active_index = None
        self.active_specialist = "unobserved"
        self.candidate_specialist = "unobserved"
        self.challenger_index = None
        self.challenger_specialist = "none"
        self.challenger_streak = 0
        self.broadcast_age = 0
        self.last_event = "disabled" if not self.enabled else "waiting"
        self.last_ignition = False
        self.confidence = 0.0
        self.candidate_confidence = 0.0
        self.margin = 0.0
        self.agreement = False
        if clear_totals:
            self.observations = 0
            self.ignitions = 0
            self.releases = 0
            self.held_challenges = 0
            self.agreements = 0
            self.action_influence = 0
            self.recommendation_substitutions = 0

    def observe(self, probabilities, proxy_optimal):
        if not self.enabled:
            return
        values = [float(value) for value in probabilities]
        if len(values) != len(SPECIALISTS):
            raise ValueError("adaptive_ignition_probability_shape")
        candidate = max(range(len(values)), key=lambda index: values[index])
        candidate_confidence = values[candidate]
        self.candidate_specialist = SPECIALISTS[candidate]
        self.candidate_confidence = candidate_confidence
        self.last_ignition = False
        self.observations += 1

        if self.active_index is None:
            self.active_index = candidate
            self.broadcast_age = 1
            self.ignitions += 1
            self.last_ignition = True
            self.last_event = "initial_ignition"
        elif candidate == self.active_index:
            self.challenger_index = None
            self.challenger_streak = 0
            self.broadcast_age += 1
            self.last_event = "broadcast_supported"
        else:
            if candidate == self.challenger_index:
                self.challenger_streak += 1
            else:
                self.challenger_index = candidate
                self.challenger_streak = 1
            active_confidence = values[self.active_index]
            self.margin = candidate_confidence - active_confidence
            should_release = (
                candidate_confidence >= self.challenger_confidence
                and self.margin >= self.switch_margin
                and self.challenger_streak >= self.release_frames
            )
            if should_release:
                self.active_index = candidate
                self.broadcast_age = 1
                self.challenger_index = None
                self.challenger_streak = 0
                self.ignitions += 1
                self.releases += 1
                self.last_ignition = True
                self.last_event = "sustained_challenger_release"
            else:
                self.broadcast_age += 1
                self.held_challenges += 1
                self.last_event = "transient_challenger_held"

        self.active_specialist = SPECIALISTS[self.active_index]
        self.challenger_specialist = (
            SPECIALISTS[self.challenger_index]
            if self.challenger_index is not None
            else "none"
        )
        self.confidence = values[self.active_index]
        self.agreement = self.active_specialist == proxy_optimal
        self.agreements += int(self.agreement)

    @property
    def agreement_rate(self):
        return self.agreements / max(self.observations, 1)


class BoundedExecutiveRouter:
    """Let the conductor time recurrent/MPC handoffs without issuing actions."""

    MODES = ("passive", "recurrent_mpc", "recurrent_mpc_air")

    def __init__(
        self,
        mode="passive",
        hz=5.0,
        confidence_threshold=0.55,
        hold_seconds=0.8,
        mandatory_persistence_seconds=1.5,
        recurrent_release_frames=3,
        chatter_window_seconds=3.0,
    ):
        if mode not in self.MODES:
            raise ValueError("unsupported_systemic_conductor_control")
        self.mode = mode
        self.hz = max(float(hz), 0.1)
        self.confidence_threshold = clamp(confidence_threshold)
        self.hold_total_ticks = max(1, int(round(hold_seconds * self.hz)))
        self.mandatory_latch_total_ticks = max(
            1, int(round(mandatory_persistence_seconds * self.hz))
        )
        self.recurrent_release_frames = max(
            1, int(recurrent_release_frames)
        )
        self.chatter_window_ticks = max(
            1, int(round(chatter_window_seconds * self.hz))
        )
        self.recent_handoffs = deque()
        self.reset(clear_totals=True)

    def reset(self, clear_totals=False):
        self.active_specialist = "baseline"
        self.hold_ticks = 0
        self.mandatory_latch_ticks = 0
        self.recurrent_release_streak = 0
        self.last_handoff_tick = None
        if clear_totals:
            self.frames = 0
            self.decisions = 0
            self.handoffs = 0
            self.denials = 0
            self.safety_overrides = 0
            self.influence_frames = 0
            self.chatter_events = 0
        self.last_baseline_specialist = "recurrent"
        self.last_recommendation = "unobserved"
        self.last_confidence = 0.0
        self.last_reason = "passive"
        self.recent_handoffs.clear()

    def select_mpc(
        self,
        recommendation,
        confidence,
        baseline_mpc,
        mandatory_mpc=False,
        fallback_active=False,
    ):
        """Return whether MPC should run; fallback remains externally dominant."""
        self.frames += 1
        baseline = "predictive" if baseline_mpc else "recurrent"
        self.last_baseline_specialist = baseline
        self.last_recommendation = str(recommendation)
        self.last_confidence = clamp(confidence)

        if self.mode == "passive":
            self.active_specialist = baseline
            self.last_reason = "passive_baseline"
            return bool(baseline_mpc)

        if fallback_active:
            self.mandatory_latch_ticks = 0
            self.recurrent_release_streak = 0
            self.active_specialist = "fallback"
            self.last_reason = "fallback_priority"
            return bool(baseline_mpc)

        desired = baseline
        reason = "baseline_low_confidence"
        confident_pair = (
            recommendation in {"recurrent", "predictive"}
            and self.last_confidence >= self.confidence_threshold
        )
        if confident_pair:
            desired = recommendation
            reason = "conductor_recommendation"
            self.decisions += 1
        elif recommendation not in {"unobserved", "recurrent", "predictive"}:
            self.denials += 1
            reason = "branch_not_enabled"

        current = (
            self.active_specialist
            if self.active_specialist in {"recurrent", "predictive"}
            else baseline
        )

        # Live evidence always outranks executive timing. This includes visible
        # food, immediate obstacle risk, and grounded AIR guidance.
        if mandatory_mpc:
            self.mandatory_latch_ticks = self.mandatory_latch_total_ticks
            self.recurrent_release_streak = 0
            if desired != "predictive":
                self.safety_overrides += 1
            desired = "predictive"
            reason = "mandatory_mpc"
        elif self.mandatory_latch_ticks > 0:
            self.mandatory_latch_ticks -= 1
            self.recurrent_release_streak = 0
            desired = "predictive"
            reason = "mandatory_mpc_persistence"
        elif desired == "recurrent" and current == "predictive":
            self.recurrent_release_streak += 1
            if self.recurrent_release_streak < self.recurrent_release_frames:
                desired = "predictive"
                reason = "recurrent_release_confirmation"
                self.denials += 1
        else:
            self.recurrent_release_streak = 0

        if (
            desired != current
            and self.hold_ticks > 0
            and not mandatory_mpc
        ):
            desired = current
            self.denials += 1
            reason = "hysteresis_hold"

        if desired != current:
            self.handoffs += 1
            if (
                self.last_handoff_tick is not None
                and self.frames - self.last_handoff_tick
                <= self.chatter_window_ticks
            ):
                self.chatter_events += 1
            self.last_handoff_tick = self.frames
            self.recent_handoffs.append(self.frames)
            self.hold_ticks = self.hold_total_ticks
        else:
            self.hold_ticks = max(0, self.hold_ticks - 1)

        self.active_specialist = desired
        self.last_reason = reason
        selected_mpc = desired == "predictive"
        if selected_mpc != bool(baseline_mpc):
            self.influence_frames += 1
        return selected_mpc

    def authorizes_episodic(self, recommendation, confidence):
        if self.mode != "recurrent_mpc_air":
            return True
        return (
            recommendation == "episodic"
            and clamp(confidence) >= self.confidence_threshold
        )
