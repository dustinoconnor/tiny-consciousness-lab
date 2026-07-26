#!/usr/bin/env python3
"""Passive reward-learning conductor for the embodied Unity controller stack."""

from __future__ import annotations

import math
import json
from collections import deque
from pathlib import Path


SPECIALISTS = ("recurrent", "episodic", "predictive", "fallback")
CONTEXTS = (
    "clear_exploration",
    "familiar_hidden_goal",
    "visible_target",
    "reactive_obstacle",
    "recovery_crisis",
)
COMPUTE_COST = {
    "recurrent": 0.01,
    "episodic": 0.03,
    "predictive": 0.08,
    "fallback": 0.12,
}


def clamp(value, lower=0.0, upper=1.0):
    return float(max(lower, min(upper, value)))


def wrapped_yaw_delta(first, second):
    return abs((float(second) - float(first) + 180.0) % 360.0 - 180.0)


class PassiveEmbodiedConductor:
    """Learn controller utility while remaining causally disconnected from action."""

    def __init__(
        self,
        enabled=False,
        learning_rate=0.18,
        protocol_size=32,
        checkpoint=None,
    ):
        self.enabled = bool(enabled)
        self.learning_rate = float(learning_rate)
        self.q = {
            context: {specialist: 0.0 for specialist in SPECIALISTS}
            for context in CONTEXTS
        }
        self.visits = {
            context: {specialist: 0 for specialist in SPECIALISTS}
            for context in CONTEXTS
        }
        self.protocol = deque(maxlen=max(4, int(protocol_size)))
        self.previous = None
        self.episode_key = None
        self.context = "unobserved"
        self.recommendation = "insufficient_evidence"
        self.active_specialist = "unobserved"
        self.confidence = 0.0
        self.agreement = False
        self.agreement_rate = 0.0
        self.last_reward = 0.0
        self.updates = 0
        self.agreements = 0
        self.observations = 0
        self.recommendation_observations = 0
        self.episode_resets = 0
        self.action_influence = 0
        self.checkpoint = "none"
        self.learning_enabled = checkpoint is None
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        path = Path(checkpoint).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_type") != "passive_embodied_conductor":
            raise ValueError("unsupported_conductor_checkpoint")
        for context, values in payload.get("q_values", {}).items():
            if context not in self.q:
                continue
            for specialist, value in values.items():
                if specialist in self.q[context]:
                    self.q[context][specialist] = float(value)
        for context, values in payload.get("visits", {}).items():
            if context not in self.visits:
                continue
            for specialist, value in values.items():
                if specialist in self.visits[context]:
                    self.visits[context][specialist] = max(0, int(value))
        self.checkpoint = str(path)

    @staticmethod
    def classify_context(snapshot):
        if (
            snapshot.get("fallback_active")
            or float(snapshot.get("physics_wedge_seconds", 0.0)) >= 2.0
            or float(snapshot.get("trap_accumulation_seconds", 0.0)) >= 2.0
        ):
            return "recovery_crisis"
        if snapshot.get("food_visible"):
            return "visible_target"
        if (
            snapshot.get("hidden_goal_active")
            or snapshot.get("hidden_goal_route_selected_id") not in {None, "", "none"}
        ):
            return "familiar_hidden_goal"
        if (
            snapshot.get("blocked")
            or snapshot.get("body_collision")
            or int(snapshot.get("body_safe_actions", 8)) <= 3
        ):
            return "reactive_obstacle"
        return "clear_exploration"

    @staticmethod
    def identify_specialist(snapshot):
        if snapshot.get("fallback_active"):
            return "fallback"
        if snapshot.get("hidden_goal_active"):
            return "episodic"
        if snapshot.get("mpc_engaged"):
            return "predictive"
        return "recurrent"

    @staticmethod
    def transition_reward(previous, current):
        reward = -COMPUTE_COST[previous["active_specialist"]]
        distance = math.hypot(
            float(current.get("x", 0.0)) - float(previous.get("x", 0.0)),
            float(current.get("z", 0.0)) - float(previous.get("z", 0.0)),
        )
        if distance >= 0.05:
            reward += 0.06
        if current.get("body_collision"):
            reward -= 0.40
        if current.get("blocked"):
            reward -= 0.25
        if current.get("stuck"):
            reward -= 0.35

        if previous.get("food_visible") or current.get("food_visible"):
            yaw_delta = wrapped_yaw_delta(previous.get("yaw", 0.0), current.get("yaw", 0.0))
            if yaw_delta > 30.0:
                reward -= min(0.50, (yaw_delta - 30.0) / 180.0 * 0.50)
        if previous.get("food_visible"):
            if current.get("food_visible"):
                progress = float(previous.get("food_distance", 0.0)) - float(
                    current.get("food_distance", 0.0)
                )
                reward += clamp(progress / 2.0, -0.30, 0.30)
            else:
                reward -= 0.20

        if previous.get("hidden_goal_route_selected_id") not in {None, "", "none"}:
            index_gain = int(current.get("route_index", 0)) - int(
                previous.get("route_index", 0)
            )
            reward += clamp(index_gain * 0.04, -0.20, 0.30)
            route_gain = float(previous.get("route_distance", 0.0)) - float(
                current.get("route_distance", 0.0)
            )
            reward += clamp(route_gain / 4.0, -0.20, 0.20)

        previous_outcome = str(previous.get("trap_outcome", "inactive"))
        outcome = str(current.get("trap_outcome", "inactive"))
        if previous_outcome == "running" and outcome == "success":
            reward += 2.0
        elif previous_outcome == "running" and outcome == "timeout":
            reward -= 2.0

        previous_wedge = float(previous.get("physics_wedge_seconds", 0.0))
        current_wedge = float(current.get("physics_wedge_seconds", 0.0))
        if previous["active_specialist"] == "fallback" and previous_wedge >= 1.0:
            reward += clamp((previous_wedge - current_wedge) / 4.0, 0.0, 0.35)
        return float(max(-2.5, min(2.5, reward)))

    def _update(self, context, specialist, reward):
        self.visits[context][specialist] += 1
        visits = self.visits[context][specialist]
        rate = self.learning_rate / math.sqrt(1.0 + 0.025 * visits)
        value = self.q[context][specialist]
        self.q[context][specialist] = value + rate * (reward - value)
        self.updates += 1

    def _recommend(self, context):
        eligible = [
            specialist
            for specialist in SPECIALISTS
            if self.visits[context][specialist] >= 3
        ]
        if len(eligible) < 2:
            return "insufficient_evidence", 0.0
        total_visits = sum(self.visits[context][specialist] for specialist in eligible)
        ranked = sorted(
            eligible,
            key=lambda specialist: (
                self.q[context][specialist],
                self.visits[context][specialist],
                -SPECIALISTS.index(specialist),
            ),
            reverse=True,
        )
        best, second = ranked[:2]
        gap = max(0.0, self.q[context][best] - self.q[context][second])
        support = min(1.0, total_visits / 12.0)
        confidence = support * (0.50 + 0.50 * math.tanh(2.0 * gap))
        return best, clamp(confidence)

    def recommend(self, context):
        if context not in self.q:
            return "insufficient_evidence", 0.0
        return self._recommend(context)

    def observe(self, snapshot):
        if not self.enabled:
            return
        current = dict(snapshot)
        episode_key = (
            str(current.get("trap_course", "natural_terrain")),
            str(current.get("trap_course_variant", "standard")),
            int(current.get("trap_episode", 0)),
        )
        if self.episode_key is not None and episode_key != self.episode_key:
            self.previous = None
            self.protocol.clear()
            self.episode_resets += 1
        self.episode_key = episode_key

        reward = 0.0
        if self.previous is not None:
            reward = self.transition_reward(self.previous, current)
            if self.learning_enabled:
                self._update(
                    self.previous["context"],
                    self.previous["active_specialist"],
                    reward,
                )

        context = self.classify_context(current)
        active = self.identify_specialist(current)
        recommendation, confidence = self._recommend(context)
        agreement = recommendation == active
        self.observations += 1
        if recommendation != "insufficient_evidence":
            self.recommendation_observations += 1
            self.agreements += int(agreement)
        self.context = context
        self.recommendation = recommendation
        self.active_specialist = active
        self.confidence = confidence
        self.agreement = agreement
        self.agreement_rate = self.agreements / max(
            self.recommendation_observations, 1
        )
        self.last_reward = reward
        self.protocol.append(
            {
                "context": context,
                "recommendation": recommendation,
                "active_specialist": active,
                "confidence": confidence,
                "agreement": agreement,
                "grounded_reward": reward,
                "trap_outcome": str(current.get("trap_outcome", "inactive")),
            }
        )
        current["context"] = context
        current["active_specialist"] = active
        self.previous = current

    @property
    def context_visits(self):
        if self.context not in self.visits:
            return 0
        return sum(self.visits[self.context].values())

    @property
    def q_values(self):
        if self.context not in self.q:
            return {specialist: 0.0 for specialist in SPECIALISTS}
        return dict(self.q[self.context])
