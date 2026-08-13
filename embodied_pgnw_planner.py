#!/usr/bin/env python3
"""Stateful PGNW experiment requests for passive Unity causal probes."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dynamic_hypothesis_pool import (
    DynamicHypothesisPool,
    compile_l1_hypothesis,
)

from pgnw_hypothesis_selection_lab import (
    ACTIONS,
    HYPOTHESES,
    action_scores,
    entropy_bits,
    posterior_after,
    select_ranked,
)


class EmbodiedPGNWExperimentPlanner:
    """Choose scientific observations without issuing motor commands."""

    MODES = ("disabled", "passive", "bounded", "committed", "dynamic_committed")

    def __init__(
        self,
        mode="disabled",
        hz=5.0,
        seed=0,
        delay_seconds=10.0,
        rise_threshold=0.12,
        max_guidance_weight=0.03,
        guidance_margin_threshold=0.08,
        commitment_seconds=3.0,
        commitment_cooldown_seconds=2.0,
        max_score_regret=0.18,
        isolation_retreat_seconds=2.0,
        isolation_retreat_max_score_regret=0.35,
        hypothesis_proposer=None,
        dynamic_discovery_minimum=4,
    ):
        if mode not in self.MODES:
            raise ValueError("unsupported_embodied_pgnw_mode")
        self.mode = mode
        self.enabled = mode != "disabled"
        self.hz = max(float(hz), 0.1)
        self.delay_ticks = max(1, int(round(delay_seconds * self.hz)))
        self.rise_threshold = max(0.0, float(rise_threshold))
        self.max_guidance_weight = max(
            0.0, min(0.20, float(max_guidance_weight))
        )
        self.guidance_margin_threshold = max(
            0.01, min(0.40, float(guidance_margin_threshold))
        )
        self.commitment_ticks = max(1, int(round(commitment_seconds * self.hz)))
        self.commitment_cooldown_ticks = max(
            0, int(round(commitment_cooldown_seconds * self.hz))
        )
        self.max_score_regret = max(0.0, min(0.35, float(max_score_regret)))
        self.isolation_retreat_ticks = max(
            1, int(round(isolation_retreat_seconds * self.hz))
        )
        self.isolation_retreat_max_score_regret = max(
            self.max_score_regret,
            min(0.35, float(isolation_retreat_max_score_regret)),
        )
        self.tie_order = np.random.default_rng(seed).permutation(len(ACTIONS))
        self.dynamic = mode == "dynamic_committed"
        if self.dynamic and hypothesis_proposer is None:
            raise ValueError("dynamic_pgnw_requires_hypothesis_proposer")
        self.hypothesis_proposer = hypothesis_proposer
        self.dynamic_discovery_minimum = max(2, int(dynamic_discovery_minimum))
        self.dynamic_pool = DynamicHypothesisPool() if self.dynamic else None
        self.dynamic_observations = []
        self.dynamic_proposal_future = None
        self.dynamic_proposal_summary = None
        self.dynamic_proposal_cutoff = 0
        self.dynamic_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tiny-scientist")
            if self.dynamic
            else None
        )
        self.dynamic_proposal_status = "collecting" if self.dynamic else "disabled"
        self.dynamic_raw_l1 = ""
        self.dynamic_generation_metrics = {}
        self.dynamic_admission_error = ""
        self.posterior = np.full(
            len(HYPOTHESES), 1.0 / len(HYPOTHESES), dtype=np.float64
        )
        self.current_action = None
        self.phase = "disabled" if not self.enabled else "selecting"
        self.observed_action = None
        self.observation_start_step = None
        self.observation_due_step = None
        self.baseline_signal = 0.0
        self.confounded = False
        self.initialized = False
        self.last_pickup_total = 0
        self.last_red_total = 0
        self.last_outcome = "none"
        self.last_information_gain = 0.0
        self.experiments_selected = 0
        self.experiments_started = 0
        self.experiments_completed = 0
        self.experiments_discarded = 0
        self.protocol_mismatches = 0
        self.posterior_updates = 0
        self.guidance_active = False
        self.guidance_vector = (0.0, 0.0)
        self.guidance_distance = 0.0
        self.guidance_weight = 0.0
        self.last_effective_guidance_weight = 0.0
        self.guidance_decisions = 0
        self.action_influence = 0
        self.commitment_remaining_ticks = 0
        self.commitment_cooldown_remaining_ticks = 0
        self.commitment_starts = 0
        self.commitment_expirations = 0
        self.isolation_active = False
        self.isolation_decisions = 0
        self.isolation_action_influence = 0
        self.isolation_intervening_pickups = 0
        self.isolation_food_suppression_frames = 0
        self.isolation_retreat_remaining_ticks = 0
        self.isolation_retreat_decisions = 0
        self.isolation_retreat_action_influence = 0
        if self.enabled:
            self.select_experiment()

    @property
    def requested_experiment(self):
        return (
            ACTIONS[self.current_action]
            if self.current_action is not None
            else "none"
        )

    @property
    def map_hypothesis(self):
        if self.dynamic:
            return self.dynamic_pool.map_hypothesis
        return HYPOTHESES[int(np.argmax(self.posterior))]

    @property
    def map_confidence(self):
        if self.dynamic:
            return self.dynamic_pool.map_confidence
        return float(np.max(self.posterior))

    def select_experiment(self):
        if self.dynamic:
            if self.dynamic_pool.admission_count:
                self.current_action, _scores = self.dynamic_pool.select_experiment()
            else:
                # Before any causal ontology exists, perform a seeded balanced
                # survey rather than allowing the two null models to request
                # no-pickup waits forever.
                self.current_action = int(
                    self.tie_order[self.experiments_selected % len(ACTIONS)]
                )
        else:
            scores = action_scores("pgnw_efe", self.posterior)
            self.current_action = select_ranked(scores, self.tie_order)
        self.phase = (
            "waiting_to_observe"
            if ACTIONS[self.current_action] == "wait_no_pickup"
            else "seeking_target"
        )
        self.observed_action = None
        self.observation_start_step = None
        self.observation_due_step = None
        self.confounded = False
        self.experiments_selected += 1

    def _dynamic_summary(self):
        features = {"red": {"deltas": [], "rises": 0}, "blue": {"deltas": [], "rises": 0}}
        for action, rise, delta in self.dynamic_observations:
            if action not in {0, 1}:
                continue
            name = "red" if action == 0 else "blue"
            values = features[name]
            values["deltas"].append(float(delta))
            values["rises"] += int(bool(rise))
        if any(not values["deltas"] for values in features.values()):
            return None
        outcomes = {}
        for name, values in features.items():
            count = len(values["deltas"])
            mean_delta = round(float(np.mean(values["deltas"])), 3)
            if abs(mean_delta) < 0.0005:
                mean_delta = 0.0
            outcomes[name] = {
                "isolated_episodes": count,
                "mean_pressure_delta": mean_delta,
                "positive_pressure_fraction": values["rises"] / count,
                "mean_positive_delay_seconds": (
                    self.delay_ticks / self.hz if values["rises"] else None
                ),
            }
        return {"feature_outcomes": outcomes, "episode_count": len(self.dynamic_observations)}

    def _poll_dynamic_proposal(self):
        if not self.dynamic or self.dynamic_pool.admission_count:
            return
        if self.dynamic_proposal_status == "rejected":
            return
        if self.dynamic_proposal_future is not None:
            if not self.dynamic_proposal_future.done():
                return
            try:
                raw, metrics = self.dynamic_proposal_future.result()
                summary = self.dynamic_proposal_summary
                candidate = compile_l1_hypothesis(
                    raw, summary, self.dynamic_proposal_cutoff
                )
                self.dynamic_pool.admit(candidate)
                self.dynamic_raw_l1 = raw
                self.dynamic_generation_metrics = dict(metrics)
                self.dynamic_proposal_status = "admitted_unverified"
                self.select_experiment()
            except Exception as exc:
                self.dynamic_admission_error = str(exc)
                self.dynamic_proposal_status = "rejected"
            self.dynamic_proposal_future = None
            self.dynamic_proposal_summary = None
            return
        summary = self._dynamic_summary()
        if (
            summary is not None
            and len(self.dynamic_observations) >= self.dynamic_discovery_minimum
        ):
            self.dynamic_proposal_status = "generating"
            self.dynamic_proposal_summary = summary
            self.dynamic_proposal_cutoff = len(self.dynamic_observations)
            self.dynamic_proposal_future = self.dynamic_executor.submit(
                self.hypothesis_proposer, summary
            )

    def start_observation(self, action, step, signal, matched=True):
        self.observed_action = int(action)
        self.observation_start_step = int(step)
        self.observation_due_step = int(step) + self.delay_ticks
        self.baseline_signal = float(signal)
        self.confounded = False
        self.phase = "observing_delay"
        self.isolation_retreat_remaining_ticks = self.isolation_retreat_ticks
        self.commitment_remaining_ticks = 0
        self.commitment_cooldown_remaining_ticks = 0
        self.experiments_started += 1
        if not matched:
            self.protocol_mismatches += 1

    def update(self, step, signal, pickup_total, red_total):
        if not self.enabled:
            return
        self._poll_dynamic_proposal()
        pickup_total = int(pickup_total)
        red_total = int(red_total)
        if not self.initialized:
            self.last_pickup_total = pickup_total
            self.last_red_total = red_total
            self.initialized = True
            if self.requested_experiment == "wait_no_pickup":
                self.start_observation(2, step, signal)
            return

        pickup_delta = max(0, pickup_total - self.last_pickup_total)
        red_delta = max(0, red_total - self.last_red_total)
        blue_delta = max(0, pickup_delta - red_delta)
        self.last_pickup_total = pickup_total
        self.last_red_total = red_total

        if self.phase == "waiting_to_observe":
            self.start_observation(2, step, signal)

        if self.phase == "seeking_target" and pickup_delta > 0:
            if pickup_delta != 1 or red_delta > 1 or blue_delta > 1:
                self.start_observation(
                    0 if red_delta else 1,
                    step,
                    signal,
                    matched=False,
                )
                self.confounded = True
            else:
                actual = 0 if red_delta == 1 else 1
                self.start_observation(
                    actual,
                    step,
                    signal,
                    matched=actual == self.current_action,
                )
        elif (
            self.phase == "observing_delay"
            and pickup_delta > 0
            and int(step) > int(self.observation_start_step)
        ):
            self.confounded = True
            self.isolation_intervening_pickups += pickup_delta

        if (
            self.phase == "observing_delay"
            and self.observation_due_step is not None
            and int(step) >= self.observation_due_step
        ):
            if self.confounded:
                self.experiments_discarded += 1
                self.last_outcome = "confounded"
            else:
                rise = float(signal) - self.baseline_signal >= self.rise_threshold
                delta = float(signal) - self.baseline_signal
                if self.dynamic:
                    self.dynamic_observations.append(
                        (int(self.observed_action), bool(rise), delta)
                    )
                    if self.dynamic_pool.admission_count:
                        before_entropy = entropy_bits(self.dynamic_pool.posterior)
                        self.dynamic_pool.update(
                            self.observed_action,
                            rise,
                            len(self.dynamic_observations),
                        )
                        self.last_information_gain = before_entropy - entropy_bits(
                            self.dynamic_pool.posterior
                        )
                        if self.dynamic_pool.map_confidence >= 0.95:
                            self.dynamic_proposal_status = "verified_held_out"
                else:
                    before_entropy = entropy_bits(self.posterior)
                    self.posterior = posterior_after(
                        self.posterior,
                        self.observed_action,
                        rise,
                    )
                    self.last_information_gain = (
                        before_entropy - entropy_bits(self.posterior)
                    )
                self.last_outcome = "probe_rise" if rise else "no_probe_rise"
                self.experiments_completed += 1
                self.posterior_updates += 1
            self.select_experiment()
            self._poll_dynamic_proposal()

    def update_guidance(self, body_state):
        self.guidance_active = False
        self.isolation_active = False
        self.guidance_vector = (0.0, 0.0)
        self.guidance_distance = 0.0
        self.guidance_weight = 0.0
        self.last_effective_guidance_weight = 0.0
        cooling_down = self.commitment_cooldown_remaining_ticks > 0
        if cooling_down:
            self.commitment_cooldown_remaining_ticks -= 1
        if (
            self.mode not in {"bounded", "committed", "dynamic_committed"}
            or not isinstance(body_state, dict)
        ):
            return
        if (
            self.mode in {"committed", "dynamic_committed"}
            and self.phase == "observing_delay"
            and not self.confounded
        ):
            repulsion_x = 0.0
            repulsion_z = 0.0
            nearest = math.inf
            for prefix in ("red", "blue"):
                try:
                    visible = bool(body_state.get(f"{prefix}_food_visible", False))
                    distance = max(
                        0.1, float(body_state.get(f"{prefix}_food_distance", 0.0))
                    )
                    x = float(body_state.get(f"{prefix}_food_world_x", 0.0))
                    z = float(body_state.get(f"{prefix}_food_world_z", 0.0))
                except (TypeError, ValueError):
                    continue
                length = math.hypot(x, z)
                if not visible or length <= 1e-6:
                    continue
                weight = 1.0 / distance
                repulsion_x -= weight * x / length
                repulsion_z -= weight * z / length
                nearest = min(nearest, distance)
            length = math.hypot(repulsion_x, repulsion_z)
            if length > 1e-6:
                self.guidance_active = True
                self.isolation_active = True
                self.guidance_vector = (
                    repulsion_x / length,
                    repulsion_z / length,
                )
                self.guidance_distance = nearest
                self.guidance_weight = self.max_guidance_weight
            return
        if self.phase != "seeking_target" or self.current_action not in {0, 1}:
            return
        prefix = "red" if self.current_action == 0 else "blue"
        try:
            visible = bool(body_state.get(f"{prefix}_food_visible", False))
            distance = float(body_state.get(f"{prefix}_food_distance", 0.0))
            x = float(body_state.get(f"{prefix}_food_world_x", 0.0))
            z = float(body_state.get(f"{prefix}_food_world_z", 0.0))
        except (TypeError, ValueError):
            return
        length = math.hypot(x, z)
        if not visible or length <= 1e-6:
            self.commitment_remaining_ticks = 0
            return
        if self.mode in {"committed", "dynamic_committed"}:
            if self.commitment_remaining_ticks <= 0:
                if cooling_down:
                    return
                self.commitment_remaining_ticks = self.commitment_ticks
                self.commitment_starts += 1
            self.commitment_remaining_ticks -= 1
            if self.commitment_remaining_ticks <= 0:
                self.commitment_expirations += 1
                self.commitment_cooldown_remaining_ticks = (
                    self.commitment_cooldown_ticks
                )
        self.guidance_active = True
        self.guidance_vector = (x / length, z / length)
        self.guidance_distance = max(0.0, distance)
        self.guidance_weight = self.max_guidance_weight

    def audit(self):
        posterior = (
            self.dynamic_pool.audit()["posterior"]
            if self.dynamic
            else {
                name: float(value)
                for name, value in zip(HYPOTHESES, self.posterior)
            }
        )
        return {
            "mode": self.mode,
            "requested_experiment": self.requested_experiment,
            "phase": self.phase,
            "observed_experiment": (
                ACTIONS[self.observed_action]
                if self.observed_action is not None
                else "none"
            ),
            "map_hypothesis": self.map_hypothesis,
            "map_confidence": self.map_confidence,
            "posterior": posterior,
            "dynamic_proposal_status": self.dynamic_proposal_status,
            "dynamic_raw_l1": self.dynamic_raw_l1,
            "dynamic_generation_metrics": self.dynamic_generation_metrics,
            "dynamic_admission_error": self.dynamic_admission_error,
            "dynamic_discovery_observations": len(self.dynamic_observations),
            "dynamic_pool": self.dynamic_pool.audit() if self.dynamic else {},
            "last_outcome": self.last_outcome,
            "last_information_gain_bits": self.last_information_gain,
            "experiments_selected": self.experiments_selected,
            "experiments_started": self.experiments_started,
            "experiments_completed": self.experiments_completed,
            "experiments_discarded": self.experiments_discarded,
            "protocol_mismatches": self.protocol_mismatches,
            "posterior_updates": self.posterior_updates,
            "guidance_active": self.guidance_active,
            "guidance_weight": self.guidance_weight,
            "guidance_margin_threshold": self.guidance_margin_threshold,
            "last_effective_guidance_weight": self.last_effective_guidance_weight,
            "guidance_decisions": self.guidance_decisions,
            "action_influence": self.action_influence,
            "commitment_active": (
                self.mode in {"committed", "dynamic_committed"}
                and self.guidance_active
                and self.commitment_remaining_ticks > 0
            ),
            "commitment_remaining_seconds": (
                self.commitment_remaining_ticks / self.hz
            ),
            "commitment_starts": self.commitment_starts,
            "commitment_expirations": self.commitment_expirations,
            "max_score_regret": self.max_score_regret,
            "isolation_active": self.isolation_active,
            "isolation_decisions": self.isolation_decisions,
            "isolation_action_influence": self.isolation_action_influence,
            "isolation_intervening_pickups": (
                self.isolation_intervening_pickups
            ),
            "isolation_food_suppression_frames": (
                self.isolation_food_suppression_frames
            ),
            "isolation_retreat_active": (
                self.isolation_active
                and self.isolation_retreat_remaining_ticks > 0
            ),
            "isolation_retreat_remaining_seconds": (
                self.isolation_retreat_remaining_ticks / self.hz
            ),
            "isolation_retreat_max_score_regret": (
                self.isolation_retreat_max_score_regret
            ),
            "isolation_retreat_decisions": self.isolation_retreat_decisions,
            "isolation_retreat_action_influence": (
                self.isolation_retreat_action_influence
            ),
        }
