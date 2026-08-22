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
from pgnw_multi_hypothesis_arbitration_lab import CandidateRoute, arbitrate
from typed_causal_domain import ORDERED_ACTIONS


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
        typed_learner=None,
        typed_rule_control="passive",
        multi_hypothesis_arbitration="disabled",
        arbitration_hazard_cost=0.25,
        metabolic_learner=None,
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
        self.typed_learner = typed_learner
        self.typed = typed_learner is not None and typed_learner.enabled
        if typed_rule_control not in {"passive", "verified_protective"}:
            raise ValueError("unsupported_typed_rule_control")
        self.typed_rule_control = str(typed_rule_control)
        if multi_hypothesis_arbitration not in {
            "disabled", "passive", "bounded_verified", "bounded_dual_verified"
        }:
            raise ValueError("unsupported_multi_hypothesis_arbitration")
        self.multi_hypothesis_arbitration = str(
            multi_hypothesis_arbitration
        )
        self.arbitration_hazard_cost = max(
            0.0, min(1.0, float(arbitration_hazard_cost))
        )
        self.metabolic_learner = metabolic_learner
        self.arbitration_active = False
        self.arbitration_selected_feature = "none"
        self.arbitration_selected_score = 0.0
        self.arbitration_records = []
        self.arbitration_evaluations = 0
        self.arbitration_agreement_frames = 0
        self.arbitration_authority = 0.0
        self.arbitration_control_frames = 0
        self.arbitration_score_margin = 0.0
        self.arbitration_suppression_probability = 0.0
        self.arbitration_metabolic_relief_probability = 0.0
        self.arbitration_hunger_urgency = 0.0
        self.arbitration_memory_target = None
        self.arbitration_denial_reason = "inactive"
        self.arbitration_min_score_margin = 0.02
        self.arbitration_min_suppression_probability = 0.50
        self.arbitration_action_influence = 0
        if (
            self.multi_hypothesis_arbitration
            in {"bounded_verified", "bounded_dual_verified"}
            and self.typed_rule_control != "verified_protective"
        ):
            raise ValueError(
                "bounded_arbitration_requires_verified_protective"
            )
        if (
            self.multi_hypothesis_arbitration == "bounded_dual_verified"
            and self.metabolic_learner is None
        ):
            raise ValueError("dual_arbitration_requires_metabolic_learner")
        self.protective_rule_active = False
        self.protective_target_feature = "none"
        self.protective_rule_confidence = 0.0
        self.protective_memory_active = False
        self.protective_need_active = False
        self.protective_guidance_decisions = 0
        self.protective_action_influence = 0
        self.typed_requested_action = "none"
        self.typed_last_terminal_count = 0
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
        self.last_blue_total = 0
        self.last_yellow_total = 0
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
        if self.typed:
            return self.typed_requested_action
        return (
            ACTIONS[self.current_action]
            if self.current_action is not None
            else "none"
        )

    @property
    def map_hypothesis(self):
        if self.typed:
            return self.typed_learner.pool.map_hypothesis
        if self.dynamic:
            return self.dynamic_pool.map_hypothesis
        return HYPOTHESES[int(np.argmax(self.posterior))]

    @property
    def map_confidence(self):
        if self.typed:
            return self.typed_learner.pool.map_confidence
        if self.dynamic:
            return self.dynamic_pool.map_confidence
        return float(np.max(self.posterior))

    def select_experiment(self):
        if self.typed:
            action, _scores = self.typed_learner.pool.select_experiment()
            self.typed_requested_action = ORDERED_ACTIONS[action]
            self.phase = "seeking_red"
            self.experiments_selected += 1
            return
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

    def verified_protective_suppressor(self):
        """Return a learned specific suppressor only above the authority gate."""
        if not self.typed or self.typed_rule_control != "verified_protective":
            return None
        pool = self.typed_learner.pool
        index = int(np.argmax(pool.posterior))
        hypothesis = pool.hypotheses[index]
        confidence = float(pool.posterior[index])
        self.protective_rule_confidence = confidence
        if (
            confidence < 0.95
            or hypothesis.relation != "after_red"
            or hypothesis.suppressor not in {"blue", "yellow"}
        ):
            return None
        return hypothesis.suppressor

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

    def update(
        self,
        step,
        signal,
        pickup_total,
        red_total,
        blue_total=None,
        yellow_total=None,
    ):
        if not self.enabled:
            return
        if self.typed:
            terminal_count = (
                self.typed_learner.completed_episodes
                + self.typed_learner.discarded_episodes
            )
            if self.typed_learner.active is None:
                if terminal_count != self.typed_last_terminal_count:
                    self.typed_last_terminal_count = terminal_count
                    self.select_experiment()
                else:
                    self.phase = "seeking_red"
            elif self.typed_learner.active["second"] == "none":
                self.phase = "seeking_second"
            else:
                self.phase = "observing_delay"
            return
        self._poll_dynamic_proposal()
        pickup_total = int(pickup_total)
        red_total = int(red_total)
        blue_total = (
            max(0, pickup_total - red_total)
            if blue_total is None
            else int(blue_total)
        )
        yellow_total = 0 if yellow_total is None else int(yellow_total)
        if not self.initialized:
            self.last_pickup_total = pickup_total
            self.last_red_total = red_total
            self.last_blue_total = blue_total
            self.last_yellow_total = yellow_total
            self.initialized = True
            if self.requested_experiment == "wait_no_pickup":
                self.start_observation(2, step, signal)
            return

        pickup_delta = max(0, pickup_total - self.last_pickup_total)
        red_delta = max(0, red_total - self.last_red_total)
        blue_delta = max(0, blue_total - self.last_blue_total)
        yellow_delta = max(0, yellow_total - self.last_yellow_total)
        self.last_pickup_total = pickup_total
        self.last_red_total = red_total
        self.last_blue_total = blue_total
        self.last_yellow_total = yellow_total

        if self.phase == "waiting_to_observe":
            self.start_observation(2, step, signal)

        if self.phase == "seeking_target" and yellow_delta > 0:
            # The legacy red/blue planner has no typed yellow action. Treating
            # every non-red pickup as blue would corrupt both discovery and
            # held-out evidence, so yellow is an explicit protocol mismatch.
            self.protocol_mismatches += 1
            self.experiments_discarded += 1
            self.last_outcome = "unexpected_yellow"
            self.select_experiment()
        elif self.phase == "seeking_target" and pickup_delta > 0:
            if (
                pickup_delta != 1
                or red_delta > 1
                or blue_delta > 1
                or red_delta + blue_delta != pickup_delta
            ):
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

    def update_guidance(
        self,
        body_state,
        resource_memory=None,
        protective_need_active=None,
        protective_deadline_remaining_seconds=None,
        metabolic_need_urgency=0.0,
    ):
        self.guidance_active = False
        self.isolation_active = False
        self.guidance_vector = (0.0, 0.0)
        self.guidance_distance = 0.0
        self.guidance_weight = 0.0
        self.last_effective_guidance_weight = 0.0
        self.protective_rule_active = False
        self.protective_target_feature = "none"
        self.protective_memory_active = False
        self.protective_need_active = False
        self.arbitration_active = False
        self.arbitration_selected_feature = "none"
        self.arbitration_selected_score = 0.0
        self.arbitration_records = []
        self.arbitration_authority = 0.0
        self.arbitration_score_margin = 0.0
        self.arbitration_suppression_probability = 0.0
        self.arbitration_metabolic_relief_probability = 0.0
        self.arbitration_hunger_urgency = max(
            0.0, min(1.0, float(metabolic_need_urgency))
        )
        self.arbitration_memory_target = None
        self.arbitration_denial_reason = "inactive"
        cooling_down = self.commitment_cooldown_remaining_ticks > 0
        if cooling_down:
            self.commitment_cooldown_remaining_ticks -= 1
        if (
            self.mode not in {"bounded", "committed", "dynamic_committed"}
            or not isinstance(body_state, dict)
        ):
            return
        protective_suppressor = self.verified_protective_suppressor()
        if self.typed_rule_control == "verified_protective":
            active_episode = self.typed_learner.active if self.typed else None
            if protective_need_active is None:
                # Direct planner callers retain the legacy test harness; the
                # embodied loop supplies the authoritative metabolic state.
                protective_need_active = active_episode is not None
            if (
                self.multi_hypothesis_arbitration
                in {"passive", "bounded_verified", "bounded_dual_verified"}
                and bool(protective_need_active)
                and (
                    self.multi_hypothesis_arbitration
                    != "bounded_dual_verified"
                    or self.arbitration_hunger_urgency > 0.0
                )
                and resource_memory is not None
                and bool(getattr(resource_memory, "enabled", False))
            ):
                try:
                    position_x = float(body_state.get("x", 0.0))
                    position_z = float(body_state.get("z", 0.0))
                except (TypeError, ValueError):
                    position_x = position_z = 0.0
                candidates = []
                candidate_targets = {}
                for feature in ("yellow", "blue"):
                    selected = resource_memory.best_typed_region(
                        feature,
                        position_x,
                        position_z,
                        retain_arrived=bool(
                            body_state.get(
                                f"{feature}_food_visible",
                                False,
                            )
                        ),
                    )
                    if selected is None:
                        continue
                    _score, negative_distance, _key, entry = selected
                    candidates.append(
                        CandidateRoute(
                            feature=feature,
                            distance=-float(negative_distance),
                            memory_confidence=float(entry.confidence),
                        )
                    )
                    if hasattr(entry, "x") and hasattr(entry, "z"):
                        candidate_targets[feature] = (
                            float(entry.x),
                            float(entry.z),
                            -float(negative_distance),
                        )
                if candidates:
                    deadline = protective_deadline_remaining_seconds
                    if deadline is None:
                        deadline = self.delay_ticks / self.hz
                    result = arbitrate(
                        self.typed_learner.pool,
                        candidates,
                        hazard_cost=self.arbitration_hazard_cost,
                        deadline_remaining=max(0.001, float(deadline)),
                        metabolic_pool=(
                            self.metabolic_learner.pool
                            if self.multi_hypothesis_arbitration
                            == "bounded_dual_verified"
                            else None
                        ),
                        hunger_urgency=self.arbitration_hunger_urgency,
                    )
                    self.arbitration_active = True
                    self.arbitration_selected_feature = result[
                        "selected_feature"
                    ]
                    self.arbitration_selected_score = float(
                        result["selected_score"]
                    )
                    self.arbitration_records = result["records"]
                    self.arbitration_evaluations += 1
                    if self.arbitration_selected_feature == protective_suppressor:
                        self.arbitration_agreement_frames += 1
                    eligible = sorted(
                        (
                            record for record in result["records"]
                            if record["eligible"]
                        ),
                        key=lambda record: record["score"],
                        reverse=True,
                    )
                    if eligible:
                        self.arbitration_score_margin = (
                            math.inf if len(eligible) == 1 else
                            float(eligible[0]["score"] - eligible[1]["score"])
                        )
                        winner = next(
                            record for record in eligible
                            if record["candidate"]["feature"]
                            == self.arbitration_selected_feature
                        )
                        self.arbitration_suppression_probability = float(
                            winner["suppression_probability"]
                        )
                        self.arbitration_metabolic_relief_probability = float(
                            winner.get("metabolic_relief_probability", 0.0)
                        )
                    if self.multi_hypothesis_arbitration == "passive":
                        self.arbitration_denial_reason = "passive_only"
                    elif protective_suppressor is None:
                        self.arbitration_denial_reason = "unverified_production"
                    elif (
                        self.multi_hypothesis_arbitration
                        == "bounded_dual_verified"
                        and (
                            self.metabolic_learner.status != "verified_held_out"
                            or self.metabolic_learner.admitted_nutrient
                            not in {"blue", "yellow"}
                        )
                    ):
                        self.arbitration_denial_reason = "unverified_metabolic_rule"
                    elif (
                        self.multi_hypothesis_arbitration
                        != "bounded_dual_verified"
                        and
                        self.arbitration_selected_feature
                        != protective_suppressor
                    ):
                        self.arbitration_denial_reason = "verified_disagreement"
                    elif self.multi_hypothesis_arbitration == "bounded_dual_verified" and (
                        self.arbitration_selected_feature == protective_suppressor
                        and self.arbitration_suppression_probability
                        < self.arbitration_min_suppression_probability
                        or self.arbitration_selected_feature
                        == self.metabolic_learner.admitted_nutrient
                        and self.arbitration_metabolic_relief_probability
                        < self.arbitration_min_suppression_probability
                        or self.arbitration_selected_feature
                        not in {
                            protective_suppressor,
                            self.metabolic_learner.admitted_nutrient,
                        }
                    ):
                        self.arbitration_denial_reason = "winner_not_verified"
                    elif (
                        self.multi_hypothesis_arbitration != "bounded_dual_verified"
                        and self.arbitration_suppression_probability
                        < self.arbitration_min_suppression_probability
                    ):
                        self.arbitration_denial_reason = "low_suppression_probability"
                    elif (
                        self.arbitration_score_margin
                        < self.arbitration_min_score_margin
                    ):
                        self.arbitration_denial_reason = "low_score_margin"
                    else:
                        self.arbitration_authority = 1.0
                        self.arbitration_control_frames += 1
                        self.arbitration_denial_reason = "none"
                        self.arbitration_memory_target = candidate_targets.get(
                            self.arbitration_selected_feature
                        )
            if protective_suppressor is None or not bool(protective_need_active):
                return
            self.protective_rule_active = True
            self.protective_need_active = True
            self.protective_target_feature = protective_suppressor
            if self.arbitration_authority > 0.0:
                self.protective_target_feature = (
                    self.arbitration_selected_feature
                )
        if (
            self.mode in {"committed", "dynamic_committed"}
            and self.phase == "observing_delay"
            and not self.confounded
            and not self.protective_rule_active
        ):
            repulsion_x = 0.0
            repulsion_z = 0.0
            nearest = math.inf
            for prefix in ("red", "blue", "yellow"):
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
        if self.typed:
            if self.protective_rule_active:
                prefix = self.protective_target_feature
            elif self.phase == "seeking_red":
                prefix = "red"
            elif self.phase == "seeking_second":
                if self.typed_requested_action == "red_then_yellow":
                    prefix = "yellow"
                elif self.typed_requested_action == "red_then_blue":
                    prefix = "blue"
                else:
                    return
            else:
                return
        else:
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
            if (
                self.protective_rule_active
                and self.arbitration_authority > 0.0
                and self.arbitration_memory_target is not None
            ):
                target_x, target_z, distance = self.arbitration_memory_target
                try:
                    position_x = float(body_state.get("x", 0.0))
                    position_z = float(body_state.get("z", 0.0))
                except (TypeError, ValueError):
                    position_x = position_z = 0.0
                x = target_x - position_x
                z = target_z - position_z
                length = math.hypot(x, z)
                self.protective_memory_active = length > 1e-6
            elif (
                self.protective_rule_active
                and resource_memory is not None
                and bool(getattr(resource_memory, "active", False))
                and getattr(resource_memory, "active_feature", "none") == prefix
                and getattr(resource_memory, "target", None) is not None
            ):
                x, z = resource_memory.guidance_vector
                distance = resource_memory.distance
                length = math.hypot(x, z)
                self.protective_memory_active = length > 1e-6
            if length <= 1e-6 or not self.protective_memory_active:
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
        if self.protective_rule_active:
            self.protective_guidance_decisions += 1

    def audit(self):
        posterior = (
            self.typed_learner.pool.audit()["posterior"]
            if self.typed
            else self.dynamic_pool.audit()["posterior"]
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
            "typed_guidance": self.typed,
            "typed_rule_control": self.typed_rule_control,
            "protective_rule_active": self.protective_rule_active,
            "protective_target_feature": self.protective_target_feature,
            "protective_rule_confidence": self.protective_rule_confidence,
            "protective_memory_active": self.protective_memory_active,
            "protective_need_active": self.protective_need_active,
            "protective_guidance_decisions": self.protective_guidance_decisions,
            "protective_action_influence": self.protective_action_influence,
            "multi_hypothesis_arbitration": (
                self.multi_hypothesis_arbitration
            ),
            "arbitration_active": self.arbitration_active,
            "arbitration_selected_feature": (
                self.arbitration_selected_feature
            ),
            "arbitration_selected_score": self.arbitration_selected_score,
            "arbitration_records": self.arbitration_records,
            "arbitration_evaluations": self.arbitration_evaluations,
            "arbitration_agreement_frames": (
                self.arbitration_agreement_frames
            ),
            "arbitration_authority": self.arbitration_authority,
            "arbitration_control_frames": self.arbitration_control_frames,
            "arbitration_score_margin": self.arbitration_score_margin,
            "arbitration_suppression_probability": (
                self.arbitration_suppression_probability
            ),
            "arbitration_metabolic_relief_probability": (
                self.arbitration_metabolic_relief_probability
            ),
            "arbitration_hunger_urgency": self.arbitration_hunger_urgency,
            "arbitration_denial_reason": self.arbitration_denial_reason,
            "arbitration_min_score_margin": (
                self.arbitration_min_score_margin
            ),
            "arbitration_min_suppression_probability": (
                self.arbitration_min_suppression_probability
            ),
            "arbitration_action_influence": (
                self.arbitration_action_influence
            ),
            "typed_learner": (
                self.typed_learner.audit() if self.typed else {}
            ),
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
