#!/usr/bin/env python3
"""First Unity body loop for the functional ego.

Unity side:
    RobotUdpBridge listens on UDP 5055 and sends body state to UDP 5056.

Python side:
    This script sends up/down/left/right/idle/sleep/wake commands and updates
    a tiny fatigue/self-repair model from Unity body feedback.

This is intentionally modest. The first goal is closing the embodied cybernetic
loop:

    mind state -> body action -> world feedback -> mind update -> sleep repair
"""

import argparse
import json
import math
import random
import socket
import time
from collections import deque
from pathlib import Path

from embodied_conductor import PassiveEmbodiedConductor
from embodied_pgnw_planner import EmbodiedPGNWExperimentPlanner
from embodied_systemic_conductor import (
    BoundedExecutiveRouter,
    PassiveAdaptiveIgnitionGate,
    PassiveSystemicConductor,
)
from terrain_air_observer import PassiveTerrainAirObserver
from terrain_air_route_controller import (
    TerrainAirRouteController,
    context_vector,
    score_margin_ambiguity,
)
from terrain_escape_teacher import TerrainEscapeTeacher
from terrain_resource_memory import PassiveTerrainResourceMemory
from dynamic_hypothesis_pool import LocalL1HypothesisProposer
from tiny_scientist_production_memory import VerifiedProductionMemory
from typed_causal_domain import OrderedCausalProbeWorld, PassiveOrderedEpisodeLearner
from formulate_ordered_interaction import LocalCalibratedOrderedProposer


ACTIONS = [
    "up",
    "up_left",
    "up_right",
    "left",
    "right",
    "down",
    "down_left",
    "down_right",
    "idle",
]

MOVE_VECTORS = {
    "up": (0.0, 1.0),
    "up_left": (-0.55, 1.0),
    "up_right": (0.55, 1.0),
    "left": (-1.0, 0.45),
    "right": (1.0, 0.45),
    "down": (0.0, -1.0),
    "down_left": (-0.65, -0.75),
    "down_right": (0.65, -0.75),
    "idle": (0.0, 0.0),
    "sleep": (0.0, 0.0),
    "wake": (0.0, 0.0),
    "unstuck_respawn": (0.0, 0.0),
    "seek_food": (0.0, 1.0),
    "avoid_obstacle": (0.0, 1.0),
}

ACTION_BUCKETS = {
    "up": "forward",
    "up_left": "forward_left",
    "up_right": "forward_right",
    "left": "left",
    "right": "right",
    "down": "back",
    "down_left": "back_left",
    "down_right": "back_right",
    "seek_food": "food",
    "avoid_obstacle": "forward",
}

ACTION_CELL_DELTAS = {
    "up": (0, 1),
    "up_left": (-1, 1),
    "up_right": (1, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "down": (0, -1),
    "down_left": (-1, -1),
    "down_right": (1, -1),
    "seek_food": (0, 1),
    "avoid_obstacle": (0, 1),
}

SHADOW_ACTIONS = ["up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"]
SHADOW_VECTORS = {
    "up": (0.0, 1.0),
    "up_right": (1.0, 1.0),
    "right": (1.0, 0.0),
    "down_right": (1.0, -1.0),
    "down": (0.0, -1.0),
    "down_left": (-1.0, -1.0),
    "left": (-1.0, 0.0),
    "up_left": (-1.0, 1.0),
}


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def route_memory_adapter(adapter_type):
    return adapter_type in {"episodic_route", "art_route_library"}


def smooth_target_intercept(previous_move, target_delta, target_distance, max_turn_degrees=55.0):
    target_length = math.hypot(*target_delta)
    if target_length <= 1e-6:
        return (0.0, 0.0)
    target_angle = math.atan2(target_delta[1], target_delta[0])
    previous_length = math.hypot(*previous_move)
    if previous_length <= 1e-6:
        output_angle = target_angle
        remaining_error = 0.0
    else:
        previous_angle = math.atan2(previous_move[1], previous_move[0])
        angle_delta = (target_angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi
        max_turn = math.radians(max(0.0, float(max_turn_degrees)))
        applied_turn = max(-max_turn, min(max_turn, angle_delta))
        output_angle = previous_angle + applied_turn
        remaining_error = abs(angle_delta - applied_turn)
    alignment_speed = clamp(1.0 - remaining_error / math.pi, 0.20, 1.0)
    speed = clamp(float(target_distance) / 1.5, 0.25, 1.0) * alignment_speed
    return (math.cos(output_angle) * speed, math.sin(output_angle) * speed)


def route_waypoint_radius(
    index,
    route_count,
    standard_radius,
    terminal_radius,
    terminal_count,
):
    terminal_start = max(0, int(route_count) - max(1, int(terminal_count)))
    if index < terminal_start:
        return float(standard_radius)
    if index >= int(route_count) - 1:
        return float(terminal_radius)
    progress = (int(index) - terminal_start + 1) / max(
        1, int(route_count) - terminal_start
    )
    interpolated = float(standard_radius) + (
        float(terminal_radius) - float(standard_radius)
    ) * progress
    return max(float(terminal_radius) + 0.25, interpolated)


def route_reversal_count(waypoints, threshold_degrees=90.0):
    reversals = 0
    for first, second, third in zip(waypoints, waypoints[1:], waypoints[2:]):
        incoming = (second[0] - first[0], second[1] - first[1])
        outgoing = (third[0] - second[0], third[1] - second[1])
        incoming_length = math.hypot(*incoming)
        outgoing_length = math.hypot(*outgoing)
        if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
            continue
        cosine = clamp(
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (incoming_length * outgoing_length),
            -1.0,
            1.0,
        )
        if math.degrees(math.acos(cosine)) > float(threshold_degrees):
            reversals += 1
    return reversals


def art_route_selection_score(match, quality, route):
    alignment = clamp((float(route.get("terminal_alignment", 1.0)) + 1.0) * 0.5)
    terminal_factor = 0.35 + 0.65 * alignment
    return float(match) * clamp(float(quality)) * terminal_factor


def conductor_gate_allows(
    control_mode,
    context,
    recommendation,
    confidence,
    confidence_threshold=0.50,
):
    if control_mode == "passive":
        return True
    if control_mode != context:
        return True
    return (
        recommendation == "episodic"
        and float(confidence) >= float(confidence_threshold)
    )


def pgnw_experiment_guidance_allowed(
    mode,
    guidance_active,
    fallback_active,
    stuck,
    hunger,
    air_guided,
    resource_guided,
):
    """Keep scientific requests subordinate to survival and route controllers."""
    return bool(
        mode in {"bounded", "committed", "dynamic_committed"}
        and guidance_active
        and not fallback_active
        and not stuck
        and float(hunger) < 0.92
        and not air_guided
        and not resource_guided
    )


def pgnw_constrained_target_action(scores, alignments, max_score_regret):
    """Choose target alignment only among finite actions near the MPC optimum."""
    values = [float(value) for value in scores]
    target = [float(value) for value in alignments]
    finite = [index for index, value in enumerate(values) if math.isfinite(value)]
    if not finite or len(values) != len(target):
        return None
    best = max(values[index] for index in finite)
    regret = max(0.0, float(max_score_regret))
    admissible = [
        index for index in finite if values[index] >= best - regret
    ]
    return max(
        admissible,
        key=lambda index: (target[index], values[index], -index),
    )


def art_route_context(body_state):
    """Return the normalized egocentric geometry used for route resonance."""
    raw = body_state.get("directional_rays", []) if isinstance(body_state, dict) else []
    if not isinstance(raw, list) or len(raw) != 8:
        return None
    try:
        return [clamp(float(value)) for value in raw]
    except (TypeError, ValueError):
        return None


def fuzzy_art_similarity(features, prototype):
    """Complement-coded Fuzzy ART match; equivalent to one minus mean L1 error."""
    if features is None or len(features) != len(prototype) or not features:
        return 0.0
    encoded = features + [1.0 - value for value in features]
    template = list(prototype) + [1.0 - clamp(float(value)) for value in prototype]
    return clamp(sum(min(value, weight) for value, weight in zip(encoded, template)) / len(features))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def stable_recovery_due(
    physics_wedge_ticks,
    trap_accumulation_ticks,
    threshold_ticks,
    sustained_orbit=False,
):
    """Return true when any independent recovery detector identifies perseveration."""
    return (
        physics_wedge_ticks >= threshold_ticks
        or trap_accumulation_ticks >= threshold_ticks
        or bool(sustained_orbit)
    )


def orbit_teacher_due(sustained_orbit, route_active=False, route_pending=False):
    """Give episodic guidance one bounded attempt before invoking the teacher."""
    return bool(sustained_orbit) and not (bool(route_active) or bool(route_pending))


def trajectory_orbit_metrics(samples, min_path=14.0, max_efficiency=0.22, min_evidence=0.06):
    """Measure sustained motion that repeatedly returns to the same local region."""
    samples = list(samples)
    if len(samples) < 2:
        return {"detected": False, "path": 0.0, "net": 0.0, "efficiency": 1.0, "evidence": 0.0}
    path = sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(samples, samples[1:])
    )
    net = math.hypot(samples[-1][0] - samples[0][0], samples[-1][1] - samples[0][1])
    efficiency = net / max(path, 1e-6)
    evidence = sum(bool(sample[2]) or bool(sample[3]) for sample in samples) / len(samples)
    return {
        "detected": path >= min_path and efficiency <= max_efficiency and evidence >= min_evidence,
        "path": path,
        "net": net,
        "efficiency": efficiency,
        "evidence": evidence,
    }


def orbit_recovery_should_finish(
    elapsed_ticks,
    displacement,
    path_clear,
    min_ticks,
    max_ticks,
    exit_displacement=4.0,
):
    """End a latched recovery after grounded escape progress or its time bound."""
    if elapsed_ticks >= max_ticks:
        return True
    return (
        elapsed_ticks >= min_ticks
        and displacement >= exit_displacement
        and path_clear
    )


def metrics_from_state(crosstalk, complexity, memory, prediction_error, trap_pressure=0.0, collision_pressure=0.0, progress=0.0):
    latency = clamp(0.15 + 0.75 * complexity + 0.30 * crosstalk)
    fatigue_report = clamp(0.36 * crosstalk + 0.30 * complexity + 0.22 * prediction_error + 0.12 * latency)
    instability = (
        0.55 * crosstalk
        + 0.28 * complexity
        + 0.34 * prediction_error
        + 0.48 * trap_pressure
        + 0.38 * collision_pressure
        - 0.22 * progress
    )
    delusion_index = sigmoid(8.0 * (instability - 0.55))
    state_separability = clamp(memory * (1.0 - 0.62 * crosstalk) * (1.0 - 0.24 * complexity))
    return {
        "fatigue_report": fatigue_report,
        "delusion_index": delusion_index,
        "state_separability": state_separability,
    }


def dream_repair(crosstalk, complexity, memory, sleep_steps):
    for step in range(sleep_steps):
        crosstalk *= 0.948
        complexity *= 0.970
        if step > 85:
            memory *= 0.995
        if step > 140:
            memory *= 0.990
    return clamp(crosstalk), clamp(complexity), clamp(memory)


def successor_handoff(crosstalk, complexity, memory, prediction_error, repair_strength=1.0):
    strength = clamp(repair_strength, 0.0, 1.0)
    crosstalk = crosstalk * (1.0 - 0.68 * strength)
    complexity = complexity * (1.0 - 0.52 * strength)
    prediction_error = prediction_error * (1.0 - 0.36 * strength)
    memory = memory * (1.0 - 0.003 * strength)
    return (
        clamp(crosstalk),
        clamp(complexity),
        clamp(memory),
        clamp(prediction_error),
    )


class UnityBodyLink:
    def __init__(self, unity_host="127.0.0.1", unity_port=5055, listen_port=5056):
        self.unity_addr = (unity_host, unity_port)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", listen_port))
        self.receiver.setblocking(False)

    def send(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.sender.sendto(data, self.unity_addr)

    def receive_latest(self):
        latest = None
        while True:
            try:
                data, _addr = self.receiver.recvfrom(8192)
            except BlockingIOError:
                return latest
            try:
                latest = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue


class EmbodiedDynamicsObserver:
    """Passive temporal-coordination and criticality proxies for live telemetry."""

    def __init__(self, hz, modules=6, bus_capacity=3, window_seconds=2.0):
        self.hz = max(float(hz), 0.1)
        self.modules = int(modules)
        self.bus_capacity = max(1, int(bus_capacity))
        self.window_ticks = max(4, int(round(self.hz * window_seconds)))
        self.tick = 0
        self.previous = None
        self.last_event_ticks = [0] * self.modules
        self.previous_event_count = 0
        self.propagation_ratio = 1.0
        self.coherence = 0.0
        self.active_modules = 0
        self.bus_pressure = 0.0
        self.criticality_score = 1.0
        self.criticality_regime = "near_critical_proxy"
        self.recommended_gain = 1.16
        self.binding_ready = False

    def update(self, values, observed_noise):
        values = [clamp(float(value)) for value in values]
        if len(values) != self.modules:
            raise ValueError("observer_module_count_mismatch")
        if self.previous is None:
            changed = [True] * self.modules
        else:
            changed = [abs(value - old) >= 0.06 for value, old in zip(values, self.previous)]
        for index, event in enumerate(changed):
            if event:
                self.last_event_ticks[index] = self.tick

        vectors_x = 0.0
        vectors_y = 0.0
        total_weight = 0.0
        active = 0
        for index, event_tick in enumerate(self.last_event_ticks):
            age = self.tick - event_tick
            recency = math.exp(-age / self.window_ticks)
            weight = recency * (0.25 + 0.75 * values[index])
            if recency >= 0.25:
                active += 1
            phase = 2.0 * math.pi * ((event_tick % self.window_ticks) / self.window_ticks)
            vectors_x += weight * math.cos(phase)
            vectors_y += weight * math.sin(phase)
            total_weight += weight
        self.coherence = clamp(math.hypot(vectors_x, vectors_y) / max(total_weight, 1e-8))
        self.active_modules = active
        self.bus_pressure = clamp(max(0, active - self.bus_capacity) / self.bus_capacity)

        event_count = sum(changed)
        instantaneous_ratio = (event_count + 0.5) / (self.previous_event_count + 0.5)
        instantaneous_ratio = clamp(instantaneous_ratio, 0.20, 2.0)
        self.propagation_ratio = 0.86 * self.propagation_ratio + 0.14 * instantaneous_ratio
        self.criticality_score = clamp(math.exp(-abs(math.log(max(self.propagation_ratio, 1e-6)))))
        if self.propagation_ratio < 0.80:
            self.criticality_regime = "subcritical_proxy"
        elif self.propagation_ratio > 1.20:
            self.criticality_regime = "supercritical_proxy"
        else:
            self.criticality_regime = "near_critical_proxy"
        self.recommended_gain = 1.16 + 0.17 * clamp(float(observed_noise))
        self.binding_ready = self.coherence >= 0.70 and active >= 2 and self.bus_pressure <= 0.67
        self.previous = values
        self.previous_event_count = event_count
        self.tick += 1


class EmbodiedAdaptiveResonanceObserver:
    """Passive Fuzzy-ART category learning over embodied telemetry."""

    def __init__(self, vigilance=0.82, choice=0.01, learning_rate=0.18, max_categories=24):
        self.vigilance = clamp(vigilance)
        self.choice = max(float(choice), 1e-6)
        self.learning_rate = clamp(learning_rate)
        self.max_categories = max(1, int(max_categories))
        self.templates = []
        self.label_counts = []
        self.category = "unassigned"
        self.category_label = "unlearned"
        self.evidence_label = "unobserved"
        self.match = 0.0
        self.resonance = False
        self.novel = False
        self.unknown = False
        self.mismatch_resets = 0
        self.mismatch_resets_total = 0
        self.novel_events = 0
        self.unknown_events = 0
        self.category_switches = 0
        self.previous_category = None

    @staticmethod
    def complement_code(features):
        values = [clamp(float(value)) for value in features]
        return values + [1.0 - value for value in values]

    def update(self, features, evidence_label):
        encoded = self.complement_code(features)
        input_mass = max(sum(encoded), 1e-8)
        candidates = []
        for index, template in enumerate(self.templates):
            intersection = sum(min(value, weight) for value, weight in zip(encoded, template))
            choice_score = intersection / (self.choice + sum(template))
            match = intersection / input_mass
            candidates.append((choice_score, match, index))
        candidates.sort(reverse=True)

        selected = None
        selected_match = 0.0
        resets = 0
        for _choice_score, match, index in candidates:
            if match >= self.vigilance:
                selected = index
                selected_match = match
                break
            resets += 1

        self.evidence_label = str(evidence_label or "unobserved")
        self.mismatch_resets = resets
        self.mismatch_resets_total += resets
        self.novel = False
        self.unknown = False
        self.resonance = selected is not None

        if selected is None and len(self.templates) < self.max_categories:
            selected = len(self.templates)
            self.templates.append(list(encoded))
            self.label_counts.append({})
            selected_match = 1.0
            self.novel = True
            self.novel_events += 1
        elif selected is None:
            self.category = "unknown"
            self.category_label = self.evidence_label
            self.match = max((match for _score, match, _index in candidates), default=0.0)
            self.unknown = True
            self.unknown_events += 1
            if self.previous_category != self.category:
                self.category_switches += 1
            self.previous_category = self.category
            return

        if self.resonance:
            beta = self.learning_rate
            old = self.templates[selected]
            self.templates[selected] = [
                beta * min(value, weight) + (1.0 - beta) * weight
                for value, weight in zip(encoded, old)
            ]

        counts = self.label_counts[selected]
        counts[self.evidence_label] = counts.get(self.evidence_label, 0) + 1
        self.category = f"art_{selected + 1:02d}"
        self.category_label = max(counts, key=lambda label: (counts[label], label))
        self.match = selected_match
        if self.previous_category is not None and self.previous_category != self.category:
            self.category_switches += 1
        self.previous_category = self.category


class ShadowRecorder:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.rows = 0

    def write(self, ego, body_state, active_action):
        if not ego.shadow_ready or not isinstance(body_state, dict):
            return
        row = {
            "time": time.time(),
            "step": ego.steps,
            "controller_seed": getattr(ego, "controller_seed", 0),
            "position": [body_state.get("x"), body_state.get("y"), body_state.get("z")],
            "yaw": body_state.get("yaw"),
            "rays": body_state.get("directional_rays"),
            "body_clearance": body_state.get("directional_body_clearance"),
            "food_visible": bool(body_state.get("food_visible", False)),
            "food_feature": str(body_state.get("food_feature", "none")),
            "food_distance": body_state.get("food_distance"),
            "food_world": [body_state.get("food_world_x"), body_state.get("food_world_z")],
            "red_food_visible": bool(body_state.get("red_food_visible", False)),
            "red_food_distance": body_state.get("red_food_distance"),
            "red_food_world": [body_state.get("red_food_world_x"), body_state.get("red_food_world_z")],
            "blue_food_visible": bool(body_state.get("blue_food_visible", False)),
            "blue_food_distance": body_state.get("blue_food_distance"),
            "blue_food_world": [body_state.get("blue_food_world_x"), body_state.get("blue_food_world_z")],
            "yellow_food_visible": bool(body_state.get("yellow_food_visible", False)),
            "yellow_food_distance": body_state.get("yellow_food_distance"),
            "yellow_food_world": [body_state.get("yellow_food_world_x"), body_state.get("yellow_food_world_z")],
            "food_available_in_radius": int(body_state.get("food_available_in_radius", 0) or 0),
            "food_visible_in_radius": int(body_state.get("food_visible_in_radius", 0) or 0),
            "food_occluded_in_radius": int(body_state.get("food_occluded_in_radius", 0) or 0),
            "nearest_available_food_distance": body_state.get("nearest_available_food_distance"),
            "hunger": ego.hunger,
            "active_action": active_action,
            "active_intent": ego.intent_label(active_action),
            "shadow_action": ego.shadow_action,
            "shadow_probabilities": ego.shadow_probabilities,
            "shadow_confidence": ego.shadow_confidence,
            "shadow_entropy": ego.shadow_entropy,
            "shadow_agreement": ego.shadow_agreement,
            "shadow_takeover": ego.shadow_takeover,
            "shadow_takeover_steps": ego.shadow_takeover_steps,
            "shadow_world_move": list(ego.shadow_world_move),
            "shadow_continuous_intercept": ego.shadow_continuous_intercept,
            "shadow_episode_resets": ego.shadow_episode_resets,
            "shadow_mpc": ego.shadow_mpc,
            "shadow_mpc_engaged": ego.shadow_mpc_engaged,
            "shadow_mpc_score": ego.shadow_mpc_score,
            "shadow_mpc_mode": ego.shadow_mpc_mode,
            "shadow_mpc_horizon": ego.shadow_mpc_horizon,
            "shadow_mpc_depth": ego.shadow_mpc_depth,
            "shadow_mpc_uncertainty_stops": ego.shadow_mpc_uncertainty_stops,
            "orbit_adapter_enabled": ego.orbit_adapter_enabled,
            "orbit_adapter_active": ego.orbit_adapter_active,
            "orbit_adapter_action": ego.orbit_adapter_action,
            "orbit_adapter_confidence": ego.orbit_adapter_confidence,
            "orbit_adapter_confidence_gate": ego.orbit_adapter_confidence_gate,
            "orbit_adapter_events": ego.orbit_adapter_events,
            "orbit_adapter_remaining_seconds": ego.orbit_adapter_hold_ticks / ego.hz,
            "orbit_adapter_displacement": ego.orbit_adapter_displacement,
            "orbit_path": ego.orbit_path,
            "orbit_net": ego.orbit_net,
            "orbit_efficiency": ego.orbit_efficiency,
            "hidden_goal_adapter_enabled": ego.hidden_goal_adapter_enabled,
            "hidden_goal_adapter_active": ego.hidden_goal_adapter_active,
            "hidden_goal_adapter_action": ego.hidden_goal_adapter_action,
            "hidden_goal_adapter_confidence": ego.hidden_goal_adapter_confidence,
            "hidden_goal_adapter_events": ego.hidden_goal_adapter_events,
            "hidden_goal_route_index": ego.hidden_goal_route_index,
            "hidden_goal_route_count": len(ego.hidden_goal_route),
            "hidden_goal_route_distance": ego.hidden_goal_route_distance,
            "hidden_goal_route_vetoes": ego.hidden_goal_route_vetoes,
            "hidden_goal_route_terminal_holds": ego.hidden_goal_route_terminal_holds,
            "hidden_goal_route_terminal_active": ego.hidden_goal_route_terminal_active,
            "hidden_goal_route_selected_id": ego.hidden_goal_route_selected_id,
            "hidden_goal_route_art_match": ego.hidden_goal_route_art_match,
            "hidden_goal_food_latched": ego.hidden_goal_food_latch_ticks > 0,
            "hidden_goal_food_latch_seconds": ego.hidden_goal_food_latch_ticks / ego.hz,
            "hidden_goal_adapter_lateral_sign": ego.hidden_goal_adapter_lateral_sign,
            "hidden_goal_adapter_lateral_resets": ego.hidden_goal_adapter_lateral_resets,
            "food_sensor_radius": ego.shadow_food_sensor_radius,
            "blocked": bool(body_state.get("blocked", False)),
            "body_collision": bool(body_state.get("horizontal_collision", False)),
            "stuck": ego.current_stuck,
            "stuck_events": ego.stuck_events,
            "physics_wedge_seconds": ego.physics_wedge_ticks / ego.hz,
            "trap_accumulation_seconds": ego.trap_accumulation_ticks / ego.hz,
            "fallback_active": ego.shadow_fallback_hold_ticks > 0,
            "fallback_seconds_remaining": ego.shadow_fallback_hold_ticks / ego.hz,
            "unstuck_respawns": ego.unstuck_respawns,
            "trap_pressure": ego.local_trap_pressure(body_state),
            "survival_state": ego.survival_state,
            "survival_failures": ego.survival_failure_events,
            "survival_failure_reason": ego.survival_failure_reason,
            "critical_hunger_seconds": ego.critical_hunger_ticks / ego.hz,
            "forage_lapse_seconds": ego.forage_lapse_ticks / ego.hz,
            "seconds_since_food": ego.ticks_since_food / ego.hz,
            "generation": ego.generation,
            "handoff_events": ego.handoff_events,
            "workspace_problem": ego.workspace_packet["problem"],
            "workspace_strategy": ego.workspace_packet["strategy"],
            "trap_course": body_state.get("trap_course", "natural_terrain"),
            "trap_course_variant": body_state.get("trap_course_variant", "standard"),
            "trap_episode": body_state.get("trap_episode", 0),
            "trap_successes": body_state.get("trap_successes", 0),
            "trap_failures": body_state.get("trap_failures", 0),
            "trap_outcome": body_state.get("trap_outcome", "inactive"),
            "mushroom_pickups_total": body_state.get("mushroom_pickups_total", 0),
            "mushroom_reward_total": body_state.get("mushroom_reward_total", 0.0),
            "mushroom_feature": str(body_state.get("mushroom_feature", "none")),
            "red_mushroom_pickups_total": int(
                body_state.get("red_mushroom_pickups_total", 0) or 0
            ),
            "blue_mushroom_pickups_total": int(
                body_state.get("blue_mushroom_pickups_total", 0) or 0
            ),
            "yellow_flower_pickups_total": int(
                body_state.get("yellow_flower_pickups_total", 0) or 0
            ),
            "mushrooms_eaten": ego.mushrooms_eaten,
            "causal_probe_signal": ego.causal_probe_signal,
            "causal_probe_events": ego.causal_probe_events,
            "causal_probe_pending_events": len(ego.causal_probe_due_steps),
            "causal_probe_cancelled_events": (
                ego.causal_probe_world.cancelled_events
            ),
            "causal_probe_ambiguous_pickup_frames": (
                ego.causal_probe_world.ambiguous_pickup_frames
            ),
            "causal_probe_action_influence": 0,
            "causal_probe_hunger_cost": ego.causal_probe_hunger_cost,
            "causal_probe_hunger_cost_events": (
                ego.causal_probe_hunger_cost_events
            ),
            "causal_probe_hunger_cost_total": (
                ego.causal_probe_hunger_cost_total
            ),
            "typed_interaction_learning": ego.typed_interaction_learner.audit(),
            # Read-only aliases keep historical Tiny Scientist analyzers able
            # to consume new recordings without restoring metabolic effects.
            "metabolic_pressure": ego.metabolic_pressure,
            "metabolic_challenge_events": ego.metabolic_challenge_events,
            "metabolic_pending_challenges": len(ego.metabolic_challenge_due_steps),
            "tiny_scientist_memory": ego.tiny_scientist_memory.audit(),
            "tiny_scientist_rule_control": ego.tiny_scientist_rule_control,
            "tiny_scientist_rule_active": ego.tiny_scientist_rule_active,
            "tiny_scientist_rule_target_feature": ego.tiny_scientist_rule_target_feature,
            "tiny_scientist_rule_predicted_pressure_risk": ego.tiny_scientist_rule_predicted_pressure_risk,
            "tiny_scientist_rule_red_utility": ego.tiny_scientist_rule_red_utility,
            "tiny_scientist_rule_blue_utility": ego.tiny_scientist_rule_blue_utility,
            "tiny_scientist_rule_guidance_weight": ego.tiny_scientist_rule_guidance_weight,
            "tiny_scientist_rule_action_influence": ego.tiny_scientist_rule_action_influence,
            "pgnw_experiment": ego.pgnw_experiment_planner.audit(),
            "conductor_observer_mode": (
                (
                    "live_bounded_familiar_hidden_goal"
                    if ego.conductor_control == "familiar_hidden_goal"
                    else "passive_reward_learning"
                    if ego.conductor_observer.learning_enabled
                    else "passive_frozen_checkpoint"
                )
                if ego.conductor_observer.enabled
                else "disabled"
            ),
            "conductor_context": ego.conductor_observer.context,
            "conductor_recommendation": ego.conductor_observer.recommendation,
            "conductor_active_specialist": ego.conductor_observer.active_specialist,
            "conductor_confidence": ego.conductor_observer.confidence,
            "conductor_agreement": ego.conductor_observer.agreement,
            "conductor_agreement_rate": ego.conductor_observer.agreement_rate,
            "conductor_grounded_reward": ego.conductor_observer.last_reward,
            "conductor_updates": ego.conductor_observer.updates,
            "conductor_context_visits": ego.conductor_observer.context_visits,
            "conductor_protocol_entries": len(ego.conductor_observer.protocol),
            "conductor_episode_resets": ego.conductor_observer.episode_resets,
            "conductor_action_influence": ego.conductor_observer.action_influence,
            "conductor_control_mode": ego.conductor_control,
            "conductor_gate_active": ego.conductor_gate_active,
            "conductor_gate_recommendation": ego.conductor_gate_recommendation,
            "conductor_gate_confidence": ego.conductor_gate_confidence,
            "conductor_gate_frames": ego.conductor_gate_frames,
            "conductor_gate_decisions": ego.conductor_gate_decisions,
            "conductor_gate_denials": ego.conductor_gate_denials,
            "conductor_q_values": ego.conductor_observer.q_values,
            "systemic_conductor_mode": (
                (
                    "bounded_recurrent_mpc"
                    if ego.systemic_router.mode != "passive"
                    else "passive_frozen_checkpoint"
                )
                if ego.systemic_conductor.enabled
                else "disabled"
            ),
            "systemic_conductor_features": ego.systemic_conductor.features,
            "systemic_conductor_scores": ego.systemic_conductor.scores,
            "systemic_conductor_probabilities": (
                ego.systemic_conductor.probabilities
            ),
            "systemic_conductor_raw_probabilities": (
                ego.systemic_conductor.raw_probabilities
            ),
            "systemic_conductor_raw_recommendation": (
                ego.systemic_conductor.raw_recommendation
            ),
            "systemic_conductor_recommendation": (
                ego.systemic_conductor.recommendation
            ),
            "systemic_conductor_confidence": ego.systemic_conductor.confidence,
            "systemic_conductor_entropy_bits": ego.systemic_conductor.entropy,
            "systemic_conductor_proxy_context": (
                ego.systemic_conductor.proxy_context
            ),
            "systemic_conductor_proxy_optimal": (
                ego.systemic_conductor.proxy_optimal
            ),
            "systemic_conductor_agreement": ego.systemic_conductor.agreement,
            "systemic_conductor_agreement_rate": (
                ego.systemic_conductor.agreement_rate
            ),
            "systemic_conductor_observations": (
                ego.systemic_conductor.observations
            ),
            "systemic_conductor_action_influence": (
                ego.systemic_conductor.action_influence
            ),
            "adaptive_gnw_mode": (
                ego.adaptive_gnw_control
            ),
            "adaptive_gnw_active_specialist": (
                ego.adaptive_gnw_gate.active_specialist
            ),
            "adaptive_gnw_challenger": (
                ego.adaptive_gnw_gate.challenger_specialist
            ),
            "adaptive_gnw_challenger_streak": (
                ego.adaptive_gnw_gate.challenger_streak
            ),
            "adaptive_gnw_broadcast_age": (
                ego.adaptive_gnw_gate.broadcast_age
            ),
            "adaptive_gnw_event": ego.adaptive_gnw_gate.last_event,
            "adaptive_gnw_ignition": ego.adaptive_gnw_gate.last_ignition,
            "adaptive_gnw_confidence": ego.adaptive_gnw_gate.confidence,
            "adaptive_gnw_margin": ego.adaptive_gnw_gate.margin,
            "adaptive_gnw_agreement": ego.adaptive_gnw_gate.agreement,
            "adaptive_gnw_agreement_rate": (
                ego.adaptive_gnw_gate.agreement_rate
            ),
            "adaptive_gnw_ignitions": ego.adaptive_gnw_gate.ignitions,
            "adaptive_gnw_releases": ego.adaptive_gnw_gate.releases,
            "adaptive_gnw_held_challenges": (
                ego.adaptive_gnw_gate.held_challenges
            ),
            "adaptive_gnw_action_influence": (
                ego.adaptive_gnw_gate.action_influence
            ),
            "adaptive_gnw_recommendation_substitutions": (
                ego.adaptive_gnw_gate.recommendation_substitutions
            ),
            "systemic_router_active_specialist": (
                ego.systemic_router.active_specialist
            ),
            "systemic_router_baseline_specialist": (
                ego.systemic_router.last_baseline_specialist
            ),
            "systemic_router_reason": ego.systemic_router.last_reason,
            "systemic_router_handoffs": ego.systemic_router.handoffs,
            "systemic_router_decisions": ego.systemic_router.decisions,
            "systemic_router_denials": ego.systemic_router.denials,
            "systemic_router_safety_overrides": (
                ego.systemic_router.safety_overrides
            ),
            "systemic_router_influence_frames": (
                ego.systemic_router.influence_frames
            ),
            "systemic_router_chatter_events": (
                ego.systemic_router.chatter_events
            ),
            "systemic_router_hold_seconds": (
                ego.systemic_router.hold_ticks / ego.hz
            ),
            "systemic_router_mpc_latch_seconds": (
                ego.systemic_router.mandatory_latch_ticks / ego.hz
            ),
            "systemic_router_recurrent_release_streak": (
                ego.systemic_router.recurrent_release_streak
            ),
            "sync_observer_mode": "passive",
            "sync_coherence": ego.dynamics_observer.coherence,
            "sync_active_modules": ego.dynamics_observer.active_modules,
            "sync_bus_pressure": ego.dynamics_observer.bus_pressure,
            "sync_binding_ready": ego.dynamics_observer.binding_ready,
            "criticality_observer_mode": "passive_proxy",
            "criticality_propagation_ratio": ego.dynamics_observer.propagation_ratio,
            "criticality_score": ego.dynamics_observer.criticality_score,
            "criticality_regime": ego.dynamics_observer.criticality_regime,
            "criticality_recommended_gain": ego.dynamics_observer.recommended_gain,
            "art_observer_mode": "passive_fuzzy_art",
            "art_category": ego.art_observer.category,
            "art_category_label": ego.art_observer.category_label,
            "art_evidence_label": ego.art_observer.evidence_label,
            "art_match": ego.art_observer.match,
            "art_resonance": ego.art_observer.resonance,
            "art_novel": ego.art_observer.novel,
            "art_unknown": ego.art_observer.unknown,
            "art_mismatch_resets": ego.art_observer.mismatch_resets,
            "art_mismatch_resets_total": ego.art_observer.mismatch_resets_total,
            "art_category_count": len(ego.art_observer.templates),
            "art_category_switches": ego.art_observer.category_switches,
            "terrain_air_mode": (
                "passive_retrieval"
                if ego.terrain_air_observer.enabled
                else "disabled"
            ),
            "terrain_air_recalled_action": ego.terrain_air_observer.recalled_action,
            "terrain_air_distance": ego.terrain_air_observer.distance,
            "terrain_air_confidence": ego.terrain_air_observer.confidence,
            "terrain_air_resonance": ego.terrain_air_observer.resonance,
            "terrain_air_agreement": ego.terrain_air_observer.agreement,
            "terrain_air_agreement_rate": ego.terrain_air_observer.agreement_rate,
            "terrain_air_queries": ego.terrain_air_observer.queries,
            "terrain_air_resonances": ego.terrain_air_observer.resonances,
            "terrain_air_action_influence": ego.terrain_air_observer.action_influence,
            "terrain_air_route_mode": ego.terrain_air_route_controller.control_mode,
            "terrain_air_route_active": ego.terrain_air_route_controller.active,
            "terrain_air_route_pending": ego.terrain_air_route_controller.pending,
            "terrain_air_route_recommendation": ego.terrain_air_route_controller.recommendation,
            "terrain_air_route_reason": ego.terrain_air_route_controller.reason,
            "terrain_air_route_match": ego.terrain_air_route_controller.match,
            "terrain_air_route_vigilance": (
                ego.terrain_air_route_controller.effective_vigilance
            ),
            "terrain_air_route_index": ego.terrain_air_route_controller.route_index,
            "terrain_air_route_count": ego.terrain_air_route_controller.route_count,
            "terrain_air_route_distance": ego.terrain_air_route_controller.route_distance,
            "terrain_air_route_remaining_seconds": (
                ego.terrain_air_route_controller.remaining_ticks / ego.hz
            ),
            "terrain_air_route_recommendations": ego.terrain_air_route_controller.recommendations,
            "terrain_air_route_interventions": ego.terrain_air_route_controller.interventions,
            "terrain_air_route_authorized_activations": (
                ego.terrain_air_route_controller.authorized_activations
            ),
            "terrain_air_route_authorization_denials": (
                ego.terrain_air_route_controller.authorization_denials
            ),
            "terrain_air_route_releases": ego.terrain_air_route_controller.releases,
            "terrain_air_route_release_reason": ego.terrain_air_route_controller.release_reason,
            "terrain_air_route_sensor_vetoes": ego.terrain_air_route_controller.sensor_vetoes,
            "terrain_air_route_action_influence": ego.terrain_air_route_controller.action_influence,
            "terrain_air_route_guidance_vector": list(
                ego.terrain_air_route_controller.guidance_vector
            ),
            "terrain_air_route_guidance_weight": ego.terrain_air_route_controller.guidance_weight,
            "terrain_air_route_guidance_decisions": (
                ego.terrain_air_route_controller.guidance_decisions
            ),
            "terrain_air_route_guidance_action_changes": (
                ego.terrain_air_route_controller.guidance_action_changes
            ),
            "escape_teacher_enabled": (
                ego.terrain_air_route_controller.teacher_memory_path is not None
            ),
            "escape_teacher_active": ego.escape_teacher.active,
            "escape_teacher_action": ego.escape_teacher.action_name,
            "escape_teacher_elapsed_seconds": (
                ego.escape_teacher.elapsed_ticks / ego.hz
            ),
            "escape_teacher_displacement": ego.escape_teacher.displacement,
            "escape_teacher_efficiency": ego.escape_teacher.efficiency,
            "escape_teacher_events": ego.escape_teacher.events,
            "escape_teacher_successes": ego.escape_teacher.successes,
            "escape_teacher_failures": ego.escape_teacher.failures,
            "escape_teacher_safety_reselections": (
                ego.escape_teacher.safety_reselections
            ),
            "escape_teacher_last_outcome": ego.escape_teacher.last_outcome,
            "escape_teacher_last_route_id": ego.escape_teacher.last_route_id,
            "escape_teacher_memory_routes": len(
                ego.terrain_air_route_controller.teacher_route_payloads
            ),
            "terrain_air_route_unguided_action": (
                ego.terrain_air_route_controller.last_unguided_action
            ),
            "terrain_air_route_guided_action": (
                ego.terrain_air_route_controller.last_guided_action
            ),
            "terrain_air_route_unguided_margin": (
                ego.terrain_air_route_controller.last_unguided_margin
            ),
            "terrain_air_route_margin_ambiguity": (
                ego.terrain_air_route_controller.last_margin_ambiguity
            ),
            "terrain_air_route_effective_guidance_weight": (
                ego.terrain_air_route_controller.last_effective_guidance_weight
            ),
            "resource_memory_enabled": ego.resource_memory.enabled,
            "resource_memory_hunger_gate": ego.resource_memory.hunger_gate,
            "resource_memory_active": ego.resource_memory.active,
            "resource_memory_recommendation": ego.resource_memory.recommendation,
            "resource_memory_target": (
                list(ego.resource_memory.target)
                if ego.resource_memory.target is not None
                else None
            ),
            "resource_memory_guidance_vector": list(
                ego.resource_memory.guidance_vector
            ),
            "resource_memory_distance": ego.resource_memory.distance,
            "resource_memory_confidence": ego.resource_memory.confidence,
            "resource_memory_regions": len(ego.resource_memory.entries),
            "resource_memory_typed_regions": len(
                ego.resource_memory.typed_entries
            ),
            "resource_memory_active_feature": (
                ego.resource_memory.active_feature
            ),
            "resource_memory_encodings": ego.resource_memory.encodings,
            "resource_memory_typed_encodings": (
                ego.resource_memory.typed_encodings
            ),
            "resource_memory_queries": ego.resource_memory.queries,
            "resource_memory_typed_queries": ego.resource_memory.typed_queries,
            "resource_memory_recommendations": ego.resource_memory.recommendations,
            "resource_memory_typed_recommendations": (
                ego.resource_memory.typed_recommendations
            ),
            "resource_memory_recommendation_frames": (
                ego.resource_memory.recommendation_frames
            ),
            "resource_memory_pickups_after_recommendation": (
                ego.resource_memory.pickups_after_recommendation
            ),
            "resource_memory_stale_arrivals": (
                ego.resource_memory.counterfactual_stale_arrivals
            ),
            "resource_memory_release_reason": ego.resource_memory.release_reason,
            "resource_memory_action_influence": ego.resource_memory.action_influence,
            "resource_memory_control_mode": ego.resource_memory.control_mode,
            "resource_memory_guidance_decisions": (
                ego.resource_memory.guidance_decisions
            ),
            "resource_memory_guidance_action_changes": (
                ego.resource_memory.guidance_action_changes
            ),
            "resource_memory_unguided_action": (
                ego.resource_memory.last_unguided_action
            ),
            "resource_memory_guided_action": (
                ego.resource_memory.last_guided_action
            ),
            "resource_memory_unguided_margin": (
                ego.resource_memory.last_unguided_margin
            ),
            "resource_memory_margin_ambiguity": (
                ego.resource_memory.last_margin_ambiguity
            ),
            "resource_memory_effective_guidance_weight": (
                ego.resource_memory.last_effective_guidance_weight
            ),
        }
        self.handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.rows += 1
        if self.rows % 25 == 0:
            self.handle.flush()

    def close(self):
        self.handle.flush()
        self.handle.close()


class EmbodiedFunctionalEgo:
    def __init__(
        self,
        hz=5.0,
        sleep_seconds=60.0,
        min_awake_seconds=300.0,
        sleep_threshold=0.82,
        wake_seconds=3.0,
        maintenance_mode="handoff",
        handoff_threshold=0.62,
        emergency_sleep_threshold=0.92,
        handoff_cooldown_seconds=180.0,
        memory_cell_size=2.0,
        obstacle_memory_decay=0.996,
        route_exploration=0.35,
        dopamine=0.35,
        norepinephrine=0.35,
        acetylcholine=0.35,
        noise_injection=0.0,
        calcium_gate=0.45,
        unstuck_respawn_seconds=45.0,
        shadow_checkpoint=None,
        shadow_control="passive",
        shadow_control_confidence=0.55,
        shadow_mpc=False,
        orbit_exit_adapter=None,
        orbit_adapter_confidence=0.70,
        hidden_goal_adapter=None,
        hidden_goal_adapter_confidence=0.30,
        hidden_goal_adapter_commit_seconds=0.4,
        hidden_goal_adapter_lateral_bias=0.0,
        passive_conductor=False,
        conductor_checkpoint=None,
        conductor_control="passive",
        systemic_conductor_checkpoint=None,
        systemic_conductor_control="passive",
        systemic_conductor_confidence=0.55,
        adaptive_gnw_passive=False,
        adaptive_gnw_control="disabled",
        tiny_scientist_rule_control="passive",
        tiny_scientist_experiment_control="disabled",
        tiny_scientist_experiment_seed=0,
        tiny_scientist_experiment_guidance_weight=0.03,
        tiny_scientist_experiment_guidance_margin=0.08,
        tiny_scientist_experiment_commit_seconds=3.0,
        tiny_scientist_experiment_commit_cooldown_seconds=2.0,
        tiny_scientist_experiment_max_score_regret=0.18,
        tiny_scientist_experiment_isolation_retreat_seconds=2.0,
        tiny_scientist_experiment_isolation_retreat_max_score_regret=0.35,
        tiny_scientist_hypothesis_proposer=None,
        causal_probe_delay_seconds=10.0,
        causal_probe_cancellation_feature="none",
        causal_probe_hunger_cost=0.0,
        typed_interaction_learning=False,
        typed_interaction_memory=None,
        typed_interaction_discovery_memory=None,
        typed_interaction_hypothesis_proposer=None,
        typed_interaction_rule_control="passive",
    ):
        self.crosstalk = 0.07
        self.complexity = 0.12
        self.memory = 0.98
        self.prediction_error = 0.18
        self.fatigue_report = 0.0
        self.delusion_index = 0.0
        self.sleep_remaining = 0
        self.sleep_total_ticks = 0
        self.wake_remaining = 0
        self.sleep_repair_credit = 0.0
        self.hz = max(hz, 0.1)
        self.sleep_seconds = max(sleep_seconds, 1.0)
        self.wake_total_ticks = max(1, int(round(wake_seconds * self.hz)))
        self.min_awake_ticks = max(0, int(round(min_awake_seconds * self.hz)))
        self.sleep_threshold = clamp(sleep_threshold)
        self.maintenance_mode = maintenance_mode
        self.handoff_threshold = clamp(handoff_threshold)
        self.emergency_sleep_threshold = clamp(emergency_sleep_threshold)
        self.handoff_cooldown_total_ticks = max(1, int(round(handoff_cooldown_seconds * self.hz)))
        self.handoff_cooldown = 0
        self.handoff_events = 0
        self.visible_sleep_events = 0
        self.generation = 1
        self.last_maintenance = "wake"
        self.awake_ticks = 0
        self.last_action = "idle"
        self.turn_bias = 1
        self.steps = 0
        self.position_history = deque(maxlen=max(4, int(round(self.hz * 4.0))))
        self.escape_action = None
        self.escape_ticks = 0
        self.breakout_plan = deque()
        self.breakout_events = 0
        self.breakout_style = "none"
        self.cluster_escalation = 0
        self.last_breakout_cell = None
        self.stuck_cooldown = 0
        self.stuck_events = 0
        self.current_stuck = False
        self.physics_wedge_ticks = 0
        self.trap_accumulation_ticks = 0
        self.unstuck_respawn_ticks = 0 if unstuck_respawn_seconds <= 0 else max(1, int(round(unstuck_respawn_seconds * self.hz)))
        self.unstuck_respawns = 0
        self.contact_probe_ticks = 0
        self.heading_action = "up"
        self.heading_ticks = 0
        self.memory_cell_size = max(memory_cell_size, 0.25)
        self.obstacle_memory_decay = clamp(obstacle_memory_decay, 0.90, 1.0)
        self.route_exploration = clamp(route_exploration)
        self.base_route_exploration = self.route_exploration
        self.dopamine_baseline = clamp(dopamine)
        self.dopamine = self.dopamine_baseline
        self.dopamine_food_boost = 0.0
        self.hunger = 0.25
        self.mushrooms_eaten = 0
        self.last_mushroom_pickup_total = 0
        self.last_mushroom_reward_total = 0.0
        self.last_red_mushroom_pickup_total = 0
        self.last_blue_mushroom_pickup_total = 0
        self.last_yellow_flower_pickup_total = 0
        self.last_consumable_feature = "none"
        self.causal_probe_world = OrderedCausalProbeWorld(
            hz=self.hz,
            delay_seconds=causal_probe_delay_seconds,
            cancellation_feature=causal_probe_cancellation_feature,
        )
        self.causal_probe_delay_ticks = self.causal_probe_world.delay_ticks
        self.causal_probe_due_steps = self.causal_probe_world.pending_due_steps
        self.causal_probe_signal = 0.0
        self.causal_probe_events = 0
        self.causal_probe_hunger_cost = clamp(causal_probe_hunger_cost)
        self.causal_probe_hunger_cost_events = 0
        self.causal_probe_hunger_cost_total = 0.0
        self.typed_interaction_learner = PassiveOrderedEpisodeLearner(
            hz=self.hz,
            delay_seconds=causal_probe_delay_seconds,
            enabled=typed_interaction_learning,
            memory_path=typed_interaction_memory,
            discovery_memory_path=typed_interaction_discovery_memory,
            hypothesis_proposer=typed_interaction_hypothesis_proposer,
        )
        self.food_feedback_initialized = False
        self.ticks_since_food = 0
        self.food_seek_ticks = 0
        self.food_seek_move = (0.0, 0.0)
        self.avoidance_move = (0.0, 0.0)
        self.avoidance_move_ticks = 0
        self.last_body_obstacle_visible = False
        self.food_lock_ticks = 0
        self.foraging_commit_ticks = 0
        self.sensory_focus_gain = 0.0
        self.sensory_focus_events = 0
        self.norepinephrine = clamp(norepinephrine)
        self.acetylcholine = clamp(acetylcholine)
        self.noise_injection = clamp(noise_injection)
        self.calcium_gate = clamp(calcium_gate)
        self.workspace_unreliable = False
        self.previous_workspace_problem = "none"
        self.locked_workspace_steps = 0
        self.no_progress_steps = 0
        self.reality_gate_brakes = 0
        self.meta_monitor_brakes = 0
        self.hunger_anchor_steps = 0
        self.false_food_reports = 0
        self.false_trap_reports = 0
        self.critical_hunger_ticks = 0
        self.forage_lapse_ticks = 0
        self.survival_failure_events = 0
        self.survival_failed = False
        self.survival_state = "stable"
        self.survival_failure_reason = "none"
        self.last_clear_progress = False
        self.valence = 0.0
        self.arousal = 0.0
        self.trap_pressure = 0.0
        self.workspace_packet = self.empty_workspace()
        self.workspace_promotions = 0
        self.obstacle_memory = {}
        self.recent_failures = {}
        self.escape_attempts = {}
        self.trap_memory = {}
        self.obstacle_events = 0
        self.shadow_enabled = False
        self.shadow_ready = False
        self.shadow_policy = None
        self.shadow_safety_mask = None
        self.shadow_hidden = None
        self.shadow_torch = None
        self.shadow_action = "none"
        self.shadow_confidence = 0.0
        self.shadow_entropy = 0.0
        self.shadow_agreement = 0.0
        self.shadow_probabilities = [0.0] * 8
        self.shadow_world_move = (0.0, 1.0)
        self.shadow_continuous_intercept = False
        self.shadow_last_reward = 0.0
        self.shadow_previous_action = 0
        self.shadow_previous_position = None
        self.shadow_previous_proposal = "none"
        self.shadow_world_move = (0.0, 1.0)
        self.shadow_continuous_intercept = False
        self.shadow_ray_range = 6.0
        self.shadow_error = "disabled"
        self.shadow_control = shadow_control
        self.shadow_control_confidence = clamp(shadow_control_confidence)
        self.shadow_takeover = False
        self.shadow_takeover_steps = 0
        self.shadow_episode_key = None
        self.shadow_episode_resets = 0
        self.shadow_episode_idle_active = False
        self.shadow_episode_idle_ticks = 0
        self.shadow_body_safe_actions = 0
        self.shadow_mpc = bool(shadow_mpc)
        self.shadow_mpc_engaged = False
        self.shadow_mpc_hold_ticks = 0
        self.shadow_fallback_hold_ticks = 0
        self.shadow_mpc_score = 0.0
        self.shadow_mpc_mode = "recurrent"
        self.shadow_mpc_horizon = 0
        self.shadow_mpc_depth = 0.0
        self.shadow_mpc_uncertainty_stops = 0
        self.shadow_mpc_planning_frames = 0
        self.shadow_mpc_critical_frames = 0
        self.orbit_adapter = None
        self.orbit_adapter_enabled = False
        self.orbit_adapter_error = "disabled"
        self.orbit_adapter_active = False
        self.orbit_adapter_action = "none"
        self.orbit_adapter_confidence = 0.0
        self.orbit_adapter_confidence_gate = clamp(orbit_adapter_confidence)
        self.orbit_adapter_events = 0
        self.orbit_adapter_hold_ticks = 0
        self.orbit_adapter_action_ticks = 0
        self.orbit_adapter_elapsed_ticks = 0
        self.orbit_adapter_selected = None
        self.orbit_adapter_start_position = None
        self.orbit_adapter_displacement = 0.0
        self.orbit_history = deque(maxlen=max(10, int(round(self.hz * 10.0))))
        self.orbit_path = 0.0
        self.orbit_net = 0.0
        self.orbit_efficiency = 1.0
        self.escape_teacher = TerrainEscapeTeacher(hz=self.hz)
        self.hidden_goal_adapter = None
        self.hidden_goal_adapter_type = "none"
        self.hidden_goal_adapter_temporal_steps = 1
        self.hidden_goal_adapter_history = deque()
        self.hidden_goal_route = []
        self.hidden_goal_route_library = []
        self.hidden_goal_route_library_vigilance = 0.86
        self.hidden_goal_route_library_exploration = 0.0
        self.hidden_goal_route_selected_id = "none"
        self.hidden_goal_route_art_match = 0.0
        self.hidden_goal_route_index = 0
        self.hidden_goal_route_origin = None
        self.hidden_goal_route_radius = 0.85
        self.hidden_goal_route_terminal_radius = 0.30
        self.hidden_goal_route_terminal_count = 5
        self.hidden_goal_route_terminal_extension = 2.0
        self.hidden_goal_route_distance = 0.0
        self.hidden_goal_route_vetoes = 0
        self.hidden_goal_route_terminal_holds = 0
        self.hidden_goal_route_terminal_active = False
        self.hidden_goal_food_latch_total_ticks = max(1, int(round(self.hz * 2.0)))
        self.hidden_goal_food_latch_ticks = 0
        self.hidden_goal_food_target = None
        self.hidden_goal_adapter_enabled = False
        self.hidden_goal_adapter_error = "disabled"
        self.hidden_goal_adapter_active = False
        self.hidden_goal_adapter_action = "none"
        self.hidden_goal_adapter_confidence = 0.0
        self.hidden_goal_adapter_confidence_gate = clamp(hidden_goal_adapter_confidence)
        self.hidden_goal_adapter_commit_ticks = max(
            1, int(round(self.hz * hidden_goal_adapter_commit_seconds))
        )
        self.hidden_goal_adapter_lateral_bias = max(0.0, float(hidden_goal_adapter_lateral_bias))
        self.hidden_goal_adapter_events = 0
        self.hidden_goal_adapter_selected = None
        self.hidden_goal_adapter_hold_ticks = 0
        self.hidden_goal_adapter_lateral_sign = 0
        self.hidden_goal_adapter_lateral_start_position = None
        self.hidden_goal_adapter_lateral_elapsed_ticks = 0
        self.hidden_goal_adapter_lateral_resets = 0
        self.shadow_food_sensor_radius = 16.0
        self.trap_course_label = "natural_terrain"
        self.trap_course_variant = "standard"
        self.trap_course_episode = 0
        self.trap_course_successes = 0
        self.trap_course_failures = 0
        self.trap_course_outcome = "inactive"
        self.dynamics_observer = EmbodiedDynamicsObserver(self.hz)
        self.art_observer = EmbodiedAdaptiveResonanceObserver()
        self.terrain_air_observer = PassiveTerrainAirObserver()
        self.terrain_air_route_controller = TerrainAirRouteController()
        self.resource_memory = PassiveTerrainResourceMemory()
        self.tiny_scientist_memory = VerifiedProductionMemory()
        if tiny_scientist_rule_control != "passive":
            raise ValueError("tiny_scientist_rule_control_is_passive_only")
        self.tiny_scientist_rule_control = tiny_scientist_rule_control
        self.tiny_scientist_rule_active = False
        self.tiny_scientist_rule_target_feature = "none"
        self.tiny_scientist_rule_predicted_pressure_risk = 0.0
        self.tiny_scientist_rule_guidance_vector = (0.0, 0.0)
        self.tiny_scientist_rule_red_vector = (0.0, 0.0)
        self.tiny_scientist_rule_red_utility = 0.0
        self.tiny_scientist_rule_blue_utility = 0.0
        self.tiny_scientist_rule_guidance_weight = 0.0
        self.tiny_scientist_rule_decisions = 0
        self.tiny_scientist_rule_action_influence = 0
        if (
            typed_interaction_rule_control == "verified_protective"
            and not typed_interaction_learning
        ):
            raise ValueError("verified_protective_requires_typed_learning")
        if (
            typed_interaction_rule_control == "verified_protective"
            and tiny_scientist_experiment_control != "committed"
        ):
            raise ValueError("verified_protective_requires_committed_pgnw")
        if (
            tiny_scientist_experiment_control in {
                "bounded", "committed", "dynamic_committed"
            }
            and not self.shadow_mpc
        ):
            raise ValueError("bounded_pgnw_experiment_requires_shadow_mpc")
        self.pgnw_experiment_planner = EmbodiedPGNWExperimentPlanner(
            mode=tiny_scientist_experiment_control,
            hz=self.hz,
            seed=tiny_scientist_experiment_seed,
            max_guidance_weight=tiny_scientist_experiment_guidance_weight,
            guidance_margin_threshold=tiny_scientist_experiment_guidance_margin,
            commitment_seconds=tiny_scientist_experiment_commit_seconds,
            commitment_cooldown_seconds=(
                tiny_scientist_experiment_commit_cooldown_seconds
            ),
            max_score_regret=tiny_scientist_experiment_max_score_regret,
            isolation_retreat_seconds=(
                tiny_scientist_experiment_isolation_retreat_seconds
            ),
            isolation_retreat_max_score_regret=(
                tiny_scientist_experiment_isolation_retreat_max_score_regret
            ),
            hypothesis_proposer=tiny_scientist_hypothesis_proposer,
            typed_learner=(
                self.typed_interaction_learner
                if typed_interaction_learning
                else None
            ),
            typed_rule_control=typed_interaction_rule_control,
        )
        self.conductor_control = str(conductor_control)
        if self.conductor_control != "passive" and not conductor_checkpoint:
            raise ValueError("live_conductor_requires_checkpoint")
        self.conductor_gate_active = False
        self.conductor_gate_recommendation = "insufficient_evidence"
        self.conductor_gate_confidence = 0.0
        self.conductor_gate_frames = 0
        self.conductor_gate_decisions = 0
        self.conductor_gate_denials = 0
        self.conductor_observer = PassiveEmbodiedConductor(
            enabled=passive_conductor or self.conductor_control != "passive",
            checkpoint=conductor_checkpoint,
        )
        self.systemic_conductor = PassiveSystemicConductor(
            checkpoint=systemic_conductor_checkpoint,
        )
        self.adaptive_gnw_control = str(adaptive_gnw_control)
        if adaptive_gnw_passive and self.adaptive_gnw_control == "disabled":
            self.adaptive_gnw_control = "passive"
        if self.adaptive_gnw_control not in {"disabled", "passive", "bounded"}:
            raise ValueError("unsupported_adaptive_gnw_control")
        if (
            self.adaptive_gnw_control != "disabled"
            and not systemic_conductor_checkpoint
        ):
            raise ValueError("adaptive_gnw_passive_requires_systemic_checkpoint")
        if (
            self.adaptive_gnw_control == "bounded"
            and systemic_conductor_control == "passive"
        ):
            raise ValueError("bounded_adaptive_gnw_requires_active_router")
        self.adaptive_gnw_gate = PassiveAdaptiveIgnitionGate(
            enabled=self.adaptive_gnw_control != "disabled",
        )
        if (
            systemic_conductor_control != "passive"
            and not systemic_conductor_checkpoint
        ):
            raise ValueError("active_systemic_conductor_requires_checkpoint")
        if systemic_conductor_control != "passive" and not self.shadow_mpc:
            raise ValueError("active_systemic_conductor_requires_shadow_mpc")
        self.systemic_router = BoundedExecutiveRouter(
            mode=systemic_conductor_control,
            hz=self.hz,
            confidence_threshold=systemic_conductor_confidence,
        )
        if shadow_checkpoint:
            self.enable_shadow_policy(shadow_checkpoint)
        if orbit_exit_adapter:
            self.enable_orbit_adapter(orbit_exit_adapter)
        if hidden_goal_adapter:
            self.enable_hidden_goal_adapter(hidden_goal_adapter)

    @property
    def metabolic_pressure(self):
        """Deprecated read/write alias for passive causal-probe telemetry."""
        return self.causal_probe_signal

    @metabolic_pressure.setter
    def metabolic_pressure(self, value):
        self.causal_probe_signal = clamp(value)

    @property
    def metabolic_challenge_delay_ticks(self):
        return self.causal_probe_delay_ticks

    @property
    def metabolic_challenge_due_steps(self):
        return self.causal_probe_due_steps

    @property
    def metabolic_challenge_events(self):
        return self.causal_probe_events

    def systemic_executive_signal(self):
        if (
            self.adaptive_gnw_control == "bounded"
            and self.adaptive_gnw_gate.active_specialist
            in {"recurrent", "episodic", "predictive", "fallback"}
        ):
            return (
                self.adaptive_gnw_gate.active_specialist,
                self.adaptive_gnw_gate.confidence,
            )
        return (
            self.systemic_conductor.recommendation,
            self.systemic_conductor.confidence,
        )

    def enable_shadow_policy(self, checkpoint_path):
        try:
            import torch
            from upgraded_foraging_pipeline import load_checkpoint, safety_mask_logits

            path = Path(checkpoint_path).expanduser().resolve()
            self.shadow_policy, payload = load_checkpoint(path)
            self.shadow_safety_mask = safety_mask_logits
            self.shadow_torch = torch
            self.shadow_hidden = self.shadow_policy.initial_state(1)
            self.shadow_ray_range = float(payload["config"].get("ray_range", 6.0))
            self.shadow_enabled = True
            self.shadow_error = "waiting_for_8_rays"
        except Exception as exc:
            self.shadow_enabled = False
            self.shadow_error = f"load_failed:{type(exc).__name__}"

    def enable_orbit_adapter(self, checkpoint_path):
        try:
            if self.shadow_torch is None or self.shadow_policy is None:
                raise RuntimeError("shadow_policy_required")
            payload = self.shadow_torch.load(
                Path(checkpoint_path).expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
            hidden_dim = int(payload["hidden_dim"])
            if hidden_dim != self.shadow_policy.hidden_dim:
                raise ValueError("hidden_dim_mismatch")
            adapter = self.shadow_torch.nn.Sequential(
                self.shadow_torch.nn.LayerNorm(hidden_dim),
                self.shadow_torch.nn.Linear(hidden_dim, len(SHADOW_ACTIONS)),
            )
            adapter.load_state_dict(payload["state_dict"])
            self.orbit_adapter = adapter.eval()
            self.orbit_adapter_enabled = True
            self.orbit_adapter_error = "none"
        except Exception as exc:
            self.orbit_adapter = None
            self.orbit_adapter_enabled = False
            self.orbit_adapter_error = f"load_failed:{type(exc).__name__}"

    def enable_hidden_goal_adapter(self, checkpoint_path):
        try:
            if self.shadow_torch is None or self.shadow_policy is None:
                raise RuntimeError("shadow_policy_required")
            payload = self.shadow_torch.load(
                Path(checkpoint_path).expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
            adapter_type = str(payload.get("adapter_type", "linear"))
            hidden_dim = int(payload.get("hidden_dim", payload.get("input_dim", 0)))
            if adapter_type in {"episodic_route", "art_route_library"}:
                if adapter_type == "art_route_library":
                    library = list(payload.get("routes", []))
                    if not library:
                        raise ValueError("empty_art_route_library")
                    for item in library:
                        if not item.get("waypoints") or len(item.get("prototype", [])) != 8:
                            raise ValueError("invalid_art_route_entry")
                    self.hidden_goal_route_library = library
                    self.hidden_goal_route_library_vigilance = clamp(
                        float(payload.get("art_vigilance", 0.86))
                    )
                    self.hidden_goal_route_library_exploration = clamp(
                        float(payload.get("selection_exploration_rate", 0.0))
                    )
                    route = []
                else:
                    self.hidden_goal_route_library = []
                    route = [tuple(map(float, point)) for point in payload["waypoints"]]
                if adapter_type == "episodic_route" and not route:
                    raise ValueError("empty_episodic_route")
                adapter = None
                temporal_steps = 1
                self.hidden_goal_route = route
                self.hidden_goal_route_radius = float(payload.get("waypoint_radius", 0.85))
                self.hidden_goal_route_terminal_radius = float(
                    payload.get("terminal_waypoint_radius", 0.30)
                )
                self.hidden_goal_route_terminal_count = max(
                    1, int(payload.get("terminal_waypoint_count", 5))
                )
                self.hidden_goal_route_terminal_extension = max(
                    0.25, float(payload.get("terminal_extension", 2.0))
                )
                self.hidden_goal_food_latch_total_ticks = max(
                    1, int(round(self.hz * float(payload.get("target_latch_seconds", 2.0))))
                )
            elif hidden_dim != self.shadow_policy.hidden_dim:
                raise ValueError("hidden_dim_mismatch")
            elif adapter_type == "temporal_gru":
                temporal_dim = int(payload["temporal_dim"])

                class TemporalAdapter(self.shadow_torch.nn.Module):
                    def __init__(module_self):
                        super().__init__()
                        module_self.input_norm = self.shadow_torch.nn.LayerNorm(hidden_dim)
                        module_self.memory = self.shadow_torch.nn.GRU(
                            hidden_dim, temporal_dim, batch_first=True
                        )
                        module_self.output_norm = self.shadow_torch.nn.LayerNorm(temporal_dim)
                        module_self.actor = self.shadow_torch.nn.Linear(
                            temporal_dim, len(SHADOW_ACTIONS)
                        )

                    def forward(module_self, sequence):
                        values, _hidden = module_self.memory(module_self.input_norm(sequence))
                        return module_self.actor(module_self.output_norm(values[:, -1]))

                adapter = TemporalAdapter()
                temporal_steps = max(2, int(payload["temporal_steps"]))
            elif adapter_type == "linear":
                adapter = self.shadow_torch.nn.Sequential(
                    self.shadow_torch.nn.LayerNorm(hidden_dim),
                    self.shadow_torch.nn.Linear(hidden_dim, len(SHADOW_ACTIONS)),
                )
                temporal_steps = 1
            else:
                raise ValueError("unsupported_adapter_type")
            if adapter is not None:
                adapter.load_state_dict(payload["state_dict"])
                adapter = adapter.eval()
            self.hidden_goal_adapter = adapter
            self.hidden_goal_adapter_type = adapter_type
            self.hidden_goal_adapter_temporal_steps = temporal_steps
            self.hidden_goal_adapter_history = deque(maxlen=temporal_steps)
            self.hidden_goal_route_index = 0
            self.hidden_goal_route_origin = None
            self.hidden_goal_route_selected_id = "none"
            self.hidden_goal_route_art_match = 0.0
            self.hidden_goal_route_distance = 0.0
            self.hidden_goal_route_vetoes = 0
            self.hidden_goal_route_terminal_holds = 0
            self.hidden_goal_route_terminal_active = False
            self.hidden_goal_food_latch_ticks = 0
            self.hidden_goal_food_target = None
            self.hidden_goal_adapter_enabled = True
            self.hidden_goal_adapter_error = "none"
        except Exception as exc:
            self.hidden_goal_adapter = None
            self.hidden_goal_adapter_type = "none"
            self.hidden_goal_adapter_temporal_steps = 1
            self.hidden_goal_adapter_history = deque()
            self.hidden_goal_route = []
            self.hidden_goal_route_library = []
            self.hidden_goal_route_selected_id = "none"
            self.hidden_goal_route_art_match = 0.0
            self.hidden_goal_route_index = 0
            self.hidden_goal_route_origin = None
            self.hidden_goal_route_distance = 0.0
            self.hidden_goal_route_vetoes = 0
            self.hidden_goal_route_terminal_holds = 0
            self.hidden_goal_route_terminal_active = False
            self.hidden_goal_food_latch_ticks = 0
            self.hidden_goal_food_target = None
            self.hidden_goal_adapter_enabled = False
            self.hidden_goal_adapter_error = f"load_failed:{type(exc).__name__}"

    def select_art_route(self, body_state):
        features = art_route_context(body_state)
        if features is None or not self.hidden_goal_route_library:
            return False
        scored = []
        for route in self.hidden_goal_route_library:
            match = fuzzy_art_similarity(features, route["prototype"])
            quality = clamp(float(route.get("quality", 0.5)))
            scored.append((match, quality, route))
        self.hidden_goal_route_art_match = max(match for match, _quality, _route in scored)
        resonant = [
            item for item in scored
            if item[0] >= self.hidden_goal_route_library_vigilance
        ]
        if not resonant:
            return False
        if random.random() < self.hidden_goal_route_library_exploration:
            weights = [
                max(
                    1e-6,
                    art_route_selection_score(
                        match,
                        quality,
                        route,
                    ),
                )
                for match, quality, route in resonant
            ]
            _match, _quality, selected = random.choices(resonant, weights=weights, k=1)[0]
        else:
            _match, _quality, selected = max(
                resonant,
                key=lambda item: (
                    art_route_selection_score(
                        item[0],
                        item[1],
                        item[2],
                    ),
                    item[0],
                    item[2].get("route_id", ""),
                ),
            )
        self.hidden_goal_route = [
            tuple(map(float, point)) for point in selected["waypoints"]
        ]
        self.hidden_goal_route_selected_id = str(selected.get("route_id", "unnamed"))
        self.hidden_goal_route_art_match = float(_match)
        self.hidden_goal_route_index = 0
        self.hidden_goal_route_origin = None
        return bool(self.hidden_goal_route)

    def reset_shadow_episode_state(self):
        """Clear short-lived learned navigation state after a Unity episode reset."""
        if self.shadow_policy is not None:
            self.shadow_hidden = self.shadow_policy.initial_state(1)
        self.shadow_previous_action = 0
        self.shadow_previous_position = None
        self.shadow_previous_proposal = "none"
        self.shadow_last_reward = 0.0
        self.shadow_mpc_engaged = False
        self.shadow_mpc_hold_ticks = 0
        self.shadow_fallback_hold_ticks = 0
        self.shadow_action = "none"
        self.shadow_world_move = (0.0, 0.0)
        self.shadow_continuous_intercept = False
        self.shadow_takeover = False
        self.shadow_episode_idle_ticks = 1
        self.orbit_history.clear()
        self.orbit_path = 0.0
        self.orbit_net = 0.0
        self.orbit_efficiency = 1.0
        self.orbit_adapter_active = False
        self.orbit_adapter_action = "none"
        self.orbit_adapter_confidence = 0.0
        self.orbit_adapter_hold_ticks = 0
        self.orbit_adapter_action_ticks = 0
        self.orbit_adapter_elapsed_ticks = 0
        self.orbit_adapter_selected = None
        self.orbit_adapter_start_position = None
        self.orbit_adapter_displacement = 0.0
        escape_teacher = getattr(self, "escape_teacher", None)
        if escape_teacher is not None:
            escape_teacher.reset("episode_reset")
        self.hidden_goal_adapter_active = False
        self.hidden_goal_adapter_action = "none"
        self.hidden_goal_adapter_confidence = 0.0
        self.hidden_goal_adapter_selected = None
        self.hidden_goal_adapter_hold_ticks = 0
        self.hidden_goal_adapter_lateral_sign = 0
        self.hidden_goal_adapter_lateral_start_position = None
        self.hidden_goal_adapter_lateral_elapsed_ticks = 0
        self.hidden_goal_adapter_history.clear()
        self.hidden_goal_route_index = 0
        self.hidden_goal_route_origin = None
        if getattr(self, "hidden_goal_adapter_type", "none") == "art_route_library":
            self.hidden_goal_route = []
        self.hidden_goal_route_selected_id = "none"
        self.hidden_goal_route_art_match = 0.0
        self.hidden_goal_route_distance = 0.0
        self.hidden_goal_route_vetoes = 0
        self.hidden_goal_route_terminal_holds = 0
        self.hidden_goal_route_terminal_active = False
        self.hidden_goal_food_latch_ticks = 0
        self.hidden_goal_food_target = None
        terrain_air_route_controller = getattr(
            self, "terrain_air_route_controller", None
        )
        if terrain_air_route_controller is not None:
            terrain_air_route_controller.reset("episode_reset")
        systemic_router = getattr(self, "systemic_router", None)
        if systemic_router is not None:
            systemic_router.reset()
        adaptive_gnw_gate = getattr(self, "adaptive_gnw_gate", None)
        if adaptive_gnw_gate is not None:
            adaptive_gnw_gate.reset()
        self.shadow_episode_resets += 1

    def synchronize_shadow_episode(self, body_state):
        label = str(body_state.get("trap_course", "natural_terrain") or "natural_terrain")
        try:
            episode = int(body_state.get("trap_episode", 0) or 0)
        except (TypeError, ValueError):
            episode = 0
        variant = str(body_state.get("trap_course_variant", "standard") or "standard")
        key = (label, variant, episode)
        if self.shadow_episode_key is None:
            self.shadow_episode_key = key
            return False
        if key == self.shadow_episode_key:
            return False
        self.shadow_episode_key = key
        self.reset_shadow_episode_state()
        return True

    def update_shadow_policy(self, body_state, active_action):
        was_shadow_takeover = self.shadow_takeover
        previous_world_move = self.shadow_world_move
        self.shadow_episode_idle_active = False
        self.shadow_ready = False
        self.shadow_takeover = False
        if not self.shadow_enabled or not isinstance(body_state, dict):
            return
        episode_changed = self.synchronize_shadow_episode(body_state)
        if episode_changed or self.shadow_episode_idle_ticks > 0:
            self.shadow_episode_idle_active = True
            self.shadow_episode_idle_ticks = max(0, self.shadow_episode_idle_ticks - 1)
            return
        rays = body_state.get("directional_rays")
        if not isinstance(rays, list) or len(rays) != 8:
            self.shadow_error = "waiting_for_8_rays"
            return
        body_clearance = body_state.get("directional_body_clearance")
        if not isinstance(body_clearance, list) or len(body_clearance) != 8:
            self.shadow_error = "waiting_for_body_clearance"
            return
        try:
            rays = [clamp(float(value)) for value in rays]
            body_clearance = [clamp(float(value)) for value in body_clearance]
            position = (float(body_state.get("x", 0.0)), float(body_state.get("z", 0.0)))
            raw_food_visible = bool(body_state.get("food_visible", False))
            food_distance = max(0.0, float(body_state.get("food_distance", 0.0)))
            direction_x = float(body_state.get("food_world_x", 0.0))
            direction_z = float(body_state.get("food_world_z", 0.0))
        except (TypeError, ValueError):
            self.shadow_error = "invalid_telemetry"
            return

        food_visible = 1.0 if raw_food_visible else 0.0
        course_finished = str(body_state.get("trap_outcome", "running")) in {
            "success",
            "timeout",
        }
        if course_finished:
            food_visible = 0.0
            self.hidden_goal_food_latch_ticks = 0
            self.hidden_goal_food_target = None
        elif raw_food_visible and route_memory_adapter(self.hidden_goal_adapter_type):
            direction_length = max(1e-6, math.hypot(direction_x, direction_z))
            self.hidden_goal_food_target = (
                position[0] + direction_x / direction_length * food_distance,
                position[1] + direction_z / direction_length * food_distance,
            )
            self.hidden_goal_food_latch_ticks = self.hidden_goal_food_latch_total_ticks
        elif (
            route_memory_adapter(self.hidden_goal_adapter_type)
            and self.hidden_goal_food_latch_ticks > 0
            and self.hidden_goal_food_target is not None
        ):
            direction_x = self.hidden_goal_food_target[0] - position[0]
            direction_z = self.hidden_goal_food_target[1] - position[1]
            food_distance = math.hypot(direction_x, direction_z)
            direction_length = max(1e-6, food_distance)
            direction_x /= direction_length
            direction_z /= direction_length
            food_visible = 1.0
            self.hidden_goal_food_latch_ticks -= 1
        elif not raw_food_visible:
            self.hidden_goal_food_latch_ticks = 0
            self.hidden_goal_food_target = None
        food_scale = min(1.0, food_distance / 14.0) if food_visible else 0.0
        food_x = direction_x * food_scale
        food_z = direction_z * food_scale

        orbit_evidence = (
            bool(body_state.get("blocked", False))
            or bool(body_state.get("horizontal_collision", False))
            or food_visible > 0.0
        )
        self.orbit_history.append((position[0], position[1], orbit_evidence, food_visible > 0.0))
        orbit = trajectory_orbit_metrics(self.orbit_history)
        self.orbit_path = float(orbit["path"])
        self.orbit_net = float(orbit["net"])
        self.orbit_efficiency = float(orbit["efficiency"])
        orbit_detected = bool(orbit["detected"]) and len(self.orbit_history) == self.orbit_history.maxlen
        self.orbit_adapter_active = False
        air_route_selected = None
        course_label = body_state.get("trap_course", "natural_terrain")
        if (
            self.terrain_air_route_controller.enabled
            and self.shadow_control == "terrain"
            and course_label in {None, "", "natural_terrain"}
        ):
            air_route_selected = self.terrain_air_route_controller.update(
                body_state,
                self.hunger,
                self.orbit_path,
                self.orbit_efficiency,
                self.physics_wedge_ticks / self.hz,
                self.trap_accumulation_ticks / self.hz,
                SHADOW_ACTIONS,
                SHADOW_VECTORS,
                body_clearance,
                fallback_active=self.shadow_fallback_hold_ticks > 0,
                episodic_authorized=self.systemic_router.authorizes_episodic(
                    self.systemic_conductor.recommendation,
                    self.systemic_conductor.confidence,
                ),
            )

        previous = [0.0] * 8
        previous[self.shadow_previous_action] = 1.0
        observation = rays + [food_visible, food_x, food_z, self.hunger] + previous + [self.shadow_last_reward]
        torch = self.shadow_torch
        with torch.no_grad():
            obs = torch.tensor([observation], dtype=torch.float32)
            logits, _value, self.shadow_hidden = self.shadow_policy.step(obs, self.shadow_hidden)
            if self.hidden_goal_adapter_enabled:
                self.hidden_goal_adapter_history.append(self.shadow_hidden.detach().clone())
            logits = self.shadow_safety_mask(logits, obs)
            body_blocked = torch.tensor([[value < 0.5 for value in body_clearance]], dtype=torch.bool)
            if bool(torch.all(body_blocked).item()):
                body_blocked[0, int(max(range(8), key=lambda index: rays[index]))] = False
            logits = logits.masked_fill(body_blocked, -1e9)
            probabilities = torch.softmax(logits, dim=-1)[0]
            self.shadow_entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()).item())
            recurrent_selected = int(torch.argmax(probabilities).item())
            tactical_obstacle = (
                rays[recurrent_selected] < 0.72
                or bool(body_state.get("blocked", False))
                or bool(body_state.get("horizontal_collision", False))
            )
            if self.shadow_control == "terrain":
                mpc_needed = (
                    food_visible > 0.0
                    or tactical_obstacle
                    or (
                        self.terrain_air_route_controller.control_mode == "guided"
                        and self.terrain_air_route_controller.active
                    )
                    or (
                        self.resource_memory.control_mode == "guided"
                        and self.resource_memory.active
                    )
                )
            else:
                mpc_needed = food_visible > 0.0
            if mpc_needed:
                self.shadow_mpc_hold_ticks = max(
                    self.shadow_mpc_hold_ticks,
                    max(2, int(round(self.hz * 1.5))),
                )
            else:
                self.shadow_mpc_hold_ticks = max(0, self.shadow_mpc_hold_ticks - 1)
            baseline_mpc_engaged = self.shadow_mpc and (
                mpc_needed or self.shadow_mpc_hold_ticks > 0
            )
            executive_recommendation, executive_confidence = (
                self.systemic_executive_signal()
            )
            if (
                self.adaptive_gnw_control == "bounded"
                and executive_recommendation
                != self.systemic_conductor.recommendation
            ):
                self.adaptive_gnw_gate.recommendation_substitutions += 1
            routed_mpc_engaged = self.systemic_router.select_mpc(
                executive_recommendation,
                executive_confidence,
                baseline_mpc_engaged,
                mandatory_mpc=self.shadow_mpc and mpc_needed,
                fallback_active=self.shadow_fallback_hold_ticks > 0,
            )
            self.shadow_mpc_engaged = self.shadow_mpc and routed_mpc_engaged
            self.systemic_conductor.action_influence = (
                self.systemic_router.influence_frames
            )
            if self.shadow_mpc_engaged:
                selected, self.shadow_mpc_score = self.select_mpc_action(
                    obs,
                    probabilities,
                    self.shadow_hidden,
                    body_clearance,
                )
            else:
                selected = recurrent_selected
                self.shadow_mpc_score = 0.0
                self.shadow_mpc_mode = "recurrent"
                self.shadow_mpc_horizon = 0
                self.shadow_mpc_depth = 0.0
                self.shadow_mpc_uncertainty_stops = 0
            self.hidden_goal_adapter_active = False
            self.conductor_gate_active = False
            self.conductor_gate_recommendation = "insufficient_evidence"
            self.conductor_gate_confidence = 0.0
            conductor_allows_episodic = True
            if (
                self.conductor_control == "familiar_hidden_goal"
                and self.trap_course_label == "lwall"
                and food_visible <= 0.0
            ):
                (
                    self.conductor_gate_recommendation,
                    self.conductor_gate_confidence,
                ) = self.conductor_observer.recommend("familiar_hidden_goal")
                self.conductor_gate_active = True
                self.conductor_gate_decisions += 1
                conductor_allows_episodic = conductor_gate_allows(
                    self.conductor_control,
                    "familiar_hidden_goal",
                    self.conductor_gate_recommendation,
                    self.conductor_gate_confidence,
                )
                if not conductor_allows_episodic:
                    self.conductor_gate_denials += 1
            if (
                self.hidden_goal_adapter_enabled
                and self.hidden_goal_adapter_type == "art_route_library"
                and self.trap_course_label == "lwall"
                and food_visible <= 0.0
                and conductor_allows_episodic
                and not self.hidden_goal_route
            ):
                self.select_art_route(body_state)
            hidden_goal_scope = (
                self.hidden_goal_adapter_enabled
                and self.trap_course_label == "lwall"
                and food_visible <= 0.0
                and conductor_allows_episodic
                and len(self.hidden_goal_adapter_history)
                >= self.hidden_goal_adapter_temporal_steps
                and (
                    self.hidden_goal_adapter_type != "art_route_library"
                    or bool(self.hidden_goal_route)
                )
            )
            if hidden_goal_scope:
                if self.hidden_goal_adapter_type in {"episodic_route", "art_route_library"}:
                    if self.hidden_goal_route_origin is None:
                        self.hidden_goal_route_origin = position
                    route_last = len(self.hidden_goal_route) - 1
                    while self.hidden_goal_route_index < route_last:
                        waypoint = self.hidden_goal_route[self.hidden_goal_route_index]
                        target = (
                            self.hidden_goal_route_origin[0] + waypoint[0],
                            self.hidden_goal_route_origin[1] + waypoint[1],
                        )
                        delta = (target[0] - position[0], target[1] - position[1])
                        waypoint_radius = route_waypoint_radius(
                            self.hidden_goal_route_index,
                            len(self.hidden_goal_route),
                            self.hidden_goal_route_radius,
                            self.hidden_goal_route_terminal_radius,
                            self.hidden_goal_route_terminal_count,
                        )
                        if math.hypot(*delta) > waypoint_radius:
                            break
                        self.hidden_goal_route_index += 1
                    waypoint = self.hidden_goal_route[self.hidden_goal_route_index]
                    target = (
                        self.hidden_goal_route_origin[0] + waypoint[0],
                        self.hidden_goal_route_origin[1] + waypoint[1],
                    )
                    delta = (target[0] - position[0], target[1] - position[1])
                    self.hidden_goal_route_distance = math.hypot(*delta)
                    if (
                        self.hidden_goal_route_index == route_last
                        and self.hidden_goal_route_distance <= self.hidden_goal_route_terminal_radius
                    ):
                        self.hidden_goal_route_terminal_active = True
                    if self.hidden_goal_route_terminal_active:
                        previous = self.hidden_goal_route[max(0, route_last - 1)]
                        terminal_vector = (waypoint[0] - previous[0], waypoint[1] - previous[1])
                        terminal_length = max(1e-6, math.hypot(*terminal_vector))
                        terminal_direction = (
                            terminal_vector[0] / terminal_length,
                            terminal_vector[1] / terminal_length,
                        )
                        delta = (
                            target[0] + terminal_direction[0] * self.hidden_goal_route_terminal_extension
                            - position[0],
                            target[1] + terminal_direction[1] * self.hidden_goal_route_terminal_extension
                            - position[1],
                        )
                        self.hidden_goal_route_terminal_holds += 1
                    length = max(1e-6, math.hypot(*delta))
                    desired = (delta[0] / length, delta[1] / length)
                    ideal = max(
                        range(len(SHADOW_ACTIONS)),
                        key=lambda index: (
                            SHADOW_VECTORS[SHADOW_ACTIONS[index]][0] * desired[0]
                            + SHADOW_VECTORS[SHADOW_ACTIONS[index]][1] * desired[1]
                        ),
                    )
                    safe = [index for index, clear in enumerate(body_clearance) if clear >= 0.5]
                    candidate = max(
                        safe,
                        key=lambda index: (
                            SHADOW_VECTORS[SHADOW_ACTIONS[index]][0] * desired[0]
                            + SHADOW_VECTORS[SHADOW_ACTIONS[index]][1] * desired[1]
                        ),
                        default=selected,
                    )
                    if candidate != ideal:
                        self.hidden_goal_route_vetoes += 1
                    candidate_confidence = 1.0
                    adapter_logits = None
                elif self.hidden_goal_adapter_type == "temporal_gru":
                    adapter_input = self.shadow_torch.stack(
                        list(self.hidden_goal_adapter_history), dim=1
                    )
                    adapter_logits = self.hidden_goal_adapter(adapter_input)
                else:
                    adapter_input = self.shadow_hidden
                    adapter_logits = self.hidden_goal_adapter(adapter_input)
                if adapter_logits is not None:
                    adapter_logits = self.shadow_safety_mask(adapter_logits, obs)
                    adapter_logits = adapter_logits.masked_fill(body_blocked, -1e9)
                if adapter_logits is not None and self.hidden_goal_adapter_lateral_sign:
                    lateral_penalty = self.shadow_torch.tensor(
                        [
                            self.hidden_goal_adapter_lateral_bias
                            if SHADOW_VECTORS[action][0] * self.hidden_goal_adapter_lateral_sign < 0.0
                            else 0.0
                            for action in SHADOW_ACTIONS
                        ],
                        dtype=adapter_logits.dtype,
                        device=adapter_logits.device,
                    )
                    adapter_logits = adapter_logits - lateral_penalty.unsqueeze(0)
                if adapter_logits is not None:
                    adapter_probabilities = self.shadow_torch.softmax(adapter_logits, dim=-1)[0]
                    candidate = int(self.shadow_torch.argmax(adapter_probabilities).item())
                    candidate_confidence = float(adapter_probabilities[candidate].item())
                candidate_lateral = SHADOW_VECTORS[SHADOW_ACTIONS[candidate]][0]
                if (
                    self.hidden_goal_adapter_lateral_bias > 0.0
                    and not self.hidden_goal_adapter_lateral_sign
                    and candidate_confidence >= self.hidden_goal_adapter_confidence_gate
                    and abs(candidate_lateral) > 0.0
                ):
                    self.hidden_goal_adapter_lateral_sign = 1 if candidate_lateral > 0.0 else -1
                    self.hidden_goal_adapter_lateral_start_position = position
                    self.hidden_goal_adapter_lateral_elapsed_ticks = 0
                held_action_safe = (
                    self.hidden_goal_adapter_selected is not None
                    and body_clearance[self.hidden_goal_adapter_selected] >= 0.5
                )
                if self.hidden_goal_adapter_hold_ticks <= 0 or not held_action_safe:
                    self.hidden_goal_adapter_selected = candidate
                    self.hidden_goal_adapter_hold_ticks = self.hidden_goal_adapter_commit_ticks
                if candidate_confidence >= self.hidden_goal_adapter_confidence_gate:
                    selected = self.hidden_goal_adapter_selected
                    self.hidden_goal_adapter_hold_ticks -= 1
                    self.hidden_goal_adapter_active = True
                    if self.conductor_gate_active:
                        self.conductor_gate_frames += 1
                        self.conductor_observer.action_influence += 1
                    self.hidden_goal_adapter_action = SHADOW_ACTIONS[selected]
                    self.hidden_goal_adapter_confidence = candidate_confidence
                    self.hidden_goal_adapter_events += 1
                    self.shadow_mpc_mode = (
                        "lwall_art_route"
                        if self.hidden_goal_adapter_type == "art_route_library"
                        else "lwall_episodic_route"
                        if self.hidden_goal_adapter_type == "episodic_route"
                        else "lwall_hidden_goal_adapter"
                    )
                    self.hidden_goal_adapter_lateral_elapsed_ticks += 1
                    if self.hidden_goal_adapter_lateral_start_position is not None:
                        lateral_displacement = math.hypot(
                            position[0] - self.hidden_goal_adapter_lateral_start_position[0],
                            position[1] - self.hidden_goal_adapter_lateral_start_position[1],
                        )
                        if (
                            self.hidden_goal_adapter_lateral_elapsed_ticks
                            >= max(2, int(round(self.hz * 2.0)))
                            and lateral_displacement < 1.0
                        ):
                            self.hidden_goal_adapter_lateral_sign = 0
                            self.hidden_goal_adapter_lateral_start_position = None
                            self.hidden_goal_adapter_lateral_elapsed_ticks = 0
                            self.hidden_goal_adapter_lateral_resets += 1
                else:
                    self.hidden_goal_adapter_action = "none"
                    self.hidden_goal_adapter_confidence = candidate_confidence
            else:
                self.hidden_goal_adapter_action = "none"
                self.hidden_goal_adapter_confidence = 0.0
                self.hidden_goal_adapter_selected = None
                self.hidden_goal_adapter_hold_ticks = 0
                self.hidden_goal_adapter_lateral_sign = 0
                self.hidden_goal_adapter_lateral_start_position = None
                self.hidden_goal_adapter_lateral_elapsed_ticks = 0
            recovery_running = (
                self.orbit_adapter_enabled
                and self.orbit_adapter_selected is not None
                and self.orbit_adapter_hold_ticks > 0
            )
            if (orbit_detected or recovery_running) and self.orbit_adapter_enabled:
                adapter_logits = self.orbit_adapter(self.shadow_hidden)
                adapter_logits = self.shadow_safety_mask(adapter_logits, obs)
                adapter_logits = adapter_logits.masked_fill(body_blocked, -1e9)
                adapter_probabilities = self.shadow_torch.softmax(adapter_logits, dim=-1)[0]
                candidate = int(self.shadow_torch.argmax(adapter_probabilities).item())
                candidate_confidence = float(adapter_probabilities[candidate].item())
                if not recovery_running and candidate_confidence >= self.orbit_adapter_confidence_gate:
                    self.orbit_adapter_selected = candidate
                    self.orbit_adapter_hold_ticks = max(2, int(round(self.hz * 6.0)))
                    self.orbit_adapter_action_ticks = max(2, int(round(self.hz * 0.8)))
                    self.orbit_adapter_elapsed_ticks = 0
                    self.orbit_adapter_start_position = position
                    self.orbit_adapter_displacement = 0.0
                    self.orbit_adapter_events += 1
                    recovery_running = True
                if recovery_running:
                    if (
                        body_clearance[self.orbit_adapter_selected] < 0.5
                        or self.orbit_adapter_action_ticks <= 0
                    ):
                        # Commit to recovery, not one direction: refresh the safe
                        # sub-action as local geometry changes along the exit path.
                        self.orbit_adapter_selected = candidate
                        self.orbit_adapter_action_ticks = max(
                            2, int(round(self.hz * 0.8))
                        )
                    selected = self.orbit_adapter_selected
                    self.orbit_adapter_elapsed_ticks += 1
                    self.orbit_adapter_hold_ticks -= 1
                    self.orbit_adapter_action_ticks -= 1
                    if self.orbit_adapter_start_position is not None:
                        self.orbit_adapter_displacement = math.hypot(
                            position[0] - self.orbit_adapter_start_position[0],
                            position[1] - self.orbit_adapter_start_position[1],
                        )
                    self.orbit_adapter_confidence = float(adapter_probabilities[selected].item())
                    self.orbit_adapter_action = SHADOW_ACTIONS[selected]
                    self.orbit_adapter_active = True
                    self.shadow_mpc_mode = "orbit_exit_adapter"
                    path_clear = (
                        body_clearance[selected] >= 0.5
                        and not bool(body_state.get("blocked", False))
                        and not bool(body_state.get("horizontal_collision", False))
                    )
                    if orbit_recovery_should_finish(
                        self.orbit_adapter_elapsed_ticks,
                        self.orbit_adapter_displacement,
                        path_clear,
                        max(2, int(round(self.hz * 2.5))),
                        max(2, int(round(self.hz * 6.0))),
                    ):
                        self.orbit_adapter_hold_ticks = 0
                        self.orbit_adapter_action_ticks = 0
                        self.orbit_adapter_elapsed_ticks = 0
                        self.orbit_adapter_selected = None
                        self.orbit_adapter_start_position = None
                else:
                    self.orbit_adapter_hold_ticks = 0
                    self.orbit_adapter_action_ticks = 0
                    self.orbit_adapter_elapsed_ticks = 0
                    self.orbit_adapter_selected = None
                    self.orbit_adapter_start_position = None
                    self.orbit_adapter_displacement = 0.0
                    self.orbit_adapter_action = "none"
                    self.orbit_adapter_confidence = candidate_confidence
            else:
                self.orbit_adapter_hold_ticks = 0
                self.orbit_adapter_action_ticks = 0
                self.orbit_adapter_elapsed_ticks = 0
                self.orbit_adapter_selected = None
                self.orbit_adapter_start_position = None
                self.orbit_adapter_displacement = 0.0
                self.orbit_adapter_action = "none"
                self.orbit_adapter_confidence = 0.0
            if air_route_selected is not None:
                selected = air_route_selected
                self.shadow_mpc_mode = "terrain_air_bounded_route"
            teacher_enabled = (
                self.terrain_air_route_controller.enabled
                and self.terrain_air_route_controller.teacher_memory_path is not None
                and self.shadow_control == "terrain"
                and course_label in {None, "", "natural_terrain"}
            )
            if self.escape_teacher.active and self.shadow_fallback_hold_ticks > 0:
                self.escape_teacher.reset("stable_fallback")
            if (
                teacher_enabled
                and not self.escape_teacher.active
                and orbit_detected
                and self.terrain_air_route_controller.recommendation == "no_resonance"
                and not self.terrain_air_route_controller.active
                and not self.terrain_air_route_controller.pending
                and self.shadow_fallback_hold_ticks <= 0
            ):
                prototype = context_vector(
                    body_state,
                    self.hunger,
                    self.orbit_path,
                    self.orbit_efficiency,
                    self.physics_wedge_ticks / self.hz,
                    self.trap_accumulation_ticks / self.hz,
                )
                if prototype is not None:
                    self.escape_teacher.start(
                        body_state,
                        prototype,
                        SHADOW_ACTIONS,
                        rays,
                        body_clearance,
                    )
            if self.escape_teacher.active:
                previous_successes = self.escape_teacher.successes
                teacher_selected = self.escape_teacher.update(
                    body_state,
                    SHADOW_ACTIONS,
                    rays,
                    body_clearance,
                    self.terrain_air_route_controller,
                )
                if self.escape_teacher.successes > previous_successes:
                    self.orbit_history.clear()
                    self.orbit_path = 0.0
                    self.orbit_net = 0.0
                    self.orbit_efficiency = 1.0
                    orbit_detected = False
                if teacher_selected is not None:
                    selected = teacher_selected
                    self.shadow_mpc_mode = "escape_teacher"
        if self.shadow_previous_position is not None and self.shadow_previous_proposal in SHADOW_VECTORS:
            dx = position[0] - self.shadow_previous_position[0]
            dz = position[1] - self.shadow_previous_position[1]
            distance = math.hypot(dx, dz)
            if distance >= 0.01:
                proposed = SHADOW_VECTORS[self.shadow_previous_proposal]
                proposed_norm = math.hypot(*proposed)
                cosine = (dx * proposed[0] + dz * proposed[1]) / (distance * proposed_norm)
                self.shadow_agreement = clamp(0.5 * (cosine + 1.0))
        self.shadow_action = SHADOW_ACTIONS[selected]
        selected_vector = SHADOW_VECTORS[self.shadow_action]
        selected_length = max(1e-6, math.hypot(*selected_vector))
        self.shadow_world_move = (
            selected_vector[0] / selected_length,
            selected_vector[1] / selected_length,
        )
        self.shadow_continuous_intercept = False
        if (
            route_memory_adapter(self.hidden_goal_adapter_type)
            and food_visible > 0.0
            and self.hidden_goal_food_target is not None
        ):
            target_delta = (
                self.hidden_goal_food_target[0] - position[0],
                self.hidden_goal_food_target[1] - position[1],
            )
            target_distance = math.hypot(*target_delta)
            if target_distance > 1e-4:
                target_direction = (
                    target_delta[0] / target_distance,
                    target_delta[1] / target_distance,
                )
                nearest_target_action = max(
                    range(len(SHADOW_ACTIONS)),
                    key=lambda index: (
                        SHADOW_VECTORS[SHADOW_ACTIONS[index]][0] * target_direction[0]
                        + SHADOW_VECTORS[SHADOW_ACTIONS[index]][1] * target_direction[1]
                    ),
                )
                if body_clearance[nearest_target_action] >= 0.5:
                    self.shadow_world_move = smooth_target_intercept(
                        previous_world_move,
                        target_delta,
                        target_distance,
                    )
                    self.shadow_continuous_intercept = True
        self.shadow_confidence = float(torch.max(probabilities).item()) if self.shadow_mpc else float(probabilities[selected].item())
        self.shadow_probabilities = [float(value) for value in probabilities.tolist()]
        self.shadow_body_safe_actions = sum(value >= 0.5 for value in body_clearance)
        self.shadow_previous_position = position
        self.shadow_previous_proposal = self.shadow_action
        self.shadow_last_reward = 0.0
        self.shadow_ready = True
        self.shadow_error = "none"
        selected_ray = rays[selected]
        clearance_required = 0.75 * math.hypot(*SHADOW_VECTORS[self.shadow_action]) / self.shadow_ray_range
        confidence_gate = max(0.0, self.shadow_control_confidence - (0.12 if was_shadow_takeover else 0.0))
        base_learned_control = (
            self.shadow_confidence >= confidence_gate
            and selected_ray >= clearance_required
            and body_clearance[selected] >= 0.5
            and not self.sleeping
            and not self.waking
        )
        safe_food_control = (
            self.shadow_control in {"food", "course"}
            and food_visible > 0.0
            and not self.current_stuck
            and not bool(body_state.get("blocked", False))
            and not bool(body_state.get("horizontal_collision", False))
            and self.local_trap_pressure(body_state) < 0.20
        )
        course_label = body_state.get("trap_course", "natural_terrain")
        if course_label not in {None, "", "natural_terrain"}:
            self.shadow_food_sensor_radius = 13.0 if self.hunger >= 0.92 else 10.0 if self.hunger >= 0.70 else 7.0
        else:
            self.shadow_food_sensor_radius = 28.0 if self.hunger >= 0.92 else 22.0 if self.hunger >= 0.70 else 16.0
        safe_course_control = (
            self.shadow_control == "course"
            and course_label not in {None, "", "natural_terrain"}
        )
        safe_terrain_control = (
            self.shadow_control == "terrain"
            and course_label in {None, "", "natural_terrain"}
        )
        fallback_threshold = int(round(self.hz * 8.0))
        route_controller = self.terrain_air_route_controller
        if stable_recovery_due(
            self.physics_wedge_ticks,
            self.trap_accumulation_ticks,
            fallback_threshold,
            sustained_orbit=orbit_teacher_due(
                orbit_detected,
                route_active=route_controller.active or self.escape_teacher.active,
                route_pending=route_controller.pending,
            ),
        ) and self.shadow_fallback_hold_ticks <= 0:
            self.shadow_fallback_hold_ticks = max(1, int(round(self.hz * 12.0)))
            self.breakout_plan.clear()
            self.escape_ticks = 0
            self.escape_action = None
            self.stuck_cooldown = 0
        fallback_active = self.shadow_fallback_hold_ticks > 0
        if fallback_active:
            self.shadow_fallback_hold_ticks -= 1
        self.shadow_takeover = base_learned_control and (
            safe_food_control or safe_course_control or safe_terrain_control
        ) and not fallback_active
        self.shadow_previous_action = selected if self.shadow_takeover else self.action_index(active_action)
        if self.shadow_takeover:
            self.shadow_takeover_steps += 1

    def select_mpc_action(self, obs, probabilities, next_hidden, body_clearance):
        normalized_entropy = self.shadow_entropy / math.log(len(SHADOW_ACTIONS)) if self.shadow_entropy > 0.0 else 0.0
        proposed = int(self.shadow_torch.argmax(probabilities).item())
        proposed_clearance = float(obs[0, proposed].item())
        horizon = 4
        if normalized_entropy >= 0.55 or proposed_clearance < 0.48 or self.physics_wedge_ticks > 0:
            horizon = 6
        if self.hunger >= 0.70 or self.physics_wedge_ticks >= int(round(self.hz * 3.0)):
            horizon = max(horizon, 6)
        self.shadow_mpc_mode = "critical_targeting" if self.hunger >= 0.92 else "adaptive_stochastic"
        self.shadow_mpc_horizon = horizon
        self.shadow_mpc_planning_frames += 1
        if self.hunger >= 0.92:
            self.shadow_mpc_critical_frames += 1
        return self.select_stochastic_mpc_action(
            obs,
            probabilities,
            next_hidden,
            body_clearance,
            horizon=horizon,
        )

    def select_consensus_mpc_action(self, obs, probabilities, next_hidden, body_clearance, horizon=4):
        torch = self.shadow_torch
        previous = int(torch.argmax(obs[0, 12:20]).item())
        scores = [-math.inf] * len(SHADOW_ACTIONS)
        with torch.no_grad():
            for root in range(len(SHADOW_ACTIONS)):
                if body_clearance[root] < 0.5:
                    continue
                imagined_hidden = next_hidden.clone()
                score = 0.18 * float(torch.log(probabilities[root].clamp_min(1e-8)).item())
                prior_action = previous
                for depth in range(horizon):
                    action = torch.tensor([root], dtype=torch.long)
                    ensemble = self.shadow_policy.predict_core(imagined_hidden, action)
                    core = ensemble.mean(dim=0)
                    core[:, :8] = core[:, :8].clamp(0.0, 1.0)
                    core[:, 8:9] = core[:, 8:9].clamp(0.0, 1.0)
                    core[:, 9:11] = core[:, 9:11].clamp(-1.0, 1.0)
                    core[:, 11:12] = core[:, 11:12].clamp(0.0, 1.0)
                    uncertainty = float(torch.var(ensemble, dim=0).mean().item())
                    clearance = float(core[0, root].item())
                    visible = float(core[0, 8].item())
                    food_distance = float(torch.linalg.vector_norm(core[0, 9:11]).item())
                    collision_risk = max(0.0, 0.12 - clearance) * 8.0
                    old_vector = SHADOW_VECTORS[SHADOW_ACTIONS[prior_action]]
                    new_vector = SHADOW_VECTORS[SHADOW_ACTIONS[root]]
                    cosine = (old_vector[0] * new_vector[0] + old_vector[1] * new_vector[1]) / (
                        math.hypot(*old_vector) * math.hypot(*new_vector)
                    )
                    jerk = 0.5 * (1.0 - max(-1.0, min(1.0, cosine))) if depth == 0 else 0.0
                    score += 0.08 * visible * (1.0 - min(1.0, food_distance))
                    score -= 0.34 * collision_risk + 0.025 * jerk + 0.16 * uncertainty
                    previous_one_hot = torch.zeros(1, len(SHADOW_ACTIONS))
                    previous_one_hot[0, root] = 1.0
                    estimated_reward = torch.tensor([[max(-0.2, min(0.2, score / (depth + 1)))]])
                    imagined_obs = torch.cat([core, previous_one_hot, estimated_reward], dim=-1)
                    _logits, _value, imagined_hidden = self.shadow_policy.step(imagined_obs, imagined_hidden)
                    prior_action = root
                scores[root] = score
        selected = max(range(len(scores)), key=lambda index: scores[index])
        return selected, float(scores[selected])

    def select_stochastic_mpc_action(
        self,
        obs,
        probabilities,
        next_hidden,
        body_clearance,
        horizon,
        samples=3,
        uncertainty_budget=0.0015,
    ):
        torch = self.shadow_torch
        roots = len(SHADOW_ACTIONS)
        batch = roots * samples
        root_actions = torch.arange(roots, dtype=torch.long).repeat_interleave(samples)
        imagined_hidden = next_hidden.repeat(batch, 1)
        scores = 0.18 * torch.log(probabilities[root_actions].clamp_min(1e-8))
        cumulative_uncertainty = torch.zeros(batch)
        active = torch.ones(batch, dtype=torch.bool)
        depths = torch.zeros(batch)
        previous = int(torch.argmax(obs[0, 12:20]).item())
        previous_actions = torch.full((batch,), previous, dtype=torch.long)
        move_tensor = torch.tensor(
            [SHADOW_VECTORS[action] for action in SHADOW_ACTIONS],
            dtype=torch.float32,
        )
        move_tensor /= torch.linalg.vector_norm(move_tensor, dim=-1, keepdim=True)
        uncertainty_stops = 0

        with torch.no_grad():
            for depth in range(horizon):
                ensemble = self.shadow_policy.predict_core(imagined_hidden, root_actions)
                disagreement = torch.var(ensemble, dim=0).mean(dim=-1)
                consensus = ensemble.mean(dim=0)
                head_indices = torch.randint(ensemble.shape[0], (batch,))
                batch_indices = torch.arange(batch)
                sampled = ensemble[head_indices, batch_indices]
                core = (consensus + 0.35 * (sampled - consensus)).clone()
                core[:, :8] = core[:, :8].clamp(0.0, 1.0)
                core[:, 8:9] = core[:, 8:9].clamp(0.0, 1.0)
                core[:, 9:11] = core[:, 9:11].clamp(-1.0, 1.0)
                core[:, 11:12] = core[:, 11:12].clamp(0.0, 1.0)

                cumulative_uncertainty += disagreement * active.float()
                trusted = active & (cumulative_uncertainty <= uncertainty_budget)
                uncertainty_stops += int(torch.sum(active & ~trusted).item())
                step_mask = active.float()
                clearance = core[batch_indices, root_actions]
                visible = torch.sigmoid(5.0 * (core[:, 8] - 0.5))
                food_distance = torch.linalg.vector_norm(core[:, 9:11], dim=-1).clamp(0.0, 1.0)
                collision_risk = torch.clamp(0.12 - clearance, min=0.0) * 8.0
                cosine = torch.sum(move_tensor[previous_actions] * move_tensor[root_actions], dim=-1).clamp(-1.0, 1.0)
                jerk = 0.5 * (1.0 - cosine) if depth == 0 else torch.zeros(batch)
                step_score = (
                    0.08 * visible * (1.0 - food_distance)
                    - 0.34 * collision_risk
                    - 0.025 * jerk
                    - 0.16 * disagreement
                )
                scores += step_mask * step_score
                depths += step_mask

                previous_one_hot = torch.zeros(batch, roots)
                previous_one_hot[batch_indices, root_actions] = 1.0
                estimated_reward = torch.clamp(scores / float(depth + 1), -0.2, 0.2).unsqueeze(-1)
                imagined_obs = torch.cat([core, previous_one_hot, estimated_reward], dim=-1)
                _logits, _value, imagined_hidden = self.shadow_policy.step(imagined_obs, imagined_hidden)
                previous_actions = root_actions
                active = trusted
                if not bool(torch.any(active).item()):
                    break

        sample_scores = scores.reshape(roots, samples)
        mean = sample_scores.mean(dim=-1)
        downside = torch.quantile(sample_scores, 0.25, dim=-1)
        spread = sample_scores.std(dim=-1, unbiased=False)
        risk_adjusted = 0.75 * mean + 0.25 * downside - 0.25 * spread
        for root, clearance in enumerate(body_clearance):
            if clearance < 0.5:
                risk_adjusted[root] = -math.inf
        air_controller = self.terrain_air_route_controller
        guidance_applied = False
        unguided_selected = int(torch.argmax(risk_adjusted).item())
        if (
            air_controller.control_mode == "guided"
            and air_controller.active
            and air_controller.guidance_weight > 0.0
        ):
            guidance = torch.tensor(
                air_controller.guidance_vector,
                dtype=move_tensor.dtype,
            )
            guidance_length = torch.linalg.vector_norm(guidance)
            if float(guidance_length.item()) > 1e-6:
                finite_scores = risk_adjusted[torch.isfinite(risk_adjusted)]
                if len(finite_scores) >= 2:
                    top_scores = torch.topk(finite_scores, 2).values
                    air_controller.last_unguided_margin = float(
                        (top_scores[0] - top_scores[1]).item()
                    )
                else:
                    air_controller.last_unguided_margin = math.inf
                ambiguity = score_margin_ambiguity(
                    air_controller.last_unguided_margin,
                    air_controller.guidance_margin_threshold,
                )
                effective_weight = air_controller.guidance_weight * ambiguity
                air_controller.last_margin_ambiguity = ambiguity
                air_controller.last_effective_guidance_weight = effective_weight
                if effective_weight > 0.0:
                    guidance /= guidance_length
                    alignment = move_tensor @ guidance
                    risk_adjusted += effective_weight * alignment
                    air_controller.action_influence += 1
                    air_controller.guidance_decisions += 1
                    guidance_applied = True
                    self.shadow_mpc_mode = "terrain_air_guided_mpc"
        resource_memory = self.resource_memory
        resource_memory.last_effective_guidance_weight = 0.0
        resource_memory.last_margin_ambiguity = 0.0
        resource_unguided = int(torch.argmax(risk_adjusted).item())
        if (
            resource_memory.control_mode == "guided"
            and resource_memory.active
            and not self.pgnw_experiment_planner.protective_memory_active
            and not air_controller.active
            and self.shadow_fallback_hold_ticks <= 0
        ):
            finite_scores = risk_adjusted[torch.isfinite(risk_adjusted)]
            if len(finite_scores) >= 2:
                top_scores = torch.topk(finite_scores, 2).values
                margin = float((top_scores[0] - top_scores[1]).item())
            else:
                margin = math.inf
            ambiguity = score_margin_ambiguity(
                margin,
                resource_memory.guidance_margin_threshold,
            )
            pressure = clamp(
                (self.hunger - resource_memory.hunger_gate)
                / max(1e-6, 1.0 - resource_memory.hunger_gate)
            )
            effective_weight = (
                resource_memory.max_guidance_weight
                * resource_memory.confidence
                * (0.30 + 0.70 * pressure)
                * ambiguity
            )
            resource_memory.last_unguided_margin = margin
            resource_memory.last_margin_ambiguity = ambiguity
            resource_memory.last_effective_guidance_weight = effective_weight
            if effective_weight > 0.0:
                guidance = torch.tensor(
                    resource_memory.guidance_vector,
                    dtype=move_tensor.dtype,
                )
                guidance_length = torch.linalg.vector_norm(guidance)
                if float(guidance_length.item()) > 1e-6:
                    guidance /= guidance_length
                    risk_adjusted += effective_weight * (move_tensor @ guidance)
                    resource_memory.action_influence += 1
                    resource_memory.guidance_decisions += 1
                    self.shadow_mpc_mode = "resource_memory_guided_mpc"
        experiment_planner = self.pgnw_experiment_planner
        experiment_planner.last_effective_guidance_weight = 0.0
        experiment_unguided = int(torch.argmax(risk_adjusted).item())
        experiment_committed_selected = None
        experiment_retreat_active = False
        if pgnw_experiment_guidance_allowed(
            experiment_planner.mode,
            experiment_planner.guidance_active,
            self.shadow_fallback_hold_ticks > 0,
            self.current_stuck,
            self.hunger,
            air_controller.control_mode == "guided" and air_controller.active,
            (
                resource_memory.control_mode == "guided"
                and resource_memory.active
                and not experiment_planner.protective_memory_active
            ),
        ):
            guidance = torch.tensor(
                experiment_planner.guidance_vector,
                dtype=move_tensor.dtype,
            )
            guidance_length = torch.linalg.vector_norm(guidance)
            if float(guidance_length.item()) > 1e-6:
                guidance /= guidance_length
                target_alignment = move_tensor @ guidance
                if experiment_planner.mode in {"committed", "dynamic_committed"}:
                    experiment_regret = experiment_planner.max_score_regret
                    if (
                        experiment_planner.isolation_active
                        and experiment_planner.isolation_retreat_remaining_ticks > 0
                    ):
                        experiment_retreat_active = True
                        experiment_regret = (
                            experiment_planner.isolation_retreat_max_score_regret
                        )
                    experiment_committed_selected = (
                        pgnw_constrained_target_action(
                            risk_adjusted.detach().tolist(),
                            target_alignment.detach().tolist(),
                            experiment_regret,
                        )
                    )
                    if experiment_committed_selected is not None:
                        experiment_planner.last_effective_guidance_weight = (
                            experiment_regret
                        )
                        experiment_planner.guidance_decisions += 1
                        if experiment_planner.isolation_active:
                            experiment_planner.isolation_decisions += 1
                            if experiment_retreat_active:
                                experiment_planner.isolation_retreat_decisions += 1
                                experiment_planner.isolation_retreat_remaining_ticks -= 1
                            self.shadow_mpc_mode = "pgnw_isolation_mpc"
                        else:
                            self.shadow_mpc_mode = "pgnw_committed_mpc"
                else:
                    finite_scores = risk_adjusted[torch.isfinite(risk_adjusted)]
                    if len(finite_scores) >= 2:
                        top_scores = torch.topk(finite_scores, 2).values
                        margin = float((top_scores[0] - top_scores[1]).item())
                    else:
                        margin = math.inf
                    ambiguity = score_margin_ambiguity(
                        margin,
                        experiment_planner.guidance_margin_threshold,
                    )
                    effective_weight = (
                        experiment_planner.guidance_weight * ambiguity
                    )
                    if effective_weight > 0.0:
                        risk_adjusted += effective_weight * (
                            target_alignment
                        )
                        experiment_planner.last_effective_guidance_weight = (
                            effective_weight
                        )
                        experiment_planner.guidance_decisions += 1
                        self.shadow_mpc_mode = "pgnw_experiment_guided_mpc"
        selected = (
            experiment_committed_selected
            if experiment_committed_selected is not None
            else int(torch.argmax(risk_adjusted).item())
        )
        if experiment_planner.last_effective_guidance_weight > 0.0:
            changed = int(selected != experiment_unguided)
            experiment_planner.action_influence += changed
            if experiment_planner.protective_rule_active:
                experiment_planner.protective_action_influence += changed
            if experiment_planner.isolation_active:
                experiment_planner.isolation_action_influence += changed
                if experiment_retreat_active:
                    experiment_planner.isolation_retreat_action_influence += (
                        changed
                    )
        resource_memory.last_unguided_action = SHADOW_ACTIONS[
            resource_unguided
        ]
        resource_memory.last_guided_action = SHADOW_ACTIONS[selected]
        if resource_memory.last_effective_guidance_weight > 0.0:
            resource_memory.guidance_action_changes += int(
                selected != resource_unguided
            )
        if guidance_applied:
            air_controller.last_unguided_action = SHADOW_ACTIONS[unguided_selected]
            air_controller.last_guided_action = SHADOW_ACTIONS[selected]
            air_controller.guidance_action_changes += int(
                selected != unguided_selected
            )
        elif air_controller.control_mode == "guided":
            air_controller.last_unguided_action = SHADOW_ACTIONS[unguided_selected]
            air_controller.last_guided_action = SHADOW_ACTIONS[selected]
        self.shadow_mpc_depth = float(depths.reshape(roots, samples)[selected].mean().item())
        self.shadow_mpc_uncertainty_stops = uncertainty_stops
        return selected, float(risk_adjusted[selected].item())

    @staticmethod
    def action_index(action):
        if action in SHADOW_ACTIONS:
            return SHADOW_ACTIONS.index(action)
        vector = MOVE_VECTORS.get(action, (0.0, 0.0))
        return max(
            range(8),
            key=lambda index: vector[0] * SHADOW_VECTORS[SHADOW_ACTIONS[index]][0]
            + vector[1] * SHADOW_VECTORS[SHADOW_ACTIONS[index]][1],
        )

    @staticmethod
    def action_agreement(active_action, shadow_action):
        active = MOVE_VECTORS.get(active_action, (0.0, 0.0))
        shadow = SHADOW_VECTORS.get(shadow_action, (0.0, 0.0))
        active_norm = math.hypot(*active)
        shadow_norm = math.hypot(*shadow)
        if active_norm < 1e-6 or shadow_norm < 1e-6:
            return 0.0
        cosine = (active[0] * shadow[0] + active[1] * shadow[1]) / (active_norm * shadow_norm)
        return clamp(0.5 * (cosine + 1.0))

    @property
    def sleeping(self):
        return self.sleep_remaining > 0

    @property
    def waking(self):
        return self.wake_remaining > 0

    def update_from_body(self, body_state):
        if isinstance(body_state, dict):
            self.trap_course_label = body_state.get("trap_course", self.trap_course_label)
            self.trap_course_variant = body_state.get(
                "trap_course_variant", self.trap_course_variant
            )
            self.trap_course_episode = int(body_state.get("trap_episode", self.trap_course_episode) or 0)
            self.trap_course_successes = int(body_state.get("trap_successes", self.trap_course_successes) or 0)
            self.trap_course_failures = int(body_state.get("trap_failures", self.trap_course_failures) or 0)
            self.trap_course_outcome = body_state.get("trap_outcome", self.trap_course_outcome)
        self.apply_control_overrides(body_state)
        self.apply_food_feedback(body_state)
        if isinstance(body_state, dict):
            self.pgnw_experiment_planner.update(
                self.steps,
                self.causal_probe_signal,
                body_state.get("mushroom_pickups_total", 0) or 0,
                body_state.get("red_mushroom_pickups_total", 0) or 0,
                body_state.get("blue_mushroom_pickups_total"),
                body_state.get("yellow_flower_pickups_total"),
            )
            experiment_planner = self.pgnw_experiment_planner
            protective_request = None
            protective_need_active = bool(
                self.causal_probe_world.pending_due_steps
            )
            if (
                experiment_planner.typed_rule_control == "verified_protective"
                and protective_need_active
            ):
                protective_request = (
                    experiment_planner.verified_protective_suppressor()
                )
            self.resource_memory.update(
                body_state,
                self.hunger,
                requested_feature=protective_request,
            )
            experiment_planner.update_guidance(
                body_state,
                resource_memory=self.resource_memory,
                protective_need_active=protective_need_active,
            )
        self.ticks_since_food += 1
        food_visible_now = self.food_visible(body_state)
        self.last_body_obstacle_visible = self.obstacle_visible(body_state)
        if food_visible_now:
            target_focus = clamp(0.35 + 0.65 * self.acetylcholine)
            self.sensory_focus_gain = clamp(0.72 * self.sensory_focus_gain + 0.28 * target_focus)
            self.sensory_focus_events += 1
        else:
            self.sensory_focus_gain = clamp(0.88 * self.sensory_focus_gain)
        blocked = bool(body_state and body_state.get("blocked", False))
        grounded = bool(body_state is None or body_state.get("grounded", True))
        moving = self.last_action not in {"idle", "sleep", "wake"}
        self.current_stuck = self.sample_stuck(body_state)
        self.decay_obstacle_memory()
        horizontal_collision = bool(body_state and body_state.get("horizontal_collision", False))

        body_error = 0.18
        collision_pressure = 0.0
        clear_progress = moving and not blocked and body_state is not None and not self.current_stuck
        self.last_clear_progress = clear_progress
        blocked_contact = blocked or self.current_stuck or (body_state and body_state.get("animation") == "Idle")
        stalled_side_contact = horizontal_collision and not clear_progress
        contact_probe = moving and body_state is not None and (
            blocked_contact
            or stalled_side_contact
        )
        if contact_probe:
            self.contact_probe_ticks += 1
        elif clear_progress:
            self.contact_probe_ticks = 0
        else:
            self.contact_probe_ticks = max(0, self.contact_probe_ticks - 4)
        if moving and self.current_stuck and not self.sleeping and not self.waking:
            self.physics_wedge_ticks += 1
        elif clear_progress:
            self.physics_wedge_ticks = 0
        else:
            self.physics_wedge_ticks = max(0, self.physics_wedge_ticks - 2)
        critical_orbit = self.hunger >= 0.92 and not self.sleeping and not self.waking
        if critical_orbit and moving and self.current_stuck:
            self.trap_accumulation_ticks += 2
        elif critical_orbit and moving and horizontal_collision:
            self.trap_accumulation_ticks += 1
        elif critical_orbit and clear_progress:
            self.trap_accumulation_ticks = max(0, self.trap_accumulation_ticks - 1)
        else:
            self.trap_accumulation_ticks = max(0, self.trap_accumulation_ticks - 4)
        if not self.sleeping:
            self.hunger = clamp(self.hunger + 0.0010 + 0.0008 * self.noise_injection)
        if clear_progress and self.trap_pressure < 0.18:
            self.cluster_escalation = max(0, self.cluster_escalation - 1)
        if blocked and moving:
            body_error += 0.55
            collision_pressure = max(collision_pressure, 0.75)
            self.remember_failed_action(body_state, self.last_action, strength=1.0)
            self.remember_trap_pressure(body_state, 0.34)
        if horizontal_collision and moving and not clear_progress:
            body_error += 0.38
            collision_pressure = max(collision_pressure, 0.62)
            self.remember_failed_action(body_state, self.last_action, strength=0.70)
            self.remember_trap_pressure(body_state, 0.18)
        if not grounded:
            body_error += 0.20
        if moving and body_state and body_state.get("animation") == "Idle":
            body_error += 0.18
            collision_pressure = max(collision_pressure, 0.38)
        if self.current_stuck:
            body_error += 0.35
            collision_pressure = max(collision_pressure, 0.60)
            self.remember_failed_action(body_state, self.last_action, strength=0.80)
            self.remember_trap_pressure(body_state, 0.26)

        self.prediction_error = clamp(0.88 * self.prediction_error + 0.12 * body_error)
        if not self.sleeping:
            self.crosstalk = clamp(self.crosstalk + 0.0025 + 0.010 * self.prediction_error)
            self.complexity = clamp(self.complexity + 0.0015 + 0.006 * self.prediction_error)
            repair_gain = 0.002 + 0.008 * self.acetylcholine
            self.crosstalk = clamp(self.crosstalk * (0.999 - repair_gain))
            self.complexity = clamp(self.complexity * (0.999 - 0.45 * repair_gain))
            if clear_progress:
                self.crosstalk = clamp(self.crosstalk * (0.986 - 0.006 * self.acetylcholine))
                self.complexity = clamp(self.complexity * 0.992)
                self.prediction_error = clamp(self.prediction_error * 0.965)
            altered_gain = 0.45 + 0.90 * self.calcium_gate
            noise_focus_gate = 1.0 - 0.68 * self.sensory_focus_gain
            self.crosstalk = clamp(self.crosstalk + 0.006 * self.noise_injection * altered_gain * noise_focus_gate)
            self.prediction_error = clamp(self.prediction_error + 0.004 * self.noise_injection * altered_gain * noise_focus_gate)
            if self.sensory_focus_gain > 0.05:
                self.crosstalk = clamp(self.crosstalk * (1.0 - 0.020 * self.sensory_focus_gain))
                self.prediction_error = clamp(self.prediction_error * (1.0 - 0.016 * self.sensory_focus_gain))

        trap_pressure = self.local_trap_pressure(body_state)
        progress = 1.0 if clear_progress else 0.0
        metrics = metrics_from_state(
            self.crosstalk,
            self.complexity,
            self.memory,
            self.prediction_error,
            trap_pressure=trap_pressure,
            collision_pressure=collision_pressure,
            progress=progress,
        )
        self.fatigue_report = metrics["fatigue_report"]
        self.delusion_index = metrics["delusion_index"]
        self.update_affect(metrics, body_state)
        self.update_workspace(body_state)
        self.update_survival_monitor(body_state)
        self.update_dynamics_observer(body_state)
        self.update_art_observer(body_state)

    def update_dynamics_observer(self, body_state):
        rays = body_state.get("directional_rays", []) if isinstance(body_state, dict) else []
        obstacle_signal = 0.0
        if isinstance(rays, list) and rays:
            try:
                obstacle_signal = 1.0 - min(clamp(float(value)) for value in rays)
            except (TypeError, ValueError):
                obstacle_signal = 0.0
        values = [
            obstacle_signal,
            1.0 if self.food_visible(body_state) else 0.0,
            clamp(0.5 * (self.valence + 1.0)),
            self.arousal,
            self.local_trap_pressure(body_state),
            self.workspace_packet["confidence"],
        ]
        observed_noise = clamp(
            0.42 * self.noise_injection
            + 0.24 * self.delusion_index
            + 0.22 * self.prediction_error
            + 0.12 * self.calcium_gate
        )
        self.dynamics_observer.update(values, observed_noise)

    def update_art_observer(self, body_state):
        if not isinstance(body_state, dict):
            return

        def normalized_values(key, count, default=1.0):
            values = body_state.get(key, [])
            if not isinstance(values, list) or len(values) != count:
                return [default] * count
            try:
                return [clamp(float(value)) for value in values]
            except (TypeError, ValueError):
                return [default] * count

        rays = normalized_values("directional_rays", 8)
        clearance = normalized_values("directional_body_clearance", 8)
        food_visible = 1.0 if self.food_visible(body_state) else 0.0
        try:
            food_distance = clamp(float(body_state.get("food_distance", 28.0)) / 28.0)
        except (TypeError, ValueError):
            food_distance = 1.0
        blocked = 1.0 if body_state.get("blocked", False) else 0.0
        collision = 1.0 if body_state.get("horizontal_collision", False) else 0.0
        stuck = 1.0 if self.current_stuck else 0.0
        trap_pressure = self.local_trap_pressure(body_state)
        features = rays + clearance + [
            food_visible,
            food_distance,
            blocked,
            collision,
            stuck,
            self.hunger,
            trap_pressure,
            clamp(0.5 * (self.valence + 1.0)),
            self.arousal,
            self.workspace_packet["confidence"],
        ]

        if self.sleeping or self.waking:
            evidence_label = "maintenance"
        elif food_visible > 0.0:
            evidence_label = "food_visible"
        elif self.current_stuck or self.physics_wedge_ticks >= max(2, int(round(self.hz * 1.0))):
            evidence_label = "wedge_orbit"
        elif blocked > 0.0 or collision > 0.0:
            evidence_label = "contact_obstacle"
        elif trap_pressure >= 0.45:
            evidence_label = "local_obstruction"
        elif self.hunger >= 0.92:
            evidence_label = "critical_foraging"
        else:
            evidence_label = "clear_traversal"
        self.art_observer.update(features, evidence_label)

    def update_conductor_observer(self, body_state):
        if not self.conductor_observer.enabled or not self.shadow_ready:
            return
        fallback_active = self.shadow_fallback_hold_ticks > 0
        self.conductor_observer.observe(
            {
                "trap_course": self.trap_course_label,
                "trap_course_variant": self.trap_course_variant,
                "trap_episode": self.trap_course_episode,
                "trap_outcome": self.trap_course_outcome,
                "x": body_state.get("x", 0.0),
                "z": body_state.get("z", 0.0),
                "yaw": body_state.get("yaw", 0.0),
                "food_visible": bool(body_state.get("food_visible", False)),
                "food_distance": body_state.get("food_distance", 0.0),
                "blocked": bool(body_state.get("blocked", False)),
                "body_collision": bool(body_state.get("horizontal_collision", False)),
                "stuck": self.current_stuck,
                "body_safe_actions": self.shadow_body_safe_actions,
                "physics_wedge_seconds": self.physics_wedge_ticks / self.hz,
                "trap_accumulation_seconds": self.trap_accumulation_ticks / self.hz,
                "fallback_active": fallback_active,
                "hidden_goal_active": self.hidden_goal_adapter_active,
                "hidden_goal_route_selected_id": self.hidden_goal_route_selected_id,
                "route_index": self.hidden_goal_route_index,
                "route_distance": self.hidden_goal_route_distance,
                "mpc_engaged": self.shadow_mpc_engaged,
            }
        )

    def update_systemic_conductor(self, body_state):
        if not self.systemic_conductor.enabled or not self.shadow_ready:
            return
        self.systemic_conductor.observe(
            {
                "food_visible": self.food_visible(body_state),
                "blocked": bool(body_state.get("blocked", False)),
                "body_collision": bool(
                    body_state.get("horizontal_collision", False)
                ),
                "stuck": self.current_stuck,
                "physics_wedge_seconds": self.physics_wedge_ticks / self.hz,
                "trap_accumulation_seconds": self.trap_accumulation_ticks / self.hz,
                "fallback_active": self.shadow_fallback_hold_ticks > 0,
                "hidden_goal_active": self.hidden_goal_adapter_active,
                "terrain_air_route_active": self.terrain_air_route_controller.active,
                "terrain_air_route_pending": self.terrain_air_route_controller.pending,
                "terrain_air_route_match": self.terrain_air_route_controller.match,
                "terrain_air_resonance": self.terrain_air_observer.resonance,
                "terrain_air_confidence": self.terrain_air_observer.confidence,
                "orbit_path": self.orbit_path,
                "orbit_efficiency": self.orbit_efficiency,
                "trap_pressure": self.local_trap_pressure(body_state),
                "art_match": self.art_observer.match,
                "art_resonance": self.art_observer.resonance,
                "art_novel": self.art_observer.novel,
                "directional_rays": body_state.get("directional_rays", []),
                "directional_body_clearance": body_state.get(
                    "directional_body_clearance", []
                ),
            }
        )
        self.adaptive_gnw_gate.observe(
            self.systemic_conductor.raw_probabilities,
            self.systemic_conductor.proxy_optimal,
        )

    def apply_food_feedback(self, body_state):
        self.shadow_last_reward = 0.0
        self.causal_probe_signal = clamp(self.causal_probe_signal * 0.985)
        if isinstance(body_state, dict):
            try:
                pickup_total = int(body_state.get("mushroom_pickups_total", body_state.get("ate_mushroom", 0)))
            except (TypeError, ValueError):
                pickup_total = self.last_mushroom_pickup_total
            try:
                reward_total = float(body_state.get("mushroom_reward_total", body_state.get("mushroom_reward", 0.0)))
            except (TypeError, ValueError):
                reward_total = self.last_mushroom_reward_total
            try:
                red_pickup_total = int(
                    body_state.get(
                        "red_mushroom_pickups_total",
                        self.last_red_mushroom_pickup_total,
                    )
                    or 0
                )
            except (TypeError, ValueError):
                red_pickup_total = self.last_red_mushroom_pickup_total
            try:
                blue_pickup_total = int(
                    body_state.get(
                        "blue_mushroom_pickups_total",
                        max(0, pickup_total - red_pickup_total),
                    )
                    or 0
                )
            except (TypeError, ValueError):
                blue_pickup_total = self.last_blue_mushroom_pickup_total
            try:
                yellow_pickup_total = int(
                    body_state.get(
                        "yellow_flower_pickups_total",
                        self.last_yellow_flower_pickup_total,
                    )
                    or 0
                )
            except (TypeError, ValueError):
                yellow_pickup_total = self.last_yellow_flower_pickup_total

            if not self.food_feedback_initialized:
                self.last_mushroom_pickup_total = pickup_total
                self.last_mushroom_reward_total = reward_total
                self.last_red_mushroom_pickup_total = red_pickup_total
                self.last_blue_mushroom_pickup_total = blue_pickup_total
                self.last_yellow_flower_pickup_total = yellow_pickup_total
                self.food_feedback_initialized = True
                pickup_total = self.last_mushroom_pickup_total
                reward_total = self.last_mushroom_reward_total

            if pickup_total < self.last_mushroom_pickup_total or reward_total < self.last_mushroom_reward_total:
                self.last_mushroom_pickup_total = 0
                self.last_mushroom_reward_total = 0.0
                self.last_red_mushroom_pickup_total = 0
                self.last_blue_mushroom_pickup_total = 0
                self.last_yellow_flower_pickup_total = 0
                self.mushrooms_eaten = 0
                self.causal_probe_world.reset()
                self.typed_interaction_learner.reset()
                self.causal_probe_signal = 0.0

            eaten = max(0, pickup_total - self.last_mushroom_pickup_total)
            red_eaten = max(
                0, red_pickup_total - self.last_red_mushroom_pickup_total
            )
            blue_eaten = max(
                0, blue_pickup_total - self.last_blue_mushroom_pickup_total
            )
            yellow_eaten = max(
                0, yellow_pickup_total - self.last_yellow_flower_pickup_total
            )
            reward = max(0.0, reward_total - self.last_mushroom_reward_total)
            if eaten > 0:
                self.last_consumable_feature = str(
                    body_state.get("mushroom_feature", "unknown") or "unknown"
                )
                if reward <= 0.0:
                    reward = 0.35 * eaten
                self.mushrooms_eaten += eaten
                self.hunger = clamp(self.hunger - 0.34 * eaten)
                self.ticks_since_food = 0
                self.critical_hunger_ticks = 0
                self.forage_lapse_ticks = 0
                self.survival_failed = False
                self.dopamine_food_boost = clamp(self.dopamine_food_boost + reward)
                self.shadow_last_reward = reward
            can_start_typed_episode = len(self.causal_probe_due_steps) == 0
            self.typed_interaction_learner.observe_pickups(
                self.steps,
                red=red_eaten,
                blue=blue_eaten,
                yellow=yellow_eaten,
                can_start=can_start_typed_episode,
            )
            self.causal_probe_world.register_pickups(
                self.steps,
                red=red_eaten,
                blue=blue_eaten,
                yellow=yellow_eaten,
            )
            self.last_mushroom_pickup_total = pickup_total
            self.last_mushroom_reward_total = reward_total
            self.last_red_mushroom_pickup_total = red_pickup_total
            self.last_blue_mushroom_pickup_total = blue_pickup_total
            self.last_yellow_flower_pickup_total = yellow_pickup_total

        due = self.causal_probe_world.pop_due(self.steps)
        self.typed_interaction_learner.observe_deadline(
            self.steps, observed_probe_events=due
        )
        self.typed_interaction_learner.poll_formulation()
        if due:
            self.causal_probe_signal = clamp(self.causal_probe_signal + 0.34 * due)
            self.causal_probe_events += due
            applied_cost = self.causal_probe_hunger_cost * due
            if applied_cost > 0.0:
                self.hunger = clamp(self.hunger + applied_cost)
                self.causal_probe_hunger_cost_events += due
                self.causal_probe_hunger_cost_total += applied_cost

        self.dopamine_food_boost *= 0.992
        self.dopamine = clamp(self.dopamine + self.dopamine_food_boost)

    def empty_workspace(self):
        return {
            "intent": "continue_heading",
            "problem": "none",
            "strategy": "local_probe",
            "feeling": "calm_positive_valence",
            "confidence": 0.0,
        }

    def update_tiny_scientist_rule_targeting(self, body_state):
        """Keep verified rules observational; they cannot target or steer."""
        self.tiny_scientist_rule_active = False
        self.tiny_scientist_rule_target_feature = "none"
        self.tiny_scientist_rule_predicted_pressure_risk = 0.0
        self.tiny_scientist_rule_guidance_vector = (0.0, 0.0)
        self.tiny_scientist_rule_red_vector = (0.0, 0.0)
        self.tiny_scientist_rule_red_utility = 0.0
        self.tiny_scientist_rule_blue_utility = 0.0
        self.tiny_scientist_rule_guidance_weight = 0.0

    def apply_control_overrides(self, body_state):
        controls = {}
        if isinstance(body_state, dict):
            raw = body_state.get("controls", {})
            if isinstance(raw, dict):
                controls.update(raw)
            for key in (
                "dopamine",
                "norepinephrine",
                "acetylcholine",
                "delusion_drive",
                "noise_injection",
                "calcium_gate",
                "route_exploration",
            ):
                if key in body_state:
                    controls[key] = body_state[key]

        self.dopamine_baseline = self.control_value(controls, "dopamine", self.dopamine_baseline)
        self.dopamine = self.dopamine_baseline
        self.norepinephrine = self.control_value(controls, "norepinephrine", self.norepinephrine)
        self.acetylcholine = self.control_value(controls, "acetylcholine", self.acetylcholine)
        noise_fallback = controls.get("delusion_drive", self.noise_injection)
        self.noise_injection = self.control_value(controls, "noise_injection", noise_fallback)
        self.calcium_gate = self.control_value(controls, "calcium_gate", self.calcium_gate)
        exploration = self.control_value(controls, "route_exploration", self.base_route_exploration)
        self.route_exploration = clamp(exploration + 0.25 * self.dopamine + 0.20 * self.norepinephrine)

    def food_visible(self, body_state):
        return bool(body_state and body_state.get("food_visible", False))

    def obstacle_visible(self, body_state):
        if not body_state or not body_state.get("obstacle_visible", False):
            return False
        try:
            distance = float(body_state.get("obstacle_distance", 999.0))
        except (TypeError, ValueError):
            return False
        return 0.35 < distance <= 5.8

    def obstacle_avoid_move(self, body_state):
        if not self.obstacle_visible(body_state):
            return None
        try:
            x = float(body_state.get("avoid_move_x", 0.0))
            z = float(body_state.get("avoid_move_z", 0.0))
            distance = float(body_state.get("obstacle_distance", 999.0))
        except (TypeError, ValueError):
            return None
        length = math.hypot(x, z)
        if length < 0.05:
            return None
        strength = clamp((5.8 - distance) / 4.8, 0.08, 0.45)
        return (x / length, z / length, strength)

    def blend_obstacle_avoidance(self, body_state, base_move, strength_scale=1.0):
        if self.food_visible(body_state):
            try:
                food_distance = float(body_state.get("food_distance", 999.0))
            except (TypeError, ValueError):
                food_distance = 999.0
            if food_distance < 7.0 and not bool(body_state.get("blocked", False)):
                return base_move
        avoid = self.obstacle_avoid_move(body_state)
        if avoid is None:
            return base_move
        ax, az, strength = avoid
        strength = clamp(strength * strength_scale, 0.0, 0.50)
        bx, bz = base_move
        x = bx * (1.0 - strength) + ax * strength
        z = bz * (1.0 - strength) + az * strength
        length = math.hypot(x, z)
        if length < 0.05:
            return base_move
        return (x / length, z / length)

    def choose_food_action(self, body_state):
        if not self.food_visible(body_state):
            self.food_lock_ticks = max(0, self.food_lock_ticks - 1)
            return None

        try:
            x = float(body_state.get("food_move_x", body_state.get("food_dir_x", 0.0)))
            z = float(body_state.get("food_move_z", body_state.get("food_dir_z", 0.0)))
            distance = float(body_state.get("food_distance", 999.0))
        except (TypeError, ValueError):
            return None

        if distance > 16.5:
            return None
        length = math.hypot(x, z)
        if length < 0.05:
            return None

        target_x = x / length
        target_z = z / length
        if self.food_lock_ticks > 0:
            old_x, old_z = self.food_seek_move
            blend = 0.16
            smooth_x = old_x * (1.0 - blend) + target_x * blend
            smooth_z = old_z * (1.0 - blend) + target_z * blend
            smooth_length = math.hypot(smooth_x, smooth_z)
            if smooth_length > 0.05:
                target_x = smooth_x / smooth_length
                target_z = smooth_z / smooth_length
        if abs(target_x) < 0.16:
            target_x = 0.0

        self.food_seek_move = (target_x, target_z)
        self.food_lock_ticks = max(self.food_lock_ticks, int(round(self.hz * 2.5)))
        self.food_seek_ticks = max(self.food_seek_ticks, int(round(self.hz * 2.0)))
        self.foraging_commit_ticks = max(self.foraging_commit_ticks, int(round(self.hz * 3.5)))
        return "seek_food"

    def control_value(self, controls, key, fallback):
        try:
            return clamp(float(controls.get(key, fallback)))
        except (TypeError, ValueError):
            return fallback

    def apply_reality_gate(self, packet, body_state):
        if packet["confidence"] <= 0.0:
            return packet

        blocked = bool(body_state and body_state.get("blocked", False))
        food_visible = self.food_visible(body_state)
        gated = dict(packet)

        if packet["problem"] == "local_obstruction_cluster" and not blocked and self.trap_pressure < 0.22:
            gated["confidence"] *= 0.28
            self.reality_gate_brakes += 1
            self.false_trap_reports += 1
        elif packet["problem"] == "food_visible" and not food_visible:
            gated["confidence"] *= 0.28
            self.reality_gate_brakes += 1
            self.false_food_reports += 1

        if self.workspace_unreliable and gated["confidence"] > 0.0:
            gated["confidence"] *= 0.55
            self.meta_monitor_brakes += 1

        if gated["problem"] == "local_obstruction_cluster" and gated["confidence"] < 0.42:
            if self.contact_probe_ticks > 0:
                gated["problem"] = "contact_probe"
                gated["strategy"] = "capsule_slide"
                gated["confidence"] = max(0.22, gated["confidence"])
            else:
                gated["confidence"] = 0.0

        if gated["confidence"] < 0.18:
            empty = self.empty_workspace()
            empty["feeling"] = self.feeling_label()
            return empty
        gated["confidence"] = clamp(gated["confidence"])
        return gated

    def update_workspace_monitor(self, packet):
        high_confidence = packet["confidence"] > 0.72 and packet["problem"] != "none"
        if high_confidence and packet["problem"] == self.previous_workspace_problem:
            self.locked_workspace_steps += 1
        else:
            self.locked_workspace_steps = 0
        self.previous_workspace_problem = packet["problem"]

        if self.last_clear_progress:
            self.no_progress_steps = max(0, self.no_progress_steps - 2)
            self.workspace_unreliable = False
            return

        if high_confidence and self.last_action not in {"idle", "sleep", "wake"}:
            self.no_progress_steps += 1

        if self.locked_workspace_steps >= max(5, int(round(self.hz * 1.0))) and self.no_progress_steps >= max(5, int(round(self.hz * 1.0))):
            self.workspace_unreliable = True

    def update_survival_monitor(self, body_state):
        critical_threshold = 0.92
        starvation_threshold = 0.98
        hungry_threshold = 0.70
        failure_grace_ticks = int(round(self.hz * 300.0))
        no_food_failure_ticks = int(round(self.hz * 480.0))
        food_seen = self.food_visible(body_state)
        foraging = self.last_action == "seek_food" or (self.foraging_commit_ticks > 0 and self.last_action == "up")

        if self.hunger >= starvation_threshold:
            self.critical_hunger_ticks += 1
        elif self.hunger >= critical_threshold:
            self.critical_hunger_ticks = max(0, self.critical_hunger_ticks - 1)
        else:
            self.critical_hunger_ticks = max(0, self.critical_hunger_ticks - 4)

        if self.hunger >= hungry_threshold and food_seen and not foraging and self.trap_pressure < 0.45:
            self.forage_lapse_ticks += 1
        else:
            self.forage_lapse_ticks = max(0, self.forage_lapse_ticks - 2)

        critical_failure = (
            self.awake_ticks >= failure_grace_ticks
            and self.ticks_since_food >= no_food_failure_ticks
            and self.critical_hunger_ticks >= int(round(self.hz * 180.0))
        )
        lapse_warning = self.forage_lapse_ticks >= int(round(self.hz * 45.0))
        failing_now = critical_failure
        if failing_now and not self.survival_failed:
            self.survival_failure_events += 1
            self.survival_failed = True
            self.survival_failure_reason = "sustained_starvation"
        elif self.hunger < 0.55:
            self.survival_failed = False

        if failing_now:
            self.survival_state = "failing"
        elif self.hunger >= critical_threshold:
            self.survival_state = "critical_hunger"
        elif lapse_warning or self.forage_lapse_ticks > int(round(self.hz * 4.0)):
            self.survival_state = "forage_lapse"
        elif self.hunger >= hungry_threshold:
            self.survival_state = "hungry"
        else:
            self.survival_state = "stable"

    def update_affect(self, metrics, body_state):
        trap_pressure = self.local_trap_pressure(body_state)
        self.trap_pressure = trap_pressure
        blocked = bool(body_state and body_state.get("blocked", False))
        progress_signal = 0.18 if self.last_action not in {"idle", "sleep", "wake"} and not blocked else -0.18
        calm = 1.0 - max(self.fatigue_report, self.delusion_index, trap_pressure)
        self.valence = max(-1.0, min(1.0, 0.55 * calm + progress_signal - 0.25 * self.prediction_error))
        self.arousal = clamp(
            0.35 * self.norepinephrine
            + 0.30 * self.fatigue_report
            + 0.20 * trap_pressure
            + 0.15 * self.prediction_error
        )

    def update_workspace(self, body_state):
        blocked = bool(body_state and body_state.get("blocked", False))
        moving = self.last_action not in {"idle", "sleep", "wake"}
        food_visible = self.food_visible(body_state)
        contact_evidence = blocked or self.current_stuck
        contact_probe = contact_evidence and self.contact_probe_ticks >= max(2, int(round(self.hz * 0.8)))
        persistent_contact = contact_evidence and self.contact_probe_ticks >= max(5, int(round(self.hz * 2.5)))
        obstacle_evidence = persistent_contact or blocked or self.current_stuck or self.breakout_plan or self.escape_ticks > 0
        tension = clamp(
            0.38 * self.trap_pressure
            + 0.26 * self.delusion_index
            + 0.20 * self.prediction_error
            + (0.16 if blocked and moving else 0.0)
        )
        sensory_focus = self.sensory_focus_gain if food_visible else 0.0
        false_salience = self.noise_injection * self.calcium_gate * (1.0 - 0.72 * sensory_focus)

        if self.sleeping:
            packet = {
                "intent": "restore_substrate",
                "problem": "fatigue_pressure",
                "strategy": "visible_sleep_repair",
                "feeling": "low_arousal_repair",
                "confidence": 0.92,
            }
        elif obstacle_evidence:
            packet = {
                "intent": "continue_heading",
                "problem": "local_obstruction_cluster",
                "strategy": self.active_escape_strategy(tension),
                "feeling": self.feeling_label(),
                "confidence": clamp(0.35 + tension + 0.10 * min(self.breakout_events, 3) + 0.16 * false_salience),
            }
        elif contact_probe:
            packet = {
                "intent": "continue_heading",
                "problem": "contact_probe",
                "strategy": "capsule_slide",
                "feeling": self.feeling_label(),
                "confidence": clamp(0.38 + 0.04 * self.contact_probe_ticks),
            }
        elif food_visible and not blocked and self.trap_pressure < 0.45:
            packet = {
                "intent": "seek_food",
                "problem": "food_visible",
                "strategy": "approach_mushroom",
                "feeling": self.feeling_label(),
                "confidence": clamp(0.72 + 0.20 * self.hunger + 0.12 * sensory_focus - 0.20 * self.trap_pressure),
            }
        elif false_salience > 0.72 and self.hunger > 0.42 and random.random() < 0.10:
            packet = {
                "intent": "seek_food",
                "problem": "food_visible",
                "strategy": "approach_mushroom",
                "feeling": "high_arousal_neutral_valence",
                "confidence": clamp(0.45 + 0.35 * false_salience),
            }
        elif self.hunger > (0.58 if self.workspace_unreliable else 0.72) and self.trap_pressure < 0.62:
            packet = {
                "intent": "seek_food",
                "problem": "metabolic_need",
                "strategy": "forage_scan",
                "feeling": self.feeling_label(),
                "confidence": clamp(0.30 + 0.58 * self.hunger - 0.18 * self.trap_pressure),
            }
        elif moving and not blocked and self.trap_pressure < 0.22 and self.delusion_index < 0.34:
            packet = {
                "intent": "continue_heading",
                "problem": "path_clear",
                "strategy": "continue_forward",
                "feeling": self.feeling_label(),
                "confidence": clamp(0.75 - self.trap_pressure - 0.25 * self.delusion_index + 0.12 * self.calcium_gate),
            }
        else:
            packet = self.empty_workspace()
            packet["feeling"] = self.feeling_label()

        packet = self.apply_reality_gate(packet, body_state)
        self.update_workspace_monitor(packet)

        if packet != self.workspace_packet and packet["confidence"] > 0.0:
            self.workspace_promotions += 1
        self.workspace_packet = packet

    def feeling_label(self):
        if self.valence < -0.25 and self.arousal > 0.55:
            return "high_arousal_negative_valence"
        if self.valence < -0.10:
            return "negative_valence"
        if self.arousal > 0.55:
            return "high_arousal_neutral_valence"
        if self.valence > 0.25:
            return "low_arousal_positive_valence"
        return "calm_neutral_valence"

    def sample_stuck(self, body_state):
        if body_state is None or self.sleeping or self.waking:
            return False

        x = float(body_state.get("x", 0.0))
        z = float(body_state.get("z", 0.0))
        self.position_history.append((x, z))
        if len(self.position_history) < self.position_history.maxlen:
            return False

        start_x, start_z = self.position_history[0]
        distance = math.hypot(x - start_x, z - start_z)
        path_length = sum(
            math.hypot(current_x - previous_x, current_z - previous_z)
            for (previous_x, previous_z), (current_x, current_z)
            in zip(self.position_history, list(self.position_history)[1:])
        )
        xs = [sample_x for sample_x, _sample_z in self.position_history]
        zs = [sample_z for _sample_x, sample_z in self.position_history]
        pocket_span = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
        trying_to_move = self.last_action not in {"idle", "sleep", "wake"}
        contact_evidence = (
            bool(body_state.get("blocked", False))
            or bool(body_state.get("horizontal_collision", False))
            or body_state.get("animation") == "Idle"
        )
        stalled = distance < 0.80
        orbiting = path_length >= 1.0 and pocket_span < 1.8
        return trying_to_move and contact_evidence and (stalled or orbiting)

    def memory_cell(self, body_state):
        if body_state is None:
            return None
        x = float(body_state.get("x", 0.0))
        z = float(body_state.get("z", 0.0))
        return (
            int(round(x / self.memory_cell_size)),
            int(round(z / self.memory_cell_size)),
        )

    def nearby_cells(self, cell):
        if cell is None:
            return []
        cx, cz = cell
        return [
            (cx + dx, cz + dz)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

    def projected_cell(self, cell, action):
        if cell is None:
            return None
        delta = ACTION_CELL_DELTAS.get(action)
        if delta is None:
            return None
        return (cell[0] + delta[0], cell[1] + delta[1])

    def remember_failed_action(self, body_state, action, strength=1.0):
        bucket = ACTION_BUCKETS.get(action)
        cell = self.memory_cell(body_state)
        if bucket is None or cell is None:
            return

        cells = [cell]
        ahead = self.projected_cell(cell, action)
        if ahead is not None:
            cells.append(ahead)

        for index, failed_cell in enumerate(cells):
            key = (failed_cell, bucket)
            gain = 0.34 if index == 0 else 0.48
            self.obstacle_memory[key] = clamp(self.obstacle_memory.get(key, 0.0) + gain * strength)
        self.obstacle_events += 1

        if bucket == "forward":
            for neighbor in ("forward_left", "forward_right"):
                for index, failed_cell in enumerate(cells):
                    neighbor_key = (failed_cell, neighbor)
                    gain = 0.16 if index == 0 else 0.22
                    self.obstacle_memory[neighbor_key] = clamp(self.obstacle_memory.get(neighbor_key, 0.0) + gain * strength)

        self.recent_failures[(cell, bucket)] = max(
            self.recent_failures.get((cell, bucket), 0),
            max(6, int(round(self.hz * 7.0 * strength))),
        )

    def decay_obstacle_memory(self):
        stale = []
        for key, value in self.obstacle_memory.items():
            faded = value * self.obstacle_memory_decay
            if faded < 0.05:
                stale.append(key)
            else:
                self.obstacle_memory[key] = faded
        for key in stale:
            del self.obstacle_memory[key]

        expired = []
        for key, ticks in self.recent_failures.items():
            ticks -= 1
            if ticks <= 0:
                expired.append(key)
            else:
                self.recent_failures[key] = ticks
        for key in expired:
            del self.recent_failures[key]

        stale_attempts = []
        for key, value in self.escape_attempts.items():
            faded = value * 0.985
            if faded < 0.03:
                stale_attempts.append(key)
            else:
                self.escape_attempts[key] = faded
        for key in stale_attempts:
            del self.escape_attempts[key]

        stale_traps = []
        for cell, value in self.trap_memory.items():
            faded = value * 0.992
            if faded < 0.04:
                stale_traps.append(cell)
            else:
                self.trap_memory[cell] = faded
        for cell in stale_traps:
            del self.trap_memory[cell]

    def remember_trap_pressure(self, body_state, amount):
        cell = self.memory_cell(body_state)
        if cell is None:
            return
        self.trap_memory[cell] = clamp(self.trap_memory.get(cell, 0.0) + amount)
        ahead = self.projected_cell(cell, self.last_action)
        if ahead is not None:
            self.trap_memory[ahead] = clamp(self.trap_memory.get(ahead, 0.0) + 0.5 * amount)

    def local_trap_pressure(self, body_state):
        cell = self.memory_cell(body_state)
        if cell is None:
            return 0.0
        pressure = self.trap_memory.get(cell, 0.0)
        for near_cell in self.nearby_cells(cell):
            if near_cell != cell:
                pressure += 0.22 * self.trap_memory.get(near_cell, 0.0)
        return clamp(pressure)

    def obstacle_penalty(self, body_state, action):
        bucket = ACTION_BUCKETS.get(action)
        cell = self.memory_cell(body_state)
        if bucket is None or cell is None:
            return 0.0

        penalty = 0.0
        ahead = self.projected_cell(cell, action)
        scan_cells = [(cell, 0.85)]
        if ahead is not None:
            scan_cells.append((ahead, 1.0))

        for base_cell, base_weight in scan_cells:
            for near_cell in self.nearby_cells(base_cell):
                weight = base_weight if near_cell == base_cell else 0.35 * base_weight
                penalty += weight * self.obstacle_memory.get((near_cell, bucket), 0.0)

        if (cell, bucket) in self.recent_failures:
            penalty += 0.65
        return clamp(penalty)

    def least_bad_escape(self, body_state, candidates):
        available = [action for action in candidates if action]
        if not available:
            return "down"
        return min(
            available,
            key=lambda action: (
                self.obstacle_penalty(body_state, action),
                0 if action.startswith("up_") else 1,
            ),
        )

    def escape_attempt_penalty(self, body_state, action):
        bucket = ACTION_BUCKETS.get(action)
        cell = self.memory_cell(body_state)
        if bucket is None or cell is None:
            return 0.0
        penalty = self.escape_attempts.get((cell, bucket), 0.0)
        ahead = self.projected_cell(cell, action)
        if ahead is not None:
            penalty += 0.45 * self.escape_attempts.get((ahead, bucket), 0.0)
        return clamp(penalty)

    def mark_escape_attempt(self, body_state, action):
        bucket = ACTION_BUCKETS.get(action)
        cell = self.memory_cell(body_state)
        if bucket is None or cell is None:
            return
        self.escape_attempts[(cell, bucket)] = clamp(self.escape_attempts.get((cell, bucket), 0.0) + 0.45)
        ahead = self.projected_cell(cell, action)
        if ahead is not None:
            self.escape_attempts[(ahead, bucket)] = clamp(self.escape_attempts.get((ahead, bucket), 0.0) + 0.25)

    def choose_escape(self, body_state, candidates):
        available = [action for action in candidates if action]
        if not available:
            return "down"

        scored = []
        for action in available:
            obstacle = self.obstacle_penalty(body_state, action)
            repeated = self.escape_attempt_penalty(body_state, action)
            forward_bonus = -0.08 if action.startswith("up_") else 0.0
            jitter = random.uniform(-0.18, 0.18) * self.route_exploration
            score = max(0.0, obstacle + 0.70 * repeated + forward_bonus + jitter)
            scored.append((action, score))

        temperature = 0.18 + 0.40 * self.route_exploration
        weights = [math.exp(-score / temperature) for _action, score in scored]
        choice = random.choices([action for action, _score in scored], weights=weights, k=1)[0]
        self.mark_escape_attempt(body_state, choice)
        return choice

    def choose_action(self, body_state):
        if self.handoff_cooldown > 0:
            self.handoff_cooldown -= 1

        if self.sleeping:
            self.sleep_remaining -= 1
            # Spread the lab's 50-step repair sweet spot across however long
            # Unity holds the body in its sleeping animation.
            self.sleep_repair_credit += 50.0 / max(self.sleep_total_ticks, 1)
            while self.sleep_repair_credit >= 1.0:
                self.crosstalk, self.complexity, self.memory = dream_repair(
                    self.crosstalk,
                    self.complexity,
                    self.memory,
                    1,
                )
                self.sleep_repair_credit -= 1.0
            self.last_action = "sleep"
            self.last_maintenance = "visible_sleep"
            if self.sleep_remaining <= 0:
                self.wake_remaining = self.wake_total_ticks
            return "sleep"

        if self.waking:
            self.wake_remaining -= 1
            self.last_action = "wake"
            self.last_maintenance = "wake"
            return "wake"

        self.awake_ticks += 1
        urgency = clamp(0.55 * self.fatigue_report + 0.30 * self.delusion_index + 0.15 * self.complexity)
        emergency = urgency > self.emergency_sleep_threshold and self.delusion_index > 0.86
        maintenance_due = self.awake_ticks >= self.min_awake_ticks
        if maintenance_due and self.maintenance_mode == "visible_sleep" and urgency > self.sleep_threshold:
            return self.start_visible_sleep()
        if maintenance_due and self.maintenance_mode in {"handoff", "hybrid"}:
            if emergency and self.maintenance_mode == "hybrid":
                return self.start_visible_sleep()
            if urgency > self.handoff_threshold and self.handoff_cooldown <= 0:
                self.perform_successor_handoff(urgency)

        if body_state is None:
            self.last_action = "idle"
            return "idle"

        persistent_wedge = max(self.physics_wedge_ticks, self.trap_accumulation_ticks)
        if self.unstuck_respawn_ticks > 0 and persistent_wedge >= self.unstuck_respawn_ticks:
            self.survival_failure_events += 1
            self.survival_failure_reason = "persistent_physics_wedge"
            self.unstuck_respawns += 1
            self.physics_wedge_ticks = 0
            self.trap_accumulation_ticks = 0
            self.contact_probe_ticks = 0
            self.stuck_cooldown = max(self.stuck_cooldown, int(round(self.hz * 8.0)))
            self.breakout_plan.clear()
            self.breakout_style = "none"
            self.escape_ticks = 0
            self.escape_action = None
            self.heading_ticks = 0
            self.last_action = "unstuck_respawn"
            return "unstuck_respawn"

        if self.survival_failed:
            self.unstuck_respawns += 1
            self.physics_wedge_ticks = 0
            self.trap_accumulation_ticks = 0
            self.contact_probe_ticks = 0
            self.stuck_cooldown = max(self.stuck_cooldown, int(round(self.hz * 8.0)))
            self.breakout_plan.clear()
            self.breakout_style = "none"
            self.escape_ticks = 0
            self.escape_action = None
            self.heading_ticks = 0
            self.hunger = 0.25
            self.ticks_since_food = 0
            self.critical_hunger_ticks = 0
            self.forage_lapse_ticks = 0
            self.survival_failed = False
            self.survival_state = "stable"
            if self.survival_failure_reason == "none":
                self.survival_failure_reason = "failure_respawn"
            self.last_action = "unstuck_respawn"
            return "unstuck_respawn"

        forward_clear = bool(body_state.get("forward_clear", True))
        left_clear = bool(body_state.get("left_clear", True))
        right_clear = bool(body_state.get("right_clear", True))
        trap_pressure = self.local_trap_pressure(body_state)

        if self.stuck_cooldown > 0:
            self.stuck_cooldown -= 1
        if self.foraging_commit_ticks > 0:
            self.foraging_commit_ticks -= 1

        if self.breakout_plan:
            action = self.breakout_plan.popleft()
            if not self.breakout_plan:
                self.breakout_style = "none"
            if action in {"up_left", "left", "down_left"} and not left_clear:
                action = "up_right" if right_clear else "down"
            elif action in {"up_right", "right", "down_right"} and not right_clear:
                action = "up_left" if left_clear else "down"
            self.last_action = action
            return action

        breakout_threshold = max(0.42, 0.82 - 0.28 * self.norepinephrine)
        stuck_breakout_threshold = max(0.32, 0.56 - 0.20 * self.norepinephrine)
        workspace_breakout = (
            self.workspace_packet["problem"] == "local_obstruction_cluster"
            and self.workspace_packet["strategy"] == "breakout_arc"
            and self.workspace_packet["confidence"] > 0.64
            and not self.workspace_unreliable
        )

        if (trap_pressure > breakout_threshold or workspace_breakout) and self.escape_ticks <= 0:
            self.start_breakout(body_state, left_clear, right_clear)
            action = self.breakout_plan.popleft()
            self.last_action = action
            return action

        food_blocked_breakout = (
            self.food_visible(body_state)
            and self.current_stuck
            and self.contact_probe_ticks >= max(4, int(round(self.hz * 4.0)))
        )
        if food_blocked_breakout and self.escape_ticks <= 0:
            self.start_breakout(body_state, left_clear, right_clear)
            action = self.breakout_plan.popleft()
            self.last_action = action
            return action

        slide_patience_ticks = max(5, int(round(self.hz * 2.5)))
        contact_stuck = self.current_stuck and self.contact_probe_ticks >= slide_patience_ticks
        if (self.current_stuck or contact_stuck) and self.stuck_cooldown <= 0 and self.escape_ticks <= 0:
            self.stuck_events += 1
            self.cluster_escalation += 1
            if trap_pressure > stuck_breakout_threshold:
                self.start_breakout(body_state, left_clear, right_clear)
                action = self.breakout_plan.popleft()
                self.last_action = action
                return action
            self.escape_ticks = max(4, int(round(self.hz * (1.4 if contact_stuck and not self.current_stuck else 3.0))))
            self.stuck_cooldown = max(6, int(round(self.hz * (2.5 if contact_stuck and not self.current_stuck else 5.0))))
            if not forward_clear:
                if left_clear and right_clear:
                    self.escape_action = self.choose_escape(body_state, ["up_left", "up_right", "down_left", "down_right"])
                    self.turn_bias *= -1
                elif left_clear:
                    self.escape_action = self.choose_escape(body_state, ["up_left", "down_left", "down"])
                elif right_clear:
                    self.escape_action = self.choose_escape(body_state, ["up_right", "down_right", "down"])
                else:
                    self.escape_action = "down"
            else:
                self.escape_action = self.choose_escape(body_state, ["up_left", "up_right", "down_left", "down_right"])

        if self.escape_ticks > 0:
            self.escape_ticks -= 1
            if self.escape_ticks > int(round(self.hz * 2.0)):
                action = "down"
            else:
                action = self.escape_action or ("up_left" if self.turn_bias < 0 else "up_right")
            self.last_action = action
            return action

        food_action = None
        hunger_anchor = self.hunger > (0.58 if self.workspace_unreliable else 0.72)
        foraging_committed = self.foraging_commit_ticks > 0
        experiment_planner = self.pgnw_experiment_planner
        isolation_allowed = pgnw_experiment_guidance_allowed(
            experiment_planner.mode,
            experiment_planner.isolation_active,
            self.shadow_fallback_hold_ticks > 0,
            self.current_stuck,
            self.hunger,
            self.terrain_air_route_controller.control_mode == "guided"
            and self.terrain_air_route_controller.active,
            self.resource_memory.control_mode == "guided"
            and self.resource_memory.active
            and not experiment_planner.protective_memory_active,
        )
        if isolation_allowed:
            self.foraging_commit_ticks = 0
            experiment_planner.isolation_food_suppression_frames += 1
        elif trap_pressure < (0.62 if hunger_anchor else 0.35):
            food_action = self.choose_food_action(body_state)
        try:
            food_distance = float(body_state.get("food_distance", 999.0))
        except (TypeError, ValueError):
            food_distance = 999.0
        close_food_pickup = food_action is not None and food_distance < 6.0 and not self.current_stuck and trap_pressure < 0.55

        if close_food_pickup:
            if hunger_anchor:
                self.hunger_anchor_steps += 1
            action = food_action
        elif not forward_clear:
            self.heading_ticks = 0
            self.remember_failed_action(body_state, "up", strength=0.65)
            if left_clear and right_clear:
                action = self.choose_escape(body_state, ["up_left", "up_right", "down_left", "down_right"])
                self.turn_bias *= -1
            elif left_clear:
                action = self.choose_escape(body_state, ["up_left", "down_left", "down"])
            elif right_clear:
                action = self.choose_escape(body_state, ["up_right", "down_right", "down"])
            else:
                action = "down"
        elif food_action is not None:
            if hunger_anchor:
                self.hunger_anchor_steps += 1
            action = food_action
        elif foraging_committed and trap_pressure < 0.35:
            if self.heading_ticks <= 0:
                self.heading_action = self.choose_heading(body_state, left_clear, right_clear)
                self.heading_ticks = random.randint(
                    max(8, int(round(self.hz * 8.0))),
                    max(14, int(round(self.hz * 18.0))),
                )
            action = self.heading_action
            self.heading_ticks -= 1
        else:
            if self.heading_ticks <= 0:
                self.heading_action = self.choose_heading(body_state, left_clear, right_clear)
                self.heading_ticks = random.randint(
                    max(10, int(round(self.hz * 12.0))),
                    max(16, int(round(self.hz * 28.0))),
                )

            action = self.heading_action
            self.heading_ticks -= 1
            if action in {"left", "up_left", "down_left"} and not left_clear:
                action = "up_right" if right_clear else "up"
                self.heading_ticks = 0
            elif action in {"right", "up_right", "down_right"} and not right_clear:
                action = "up_left" if left_clear else "up"
                self.heading_ticks = 0

            if random.random() < 0.0015:
                action = "idle"

        if action not in {"seek_food", "sleep", "wake", "idle"} and self.obstacle_visible(body_state) and action.startswith("up"):
            base_move = MOVE_VECTORS.get(action, MOVE_VECTORS["up"])
            self.avoidance_move = self.blend_obstacle_avoidance(body_state, base_move, strength_scale=0.35)
            self.avoidance_move_ticks = max(self.avoidance_move_ticks, int(round(self.hz * 1.2)))
            if self.avoidance_move != base_move:
                action = "avoid_obstacle"

        self.last_action = action
        return action

    def start_visible_sleep(self):
        self.sleep_total_ticks = max(1, int(round(self.sleep_seconds * self.hz)))
        self.sleep_remaining = self.sleep_total_ticks
        self.sleep_repair_credit = 0.0
        self.awake_ticks = 0
        self.visible_sleep_events += 1
        self.last_action = "sleep"
        self.last_maintenance = "visible_sleep_start"
        return "sleep"

    def perform_successor_handoff(self, urgency):
        repair_strength = 0.80 + 0.18 * clamp((urgency - self.handoff_threshold) / 0.30)
        self.crosstalk, self.complexity, self.memory, self.prediction_error = successor_handoff(
            self.crosstalk,
            self.complexity,
            self.memory,
            self.prediction_error,
            repair_strength=repair_strength,
        )
        metrics = metrics_from_state(
            self.crosstalk,
            self.complexity,
            self.memory,
            self.prediction_error,
            trap_pressure=self.trap_pressure,
            collision_pressure=0.0,
        )
        self.fatigue_report = metrics["fatigue_report"]
        self.delusion_index = metrics["delusion_index"]
        self.handoff_events += 1
        self.generation += 1
        self.handoff_cooldown = self.handoff_cooldown_total_ticks
        self.awake_ticks = 0
        self.last_maintenance = "successor_handoff"

    def start_breakout(self, body_state, left_clear, right_clear):
        self.breakout_events += 1
        self.escape_ticks = 0
        self.stuck_cooldown = max(8, int(round(self.hz * 6.0)))
        cell = self.memory_cell(body_state)
        if cell == self.last_breakout_cell:
            self.cluster_escalation += 1
        else:
            self.cluster_escalation = max(0, self.cluster_escalation - 1)
        self.last_breakout_cell = cell
        prefer_left = self.choose_breakout_side(body_state, left_clear, right_clear) == "left"
        side = "left" if prefer_left else "right"
        diagonal = "up_left" if prefer_left else "up_right"
        reverse_diagonal = "down_left" if prefer_left else "down_right"
        both_sides_clear = left_clear and right_clear
        boxed = not left_clear and not right_clear
        force_long_arc = boxed or self.cluster_escalation >= 3 or self.trap_pressure > 0.72

        style_roll = random.random()
        exploration = clamp(self.route_exploration + 0.18 * self.dopamine)
        if force_long_arc:
            style = "backtrack_arc"
        elif both_sides_clear and style_roll < 0.30 + 0.35 * exploration:
            style = "edge_follow"
        elif style_roll < 0.58 + 0.25 * exploration:
            style = "sweep"
        else:
            style = "step_around"

        self.breakout_style = f"{style}_{side}"
        self.breakout_plan = deque(self.build_breakout_plan(style, side, diagonal, reverse_diagonal, force_long_arc))
        self.heading_ticks = 0
        self.heading_action = diagonal
        self.mark_escape_attempt(body_state, diagonal)
        self.mark_escape_attempt(body_state, reverse_diagonal)

    def build_breakout_plan(self, style, side, diagonal, reverse_diagonal, force_long_arc=False):
        sidestep_ticks = max(4, int(round(self.hz * 1.0)))
        sweep_ticks = max(6, int(round(self.hz * 1.6)))
        arc_ticks = max(8, int(round(self.hz * 2.2)))
        settle_ticks = max(5, int(round(self.hz * 1.2)))
        back_ticks = max(3, int(round(self.hz * 0.8)))
        if force_long_arc:
            back_ticks = max(back_ticks, int(round(self.hz * 1.8)))
            arc_ticks = max(arc_ticks, int(round(self.hz * 3.8)))
            settle_ticks = max(settle_ticks, int(round(self.hz * 2.0)))

        if style == "edge_follow":
            # Human-ish: rotate/slide along the obstacle, then try to continue the old heading.
            return (
                [side] * sidestep_ticks
                + [diagonal] * sweep_ticks
                + [side] * max(2, sidestep_ticks // 2)
                + ["up"] * settle_ticks
            )

        if style == "sweep":
            return (
                [side] * sweep_ticks
                + [diagonal] * arc_ticks
                + ["up"] * settle_ticks
            )

        if style == "step_around":
            return (
                [reverse_diagonal] * max(2, back_ticks // 2)
                + [side] * sidestep_ticks
                + [diagonal] * sweep_ticks
                + ["up"] * settle_ticks
            )

        return (
            ["down"] * back_ticks
            + [reverse_diagonal] * max(3, back_ticks // 2)
            + [side] * arc_ticks
            + [diagonal] * arc_ticks
            + ["up"] * settle_ticks
        )

    def active_escape_strategy(self, tension):
        if self.breakout_plan and self.breakout_style != "none":
            return self.breakout_style
        if tension > 0.62:
            return "breakout_arc"
        return "local_escape"

    def choose_breakout_side(self, body_state, left_clear, right_clear):
        if left_clear and not right_clear:
            return "left"
        if right_clear and not left_clear:
            return "right"
        if not left_clear and not right_clear:
            return "left" if self.turn_bias < 0 else "right"

        left_score = (
            self.obstacle_penalty(body_state, "up_left")
            + 0.50 * self.escape_attempt_penalty(body_state, "up_left")
            + random.uniform(-0.12, 0.12)
        )
        right_score = (
            self.obstacle_penalty(body_state, "up_right")
            + 0.50 * self.escape_attempt_penalty(body_state, "up_right")
            + random.uniform(-0.12, 0.12)
        )
        if abs(left_score - right_score) < 0.10:
            self.turn_bias *= -1
            return "left" if self.turn_bias < 0 else "right"
        return "left" if left_score < right_score else "right"

    def choose_heading(self, body_state, left_clear, right_clear):
        choices = [("up", 0.20), ("down", 0.18)]
        if left_clear:
            choices.append(("up_left", 0.17))
            choices.append(("left", 0.12))
            choices.append(("down_left", 0.13))
        if right_clear:
            choices.append(("up_right", 0.17))
            choices.append(("right", 0.12))
            choices.append(("down_right", 0.13))

        scored_choices = []
        for action, weight in choices:
            penalty = self.obstacle_penalty(body_state, action)
            scored_choices.append((action, max(0.01, weight * (1.0 - 0.88 * penalty))))

        total = sum(weight for _action, weight in scored_choices)
        roll = random.random() * total
        running = 0.0
        for action, weight in scored_choices:
            running += weight
            if roll <= running:
                return action

        return "up"

    def command_payload(self, action):
        if self.shadow_episode_idle_active:
            action = "idle"
        mode = "sleep" if action == "sleep" else "wake"
        if action == "seek_food":
            move_x, move_z = self.food_seek_move
        elif action == "avoid_obstacle":
            move_x, move_z = self.avoidance_move
            self.avoidance_move_ticks = max(0, self.avoidance_move_ticks - 1)
        else:
            move_x, move_z = MOVE_VECTORS.get(action, (0.0, 0.0))
        payload = {
            "action": action,
            "mode": mode,
            "move_x": round(move_x, 4),
            "move_z": round(move_z, 4),
            "fatigue": round(self.fatigue_report, 4),
            "delusion": round(self.delusion_index, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "crosstalk": round(self.crosstalk, 4),
            "complexity": round(self.complexity, 4),
            "prediction_error": round(self.prediction_error, 4),
            "sleep_remaining": self.sleep_remaining,
            "wake_remaining": self.wake_remaining,
            "generation": self.generation,
            "maintenance": self.last_maintenance,
            "handoff_events": self.handoff_events,
            "breakout_events": self.breakout_events,
            "breakout_plan_remaining": len(self.breakout_plan),
            "heading_action": self.heading_action,
            "heading_ticks": self.heading_ticks,
            "escape_ticks": self.escape_ticks,
            "stuck_events": self.stuck_events,
            "physics_wedge_seconds": round(self.physics_wedge_ticks / self.hz, 1),
            "trap_accumulation_seconds": round(self.trap_accumulation_ticks / self.hz, 1),
            "unstuck_respawns": self.unstuck_respawns,
            "trap_cells": len(self.trap_memory),
            "trap_pressure": round(self.trap_pressure, 4),
            "route_exploration": round(self.route_exploration, 4),
            "dopamine": round(self.dopamine, 4),
            "dopamine_food_boost": round(self.dopamine_food_boost, 4),
            "mushrooms_eaten": self.mushrooms_eaten,
            "norepinephrine": round(self.norepinephrine, 4),
            "acetylcholine": round(self.acetylcholine, 4),
            "delusion_drive": round(self.noise_injection, 4),
            "noise_injection": round(self.noise_injection, 4),
            "calcium_gate": round(self.calcium_gate, 4),
            "hunger": round(self.hunger, 4),
            "workspace_unreliable": 1.0 if self.workspace_unreliable else 0.0,
            "reality_gate_brakes": self.reality_gate_brakes,
            "meta_monitor_brakes": self.meta_monitor_brakes,
            "hunger_anchor_steps": self.hunger_anchor_steps,
            "false_food_reports": self.false_food_reports,
            "false_trap_reports": self.false_trap_reports,
            "contact_probe_seconds": round(self.contact_probe_ticks / self.hz, 1),
            "sensory_focus": round(self.sensory_focus_gain, 4),
            "sensory_focus_events": self.sensory_focus_events,
            "obstacle_visible": self.last_body_obstacle_visible,
            "survival_state": self.survival_state,
            "critical_hunger_seconds": round(self.critical_hunger_ticks / self.hz, 1),
            "forage_lapse_seconds": round(self.forage_lapse_ticks / self.hz, 1),
            "survival_failures": self.survival_failure_events,
            "survival_failure_reason": self.survival_failure_reason,
            "run_seconds": round(self.awake_ticks / self.hz, 1),
            "seconds_since_food": round(self.ticks_since_food / self.hz, 1),
            "intent": self.intent_label(action),
            "workspace_intent": self.workspace_packet["intent"],
            "workspace_problem": self.workspace_packet["problem"],
            "workspace_strategy": self.workspace_packet["strategy"],
            "workspace_feeling": self.workspace_packet["feeling"],
            "workspace_confidence": round(self.workspace_packet["confidence"], 4),
            "workspace_promotions": self.workspace_promotions,
            "shadow_enabled": self.shadow_enabled,
            "shadow_ready": self.shadow_ready,
            "shadow_action": self.shadow_action,
            "shadow_confidence": round(self.shadow_confidence, 4),
            "shadow_entropy": round(self.shadow_entropy, 4),
            "shadow_agreement": round(self.shadow_agreement, 4),
            "shadow_error": self.shadow_error,
            "shadow_takeover": self.shadow_takeover,
            "shadow_world_x": round(self.shadow_world_move[0], 4),
            "shadow_world_z": round(self.shadow_world_move[1], 4),
            "shadow_continuous": self.shadow_continuous_intercept,
            "shadow_takeover_steps": self.shadow_takeover_steps,
            "shadow_episode_resets": self.shadow_episode_resets,
            "shadow_body_safe_actions": self.shadow_body_safe_actions,
            "shadow_mpc": self.shadow_mpc,
            "shadow_mpc_engaged": self.shadow_mpc_engaged,
            "shadow_mpc_score": round(self.shadow_mpc_score, 4),
            "shadow_mpc_mode": self.shadow_mpc_mode,
            "shadow_mpc_horizon": self.shadow_mpc_horizon,
            "shadow_mpc_depth": round(self.shadow_mpc_depth, 3),
            "shadow_mpc_uncertainty_stops": self.shadow_mpc_uncertainty_stops,
            "shadow_mpc_planning_frames": self.shadow_mpc_planning_frames,
            "shadow_mpc_critical_frames": self.shadow_mpc_critical_frames,
            "orbit_adapter_enabled": self.orbit_adapter_enabled,
            "orbit_adapter_active": self.orbit_adapter_active,
            "orbit_adapter_action": self.orbit_adapter_action,
            "orbit_adapter_confidence": round(self.orbit_adapter_confidence, 4),
            "orbit_adapter_confidence_gate": round(self.orbit_adapter_confidence_gate, 4),
            "orbit_adapter_events": self.orbit_adapter_events,
            "orbit_adapter_remaining_seconds": round(self.orbit_adapter_hold_ticks / self.hz, 2),
            "orbit_adapter_displacement": round(self.orbit_adapter_displacement, 3),
            "orbit_path": round(self.orbit_path, 3),
            "orbit_net": round(self.orbit_net, 3),
            "orbit_efficiency": round(self.orbit_efficiency, 4),
            "escape_teacher_active": self.escape_teacher.active,
            "escape_teacher_action": self.escape_teacher.action_name,
            "escape_teacher_elapsed_seconds": round(
                self.escape_teacher.elapsed_ticks / self.hz, 2
            ),
            "escape_teacher_displacement": round(
                self.escape_teacher.displacement, 3
            ),
            "escape_teacher_efficiency": round(
                self.escape_teacher.efficiency, 4
            ),
            "escape_teacher_events": self.escape_teacher.events,
            "escape_teacher_successes": self.escape_teacher.successes,
            "escape_teacher_failures": self.escape_teacher.failures,
            "escape_teacher_last_outcome": self.escape_teacher.last_outcome,
            "escape_teacher_memory_routes": len(
                self.terrain_air_route_controller.teacher_route_payloads
            ),
            "hidden_goal_adapter_enabled": self.hidden_goal_adapter_enabled,
            "hidden_goal_adapter_type": self.hidden_goal_adapter_type,
            "hidden_goal_adapter_active": self.hidden_goal_adapter_active,
            "hidden_goal_adapter_action": self.hidden_goal_adapter_action,
            "hidden_goal_adapter_confidence": round(self.hidden_goal_adapter_confidence, 4),
            "hidden_goal_adapter_events": self.hidden_goal_adapter_events,
            "hidden_goal_route_index": self.hidden_goal_route_index,
            "hidden_goal_route_count": len(self.hidden_goal_route),
            "hidden_goal_route_distance": round(self.hidden_goal_route_distance, 3),
            "hidden_goal_route_vetoes": self.hidden_goal_route_vetoes,
            "hidden_goal_route_terminal_holds": self.hidden_goal_route_terminal_holds,
            "hidden_goal_route_terminal_active": self.hidden_goal_route_terminal_active,
            "hidden_goal_route_selected_id": self.hidden_goal_route_selected_id,
            "hidden_goal_route_art_match": round(self.hidden_goal_route_art_match, 4),
            "hidden_goal_food_latched": self.hidden_goal_food_latch_ticks > 0,
            "hidden_goal_food_latch_seconds": round(
                self.hidden_goal_food_latch_ticks / self.hz, 2
            ),
            "hidden_goal_adapter_lateral_sign": self.hidden_goal_adapter_lateral_sign,
            "hidden_goal_adapter_lateral_resets": self.hidden_goal_adapter_lateral_resets,
            "food_sensor_radius": round(self.shadow_food_sensor_radius, 2),
            "trap_course": self.trap_course_label,
            "trap_course_variant": self.trap_course_variant,
            "trap_episode": self.trap_course_episode,
            "trap_successes": self.trap_course_successes,
            "trap_failures": self.trap_course_failures,
            "trap_outcome": self.trap_course_outcome,
            "conductor_observer_mode": (
                (
                    "live_bounded_familiar_hidden_goal"
                    if self.conductor_control == "familiar_hidden_goal"
                    else "passive_reward_learning"
                    if self.conductor_observer.learning_enabled
                    else "passive_frozen_checkpoint"
                )
                if self.conductor_observer.enabled
                else "disabled"
            ),
            "conductor_context": self.conductor_observer.context,
            "conductor_recommendation": self.conductor_observer.recommendation,
            "conductor_active_specialist": self.conductor_observer.active_specialist,
            "conductor_confidence": round(self.conductor_observer.confidence, 4),
            "conductor_agreement": self.conductor_observer.agreement,
            "conductor_agreement_rate": round(
                self.conductor_observer.agreement_rate, 4
            ),
            "conductor_grounded_reward": round(
                self.conductor_observer.last_reward, 4
            ),
            "conductor_updates": self.conductor_observer.updates,
            "conductor_context_visits": self.conductor_observer.context_visits,
            "conductor_protocol_entries": len(self.conductor_observer.protocol),
            "conductor_episode_resets": self.conductor_observer.episode_resets,
            "conductor_action_influence": self.conductor_observer.action_influence,
            "conductor_control_mode": self.conductor_control,
            "conductor_gate_active": self.conductor_gate_active,
            "conductor_gate_recommendation": self.conductor_gate_recommendation,
            "conductor_gate_confidence": round(self.conductor_gate_confidence, 4),
            "conductor_gate_frames": self.conductor_gate_frames,
            "conductor_gate_decisions": self.conductor_gate_decisions,
            "conductor_gate_denials": self.conductor_gate_denials,
            "systemic_conductor_mode": (
                (
                    "bounded_recurrent_mpc"
                    if self.systemic_router.mode != "passive"
                    else "passive_frozen_checkpoint"
                )
                if self.systemic_conductor.enabled
                else "disabled"
            ),
            "systemic_conductor_recommendation": (
                self.systemic_conductor.recommendation
            ),
            "systemic_conductor_confidence": round(
                self.systemic_conductor.confidence, 4
            ),
            "systemic_conductor_entropy_bits": round(
                self.systemic_conductor.entropy, 4
            ),
            "systemic_conductor_proxy_context": (
                self.systemic_conductor.proxy_context
            ),
            "systemic_conductor_proxy_optimal": (
                self.systemic_conductor.proxy_optimal
            ),
            "systemic_conductor_agreement": self.systemic_conductor.agreement,
            "systemic_conductor_agreement_rate": round(
                self.systemic_conductor.agreement_rate, 4
            ),
            "systemic_conductor_observations": (
                self.systemic_conductor.observations
            ),
            "systemic_conductor_action_influence": (
                self.systemic_conductor.action_influence
            ),
            "systemic_router_active_specialist": (
                self.systemic_router.active_specialist
            ),
            "systemic_router_reason": self.systemic_router.last_reason,
            "systemic_router_handoffs": self.systemic_router.handoffs,
            "systemic_router_safety_overrides": (
                self.systemic_router.safety_overrides
            ),
            "systemic_router_influence_frames": (
                self.systemic_router.influence_frames
            ),
            "systemic_router_chatter_events": (
                self.systemic_router.chatter_events
            ),
            "systemic_router_mpc_latch_seconds": round(
                self.systemic_router.mandatory_latch_ticks / self.hz, 2
            ),
            "systemic_router_recurrent_release_streak": (
                self.systemic_router.recurrent_release_streak
            ),
            "resource_memory_mode": (
                f"{self.resource_memory.control_mode}_coordinate_recall"
                if self.resource_memory.enabled
                else "disabled"
            ),
            "resource_memory_active": self.resource_memory.active,
            "resource_memory_recommendation": self.resource_memory.recommendation,
            "resource_memory_distance": round(self.resource_memory.distance, 3),
            "resource_memory_confidence": round(
                self.resource_memory.confidence, 4
            ),
            "resource_memory_regions": len(self.resource_memory.entries),
            "resource_memory_typed_regions": len(
                self.resource_memory.typed_entries
            ),
            "resource_memory_active_feature": (
                self.resource_memory.active_feature
            ),
            "resource_memory_typed_encodings": (
                self.resource_memory.typed_encodings
            ),
            "resource_memory_typed_recommendations": (
                self.resource_memory.typed_recommendations
            ),
            "resource_memory_action_influence": (
                self.resource_memory.action_influence
            ),
            "pgnw_experiment_mode": self.pgnw_experiment_planner.mode,
            "pgnw_experiment_request": (
                self.pgnw_experiment_planner.requested_experiment
            ),
            "pgnw_experiment_phase": self.pgnw_experiment_planner.phase,
            "pgnw_experiment_map_hypothesis": (
                self.pgnw_experiment_planner.map_hypothesis
            ),
            "pgnw_experiment_map_confidence": round(
                self.pgnw_experiment_planner.map_confidence, 4
            ),
            "pgnw_experiment_last_outcome": (
                self.pgnw_experiment_planner.last_outcome
            ),
            "pgnw_experiment_completed": (
                self.pgnw_experiment_planner.experiments_completed
            ),
            "pgnw_experiment_discarded": (
                self.pgnw_experiment_planner.experiments_discarded
            ),
            "pgnw_experiment_guidance_weight": (
                self.pgnw_experiment_planner.last_effective_guidance_weight
            ),
            "pgnw_experiment_action_influence": (
                self.pgnw_experiment_planner.action_influence
            ),
            "sync_observer_mode": "passive",
            "sync_coherence": round(self.dynamics_observer.coherence, 4),
            "sync_active_modules": self.dynamics_observer.active_modules,
            "sync_bus_pressure": round(self.dynamics_observer.bus_pressure, 4),
            "sync_binding_ready": self.dynamics_observer.binding_ready,
            "criticality_observer_mode": "passive_proxy",
            "criticality_propagation_ratio": round(self.dynamics_observer.propagation_ratio, 4),
            "criticality_score": round(self.dynamics_observer.criticality_score, 4),
            "criticality_regime": self.dynamics_observer.criticality_regime,
            "criticality_recommended_gain": round(self.dynamics_observer.recommended_gain, 4),
            "art_observer_mode": "passive_fuzzy_art",
            "art_category": self.art_observer.category,
            "art_category_label": self.art_observer.category_label,
            "art_evidence_label": self.art_observer.evidence_label,
            "art_match": round(self.art_observer.match, 4),
            "art_resonance": self.art_observer.resonance,
            "art_novel": self.art_observer.novel,
            "art_unknown": self.art_observer.unknown,
            "art_mismatch_resets": self.art_observer.mismatch_resets,
            "art_mismatch_resets_total": self.art_observer.mismatch_resets_total,
            "art_category_count": len(self.art_observer.templates),
            "art_category_switches": self.art_observer.category_switches,
        }
        # Recovery commands must reach Unity even when learned control was active
        # on the sensor frame that triggered them.
        if self.shadow_takeover and action != "unstuck_respawn":
            if self.shadow_control == "course":
                payload["action"] = "gru_course"
                payload["intent"] = "learned_course_control"
            elif self.shadow_control == "terrain":
                payload["action"] = "gru_terrain"
                payload["intent"] = "learned_terrain_control"
            else:
                payload["action"] = "gru_food"
                payload["intent"] = "learned_food_pursuit"
        return payload

    def intent_label(self, action):
        if self.sleeping:
            return "visible_sleep_repair"
        if self.waking:
            return "waking"
        if self.breakout_plan:
            return "breakout_arc"
        if action == "seek_food" or (self.foraging_commit_ticks > 0 and action == "up"):
            return "seek_food"
        if action == "avoid_obstacle":
            return "anticipatory_avoidance"
        if action == "unstuck_respawn":
            return "physics_wedge_recovery"
        if self.escape_ticks > 0:
            return "local_escape"
        if action == "idle":
            return "observe"
        if action.startswith("up"):
            return "continue_heading"
        if action.startswith("down"):
            return "retreat"
        return "route_adjust"

    def status_line(self, body_state):
        trap_pressure = self.local_trap_pressure(body_state)
        body = "no_body" if body_state is None else (
            f"{body_state.get('animation')} pos=({body_state.get('x', 0):.1f},"
            f"{body_state.get('z', 0):.1f}) blocked={body_state.get('blocked')}"
        )
        return (
            f"step={self.steps:04d} action={self.last_action:>5} "
            f"valence={self.valence:+.2f} arousal={self.arousal:.2f} "
            f"fatigue={self.fatigue_report:.2f} delusion={self.delusion_index:.2f} "
            f"DA={self.dopamine:.2f} NE={self.norepinephrine:.2f} ACh={self.acetylcholine:.2f} "
            f"noise={self.noise_injection:.2f} Ca={self.calcium_gate:.2f} hunger={self.hunger:.2f} "
            f"survival={self.survival_state} "
            f"unreliable={int(self.workspace_unreliable)} "
            f"gen={self.generation:02d} handoffs={self.handoff_events:02d} "
            f"sleep={self.sleep_remaining:03d} wake={self.wake_remaining:02d} "
            f"maint={self.last_maintenance} "
            f"ws={self.workspace_packet['problem']}:{self.workspace_packet['strategy']}:{self.workspace_packet['confidence']:.2f} "
            f"heading={self.heading_action}:{self.heading_ticks:02d} "
            f"escape={self.escape_ticks:02d} stucks={self.stuck_events:02d} "
            f"wedge={self.physics_wedge_ticks / self.hz:.1f}s "
            f"trap_accum={self.trap_accumulation_ticks / self.hz:.1f}s respawns={self.unstuck_respawns:02d} "
            f"obstacles={len(self.obstacle_memory):02d}/{self.obstacle_events:02d} "
            f"routes={len(self.escape_attempts):02d} traps={len(self.trap_memory):02d}:{trap_pressure:.2f} "
            f"breakouts={self.breakout_events:02d} "
            f"shadow={self.shadow_action}:{self.shadow_confidence:.2f}/{self.shadow_agreement:.2f} "
            f"takeover={int(self.shadow_takeover)} "
            f"conductor={self.conductor_observer.recommendation}:{self.conductor_observer.confidence:.2f}/"
            f"{self.conductor_observer.active_specialist} "
            f"systemic={self.systemic_conductor.recommendation}:"
            f"{self.systemic_conductor.confidence:.2f}/"
            f"{self.systemic_conductor.proxy_context}->"
            f"{self.systemic_router.active_specialist} "
            f"art={self.art_observer.category}:{self.art_observer.category_label}:{self.art_observer.match:.2f} "
            f"air={self.terrain_air_observer.recalled_action}:"
            f"{self.terrain_air_observer.confidence:.2f}/"
            f"{self.terrain_air_observer.agreement_rate:.2f} "
            f"science={self.pgnw_experiment_planner.requested_experiment}:"
            f"{self.pgnw_experiment_planner.phase}:"
            f"{self.pgnw_experiment_planner.map_confidence:.2f} "
            f"recent={len(self.recent_failures):02d} {body}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the first Python-to-Unity embodied functional ego loop.")
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--unity-port", type=int, default=5055)
    parser.add_argument("--listen-port", type=int, default=5056)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until Ctrl-C.")
    parser.add_argument("--seed", type=int, default=0, help="Seed Python and Torch randomness for repeatable controller trials.")
    parser.add_argument(
        "--course-episodes",
        type=int,
        default=0,
        help="Stop after this many course episodes finish. 0 disables automatic stopping.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=60.0, help="How long Unity should keep the body asleep.")
    parser.add_argument("--wake-seconds", type=float, default=3.0, help="How long to send explicit wake commands before walking again.")
    parser.add_argument("--min-awake-seconds", type=float, default=300.0, help="Minimum awake time before autonomous sleep can start.")
    parser.add_argument("--sleep-threshold", type=float, default=0.82, help="Fatigue urgency threshold for autonomous sleep.")
    parser.add_argument(
        "--maintenance-mode",
        choices=["handoff", "hybrid", "visible_sleep"],
        default="handoff",
        help="handoff keeps the body online, hybrid sleeps only in emergencies, visible_sleep preserves the original sit-down behavior.",
    )
    parser.add_argument("--handoff-threshold", type=float, default=0.62, help="Fatigue urgency threshold for invisible successor handoff.")
    parser.add_argument("--emergency-sleep-threshold", type=float, default=0.92, help="Urgency threshold for visible emergency sleep in hybrid mode.")
    parser.add_argument("--handoff-cooldown-seconds", type=float, default=180.0, help="Minimum time between invisible successor handoffs.")
    parser.add_argument("--memory-cell-size", type=float, default=2.0, help="World-space size for coarse obstacle memory cells.")
    parser.add_argument("--obstacle-memory-decay", type=float, default=0.996, help="Per-tick decay for remembered bad route directions.")
    parser.add_argument("--route-exploration", type=float, default=0.35, help="How strongly the body tries alternate escape routes after repeated local failures.")
    parser.add_argument("--dopamine", type=float, default=0.35, help="Baseline exploration/novelty dose shown in the HUD.")
    parser.add_argument("--norepinephrine", type=float, default=0.35, help="Baseline arousal/urgency dose; higher values trigger breakouts sooner.")
    parser.add_argument("--acetylcholine", type=float, default=0.35, help="Baseline cleanup/attention dose; higher values improve waking repair.")
    parser.add_argument("--noise-injection", type=float, default=0.0, help="Injected perception/noise pressure for testing unstable behavior.")
    parser.add_argument("--calcium-gate", type=float, default=0.45, help="Excitability/promotion gate; high values amplify weak signals and false salience.")
    parser.add_argument(
        "--initial-hunger",
        type=float,
        default=None,
        help="Diagnostic initial hunger override in the normalized 0..1 range.",
    )
    parser.add_argument(
        "--initial-causal-probe-signal",
        type=float,
        default=None,
        help=(
            "Diagnostic initial value for the passive delayed causal signal. "
            "It has zero motor, routing, workspace, or neuromodulatory influence."
        ),
    )
    parser.add_argument(
        "--causal-probe-delay-seconds",
        type=float,
        default=10.0,
        help=(
            "Passive red-pickup probe delay. The typed yellow protocol uses "
            "30 seconds to permit a physical second intervention."
        ),
    )
    parser.add_argument(
        "--causal-probe-cancellation-feature",
        choices=["none", "blue", "yellow"],
        default="none",
        help=(
            "Hidden environment transition that cancels one pending red event; "
            "never exposed to the hypothesis pool or controller."
        ),
    )
    parser.add_argument(
        "--causal-probe-hunger-cost",
        type=float,
        default=0.0,
        help=(
            "Bounded hunger increase when an uncancelled delayed probe becomes "
            "due. Zero preserves the passive-probe protocol."
        ),
    )
    parser.add_argument(
        "--typed-interaction-learning",
        action="store_true",
        help=(
            "Passively admit uncontaminated red-led ordered episodes and update "
            "the symmetric candidate pool only at the observation deadline."
        ),
    )
    parser.add_argument(
        "--typed-interaction-memory",
        default=None,
        help=(
            "Optional JSON memory for carrying accepted typed-interaction "
            "posteriors across Unity process boundaries."
        ),
    )
    parser.add_argument(
        "--typed-interaction-discovery-memory",
        default=None,
        help=(
            "Optional read-only discovery checkpoint. It is loaded initially, "
            "while new episodes are written only to --typed-interaction-memory."
        ),
    )
    parser.add_argument(
        "--typed-interaction-sealed-discovery",
        action="store_true",
        help=(
            "Require the preregistered three-observation discovery profile, "
            "a distinct fresh writable memory, and a fresh explicit shadow log."
        ),
    )
    parser.add_argument(
        "--typed-interaction-formulation",
        action="store_true",
        help="Run calibrated Gemma ordered-role formulation asynchronously.",
    )
    parser.add_argument(
        "--typed-interaction-formulation-model",
        default="google/gemma-3-1b-it",
    )
    parser.add_argument(
        "--typed-interaction-rule-control",
        choices=["passive", "verified_protective"],
        default="passive",
        help=(
            "Keep learned typed rules passive or let a >=0.95 specific "
            "after-red suppressor request bounded committed PGNW guidance."
        ),
    )
    parser.add_argument(
        "--initial-metabolic-pressure",
        type=float,
        default=None,
        help=(
            "Deprecated alias for --initial-causal-probe-signal."
        ),
    )
    parser.add_argument(
        "--diagnostic-teleport",
        nargs=2,
        type=float,
        metavar=("X", "Z"),
        default=None,
        help="Teleport once to a terrain X/Z coordinate before normal control begins.",
    )
    parser.add_argument("--unstuck-respawn-seconds", type=float, default=45.0, help="Respawn after this many seconds in a persistent physics wedge; use 0 to disable.")
    parser.add_argument("--delusion-drive", type=float, default=None, help="Deprecated alias for --noise-injection.")
    parser.add_argument(
        "--shadow-policy",
        nargs="?",
        const="checkpoints/upgraded_foraging/best.pt",
        default=None,
        help="Run the learned recurrent policy passively; optionally provide a checkpoint path.",
    )
    parser.add_argument(
        "--shadow-log",
        default=None,
        help="JSONL telemetry path. With --shadow-policy, the default is a timestamped file under outputs/unity_shadow.",
    )
    parser.add_argument("--no-shadow-log", action="store_true", help="Disable automatic shadow telemetry recording.")
    parser.add_argument(
        "--shadow-control",
        choices=["passive", "food", "course", "terrain"],
        default="passive",
        help="Allow the learned policy to control only the selected validated context.",
    )
    parser.add_argument(
        "--shadow-control-confidence",
        type=float,
        default=0.55,
        help="Minimum learned-policy confidence for bounded takeover.",
    )
    parser.add_argument(
        "--shadow-mpc",
        action="store_true",
        help="Use four-step policy-weighted MPC for learned-policy action selection.",
    )
    parser.add_argument(
        "--orbit-exit-adapter",
        nargs="?",
        const="checkpoints/orbit_exit_adapter/best.pt",
        default=None,
        help=(
            "Trigger the learned exit readout on sustained trajectory orbits and "
            "latch its safe recovery plan until grounded escape progress or timeout."
        ),
    )
    parser.add_argument(
        "--orbit-adapter-confidence",
        type=float,
        default=0.70,
        help="Minimum learned exit confidence required to override the base controller.",
    )
    parser.add_argument(
        "--hidden-goal-adapter",
        nargs="?",
        const="checkpoints/lwall_hidden_goal_adapter/best.pt",
        default=None,
        help="Use the frozen-GRU L-wall hidden-goal readout while course food is unseen.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-confidence",
        type=float,
        default=0.30,
        help="Minimum L-wall hidden-goal readout confidence required for bounded takeover.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-commit-seconds",
        type=float,
        default=0.4,
        help="Hold a safe L-wall hidden-goal action before reconsidering it.",
    )
    parser.add_argument(
        "--hidden-goal-adapter-lateral-bias",
        type=float,
        default=0.0,
        help="Soft logit penalty for reversing the latched L-wall bypass side.",
    )
    parser.add_argument(
        "--passive-conductor",
        action="store_true",
        help=(
            "Learn and report controller-routing utility without changing actions. "
            "The observer is causally disconnected from motor selection."
        ),
    )
    parser.add_argument(
        "--conductor-checkpoint",
        default=None,
        help="Optional offline passive-conductor JSON checkpoint.",
    )
    parser.add_argument(
        "--conductor-control",
        choices=["passive", "familiar_hidden_goal"],
        default="passive",
        help=(
            "Allow a frozen conductor checkpoint to gate only its validated context. "
            "Other controller contexts remain unchanged."
        ),
    )
    parser.add_argument(
        "--systemic-conductor-checkpoint",
        default=None,
        help=(
            "Load the frozen four-context Bunge-style systemic conductor."
        ),
    )
    parser.add_argument(
        "--systemic-conductor-control",
        choices=["passive", "recurrent_mpc", "recurrent_mpc_air"],
        default="passive",
        help=(
            "Keep recommendations passive or allow bounded recurrent/MPC "
            "executive timing. The conductor never emits motor actions."
        ),
    )
    parser.add_argument(
        "--systemic-conductor-confidence",
        type=float,
        default=0.55,
        help="Minimum recommendation margin for bounded executive routing.",
    )
    parser.add_argument(
        "--adaptive-gnw-passive",
        action="store_true",
        help=(
            "Observe the reward-calibrated adaptive GNW ignition gate without "
            "allowing it to influence routing or motor control."
        ),
    )
    parser.add_argument(
        "--adaptive-gnw-control",
        choices=["disabled", "passive", "bounded"],
        default="disabled",
        help=(
            "Disable adaptive GNW, observe it passively, or let its broadcast "
            "feed the bounded systemic router. Motor actions remain owned by "
            "the recurrent/MPC specialists and stable fallback."
        ),
    )
    parser.add_argument(
        "--terrain-air-memory",
        default=None,
        help=(
            "Load a frozen AIR-selected terrain memory for passive retrieval "
            "telemetry. It cannot influence motor control."
        ),
    )
    parser.add_argument(
        "--terrain-air-route-library",
        default=None,
        help=(
            "Load an AIR-selected ART terrain-route library for gated passive "
            "recommendation or bounded intervention."
        ),
    )
    parser.add_argument(
        "--terrain-air-route-control",
        choices=["passive", "bounded", "guided"],
        default="passive",
        help=(
            "Keep AIR terrain routes observational, permit historical direct "
            "replay, or use a soft AIR heading prior inside grounded MPC."
        ),
    )
    parser.add_argument(
        "--terrain-air-route-seconds",
        type=float,
        default=4.0,
        help="Maximum duration of each bounded AIR route intervention.",
    )
    parser.add_argument(
        "--terrain-air-max-guidance-weight",
        type=float,
        default=0.05,
        help="Maximum AIR heading bonus added to otherwise grounded MPC scores.",
    )
    parser.add_argument(
        "--terrain-air-guidance-margin",
        type=float,
        default=0.05,
        help=(
            "MPC top-two score margin where AIR influence reaches zero; "
            "smaller margins receive a bounded tie-breaking prior."
        ),
    )
    parser.add_argument(
        "--terrain-air-teacher-memory",
        default=None,
        help=(
            "Persist quality-gated deterministic escape demonstrations in a "
            "separate ART route memory. Omit to disable the escape teacher."
        ),
    )
    parser.add_argument(
        "--terrain-resource-memory",
        default=None,
        help=(
            "Persist rewarded Unity resource regions and emit hunger-gated "
            "passive recommendations. Recommendations cannot affect actions."
        ),
    )
    parser.add_argument(
        "--terrain-resource-hunger-gate",
        type=float,
        default=0.70,
        help="Minimum hunger required for passive resource-memory retrieval.",
    )
    parser.add_argument(
        "--terrain-resource-control",
        choices=["passive", "guided"],
        default="passive",
        help=(
            "Keep resource recall passive or add a small uncertainty-gated "
            "heading prior inside collision-masked MPC."
        ),
    )
    parser.add_argument(
        "--terrain-resource-max-guidance-weight",
        type=float,
        default=0.03,
        help="Maximum resource-memory bonus applied to grounded MPC scores.",
    )
    parser.add_argument(
        "--tiny-scientist-production-memory",
        default=None,
        help=(
            "Load verified Tiny Scientist rules for passive audit telemetry only. "
            "This option has zero motor, routing, and MPC influence."
        ),
    )
    parser.add_argument(
        "--tiny-scientist-rule-control",
        choices=["passive"],
        default="passive",
        help=(
            "Keep committed Tiny Scientist rules read-only with zero motor, "
            "routing, workspace, and MPC influence."
        ),
    )
    parser.add_argument(
        "--tiny-scientist-experiment-control",
        choices=[
            "disabled", "passive", "bounded", "committed", "dynamic_committed"
        ],
        default="disabled",
        help=(
            "Let PGNW select red, blue, or no-pickup observations. Passive "
            "logs requests only; bounded adds a capped uncertainty-gated MPC "
            "preference; committed chooses target alignment only within a "
            "safety-filtered MPC regret bound; dynamic_committed first admits "
            "a Gemma-proposed L1 hypothesis from discovery evidence."
        ),
    )
    parser.add_argument(
        "--tiny-scientist-dynamic-model",
        default="google/gemma-3-1b-it",
        help="Local base model used only by dynamic_committed proposal generation.",
    )
    parser.add_argument(
        "--tiny-scientist-dynamic-adapter",
        default=None,
        help="L1 LoRA adapter required by dynamic_committed proposal generation.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-guidance-weight",
        type=float,
        default=0.03,
        help="Maximum PGNW experiment preference added to ambiguous MPC scores.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-guidance-margin",
        type=float,
        default=0.08,
        help="MPC score-margin range in which PGNW experiment guidance may act.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-commit-seconds",
        type=float,
        default=3.0,
        help="Maximum duration of each constrained PGNW target commitment.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-commit-cooldown-seconds",
        type=float,
        default=2.0,
        help="Cooldown before PGNW may recommit to a still-visible target.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-max-score-regret",
        type=float,
        default=0.18,
        help="Largest MPC-score loss admitted for target-aligned commitment.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-isolation-retreat-seconds",
        type=float,
        default=2.0,
        help="Brief post-pickup interval allowed stronger safe food repulsion.",
    )
    parser.add_argument(
        "--tiny-scientist-experiment-isolation-retreat-max-score-regret",
        type=float,
        default=0.35,
        help="Collision-safe MPC regret bound during immediate post-pickup retreat.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    if args.shadow_policy:
        try:
            import torch

            torch.manual_seed(args.seed)
        except ImportError:
            pass

    if (
        args.tiny_scientist_experiment_control == "dynamic_committed"
        and not args.tiny_scientist_dynamic_adapter
    ):
        parser.error("dynamic_committed requires --tiny-scientist-dynamic-adapter")
    if args.typed_interaction_sealed_discovery:
        from typed_causal_domain import validate_sealed_discovery_memory

        if not args.typed_interaction_formulation or not args.typed_interaction_learning:
            parser.error("sealed discovery requires learning and formulation")
        if not args.typed_interaction_discovery_memory or not args.typed_interaction_memory:
            parser.error("sealed discovery requires separate discovery and run memories")
        discovery_path = Path(args.typed_interaction_discovery_memory).resolve()
        run_memory_path = Path(args.typed_interaction_memory).resolve()
        if discovery_path == run_memory_path:
            parser.error("sealed discovery input and writable memory must differ")
        try:
            validate_sealed_discovery_memory(discovery_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid sealed discovery: {exc}")
        if run_memory_path.exists():
            parser.error(f"writable typed memory already exists: {run_memory_path}")
        if not args.shadow_log:
            parser.error("sealed discovery requires an explicit fresh --shadow-log")
        shadow_log_path = Path(args.shadow_log).resolve()
        if shadow_log_path.exists():
            parser.error(f"shadow log already exists: {shadow_log_path}")
    hypothesis_proposer = None
    if args.tiny_scientist_experiment_control == "dynamic_committed":
        hypothesis_proposer = LocalL1HypothesisProposer(
            args.tiny_scientist_dynamic_model,
            args.tiny_scientist_dynamic_adapter,
        )
    typed_interaction_proposer = (
        LocalCalibratedOrderedProposer(args.typed_interaction_formulation_model)
        if args.typed_interaction_formulation
        else None
    )

    link = UnityBodyLink(args.unity_host, args.unity_port, args.listen_port)
    ego = EmbodiedFunctionalEgo(
        hz=args.hz,
        sleep_seconds=args.sleep_seconds,
        min_awake_seconds=args.min_awake_seconds,
        sleep_threshold=args.sleep_threshold,
        wake_seconds=args.wake_seconds,
        maintenance_mode=args.maintenance_mode,
        handoff_threshold=args.handoff_threshold,
        emergency_sleep_threshold=args.emergency_sleep_threshold,
        handoff_cooldown_seconds=args.handoff_cooldown_seconds,
        memory_cell_size=args.memory_cell_size,
        obstacle_memory_decay=args.obstacle_memory_decay,
        route_exploration=args.route_exploration,
        dopamine=args.dopamine,
        norepinephrine=args.norepinephrine,
        acetylcholine=args.acetylcholine,
        noise_injection=args.noise_injection if args.delusion_drive is None else args.delusion_drive,
        calcium_gate=args.calcium_gate,
        unstuck_respawn_seconds=args.unstuck_respawn_seconds,
        shadow_checkpoint=args.shadow_policy,
        shadow_control=args.shadow_control,
        shadow_control_confidence=args.shadow_control_confidence,
        shadow_mpc=args.shadow_mpc,
        orbit_exit_adapter=args.orbit_exit_adapter,
        orbit_adapter_confidence=args.orbit_adapter_confidence,
        hidden_goal_adapter=args.hidden_goal_adapter,
        hidden_goal_adapter_confidence=args.hidden_goal_adapter_confidence,
        hidden_goal_adapter_commit_seconds=args.hidden_goal_adapter_commit_seconds,
        hidden_goal_adapter_lateral_bias=args.hidden_goal_adapter_lateral_bias,
        passive_conductor=args.passive_conductor,
        conductor_checkpoint=args.conductor_checkpoint,
        conductor_control=args.conductor_control,
        systemic_conductor_checkpoint=args.systemic_conductor_checkpoint,
        systemic_conductor_control=args.systemic_conductor_control,
        systemic_conductor_confidence=args.systemic_conductor_confidence,
        adaptive_gnw_passive=args.adaptive_gnw_passive,
        adaptive_gnw_control=args.adaptive_gnw_control,
        tiny_scientist_rule_control=args.tiny_scientist_rule_control,
        tiny_scientist_experiment_control=(
            args.tiny_scientist_experiment_control
        ),
        tiny_scientist_experiment_seed=args.seed,
        tiny_scientist_experiment_guidance_weight=(
            args.tiny_scientist_experiment_guidance_weight
        ),
        tiny_scientist_experiment_guidance_margin=(
            args.tiny_scientist_experiment_guidance_margin
        ),
        tiny_scientist_experiment_commit_seconds=(
            args.tiny_scientist_experiment_commit_seconds
        ),
        tiny_scientist_experiment_commit_cooldown_seconds=(
            args.tiny_scientist_experiment_commit_cooldown_seconds
        ),
        tiny_scientist_experiment_max_score_regret=(
            args.tiny_scientist_experiment_max_score_regret
        ),
        tiny_scientist_experiment_isolation_retreat_seconds=(
            args.tiny_scientist_experiment_isolation_retreat_seconds
        ),
        tiny_scientist_experiment_isolation_retreat_max_score_regret=(
            args.tiny_scientist_experiment_isolation_retreat_max_score_regret
        ),
        tiny_scientist_hypothesis_proposer=hypothesis_proposer,
        causal_probe_delay_seconds=args.causal_probe_delay_seconds,
        causal_probe_cancellation_feature=(
            args.causal_probe_cancellation_feature
        ),
        causal_probe_hunger_cost=args.causal_probe_hunger_cost,
        typed_interaction_learning=args.typed_interaction_learning,
        typed_interaction_memory=args.typed_interaction_memory,
        typed_interaction_discovery_memory=(
            args.typed_interaction_discovery_memory
        ),
        typed_interaction_hypothesis_proposer=typed_interaction_proposer,
        typed_interaction_rule_control=args.typed_interaction_rule_control,
    )
    ego.controller_seed = args.seed
    ego.terrain_air_observer = PassiveTerrainAirObserver(args.terrain_air_memory)
    ego.terrain_air_route_controller = TerrainAirRouteController(
        args.terrain_air_route_library,
        control_mode=args.terrain_air_route_control,
        hz=args.hz,
        max_control_seconds=args.terrain_air_route_seconds,
        max_guidance_weight=args.terrain_air_max_guidance_weight,
        guidance_margin_threshold=args.terrain_air_guidance_margin,
        teacher_memory=args.terrain_air_teacher_memory,
    )
    ego.resource_memory = PassiveTerrainResourceMemory(
        args.terrain_resource_memory,
        hunger_gate=args.terrain_resource_hunger_gate,
        hz=args.hz,
        control_mode=args.terrain_resource_control,
        max_guidance_weight=args.terrain_resource_max_guidance_weight,
    )
    ego.tiny_scientist_memory = VerifiedProductionMemory(
        args.tiny_scientist_production_memory
    )
    if args.initial_hunger is not None:
        ego.hunger = clamp(args.initial_hunger)
    initial_probe = args.initial_causal_probe_signal
    if initial_probe is None:
        initial_probe = args.initial_metabolic_pressure
    if initial_probe is not None:
        ego.causal_probe_signal = clamp(initial_probe)
    delay = 1.0 / max(args.hz, 0.1)
    started = time.time()
    latest_body = None
    last_payload = {"action": "idle", "mode": "wake", "fatigue": 0.0, "sleep_remaining": 0}
    diagnostic_teleport = args.diagnostic_teleport
    recorder = None
    completed_course_episodes = 0
    previous_course_episode = None
    previous_course_outcome = None
    if args.shadow_policy and not args.no_shadow_log:
        log_path = args.shadow_log
        if not log_path:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = f"outputs/unity_shadow/shadow_{stamp}.jsonl"
        recorder = ShadowRecorder(log_path)

    print(f"Sending commands to Unity UDP {args.unity_host}:{args.unity_port}")
    print(f"Listening for robot state on UDP 127.0.0.1:{args.listen_port}")
    if recorder is not None:
        print(f"Recording shadow telemetry to {recorder.path}")
    print("Press Ctrl-C to stop.")

    try:
        while args.duration <= 0.0 or time.time() - started < args.duration:
            received = link.receive_latest()
            if received is not None:
                latest_body = received
                if diagnostic_teleport is not None:
                    teleport_x, teleport_z = diagnostic_teleport
                    last_payload = {
                        "action": "diagnostic_teleport",
                        "mode": "wake",
                        "teleport_x": teleport_x,
                        "teleport_z": teleport_z,
                    }
                    print(f"Diagnostic teleport sent to ({teleport_x:.2f}, {teleport_z:.2f}).")
                    diagnostic_teleport = None
                else:
                    ego.update_from_body(latest_body)
                    ego.update_tiny_scientist_rule_targeting(latest_body)
                    action = ego.choose_action(latest_body)
                    ego.update_shadow_policy(latest_body, action)
                    ego.terrain_air_observer.update(
                        latest_body, ego.hunger, ego.shadow_action
                    )
                    ego.update_conductor_observer(latest_body)
                    ego.update_systemic_conductor(latest_body)
                    if recorder is not None:
                        recorder.write(ego, latest_body, action)
                    last_payload = ego.command_payload(action)
                    if ego.steps % int(max(args.hz, 1.0)) == 0:
                        print(ego.status_line(latest_body))
                    ego.steps += 1
                    course_episode = int(latest_body.get("trap_episode", 0) or 0)
                    course_outcome = str(latest_body.get("trap_outcome", "inactive"))
                    if (
                        course_episode == previous_course_episode
                        and previous_course_outcome == "running"
                        and course_outcome in {"success", "timeout"}
                    ):
                        completed_course_episodes += 1
                        print(
                            f"Course episode {course_episode} finished: {course_outcome} "
                            f"({completed_course_episodes}/{args.course_episodes or 'unbounded'})."
                        )
                    previous_course_episode = course_episode
                    previous_course_outcome = course_outcome
                    if args.course_episodes > 0 and completed_course_episodes >= args.course_episodes:
                        break
            link.send(last_payload)
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopping embodied loop.")
    finally:
        if recorder is not None:
            recorder.close()
        link.send({"action": "idle", "mode": "wake", "fatigue": ego.fatigue_report, "sleep_remaining": 0})


if __name__ == "__main__":
    main()
