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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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
            "position": [body_state.get("x"), body_state.get("y"), body_state.get("z")],
            "rays": body_state.get("directional_rays"),
            "body_clearance": body_state.get("directional_body_clearance"),
            "food_visible": bool(body_state.get("food_visible", False)),
            "food_distance": body_state.get("food_distance"),
            "food_world": [body_state.get("food_world_x"), body_state.get("food_world_z")],
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
            "shadow_mpc": ego.shadow_mpc,
            "shadow_mpc_engaged": ego.shadow_mpc_engaged,
            "shadow_mpc_score": ego.shadow_mpc_score,
            "shadow_mpc_mode": ego.shadow_mpc_mode,
            "shadow_mpc_horizon": ego.shadow_mpc_horizon,
            "shadow_mpc_depth": ego.shadow_mpc_depth,
            "shadow_mpc_uncertainty_stops": ego.shadow_mpc_uncertainty_stops,
            "food_sensor_radius": ego.shadow_food_sensor_radius,
            "blocked": bool(body_state.get("blocked", False)),
            "body_collision": bool(body_state.get("horizontal_collision", False)),
            "stuck": ego.current_stuck,
            "stuck_events": ego.stuck_events,
            "physics_wedge_seconds": ego.physics_wedge_ticks / ego.hz,
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
            "trap_episode": body_state.get("trap_episode", 0),
            "trap_successes": body_state.get("trap_successes", 0),
            "trap_failures": body_state.get("trap_failures", 0),
            "trap_outcome": body_state.get("trap_outcome", "inactive"),
            "mushroom_pickups_total": body_state.get("mushroom_pickups_total", 0),
            "mushroom_reward_total": body_state.get("mushroom_reward_total", 0.0),
            "mushrooms_eaten": ego.mushrooms_eaten,
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
        self.shadow_last_reward = 0.0
        self.shadow_previous_action = 0
        self.shadow_previous_position = None
        self.shadow_previous_proposal = "none"
        self.shadow_ray_range = 6.0
        self.shadow_error = "disabled"
        self.shadow_control = shadow_control
        self.shadow_control_confidence = clamp(shadow_control_confidence)
        self.shadow_takeover = False
        self.shadow_takeover_steps = 0
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
        self.shadow_food_sensor_radius = 16.0
        self.trap_course_label = "natural_terrain"
        self.trap_course_episode = 0
        self.trap_course_successes = 0
        self.trap_course_failures = 0
        self.trap_course_outcome = "inactive"
        if shadow_checkpoint:
            self.enable_shadow_policy(shadow_checkpoint)

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

    def update_shadow_policy(self, body_state, active_action):
        was_shadow_takeover = self.shadow_takeover
        self.shadow_ready = False
        self.shadow_takeover = False
        if not self.shadow_enabled or not isinstance(body_state, dict):
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
            food_visible = 1.0 if body_state.get("food_visible", False) else 0.0
            food_distance = max(0.0, float(body_state.get("food_distance", 0.0)))
            food_scale = min(1.0, food_distance / 14.0) if food_visible else 0.0
            food_x = float(body_state.get("food_world_x", 0.0)) * food_scale
            food_z = float(body_state.get("food_world_z", 0.0)) * food_scale
            position = (float(body_state.get("x", 0.0)), float(body_state.get("z", 0.0)))
        except (TypeError, ValueError):
            self.shadow_error = "invalid_telemetry"
            return

        previous = [0.0] * 8
        previous[self.shadow_previous_action] = 1.0
        observation = rays + [food_visible, food_x, food_z, self.hunger] + previous + [self.shadow_last_reward]
        torch = self.shadow_torch
        with torch.no_grad():
            obs = torch.tensor([observation], dtype=torch.float32)
            logits, _value, self.shadow_hidden = self.shadow_policy.step(obs, self.shadow_hidden)
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
                mpc_needed = food_visible > 0.0 or tactical_obstacle
            else:
                mpc_needed = food_visible > 0.0
            if mpc_needed:
                self.shadow_mpc_hold_ticks = max(
                    self.shadow_mpc_hold_ticks,
                    max(2, int(round(self.hz * 1.5))),
                )
            else:
                self.shadow_mpc_hold_ticks = max(0, self.shadow_mpc_hold_ticks - 1)
            self.shadow_mpc_engaged = self.shadow_mpc and (
                mpc_needed or self.shadow_mpc_hold_ticks > 0
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
        if self.physics_wedge_ticks >= fallback_threshold and self.shadow_fallback_hold_ticks <= 0:
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
        selected = int(torch.argmax(risk_adjusted).item())
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
            self.trap_course_episode = int(body_state.get("trap_episode", self.trap_course_episode) or 0)
            self.trap_course_successes = int(body_state.get("trap_successes", self.trap_course_successes) or 0)
            self.trap_course_failures = int(body_state.get("trap_failures", self.trap_course_failures) or 0)
            self.trap_course_outcome = body_state.get("trap_outcome", self.trap_course_outcome)
        self.apply_control_overrides(body_state)
        self.apply_food_feedback(body_state)
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

    def apply_food_feedback(self, body_state):
        self.shadow_last_reward = 0.0
        if isinstance(body_state, dict):
            try:
                pickup_total = int(body_state.get("mushroom_pickups_total", body_state.get("ate_mushroom", 0)))
            except (TypeError, ValueError):
                pickup_total = self.last_mushroom_pickup_total
            try:
                reward_total = float(body_state.get("mushroom_reward_total", body_state.get("mushroom_reward", 0.0)))
            except (TypeError, ValueError):
                reward_total = self.last_mushroom_reward_total

            if not self.food_feedback_initialized:
                self.last_mushroom_pickup_total = pickup_total
                self.last_mushroom_reward_total = reward_total
                self.food_feedback_initialized = True
                pickup_total = self.last_mushroom_pickup_total
                reward_total = self.last_mushroom_reward_total

            if pickup_total < self.last_mushroom_pickup_total or reward_total < self.last_mushroom_reward_total:
                self.last_mushroom_pickup_total = 0
                self.last_mushroom_reward_total = 0.0
                self.mushrooms_eaten = 0

            eaten = max(0, pickup_total - self.last_mushroom_pickup_total)
            reward = max(0.0, reward_total - self.last_mushroom_reward_total)
            if eaten > 0:
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
            self.last_mushroom_pickup_total = pickup_total
            self.last_mushroom_reward_total = reward_total

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

        if self.unstuck_respawn_ticks > 0 and self.physics_wedge_ticks >= self.unstuck_respawn_ticks:
            self.survival_failure_events += 1
            self.survival_failure_reason = "persistent_physics_wedge"
            self.unstuck_respawns += 1
            self.physics_wedge_ticks = 0
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
        if trap_pressure < (0.62 if hunger_anchor else 0.35):
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
            "shadow_world_x": SHADOW_VECTORS.get(self.shadow_action, (0.0, 0.0))[0],
            "shadow_world_z": SHADOW_VECTORS.get(self.shadow_action, (0.0, 0.0))[1],
            "shadow_takeover_steps": self.shadow_takeover_steps,
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
            "food_sensor_radius": round(self.shadow_food_sensor_radius, 2),
            "trap_course": self.trap_course_label,
            "trap_episode": self.trap_course_episode,
            "trap_successes": self.trap_course_successes,
            "trap_failures": self.trap_course_failures,
            "trap_outcome": self.trap_course_outcome,
        }
        if self.shadow_takeover:
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
            f"wedge={self.physics_wedge_ticks / self.hz:.1f}s respawns={self.unstuck_respawns:02d} "
            f"obstacles={len(self.obstacle_memory):02d}/{self.obstacle_events:02d} "
            f"routes={len(self.escape_attempts):02d} traps={len(self.trap_memory):02d}:{trap_pressure:.2f} "
            f"breakouts={self.breakout_events:02d} "
            f"shadow={self.shadow_action}:{self.shadow_confidence:.2f}/{self.shadow_agreement:.2f} "
            f"takeover={int(self.shadow_takeover)} "
            f"recent={len(self.recent_failures):02d} {body}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the first Python-to-Unity embodied functional ego loop.")
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--unity-port", type=int, default=5055)
    parser.add_argument("--listen-port", type=int, default=5056)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until Ctrl-C.")
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
    args = parser.parse_args()

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
    )
    if args.initial_hunger is not None:
        ego.hunger = clamp(args.initial_hunger)
    delay = 1.0 / max(args.hz, 0.1)
    started = time.time()
    latest_body = None
    last_payload = {"action": "idle", "mode": "wake", "fatigue": 0.0, "sleep_remaining": 0}
    recorder = None
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
                ego.update_from_body(latest_body)
                action = ego.choose_action(latest_body)
                ego.update_shadow_policy(latest_body, action)
                if recorder is not None:
                    recorder.write(ego, latest_body, action)
                last_payload = ego.command_payload(action)
                if ego.steps % int(max(args.hz, 1.0)) == 0:
                    print(ego.status_line(latest_body))
                ego.steps += 1
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
