"""Bounded AIR/ART terrain-route recommendation and intervention controller."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def intervention_reason(wedge_seconds, trap_seconds, orbit_path, orbit_efficiency):
    if float(wedge_seconds) >= 1.0:
        return "physical_wedge"
    if float(trap_seconds) >= 2.0:
        return "trap_accumulation"
    if float(orbit_path) >= 4.0 and float(orbit_efficiency) <= 0.25:
        return "low_efficiency_orbit"
    return "none"


def score_margin_ambiguity(margin, threshold):
    threshold = max(float(threshold), 1e-6)
    return clamp(1.0 - max(0.0, float(margin)) / threshold)


def context_vector(
    body_state,
    hunger,
    orbit_path,
    orbit_efficiency,
    wedge_seconds,
    trap_seconds,
):
    rays = np.asarray(body_state.get("directional_rays", []), dtype=np.float64)
    clearance = np.asarray(
        body_state.get("directional_body_clearance", []), dtype=np.float64
    )
    if rays.size != 8 or clearance.size != 8:
        return None
    return np.r_[
        np.clip(rays, 0.0, 1.0),
        np.clip(clearance, 0.0, 1.0),
        clamp(hunger),
        float(bool(body_state.get("food_visible", False))),
        clamp(orbit_efficiency),
        clamp(float(orbit_path) / 20.0),
        clamp(float(wedge_seconds) / 8.0),
        clamp(float(trap_seconds) / 8.0),
    ]


def local_to_world(local, yaw_degrees):
    x, forward = map(float, local)
    yaw = math.radians(float(yaw_degrees))
    right_vector = (math.cos(yaw), -math.sin(yaw))
    forward_vector = (math.sin(yaw), math.cos(yaw))
    return (
        right_vector[0] * x + forward_vector[0] * forward,
        right_vector[1] * x + forward_vector[1] * forward,
    )


class TerrainAirRouteController:
    def __init__(
        self,
        checkpoint=None,
        control_mode="passive",
        hz=5.0,
        max_control_seconds=4.0,
        cooldown_seconds=8.0,
        max_guidance_weight=0.05,
        guidance_margin_threshold=0.05,
    ):
        if control_mode not in {"passive", "bounded", "guided"}:
            raise ValueError(
                "terrain AIR route control must be passive, bounded, or guided"
            )
        self.enabled = checkpoint is not None
        self.control_mode = control_mode
        self.hz = max(0.1, float(hz))
        self.max_control_ticks = max(1, int(round(self.hz * max_control_seconds)))
        self.cooldown_ticks_total = max(1, int(round(self.hz * cooldown_seconds)))
        self.max_guidance_weight = clamp(max_guidance_weight, 0.0, 1.0)
        self.guidance_margin_threshold = max(
            1e-6, float(guidance_margin_threshold)
        )
        self.vigilance = 0.78
        self.routes = []
        self.active = False
        self.recommendation = "none"
        self.reason = "none"
        self.match = 0.0
        self.route_index = 0
        self.route_count = 0
        self.route_distance = 0.0
        self.remaining_ticks = 0
        self.cooldown_ticks = 0
        self.recommendations = 0
        self.interventions = 0
        self.action_influence = 0
        self.guidance_decisions = 0
        self.guidance_action_changes = 0
        self.last_unguided_action = "none"
        self.last_guided_action = "none"
        self.last_unguided_margin = 0.0
        self.last_margin_ambiguity = 0.0
        self.last_effective_guidance_weight = 0.0
        self.sensor_vetoes = 0
        self.releases = 0
        self.release_reason = "none"
        self.activation_food_visible = False
        self.guidance_vector = (0.0, 0.0)
        self.guidance_weight = 0.0
        self.last_margin_ambiguity = 0.0
        self.last_effective_guidance_weight = 0.0
        self.world_waypoints = []
        self.start_position = None
        self.previous_position = None
        if not self.enabled:
            return
        payload = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        if payload.get("format") != "terrain_air_art_route_library_v1":
            raise ValueError("unsupported terrain AIR route checkpoint")
        self.vigilance = clamp(payload.get("art_vigilance", 0.78))
        for route in payload.get("routes", []):
            prototype = np.asarray(route.get("prototype", []), dtype=np.float64)
            waypoints = route.get("waypoints", [])
            if prototype.size != 22 or not waypoints:
                continue
            self.routes.append(
                {
                    "id": f"terrain_episode_{route.get('episode_id', len(self.routes))}",
                    "prototype": np.clip(prototype, 0.0, 1.0),
                    "waypoints": [tuple(map(float, point)) for point in waypoints],
                }
            )
        if not self.routes:
            raise ValueError("terrain AIR route checkpoint contains no valid routes")

    def reset(self, release_reason="reset"):
        if self.active:
            self.releases += 1
        self.active = False
        self.world_waypoints = []
        self.route_index = 0
        self.route_count = 0
        self.route_distance = 0.0
        self.remaining_ticks = 0
        self.release_reason = release_reason
        self.guidance_vector = (0.0, 0.0)
        self.guidance_weight = 0.0
        self.last_margin_ambiguity = 0.0
        self.last_effective_guidance_weight = 0.0
        self.cooldown_ticks = self.cooldown_ticks_total
        self.start_position = None

    def select_route(self, context):
        distances = [
            float(np.mean(np.abs(route["prototype"] - context)))
            for route in self.routes
        ]
        index = int(np.argmin(distances))
        match = max(0.0, 1.0 - distances[index])
        return (self.routes[index] if match >= self.vigilance else None), match

    def activate(self, route, match, reason, position, yaw, food_visible):
        self.recommendation = route["id"]
        self.reason = reason
        self.match = match
        self.recommendations += 1
        self.cooldown_ticks = self.cooldown_ticks_total
        if self.control_mode == "passive":
            return
        self.world_waypoints = []
        for waypoint in route["waypoints"]:
            dx, dz = local_to_world(waypoint, yaw)
            self.world_waypoints.append((position[0] + dx, position[1] + dz))
        self.active = True
        self.route_index = 0
        self.route_count = len(self.world_waypoints)
        self.remaining_ticks = self.max_control_ticks
        self.activation_food_visible = bool(food_visible)
        self.start_position = position
        self.interventions += 1
        self.release_reason = "none"

    @staticmethod
    def necessity_strength(reason, wedge_seconds, trap_seconds, orbit_path, orbit_efficiency):
        if reason == "physical_wedge":
            return clamp(float(wedge_seconds) / 4.0, 0.25, 1.0)
        if reason == "trap_accumulation":
            return clamp(float(trap_seconds) / 6.0, 0.33, 1.0)
        if reason == "low_efficiency_orbit":
            inefficiency = clamp(
                (0.35 - float(orbit_efficiency)) / 0.20, 0.25, 1.0
            )
            path_pressure = clamp(float(orbit_path) / 12.0, 0.33, 1.0)
            return max(inefficiency, path_pressure)
        return 0.0

    def update(
        self,
        body_state,
        hunger,
        orbit_path,
        orbit_efficiency,
        wedge_seconds,
        trap_seconds,
        actions,
        vectors,
        body_clearance,
        fallback_active=False,
    ):
        if not self.enabled:
            return None
        position = (
            float(body_state.get("x", 0.0) or 0.0),
            float(body_state.get("z", 0.0) or 0.0),
        )
        if (
            self.previous_position is not None
            and math.dist(position, self.previous_position) > 25.0
        ):
            self.reset("position_jump")
        self.previous_position = position
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1

        food_visible = bool(body_state.get("food_visible", False))
        if self.active:
            if fallback_active:
                self.reset("stable_fallback")
                return None
            if not self.activation_food_visible and food_visible:
                self.reset("new_target_grounding")
                return None
            if self.remaining_ticks <= 0:
                self.reset(
                    "guidance_timeout"
                    if self.control_mode == "guided"
                    else "bounded_timeout"
                )
                return None
            if float(orbit_efficiency) > 0.45:
                self.reset("progress_restored")
                return None
            if not self.world_waypoints:
                self.reset("empty_route")
                return None
            while self.route_index < len(self.world_waypoints) - 1:
                if math.dist(position, self.world_waypoints[self.route_index]) > 1.0:
                    break
                self.route_index += 1
            target = self.world_waypoints[self.route_index]
            delta = (target[0] - position[0], target[1] - position[1])
            self.route_distance = math.hypot(*delta)
            if (
                self.route_index == len(self.world_waypoints) - 1
                and self.route_distance <= 0.75
            ):
                self.reset("route_complete")
                return None
            desired_length = max(self.route_distance, 1e-6)
            desired = (delta[0] / desired_length, delta[1] / desired_length)
            self.guidance_vector = desired
            severity = self.necessity_strength(
                self.reason,
                wedge_seconds,
                trap_seconds,
                orbit_path,
                orbit_efficiency,
            )
            self.guidance_weight = self.max_guidance_weight * self.match * severity
            self.remaining_ticks -= 1
            if self.control_mode == "guided":
                return None
            ideal = max(
                range(len(actions)),
                key=lambda index: (
                    vectors[actions[index]][0] * desired[0]
                    + vectors[actions[index]][1] * desired[1]
                ),
            )
            safe = [
                index
                for index, clearance in enumerate(body_clearance)
                if float(clearance) >= 0.5
            ]
            if not safe:
                self.reset("no_safe_action")
                return None
            selected = max(
                safe,
                key=lambda index: (
                    vectors[actions[index]][0] * desired[0]
                    + vectors[actions[index]][1] * desired[1]
                ),
            )
            self.sensor_vetoes += int(selected != ideal)
            self.action_influence += 1
            return selected

        reason = intervention_reason(
            wedge_seconds, trap_seconds, orbit_path, orbit_efficiency
        )
        self.reason = reason
        if reason == "none" or self.cooldown_ticks > 0:
            return None
        context = context_vector(
            body_state,
            hunger,
            orbit_path,
            orbit_efficiency,
            wedge_seconds,
            trap_seconds,
        )
        if context is None:
            return None
        route, match = self.select_route(context)
        self.match = match
        if route is None:
            self.recommendation = "no_resonance"
            self.cooldown_ticks = self.cooldown_ticks_total
            return None
        self.activate(
            route,
            match,
            reason,
            position,
            float(body_state.get("yaw", 0.0) or 0.0),
            food_visible,
        )
        if self.control_mode in {"bounded", "guided"}:
            return self.update(
                body_state,
                hunger,
                orbit_path,
                orbit_efficiency,
                wedge_seconds,
                trap_seconds,
                actions,
                vectors,
                body_clearance,
                fallback_active=fallback_active,
            )
        return None
