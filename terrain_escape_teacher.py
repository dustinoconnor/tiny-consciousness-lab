"""Deterministic escape demonstrations for cold-start terrain ART memory."""

from __future__ import annotations

import math


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def world_to_local(origin, yaw_degrees, point):
    delta_x = float(point[0]) - float(origin[0])
    delta_z = float(point[1]) - float(origin[1])
    yaw = math.radians(float(yaw_degrees))
    right = (math.cos(yaw), -math.sin(yaw))
    forward = (math.sin(yaw), math.cos(yaw))
    return (
        delta_x * right[0] + delta_z * right[1],
        delta_x * forward[0] + delta_z * forward[1],
    )


def widest_safe_action(rays, body_clearance, minimum_clearance=0.50):
    safe = [
        index
        for index, clearance in enumerate(body_clearance)
        if float(clearance) >= minimum_clearance
    ]
    if not safe:
        return None
    return max(
        safe,
        key=lambda index: (
            0.58 * clamp(rays[index]) + 0.42 * clamp(body_clearance[index]),
            clamp(body_clearance[index]),
            clamp(rays[index]),
            -index,
        ),
    )


class TerrainEscapeTeacher:
    def __init__(
        self,
        hz=5.0,
        min_commit_seconds=2.5,
        max_commit_seconds=5.0,
        min_displacement=4.0,
        min_efficiency=0.65,
        max_collision_fraction=0.20,
        waypoint_spacing=1.0,
    ):
        self.hz = max(0.1, float(hz))
        self.min_commit_ticks = max(1, int(round(self.hz * min_commit_seconds)))
        self.max_commit_ticks = max(
            self.min_commit_ticks,
            int(round(self.hz * max_commit_seconds)),
        )
        self.min_displacement = float(min_displacement)
        self.min_efficiency = float(min_efficiency)
        self.max_collision_fraction = float(max_collision_fraction)
        self.waypoint_spacing = float(waypoint_spacing)
        self.active = False
        self.action_index = None
        self.action_name = "none"
        self.elapsed_ticks = 0
        self.start_position = None
        self.start_yaw = 0.0
        self.prototype = None
        self.positions = []
        self.collision_ticks = 0
        self.path_length = 0.0
        self.displacement = 0.0
        self.efficiency = 0.0
        self.events = 0
        self.successes = 0
        self.failures = 0
        self.safety_reselections = 0
        self.last_outcome = "none"
        self.last_route_id = "none"

    def reset(self, outcome="reset"):
        if self.active and outcome not in {"success", "timeout"}:
            self.last_outcome = outcome
        self.active = False
        self.action_index = None
        self.action_name = "none"
        self.elapsed_ticks = 0
        self.start_position = None
        self.start_yaw = 0.0
        self.prototype = None
        self.positions = []
        self.collision_ticks = 0
        self.path_length = 0.0
        self.displacement = 0.0
        self.efficiency = 0.0

    def start(
        self,
        body_state,
        prototype,
        actions,
        rays,
        body_clearance,
    ):
        selected = widest_safe_action(rays, body_clearance)
        if selected is None:
            self.last_outcome = "no_safe_action"
            return False
        position = (
            float(body_state.get("x", 0.0) or 0.0),
            float(body_state.get("z", 0.0) or 0.0),
        )
        self.active = True
        self.action_index = selected
        self.action_name = actions[selected]
        self.elapsed_ticks = 0
        self.start_position = position
        self.start_yaw = float(body_state.get("yaw", 0.0) or 0.0)
        self.prototype = [clamp(value) for value in prototype]
        self.positions = [position]
        self.collision_ticks = 0
        self.path_length = 0.0
        self.displacement = 0.0
        self.efficiency = 0.0
        self.events += 1
        self.last_outcome = "active"
        return True

    def _route_waypoints(self):
        local_points = [
            world_to_local(self.start_position, self.start_yaw, point)
            for point in self.positions[1:]
        ]
        selected = []
        previous = (0.0, 0.0)
        for point in local_points:
            if math.dist(point, previous) >= self.waypoint_spacing:
                selected.append(point)
                previous = point
        if local_points and (
            not selected or math.dist(local_points[-1], selected[-1]) > 0.25
        ):
            selected.append(local_points[-1])
        return [[float(x), float(z)] for x, z in selected]

    def update(
        self,
        body_state,
        actions,
        rays,
        body_clearance,
        route_controller,
    ):
        if not self.active:
            return None
        position = (
            float(body_state.get("x", 0.0) or 0.0),
            float(body_state.get("z", 0.0) or 0.0),
        )
        if math.dist(position, self.positions[-1]) > 25.0:
            self.failures += 1
            self.reset("position_jump")
            return None
        self.path_length += math.dist(position, self.positions[-1])
        self.positions.append(position)
        self.elapsed_ticks += 1
        if bool(body_state.get("horizontal_collision", False)):
            self.collision_ticks += 1
        self.displacement = math.dist(position, self.start_position)
        self.efficiency = self.displacement / max(self.path_length, 1e-6)

        selected = self.action_index
        if float(body_clearance[selected]) < 0.45:
            replacement = widest_safe_action(rays, body_clearance)
            if replacement is None:
                self.failures += 1
                self.reset("no_safe_action")
                return None
            if replacement != selected:
                self.safety_reselections += 1
                self.action_index = replacement
                self.action_name = actions[replacement]
                selected = replacement

        collision_fraction = self.collision_ticks / max(self.elapsed_ticks, 1)
        successful = (
            self.elapsed_ticks >= self.min_commit_ticks
            and self.displacement >= self.min_displacement
            and self.efficiency >= self.min_efficiency
            and collision_fraction <= self.max_collision_fraction
            and float(body_clearance[selected]) >= 0.50
        )
        if successful:
            waypoints = self._route_waypoints()
            route_id = route_controller.add_teacher_route(
                self.prototype,
                waypoints,
                {
                    "teacher_action": self.action_name,
                    "duration_seconds": self.elapsed_ticks / self.hz,
                    "displacement": self.displacement,
                    "path_efficiency": self.efficiency,
                    "collision_fraction": collision_fraction,
                },
            )
            if route_id is not None:
                self.successes += 1
                self.last_route_id = route_id
                self.last_outcome = "success"
                self.reset("success")
                return None
        if self.elapsed_ticks >= self.max_commit_ticks:
            self.failures += 1
            self.last_outcome = "timeout"
            self.reset("timeout")
            return None
        return selected
