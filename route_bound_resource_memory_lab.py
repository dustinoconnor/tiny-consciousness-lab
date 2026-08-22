#!/usr/bin/env python3
"""Test food-region memory bound to successful local approach trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from resource_memory_lab import (
    FAILED_MEMORY_SUPPRESSION_STEPS,
    MAX_RESOURCE_GUIDANCE_STEPS,
    MEMORY_HUNGER_GATE,
    MEMORY_REFRACTORY_STEPS,
    OUTPUT as COORDINATE_OUTPUT,
    VISIBLE_TARGET_STALE_STEPS,
    RenewableSparseTerrain,
)
from starvation_exploration_lab import MAX_STEPS, WORLD_SIZE, select_action
from upgraded_foraging_pipeline import MOVES, load_checkpoint


CHECKPOINT = Path("checkpoints/unity_mpc/best.pt")
OUTPUT = Path("outputs/route_bound_resource_memory_metrics.json")
CONDITIONS = [
    "no_memory",
    "coordinate_memory",
    "route_memory",
    "shuffled_route_memory",
    "memory_reset",
]
MEMORY_CELL_SIZE = 4.0
APPROACH_HISTORY_STEPS = 100
APPROACH_RADIUS = 14.0
WAYPOINT_SPACING = 1.25
WAYPOINT_RADIUS = 1.75
ROUTE_ENTRY_RADIUS = 3.5


def compress_approach(history, reward_position):
    points = [np.asarray(point, dtype=np.float32) for point in history]
    if not points:
        return []
    near = [
        index
        for index, point in enumerate(points)
        if float(np.linalg.norm(point - reward_position)) <= APPROACH_RADIUS
    ]
    if not near:
        return []
    source = points[near[0] :]
    selected = [source[0]]
    for point in source[1:]:
        if float(np.linalg.norm(point - selected[-1])) >= WAYPOINT_SPACING:
            selected.append(point)
    if float(np.linalg.norm(reward_position - selected[-1])) > 0.25:
        selected.append(np.asarray(reward_position, dtype=np.float32))
    return [point.copy() for point in selected]


def route_efficiency(route):
    if len(route) < 2:
        return 0.0
    path = sum(
        float(np.linalg.norm(current - previous))
        for previous, current in zip(route, route[1:])
    )
    net = float(np.linalg.norm(route[-1] - route[0]))
    return net / max(path, 1e-6)


@dataclass
class RouteResourceEpisode:
    position: np.ndarray
    value: float
    rewards: int
    last_reward_step: int
    approach: list[np.ndarray]
    approach_efficiency: float
    failures: int = 0
    suppressed_until: int = 0


class RouteResourceMemory:
    def __init__(self, cell_size=MEMORY_CELL_SIZE):
        self.cell_size = float(cell_size)
        self.entries = {}
        self.recalls = 0

    def cell(self, position):
        return tuple(np.floor(np.asarray(position) / self.cell_size).astype(int))

    def clear(self):
        self.entries.clear()

    def record_reward(self, position, step, history):
        key = self.cell(position)
        point = np.asarray(position, dtype=np.float32)
        approach = compress_approach(history, point)
        efficiency = route_efficiency(approach)
        entry = self.entries.get(key)
        if entry is None:
            self.entries[key] = RouteResourceEpisode(
                point,
                1.0,
                1,
                int(step),
                approach,
                efficiency,
            )
            return
        weight = 1.0 / (entry.rewards + 1)
        entry.position = (1.0 - weight) * entry.position + weight * point
        entry.value = min(2.0, 0.85 * entry.value + 0.35)
        entry.rewards += 1
        entry.last_reward_step = int(step)
        entry.failures = max(0, entry.failures - 1)
        entry.suppressed_until = 0
        if approach and efficiency >= entry.approach_efficiency:
            entry.approach = approach
            entry.approach_efficiency = efficiency

    def record_failure(self, key, step):
        entry = self.entries.get(key)
        if entry is None:
            return
        entry.failures += 1
        entry.value = max(0.10, entry.value * 0.60)
        entry.suppressed_until = int(step) + FAILED_MEMORY_SUPPRESSION_STEPS

    def recall(self, position, step):
        candidates = []
        for key, entry in self.entries.items():
            age = max(0, int(step) - entry.last_reward_step)
            if age < MEMORY_REFRACTORY_STEPS or int(step) < entry.suppressed_until:
                continue
            distance = float(np.linalg.norm(entry.position - position))
            if distance < 3.0:
                continue
            confidence = 1.0 - math.exp(-entry.rewards / 2.0)
            freshness = math.exp(-age / 1800.0)
            reliability = 1.0 / (1.0 + entry.failures)
            score = (
                entry.value * confidence * freshness * reliability
                - 0.006 * distance
            )
            candidates.append((score, key, entry))
        if not candidates:
            return None
        self.recalls += 1
        _score, key, entry = max(candidates, key=lambda item: item[0])
        return key, entry


def shuffle_point(point):
    return np.asarray(
        [
            (float(point[0]) + WORLD_SIZE * 0.37) % WORLD_SIZE,
            (float(point[1]) + WORLD_SIZE * 0.61) % WORLD_SIZE,
        ],
        dtype=np.float32,
    )


def memory_action(env, baseline_action, target, route_weight):
    delta = np.asarray(target, dtype=np.float32) - env.pos
    distance = float(np.linalg.norm(delta))
    if distance < 0.5:
        return baseline_action
    direction = delta / max(distance, 1e-6)
    safe = env.safe_actions()
    scores = []
    for action in range(len(MOVES)):
        if action not in safe:
            scores.append(-math.inf)
            continue
        move = MOVES[action].astype(np.float32)
        move /= np.linalg.norm(move)
        alignment = float(np.dot(move, direction))
        continuity = 1.0 if action == baseline_action else 0.0
        scores.append(
            route_weight * alignment
            + 0.30 * env.novelty_score(action)
            + 0.48 * continuity
        )
    return int(np.argmax(scores))


def run_episode(policy, condition, seed):
    env = RenewableSparseTerrain(seed, sight=7.0)
    env.reset()
    env.reset_resources()
    obs = torch.tensor(env.observe()).unsqueeze(0)
    hidden = policy.initial_state(1)
    novelty_target = None
    memory = RouteResourceMemory()
    approach_history = deque([env.pos.copy()], maxlen=APPROACH_HISTORY_STEPS)
    recalled_key = None
    recalled_entry = None
    recalled_position = None
    recalled_route = []
    route_index = 0
    route_active = False
    recall_started = 0
    recall_pending = False
    last_pickup_step = 0
    previous_action = 0
    reversals = 0
    memory_guidance_steps = 0
    route_guidance_steps = 0
    pickups_after_recall = 0

    for _step in range(MAX_STEPS):
        if condition == "memory_reset":
            memory.clear()
            recalled_key = None
            recalled_entry = None
            recalled_position = None
            recalled_route = []
            route_active = False
        visible = env.food_visible() is not None
        stale_visible = (
            visible
            and bool(memory.entries)
            and env.steps - last_pickup_step >= VISIBLE_TARGET_STALE_STEPS
        )
        recall_allowed = condition not in {"no_memory", "memory_reset"}
        if (
            recall_allowed
            and recalled_entry is None
            and (not visible or stale_visible)
            and env.hunger >= MEMORY_HUNGER_GATE
        ):
            recalled = memory.recall(env.pos, env.steps)
            if recalled is not None:
                recalled_key, recalled_entry = recalled
                recalled_position = recalled_entry.position.copy()
                recalled_route = [
                    point.copy() for point in recalled_entry.approach
                ]
                if condition == "shuffled_route_memory":
                    recalled_position = shuffle_point(recalled_position)
                    recalled_route = [shuffle_point(point) for point in recalled_route]
                route_index = 0
                route_active = False
                recall_started = env.steps
                recall_pending = True

        if recalled_entry is not None:
            if (
                not route_active
                and recalled_route
                and float(np.linalg.norm(recalled_route[0] - env.pos))
                <= ROUTE_ENTRY_RADIUS
            ):
                route_active = True
            while route_active and route_index < len(recalled_route):
                if (
                    float(np.linalg.norm(recalled_route[route_index] - env.pos))
                    > WAYPOINT_RADIUS
                ):
                    break
                route_index += 1
            if env.steps - recall_started >= MAX_RESOURCE_GUIDANCE_STEPS:
                memory.record_failure(recalled_key, env.steps)
                recalled_key = None
                recalled_entry = None
                recalled_position = None
                recalled_route = []
                route_active = False
                recall_pending = False

        baseline, hidden, novelty_target = select_action(
            policy,
            obs,
            hidden,
            env,
            "regional_novelty",
            novelty_target,
        )
        action = baseline
        if recalled_entry is not None and (not visible or stale_visible):
            target = recalled_position
            route_weight = 0.72
            if (
                condition in {"route_memory", "shuffled_route_memory"}
                and route_active
            ):
                if route_index < len(recalled_route):
                    target = recalled_route[route_index]
                    route_weight = 0.92
                    route_guidance_steps += 1
            action = memory_action(env, baseline, target, route_weight)
            memory_guidance_steps += 1
        if float(np.dot(MOVES[previous_action], MOVES[action])) < 0.0:
            reversals += 1
        result = env.step(action)
        approach_history.append(env.pos.copy())
        if result.ate:
            memory.record_reward(
                env.last_pickup_position,
                env.steps,
                list(approach_history),
            )
            if recall_pending:
                pickups_after_recall += 1
            recalled_key = None
            recalled_entry = None
            recalled_position = None
            recalled_route = []
            route_active = False
            recall_pending = False
            novelty_target = None
            last_pickup_step = env.steps
            approach_history.clear()
            approach_history.append(env.pos.copy())
        previous_action = action
        obs = torch.tensor(result.obs).unsqueeze(0)

    meal_boundaries = [0] + env.pickup_steps + [MAX_STEPS]
    meal_gaps = [
        later - earlier
        for earlier, later in zip(meal_boundaries, meal_boundaries[1:])
    ]
    route_entries = sum(bool(entry.approach) for entry in memory.entries.values())
    return {
        "first_pickup": env.first_pickup_step is not None,
        "first_pickup_step": env.first_pickup_step or MAX_STEPS,
        "pickups": env.pickups,
        "collisions": env.collisions,
        "unique_cells": len(env.visits),
        "revisit_ratio": 1.0 - len(env.visits) / max(1, sum(env.visits.values())),
        "reversals": reversals,
        "max_meal_gap_steps": max(meal_gaps),
        "memory_entries": len(memory.entries),
        "route_entries": route_entries,
        "memory_recalls": memory.recalls,
        "memory_guidance_steps": memory_guidance_steps,
        "route_guidance_steps": route_guidance_steps,
        "pickups_after_recall": pickups_after_recall,
    }


def summarize(rows):
    keys = [
        "first_pickup",
        "first_pickup_step",
        "pickups",
        "collisions",
        "unique_cells",
        "revisit_ratio",
        "reversals",
        "max_meal_gap_steps",
        "memory_entries",
        "route_entries",
        "memory_recalls",
        "memory_guidance_steps",
        "route_guidance_steps",
        "pickups_after_recall",
    ]
    return {
        "episodes": len(rows),
        **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in keys},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    policy, _payload = load_checkpoint(args.checkpoint)
    policy.eval()
    episodes = 4 if args.quick else max(8, args.episodes)
    results = {}
    episode_rows = {}
    for condition in CONDITIONS:
        print(f"evaluating {condition}", flush=True)
        rows = [
            run_episode(policy, condition, 130_031 + episode * 997)
            for episode in range(episodes)
        ]
        episode_rows[condition] = rows
        results[condition] = summarize(rows)
        print(json.dumps(results[condition], indent=2), flush=True)

    baseline = results["no_memory"]
    coordinate = results["coordinate_memory"]
    candidate = results["route_memory"]
    shuffled = results["shuffled_route_memory"]
    reset = results["memory_reset"]
    criteria = {
        "route_memory_improves_pickups_10pct_over_baseline": (
            candidate["mean_pickups"] >= baseline["mean_pickups"] * 1.10
        ),
        "route_memory_reduces_meal_gap_10pct": (
            candidate["mean_max_meal_gap_steps"]
            <= baseline["mean_max_meal_gap_steps"] * 0.90
        ),
        "route_binding_beats_coordinate_only": (
            candidate["mean_pickups"] >= coordinate["mean_pickups"]
            and candidate["mean_max_meal_gap_steps"]
            <= coordinate["mean_max_meal_gap_steps"]
        ),
        "route_memory_beats_shuffled": (
            candidate["mean_pickups"] >= shuffled["mean_pickups"] * 1.10
        ),
        "memory_reset_matches_baseline": (
            abs(reset["mean_pickups"] - baseline["mean_pickups"]) <= 0.25
        ),
        "collision_regression_at_most_10pct": (
            candidate["mean_collisions"] <= baseline["mean_collisions"] * 1.10 + 1.0
        ),
        "route_memory_is_actively_used": (
            candidate["mean_route_guidance_steps"] > 0.0
            and candidate["mean_pickups_after_recall"] > 0.0
        ),
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "frozen_recurrent_policy": True,
            "renewable_resource_patches": True,
            "coordinate_memory_reference": str(COORDINATE_OUTPUT),
            "approach_history_steps": APPROACH_HISTORY_STEPS,
            "approach_radius": APPROACH_RADIUS,
            "waypoint_spacing": WAYPOINT_SPACING,
            "route_entry_radius": ROUTE_ENTRY_RADIUS,
            "memory_role": "bounded resource subgoal plus successful local approach",
        },
        "conditions": results,
        "episode_rows": episode_rows,
        "criteria": criteria,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"criteria": criteria, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
