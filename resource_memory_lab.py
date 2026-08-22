#!/usr/bin/env python3
"""Test valence-weighted episodic resource memory under food scarcity.

All conditions use the same frozen recurrent policy, regional novelty search,
obstacles, food sites, and random seeds. The experimental condition remembers
coarse rewarded regions and supplies a temporary subgoal prior when hungry and
food is not visible. Shuffled coordinates and per-step memory erasure are
capacity-matched controls.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from starvation_exploration_lab import (
    MAX_STEPS,
    PICKUP_RADIUS,
    STEP_LENGTH,
    StepResult,
    VISIT_CELL_SIZE,
    WORLD_SIZE,
    SparseTerrain,
    select_action,
)
from upgraded_foraging_pipeline import MOVES, load_checkpoint


CHECKPOINT = Path("checkpoints/unity_mpc/best.pt")
OUTPUT = Path("outputs/resource_memory_metrics.json")
CONDITIONS = ["no_memory", "resource_memory", "shuffled_memory", "memory_reset"]
RESOURCE_RESPAWN_STEPS = 120
RESOURCE_PATCH_COUNT = 3
SITES_PER_PATCH = 3
MEMORY_REFRACTORY_STEPS = 80
MEMORY_HUNGER_GATE = 0.55
VISIBLE_TARGET_STALE_STEPS = 240
MAX_RESOURCE_GUIDANCE_STEPS = 60
FAILED_MEMORY_SUPPRESSION_STEPS = 300


class RenewableSparseTerrain(SparseTerrain):
    def __init__(self, seed, sight):
        self.cooldowns = None
        self.last_pickup_position = None
        super().__init__(seed, sight)
        self.resource_sites = self._make_resource_patches()
        self.cooldowns = [0 for _ in self.resource_sites]

    def _make_resource_patches(self):
        centers = []
        for point in self.food:
            if all(float(np.linalg.norm(point - center)) >= 12.0 for center in centers):
                centers.append(point.copy())
            if len(centers) >= RESOURCE_PATCH_COUNT:
                break
        sites = []
        offsets = [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.5, 0.0], dtype=np.float32),
            np.asarray([-1.5, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.5], dtype=np.float32),
            np.asarray([0.0, -1.5], dtype=np.float32),
        ]
        for center in centers:
            patch = []
            for offset in offsets:
                candidate = np.clip(
                    center + offset,
                    2.0,
                    WORLD_SIZE - 2.0,
                ).astype(np.float32)
                if self.point_blocked(candidate):
                    continue
                patch.append(candidate)
                if len(patch) >= SITES_PER_PATCH:
                    break
            sites.extend(patch)
        if len(sites) < RESOURCE_PATCH_COUNT * SITES_PER_PATCH:
            raise RuntimeError("could not construct renewable resource patches")
        return sites

    def reset_resources(self):
        self.food = [point.copy() for point in self.resource_sites]
        self.cooldowns = [0 for _ in self.resource_sites]
        self.last_pickup_position = None

    def visible_food(self):
        if self.cooldowns is None:
            return super().visible_food()
        candidates = []
        for index, point in enumerate(self.resource_sites):
            if self.cooldowns[index] > 0:
                continue
            distance = float(np.linalg.norm(point - self.pos))
            if distance > self.sight:
                continue
            samples = max(2, int(math.ceil(distance / 0.25)))
            occluded = any(
                self.point_blocked(self.pos + alpha * (point - self.pos))
                for alpha in np.linspace(0.0, 1.0, samples)[1:-1]
            )
            if not occluded:
                candidates.append((distance, index, point))
        return min(candidates, default=None, key=lambda item: item[0])

    def step(self, action):
        for index in range(len(self.cooldowns)):
            self.cooldowns[index] = max(0, self.cooldowns[index] - 1)
        action = int(action)
        direction = MOVES[action].astype(np.float32)
        direction /= np.linalg.norm(direction)
        candidate = self.pos + direction * STEP_LENGTH
        collision = self.body_blocked(candidate)
        if collision:
            self.collisions += 1
        else:
            self.pos = candidate
            self.path_length += STEP_LENGTH
        cell = self.visit_cell()
        new_cell = cell not in self.visits
        self.visits[cell] += 1
        self.steps += 1
        self.hunger = min(1.0, self.hunger + 1.2 / MAX_STEPS)
        self.last_action = action
        ate = False
        self.last_pickup_position = None
        for index, point in enumerate(self.resource_sites):
            if self.cooldowns[index] > 0:
                continue
            if float(np.linalg.norm(point - self.pos)) <= PICKUP_RADIUS:
                ate = True
                self.pickups += 1
                if self.first_pickup_step is None:
                    self.first_pickup_step = self.steps
                self.pickup_steps.append(self.steps)
                self.last_reward = 2.0
                self.last_pickup_position = point.copy()
                self.cooldowns[index] = RESOURCE_RESPAWN_STEPS
                self.hunger = 0.35
                break
        if not ate:
            self.last_reward = -0.004 - (0.24 if collision else 0.0)
        return StepResult(self.observe(), ate, collision, new_cell)


@dataclass
class ResourceEpisode:
    position: np.ndarray
    value: float
    rewards: int
    last_reward_step: int
    failures: int = 0
    suppressed_until: int = 0


class ResourceMemory:
    def __init__(self, cell_size=VISIT_CELL_SIZE):
        self.cell_size = float(cell_size)
        self.entries = {}
        self.recalls = 0

    def cell(self, position):
        return tuple(np.floor(np.asarray(position) / self.cell_size).astype(int))

    def clear(self):
        self.entries.clear()

    def record_reward(self, position, step):
        key = self.cell(position)
        point = np.asarray(position, dtype=np.float32)
        entry = self.entries.get(key)
        if entry is None:
            self.entries[key] = ResourceEpisode(point, 1.0, 1, int(step))
            return
        weight = 1.0 / (entry.rewards + 1)
        entry.position = (1.0 - weight) * entry.position + weight * point
        entry.value = min(2.0, 0.85 * entry.value + 0.35)
        entry.rewards += 1
        entry.last_reward_step = int(step)
        entry.failures = max(0, entry.failures - 1)
        entry.suppressed_until = 0

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
            if age < MEMORY_REFRACTORY_STEPS:
                continue
            if int(step) < entry.suppressed_until:
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
            candidates.append((score, key, entry.position))
        if not candidates:
            return None
        self.recalls += 1
        _score, key, position = max(candidates, key=lambda item: item[0])
        return key, position.copy()


def shuffled_target(target):
    return np.asarray(
        [
            (float(target[0]) + WORLD_SIZE * 0.37) % WORLD_SIZE,
            (float(target[1]) + WORLD_SIZE * 0.61) % WORLD_SIZE,
        ],
        dtype=np.float32,
    )


def resource_guided_action(env, baseline_action, target):
    delta = np.asarray(target, dtype=np.float32) - env.pos
    length = float(np.linalg.norm(delta))
    if length < 3.0:
        return baseline_action
    direction = delta / max(length, 1e-6)
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
            0.72 * alignment
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
    resource_target = None
    resource_target_key = None
    resource_target_started = 0
    memory = ResourceMemory()
    reversals = 0
    previous_action = 0
    memory_guidance_steps = 0
    pickup_after_recall = 0
    recall_pending = False
    last_pickup_step = 0

    for _step in range(MAX_STEPS):
        if condition == "memory_reset":
            memory.clear()
            resource_target = None
            resource_target_key = None
        visible = env.food_visible() is not None
        stale_visible_target = (
            visible
            and bool(memory.entries)
            and env.steps - last_pickup_step >= VISIBLE_TARGET_STALE_STEPS
        )
        if (
            condition != "no_memory"
            and (not visible or stale_visible_target)
            and env.hunger >= MEMORY_HUNGER_GATE
            and (
                resource_target is None
                or float(np.linalg.norm(resource_target - env.pos)) < 3.0
            )
        ):
            recalled = memory.recall(env.pos, env.steps)
            if recalled is not None:
                resource_target_key, resource_target = recalled
                resource_target_started = env.steps
            if resource_target is not None and condition == "shuffled_memory":
                resource_target = shuffled_target(resource_target)
            recall_pending = resource_target is not None

        if (
            resource_target is not None
            and env.steps - resource_target_started >= MAX_RESOURCE_GUIDANCE_STEPS
        ):
            memory.record_failure(resource_target_key, env.steps)
            resource_target = None
            resource_target_key = None
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
        if resource_target is not None and (not visible or stale_visible_target):
            action = resource_guided_action(env, baseline, resource_target)
            memory_guidance_steps += 1
        if float(np.dot(MOVES[previous_action], MOVES[action])) < 0.0:
            reversals += 1
        result = env.step(action)
        if result.ate:
            memory.record_reward(env.last_pickup_position, env.steps)
            if recall_pending:
                pickup_after_recall += 1
            recall_pending = False
            resource_target = None
            resource_target_key = None
            novelty_target = None
            last_pickup_step = env.steps
        previous_action = action
        obs = torch.tensor(result.obs).unsqueeze(0)

    meal_boundaries = [0] + env.pickup_steps + [MAX_STEPS]
    meal_gaps = [
        later - earlier
        for earlier, later in zip(meal_boundaries, meal_boundaries[1:])
    ]
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
        "memory_recalls": memory.recalls,
        "memory_guidance_steps": memory_guidance_steps,
        "pickups_after_recall": pickup_after_recall,
    }


def summarize(rows):
    return {
        "episodes": len(rows),
        "first_pickup_success_rate": float(np.mean([row["first_pickup"] for row in rows])),
        "mean_first_pickup_steps": float(np.mean([row["first_pickup_step"] for row in rows])),
        "mean_pickups": float(np.mean([row["pickups"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_unique_cells": float(np.mean([row["unique_cells"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "mean_reversals": float(np.mean([row["reversals"] for row in rows])),
        "mean_max_meal_gap_steps": float(np.mean([row["max_meal_gap_steps"] for row in rows])),
        "mean_memory_entries": float(np.mean([row["memory_entries"] for row in rows])),
        "mean_memory_recalls": float(np.mean([row["memory_recalls"] for row in rows])),
        "mean_memory_guidance_steps": float(np.mean([row["memory_guidance_steps"] for row in rows])),
        "mean_pickups_after_recall": float(np.mean([row["pickups_after_recall"] for row in rows])),
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
    conditions = {}
    for condition in CONDITIONS:
        print(f"evaluating {condition}", flush=True)
        rows = [
            run_episode(policy, condition, 120_031 + episode * 997)
            for episode in range(episodes)
        ]
        conditions[condition] = summarize(rows)
        print(json.dumps(conditions[condition], indent=2), flush=True)

    baseline = conditions["no_memory"]
    candidate = conditions["resource_memory"]
    shuffled = conditions["shuffled_memory"]
    reset = conditions["memory_reset"]
    criteria = {
        "valid_memory_improves_pickups_10pct": (
            candidate["mean_pickups"] >= baseline["mean_pickups"] * 1.10
        ),
        "valid_memory_reduces_meal_gap_10pct": (
            candidate["mean_max_meal_gap_steps"]
            <= baseline["mean_max_meal_gap_steps"] * 0.90
        ),
        "valid_memory_beats_shuffled_locations": (
            candidate["mean_pickups"] >= shuffled["mean_pickups"] * 1.10
        ),
        "memory_reset_removes_majority_of_gain": (
            reset["mean_pickups"]
            <= baseline["mean_pickups"]
            + 0.40 * (candidate["mean_pickups"] - baseline["mean_pickups"])
        ),
        "collision_regression_at_most_10pct": (
            candidate["mean_collisions"] <= baseline["mean_collisions"] * 1.10 + 1.0
        ),
        "memory_is_actively_used": (
            candidate["mean_memory_guidance_steps"] > 0.0
            and candidate["mean_pickups_after_recall"] > 0.0
        ),
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "frozen_recurrent_policy": True,
            "renewable_fixed_resource_sites": True,
            "resource_respawn_steps": RESOURCE_RESPAWN_STEPS,
            "resource_patch_count": RESOURCE_PATCH_COUNT,
            "sites_per_patch": SITES_PER_PATCH,
            "memory_hunger_gate": MEMORY_HUNGER_GATE,
            "visible_target_stale_steps": VISIBLE_TARGET_STALE_STEPS,
            "max_resource_guidance_steps": MAX_RESOURCE_GUIDANCE_STEPS,
            "failed_memory_suppression_steps": FAILED_MEMORY_SUPPRESSION_STEPS,
            "coarse_memory_cell_size": VISIT_CELL_SIZE,
            "resource_coordinates_never_visible_outside_sensor_range": True,
            "memory_role": "temporary_subgoal_prior",
        },
        "conditions": conditions,
        "criteria": criteria,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"criteria": criteria, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
