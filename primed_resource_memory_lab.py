#!/usr/bin/env python3
"""Isolate resource-memory encoding from delayed, occluded recall."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from resource_memory_lab import (
    FAILED_MEMORY_SUPPRESSION_STEPS,
    MEMORY_HUNGER_GATE,
    RenewableSparseTerrain,
)
from route_bound_resource_memory_lab import memory_action, shuffle_point
from starvation_exploration_lab import MAX_STEPS, WORLD_SIZE, select_action
from upgraded_foraging_pipeline import MOVES, load_checkpoint


CHECKPOINT = Path("checkpoints/unity_mpc/best.pt")
OUTPUT = Path("outputs/primed_resource_memory_metrics.json")
CONDITIONS = [
    "no_memory",
    "coordinate_memory",
    "anchor_memory",
    "shuffled_anchor_memory",
    "memory_reset",
]
PRIME_MAX_STEPS = 180
PRIME_DISTANCE = 5.5
RECALL_START_MIN_DISTANCE = 18.0
RECALL_START_MAX_DISTANCE = 34.0
RECALL_GUIDANCE_STEPS = 180
ANCHOR_RADIUS = 1.75
ANCHOR_SEARCH_STEPS = 24
STALE_ESCAPE_STEPS = 30


@dataclass
class ResourcePrecedent:
    region: np.ndarray
    anchor: np.ndarray
    value: float = 1.0
    successes: int = 1
    failures: int = 0
    suppressed_until: int = 0


class PrimedResourceMemory:
    def __init__(self):
        self.entries: list[ResourcePrecedent] = []
        self.recalls = 0

    def clear(self):
        self.entries.clear()

    def record(self, region, anchor):
        self.entries = [
            ResourcePrecedent(
                np.asarray(region, dtype=np.float32).copy(),
                np.asarray(anchor, dtype=np.float32).copy(),
            )
        ]

    def recall(self, position, step):
        candidates = []
        for entry in self.entries:
            if step < entry.suppressed_until:
                continue
            distance = float(np.linalg.norm(entry.region - position))
            reliability = entry.successes / (
                entry.successes + entry.failures + 1.0
            )
            score = entry.value * reliability - 0.004 * distance
            candidates.append((score, entry))
        if not candidates:
            return None
        self.recalls += 1
        return max(candidates, key=lambda item: item[0])[1]

    def reward(self, entry):
        entry.successes += 1
        entry.value = min(2.0, entry.value + 0.25)

    def penalize(self, entry, step):
        entry.failures += 1
        entry.value = max(0.10, entry.value * 0.50)
        entry.suppressed_until = step + FAILED_MEMORY_SUPPRESSION_STEPS


def patch_centroid(env, pickup_position):
    distances = [
        float(np.linalg.norm(point - pickup_position))
        for point in env.resource_sites
    ]
    site_index = int(np.argmin(distances))
    patch_start = (site_index // 3) * 3
    return np.mean(
        env.resource_sites[patch_start : patch_start + 3],
        axis=0,
    ).astype(np.float32)


def final_visible_anchor(history):
    """Return the start of the final uninterrupted food-visible approach."""
    if not history:
        raise ValueError("approach history cannot be empty")
    index = len(history) - 1
    while index > 0 and history[index - 1][1]:
        index -= 1
    if not history[index][1]:
        return history[-1][0].copy()
    return history[index][0].copy()


def find_prime_start(env):
    target = env.resource_sites[0]
    directions = [
        move.astype(np.float32) / np.linalg.norm(move)
        for move in MOVES
    ]
    for distance in (PRIME_DISTANCE, 4.5, 3.5):
        for direction in directions:
            candidate = target - direction * distance
            if env.body_blocked(candidate):
                continue
            original = env.pos.copy()
            env.pos = candidate.astype(np.float32)
            target_visible = env.visible_food() is not None
            env.pos = original
            if target_visible:
                return candidate.astype(np.float32)
    raise RuntimeError("could not place a visible prime start")


def prime_memory(policy, env):
    env.reset()
    env.reset_resources()
    env.pos = find_prime_start(env)
    env.hunger = 0.82
    env.visits = Counter({env.visit_cell(): 1})
    obs = torch.tensor(env.observe()).unsqueeze(0)
    hidden = policy.initial_state(1)
    novelty_target = None
    history = []
    for _ in range(PRIME_MAX_STEPS):
        history.append((env.pos.copy(), env.food_visible()))
        action, hidden, novelty_target = select_action(
            policy,
            obs,
            hidden,
            env,
            "regional_novelty",
            novelty_target,
        )
        result = env.step(action)
        obs = torch.tensor(result.obs).unsqueeze(0)
        if result.ate:
            return {
                "steps": env.steps,
                "region": patch_centroid(env, env.last_pickup_position),
                "anchor": final_visible_anchor(history),
            }
    raise RuntimeError("prime phase failed to produce a pickup")


def find_recall_start(env, region, seed):
    rng = np.random.default_rng(seed ^ 0xA17E)
    original = env.pos.copy()
    try:
        for _ in range(5000):
            candidate = rng.uniform(
                2.0,
                WORLD_SIZE - 2.0,
                size=2,
            ).astype(np.float32)
            distance = float(np.linalg.norm(candidate - region))
            if not RECALL_START_MIN_DISTANCE <= distance <= RECALL_START_MAX_DISTANCE:
                continue
            if env.body_blocked(candidate):
                continue
            env.pos = candidate
            if env.visible_food() is None:
                return candidate.copy()
    finally:
        env.pos = original
    raise RuntimeError("could not place an occluded recall start")


def reset_for_recall(env, start):
    env.reset()
    env.reset_resources()
    env.pos = np.asarray(start, dtype=np.float32).copy()
    env.hunger = 0.82
    env.visits = Counter({env.visit_cell(): 1})
    return env.observe()


def escape_action(env, region, baseline_action):
    away = env.pos - region
    distance = float(np.linalg.norm(away))
    if distance < 1e-6:
        return baseline_action
    direction = away / distance
    scores = []
    safe = env.safe_actions()
    for action, move in enumerate(MOVES):
        if action not in safe:
            scores.append(-math.inf)
            continue
        unit = move.astype(np.float32) / np.linalg.norm(move)
        scores.append(
            float(np.dot(unit, direction))
            + 0.35 * env.novelty_score(action)
            + 0.20 * float(action == baseline_action)
        )
    return int(np.argmax(scores))


def run_episode(policy, condition, seed):
    env = RenewableSparseTerrain(seed, sight=7.0)
    prime = prime_memory(policy, env)
    start = find_recall_start(env, prime["region"], seed)
    memory = PrimedResourceMemory()
    memory.record(prime["region"], prime["anchor"])
    if condition in {"no_memory", "memory_reset"}:
        memory.clear()

    obs = torch.tensor(reset_for_recall(env, start)).unsqueeze(0)
    hidden = policy.initial_state(1)
    novelty_target = None
    active = None
    active_region = None
    active_anchor = None
    recall_started = 0
    anchor_reached_at = None
    escape_until = 0
    previous_action = 0
    reversals = 0
    guidance_steps = 0
    anchor_guidance_steps = 0
    stale_recalls = 0
    pickups_after_recall = 0
    recall_pending = False

    for _ in range(MAX_STEPS):
        visible = env.food_visible()
        if (
            active is None
            and memory.entries
            and env.hunger > MEMORY_HUNGER_GATE
            and not visible
            and env.steps >= escape_until
        ):
            active = memory.recall(env.pos, env.steps)
            if active is not None:
                active_region = active.region.copy()
                active_anchor = active.anchor.copy()
                if condition == "shuffled_anchor_memory":
                    active_region = shuffle_point(active_region)
                    active_anchor = shuffle_point(active_anchor)
                recall_started = env.steps
                anchor_reached_at = None
                recall_pending = True

        if active is not None and visible:
            active = None
            active_region = None
            active_anchor = None
            anchor_reached_at = None

        baseline, hidden, novelty_target = select_action(
            policy,
            obs,
            hidden,
            env,
            "regional_novelty",
            novelty_target,
        )
        action = baseline
        if env.steps < escape_until and active_region is not None:
            action = escape_action(env, active_region, baseline)
        elif active is not None and not visible:
            target = active_region
            weight = 0.72
            if condition in {"anchor_memory", "shuffled_anchor_memory"}:
                target = active_anchor
                weight = 0.90
                anchor_guidance_steps += 1
                if (
                    float(np.linalg.norm(env.pos - active_anchor))
                    <= ANCHOR_RADIUS
                ):
                    if anchor_reached_at is None:
                        anchor_reached_at = env.steps
                    target = active_region
                    weight = 0.78
            action = memory_action(env, baseline, target, weight)
            guidance_steps += 1

            stale_anchor = (
                anchor_reached_at is not None
                and env.steps - anchor_reached_at >= ANCHOR_SEARCH_STEPS
            )
            guidance_expired = (
                env.steps - recall_started >= RECALL_GUIDANCE_STEPS
            )
            if stale_anchor or guidance_expired:
                memory.penalize(active, env.steps)
                stale_recalls += 1
                escape_until = env.steps + STALE_ESCAPE_STEPS
                active = None
                anchor_reached_at = None

        if float(np.dot(MOVES[previous_action], MOVES[action])) < 0.0:
            reversals += 1
        result = env.step(action)
        if result.ate:
            if recall_pending:
                pickups_after_recall += 1
            if memory.entries:
                memory.reward(memory.entries[0])
            active = None
            active_region = None
            active_anchor = None
            anchor_reached_at = None
            recall_pending = False
            novelty_target = None
        previous_action = action
        obs = torch.tensor(result.obs).unsqueeze(0)

    meal_boundaries = [0] + env.pickup_steps + [MAX_STEPS]
    meal_gaps = [
        later - earlier
        for earlier, later in zip(meal_boundaries, meal_boundaries[1:])
    ]
    return {
        "prime_steps": prime["steps"],
        "first_pickup": env.first_pickup_step is not None,
        "first_pickup_step": env.first_pickup_step or MAX_STEPS,
        "pickups": env.pickups,
        "collisions": env.collisions,
        "unique_cells": len(env.visits),
        "revisit_ratio": 1.0 - len(env.visits) / max(1, sum(env.visits.values())),
        "reversals": reversals,
        "max_meal_gap_steps": max(meal_gaps),
        "memory_recalls": memory.recalls,
        "guidance_steps": guidance_steps,
        "anchor_guidance_steps": anchor_guidance_steps,
        "stale_recalls": stale_recalls,
        "pickups_after_recall": pickups_after_recall,
    }


def summarize(rows):
    keys = rows[0].keys()
    return {
        "episodes": len(rows),
        **{
            f"mean_{key}": float(np.mean([row[key] for row in rows]))
            for key in keys
        },
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
            run_episode(policy, condition, 220_003 + episode * 1291)
            for episode in range(episodes)
        ]
        episode_rows[condition] = rows
        results[condition] = summarize(rows)
        print(json.dumps(results[condition], indent=2), flush=True)

    baseline = results["no_memory"]
    coordinate = results["coordinate_memory"]
    anchor = results["anchor_memory"]
    shuffled = results["shuffled_anchor_memory"]
    reset = results["memory_reset"]
    criteria = {
        "coordinate_memory_improves_pickups_10pct": (
            coordinate["mean_pickups"] >= baseline["mean_pickups"] * 1.10
        ),
        "coordinate_memory_reduces_meal_gap_10pct": (
            coordinate["mean_max_meal_gap_steps"]
            <= baseline["mean_max_meal_gap_steps"] * 0.90
        ),
        "coordinate_memory_beats_shuffled_pickups_10pct": (
            coordinate["mean_pickups"] >= shuffled["mean_pickups"] * 1.10
        ),
        "coordinate_memory_is_used": (
            coordinate["mean_guidance_steps"] > 0.0
            and coordinate["mean_pickups_after_recall"] > 0.0
        ),
        "anchor_memory_improves_pickups_10pct": (
            anchor["mean_pickups"] >= baseline["mean_pickups"] * 1.10
        ),
        "anchor_memory_reduces_meal_gap_10pct": (
            anchor["mean_max_meal_gap_steps"]
            <= baseline["mean_max_meal_gap_steps"] * 0.90
        ),
        "anchor_memory_beats_coordinate_pickups": (
            anchor["mean_pickups"] >= coordinate["mean_pickups"]
        ),
        "anchor_memory_beats_coordinate_meal_gap": (
            anchor["mean_max_meal_gap_steps"]
            <= coordinate["mean_max_meal_gap_steps"]
        ),
        "anchor_memory_beats_shuffled_pickups_10pct": (
            anchor["mean_pickups"] >= shuffled["mean_pickups"] * 1.10
        ),
        "memory_reset_matches_baseline": (
            abs(reset["mean_pickups"] - baseline["mean_pickups"]) <= 0.25
        ),
        "no_collision_regression": (
            anchor["mean_collisions"] <= baseline["mean_collisions"] + 1.0
        ),
        "anchor_memory_is_used": (
            anchor["mean_anchor_guidance_steps"] > 0.0
            and anchor["mean_pickups_after_recall"] > 0.0
        ),
    }
    coordinate_checks = [
        "coordinate_memory_improves_pickups_10pct",
        "coordinate_memory_reduces_meal_gap_10pct",
        "coordinate_memory_beats_shuffled_pickups_10pct",
        "coordinate_memory_is_used",
        "memory_reset_matches_baseline",
        "no_collision_regression",
    ]
    anchor_checks = [
        "anchor_memory_improves_pickups_10pct",
        "anchor_memory_reduces_meal_gap_10pct",
        "anchor_memory_beats_coordinate_pickups",
        "anchor_memory_beats_coordinate_meal_gap",
        "anchor_memory_beats_shuffled_pickups_10pct",
        "memory_reset_matches_baseline",
        "no_collision_regression",
        "anchor_memory_is_used",
    ]
    criteria["coordinate_memory_approved"] = (
        all(criteria[name] for name in coordinate_checks) and not args.quick
    )
    criteria["approach_anchor_approved"] = (
        all(criteria[name] for name in anchor_checks) and not args.quick
    )
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "frozen_recurrent_policy": True,
            "phase_1": "near-resource autonomous encoding",
            "phase_2": "distant food-invisible recall",
            "memory_payload": "resource region plus sustained-visibility approach anchor",
            "hunger_gate": MEMORY_HUNGER_GATE,
            "visible_food_overrides_memory": True,
            "stale_recall_escape_steps": STALE_ESCAPE_STEPS,
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
