#!/usr/bin/env python3
"""Test large-scale forage relocation without changing the frozen policy.

The lab separates food-sensor range from regional exploration. It starts the
agent in a food-sparse patch and compares the current critical-hunger policy
against an enlarged sensor, a local visit-novelty objective, and persistent
regional novelty seeking. No condition is given a food coordinate.
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from upgraded_foraging_pipeline import ACTION_NAMES, CORE_DIM, MOVES, RAY_RANGE, load_checkpoint
from unity_posttraining_lab import AGENT_RADIUS, PICKUP_RADIUS, RAY_STEP, STEP_LENGTH, mpc_action


CHECKPOINT = Path("checkpoints/unity_mpc/best.pt")
OUTPUT = Path("outputs/starvation_exploration_metrics.json")
WORLD_SIZE = 64.0
MAX_STEPS = 1800
HUNGER_RISE_PER_EPISODE = 1.2
NORMAL_SIGHT = 7.0
EXTENDED_SIGHT = 13.0
VISIT_CELL_SIZE = 4.0
CONDITIONS = ["current", "radius_only", "novelty", "regional_novelty"]


@dataclass
class StepResult:
    obs: np.ndarray
    ate: bool
    collision: bool
    new_cell: bool


class SparseTerrain:
    """Unity-scaled local physics inside a larger food-sparse world."""

    def __init__(self, seed, sight):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.sight = float(sight)
        self.blocked = self.make_obstacles()
        self.food = self.make_food()
        self.reset()

    def make_obstacles(self):
        blocked = set()
        for _ in range(70):
            x = int(self.rng.integers(2, int(WORLD_SIZE) - 3))
            z = int(self.rng.integers(2, int(WORLD_SIZE) - 3))
            length = int(self.rng.integers(1, 5))
            horizontal = bool(self.rng.integers(0, 2))
            for offset in range(length):
                cell = (x + offset, z) if horizontal else (x, z + offset)
                blocked.add(cell)
        start = np.asarray([49.5, 49.5])
        blocked = {
            cell for cell in blocked
            if np.linalg.norm(np.asarray(cell, dtype=np.float32) + 0.5 - start) > 5.0
        }
        return blocked

    def make_food(self):
        food = []
        while len(food) < 22:
            point = self.rng.uniform(3.0, WORLD_SIZE - 3.0, size=2).astype(np.float32)
            # The southeast start region deliberately contains no food.
            if point[0] > 38.0 and point[1] > 38.0:
                continue
            if self.point_blocked(point):
                continue
            if all(np.linalg.norm(point - existing) > 2.0 for existing in food):
                food.append(point)
        return food

    def reset(self):
        self.pos = np.asarray([49.5, 49.5], dtype=np.float32)
        self.steps = 0
        self.hunger = 0.76
        self.last_action = 0
        self.last_reward = 0.0
        self.collisions = 0
        self.pickups = 0
        self.path_length = 0.0
        self.first_pickup_step = None
        self.pickup_steps = []
        self.visits = Counter({self.visit_cell(): 1})
        return self.observe()

    def visit_cell(self, position=None):
        point = self.pos if position is None else position
        return tuple(np.floor(np.asarray(point) / VISIT_CELL_SIZE).astype(int))

    def point_blocked(self, point):
        x, z = float(point[0]), float(point[1])
        if x < 0.0 or z < 0.0 or x >= WORLD_SIZE or z >= WORLD_SIZE:
            return True
        return (int(math.floor(x)), int(math.floor(z))) in self.blocked

    def body_blocked(self, point):
        x, z = float(point[0]), float(point[1])
        if x < AGENT_RADIUS or z < AGENT_RADIUS:
            return True
        if x > WORLD_SIZE - AGENT_RADIUS or z > WORLD_SIZE - AGENT_RADIUS:
            return True
        x0, x1 = int(math.floor(x - AGENT_RADIUS)), int(math.floor(x + AGENT_RADIUS))
        z0, z1 = int(math.floor(z - AGENT_RADIUS)), int(math.floor(z + AGENT_RADIUS))
        for cell_x in range(max(0, x0), min(int(WORLD_SIZE) - 1, x1) + 1):
            for cell_z in range(max(0, z0), min(int(WORLD_SIZE) - 1, z1) + 1):
                if (cell_x, cell_z) not in self.blocked:
                    continue
                nearest_x = min(max(x, cell_x), cell_x + 1.0)
                nearest_z = min(max(z, cell_z), cell_z + 1.0)
                if math.hypot(x - nearest_x, z - nearest_z) < AGENT_RADIUS:
                    return True
        return False

    def ray_distance(self, move):
        direction = move.astype(np.float32) / np.linalg.norm(move)
        for distance in np.arange(RAY_STEP, RAY_RANGE + RAY_STEP, RAY_STEP):
            if self.point_blocked(self.pos + direction * distance):
                return float(np.clip((distance - RAY_STEP) / RAY_RANGE, 0.0, 1.0))
        return 1.0

    def visible_food(self):
        candidates = []
        for index, point in enumerate(self.food):
            distance = float(np.linalg.norm(point - self.pos))
            if distance > self.sight:
                continue
            samples = max(2, int(math.ceil(distance / RAY_STEP)))
            occluded = any(
                self.point_blocked(self.pos + alpha * (point - self.pos))
                for alpha in np.linspace(0.0, 1.0, samples)[1:-1]
            )
            if not occluded:
                candidates.append((distance, index, point))
        return min(candidates, default=None, key=lambda item: item[0])

    def food_visible(self):
        return self.visible_food() is not None

    def observe(self):
        rays = np.asarray([self.ray_distance(move) for move in MOVES], dtype=np.float32)
        visible = self.visible_food()
        food_delta = np.zeros(2, dtype=np.float32)
        if visible is not None:
            food_delta = np.clip((visible[2] - self.pos) / 14.0, -1.0, 1.0)
        previous = np.zeros(len(MOVES), dtype=np.float32)
        previous[self.last_action] = 1.0
        return np.concatenate(
            [rays, [float(visible is not None)], food_delta, [self.hunger], previous, [self.last_reward]]
        ).astype(np.float32)

    def step(self, action):
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
        self.hunger = min(1.0, self.hunger + HUNGER_RISE_PER_EPISODE / MAX_STEPS)
        self.last_action = action
        ate = False
        for index, point in enumerate(self.food):
            if float(np.linalg.norm(point - self.pos)) <= PICKUP_RADIUS:
                ate = True
                self.pickups += 1
                if self.first_pickup_step is None:
                    self.first_pickup_step = self.steps
                self.pickup_steps.append(self.steps)
                self.last_reward = 2.0
                replacement = self.make_replacement_food()
                self.food[index] = replacement
                self.hunger = 0.35
                break
        if not ate:
            self.last_reward = -0.004 - (0.24 if collision else 0.0)
        return StepResult(self.observe(), ate, collision, new_cell)

    def make_replacement_food(self):
        for _ in range(1000):
            point = self.rng.uniform(3.0, WORLD_SIZE - 3.0, size=2).astype(np.float32)
            if point[0] > 38.0 and point[1] > 38.0:
                continue
            if not self.point_blocked(point):
                return point
        raise RuntimeError("could not place replacement food")

    def safe_actions(self):
        safe = []
        for action, move in enumerate(MOVES):
            direction = move.astype(np.float32) / np.linalg.norm(move)
            if not self.body_blocked(self.pos + direction * STEP_LENGTH):
                safe.append(action)
        return safe or [int(np.argmax(self.observe()[: len(MOVES)]))]

    def novelty_score(self, action):
        direction = MOVES[action].astype(np.float32)
        direction /= np.linalg.norm(direction)
        weighted_visits = 0.0
        for distance, weight in ((4.0, 0.25), (8.0, 0.35), (12.0, 0.40)):
            projected = np.clip(self.pos + direction * distance, 0.0, WORLD_SIZE - 0.01)
            weighted_visits += weight * self.visits.get(self.visit_cell(projected), 0)
        return 1.0 / math.sqrt(1.0 + weighted_visits)

    def choose_exploration_target(self):
        """Choose an under-visited coarse region without consulting food."""
        candidates = []
        for x in np.arange(6.0, WORLD_SIZE - 5.0, 8.0):
            for z in np.arange(6.0, WORLD_SIZE - 5.0, 8.0):
                target = np.asarray([x, z], dtype=np.float32)
                if self.point_blocked(target):
                    continue
                visits = self.visits.get(self.visit_cell(target), 0)
                distance = float(np.linalg.norm(target - self.pos))
                if distance < 8.0:
                    continue
                # Prefer genuinely unvisited regions, with distance only breaking ties.
                score = -4.0 * visits + min(distance, 32.0) / 32.0
                candidates.append((score, distance, target))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


def select_action(policy, obs, hidden, env, condition, exploration_target):
    visible = env.food_visible()
    novelty_active = condition in {"novelty", "regional_novelty"} and not visible and env.hunger >= 0.70
    if visible or (env.hunger >= 0.92 and not novelty_active):
        action, next_hidden = mpc_action(policy, obs, hidden, env, horizon=6)
        return action, next_hidden, exploration_target

    with torch.no_grad():
        logits, _value, next_hidden = policy.step(obs, hidden)
        safe = env.safe_actions()
        mask = torch.full_like(logits, -1e9)
        mask[0, safe] = 0.0
        logits = logits + mask
        probabilities = torch.softmax(logits, dim=-1)[0]
        recurrent = int(torch.argmax(probabilities))

    if not novelty_active:
        return recurrent, next_hidden, exploration_target

    if condition == "regional_novelty":
        if exploration_target is None or float(np.linalg.norm(exploration_target - env.pos)) < 5.0:
            exploration_target = env.choose_exploration_target()
        if exploration_target is not None:
            target_direction = exploration_target - env.pos
            target_direction /= max(1e-6, float(np.linalg.norm(target_direction)))
            scores = []
            for action in range(len(MOVES)):
                if action not in safe:
                    scores.append(-math.inf)
                    continue
                move = MOVES[action].astype(np.float32)
                move /= np.linalg.norm(move)
                alignment = float(np.dot(move, target_direction))
                prior = 0.04 * math.log(max(1e-8, float(probabilities[action])))
                scores.append(1.6 * alignment + 0.35 * env.novelty_score(action) + prior)
            return int(np.argmax(scores)), next_hidden, exploration_target

    previous = env.last_action
    scores = []
    for action in range(len(MOVES)):
        if action not in safe:
            scores.append(-math.inf)
            continue
        policy_prior = 0.10 * math.log(max(1e-8, float(probabilities[action])))
        novelty = env.novelty_score(action)
        cosine = float(np.dot(MOVES[previous], MOVES[action])) / (
            float(np.linalg.norm(MOVES[previous])) * float(np.linalg.norm(MOVES[action]))
        )
        turn_cost = 0.08 * 0.5 * (1.0 - float(np.clip(cosine, -1.0, 1.0)))
        scores.append(policy_prior + 2.0 * novelty - turn_cost)
    selected = int(np.argmax(scores))
    if float(np.dot(MOVES[previous], MOVES[selected])) < 0.0:
        non_reversing = [
            action for action in safe
            if float(np.dot(MOVES[previous], MOVES[action])) >= 0.0
        ]
        if non_reversing:
            selected = max(non_reversing, key=lambda action: scores[action])
    return selected, next_hidden, exploration_target


def run_episode(policy, condition, seed):
    sight = EXTENDED_SIGHT if condition == "radius_only" else NORMAL_SIGHT
    env = SparseTerrain(seed, sight)
    obs = torch.tensor(env.reset()).unsqueeze(0)
    hidden = policy.initial_state(1)
    exploration_target = None
    reversals = 0
    previous_action = 0

    for _step in range(MAX_STEPS):
        action, hidden, exploration_target = select_action(
            policy, obs, hidden, env, condition, exploration_target
        )
        if float(np.dot(MOVES[previous_action], MOVES[action])) < 0.0:
            reversals += 1
        result = env.step(action)
        previous_action = action
        obs = torch.tensor(result.obs).unsqueeze(0)

    total_visits = sum(env.visits.values())
    meal_boundaries = [0] + env.pickup_steps + [MAX_STEPS]
    meal_gaps = [later - earlier for earlier, later in zip(meal_boundaries, meal_boundaries[1:])]
    return {
        "first_pickup": env.first_pickup_step is not None,
        "first_pickup_step": env.first_pickup_step or MAX_STEPS,
        "pickups": env.pickups,
        "collisions": env.collisions,
        "unique_cells": len(env.visits),
        "revisit_ratio": 1.0 - len(env.visits) / max(1, total_visits),
        "path_length": env.path_length,
        "reversals": reversals,
        "max_meal_gap_steps": max(meal_gaps),
    }


def summarize(rows):
    first_steps = [row["first_pickup_step"] for row in rows if row["first_pickup"]]
    return {
        "episodes": len(rows),
        "first_pickup_success_rate": float(np.mean([row["first_pickup"] for row in rows])),
        "mean_first_pickup_steps_on_success": float(np.mean(first_steps)) if first_steps else MAX_STEPS,
        "mean_pickups": float(np.mean([row["pickups"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_unique_cells": float(np.mean([row["unique_cells"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "mean_path_length": float(np.mean([row["path_length"] for row in rows])),
        "mean_reversals": float(np.mean([row["reversals"] for row in rows])),
        "mean_max_meal_gap_steps": float(np.mean([row["max_meal_gap_steps"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    policy, _payload = load_checkpoint(args.checkpoint)
    policy.eval()
    episodes = 4 if args.quick else max(4, args.episodes)
    results = {}
    for condition in CONDITIONS:
        print(f"evaluating {condition}", flush=True)
        rows = [run_episode(policy, condition, 90_001 + episode * 997) for episode in range(episodes)]
        results[condition] = summarize(rows)
        print(json.dumps(results[condition], indent=2), flush=True)

    current = results["current"]
    radius = results["radius_only"]
    candidate = results["novelty"]
    criteria = {
        "radius_alone_does_not_solve_coverage": radius["first_pickup_success_rate"] <= current["first_pickup_success_rate"] + 0.20,
        "local_novelty_preserves_pickups": candidate["mean_pickups"] >= current["mean_pickups"],
        "local_novelty_reduces_longest_gap_10pct": candidate["mean_max_meal_gap_steps"] < current["mean_max_meal_gap_steps"] * 0.90,
        "local_novelty_preserves_collision_safety": candidate["mean_collisions"] <= current["mean_collisions"] + 1.0,
        "local_novelty_avoids_excess_reversals": candidate["mean_reversals"] <= current["mean_reversals"] * 1.25,
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "world_size": WORLD_SIZE,
            "normal_sight": NORMAL_SIGHT,
            "extended_sight": EXTENDED_SIGHT,
            "hunger_rise_per_episode": HUNGER_RISE_PER_EPISODE,
            "food_coordinates_hidden_from_exploration": True,
            "checkpoint_frozen": True,
            "selected_candidate": "hunger_weighted_local_novelty",
        },
        "conditions": results,
        "criteria": criteria,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"criteria": criteria, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
