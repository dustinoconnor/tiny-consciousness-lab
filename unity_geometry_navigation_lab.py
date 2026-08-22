#!/usr/bin/env python3
"""Unity-scale hidden-goal navigation and recurrent post-training lab.

The earlier grid approximation was too forgiving: the frozen policy solved it
even though it timed out in Unity. This lab uses the measured trap-course deck,
walls, body radius, ray range, movement stride, and episode horizon. A map-aware
teacher is used only during training; evaluation receives the normal 21-value
sensor packet and no privileged geometry or hidden goal coordinates.
"""

import argparse
import copy
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unity_posttraining_lab import ALL_COURSES, evaluate_continuous
from upgraded_foraging_pipeline import MOVES, load_checkpoint


BASE = Path("checkpoints/unity_mpc/best.pt")
CHECKPOINT = Path("checkpoints/unity_geometry_posttrained/best.pt")
METRICS = Path("outputs/unity_geometry_navigation_metrics.json")
DECK = (34.0, 38.0)
WALL_THICKNESS = 0.75
RAY_RANGE = 6.0
RAY_STEP = 0.20
FOOD_SIGHT = 7.0
STEP_LENGTH = 0.62
BODY_RADIUS = 0.48
PICKUP_RADIUS = 1.05
MAX_STEPS = 450
GRID = 0.50


COURSES = {
    "utrap": {
        "walls": [(0.0, 3.0, 6.0, WALL_THICKNESS), (-2.65, 0.0, WALL_THICKNESS, 6.0), (2.65, 0.0, WALL_THICKNESS, 6.0)],
        "start": (0.0, -1.9),
        "food": (0.0, 5.5),
    },
    "ctrap": {
        "walls": [(-4.5, 0.0, WALL_THICKNESS, 11.0), (0.0, 5.1, 9.0, WALL_THICKNESS), (0.0, -5.1, 9.0, WALL_THICKNESS)],
        "start": (-1.5, 0.0),
        "food": (-8.0, 0.0),
    },
}


def expanded_contains(point, rectangle, padding=0.0):
    x, z = float(point[0]), float(point[1])
    cx, cz, width, depth = rectangle
    return abs(x - cx) <= width * 0.5 + padding and abs(z - cz) <= depth * 0.5 + padding


class UnityGeometryCourse:
    def __init__(self, family, seed=0, food_reward=2.0):
        self.family = family
        self.seed = int(seed)
        self.food_reward = float(food_reward)
        spec = COURSES[family]
        self.walls = list(spec["walls"])
        self.start = np.asarray(spec["start"], dtype=np.float32)
        self.food = np.asarray(spec["food"], dtype=np.float32)
        self.distance_map = self.build_distance_map()
        self.reset()

    def reset(self):
        rng = np.random.default_rng(self.seed)
        self.pos = self.start + rng.uniform(-0.12, 0.12, size=2).astype(np.float32)
        self.steps = 0
        self.hunger = 0.25
        self.last_action = 0
        self.last_reward = 0.0
        self.collisions = 0
        self.path_length = 0.0
        self.visits = Counter({self.visit_cell(): 1})
        return self.observe()

    @staticmethod
    def bounds_blocked(point, padding=0.0):
        return abs(float(point[0])) >= DECK[0] * 0.5 - 0.325 - padding or abs(float(point[1])) >= DECK[1] * 0.5 - 0.325 - padding

    def point_blocked(self, point):
        return self.bounds_blocked(point) or any(expanded_contains(point, wall) for wall in self.walls)

    def body_blocked(self, point):
        return self.bounds_blocked(point, BODY_RADIUS) or any(
            expanded_contains(point, wall, BODY_RADIUS) for wall in self.walls
        )

    def ray_distance(self, move):
        direction = move.astype(np.float32) / np.linalg.norm(move)
        for distance in np.arange(RAY_STEP, RAY_RANGE + RAY_STEP, RAY_STEP):
            if self.point_blocked(self.pos + direction * distance):
                return float(np.clip((distance - RAY_STEP) / RAY_RANGE, 0.0, 1.0))
        return 1.0

    def food_visible(self):
        delta = self.food - self.pos
        distance = float(np.linalg.norm(delta))
        if distance > FOOD_SIGHT:
            return False
        for alpha in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / RAY_STEP))))[1:-1]:
            if self.point_blocked(self.pos + alpha * delta):
                return False
        return True

    def observe(self):
        rays = np.asarray([self.ray_distance(move) for move in MOVES], dtype=np.float32)
        visible = self.food_visible()
        food_delta = (self.food - self.pos) / 14.0 if visible else np.zeros(2, dtype=np.float32)
        previous = np.zeros(len(MOVES), dtype=np.float32)
        previous[self.last_action] = 1.0
        return np.concatenate([rays, [float(visible)], food_delta, [self.hunger], previous, [self.last_reward]]).astype(np.float32)

    def visit_cell(self, point=None):
        point = self.pos if point is None else point
        return int(round(float(point[0]) / GRID)), int(round(float(point[1]) / GRID))

    def grid_point(self, cell):
        return np.asarray(cell, dtype=np.float32) * GRID

    def build_distance_map(self):
        goal = self.visit_cell(self.food)
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for move in MOVES:
                previous = cell[0] - int(move[0]), cell[1] - int(move[1])
                if previous in distances or self.body_blocked(self.grid_point(previous)):
                    continue
                if abs(previous[0] * GRID) > DECK[0] * 0.5 or abs(previous[1] * GRID) > DECK[1] * 0.5:
                    continue
                distances[previous] = distances[cell] + 1
                queue.append(previous)
        return distances

    def teacher_action(self):
        choices = []
        for action, move in enumerate(MOVES):
            direction = move.astype(np.float32) / np.linalg.norm(move)
            candidate = self.pos + direction * STEP_LENGTH
            if self.body_blocked(candidate):
                continue
            cell = self.visit_cell(candidate)
            distance = self.distance_map.get(cell)
            if distance is None:
                continue
            previous = MOVES[self.last_action]
            cosine = float(np.dot(previous, move) / (np.linalg.norm(previous) * np.linalg.norm(move)))
            choices.append((distance + 0.08 * (1.0 - cosine), action))
        return min(choices)[1] if choices else int(np.argmax(self.observe()[:8]))

    def step(self, action):
        action = int(action)
        direction = MOVES[action].astype(np.float32) / np.linalg.norm(MOVES[action])
        candidate = self.pos + direction * STEP_LENGTH
        collision = self.body_blocked(candidate)
        old_distance = float(np.linalg.norm(self.food - self.pos))
        previous_action = self.last_action
        if collision:
            self.collisions += 1
        else:
            self.pos = candidate
            self.path_length += STEP_LENGTH
        cell = self.visit_cell()
        novel = cell not in self.visits
        self.visits[cell] += 1
        new_distance = float(np.linalg.norm(self.food - self.pos))
        visible = self.food_visible()
        ate = new_distance <= PICKUP_RADIUS
        cosine = float(np.dot(MOVES[previous_action], MOVES[action]) / (np.linalg.norm(MOVES[previous_action]) * np.linalg.norm(MOVES[action])))
        reward = -0.004 + (0.012 if novel else -0.004 * math.sqrt(self.visits[cell] - 1))
        reward -= 0.010 * 0.5 * (1.0 - float(np.clip(cosine, -1.0, 1.0)))
        if collision:
            reward -= 0.25
        if visible and not collision:
            reward += 0.06 * (old_distance - new_distance)
        if ate:
            reward += self.food_reward
        self.steps += 1
        self.hunger = min(1.0, self.hunger + 0.75 / MAX_STEPS)
        self.last_action = action
        self.last_reward = reward
        return self.observe(), reward, ate or self.steps >= MAX_STEPS, ate, collision


def mask_logits(logits, envs):
    blocked = []
    for env in envs:
        row = []
        for move in MOVES:
            direction = move.astype(np.float32) / np.linalg.norm(move)
            row.append(env.body_blocked(env.pos + direction * STEP_LENGTH))
        if all(row):
            row[int(np.argmax(env.observe()[:8]))] = False
        blocked.append(row)
    return logits.masked_fill(torch.tensor(blocked, dtype=torch.bool), -1e9)


def evaluate(policy, episodes=20, reset_memory=False):
    rows = []
    for family in COURSES:
        for seed in range(episodes):
            env = UnityGeometryCourse(family, 10_000 + seed)
            obs = torch.tensor(env.reset()).unsqueeze(0)
            hidden = policy.initial_state(1)
            for step in range(MAX_STEPS):
                with torch.no_grad():
                    logits, _, hidden = policy.step(obs, hidden)
                    action = int(torch.argmax(mask_logits(logits, [env]), dim=-1))
                obs_array, _, done, ate, _ = env.step(action)
                obs = torch.tensor(obs_array).unsqueeze(0)
                if reset_memory:
                    hidden = policy.initial_state(1)
                if done:
                    rows.append({"family": family, "success": int(ate), "steps": step + 1, "collisions": env.collisions, "revisit_ratio": 1.0 - len(env.visits) / max(1, sum(env.visits.values()))})
                    break
    by_family = {}
    for family in COURSES:
        subset = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "mean_steps": float(np.mean([row["steps"] for row in subset])),
            "mean_collisions": float(np.mean([row["collisions"] for row in subset])),
        }
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "by_family": by_family,
    }


def train_candidate(base, mode, updates, seed, env_count=16, rollout=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy, payload = load_checkpoint(base)
    reference = copy.deepcopy(policy).eval()
    reference_state = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    for parameter in policy.parameters():
        parameter.requires_grad = False
    trainable = [policy.actor] if mode == "readout_only" else [policy.encoder, policy.memory, policy.actor]
    for module in trainable:
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=3e-4 if mode == "readout_only" else 1.2e-4,
        weight_decay=0.01,
    )
    envs = [UnityGeometryCourse("utrap" if index % 2 == 0 else "ctrap", int(rng.integers(1_000_000))) for index in range(env_count)]
    obs = torch.tensor(np.stack([env.reset() for env in envs]))
    hidden = policy.initial_state(env_count)
    best_state = copy.deepcopy(policy.state_dict())
    best_score = (-1.0, -1e9)
    curve = []

    for update in range(updates):
        losses = []
        teacher_probability = 0.90 - 0.45 * update / max(1, updates - 1)
        for _ in range(rollout):
            logits, _, hidden = policy.step(obs, hidden)
            logits = mask_logits(logits, envs)
            teacher = torch.tensor([env.teacher_action() for env in envs], dtype=torch.long)
            losses.append(F.cross_entropy(logits, teacher))
            sampled = torch.distributions.Categorical(logits=logits).sample()
            actions = torch.where(torch.rand(env_count) < teacher_probability, teacher, sampled)
            next_obs, alive = [], []
            for index, env in enumerate(envs):
                obs_array, _, done, _, _ = env.step(int(actions[index]))
                if done:
                    family = "utrap" if rng.random() < 0.5 else "ctrap"
                    envs[index] = UnityGeometryCourse(family, int(rng.integers(1_000_000)))
                    next_obs.append(envs[index].reset())
                    alive.append(0.0)
                else:
                    next_obs.append(obs_array)
                    alive.append(1.0)
            obs = torch.tensor(np.stack(next_obs))
            hidden = hidden * torch.tensor(alive).unsqueeze(-1)
        imitation = torch.stack(losses).mean()
        anchor = torch.tensor(0.0)
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                anchor = anchor + torch.mean((parameter - reference_state[name]) ** 2)
        loss = imitation + 0.015 * anchor
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
        optimizer.step()
        hidden = hidden.detach()

        if update % 25 == 0 or update == updates - 1:
            policy.eval()
            validation = evaluate(policy, episodes=4)
            policy.train()
            score = (validation["success_rate"], -validation["mean_steps"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(policy.state_dict())
            curve.append({"update": update, "loss": float(loss.detach()), **validation})
            print(f"{mode} update={update:03d} success={validation['success_rate']:.3f} steps={validation['mean_steps']:.1f} loss={float(loss.detach()):.3f}", flush=True)
    policy.load_state_dict(best_state)
    return policy.eval(), payload, curve


def existing_course_regression(policy, episodes_per_seed=3):
    return evaluate_continuous(
        policy,
        ALL_COURSES,
        [31, 47],
        episodes_per_seed=episodes_per_seed,
    )


def save_checkpoint(policy, payload, mode, metrics):
    exported = dict(payload)
    exported["state_dict"] = policy.state_dict()
    exported["config"] = dict(payload["config"])
    exported["config"].update({
        "posttraining": "unity_geometry_navigation_v1",
        "posttraining_scope": mode,
        "privileged_teacher_removed_at_evaluation": True,
    })
    exported["unity_geometry_navigation_metrics"] = metrics
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--readout-updates", type=int, default=150)
    parser.add_argument("--recurrent-updates", type=int, default=240)
    args = parser.parse_args()
    policy, _ = load_checkpoint(args.base)
    episodes = 3 if args.quick else 20
    baseline = evaluate(policy, episodes)
    reset = evaluate(policy, episodes, reset_memory=True)
    print(json.dumps({"baseline": baseline, "memory_reset": reset}, indent=2))
    if args.evaluate_only:
        return
    if args.quick:
        args.readout_updates = 20
        args.recurrent_updates = 30
    existing_baseline = existing_course_regression(policy, 1 if args.quick else 3)
    readout, payload, readout_curve = train_candidate(args.base, "readout_only", args.readout_updates, 4101, env_count=8 if args.quick else 16, rollout=16 if args.quick else 32)
    readout_result = evaluate(readout, episodes)
    readout_existing = existing_course_regression(readout, 1 if args.quick else 3)
    candidates = [("readout_only", readout, readout_result, readout_existing, readout_curve)]
    if readout_result["success_rate"] < 0.90:
        recurrent, payload, recurrent_curve = train_candidate(args.base, "recurrent", args.recurrent_updates, 4201, env_count=8 if args.quick else 16, rollout=16 if args.quick else 32)
        recurrent_result = evaluate(recurrent, episodes)
        recurrent_existing = existing_course_regression(recurrent, 1 if args.quick else 3)
        candidates.append(("recurrent", recurrent, recurrent_result, recurrent_existing, recurrent_curve))
    eligible = [item for item in candidates if item[2]["success_rate"] >= 0.90 and item[3]["success_rate"] >= existing_baseline["success_rate"] - 0.03]
    chosen = min(eligible, key=lambda item: 0 if item[0] == "readout_only" else 1) if eligible else max(candidates, key=lambda item: (item[2]["success_rate"], item[3]["success_rate"]))
    mode, chosen_policy, result, existing, _ = chosen
    metrics = {
        "baseline": baseline,
        "memory_reset": reset,
        "existing_course_baseline": existing_baseline,
        "candidates": {item[0]: {"unity_geometry": item[2], "existing_courses": item[3], "curve": item[4]} for item in candidates},
        "selected": mode,
        "accepted": result["success_rate"] >= 0.90 and existing["success_rate"] >= existing_baseline["success_rate"] - 0.03,
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    if metrics["accepted"] and not args.quick:
        save_checkpoint(chosen_policy, payload, mode, metrics)
        print(f"exported {CHECKPOINT}")
    else:
        print("candidate rejected; no checkpoint exported")


if __name__ == "__main__":
    main()
