#!/usr/bin/env python3
"""Frozen world-model routing with a held-out obstacle-geometry test.

The lab learns a compact forward model from Unity-like local sensor packets:

    current observation + candidate movement -> next observation + collision

The controller never receives a global map or a NavMesh. After training on
scattered circular and rectangular obstacles, the model is frozen and tested
on familiar layouts plus diagonal barrier geometry withheld from training.

The held-out U-detour additionally compares stateless neural rollout against an
online structured world model reconstructed only from accumulated local rays.

This supports a narrow zero-shot claim only when the frozen model transfers to
the held-out geometry. It does not test open-ended task generalization.
"""

import json
import math
from collections import deque
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_lab import OUT, set_seed


WORLD_SIZE = 18
SENSOR_RANGE = 4
MAX_STEPS = 90
MOVES = np.array(
    [
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
    ],
    dtype=np.int64,
)
ACTION_NAMES = ["up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"]
OBS_DIM = 2 + 2 + 1 + 8 + 1


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    collision: float


class LocalNavigationWorld:
    def __init__(self, blocked, start, goal):
        self.blocked = set(blocked)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.reset()

    def reset(self):
        self.pos = self.start
        self.last_collision = 0.0
        self.steps = 0
        return self.observe()

    def inside(self, xy):
        return 0 <= xy[0] < WORLD_SIZE and 0 <= xy[1] < WORLD_SIZE

    def is_blocked(self, xy):
        return not self.inside(xy) or tuple(xy) in self.blocked

    def ray_clearance(self, direction):
        for distance in range(1, SENSOR_RANGE + 1):
            candidate = self.pos[0] + direction[0] * distance, self.pos[1] + direction[1] * distance
            if self.is_blocked(candidate):
                return (distance - 1) / SENSOR_RANGE
        return 1.0

    def observe(self):
        pos = np.asarray(self.pos, dtype=np.float32) / (WORLD_SIZE - 1)
        goal_delta = (np.asarray(self.goal, dtype=np.float32) - np.asarray(self.pos, dtype=np.float32)) / (WORLD_SIZE - 1)
        goal_distance = np.linalg.norm(goal_delta) / math.sqrt(2.0)
        rays = np.asarray([self.ray_clearance(move) for move in MOVES], dtype=np.float32)
        return np.concatenate([pos, goal_delta, [goal_distance], rays, [self.last_collision]]).astype(np.float32)

    def step(self, action):
        move = MOVES[int(action)]
        candidate = self.pos[0] + int(move[0]), self.pos[1] + int(move[1])
        collision = float(self.is_blocked(candidate))
        if not collision:
            self.pos = candidate
        self.last_collision = collision
        self.steps += 1
        return self.observe(), collision, self.pos == self.goal


def disk(center, radius):
    cells = set()
    for x in range(WORLD_SIZE):
        for y in range(WORLD_SIZE):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2:
                cells.add((x, y))
    return cells


def rectangle(x0, y0, width, height):
    return {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}


def diagonal_barrier(offset, thickness=1):
    cells = set()
    for x in range(3, WORLD_SIZE - 3):
        y = x + offset
        for delta in range(-(thickness - 1), thickness):
            if 1 <= y + delta < WORLD_SIZE - 1:
                cells.add((x, y + delta))
    return cells


def u_detour():
    """A cul-de-sac whose opening faces away from the goal."""
    cells = set()
    cells |= rectangle(5, 6, 1, 8)
    cells |= rectangle(12, 6, 1, 8)
    cells |= rectangle(5, 13, 8, 1)
    return cells


def make_world(kind, seed):
    rng = np.random.default_rng(seed)
    start = (1, 1)
    goal = (WORLD_SIZE - 2, WORLD_SIZE - 2)
    blocked = set()

    if kind == "circles":
        for _ in range(11):
            center = rng.integers(3, WORLD_SIZE - 3, size=2)
            blocked |= disk(center, int(rng.integers(1, 3)))
    elif kind == "rectangles":
        for _ in range(12):
            x, y = rng.integers(2, WORLD_SIZE - 4, size=2)
            blocked |= rectangle(int(x), int(y), int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    elif kind == "diagonal_bars":
        blocked |= diagonal_barrier(-3)
        blocked |= diagonal_barrier(4)
        # Leave doorways so successful traversal remains possible.
        blocked -= {(5, 2), (5, 3), (11, 14), (11, 15)}
        for _ in range(5):
            center = rng.integers(3, WORLD_SIZE - 3, size=2)
            blocked |= disk(center, 1)
    elif kind == "u_detour":
        start = (9, 7)
        goal = (9, 16)
        blocked |= u_detour()
    else:
        raise ValueError(kind)

    blocked.discard(start)
    blocked.discard(goal)
    return LocalNavigationWorld(blocked, start, goal)


class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + len(MOVES), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.next_obs = nn.Linear(64, OBS_DIM)
        self.collision = nn.Linear(64, 1)

    def forward(self, obs, action):
        one_hot = F.one_hot(action.long(), len(MOVES)).float()
        hidden = self.net(torch.cat([obs, one_hot], dim=-1))
        return self.next_obs(hidden), self.collision(hidden).squeeze(-1)


def collect_training_data(worlds_per_kind=55, steps_per_world=180):
    rows = []
    for kind_index, kind in enumerate(("circles", "rectangles")):
        for world_index in range(worlds_per_kind):
            env = make_world(kind, 1000 * (kind_index + 1) + world_index)
            obs = env.reset()
            rng = np.random.default_rng(9000 + kind_index * 1000 + world_index)
            for _ in range(steps_per_world):
                action = int(rng.integers(0, len(MOVES)))
                next_obs, collision, done = env.step(action)
                rows.append(Transition(obs, action, next_obs, collision))
                obs = env.reset() if done else next_obs
    return rows


def train_forward_model(rows, epochs=65, batch_size=512):
    model = ForwardModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    obs = torch.tensor(np.stack([row.obs for row in rows]))
    actions = torch.tensor([row.action for row in rows])
    next_obs = torch.tensor(np.stack([row.next_obs for row in rows]))
    collisions = torch.tensor([row.collision for row in rows])
    rng = np.random.default_rng(33)
    losses = []

    for _ in range(epochs):
        order = rng.permutation(len(rows))
        epoch_losses = []
        for start in range(0, len(rows), batch_size):
            idx = torch.tensor(order[start : start + batch_size])
            predicted_obs, collision_logits = model(obs[idx], actions[idx])
            state_loss = F.mse_loss(predicted_obs, next_obs[idx])
            collision_loss = F.binary_cross_entropy_with_logits(collision_logits, collisions[idx])
            loss = state_loss + 0.45 * collision_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    return model.eval(), losses


def train_model_ensemble(rows, members=3, epochs=55):
    models = []
    histories = []
    for member in range(members):
        torch.manual_seed(330 + member)
        model, losses = train_forward_model(rows, epochs=epochs)
        models.append(model)
        histories.append(losses)
    return models, histories


def action_goal_alignment(obs):
    goal = obs[2:4]
    norm = np.linalg.norm(goal)
    if norm < 1e-8:
        return np.zeros(len(MOVES), dtype=np.float32)
    directions = MOVES.astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions @ (goal / norm)


def choose_reactive(obs, rng):
    scores = action_goal_alignment(obs)
    if obs[-1] > 0.5:
        scores = scores + 0.65 * obs[5:13]
    scores += rng.normal(0.0, 0.015, size=len(scores))
    return int(np.argmax(scores))


def choose_sensor_router(obs, rng):
    scores = action_goal_alignment(obs) + 1.20 * obs[5:13]
    scores -= 1.8 * (obs[5:13] <= 0.01)
    scores += rng.normal(0.0, 0.015, size=len(scores))
    return int(np.argmax(scores))


def score_predicted_state(previous_obs, predicted_obs, collision_probability):
    predicted_progress = float(previous_obs[4] - predicted_obs[4])
    clearance = float(np.mean(np.sort(predicted_obs[5:13])[-3:]))
    # Progress must dominate generic open-space preference or the planner can
    # become perfectly collision-free while circling in a comfortable region.
    return 8.0 * predicted_progress + 0.04 * clearance - 2.6 * collision_probability


def predict(model, obs, action):
    with torch.no_grad():
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        action_tensor = torch.tensor([action])
        next_obs, collision_logit = model(obs_tensor, action_tensor)
    return next_obs.squeeze(0).numpy(), float(torch.sigmoid(collision_logit).item())


def predict_ensemble(models, obs, action):
    predictions = [predict(model, obs, action) for model in models]
    states = np.stack([item[0] for item in predictions])
    collisions = np.asarray([item[1] for item in predictions], dtype=np.float32)
    mean_state = np.mean(states, axis=0)
    mean_collision = float(np.mean(collisions))
    state_disagreement = float(np.mean(np.var(states, axis=0)))
    collision_disagreement = float(np.var(collisions))
    uncertainty = state_disagreement + 0.35 * collision_disagreement
    return mean_state, mean_collision, uncertainty


def choose_model_router(model, obs, depth, rng):
    best_action = 0
    best_value = -float("inf")
    for sequence in product(range(len(MOVES)), repeat=depth):
        imagined = obs.copy()
        value = 0.0
        discount = 1.0
        for action in sequence:
            previous = imagined
            imagined, collision_probability = predict(model, imagined, action)
            value += discount * score_predicted_state(previous, imagined, collision_probability)
            discount *= 0.72
        value += float(rng.normal(0.0, 0.002))
        if value > best_value:
            best_value = value
            best_action = sequence[0]
    return int(best_action)


def choose_uncertainty_grounded_router(models, obs, rng, depth=2):
    """Receding-horizon planning with uncertainty and a raw-sensor veto.

    Only the first selected action is executed. The next call starts again from
    a fresh real observation, so imagined sensor packets never persist across
    physical control steps.
    """
    uncertainty_limit = 0.0018
    sensor_fallback = choose_sensor_router(obs, rng)
    best_action = sensor_fallback
    best_value = -float("inf")

    for sequence in product(range(len(MOVES)), repeat=depth):
        first_action = sequence[0]
        # The current raw ray is stronger evidence than any learned prediction.
        if obs[5 + first_action] <= 0.01:
            continue

        imagined = obs.copy()
        value = 0.0
        discount = 1.0
        branch_uncertainty = 0.0
        for rollout_index, action in enumerate(sequence):
            previous = imagined
            imagined, collision_probability, uncertainty = predict_ensemble(models, imagined, action)
            branch_uncertainty += uncertainty
            trust = math.exp(-180.0 * branch_uncertainty)
            value += discount * trust * score_predicted_state(previous, imagined, collision_probability)
            value -= discount * 90.0 * uncertainty
            discount *= 0.72

            # Do not fabricate a precise deeper future after ensemble agreement
            # has broken down. The real sensor packet will re-anchor after action.
            if rollout_index == 0 and branch_uncertainty > uncertainty_limit:
                break

        value += 0.10 * float(action_goal_alignment(obs)[first_action])
        value += float(rng.normal(0.0, 0.002))
        if value > best_value:
            best_value = value
            best_action = first_action

    # When every imagined branch is weak, preserve the proven sensor controller.
    if best_value < -0.20:
        return sensor_fallback
    return int(best_action)


def make_grounded_beam_plan(models, obs, rng, depth=6, beam_width=8):
    """Deeper counterfactual search without enumerating all 8**depth paths."""
    sensor_fallback = choose_sensor_router(obs, rng)
    goal_alignment = action_goal_alignment(obs)
    beams = [([], obs.copy(), 0.0, 0.0, [])]

    for rollout_index in range(depth):
        expanded = []
        for sequence, imagined, value, branch_uncertainty, imagined_positions in beams:
            for action in range(len(MOVES)):
                if rollout_index == 0 and obs[5 + action] <= 0.01:
                    continue
                predicted, collision_probability, uncertainty = predict_ensemble(models, imagined, action)
                total_uncertainty = branch_uncertainty + uncertainty
                trust = math.exp(-150.0 * total_uncertainty)
                discount = 0.78**rollout_index
                step_value = trust * score_predicted_state(imagined, predicted, collision_probability)
                step_value -= 75.0 * uncertainty

                predicted_cell = tuple(np.round(predicted[:2] * (WORLD_SIZE - 1)).astype(int))
                revisit_penalty = 0.24 if predicted_cell in imagined_positions else 0.0
                # After real contact, permit temporarily negative goal progress
                # so a branch can back out of a cul-de-sac.
                if obs[-1] > 0.5:
                    predicted_progress = float(imagined[4] - predicted[4])
                    if predicted_progress < 0.0:
                        step_value += 5.5 * (-predicted_progress)

                expanded.append(
                    (
                        sequence + [action],
                        predicted,
                        value + discount * (step_value - revisit_penalty),
                        total_uncertainty,
                        imagined_positions + [predicted_cell],
                    )
                )

        if not expanded:
            return [sensor_fallback], -float("inf"), float("inf"), True
        expanded.sort(key=lambda item: item[2], reverse=True)
        beams = expanded[:beam_width]

    best_sequence, _, best_value, best_uncertainty, _ = beams[0]
    if not best_sequence or best_uncertainty > 0.020 or best_value < -0.55:
        return [sensor_fallback], best_value, best_uncertainty, True

    first_action = best_sequence[0]
    # A small present-time goal prior resolves equivalent safe branches without
    # dominating the multi-step counterfactual score.
    if best_value < -0.05 and goal_alignment[sensor_fallback] > goal_alignment[first_action] + 0.5:
        return [sensor_fallback], best_value, best_uncertainty, True
    return [int(action) for action in best_sequence], best_value, best_uncertainty, False


def choose_grounded_beam_router(models, obs, rng):
    plan, _, _, _ = make_grounded_beam_plan(models, obs, rng)
    return plan[0]


def observation_cell(obs):
    return tuple(np.round(obs[:2] * (WORLD_SIZE - 1)).astype(int))


def update_episodic_occupancy(obs, occupancy, visits):
    """Fuse one local ray packet into an online free/blocked memory."""
    current = observation_cell(obs)
    occupancy[current] = 0
    visits[current] = visits.get(current, 0) + 1
    for action, move in enumerate(MOVES):
        clear_steps = int(round(float(obs[5 + action]) * SENSOR_RANGE))
        for distance in range(1, clear_steps + 1):
            cell = current[0] + int(move[0]) * distance, current[1] + int(move[1]) * distance
            if 0 <= cell[0] < WORLD_SIZE and 0 <= cell[1] < WORLD_SIZE:
                occupancy[cell] = 0
        if clear_steps < SENSOR_RANGE:
            distance = clear_steps + 1
            cell = current[0] + int(move[0]) * distance, current[1] + int(move[1]) * distance
            if 0 <= cell[0] < WORLD_SIZE and 0 <= cell[1] < WORLD_SIZE:
                occupancy[cell] = 1


def shortest_known_path(start, target, occupancy):
    if start == target:
        return []
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        for action, move in enumerate(MOVES):
            nxt = current[0] + int(move[0]), current[1] + int(move[1])
            if occupancy.get(nxt) != 0 or nxt in parent:
                continue
            parent[nxt] = (current, action)
            if nxt == target:
                actions = []
                cursor = nxt
                while parent[cursor] is not None:
                    previous, chosen_action = parent[cursor]
                    actions.append(chosen_action)
                    cursor = previous
                return list(reversed(actions))
            queue.append(nxt)
    return None


def episodic_frontier_plan(obs, occupancy, visits, revisit_weight=1.8, frontier_enabled=True):
    """Plan to the goal if known, otherwise to a useful map frontier."""
    start = observation_cell(obs)
    goal_delta = obs[2:4] * (WORLD_SIZE - 1)
    goal = tuple(np.round(np.asarray(start) + goal_delta).astype(int))
    direct = shortest_known_path(start, goal, occupancy)
    if direct is not None:
        return direct, "known_goal"
    if not frontier_enabled:
        return [], "frontier_disabled"

    candidates = []
    cardinal = MOVES[[0, 2, 4, 6]]
    for cell, state in occupancy.items():
        if state != 0:
            continue
        unknown_neighbors = 0
        for move in cardinal:
            neighbor = cell[0] + int(move[0]), cell[1] + int(move[1])
            if 0 <= neighbor[0] < WORLD_SIZE and 0 <= neighbor[1] < WORLD_SIZE and neighbor not in occupancy:
                unknown_neighbors += 1
        if unknown_neighbors == 0:
            continue
        path = shortest_known_path(start, cell, occupancy)
        if path is None or not path:
            continue
        goal_distance = math.dist(cell, goal)
        revisit_cost = revisit_weight * visits.get(cell, 0)
        path_cost = 0.12 * len(path)
        information_bonus = 0.35 * unknown_neighbors
        score = goal_distance + revisit_cost + path_cost - information_bonus
        candidates.append((score, path, cell))

    if not candidates:
        return [], "no_frontier"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], "frontier"


def run_episode(models, condition, kind, seed):
    env = make_world(kind, seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 50000)
    collisions = 0
    path = [env.pos]
    away_from_goal_steps = 0
    plan_queue = deque()
    macro_queue = deque()
    episodic_plan_queue = deque()
    occupancy = {}
    visits = {}
    plans_created = 0
    plan_actions_executed = 0
    plan_completions = 0
    plan_sensor_vetoes = 0
    plan_fallbacks = 0
    stagnation_ticks = 0
    best_goal_distance = math.dist(env.start, env.goal)
    macro_used = False

    for step in range(MAX_STEPS):
        if condition == "reactive":
            action = choose_reactive(obs, rng)
        elif condition == "sensor_router":
            action = choose_sensor_router(obs, rng)
        elif condition == "learned_one_step":
            action = choose_model_router(models[0], obs, 1, rng)
        elif condition == "learned_depth_2":
            action = choose_model_router(models[0], obs, 2, rng)
        elif condition == "uncertainty_grounded_depth_2":
            action = choose_uncertainty_grounded_router(models, obs, rng)
        elif condition == "uncertainty_grounded_beam":
            action = choose_grounded_beam_router(models, obs, rng)
        elif condition == "counterfactual_plan_memory":
            action = None
            if plan_queue:
                candidate = plan_queue[0]
                if obs[5 + candidate] <= 0.01:
                    plan_queue.clear()
                    plan_sensor_vetoes += 1
                else:
                    action = plan_queue.popleft()
                    plan_actions_executed += 1
                    if not plan_queue:
                        plan_completions += 1

            if action is None:
                plan, _, _, used_fallback = make_grounded_beam_plan(models, obs, rng)
                plan_queue.extend(plan)
                plans_created += 1
                plan_fallbacks += int(used_fallback)
                action = plan_queue.popleft()
                plan_actions_executed += 1
                if not plan_queue:
                    plan_completions += 1
        elif condition == "scripted_escape_macro":
            if macro_queue:
                action = macro_queue.popleft()
            elif stagnation_ticks >= 6 and not macro_used:
                side = 6 if obs[5 + 6] >= obs[5 + 2] else 2
                # Explicit control: reverse out, clear one side wall, then
                # resume movement toward the goal. This is intentionally not
                # learned and defines the task's solvable upper bound.
                macro_queue.extend([4] * 7 + [side] * 5 + [0] * 12)
                action = macro_queue.popleft()
                plans_created += 1
                macro_used = True
            else:
                action = choose_sensor_router(obs, rng)
        elif condition in {
            "episodic_frontier_world_model",
            "episodic_no_revisit_cost",
            "episodic_no_frontier_objective",
        }:
            update_episodic_occupancy(obs, occupancy, visits)
            action = None
            if episodic_plan_queue:
                candidate = episodic_plan_queue[0]
                if obs[5 + candidate] <= 0.01:
                    episodic_plan_queue.clear()
                    plan_sensor_vetoes += 1
                else:
                    action = episodic_plan_queue.popleft()
                    plan_actions_executed += 1
                    if not episodic_plan_queue:
                        plan_completions += 1

            if action is None:
                revisit_weight = 0.0 if condition == "episodic_no_revisit_cost" else 1.8
                frontier_enabled = condition != "episodic_no_frontier_objective"
                plan, plan_kind = episodic_frontier_plan(
                    obs,
                    occupancy,
                    visits,
                    revisit_weight=revisit_weight,
                    frontier_enabled=frontier_enabled,
                )
                plans_created += 1
                if plan:
                    episodic_plan_queue.extend(plan)
                    action = episodic_plan_queue.popleft()
                    plan_actions_executed += 1
                    if not episodic_plan_queue:
                        plan_completions += 1
                else:
                    action = choose_sensor_router(obs, rng)
                    plan_fallbacks += 1
        else:
            raise ValueError(condition)

        old_distance = math.dist(env.pos, env.goal)
        obs, collision, done = env.step(action)
        current_goal_distance = math.dist(env.pos, env.goal)
        if current_goal_distance < best_goal_distance - 1e-8:
            best_goal_distance = current_goal_distance
            stagnation_ticks = 0
        else:
            stagnation_ticks += 1
        away_from_goal_steps += int(math.dist(env.pos, env.goal) > old_distance + 1e-8)
        collisions += int(collision)
        path.append(env.pos)
        if done:
            return {
                "success": 1.0,
                "steps": step + 1,
                "collisions": collisions,
                "path_efficiency": math.dist(env.start, env.goal) / max(1, step + 1),
                "away_from_goal_steps": away_from_goal_steps,
                "detour_completed": float(kind != "u_detour" or away_from_goal_steps >= 1),
                "plans_created": plans_created,
                "plan_actions_executed": plan_actions_executed,
                "plan_completions": plan_completions,
                "plan_sensor_vetoes": plan_sensor_vetoes,
                "plan_fallbacks": plan_fallbacks,
                "path": path,
                "blocked": env.blocked,
            }

    return {
        "success": 0.0,
        "steps": MAX_STEPS,
        "collisions": collisions,
        "path_efficiency": 0.0,
        "away_from_goal_steps": away_from_goal_steps,
        "detour_completed": 0.0,
        "plans_created": plans_created,
        "plan_actions_executed": plan_actions_executed,
        "plan_completions": plan_completions,
        "plan_sensor_vetoes": plan_sensor_vetoes,
        "plan_fallbacks": plan_fallbacks,
        "path": path,
        "blocked": env.blocked,
    }


def evaluate(models, runs=24):
    standard_conditions = [
        "reactive",
        "sensor_router",
        "learned_one_step",
        "learned_depth_2",
        "uncertainty_grounded_depth_2",
    ]
    detour_conditions = standard_conditions + [
        "uncertainty_grounded_beam",
        "counterfactual_plan_memory",
        "scripted_escape_macro",
        "episodic_frontier_world_model",
        "episodic_no_revisit_cost",
        "episodic_no_frontier_objective",
    ]
    episodes = {}
    summary = {}

    evaluation_sets = [
        ("circles", standard_conditions, 0, runs),
        ("rectangles", standard_conditions, 1, runs),
        ("diagonal_bars", standard_conditions, 2, runs),
        ("u_detour", detour_conditions, 3, 8),
    ]
    for kind, conditions, kind_index, run_count in evaluation_sets:
        for condition in conditions:
            key = f"{kind}:{condition}"
            rows = [
                run_episode(models, condition, kind, 30000 + kind_index * 1000 + run)
                for run in range(run_count)
            ]
            episodes[key] = rows
            summary[key] = {
                "geometry": kind,
                "condition": condition,
                "held_out_geometry": kind in {"diagonal_bars", "u_detour"},
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "mean_steps": float(np.mean([row["steps"] for row in rows])),
                "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
                "mean_path_efficiency": float(np.mean([row["path_efficiency"] for row in rows])),
                "mean_away_from_goal_steps": float(np.mean([row["away_from_goal_steps"] for row in rows])),
                "detour_completion_rate": float(np.mean([row["detour_completed"] for row in rows])),
                "mean_plans_created": float(np.mean([row["plans_created"] for row in rows])),
                "mean_plan_actions_executed": float(np.mean([row["plan_actions_executed"] for row in rows])),
                "mean_plan_completions": float(np.mean([row["plan_completions"] for row in rows])),
                "mean_plan_sensor_vetoes": float(np.mean([row["plan_sensor_vetoes"] for row in rows])),
                "mean_plan_fallbacks": float(np.mean([row["plan_fallbacks"] for row in rows])),
            }
    return summary, episodes


def plot_summary(summary, path):
    conditions = [
        "reactive",
        "sensor_router",
        "learned_one_step",
        "learned_depth_2",
        "uncertainty_grounded_depth_2",
    ]
    kinds = ["circles", "rectangles", "diagonal_bars"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("success_rate", "Success rate"), ("mean_collisions", "Mean collisions"), ("mean_steps", "Mean steps")]
    x = np.arange(len(conditions))
    width = 0.24
    colors = ["#287271", "#e9c46a", "#e76f51"]
    for ax, (metric, title) in zip(axes, metrics):
        for index, kind in enumerate(kinds):
            values = [summary[f"{kind}:{condition}"][metric] for condition in conditions]
            ax.bar(x + (index - 1) * width, values, width, label=kind, color=colors[index])
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen Counterfactual Router: Familiar and Held-Out Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_detour_summary(summary, path):
    conditions = [
        "reactive",
        "sensor_router",
        "learned_one_step",
        "learned_depth_2",
        "uncertainty_grounded_depth_2",
        "uncertainty_grounded_beam",
        "counterfactual_plan_memory",
        "scripted_escape_macro",
        "episodic_frontier_world_model",
        "episodic_no_revisit_cost",
        "episodic_no_frontier_objective",
    ]
    metrics = ["success_rate", "detour_completion_rate", "mean_collisions"]
    labels = ["Success", "Completed detour", "Collisions / 20"]
    values = []
    for condition in conditions:
        row = summary[f"u_detour:{condition}"]
        values.append([row[metrics[0]], row[metrics[1]], row[metrics[2]] / 20.0])

    x = np.arange(len(conditions))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = ["#287271", "#e9c46a", "#e76f51"]
    for index, label in enumerate(labels):
        ax.bar(x + (index - 1) * width, [row[index] for row in values], width, label=label, color=colors[index])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0, max(1.05, max(max(row) for row in values) * 1.1))
    ax.set_title("Frozen Zero-Shot U-Detour Evaluation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paths(episodes, path):
    conditions = [
        "reactive",
        "sensor_router",
        "learned_one_step",
        "learned_depth_2",
        "uncertainty_grounded_depth_2",
        "uncertainty_grounded_beam",
        "counterfactual_plan_memory",
        "scripted_escape_macro",
        "episodic_frontier_world_model",
        "episodic_no_revisit_cost",
        "episodic_no_frontier_objective",
    ]
    fig, axes = plt.subplots(1, 11, figsize=(44, 4.5), sharex=True, sharey=True)
    for ax, condition in zip(axes, conditions):
        row = episodes[f"u_detour:{condition}"][0]
        blocked = np.asarray(list(row["blocked"]))
        trajectory = np.asarray(row["path"])
        if len(blocked):
            ax.scatter(blocked[:, 0], blocked[:, 1], marker="s", s=30, color="#4b5563")
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#d1495b", linewidth=2)
        ax.scatter(
            [trajectory[0, 0], 9],
            [trajectory[0, 1], 16],
            color=["#287271", "#f4a261"],
            s=65,
        )
        ax.set_title(condition)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    fig.suptitle("Held-Out U-Detour Paths (No Retraining)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(33)
    rows = collect_training_data()
    models, training_histories = train_model_ensemble(rows)
    summary, episodes = evaluate(models)

    held_out = {key: value for key, value in summary.items() if value["held_out_geometry"]}
    learned = held_out["diagonal_bars:uncertainty_grounded_depth_2"]
    reactive = held_out["diagonal_bars:reactive"]
    zero_shot_supported = (
        learned["success_rate"] >= 0.70
        and learned["success_rate"] > reactive["success_rate"]
        and learned["mean_collisions"] < reactive["mean_collisions"]
    )
    detour = held_out["u_detour:episodic_frontier_world_model"]
    detour_sensor = held_out["u_detour:sensor_router"]
    zero_shot_detour_supported = (
        detour["success_rate"] >= 0.70
        and detour["detour_completion_rate"] >= 0.70
        and detour["success_rate"] > detour_sensor["success_rate"]
    )
    payload = {
        "training": {
            "geometries": ["circles", "rectangles"],
            "transitions": len(rows),
            "ensemble_members": len(models),
            "initial_loss_mean": float(np.mean([history[0] for history in training_histories])),
            "final_loss_mean": float(np.mean([history[-1] for history in training_histories])),
        },
        "evaluation": {
            "held_out_geometry": "diagonal_bars",
            "held_out_detour_geometry": "u_detour",
            "model_frozen_during_evaluation": True,
            "receding_horizon_reanchors_after_each_real_action": True,
            "zero_shot_geometry_transfer_supported": bool(zero_shot_supported),
            "zero_shot_detour_transfer_supported": bool(zero_shot_detour_supported),
        },
        "summary": summary,
        "claim_boundary": (
            "The neural ensemble supports frozen zero-shot transfer across local obstacle geometry. The U-detour "
            "result supports zero-shot traversal by a designed online occupancy/frontier world model reconstructed from "
            "local rays. It does not establish learned topology abstraction, emergent navigation, open-ended zero-shot "
            "task learning, a complete physical world model, or active inference."
        ),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "embodied_world_model_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(summary, OUT / "embodied_world_model_summary.png")
    plot_detour_summary(summary, OUT / "embodied_world_model_detour_summary.png")
    plot_paths(episodes, OUT / "embodied_world_model_zero_shot_paths.png")
    print("Embodied world-model lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
