#!/usr/bin/env python3
"""Post-train the recurrent forager on Unity-scaled continuous courses.

The original checkpoint is never overwritten. A candidate is exported only
after continuous-course, original-grid, and memory-dependence gates pass.
"""

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from upgraded_foraging_pipeline import (
    ACTION_NAMES,
    CORE_DIM,
    FOOD_SIGHT,
    MOVES,
    OBS_DIM,
    RAY_RANGE,
    RecurrentForagingPolicy,
    evaluate_policy as evaluate_grid_policy,
    generate_layout,
    load_checkpoint,
)


STEP_LENGTH = 0.44
AGENT_RADIUS = 0.20
PICKUP_RADIUS = 0.55
MAX_STEPS = 320
RAY_STEP = 0.10
TRAIN_FAMILIES = ["pocket", "l_wall", "offset_barriers", "u_trap", "narrow_corridor"]
WITHHELD_FAMILIES = ["c_shape", "zigzag_gate"]
ALL_COURSES = ["u_trap", "c_shape", "l_wall", "zigzag_gate", "offset_barriers", "narrow_corridor"]
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints/unity_posttrained")


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    ate: bool
    collision: bool
    new_cell: bool
    jerk: float


def narrow_corridor_layout(seed):
    rng = np.random.default_rng(seed)
    blocked = set()
    offset = int(rng.integers(-1, 2))
    for y in range(2, 11):
        blocked.add((5 + offset, y))
    for y in range(5, 14):
        blocked.add((9 + offset, y))
    for x in range(5 + offset, 10 + offset):
        blocked.add((x, 10))
    blocked.discard((5 + offset, 3 + int(rng.integers(0, 3))))
    blocked.discard((9 + offset, 10 + int(rng.integers(0, 3))))
    return blocked, (3 + offset, 2), (11 + offset, 12)


def layout_for(family, seed):
    if family == "narrow_corridor":
        return narrow_corridor_layout(seed)
    return generate_layout(family, seed)


class UnityContinuousCourse:
    def __init__(self, family, layout_seed, food_reward=2.0):
        self.family = family
        self.layout_seed = int(layout_seed)
        self.food_reward = float(food_reward)
        self.blocked, start, food = layout_for(family, layout_seed)
        self.start = np.asarray(start, dtype=np.float32) + 0.5
        self.food = np.asarray(food, dtype=np.float32) + 0.5
        self.distance_to_food = self.build_distance_map()
        self.reset()

    def build_distance_map(self):
        goal = tuple(np.floor(self.food).astype(int))
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            cell = queue.popleft()
            for move in MOVES:
                previous = cell[0] - int(move[0]), cell[1] - int(move[1])
                if not (0 <= previous[0] < 15 and 0 <= previous[1] < 15):
                    continue
                if previous in self.blocked or previous in distances:
                    continue
                if move[0] != 0 and move[1] != 0:
                    side_a = previous[0] + int(move[0]), previous[1]
                    side_b = previous[0], previous[1] + int(move[1])
                    if side_a in self.blocked and side_b in self.blocked:
                        continue
                distances[previous] = distances[cell] + 1
                queue.append(previous)
        return distances

    def reset(self):
        self.pos = self.start.copy()
        self.steps = 0
        self.hunger = 0.12
        self.last_action = 0
        self.last_reward = 0.0
        self.visits = Counter({self.visit_cell(): 1})
        self.collisions = 0
        self.path_length = 0.0
        return self.observe()

    def visit_cell(self, position=None):
        point = self.pos if position is None else position
        return int(math.floor(float(point[0]) * 2.0)), int(math.floor(float(point[1]) * 2.0))

    def point_blocked(self, point):
        x, y = float(point[0]), float(point[1])
        if x < 0.0 or y < 0.0 or x >= 15.0 or y >= 15.0:
            return True
        return (int(math.floor(x)), int(math.floor(y))) in self.blocked

    def body_blocked(self, point):
        x, y = float(point[0]), float(point[1])
        if x < AGENT_RADIUS or y < AGENT_RADIUS or x > 15.0 - AGENT_RADIUS or y > 15.0 - AGENT_RADIUS:
            return True
        for cell_x in range(max(0, int(math.floor(x - AGENT_RADIUS))), min(14, int(math.floor(x + AGENT_RADIUS))) + 1):
            for cell_y in range(max(0, int(math.floor(y - AGENT_RADIUS))), min(14, int(math.floor(y + AGENT_RADIUS))) + 1):
                if (cell_x, cell_y) not in self.blocked:
                    continue
                nearest_x = min(max(x, cell_x), cell_x + 1.0)
                nearest_y = min(max(y, cell_y), cell_y + 1.0)
                if math.hypot(x - nearest_x, y - nearest_y) < AGENT_RADIUS:
                    return True
        return False

    def ray_distance(self, move):
        direction = move.astype(np.float32)
        direction /= np.linalg.norm(direction)
        for distance in np.arange(RAY_STEP, RAY_RANGE + RAY_STEP, RAY_STEP):
            if self.point_blocked(self.pos + direction * distance):
                return float(np.clip((distance - RAY_STEP) / RAY_RANGE, 0.0, 1.0))
        return 1.0

    def food_visible(self):
        distance = float(np.linalg.norm(self.food - self.pos))
        if distance > FOOD_SIGHT:
            return False
        samples = max(2, int(math.ceil(distance / RAY_STEP)))
        for alpha in np.linspace(0.0, 1.0, samples)[1:-1]:
            if self.point_blocked(self.pos + alpha * (self.food - self.pos)):
                return False
        return True

    def observe(self):
        rays = np.asarray([self.ray_distance(move) for move in MOVES], dtype=np.float32)
        visible = self.food_visible()
        food_delta = (self.food - self.pos) / 14.0 if visible else np.zeros(2, dtype=np.float32)
        previous = np.zeros(len(MOVES), dtype=np.float32)
        previous[self.last_action] = 1.0
        return np.concatenate([rays, [float(visible)], food_delta, [self.hunger], previous, [self.last_reward]]).astype(np.float32)

    def step(self, action):
        action = int(action)
        previous_action = self.last_action
        direction = MOVES[action].astype(np.float32)
        direction /= np.linalg.norm(direction)
        candidate = self.pos + direction * STEP_LENGTH
        collision = self.body_blocked(candidate)
        old_distance = float(np.linalg.norm(self.food - self.pos))
        old_cell = self.visit_cell()
        if not collision:
            self.pos = candidate
            self.path_length += STEP_LENGTH
        else:
            self.collisions += 1
        new_cell = self.visit_cell() not in self.visits
        self.visits[self.visit_cell()] += 1
        revisit_count = self.visits[self.visit_cell()]
        new_distance = float(np.linalg.norm(self.food - self.pos))
        visible = self.food_visible()
        ate = new_distance <= PICKUP_RADIUS
        cosine = float(np.dot(MOVES[previous_action], MOVES[action])) / (
            float(np.linalg.norm(MOVES[previous_action])) * float(np.linalg.norm(MOVES[action]))
        )
        jerk = 0.5 * (1.0 - float(np.clip(cosine, -1.0, 1.0)))

        reward = -0.004
        reward += 0.010 if new_cell else 0.0
        reward -= 0.006 * math.sqrt(max(0, revisit_count - 1))
        reward -= 0.012 * jerk
        if collision:
            reward -= 0.24
        if visible and not collision:
            reward += 0.055 * (old_distance - new_distance)
        if ate and self.food_reward > 0.0:
            reward += self.food_reward

        self.steps += 1
        self.hunger = min(1.0, self.hunger + 1.0 / MAX_STEPS)
        done = (ate and self.food_reward > 0.0) or self.steps >= MAX_STEPS
        if self.steps >= MAX_STEPS and not ate:
            reward -= 0.45
        self.last_action = action
        self.last_reward = reward
        return StepResult(self.observe(), reward, done, ate, collision, new_cell, jerk)


class PriorityReplay:
    def __init__(self):
        self.failures = Counter()

    def record(self, env, success):
        key = (env.family, env.layout_seed)
        self.failures[key] = max(0, self.failures[key] - 1) if success else self.failures[key] + 1

    def sample(self, rng):
        active = [(key, count) for key, count in self.failures.items() if count > 0]
        if not active:
            return None
        weights = np.asarray([count for _, count in active], dtype=np.float64)
        weights /= weights.sum()
        return active[int(rng.choice(len(active), p=weights))][0]


def new_world(rng, replay, food_reward=2.0):
    replayed = replay.sample(rng) if rng.random() < 0.45 else None
    if replayed is None:
        family = TRAIN_FAMILIES[int(rng.integers(len(TRAIN_FAMILIES)))]
        seed = int(rng.integers(1, 10_000_000))
    else:
        family, seed = replayed
    return UnityContinuousCourse(family, seed, food_reward=food_reward)


def continuous_safety_mask_logits(logits, obs):
    """Mask actions whose centerline cannot clear one Unity-sized body step."""
    required_clearance = (STEP_LENGTH + AGENT_RADIUS - RAY_STEP) / RAY_RANGE
    clearance = obs[:, : len(MOVES)]
    blocked = clearance < required_clearance
    all_blocked = torch.all(blocked, dim=-1)
    if torch.any(all_blocked):
        safest = torch.argmax(clearance, dim=-1)
        blocked = blocked.clone()
        blocked[all_blocked] = True
        blocked[all_blocked, safest[all_blocked]] = False
    return logits.masked_fill(blocked, -1e9)


def body_safety_mask_logits(logits, envs):
    """Use the physics body's candidate poses as a grounded action mask."""
    blocked_rows = []
    for env in envs:
        blocked = []
        for move in MOVES:
            direction = move.astype(np.float32) / np.linalg.norm(move)
            blocked.append(env.body_blocked(env.pos + direction * STEP_LENGTH))
        if all(blocked):
            blocked[int(np.argmax(env.observe()[: len(MOVES)]))] = False
        blocked_rows.append(blocked)
    blocked = torch.tensor(blocked_rows, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(blocked, -1e9)


def oracle_action(env):
    """Return a shortest-path action used only as a post-training teacher."""
    choices = []
    for action, move in enumerate(MOVES):
        direction = move.astype(np.float32) / np.linalg.norm(move)
        candidate = env.pos + direction * STEP_LENGTH
        if env.body_blocked(candidate):
            continue
        cell = tuple(np.floor(candidate).astype(int))
        path_cost = env.distance_to_food.get(cell, 10_000)
        goal_cost = float(np.linalg.norm(candidate - env.food)) / 15.0
        turn_cost = 0.035 * (1.0 - float(np.dot(MOVES[env.last_action], move)) / (
            float(np.linalg.norm(MOVES[env.last_action])) * float(np.linalg.norm(move))
        ))
        choices.append((path_cost + goal_cost + turn_cost, action))
    if choices:
        return min(choices)[1]
    return int(np.argmax(env.observe()[: len(MOVES)]))


def posttrain(base_path, seed=701, updates=420, env_count=24, rollout=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy, source_payload = load_checkpoint(base_path)
    policy.train()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    for module in (policy.actor, policy.forward_heads):
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=0.001,
    )
    replay = PriorityReplay()
    envs = [new_world(rng, replay) for _ in range(env_count)]
    obs = torch.tensor(np.stack([env.reset() for env in envs]))
    hidden = policy.initial_state(env_count)
    recent = deque(maxlen=240)
    curves = {"success": [], "loss": [], "collision_rate": [], "validation": []}
    best_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    policy.eval()
    best_validation = evaluate_continuous(policy, TRAIN_FAMILIES, [17, 23], episodes_per_seed=2)
    policy.train()

    for update in range(updates):
        imitation_losses, entropies, masks = [], [], []
        predictions, targets = [], []
        collision_samples = []
        teacher_probability = 0.80 - 0.55 * update / max(1, updates - 1)
        for _ in range(rollout):
            logits, value, hidden = policy.step(obs, hidden)
            logits = body_safety_mask_logits(logits, envs)
            dist = torch.distributions.Categorical(logits=logits)
            teacher = torch.tensor([oracle_action(env) for env in envs], dtype=torch.long)
            sampled = dist.sample()
            use_teacher = torch.rand(env_count) < teacher_probability
            action = torch.where(use_teacher, teacher, sampled)
            prediction = policy.predict_core(hidden, action)
            next_obs, prediction_targets, shaped, alive = [], [], [], []
            for index, env in enumerate(envs):
                result = env.step(int(action[index]))
                shaped.append(result.reward)
                collision_samples.append(float(result.collision))
                prediction_targets.append(result.obs[:CORE_DIM])
                if result.done:
                    success = bool(result.ate)
                    replay.record(env, success)
                    recent.append(float(success))
                    envs[index] = new_world(rng, replay)
                    next_obs.append(envs[index].reset())
                    alive.append(0.0)
                else:
                    next_obs.append(result.obs)
                    alive.append(1.0)
            next_tensor = torch.tensor(np.stack(next_obs))
            imitation_losses.append(F.cross_entropy(logits, teacher))
            entropies.append(dist.entropy())
            masks.append(torch.tensor(alive, dtype=torch.float32))
            predictions.append(prediction)
            targets.append(torch.tensor(np.stack(prediction_targets)))
            obs = next_tensor
            hidden = hidden * masks[-1].unsqueeze(-1)

        imitation = torch.stack(imitation_losses).mean()
        entropy = -0.003 * torch.stack(entropies).mean()
        auxiliary = torch.stack(
            [F.smooth_l1_loss(prediction, target.unsqueeze(0).expand_as(prediction)) for prediction, target in zip(predictions, targets)]
        ).mean()
        loss = imitation + entropy + 0.18 * auxiliary
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
        optimizer.step()
        hidden = hidden.detach()
        curves["success"].append(float(np.mean(recent)) if recent else 0.0)
        curves["loss"].append(float(loss.detach()))
        curves["collision_rate"].append(float(np.mean(collision_samples)))
        if update % 40 == 0 or update == updates - 1:
            policy.eval()
            validation = evaluate_continuous(policy, TRAIN_FAMILIES, [17, 23], episodes_per_seed=2)
            policy.train()
            curves["validation"].append({"update": update, **validation})
            current_key = (validation["success_rate"], -validation["mean_steps"])
            best_key = (best_validation["success_rate"], -best_validation["mean_steps"])
            if current_key > best_key:
                best_validation = validation
                best_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        if update % 40 == 0 or update == updates - 1:
            print(
                f"update={update:03d} success={curves['success'][-1]:.3f} "
                f"validation={validation['success_rate']:.3f} "
                f"collision_rate={curves['collision_rate'][-1]:.3f} loss={curves['loss'][-1]:.3f}",
                flush=True,
            )
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, source_payload, curves


def mpc_action(
    policy,
    obs,
    hidden,
    env,
    horizon=4,
    policy_weight=0.08,
    uncertainty_weight=0.16,
    jerk_weight=0.025,
):
    with torch.no_grad():
        logits, _, next_hidden = policy.step(obs, hidden)
        logits = body_safety_mask_logits(logits, [env])
        probabilities = torch.softmax(logits, dim=-1)[0]
        scores = []
        for root in range(len(MOVES)):
            imagined_hidden = next_hidden.clone()
            imagined_obs = obs.clone()
            score = policy_weight * float(torch.log(probabilities[root].clamp_min(1e-8)))
            previous = int(torch.argmax(imagined_obs[0, CORE_DIM:CORE_DIM + len(MOVES)]))
            for depth in range(horizon):
                action = torch.tensor([root])
                ensemble = policy.predict_core(imagined_hidden, action)
                core = ensemble.mean(dim=0)
                uncertainty = float(torch.var(ensemble, dim=0).mean())
                clearance = float(core[0, root])
                visible = float(torch.sigmoid(5.0 * (core[0, 8] - 0.5)))
                food_distance = float(torch.linalg.vector_norm(core[0, 9:11]))
                collision_risk = max(0.0, 0.12 - clearance) * 8.0
                cosine = float(np.dot(MOVES[previous], MOVES[root])) / (
                    float(np.linalg.norm(MOVES[previous])) * float(np.linalg.norm(MOVES[root]))
                )
                jerk = 0.5 * (1.0 - np.clip(cosine, -1.0, 1.0)) if depth == 0 else 0.0
                score += (0.08 * visible * (1.0 - min(1.0, food_distance))) - 0.34 * collision_risk - jerk_weight * jerk
                score -= uncertainty_weight * uncertainty
                previous_one_hot = torch.zeros(1, len(MOVES))
                previous_one_hot[0, root] = 1.0
                estimated_reward = torch.tensor([[max(-0.2, min(0.2, score / (depth + 1)))]]).float()
                imagined_obs = torch.cat([core, previous_one_hot, estimated_reward], dim=-1)
                _, _, imagined_hidden = policy.step(imagined_obs, imagined_hidden)
                previous = root
            scores.append(score)
        return int(np.argmax(scores)), next_hidden


def evaluate_continuous(
    policy,
    families,
    seeds,
    episodes_per_seed=12,
    use_mpc=False,
    reset_memory=False,
    mpc_kwargs=None,
    mpc_requires_visible_food=False,
):
    rows = []
    for seed in seeds:
        for family_index, family in enumerate(families):
            for episode in range(episodes_per_seed):
                env = UnityContinuousCourse(family, seed * 100_000 + family_index * 10_000 + episode)
                obs = torch.tensor(env.reset()).unsqueeze(0)
                hidden = policy.initial_state(1)
                previous_action = 0
                reversals = 0
                for step in range(MAX_STEPS):
                    with torch.no_grad():
                        if use_mpc and (not mpc_requires_visible_food or env.food_visible()):
                            action, hidden = mpc_action(policy, obs, hidden, env, **(mpc_kwargs or {}))
                        else:
                            logits, _, hidden = policy.step(obs, hidden)
                            logits = body_safety_mask_logits(logits, [env])
                            action = int(torch.argmax(logits, dim=-1))
                    if np.dot(MOVES[previous_action], MOVES[action]) < 0:
                        reversals += 1
                    result = env.step(action)
                    previous_action = action
                    obs = torch.tensor(result.obs).unsqueeze(0)
                    if reset_memory:
                        hidden = policy.initial_state(1)
                    if result.done:
                        rows.append(
                            {
                                "seed": seed,
                                "family": family,
                                "success": float(result.ate),
                                "steps": step + 1,
                                "collisions": env.collisions,
                                "path_length": env.path_length,
                                "revisit_ratio": 1.0 - len(env.visits) / max(1, sum(env.visits.values())),
                                "reversals": reversals,
                            }
                        )
                        break
    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_path_length": float(np.mean([row["path_length"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "mean_reversals": float(np.mean([row["reversals"] for row in rows])),
        "by_family": {
            family: {
                "success_rate": float(np.mean([row["success"] for row in rows if row["family"] == family])),
                "mean_collisions": float(np.mean([row["collisions"] for row in rows if row["family"] == family])),
            }
            for family in families
        },
    }


def save_candidate(policy, source_payload, metrics, path):
    payload = dict(source_payload)
    payload["state_dict"] = policy.state_dict()
    payload["config"] = dict(source_payload["config"])
    payload["config"].update({"posttraining": "unity_continuous_v1", "mpc_default": metrics["selected_controller"] == "mpc"})
    payload["unity_posttraining_metrics"] = metrics
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/upgraded_foraging/best.pt")
    parser.add_argument("--updates", type=int, default=420)
    parser.add_argument("--quick", action="store_true", help="Short smoke run without export eligibility.")
    args = parser.parse_args()
    updates = 35 if args.quick else args.updates
    eval_seeds = [31] if args.quick else [31, 47, 59, 73, 89]
    episodes = 3 if args.quick else 12

    base, _ = load_checkpoint(args.base)
    print("evaluating frozen baseline on Unity-continuous courses", flush=True)
    baseline = evaluate_continuous(base, ALL_COURSES, eval_seeds, episodes_per_seed=episodes)
    print(json.dumps(baseline, indent=2), flush=True)
    policy, source_payload, curves = posttrain(args.base, updates=updates)
    raw = evaluate_continuous(policy, ALL_COURSES, eval_seeds, episodes_per_seed=episodes)
    mpc = evaluate_continuous(policy, ALL_COURSES, [31], episodes_per_seed=3, use_mpc=True)
    selected_name, selected = "raw", raw
    memory_reset = evaluate_continuous(
        policy,
        WITHHELD_FAMILIES,
        eval_seeds,
        episodes_per_seed=episodes,
        use_mpc=selected_name == "mpc",
        reset_memory=True,
    )
    grid_regression = evaluate_grid_policy(policy, ["c_shape", "zigzag_gate"], episodes_per_family=40, seed=150_000)
    criteria = {
        "continuous_success_at_least_90_percent": selected["success_rate"] >= 0.90,
        "continuous_mean_collisions_at_most_8": selected["mean_collisions"] <= 8.0,
        "original_grid_withheld_at_least_90_percent": grid_regression["success_rate"] >= 0.90,
        "memory_reset_drop_at_least_40_points": selected["success_rate"] - memory_reset["success_rate"] >= 0.40,
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    metrics = {
        "protocol": {
            "step_length": STEP_LENGTH,
            "agent_radius": AGENT_RADIUS,
            "food_sight": FOOD_SIGHT,
            "evaluation_seeds": eval_seeds,
            "episodes_per_seed_family": episodes,
            "withheld_from_posttraining": WITHHELD_FAMILIES,
        },
        "baseline": baseline,
        "posttrained_raw": raw,
        "posttrained_mpc": mpc,
        "selected_controller": selected_name,
        "selected": selected,
        "memory_reset": memory_reset,
        "original_grid_regression": grid_regression,
        "criteria": criteria,
        "curves": curves,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "unity_posttraining_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if criteria["all_passed"]:
        candidate = CHECKPOINT_DIR / "best.pt"
        save_candidate(policy, source_payload, metrics, candidate)
        print(f"exported {candidate}")
    else:
        print("candidate rejected; no checkpoint exported")
    print(json.dumps({"selected_controller": selected_name, "criteria": criteria, "metrics": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
