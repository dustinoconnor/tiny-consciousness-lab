#!/usr/bin/env python3
"""Post-train recurrent exploration from reward, then remove the teacher signal.

Training uses a hunger-weighted visit-novelty reward in procedurally rotated
food-sparse terrains. Evaluation exposes only the ordinary Unity-aligned
observation packet: rays, food visibility/direction, hunger, previous action,
and previous reward. No exploration waypoint or visit count reaches the policy.
"""

import argparse
import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from starvation_exploration_lab import MAX_STEPS, NORMAL_SIGHT, SparseTerrain
from unity_posttraining_lab import ALL_COURSES, MOVES, evaluate_continuous
from upgraded_foraging_pipeline import load_checkpoint


BASE_CHECKPOINT = Path("checkpoints/unity_mpc/best.pt")
OUTPUT = Path("outputs/starvation_posttraining_metrics.json")
CHECKPOINT = Path("checkpoints/starvation_posttrained/best.pt")
TRAIN_EPISODE_STEPS = 720
EVAL_STEPS = 1800


def rotate_point(point, turns):
    result = np.asarray(point, dtype=np.float32)
    for _ in range(turns % 4):
        result = np.asarray([64.0 - result[1], result[0]], dtype=np.float32)
    return result


def rotate_cell(cell, turns):
    x, z = cell
    for _ in range(turns % 4):
        x, z = 63 - z, x
    return x, z


class RotatedSparseTerrain(SparseTerrain):
    """Rotate the entire procedural world so barren starts span all corners."""

    def __init__(self, seed, turns):
        super().__init__(seed, NORMAL_SIGHT)
        self.turns = int(turns) % 4
        self.blocked = {rotate_cell(cell, self.turns) for cell in self.blocked}
        self.food = [rotate_point(point, self.turns) for point in self.food]
        self.start_pos = rotate_point(np.asarray([49.5, 49.5]), self.turns)
        self.reset()

    def reset(self):
        obs = super().reset()
        if hasattr(self, "start_pos"):
            self.pos = self.start_pos.copy()
            self.visits.clear()
            self.visits[self.visit_cell()] = 1
            obs = self.observe()
        return obs


def safe_logits(logits, envs):
    blocked_rows = []
    for env in envs:
        safe = set(env.safe_actions())
        blocked_rows.append([action not in safe for action in range(len(MOVES))])
    blocked = torch.tensor(blocked_rows, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(blocked, -1e9)


def make_env(rng):
    return RotatedSparseTerrain(int(rng.integers(1, 10_000_000)), int(rng.integers(0, 4)))


def shaped_reward(env, result, previous_action, action):
    cosine = float(np.dot(MOVES[previous_action], MOVES[action])) / (
        float(np.linalg.norm(MOVES[previous_action])) * float(np.linalg.norm(MOVES[action]))
    )
    jerk = 0.5 * (1.0 - float(np.clip(cosine, -1.0, 1.0)))
    hunger_gate = float(np.clip((env.hunger - 0.55) / 0.45, 0.0, 1.0))
    revisit_count = env.visits[env.visit_cell()]
    novelty = 1.0 if result.new_cell else -0.20 * math.sqrt(max(0, revisit_count - 1))
    reward = 2.0 if result.ate else -0.004
    reward += hunger_gate * 0.055 * novelty
    reward -= 0.010 * jerk
    reward -= 0.24 if result.collision else 0.0
    return reward


def evaluate_sparse(policy, seeds, reset_memory=False):
    rows = []
    for index, seed in enumerate(seeds):
        env = RotatedSparseTerrain(seed, index % 4)
        obs = torch.tensor(env.reset()).unsqueeze(0)
        hidden = policy.initial_state(1)
        previous_action = 0
        reversals = 0
        for _ in range(EVAL_STEPS):
            with torch.no_grad():
                logits, _value, hidden = policy.step(obs, hidden)
                logits = safe_logits(logits, [env])
                action = int(torch.argmax(logits, dim=-1))
            if float(np.dot(MOVES[previous_action], MOVES[action])) < 0.0:
                reversals += 1
            result = env.step(action)
            previous_action = action
            obs = torch.tensor(result.obs).unsqueeze(0)
            if reset_memory:
                hidden = policy.initial_state(1)
        boundaries = [0] + env.pickup_steps + [EVAL_STEPS]
        rows.append(
            {
                "pickups": env.pickups,
                "first_pickup_step": env.first_pickup_step or EVAL_STEPS,
                "max_meal_gap_steps": max(b - a for a, b in zip(boundaries, boundaries[1:])),
                "unique_cells": len(env.visits),
                "revisit_ratio": 1.0 - len(env.visits) / max(1, sum(env.visits.values())),
                "collisions": env.collisions,
                "reversals": reversals,
            }
        )
    return {
        "episodes": len(rows),
        "mean_pickups": float(np.mean([row["pickups"] for row in rows])),
        "mean_first_pickup_steps": float(np.mean([row["first_pickup_step"] for row in rows])),
        "mean_max_meal_gap_steps": float(np.mean([row["max_meal_gap_steps"] for row in rows])),
        "mean_unique_cells": float(np.mean([row["unique_cells"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_reversals": float(np.mean([row["reversals"] for row in rows])),
    }


def posttrain(base_path, updates, env_count=10, rollout=32, seed=1701):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy, source_payload = load_checkpoint(base_path)
    reference = copy.deepcopy(policy).eval()
    policy.train()
    for parameter in policy.forward_heads.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=2.5e-4,
    )
    envs = [make_env(rng) for _ in range(env_count)]
    obs = torch.tensor(np.stack([env.reset() for env in envs]))
    hidden = policy.initial_state(env_count)
    reference_hidden = reference.initial_state(env_count)
    episode_steps = np.zeros(env_count, dtype=np.int32)
    recent_pickups = deque(maxlen=100)
    curves = {"loss": [], "pickups": [], "kl": []}

    for update in range(updates):
        log_probs, values, rewards, masks, entropies, divergences = [], [], [], [], [], []
        for _ in range(rollout):
            logits, value, hidden = policy.step(obs, hidden)
            logits = safe_logits(logits, envs)
            with torch.no_grad():
                reference_logits, _reference_value, reference_hidden = reference.step(obs, reference_hidden)
                reference_logits = safe_logits(reference_logits, envs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_obs, step_rewards, alive = [], [], []
            for index, env in enumerate(envs):
                previous = env.last_action
                result = env.step(int(action[index]))
                step_rewards.append(shaped_reward(env, result, previous, int(action[index])))
                episode_steps[index] += 1
                if result.ate:
                    recent_pickups.append(1.0)
                if episode_steps[index] >= TRAIN_EPISODE_STEPS:
                    recent_pickups.append(float(env.pickups > 0))
                    envs[index] = make_env(rng)
                    next_obs.append(envs[index].reset())
                    episode_steps[index] = 0
                    alive.append(0.0)
                else:
                    next_obs.append(result.obs)
                    alive.append(1.0)
            alive_tensor = torch.tensor(alive, dtype=torch.float32)
            next_obs_tensor = torch.tensor(np.stack(next_obs))
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
            masks.append(alive_tensor)
            entropies.append(dist.entropy())
            divergences.append(
                F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(reference_logits, dim=-1),
                    reduction="batchmean",
                )
            )
            obs = next_obs_tensor
            hidden = hidden * alive_tensor.unsqueeze(-1)
            reference_hidden = reference_hidden * alive_tensor.unsqueeze(-1)

        with torch.no_grad():
            _logits, bootstrap, _next = policy.step(obs, hidden)
        returns = []
        running = bootstrap
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running = reward + 0.98 * running * mask
            returns.append(running)
        returns.reverse()
        advantage = torch.stack(returns) - torch.stack(values)
        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        critic_loss = 0.45 * advantage.pow(2).mean()
        entropy_loss = -0.006 * torch.stack(entropies).mean()
        kl_loss = 0.08 * torch.stack(divergences).mean()
        loss = actor_loss + critic_loss + entropy_loss + kl_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
        optimizer.step()
        hidden = hidden.detach()
        reference_hidden = reference_hidden.detach()
        curves["loss"].append(float(loss.detach()))
        curves["pickups"].append(float(np.mean(recent_pickups)) if recent_pickups else 0.0)
        curves["kl"].append(float(kl_loss.detach()))
        if update % 20 == 0 or update == updates - 1:
            print(
                f"update={update:03d} pickup_signal={curves['pickups'][-1]:.3f} "
                f"kl={curves['kl'][-1]:.4f} loss={curves['loss'][-1]:.4f}",
                flush=True,
            )
    return policy.eval(), source_payload, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--updates", type=int, default=260)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    updates = 24 if args.quick else args.updates
    sparse_seeds = [81_101 + 977 * index for index in range(4 if args.quick else 16)]
    course_seeds = [41] if args.quick else [41, 67, 89]
    course_episodes = 2 if args.quick else 6

    baseline, _payload = load_checkpoint(args.base)
    print("evaluating frozen baseline", flush=True)
    baseline_sparse = evaluate_sparse(baseline, sparse_seeds)
    baseline_courses = evaluate_continuous(
        baseline, ALL_COURSES, course_seeds, episodes_per_seed=course_episodes
    )
    candidate, source_payload, curves = posttrain(args.base, updates)
    print("evaluating teacher-free candidate", flush=True)
    candidate_sparse = evaluate_sparse(candidate, sparse_seeds)
    candidate_memory_reset = evaluate_sparse(candidate, sparse_seeds, reset_memory=True)
    candidate_courses = evaluate_continuous(
        candidate, ALL_COURSES, course_seeds, episodes_per_seed=course_episodes
    )
    criteria = {
        "meal_gap_improves_15pct": candidate_sparse["mean_max_meal_gap_steps"] <= baseline_sparse["mean_max_meal_gap_steps"] * 0.85,
        "pickups_not_reduced": candidate_sparse["mean_pickups"] >= baseline_sparse["mean_pickups"],
        "collision_safety_preserved": candidate_sparse["mean_collisions"] <= baseline_sparse["mean_collisions"] + 1.0,
        "course_success_regression_at_most_5_points": candidate_courses["success_rate"] >= baseline_courses["success_rate"] - 0.05,
        "memory_reset_reduces_sparse_pickups": candidate_memory_reset["mean_pickups"] < candidate_sparse["mean_pickups"],
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    metrics = {
        "protocol": {
            "base_checkpoint": str(args.base),
            "updates": updates,
            "teacher_signal_removed_during_evaluation": True,
            "food_coordinates_hidden_when_not_visible": True,
            "forward_heads_frozen": True,
            "procedural_start_rotations": 4,
        },
        "baseline_sparse": baseline_sparse,
        "candidate_sparse": candidate_sparse,
        "candidate_memory_reset": candidate_memory_reset,
        "baseline_courses": baseline_courses,
        "candidate_courses": candidate_courses,
        "criteria": criteria,
        "curves": curves,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if criteria["all_passed"]:
        payload = dict(source_payload)
        payload["state_dict"] = candidate.state_dict()
        payload["config"] = dict(source_payload["config"])
        payload["config"].update({"posttraining": "starvation_novelty_v1"})
        payload["starvation_posttraining_metrics"] = metrics
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, CHECKPOINT)
        print(f"exported {CHECKPOINT}")
    else:
        print("candidate rejected; no checkpoint exported")
    print(json.dumps({"criteria": criteria, "metrics": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
