#!/usr/bin/env python3
"""Compare fixed MPC with uncertainty-bounded stochastic imagination.

The checkpoint is frozen. This lab changes only inference and food-sensing
budgets, so failed experiments cannot damage the trained recurrent policy.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from upgraded_foraging_pipeline import CORE_DIM, MOVES, load_checkpoint
from unity_posttraining_lab import (
    ALL_COURSES,
    MAX_STEPS,
    RAY_STEP,
    UnityContinuousCourse,
    body_safety_mask_logits,
    mpc_action,
)


OUTPUT_PATH = Path("outputs/adaptive_stochastic_mpc_metrics.json")
CHECKPOINT_PATH = Path("checkpoints/unity_mpc/best.pt")
MOVE_TENSOR = torch.tensor(np.asarray(MOVES), dtype=torch.float32)
MOVE_TENSOR = MOVE_TENSOR / torch.linalg.vector_norm(MOVE_TENSOR, dim=-1, keepdim=True)


class HungerAdaptiveCourse(UnityContinuousCourse):
    """Expand grounded food sight as metabolic urgency rises."""

    def food_sight(self):
        if self.hunger >= 0.92:
            return 13.0
        if self.hunger >= 0.70:
            return 10.0
        return 7.0

    def food_visible(self):
        distance = float(np.linalg.norm(self.food - self.pos))
        if distance > self.food_sight():
            return False
        samples = max(2, int(math.ceil(distance / RAY_STEP)))
        for alpha in np.linspace(0.0, 1.0, samples)[1:-1]:
            if self.point_blocked(self.pos + alpha * (self.food - self.pos)):
                return False
        return True


def recurrent_action(policy, obs, hidden, env):
    with torch.no_grad():
        logits, _value, next_hidden = policy.step(obs, hidden)
        logits = body_safety_mask_logits(logits, [env])
        action = int(torch.argmax(logits, dim=-1))
    return action, next_hidden


def adaptive_stochastic_action(
    policy,
    obs,
    hidden,
    env,
    no_progress,
    rng,
    samples=4,
    uncertainty_budget=0.0015,
    jerk_weight=0.025,
):
    """Use recurrent control unless uncertainty or tactical demand earns MPC."""
    with torch.no_grad():
        logits, _value, next_hidden = policy.step(obs, hidden)
        logits = body_safety_mask_logits(logits, [env])
        probabilities = torch.softmax(logits, dim=-1)[0]
        recurrent = int(torch.argmax(probabilities))
        entropy = float((-(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()) / math.log(len(MOVES)))
        hunger = float(obs[0, 11])
        proposed_clearance = float(obs[0, recurrent])
        if hunger >= 0.92:
            action, consensus_hidden = mpc_action(policy, obs, hidden, env, horizon=6)
            return action, consensus_hidden, {
                "horizon": 6,
                "mean_depth": 6.0,
                "uncertainty_stops": 0,
                "entropy": entropy,
                "critical_rescue": 1,
            }
        planning_needed = env.food_visible() or proposed_clearance < 0.72 or no_progress >= 8
        if not planning_needed:
            return recurrent, next_hidden, {
                "horizon": 0,
                "mean_depth": 0.0,
                "uncertainty_stops": 0,
                "entropy": entropy,
                "critical_rescue": 0,
            }

        horizon = 4
        if entropy >= 0.55 or proposed_clearance < 0.48 or no_progress >= 8:
            horizon = 6
        if hunger >= 0.70 or no_progress >= 16:
            horizon = 8

        roots = len(MOVES)
        batch = roots * samples
        root_actions = torch.arange(roots, dtype=torch.long).repeat_interleave(samples)
        imagined_hidden = next_hidden.repeat(batch, 1)
        scores = 0.18 * torch.log(probabilities[root_actions].clamp_min(1e-8))
        cumulative_uncertainty = torch.zeros(batch)
        active = torch.ones(batch, dtype=torch.bool)
        depths = torch.zeros(batch)
        previous = int(torch.argmax(obs[0, CORE_DIM:CORE_DIM + roots]))
        previous_actions = torch.full((batch,), previous, dtype=torch.long)
        uncertainty_stops = 0

        for depth in range(horizon):
            ensemble = policy.predict_core(imagined_hidden, root_actions)
            disagreement = torch.var(ensemble, dim=0).mean(dim=-1)
            head_indices = torch.tensor(rng.integers(0, ensemble.shape[0], size=batch), dtype=torch.long)
            batch_indices = torch.arange(batch)
            consensus = ensemble.mean(dim=0)
            sampled = ensemble[head_indices, batch_indices]
            core = (consensus + 0.35 * (sampled - consensus)).clone()
            core[:, :8] = core[:, :8].clamp(0.0, 1.0)
            core[:, 8:9] = core[:, 8:9].clamp(0.0, 1.0)
            core[:, 9:11] = core[:, 9:11].clamp(-1.0, 1.0)
            core[:, 11:12] = core[:, 11:12].clamp(0.0, 1.0)

            cumulative_uncertainty += disagreement * active.float()
            trusted = active & (cumulative_uncertainty <= uncertainty_budget)
            uncertainty_stops += int(torch.sum(active & ~trusted))
            step_mask = active.float()
            clearance = core[batch_indices, root_actions]
            visible = torch.sigmoid(5.0 * (core[:, 8] - 0.5))
            food_distance = torch.linalg.vector_norm(core[:, 9:11], dim=-1).clamp(0.0, 1.0)
            collision_risk = torch.clamp(0.12 - clearance, min=0.0) * 8.0
            cosine = torch.sum(MOVE_TENSOR[previous_actions] * MOVE_TENSOR[root_actions], dim=-1).clamp(-1.0, 1.0)
            jerk = 0.5 * (1.0 - cosine) if depth == 0 else torch.zeros(batch)
            step_score = (
                0.08 * visible * (1.0 - food_distance)
                - 0.34 * collision_risk
                - jerk_weight * jerk
                - 0.16 * disagreement
            )
            scores += step_mask * step_score
            depths += step_mask

            previous_one_hot = torch.zeros(batch, roots)
            previous_one_hot[batch_indices, root_actions] = 1.0
            estimated_reward = torch.clamp(scores / float(depth + 1), -0.2, 0.2).unsqueeze(-1)
            imagined_obs = torch.cat([core, previous_one_hot, estimated_reward], dim=-1)
            _imagined_logits, _imagined_value, imagined_hidden = policy.step(imagined_obs, imagined_hidden)
            previous_actions = root_actions
            active = trusted
            if not torch.any(active):
                break

        sample_scores = scores.reshape(roots, samples)
        mean = sample_scores.mean(dim=-1)
        downside = torch.quantile(sample_scores, 0.25, dim=-1)
        spread = sample_scores.std(dim=-1, unbiased=False)
        risk_adjusted = 0.75 * mean + 0.25 * downside - 0.25 * spread
        action = int(torch.argmax(risk_adjusted))
        chosen_depth = depths.reshape(roots, samples)[action]
        return action, next_hidden, {
            "horizon": horizon,
            "mean_depth": float(chosen_depth.mean()),
            "uncertainty_stops": uncertainty_stops,
            "entropy": entropy,
            "critical_rescue": 0,
        }


def evaluate_condition(policy, condition, seeds, episodes_per_seed, stochastic_samples=4):
    rows = []
    planning = []
    sight_counts = Counter()
    started = time.perf_counter()
    adaptive_sight = condition == "adaptive_stochastic_hunger"
    for seed in seeds:
        for family_index, family in enumerate(ALL_COURSES):
            for episode in range(episodes_per_seed):
                layout_seed = seed * 100_000 + family_index * 10_000 + episode
                env_type = HungerAdaptiveCourse if adaptive_sight else UnityContinuousCourse
                env = env_type(family, layout_seed)
                obs = torch.tensor(env.reset()).unsqueeze(0)
                hidden = policy.initial_state(1)
                previous_action = 0
                reversals = 0
                no_progress = 0
                rng = np.random.default_rng(layout_seed + 7_000_003)

                for step in range(MAX_STEPS):
                    if adaptive_sight:
                        sight_counts[env.food_sight()] += 1
                    if condition == "recurrent":
                        action, hidden = recurrent_action(policy, obs, hidden, env)
                    elif condition == "fixed_mpc":
                        action, hidden = mpc_action(policy, obs, hidden, env, horizon=4)
                    else:
                        action, hidden, diagnostics = adaptive_stochastic_action(
                            policy,
                            obs,
                            hidden,
                            env,
                            no_progress,
                            rng,
                            samples=stochastic_samples,
                        )
                        planning.append(diagnostics)

                    if np.dot(MOVES[previous_action], MOVES[action]) < 0:
                        reversals += 1
                    result = env.step(action)
                    no_progress = 0 if result.new_cell or result.ate else no_progress + 1
                    previous_action = action
                    obs = torch.tensor(result.obs).unsqueeze(0)
                    if result.done:
                        rows.append(
                            {
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

    elapsed = time.perf_counter() - started
    planning_frames = [item for item in planning if item["horizon"] > 0]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_path_length": float(np.mean([row["path_length"] for row in rows])),
        "mean_revisit_ratio": float(np.mean([row["revisit_ratio"] for row in rows])),
        "mean_reversals": float(np.mean([row["reversals"] for row in rows])),
        "elapsed_seconds": elapsed,
        "planning_fraction": len(planning_frames) / max(1, len(planning)),
        "mean_requested_horizon": float(np.mean([item["horizon"] for item in planning_frames])) if planning_frames else 0.0,
        "mean_realized_depth": float(np.mean([item["mean_depth"] for item in planning_frames])) if planning_frames else 0.0,
        "uncertainty_stops": int(sum(item["uncertainty_stops"] for item in planning)),
        "critical_rescue_frames": int(sum(item["critical_rescue"] for item in planning)),
        "mean_policy_entropy": float(np.mean([item["entropy"] for item in planning])) if planning else 0.0,
        "food_sight_frames": {str(key): value for key, value in sorted(sight_counts.items())},
        "by_family": {
            family: {
                "success_rate": float(np.mean([row["success"] for row in rows if row["family"] == family])),
                "mean_collisions": float(np.mean([row["collisions"] for row in rows if row["family"] == family])),
            }
            for family in ALL_COURSES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = [31] if args.quick else [31, 47, 59]
    episodes = 1 if args.quick else max(1, args.episodes)
    samples = 2 if args.quick else max(2, args.samples)
    policy, _payload = load_checkpoint(args.checkpoint)
    policy.eval()
    conditions = ["recurrent", "fixed_mpc", "adaptive_stochastic", "adaptive_stochastic_hunger"]
    results = {}
    for condition in conditions:
        print(f"evaluating {condition}", flush=True)
        results[condition] = evaluate_condition(policy, condition, seeds, episodes, stochastic_samples=samples)
        print(json.dumps(results[condition], indent=2), flush=True)

    candidate = results["adaptive_stochastic_hunger"]
    fixed = results["fixed_mpc"]
    criteria = {
        "candidate_success_not_below_fixed_mpc": candidate["success_rate"] >= fixed["success_rate"],
        "candidate_collisions_not_above_fixed_mpc": candidate["mean_collisions"] <= fixed["mean_collisions"],
        "adaptive_horizon_was_used": candidate["mean_requested_horizon"] > 4.0,
        "uncertainty_bound_was_active": candidate["uncertainty_stops"] > 0,
    }
    criteria["all_passed"] = all(criteria.values()) and not args.quick
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "seeds": seeds,
            "episodes_per_seed_family": episodes,
            "stochastic_samples": samples,
            "hunger_food_sight": {"normal": 7.0, "hungry": 10.0, "critical": 13.0},
        },
        "conditions": results,
        "criteria": criteria,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"criteria": criteria, "output": str(OUTPUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
