#!/usr/bin/env python3
"""Teach hidden-goal exterior detours while preserving existing navigation.

Unity's C course places an invisible goal behind the closed back wall. This is
harder than the original withheld C-shape, whose goal lies beyond its opening.
The lab compares a frozen checkpoint, readout-only adaptation, and recurrent
post-training on randomized rotations and dimensions. A shortest-path teacher
is available only during training and is removed for every evaluation.
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

from unity_posttraining_lab import (
    ALL_COURSES,
    STEP_LENGTH,
    UnityContinuousCourse,
    body_safety_mask_logits,
    evaluate_continuous,
)
from upgraded_foraging_pipeline import MOVES, evaluate_policy, load_checkpoint


FAMILY = "c_back_goal"
BASE = Path("checkpoints/starvation_posttrained/best.pt")
CHECKPOINT = Path("checkpoints/back_goal_posttrained/best.pt")
METRICS = Path("outputs/back_goal_navigation_metrics.json")
OLD_TRAIN_FAMILIES = ["pocket", "l_wall", "offset_barriers", "u_trap", "narrow_corridor"]


def topological_oracle_action(env):
    """Follow the map-distance gradient; used only as a post-training teacher."""
    current = tuple(np.floor(env.pos).astype(int))
    choices = []
    for action, move in enumerate(MOVES):
        direction = move.astype(np.float32) / np.linalg.norm(move)
        candidate = env.pos + direction * STEP_LENGTH
        if env.body_blocked(candidate):
            continue
        neighbor = current[0] + int(move[0]), current[1] + int(move[1])
        if neighbor in env.blocked or neighbor not in env.distance_to_food:
            continue
        if move[0] != 0 and move[1] != 0:
            side_a = current[0] + int(move[0]), current[1]
            side_b = current[0], current[1] + int(move[1])
            if side_a in env.blocked and side_b in env.blocked:
                continue
        old_move = MOVES[env.last_action]
        cosine = float(np.dot(old_move, move)) / (
            float(np.linalg.norm(old_move)) * float(np.linalg.norm(move))
        )
        choices.append((env.distance_to_food[neighbor] + 0.025 * (1.0 - cosine), action))
    if choices:
        return min(choices)[1]
    return int(np.argmax(env.observe()[: len(MOVES)]))


def make_world(rng, back_goal_probability=0.60):
    if rng.random() < back_goal_probability:
        family = FAMILY
    else:
        family = OLD_TRAIN_FAMILIES[int(rng.integers(len(OLD_TRAIN_FAMILIES)))]
    return UnityContinuousCourse(family, int(rng.integers(1, 10_000_000)))


def train_candidate(base, mode, updates, seed, env_count=20, rollout=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy, source_payload = load_checkpoint(base)
    reference = copy.deepcopy(policy).eval()
    for parameter in policy.parameters():
        parameter.requires_grad = False
    modules = [policy.actor]
    if mode == "recurrent":
        modules.extend([policy.encoder, policy.memory, policy.critic])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=4e-4 if mode == "readout_only" else 1.8e-4,
        weight_decay=0.015,
    )
    envs = [make_world(rng) for _ in range(env_count)]
    obs = torch.tensor(np.stack([env.reset() for env in envs]))
    hidden = policy.initial_state(env_count)
    reference_hidden = reference.initial_state(env_count)
    recent = deque(maxlen=240)
    curves = {"loss": [], "success": [], "old_policy_kl": [], "validation": []}
    best_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    best_score = (-1.0, -1e9)

    for update in range(updates):
        imitation_losses, old_kl_losses, entropies, masks = [], [], [], []
        teacher_probability = 0.92 - 0.42 * update / max(1, updates - 1)
        for _ in range(rollout):
            logits, _value, hidden = policy.step(obs, hidden)
            logits = body_safety_mask_logits(logits, envs)
            with torch.no_grad():
                reference_logits, _reference_value, reference_hidden = reference.step(
                    obs, reference_hidden
                )
                reference_logits = body_safety_mask_logits(reference_logits, envs)
            distribution = torch.distributions.Categorical(logits=logits)
            teacher = torch.tensor(
                [topological_oracle_action(env) for env in envs], dtype=torch.long
            )
            sampled = distribution.sample()
            use_teacher = torch.rand(env_count) < teacher_probability
            actions = torch.where(use_teacher, teacher, sampled)
            old_mask = torch.tensor(
                [env.family != FAMILY for env in envs], dtype=torch.bool
            )
            imitation_losses.append(F.cross_entropy(logits, teacher))
            if bool(torch.any(old_mask)):
                old_kl_losses.append(F.kl_div(
                    F.log_softmax(logits[old_mask], dim=-1),
                    F.softmax(reference_logits[old_mask], dim=-1),
                    reduction="batchmean",
                ))
            entropies.append(distribution.entropy())

            next_obs, alive = [], []
            for index, env in enumerate(envs):
                result = env.step(int(actions[index]))
                if result.done:
                    recent.append(float(result.ate) if env.family == FAMILY else 1.0)
                    envs[index] = make_world(rng)
                    next_obs.append(envs[index].reset())
                    alive.append(0.0)
                else:
                    next_obs.append(result.obs)
                    alive.append(1.0)
            alive_tensor = torch.tensor(alive, dtype=torch.float32)
            obs = torch.tensor(np.stack(next_obs))
            hidden = hidden * alive_tensor.unsqueeze(-1)
            reference_hidden = reference_hidden * alive_tensor.unsqueeze(-1)
            masks.append(alive_tensor)

        imitation = torch.stack(imitation_losses).mean()
        old_kl = torch.stack(old_kl_losses).mean() if old_kl_losses else torch.tensor(0.0)
        entropy = -0.004 * torch.stack(entropies).mean()
        loss = imitation + 0.10 * old_kl + entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
        optimizer.step()
        hidden = hidden.detach()
        reference_hidden = reference_hidden.detach()
        curves["loss"].append(float(loss.detach()))
        curves["success"].append(float(np.mean(recent)) if recent else 0.0)
        curves["old_policy_kl"].append(float(old_kl.detach()))

        if update % 40 == 0 or update == updates - 1:
            policy.eval()
            validation = evaluate_continuous(
                policy, [FAMILY], [17, 29], episodes_per_seed=5
            )
            policy.train()
            curves["validation"].append({"update": update, **validation})
            score = (validation["success_rate"], -validation["mean_steps"])
            if score > best_score:
                best_score = score
                best_state = {
                    name: value.detach().clone() for name, value in policy.state_dict().items()
                }
            print(
                f"{mode} update={update:03d} "
                f"back_goal={validation['success_rate']:.3f} "
                f"loss={float(loss.detach()):.3f} old_kl={float(old_kl.detach()):.4f}",
                flush=True,
            )
    policy.load_state_dict(best_state)
    return policy.eval(), source_payload, curves


def evaluate_candidate(policy, back_seeds, old_seeds, episodes):
    back = evaluate_continuous(
        policy, [FAMILY], back_seeds, episodes_per_seed=episodes
    )
    back_reset = evaluate_continuous(
        policy, [FAMILY], back_seeds, episodes_per_seed=episodes, reset_memory=True
    )
    old = evaluate_continuous(
        policy, ALL_COURSES, old_seeds, episodes_per_seed=max(3, episodes // 2)
    )
    grid = evaluate_policy(
        policy, ["c_shape", "zigzag_gate"], episodes_per_family=30, seed=250_000
    )
    return {
        "back_goal": back,
        "back_goal_memory_reset": back_reset,
        "existing_continuous_courses": old,
        "original_grid_withheld": grid,
    }


def acceptance(result, baseline):
    return {
        "back_goal_success_at_least_90_percent": result["back_goal"]["success_rate"] >= 0.90,
        "back_goal_collisions_at_most_1": result["back_goal"]["mean_collisions"] <= 1.0,
        "existing_courses_at_least_90_percent": result["existing_continuous_courses"]["success_rate"] >= 0.90,
        "existing_course_regression_at_most_3_points": (
            result["existing_continuous_courses"]["success_rate"]
            >= baseline["existing_continuous_courses"]["success_rate"] - 0.03
        ),
        "original_grid_withheld_at_least_90_percent": result["original_grid_withheld"]["success_rate"] >= 0.90,
    }


def save_checkpoint(policy, source_payload, mode, metrics):
    payload = dict(source_payload)
    payload["state_dict"] = policy.state_dict()
    payload["config"] = dict(source_payload["config"])
    payload["config"].update({
        "posttraining": "back_goal_navigation_v1",
        "posttraining_scope": mode,
        "training_teacher_removed_at_evaluation": True,
    })
    payload["back_goal_navigation_metrics"] = metrics
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--readout-updates", type=int, default=200)
    parser.add_argument("--recurrent-updates", type=int, default=320)
    args = parser.parse_args()
    if args.quick:
        args.readout_updates = 24
        args.recurrent_updates = 32
    back_seeds = [101] if args.quick else [101, 211, 307, 419, 523]
    old_seeds = [31] if args.quick else [31, 47, 59]
    episodes = 3 if args.quick else 16

    baseline_policy, _ = load_checkpoint(args.base)
    print("evaluating frozen baseline", flush=True)
    baseline = evaluate_candidate(baseline_policy, back_seeds, old_seeds, episodes)

    readout, source_payload, readout_curves = train_candidate(
        args.base, "readout_only", args.readout_updates, 2601
    )
    print("evaluating readout-only candidate", flush=True)
    readout_result = evaluate_candidate(readout, back_seeds, old_seeds, episodes)
    readout_criteria = acceptance(readout_result, baseline)

    recurrent, source_payload, recurrent_curves = train_candidate(
        args.base, "recurrent", args.recurrent_updates, 2701
    )
    print("evaluating recurrent candidate", flush=True)
    recurrent_result = evaluate_candidate(recurrent, back_seeds, old_seeds, episodes)
    recurrent_criteria = acceptance(recurrent_result, baseline)

    readout_pass = all(readout_criteria.values()) and not args.quick
    recurrent_pass = all(recurrent_criteria.values()) and not args.quick
    if readout_pass:
        selected_mode, selected_policy = "readout_only", readout
    elif recurrent_pass:
        selected_mode, selected_policy = "recurrent", recurrent
    else:
        selected_mode, selected_policy = "rejected", None
    metrics = {
        "protocol": {
            "base_checkpoint": str(args.base),
            "family": FAMILY,
            "procedural_rotations": 4,
            "teacher_available_during_training_only": True,
            "selection_prefers_readout_only_when_both_pass": True,
        },
        "baseline": baseline,
        "readout_only": readout_result,
        "readout_only_criteria": readout_criteria,
        "recurrent": recurrent_result,
        "recurrent_criteria": recurrent_criteria,
        "selected_mode": selected_mode,
        "curves": {"readout_only": readout_curves, "recurrent": recurrent_curves},
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if selected_policy is not None:
        save_checkpoint(selected_policy, source_payload, selected_mode, metrics)
        print(f"exported {CHECKPOINT}")
    else:
        print("both candidates rejected")
    print(json.dumps({
        "baseline_back_goal": baseline["back_goal"]["success_rate"],
        "readout_back_goal": readout_result["back_goal"]["success_rate"],
        "recurrent_back_goal": recurrent_result["back_goal"]["success_rate"],
        "selected_mode": selected_mode,
        "metrics": str(METRICS),
    }, indent=2))


if __name__ == "__main__":
    main()
