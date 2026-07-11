#!/usr/bin/env python3
"""Causal interventions on the emergent foraging agent's latent trap axis.

The trap label is used only after policy training to fit a diagnostic direction.
The experiment then tests that direction with projection erasure in withheld
U-detours, activation patching in clear fields, dose response, and equal-norm
random controls.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from emergent_foraging_lab import (
    MAX_STEPS,
    ForagingWorld,
    evaluate,
    train_condition,
)
from tiny_lab import OUT, set_seed


def fit_trap_axis(features, labels):
    positive = features[labels == 1.0]
    negative = features[labels == 0.0]
    trap_mean = positive.mean(axis=0)
    open_mean = negative.mean(axis=0)
    axis = trap_mean - open_mean
    axis /= np.linalg.norm(axis) + 1e-8
    trap_projection = float(trap_mean @ axis)
    open_projection = float(open_mean @ axis)
    return axis.astype(np.float32), trap_mean.astype(np.float32), open_mean.astype(np.float32), trap_projection, open_projection


def make_clear_world(seed):
    env = ForagingWorld("open", seed)
    env.blocked = set()
    env.start = (5, 2)
    env.food = (5, 8)
    env.trap_cells = set()
    env.reset()
    return env


def make_blind_corridor_world(seed, forward_action):
    env = ForagingWorld("open", seed)
    if forward_action in {0, 2}:
        env.blocked = {(3, y) for y in range(11)} | {(7, y) for y in range(11)}
    else:
        env.blocked = {(x, 3) for x in range(11)} | {(x, 7) for x in range(11)}
    env.start = (5, 5)
    env.food = (0, 10)
    env.trap_cells = set()
    env.reset()
    env.last_action = forward_action
    return env


def intervene(hidden, mode, axis, scale, open_projection, random_axis=None, trap_mean=None, dose=1.0):
    if mode == "none":
        return hidden
    if mode == "erase_trap":
        projection = hidden @ axis
        return hidden + (open_projection - projection).unsqueeze(-1) * axis
    if mode == "erase_random":
        projection = hidden @ random_axis
        return hidden - projection.unsqueeze(-1) * random_axis
    if mode == "patch_trap":
        return hidden + dose * scale * axis
    if mode == "patch_random":
        return hidden + dose * scale * random_axis
    if mode == "replace_trap_mean":
        return trap_mean.unsqueeze(0).expand_as(hidden)
    raise ValueError(mode)


def run_intervention(
    policy,
    world_kind,
    mode,
    axis,
    scale,
    open_projection,
    random_axis,
    trap_mean,
    dose=1.0,
    runs=96,
    seed=12000,
):
    axis_t = torch.tensor(axis)
    random_t = torch.tensor(random_axis)
    trap_mean_t = torch.tensor(trap_mean)
    rows = []
    logit_shifts = []

    for run in range(runs):
        env = ForagingWorld("u_detour", seed + run) if world_kind == "u_detour" else make_clear_world(seed + run)
        obs = torch.tensor(env.reset()).unsqueeze(0)
        hidden = policy.initial_state(1)
        collisions = 0
        away_steps = 0
        intervention_count = 0
        injected_once = False

        for step in range(MAX_STEPS):
            with torch.no_grad():
                base_logits, _, next_hidden = policy.step(obs, hidden)

            should_intervene = False
            if mode in {"erase_trap", "erase_random"}:
                should_intervene = env.trap_label() > 0.5
            elif mode != "none":
                should_intervene = world_kind == "clear" and step == 5 and not injected_once

            if should_intervene:
                patched = intervene(
                    next_hidden,
                    mode,
                    axis_t,
                    scale,
                    open_projection,
                    random_axis=random_t,
                    trap_mean=trap_mean_t,
                    dose=dose,
                )
                with torch.no_grad():
                    patched_logits = policy.policy(patched)
                base_prob = torch.softmax(base_logits, dim=-1)
                patched_prob = torch.softmax(patched_logits, dim=-1)
                logit_shifts.append(float(torch.mean(torch.abs(patched_prob - base_prob))))
                hidden = patched
                logits = patched_logits
                intervention_count += 1
                injected_once = True
            else:
                hidden = next_hidden
                logits = base_logits

            action = int(torch.argmax(logits, dim=-1))
            old_distance = math.dist(env.pos, env.food)
            result = env.step(action)
            away_steps += int(math.dist(env.pos, env.food) > old_distance + 1e-8)
            collisions += int(result.collision)
            obs = torch.tensor(result.obs).unsqueeze(0)
            if result.done:
                rows.append(
                    {
                        "success": float(result.ate),
                        "steps": step + 1,
                        "collisions": collisions,
                        "away_steps": away_steps,
                        "interventions": intervention_count,
                    }
                )
                break

    return {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_away_steps": float(np.mean([row["away_steps"] for row in rows])),
        "mean_interventions": float(np.mean([row["interventions"] for row in rows])),
        "mean_action_probability_shift": float(np.mean(logit_shifts)) if logit_shifts else 0.0,
    }


def run_blind_corridor_probe(
    policy,
    mode,
    axis,
    scale,
    open_projection,
    random_axis,
    trap_mean,
    dose=1.0,
    runs=160,
    seed=18000,
):
    axis_t = torch.tensor(axis)
    random_t = torch.tensor(random_axis)
    trap_mean_t = torch.tensor(trap_mean)
    reverse_deltas = []
    forward_deltas = []
    total_variations = []
    reverse_argmax = []
    retreat_actions = []

    for run in range(runs):
        forward = run % 4
        reverse = (forward + 2) % 4
        env = make_blind_corridor_world(seed + run, forward)
        obs = torch.tensor(env.observe()).unsqueeze(0)
        hidden = policy.initial_state(1)

        # Establish recent forward motion while keeping food outside sight.
        for _ in range(2):
            with torch.no_grad():
                _, _, hidden = policy.step(obs, hidden)
            result = env.step(forward)
            obs = torch.tensor(result.obs).unsqueeze(0)

        with torch.no_grad():
            base_logits, _, next_hidden = policy.step(obs, hidden)
        patched_hidden = intervene(
            next_hidden,
            mode,
            axis_t,
            scale,
            open_projection,
            random_axis=random_t,
            trap_mean=trap_mean_t,
            dose=dose,
        )
        with torch.no_grad():
            patched_logits = policy.policy(patched_hidden)
        base_prob = torch.softmax(base_logits, dim=-1).squeeze(0)
        patched_prob = torch.softmax(patched_logits, dim=-1).squeeze(0)
        reverse_deltas.append(float(patched_prob[reverse] - base_prob[reverse]))
        forward_deltas.append(float(patched_prob[forward] - base_prob[forward]))
        total_variations.append(float(0.5 * torch.sum(torch.abs(patched_prob - base_prob))))
        reverse_argmax.append(float(int(torch.argmax(patched_prob)) == reverse))

        hidden = patched_hidden
        logits = patched_logits
        retreat_count = 0
        for follow_step in range(3):
            action = int(torch.argmax(logits, dim=-1))
            retreat_count += int(action == reverse)
            result = env.step(action)
            obs = torch.tensor(result.obs).unsqueeze(0)
            with torch.no_grad():
                logits, _, hidden = policy.step(obs, hidden)
        retreat_actions.append(retreat_count)

    return {
        "mean_reverse_probability_delta": float(np.mean(reverse_deltas)),
        "mean_forward_probability_delta": float(np.mean(forward_deltas)),
        "mean_total_variation": float(np.mean(total_variations)),
        "reverse_argmax_rate": float(np.mean(reverse_argmax)),
        "mean_reverse_actions_next_3": float(np.mean(retreat_actions)),
    }


def plot_results(u_results, patch_results, dose_results, blind_results, blind_dose, path):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))

    u_names = list(u_results)
    axes[0].bar(u_names, [u_results[name]["success_rate"] for name in u_names], color=["#287271", "#e76f51", "#6b7280"])
    axes[0].set_title("U-detour: latent erasure")
    axes[0].set_ylabel("mushroom success")
    axes[0].tick_params(axis="x", rotation=18)

    patch_names = list(patch_results)
    axes[1].bar(
        patch_names,
        [patch_results[name]["mean_away_steps"] for name in patch_names],
        color=["#287271", "#e76f51", "#6b7280", "#e9c46a"],
    )
    axes[1].set_title("Clear field: patched trajectory")
    axes[1].set_ylabel("steps away from mushroom")
    axes[1].tick_params(axis="x", rotation=18)

    doses = sorted(dose_results)
    axes[2].plot(doses, [dose_results[d]["mean_away_steps"] for d in doses], marker="o", label="away steps")
    axes[2].plot(
        doses,
        [dose_results[d]["mean_action_probability_shift"] * 20.0 for d in doses],
        marker="s",
        label="action shift x20",
    )
    axes[2].set_title("Trap-axis dose response")
    axes[2].set_xlabel("injection dose")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    blind_names = list(blind_results)
    axes[3].bar(
        blind_names,
        [blind_results[name]["mean_reverse_probability_delta"] for name in blind_names],
        color=["#287271", "#e76f51", "#6b7280"],
    )
    axes[3].plot(
        range(len(blind_names)),
        [blind_results[name]["mean_total_variation"] for name in blind_names],
        color="#111827",
        marker="o",
        label="total variation",
    )
    axes[3].set_title("Blind corridor: instant motor effect")
    axes[3].set_ylabel("probability change")
    axes[3].tick_params(axis="x", rotation=18)
    axes[3].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(52)
    policy, _ = train_condition(
        "recurrent_curiosity_intervention",
        recurrent=True,
        curiosity_beta=0.018,
        food_reward=2.0,
        seed=52,
    )

    _, _, probe = evaluate(
        policy,
        "u_detour",
        runs=140,
        seed=9000,
        stochastic=True,
        collect_probe=True,
    )
    axis, trap_mean, open_mean, trap_projection, open_projection = fit_trap_axis(probe[0], probe[2])
    scale = max(0.1, trap_projection - open_projection)
    rng = np.random.default_rng(77)
    random_axis = rng.normal(size=axis.shape).astype(np.float32)
    random_axis -= axis * float(random_axis @ axis)
    random_axis /= np.linalg.norm(random_axis) + 1e-8

    u_results = {
        "normal": run_intervention(policy, "u_detour", "none", axis, scale, open_projection, random_axis, trap_mean),
        "trap_erased": run_intervention(
            policy, "u_detour", "erase_trap", axis, scale, open_projection, random_axis, trap_mean
        ),
        "random_erased": run_intervention(
            policy, "u_detour", "erase_random", axis, scale, open_projection, random_axis, trap_mean
        ),
    }
    patch_results = {
        "normal": run_intervention(policy, "clear", "none", axis, scale, open_projection, random_axis, trap_mean),
        "trap_patch": run_intervention(
            policy, "clear", "patch_trap", axis, scale, open_projection, random_axis, trap_mean
        ),
        "random_patch": run_intervention(
            policy, "clear", "patch_random", axis, scale, open_projection, random_axis, trap_mean
        ),
        "trap_mean": run_intervention(
            policy, "clear", "replace_trap_mean", axis, scale, open_projection, random_axis, trap_mean
        ),
    }
    dose_results = {
        dose: run_intervention(
            policy,
            "clear",
            "patch_trap",
            axis,
            scale,
            open_projection,
            random_axis,
            trap_mean,
            dose=dose,
        )
        for dose in (0.0, 0.5, 1.0, 1.5)
    }
    blind_results = {
        "none": run_blind_corridor_probe(
            policy, "none", axis, scale, open_projection, random_axis, trap_mean
        ),
        "trap_patch": run_blind_corridor_probe(
            policy, "patch_trap", axis, scale, open_projection, random_axis, trap_mean
        ),
        "random_patch": run_blind_corridor_probe(
            policy, "patch_random", axis, scale, open_projection, random_axis, trap_mean
        ),
    }
    blind_dose = {
        dose: run_blind_corridor_probe(
            policy,
            "patch_trap",
            axis,
            scale,
            open_projection,
            random_axis,
            trap_mean,
            dose=dose,
        )
        for dose in (0.0, 0.5, 1.0, 1.5)
    }

    erasure_effect = u_results["normal"]["success_rate"] - u_results["trap_erased"]["success_rate"]
    random_erasure_effect = u_results["normal"]["success_rate"] - u_results["random_erased"]["success_rate"]
    patch_effect = patch_results["trap_patch"]["mean_away_steps"] - patch_results["normal"]["mean_away_steps"]
    random_patch_effect = patch_results["random_patch"]["mean_away_steps"] - patch_results["normal"]["mean_away_steps"]
    dose_away = [dose_results[d]["mean_away_steps"] for d in sorted(dose_results)]
    monotonic_dose = all(a <= b + 1e-8 for a, b in zip(dose_away, dose_away[1:]))
    blind_trap_effect = blind_results["trap_patch"]["mean_reverse_probability_delta"]
    blind_random_effect = blind_results["random_patch"]["mean_reverse_probability_delta"]
    blind_dose_reverse = [blind_dose[d]["mean_reverse_probability_delta"] for d in sorted(blind_dose)]
    blind_monotonic = all(a <= b + 1e-8 for a, b in zip(blind_dose_reverse, blind_dose_reverse[1:]))
    causal_supported = (
        erasure_effect > max(0.10, random_erasure_effect + 0.05)
        and blind_trap_effect > max(0.005, blind_random_effect + 0.003)
        and blind_monotonic
    )

    payload = {
        "axis": {
            "fit_trajectories": 140,
            "trap_projection": trap_projection,
            "open_projection": open_projection,
            "projection_gap": scale,
        },
        "u_detour_erasure": u_results,
        "clear_field_patching": patch_results,
        "dose_response": {str(key): value for key, value in dose_results.items()},
        "blind_corridor_patching": blind_results,
        "blind_corridor_dose_response": {str(key): value for key, value in blind_dose.items()},
        "effects": {
            "trap_erasure_success_drop": erasure_effect,
            "random_erasure_success_drop": random_erasure_effect,
            "trap_patch_away_step_increase": patch_effect,
            "random_patch_away_step_increase": random_patch_effect,
            "monotonic_dose_response": monotonic_dose,
            "blind_trap_reverse_probability_increase": blind_trap_effect,
            "blind_random_reverse_probability_increase": blind_random_effect,
            "blind_monotonic_dose_response": blind_monotonic,
            "causal_trap_axis_supported": causal_supported,
        },
        "claim_boundary": (
            "A positive result supports a causally active distributed hidden-state direction associated with trap context. "
            "It does not isolate a discrete neural circuit, prove the axis is the policy's only trap representation, "
            "or establish human-like spatial concepts or conscious counterfactual thought."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "latent_trap_intervention_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_results(
        u_results,
        patch_results,
        dose_results,
        blind_results,
        blind_dose,
        OUT / "latent_trap_intervention_summary.png",
    )
    print("Latent trap intervention lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
