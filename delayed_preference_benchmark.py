#!/usr/bin/env python3
"""Matched delayed-preference assay for recurrence and grounded valence.

The environment has two actions and a hidden preferred action. A decision earns
+1 for the preferred action and -1 otherwise. The only evidence about the
preference is the grounded outcome pulse on the following step. Preferences
reverse halfway through an episode without an explicit cue.

The feedforward control receives an explicit eight-frame window. Training
samples delays on both sides of that window; evaluation reports a delay sweep.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


OUT = Path("outputs")
METRICS = OUT / "delayed_preference_benchmark_metrics.json"
PLOT = OUT / "delayed_preference_benchmark.png"
CHECKPOINT_DIR = Path("checkpoints/delayed_preference_benchmark")
CONTEXT = 8
OBS_DIM = 1 + 2 + 2 + 4
DECISIONS = 12
TRAIN_DELAYS = (2, 4, 6, 8, 12, 16, 20)
EVAL_DELAYS = (2, 4, 6, 8, 10, 12, 16, 20, 28)
CONDITIONS = (
    ("feedforward", False, False),
    ("feedforward_valence", False, True),
    ("recurrent", True, False),
    ("recurrent_valence", True, True),
)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class PreferencePolicy(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = 32 if recurrent else 55
        if recurrent:
            self.core = nn.GRUCell(OBS_DIM, self.hidden_dim)
        else:
            self.core = nn.Sequential(nn.Linear(OBS_DIM * CONTEXT, self.hidden_dim), nn.Tanh())
        self.actor = nn.Linear(self.hidden_dim, 2)

    def initial_state(self, batch):
        if self.recurrent:
            return torch.zeros(batch, self.hidden_dim)
        return torch.zeros(batch, CONTEXT, OBS_DIM)

    def step(self, obs, state, reset_memory=False):
        if self.recurrent:
            if reset_memory:
                state = torch.zeros_like(state)
            state = self.core(obs, state)
            features = state
        else:
            state = torch.cat((state[:, 1:], obs.unsqueeze(1)), dim=1)
            features = self.core(state.reshape(obs.shape[0], -1))
        return self.actor(features), state


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def observation(batch, decision, previous_action, outcome, valence_enabled, valence_mode, rng):
    obs = np.zeros((batch, OBS_DIM), dtype=np.float32)
    obs[:, 0] = float(decision)
    obs[np.arange(batch), 1 + previous_action] = 1.0
    if valence_enabled:
        signed = outcome.copy()
        magnitude = np.abs(outcome)
        if valence_mode == "zero":
            signed.fill(0.0)
            magnitude.fill(0.0)
        elif valence_mode == "sign_flip":
            signed *= -1.0
        elif valence_mode == "shuffle":
            signed = signed[rng.permutation(batch)]
        obs[:, 3] = signed
        obs[:, 4] = magnitude
    obs[:, 5:] = rng.normal(0.0, 0.35, size=(batch, 4))
    return torch.tensor(obs)


def run_episode(
    model,
    recurrent,
    valence_enabled,
    delay,
    batch,
    rng,
    training,
    reset_memory=False,
    valence_mode="normal",
):
    state = model.initial_state(batch)
    # Exactly balanced preferences prevent a no-feedback policy from appearing
    # adaptive because of evaluation imbalance.
    preference = np.arange(batch, dtype=np.int64) % 2
    rng.shuffle(preference)
    previous_action = np.zeros(batch, dtype=np.int64)
    zero_outcome = np.zeros(batch, dtype=np.float32)
    decision_log_probs = []
    decision_entropies = []
    decision_rewards = []
    decision_correct = []

    for decision_index in range(DECISIONS):
        if decision_index == DECISIONS // 2:
            preference = 1 - preference
        obs = observation(
            batch, True, previous_action, zero_outcome, valence_enabled, valence_mode, rng
        )
        logits, state = model.step(obs, state, reset_memory=reset_memory)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if training else logits.argmax(dim=-1)
        action_np = action.detach().numpy()
        correct = action_np == preference
        reward = np.where(correct, 1.0, -1.0).astype(np.float32)
        decision_log_probs.append(distribution.log_prob(action))
        decision_entropies.append(distribution.entropy())
        decision_rewards.append(torch.tensor(reward))
        decision_correct.append(correct)
        previous_action = action_np

        # Grounded outcome appears once, immediately after the decision.
        pulse = observation(
            batch, False, previous_action, reward, valence_enabled, valence_mode, rng
        )
        _, state = model.step(pulse, state, reset_memory=reset_memory)
        for _ in range(max(0, delay - 2)):
            blank = observation(
                batch, False, previous_action, zero_outcome, valence_enabled, valence_mode, rng
            )
            _, state = model.step(blank, state, reset_memory=reset_memory)

    return {
        "log_probs": torch.stack(decision_log_probs, dim=1),
        "entropies": torch.stack(decision_entropies, dim=1),
        "rewards": torch.stack(decision_rewards, dim=1),
        "correct": np.stack(decision_correct, axis=1),
    }


def train_condition(recurrent, valence_enabled, seed, updates, batch_size):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = PreferencePolicy(recurrent)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.8e-3)
    curve = []
    for update in range(updates):
        delay = TRAIN_DELAYS[int(rng.integers(len(TRAIN_DELAYS)))]
        episode = run_episode(model, recurrent, valence_enabled, delay, batch_size, rng, True)
        rewards = episode["rewards"]
        # Immediate signed outcomes are the policy-gradient learning signal.
        # Excluding the first decision avoids training on an unknowable choice.
        policy_loss = -(episode["log_probs"][:, 1:] * rewards[:, 1:]).mean()
        entropy = episode["entropies"][:, 1:].mean()
        loss = policy_loss - 0.012 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 25 == 0 or update == updates - 1:
            correct = episode["correct"]
            curve.append(
                {
                    "update": update + 1,
                    "delay": delay,
                    "accuracy": float(correct[:, 1:].mean()),
                    "post_switch_accuracy": float(correct[:, DECISIONS // 2 :].mean()),
                }
            )
    return model.eval(), curve


def evaluate(model, recurrent, valence_enabled, delay, seed, episodes, reset_memory=False, valence_mode="normal"):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        result = run_episode(
            model,
            recurrent,
            valence_enabled,
            delay,
            episodes,
            rng,
            False,
            reset_memory=reset_memory,
            valence_mode=valence_mode,
        )
    correct = result["correct"]
    post = correct[:, DECISIONS // 2 :]
    return {
        "accuracy": float(correct[:, 1:].mean()),
        "pre_switch_accuracy": float(correct[:, 1 : DECISIONS // 2].mean()),
        "post_switch_accuracy": float(post.mean()),
        "first_post_switch_accuracy": float(post[:, 0].mean()),
        "second_post_switch_accuracy": float(post[:, 1].mean()),
        "final_post_switch_accuracy": float(post[:, -2:].mean()),
        "mean_reward": float(result["rewards"][:, 1:].mean()),
    }


def aggregate(rows):
    keys = rows[0].keys()
    return {
        key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows]))}
        for key in keys
    }


def paired_contrast(left_rows, right_rows, metric, seed=7421):
    differences = np.asarray(
        [left[metric] - right[metric] for left, right in zip(left_rows, right_rows)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(20_000, len(differences)), replace=True).mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "bootstrap_95_ci": [float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "per_seed_differences": [float(value) for value in differences],
    }


def plot(results):
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = ("#7A848C", "#4F9D75", "#D9822B", "#1976A3")
    for (name, _, _), color in zip(CONDITIONS, colors):
        means = [results[name][str(delay)]["accuracy"]["mean"] for delay in EVAL_DELAYS]
        stds = [results[name][str(delay)]["accuracy"]["std"] for delay in EVAL_DELAYS]
        ax.plot(EVAL_DELAYS, means, marker="o", linewidth=2, color=color, label=name.replace("_", " "))
        ax.fill_between(EVAL_DELAYS, np.asarray(means) - stds, np.asarray(means) + stds, color=color, alpha=0.13)
    ax.axvline(CONTEXT + 1, color="#20262B", linestyle="--", alpha=0.45, label="MLP history boundary")
    ax.axhline(0.5, color="#20262B", linestyle=":", alpha=0.5, label="chance")
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Steps between preference decisions")
    ax.set_ylabel("Preferred-action accuracy")
    ax.set_title("Grounded Outcome Use Across Temporal Delays")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)


def run(args):
    OUT.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {name: {str(delay): [] for delay in EVAL_DELAYS} for name, _, _ in CONDITIONS}
    curves = {}
    params = {}
    models = {}
    for name, recurrent, valence_enabled in CONDITIONS:
        curves[name] = []
        models[name] = []
        for seed_index in range(args.seeds):
            seed = args.seed + seed_index * 1009
            print(f"training {name} seed={seed}", flush=True)
            model, curve = train_condition(recurrent, valence_enabled, seed, args.updates, args.batch_size)
            curves[name].append(curve)
            models[name].append(model)
            params[name] = parameter_count(model)
            for delay in EVAL_DELAYS:
                raw[name][str(delay)].append(
                    evaluate(model, recurrent, valence_enabled, delay, 80_000 + seed_index * 1000, args.eval_episodes)
                )
        for seed_index, model in enumerate(models[name]):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "recurrent": recurrent,
                    "valence_enabled": valence_enabled,
                    "context": CONTEXT,
                    "obs_dim": OBS_DIM,
                    "seed": args.seed + seed_index * 1009,
                },
                CHECKPOINT_DIR / f"{name}_seed{seed_index}.pt",
            )
    results = {name: {delay: aggregate(rows) for delay, rows in delays.items()} for name, delays in raw.items()}
    contrasts = {
        "recurrent_valence_minus_feedforward_valence": {
            str(delay): paired_contrast(
                raw["recurrent_valence"][str(delay)],
                raw["feedforward_valence"][str(delay)],
                "accuracy",
                seed=7421 + delay,
            )
            for delay in EVAL_DELAYS
        },
        "delay20_convergence_rate_at_70_percent": {
            name: float(
                np.mean([row["accuracy"] >= 0.70 for row in raw[name]["20"]])
            )
            for name, _, _ in CONDITIONS
        },
    }
    ablations = {}
    raw_ablations = {}
    for mode, kwargs in {
        "normal": {},
        "hidden_reset": {"reset_memory": True},
        "valence_zero": {"valence_mode": "zero"},
        "valence_sign_flip": {"valence_mode": "sign_flip"},
        "valence_shuffle": {"valence_mode": "shuffle"},
    }.items():
        raw_ablations[mode] = {
            str(delay): [
                evaluate(
                    model,
                    True,
                    True,
                    delay,
                    190_000 + seed_index * 1000,
                    args.eval_episodes,
                    **kwargs,
                )
                for seed_index, model in enumerate(models["recurrent_valence"])
            ]
            for delay in EVAL_DELAYS
        }
        ablations[mode] = {
            delay: aggregate(rows) for delay, rows in raw_ablations[mode].items()
        }
    payload = {
        "design": {
            "seeds": args.seeds,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "train_delays": TRAIN_DELAYS,
            "eval_delays": EVAL_DELAYS,
            "feedforward_context_frames": CONTEXT,
            "parameter_counts": params,
            "balanced_hidden_preferences": True,
            "claim_boundary": "Tests use of grounded outcome feedback across delays in a small POMDP; it does not establish universal recurrent superiority or evaluate navigation skill.",
        },
        "results": results,
        "per_seed_results": raw,
        "contrasts": contrasts,
        "recurrent_valence_ablations": ablations,
        "per_seed_recurrent_valence_ablations": raw_ablations,
        "training_curves": curves,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    plot(results)
    print("\nDelayed preference benchmark complete")
    for name, _, _ in CONDITIONS:
        short = results[name]["4"]["accuracy"]
        long = results[name]["20"]["accuracy"]
        print(
            f"{name:22s} params={params[name]:4d} delay4={short['mean']:.3f}+/-{short['std']:.3f} "
            f"delay20={long['mean']:.3f}+/-{long['std']:.3f}"
        )
    print(f"Metrics: {METRICS}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=29041)
    parser.add_argument("--updates", type=int, default=520)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.updates = 45
        args.batch_size = 48
        args.eval_episodes = 64
        args.seeds = 1
    return args


if __name__ == "__main__":
    run(parse_args())
