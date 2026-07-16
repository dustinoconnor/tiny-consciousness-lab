#!/usr/bin/env python3
"""Reward-learned recurrent rhythm policy for the live MIDI generator.

The policy receives no target rhythm sequence. It samples note durations and is
rewarded for bar alignment, useful variation, groove continuity, development,
and delayed return of its own opening rhythm motif. These objectives are
engineered; the duration sequence itself is learned rather than scripted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from midi_transfer_lab import CHORDS, MOTIF, PROGRESSION, SECTION, STEPS, make_phrase, seed_all


DURATION_BEATS = np.asarray((0.25, 0.5, 1.0, 2.0), dtype=np.float32)
RHYTHM_OBS_DIM = 8 + 4 + 3 + 4 + 1 + 7 + 1
CHECKPOINT = Path("checkpoints/midi_transfer/learned_rhythm.pt")
METRICS = Path("outputs/midi_rhythm_learning_metrics.json")


def rhythm_observation(step, pitch_token, confidence, previous_action, bar_elapsed):
    obs = np.zeros(RHYTHM_OBS_DIM, dtype=np.float32)
    offset = 0
    obs[offset + step % MOTIF] = 1.0
    offset += 8
    obs[offset + SECTION[min(7, step // MOTIF)]] = 1.0
    offset += 4
    obs[offset + PROGRESSION[min(7, step // MOTIF)]] = 1.0
    offset += 3
    obs[offset + int(previous_action)] = 1.0
    offset += 4
    obs[offset] = float(np.clip(bar_elapsed / 4.0, 0.0, 1.5))
    offset += 1
    obs[offset + int(pitch_token)] = 1.0
    obs[-1] = float(confidence)
    return obs


class RhythmPolicy(nn.Module):
    def __init__(self, hidden=40):
        super().__init__()
        self.hidden_size = hidden
        self.cell = nn.GRUCell(RHYTHM_OBS_DIM, hidden)
        self.head = nn.Linear(hidden, len(DURATION_BEATS))

    def initial_state(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)

    def step(self, obs, hidden):
        hidden = self.cell(obs, hidden)
        return self.head(hidden), hidden


def rhythm_scores(actions):
    """Return objective components for an [batch, 64] action tensor."""
    durations = torch.tensor(DURATION_BEATS, device=actions.device)[actions]
    bars = durations.reshape(actions.shape[0], 8, MOTIF)
    bar_totals = bars.sum(dim=-1)
    alignment = torch.exp(-1.15 * torch.abs(bar_totals - 4.0)).mean(dim=1)

    diversity_rows = []
    groove_rows = []
    for bar in range(8):
        bar_actions = actions[:, bar * MOTIF : (bar + 1) * MOTIF]
        one_hot = torch.nn.functional.one_hot(bar_actions, len(DURATION_BEATS)).float()
        unique = (one_hot.sum(dim=1) > 0).float().sum(dim=1)
        diversity_rows.append(torch.clamp((unique - 1.0) / 2.0, 0.0, 1.0))
        change_rate = (bar_actions[:, 1:] != bar_actions[:, :-1]).float().mean(dim=1)
        groove_rows.append(torch.clamp(1.0 - torch.abs(change_rate - 0.48) / 0.48, 0.0, 1.0))
    diversity = torch.stack(diversity_rows, dim=1).mean(dim=1)
    groove = torch.stack(groove_rows, dim=1).mean(dim=1)

    opening = actions[:, :MOTIF]
    return_a = (actions[:, 32:40] == opening).float().mean(dim=1)
    return_b = (actions[:, 56:64] == opening).float().mean(dim=1)
    raw_recall = 0.5 * (return_a + return_b)
    # A constant rhythm cannot claim meaningful motif memory merely because it
    # matches itself. Recall is grounded by diversity in the opening pattern.
    recall = raw_recall * diversity_rows[0]

    variation = (actions[:, 8:16] != opening).float().mean(dim=1)
    development = torch.clamp(1.0 - torch.abs(variation - 0.38) / 0.38, 0.0, 1.0)
    total = 0.35 * alignment + 0.18 * diversity + 0.14 * groove + 0.23 * recall + 0.10 * development
    return {
        "total": total,
        "bar_alignment": alignment,
        "diversity": diversity,
        "groove": groove,
        "motif_recall": recall,
        "development": development,
    }


def pitch_batch(rng, batch_size):
    phrases = np.stack([make_phrase(rng) for _ in range(batch_size)])
    confidence = rng.uniform(0.18, 0.78, size=phrases.shape).astype(np.float32)
    return phrases, confidence


def rollout(model, rng, batch_size, reset_memory=False, greedy=False):
    pitches, confidences = pitch_batch(rng, batch_size)
    hidden = model.initial_state(batch_size)
    previous = np.full(batch_size, 1, dtype=np.int64)
    elapsed = np.zeros(batch_size, dtype=np.float32)
    actions, log_probs, entropies = [], [], []
    for step in range(STEPS):
        if reset_memory:
            hidden = model.initial_state(batch_size)
        obs = np.stack(
            [
                rhythm_observation(step, pitches[i, step], confidences[i, step], previous[i], elapsed[i])
                for i in range(batch_size)
            ]
        )
        logits, hidden = model.step(torch.tensor(obs), hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else distribution.sample()
        actions.append(action)
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        action_np = action.detach().numpy()
        elapsed += DURATION_BEATS[action_np]
        previous = action_np
        if (step + 1) % MOTIF == 0:
            elapsed.fill(0.0)
    action_tensor = torch.stack(actions, dim=1)
    return action_tensor, torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)


def summarize(score_map, actions):
    result = {name: float(values.mean()) for name, values in score_map.items()}
    counts = torch.bincount(actions.reshape(-1), minlength=len(DURATION_BEATS)).float()
    result["duration_distribution"] = {
        str(float(duration)): float(counts[index] / counts.sum())
        for index, duration in enumerate(DURATION_BEATS)
    }
    return result


def evaluate(model, seed, episodes=512, reset_memory=False):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        actions, _, _ = rollout(model, rng, episodes, reset_memory=reset_memory)
        return summarize(rhythm_scores(actions), actions)


def random_baseline(seed, episodes=512):
    rng = np.random.default_rng(seed)
    actions = torch.tensor(rng.choice(4, size=(episodes, STEPS), p=(0.18, 0.52, 0.24, 0.06)))
    return summarize(rhythm_scores(actions), actions)


def fixed_baseline(episodes=512):
    actions = torch.full((episodes, STEPS), 1, dtype=torch.long)
    return summarize(rhythm_scores(actions), actions)


def train(seed, updates, batch_size):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = RhythmPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.8e-3)
    baseline = 0.45
    curve = []
    for update in range(updates):
        actions, log_probs, entropies = rollout(model, rng, batch_size)
        scores = rhythm_scores(actions)
        reward = scores["total"].detach()
        baseline = 0.96 * baseline + 0.04 * float(reward.mean())
        advantage = reward - baseline
        policy_loss = -(log_probs.sum(dim=1) * advantage).mean() / STEPS
        entropy_bonus = entropies.mean()
        loss = policy_loss - 0.010 * entropy_bonus
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 25 == 0 or update == updates - 1:
            curve.append(
                {
                    "update": update + 1,
                    "reward": float(reward.mean()),
                    "entropy": float(entropy_bonus.detach()),
                }
            )
    return model.eval(), curve


def run(args):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(exist_ok=True)
    seed_results = []
    best_model = None
    best_reward = -1.0
    curves = []
    for index in range(args.seeds):
        seed = args.seed + index * 1009
        model, curve = train(seed, args.updates, args.batch_size)
        evaluation = evaluate(model, seed + 50_000, args.eval_episodes)
        seed_results.append(evaluation)
        curves.append(curve)
        if evaluation["total"] > best_reward:
            best_reward = evaluation["total"]
            best_model = model

    normal = evaluate(best_model, args.seed + 90_000, args.eval_episodes)
    reset = evaluate(best_model, args.seed + 90_000, args.eval_episodes, reset_memory=True)
    metrics = {
        "design": {
            "training": "policy gradient from scalar structural reward; no target rhythm sequences",
            "durations_beats": DURATION_BEATS.tolist(),
            "updates": args.updates,
            "seeds": args.seeds,
            "claim_boundary": "Reward-learned symbolic timing under engineered musical objectives; not unconstrained emergent musical understanding.",
        },
        "fixed_eighths": fixed_baseline(args.eval_episodes),
        "weighted_random": random_baseline(args.seed + 90_000, args.eval_episodes),
        "learned": normal,
        "hidden_reset": reset,
        "seed_evaluations": seed_results,
        "training_curves": curves,
    }
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "hidden_size": best_model.hidden_size,
            "obs_dim": RHYTHM_OBS_DIM,
            "durations_beats": DURATION_BEATS.tolist(),
        },
        CHECKPOINT,
    )
    METRICS.write_text(json.dumps(metrics, indent=2))
    print("Reward-learned rhythm lab complete")
    for name in ("fixed_eighths", "weighted_random", "learned", "hidden_reset"):
        row = metrics[name]
        print(
            f"{name:16s} reward={row['total']:.3f} align={row['bar_alignment']:.3f} "
            f"variety={row['diversity']:.3f} recall={row['motif_recall']:.3f}"
        )
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Metrics: {METRICS}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8117)
    parser.add_argument("--updates", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.updates = 40
        args.batch_size = 48
        args.eval_episodes = 96
        args.seeds = 1
    return args


if __name__ == "__main__":
    run(parse_args())
