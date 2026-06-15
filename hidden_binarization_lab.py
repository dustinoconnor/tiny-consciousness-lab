#!/usr/bin/env python3
"""Binarize trained hidden states and measure empirical integration.

Earlier labs measured exact Phi-like integration on hand-designed binary
circuits. This lab asks a stronger mechanistic-interpretability question:

Did a trained recurrent agent develop integrated binary dynamics inside its
hidden state while solving a task?

Pipeline:

1. Train the tiny recurrent PyTorch agent from `tiny_lab.py`.
2. Record hidden-state trajectories while the trained agent acts.
3. Select the most behaviorally active hidden units.
4. Convert their continuous activations into binary on/off states.
5. Estimate an empirical transition table: P(binary next state | binary state).
6. Compare the full transition table against its best bipartitioned
   approximation, using KL divergence as a Phi-like integration score.

This is not official IIT Phi. It is a transparent bridge from learned neural
activations to the same partition logic used in the exact toy binary labs.
"""

import itertools
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from exact_phi_lab import EPS, all_states, bipartitions, kl, state_index
from tiny_lab import LineWorld, OUT, TinyRecurrentAgent, set_seed


def expert_action(env):
    """A tiny teacher policy for LineWorld.

    Move toward the goal unless the next step would hit the hazard. If the
    hazard blocks the direct path, stay put. This makes a stable trained agent
    for hidden-state analysis instead of relying on noisy policy-gradient luck.
    """
    if env.goal > env.pos:
        action = 2
    elif env.goal < env.pos:
        action = 0
    else:
        action = 1

    move = [-1, 0, 1][action]
    next_pos = max(0, min(env.size - 1, env.pos + move))
    if next_pos == env.hazard:
        return 1
    return action


def train_expert_agent(agent, episodes=900):
    """Train policy, value, world model, and self-model from an expert teacher."""
    opt = torch.optim.Adam(agent.parameters(), lr=0.006)
    history = {"loss": [], "policy_accuracy": [], "reward": []}

    for _ in range(episodes):
        env = LineWorld()
        obs = env.reset()
        h = agent.initial_state()
        losses = []
        correct = []
        rewards = []
        last_logits = None

        for _ in range(env.max_steps):
            h, logits, value, self_logits, _ = agent.forward_step(obs, h)
            action = torch.tensor(expert_action(env))
            pred_next = agent.predict_next_obs(h, action)
            result = env.step(action)

            policy_loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), action.view(1))
            value_target = torch.tensor(result.reward, dtype=torch.float32)
            value_loss = torch.nn.functional.mse_loss(value, value_target)
            world_loss = torch.nn.functional.mse_loss(pred_next, result.obs)
            if last_logits is None:
                self_loss = torch.tensor(0.0)
            else:
                self_loss = torch.nn.functional.cross_entropy(last_logits.unsqueeze(0), action.view(1))
            last_logits = self_logits
            losses.append(policy_loss + 0.25 * value_loss + 0.30 * world_loss + 0.05 * self_loss)
            correct.append(float(torch.argmax(logits).item() == int(action)))
            rewards.append(result.reward)
            obs = result.obs
            if result.done:
                break

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        history["loss"].append(float(loss.detach()))
        history["policy_accuracy"].append(float(np.mean(correct)))
        history["reward"].append(float(np.sum(rewards)))

    return history


def collect_hidden_dataset(agent, episodes=220):
    """Run the trained agent and collect hidden states plus behavior labels."""
    rows = []
    for episode in range(episodes):
        env = LineWorld()
        obs = env.reset()
        h = agent.initial_state()
        previous_distance = abs(env.goal - env.pos)

        for t in range(env.max_steps):
            h, logits, value, _, gate = agent.forward_step(obs, h)
            action = torch.argmax(logits)
            teacher = expert_action(env)
            old_pos = env.pos
            result = env.step(action)
            distance = abs(env.goal - env.pos)
            direct_action = 2 if env.goal > old_pos else 0 if env.goal < old_pos else 1
            direct_move = [-1, 0, 1][direct_action]
            direct_next = max(0, min(env.size - 1, old_pos + direct_move))
            hazard_blocks_direct_path = direct_next == env.hazard
            conflict = int(hazard_blocks_direct_path or int(action) != teacher or result.reward < -0.02)
            rows.append(
                {
                    "episode": episode,
                    "t": t,
                    "hidden": h.detach().numpy(),
                    "gate": gate.detach().numpy(),
                    "action": int(action),
                    "teacher_action": int(teacher),
                    "reward": float(result.reward),
                    "value": float(value.detach()),
                    "pos": int(old_pos),
                    "next_pos": int(env.pos),
                    "goal": int(env.goal),
                    "hazard": int(env.hazard),
                    "distance": int(distance),
                    "conflict": conflict,
                    "terminal": bool(result.done),
                }
            )
            obs = result.obs
            previous_distance = distance
            if result.done:
                break
    return rows


def select_units(hidden, k=6):
    """Pick hidden units with the strongest mixture of variance and valence link."""
    hidden = np.asarray(hidden)
    variance = hidden.var(axis=0)
    activity = np.mean(np.abs(hidden), axis=0)
    score = variance + 0.25 * activity
    return np.argsort(score)[-k:][::-1]


def binarize(hidden, units, thresholds=None):
    """Convert selected continuous hidden units into binary states."""
    selected = np.asarray(hidden)[:, units]
    if thresholds is None:
        thresholds = np.median(selected, axis=0)
    bits = (selected > thresholds).astype(np.int8)
    return bits, thresholds


def transition_counts(bits, mask=None, smoothing=0.5):
    """Estimate P(next binary state | current binary state) from trajectory bits."""
    n = bits.shape[1]
    num_states = 2**n
    counts = np.full((num_states, num_states), smoothing, dtype=np.float64)
    mask = np.ones(len(bits) - 1, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    for i in range(len(bits) - 1):
        if not mask[i]:
            continue
        src = state_index(bits[i])
        dst = state_index(bits[i + 1])
        counts[src, dst] += 1.0
    return counts


def normalize_rows(counts):
    return counts / counts.sum(axis=1, keepdims=True)


def marginalize_transition(full_transition, n, subset, current_subset_state):
    """Average over unknown outside nodes and return subset next-state dist."""
    subset = tuple(subset)
    outside = tuple(i for i in range(n) if i not in subset)
    accum = np.zeros(2 ** len(subset), dtype=np.float64)
    outside_states = all_states(len(outside))

    for outside_state in outside_states:
        full_current = np.zeros(n, dtype=np.int8)
        for i, bit in zip(subset, current_subset_state):
            full_current[i] = bit
        for i, bit in zip(outside, outside_state):
            full_current[i] = bit
        src = state_index(full_current)
        for next_bits in all_states(n):
            prob = full_transition[src, state_index(next_bits)]
            subset_next = [next_bits[i] for i in subset]
            accum[state_index(subset_next)] += prob

    accum /= max(len(outside_states), 1)
    return accum / accum.sum()


def product_partition_distribution(full_transition, n, state, partition):
    """Build the disconnected approximation for one bipartition."""
    a, b = partition
    pa = marginalize_transition(full_transition, n, a, [state[i] for i in a])
    pb = marginalize_transition(full_transition, n, b, [state[i] for i in b])
    full = np.zeros(2**n, dtype=np.float64)
    for next_a in all_states(len(a)):
        for next_b in all_states(len(b)):
            bits = np.zeros(n, dtype=np.int8)
            for i, bit in zip(a, next_a):
                bits[i] = bit
            for i, bit in zip(b, next_b):
                bits[i] = bit
            full[state_index(bits)] += pa[state_index(next_a)] * pb[state_index(next_b)]
    return full / full.sum()


def empirical_phi_proxy(counts):
    """Partition-KL score over an empirical binary transition table."""
    transition = normalize_rows(counts)
    n = int(np.log2(transition.shape[0]))
    states = all_states(n)
    parts = bipartitions(n)
    state_phi = []
    best_partition_counts = defaultdict(int)
    state_visits = counts.sum(axis=1)

    for state in states:
        src = state_index(state)
        p = np.clip(transition[src], EPS, 1.0)
        scores = []
        for part in parts:
            q = product_partition_distribution(transition, n, state, part)
            scores.append(kl(p, q))
        best = int(np.argmin(scores))
        state_phi.append(scores[best])
        best_partition_counts[str(parts[best])] += 1

    weights = state_visits / state_visits.sum()
    return {
        "phi_proxy_mean": float(np.mean(state_phi)),
        "phi_proxy_visit_weighted": float(np.sum(np.array(state_phi) * weights)),
        "state_phi": np.array(state_phi),
        "state_visits": state_visits,
        "best_partition_counts": dict(best_partition_counts),
    }


def segment_masks(rows):
    """Create masks over transitions for ordinary vs conflict moments."""
    conflict = np.array([row["conflict"] for row in rows], dtype=bool)
    teacher_mismatch = np.array([row["action"] != row["teacher_action"] for row in rows], dtype=bool)
    reward = np.array([row["reward"] for row in rows], dtype=float)
    ordinary_mask = (~conflict[:-1]) & (~teacher_mismatch[:-1]) & (reward[:-1] >= -0.02)
    conflict_mask = conflict[:-1] | teacher_mismatch[:-1] | (reward[:-1] < -0.02)
    return {
        "all_transitions": np.ones(len(rows) - 1, dtype=bool),
        "ordinary_transitions": ordinary_mask,
        "conflict_or_negative_transitions": conflict_mask,
    }


def summarize_hidden(rows, units, thresholds, phi_results):
    hidden = np.array([row["hidden"] for row in rows])
    rewards = np.array([row["reward"] for row in rows])
    values = np.array([row["value"] for row in rows])
    return {
        "selected_hidden_units": [int(u) for u in units],
        "thresholds": [float(x) for x in thresholds],
        "rows_collected": len(rows),
        "mean_reward": float(np.mean(rewards)),
        "mean_value_prediction": float(np.mean(values)),
        "selected_unit_variance": [float(v) for v in hidden[:, units].var(axis=0)],
        "phi_by_segment": {
            name: {
                "phi_proxy_mean": data["phi_proxy_mean"],
                "phi_proxy_visit_weighted": data["phi_proxy_visit_weighted"],
                "best_partition_counts": data["best_partition_counts"],
            }
            for name, data in phi_results.items()
        },
        "note": "Empirical partition-KL integration on binarized trained hidden states; not official IIT Phi.",
    }


def plot_binary_raster(bits, rows, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(bits[:160].T, aspect="auto", cmap="Greys", interpolation="nearest")
    conflict_times = [i for i, row in enumerate(rows[:160]) if row["conflict"]]
    for t in conflict_times:
        ax.axvline(t, color="#ff8a00", lw=0.8, alpha=0.35)
    ax.set_title("Binarized Hidden-State Raster: Selected Trained Units")
    ax.set_xlabel("time step across collected episodes")
    ax.set_ylabel("selected hidden unit")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phi_segments(phi_results, path):
    names = list(phi_results)
    mean_vals = [phi_results[name]["phi_proxy_mean"] for name in names]
    weighted_vals = [phi_results[name]["phi_proxy_visit_weighted"] for name in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, mean_vals, width, label="state mean", color="#7c3aed")
    ax.bar(x + width / 2, weighted_vals, width, label="visit weighted", color="#16a3a6")
    ax.set_title("Empirical Phi Proxy on Binarized Trained Hidden States")
    ax.set_ylabel("minimum partition KL divergence, bits")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_state_phi(phi_results, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, data in phi_results.items():
        ax.plot(data["state_phi"], lw=2, label=name)
    ax.set_title("State-by-State Empirical Phi Proxy")
    ax.set_xlabel("binary state index")
    ax.set_ylabel("minimum partition KL divergence, bits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(89)
    OUT.mkdir(exist_ok=True)

    agent = TinyRecurrentAgent(obs_dim=LineWorld().obs_dim, hidden_dim=24)
    training = train_expert_agent(agent, episodes=900)
    rows = collect_hidden_dataset(agent, episodes=240)
    hidden = np.array([row["hidden"] for row in rows])
    units = select_units(hidden, k=6)
    bits, thresholds = binarize(hidden, units)

    masks = segment_masks(rows)
    phi_results = {}
    for name, mask in masks.items():
        counts = transition_counts(bits, mask=mask)
        phi_results[name] = empirical_phi_proxy(counts)

    summary = summarize_hidden(rows, units, thresholds, phi_results)
    summary["mean_training_reward_last_100"] = float(np.mean(training["reward"][-100:]))
    summary["mean_training_policy_accuracy_last_100"] = float(np.mean(training["policy_accuracy"][-100:]))
    summary["mean_training_loss_last_100"] = float(np.mean(training["loss"][-100:]))

    (OUT / "hidden_binarization_metrics.json").write_text(json.dumps(summary, indent=2))
    plot_binary_raster(bits, rows, OUT / "hidden_binarization_raster.png")
    plot_phi_segments(phi_results, OUT / "hidden_binarization_phi_segments.png")
    plot_state_phi(phi_results, OUT / "hidden_binarization_state_phi.png")

    print("Hidden binarization lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
