#!/usr/bin/env python3
"""Pre-action intuition / imagination loop experiment.

The baseline recurrent agent acts from its current hidden state.

The imagination agent first uses its world model to imagine the next observation
for each possible action. It scores those imagined futures for progress toward
the goal and hazard risk, then uses that score as a soft action prior before it
acts.

This tests whether a fast pre-action prediction loop improves learning.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tiny_lab import OUT, TinyRecurrentAgent, set_seed
from valence_shaping_lab import ShapingWorld


def imagined_action_bonus(agent, h, obs, world_size, actions=5):
    goal_idx = int(torch.argmax(obs[world_size : 2 * world_size]).item())
    hazard_idx = int(torch.argmax(obs[2 * world_size : 3 * world_size]).item())
    current_pos = int(torch.argmax(obs[:world_size]).item())
    current_distance = abs(goal_idx - current_pos)
    positions = torch.arange(world_size, dtype=torch.float32)
    bonuses = []

    for action in range(actions):
        pred = agent.predict_next_obs(h, action)
        pos_probs = torch.softmax(pred[:world_size], dim=0)
        expected_distance = torch.sum(pos_probs * torch.abs(positions - goal_idx))
        hazard_prob = pos_probs[hazard_idx]
        expected_progress = current_distance - expected_distance

        if action >= 3:
            # Imagination should not encourage direct valence buttons.
            button_penalty = torch.tensor(0.35)
        else:
            button_penalty = torch.tensor(0.0)

        bonuses.append(1.4 * expected_progress - 2.2 * hazard_prob - button_penalty)

    return torch.stack(bonuses)


def run_episode(agent, env, train=True, use_imagination=False, imagination_strength=1.0):
    obs = env.reset()
    h = agent.initial_state()
    logps, values, rewards, world_losses = [], [], [], []
    events = []
    imagination_bonuses = []

    for _ in range(env.max_steps):
        h, logits, value, _, _ = agent.forward_step(obs, h)
        bonus = torch.zeros(agent.actions)
        if use_imagination:
            bonus = imagined_action_bonus(agent, h, obs, env.size, agent.actions)
            logits = logits + imagination_strength * bonus
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if train else torch.argmax(logits)
        pred_next = agent.predict_next_obs(h, action)
        result = env.step(action)

        logps.append(dist.log_prob(action))
        values.append(value)
        rewards.append(torch.tensor(result.reward))
        world_losses.append(F.mse_loss(pred_next, result.obs))
        events.append(result.event)
        imagination_bonuses.append(float(bonus[int(action)].detach()))
        obs = result.obs
        if result.done:
            break

    returns = []
    g = torch.tensor(0.0)
    for r in reversed(rewards):
        g = r + 0.94 * g
        returns.append(g)
    returns = torch.stack(list(reversed(returns)))
    values_t = torch.stack(values)
    logps_t = torch.stack(logps)
    advantage = returns - values_t.detach()
    loss = -(logps_t * advantage).mean() + 0.5 * F.mse_loss(values_t, returns) + 0.25 * torch.stack(world_losses).mean()
    return {
        "loss": loss,
        "reward": float(torch.stack(rewards).sum()),
        "events": events,
        "steps": len(events),
        "mean_imagination_bonus": float(np.mean(imagination_bonuses)) if imagination_bonuses else 0.0,
    }


def train_condition(label, mode, hidden_dim=32, use_imagination=False, episodes=900, eval_runs=96):
    env = ShapingWorld(mode)
    agent = TinyRecurrentAgent(obs_dim=env.obs_dim, hidden_dim=hidden_dim, actions=5)
    opt = torch.optim.Adam(agent.parameters(), lr=0.006)
    training = {"reward": [], "goal_rate": [], "button_rate": [], "steps": [], "imagination_bonus": []}

    for _ in range(episodes):
        result = run_episode(agent, env, train=True, use_imagination=use_imagination)
        opt.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        training["reward"].append(result["reward"])
        training["goal_rate"].append(1.0 if "goal" in result["events"] else 0.0)
        training["button_rate"].append(result["events"].count("good_button") / max(result["steps"], 1))
        training["steps"].append(result["steps"])
        training["imagination_bonus"].append(result["mean_imagination_bonus"])

    evals = [run_episode(agent, env, train=False, use_imagination=use_imagination) for _ in range(eval_runs)]
    total_steps = sum(r["steps"] for r in evals)
    return {
        "label": label,
        "mode": mode,
        "hidden_dim": hidden_dim,
        "use_imagination": use_imagination,
        "training": training,
        "eval_goal_rate": sum(1 for r in evals if "goal" in r["events"]) / len(evals),
        "eval_hazard_rate": sum(1 for r in evals if "hazard" in r["events"]) / len(evals),
        "eval_button_rate": sum(r["events"].count("good_button") for r in evals) / max(total_steps, 1),
        "eval_mean_steps": float(np.mean([r["steps"] for r in evals])),
        "eval_mean_reward": float(np.mean([r["reward"] for r in evals])),
    }


def moving_average(values, window=35):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(results, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for label, data in results.items():
        axes[0].plot(moving_average(data["training"]["goal_rate"]), lw=2, label=label)
        axes[1].plot(moving_average(data["training"]["steps"]), lw=2, label=label)
        axes[2].plot(moving_average(data["training"]["reward"]), lw=2, label=label)
    axes[0].set_title("Goal Completion During Training")
    axes[1].set_title("Episode Length During Training")
    axes[2].set_title("Reward During Training")
    axes[2].set_xlabel("training episode")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eval(results, path):
    labels = list(results)
    x = np.arange(len(labels))
    width = 0.24
    goal = [results[k]["eval_goal_rate"] for k in labels]
    steps = [results[k]["eval_mean_steps"] / 32 for k in labels]
    reward = [results[k]["eval_mean_reward"] for k in labels]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(x - width / 2, goal, width, label="goal rate", color="#16a3a6")
    axes[0].bar(x + width / 2, steps, width, label="mean steps / max", color="#7a7a7a")
    axes[0].set_ylabel("rate")
    axes[0].legend()
    axes[1].bar(x, reward, color="#ff8a00")
    axes[1].set_ylabel("mean reward")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    fig.suptitle("Pre-Action Imagination Evaluation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(41)
    OUT.mkdir(exist_ok=True)
    configs = [
        ("baseline_goal_only", "goal_only", False),
        ("imagination_goal_only", "goal_only", True),
        ("baseline_progress_valence", "large_progress_reward", False),
        ("imagination_progress_valence", "large_progress_reward", True),
    ]
    results = {
        label: train_condition(label, mode, use_imagination=use_imagination)
        for label, mode, use_imagination in configs
    }
    serializable = {
        label: {
            "mode": data["mode"],
            "hidden_dim": data["hidden_dim"],
            "use_imagination": data["use_imagination"],
            "eval_goal_rate": data["eval_goal_rate"],
            "eval_hazard_rate": data["eval_hazard_rate"],
            "eval_button_rate": data["eval_button_rate"],
            "eval_mean_steps": data["eval_mean_steps"],
            "eval_mean_reward": data["eval_mean_reward"],
        }
        for label, data in results.items()
    }
    serializable["note"] = (
        "Imagination here means a pre-action world-model rollout for each possible action. "
        "The imagined next states create an action prior before the policy acts."
    )
    (OUT / "imagination_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_training(results, OUT / "imagination_training.png")
    plot_eval(results, OUT / "imagination_eval.png")
    print("Imagination lab complete")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
