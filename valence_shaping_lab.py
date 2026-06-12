#!/usr/bin/env python3
"""Valence shaping tests.

This experiment asks whether partial positive valence helps or degrades task
solving. The agent has a real external goal plus an optional good-valence
button. Different conditions change when that button pays out.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tiny_lab import OUT, TinyRecurrentAgent, set_seed


@dataclass
class StepResult:
    obs: torch.Tensor
    reward: float
    done: bool
    event: str
    distance_decreased: bool


class ShapingWorld:
    def __init__(self, mode, size=9, max_steps=32, button_reward=0.08, button_cost=-0.05):
        self.mode = mode
        self.size = size
        self.max_steps = max_steps
        self.button_reward = button_reward
        self.button_cost = button_cost
        self.reset()

    @property
    def obs_dim(self):
        # position, goal, hazard, last reward, last action one-hot, recent progress scalar
        return self.size * 3 + 1 + 5 + 1

    def reset(self):
        self.t = 0
        self.pos = self.size // 2
        self.goal = random.choice([0, self.size - 1])
        self.hazard = random.choice([i for i in range(self.size) if i not in {self.pos, self.goal}])
        self.last_reward = 0.0
        self.last_action = 1
        self.recent_progress = 0.0
        self.button_presses = 0
        self.progress_credit = 0
        return self.observe()

    def distance(self):
        return abs(self.goal - self.pos)

    def observe(self):
        v = torch.zeros(self.obs_dim)
        v[self.pos] = 1.0
        v[self.size + self.goal] = 1.0
        v[self.size * 2 + self.hazard] = 1.0
        v[self.size * 3] = self.last_reward
        v[self.size * 3 + 1 + self.last_action] = 1.0
        v[-1] = self.recent_progress
        return v

    def button_value(self, distance_decreased):
        if self.mode in {"goal_only", "small_progress_reward", "large_progress_reward"}:
            return self.button_cost
        if self.mode == "direct_positive":
            return self.button_reward
        if self.mode == "decaying_positive":
            self.button_presses += 1
            return max(self.button_cost, self.button_reward * (0.55 ** (self.button_presses - 1)))
        if self.mode == "progress_gated":
            if self.progress_credit > 0:
                self.progress_credit -= 1
                return self.button_reward
            return self.button_cost
        if self.mode == "negative_button":
            return -0.15
        raise ValueError(self.mode)

    def step(self, action):
        action = int(action)
        old_distance = self.distance()
        self.t += 1
        reward = -0.02
        done = False
        event = "move"
        distance_decreased = False

        if action in (0, 1, 2):
            move = [-1, 0, 1][action]
            self.pos = max(0, min(self.size - 1, self.pos + move))
            distance_decreased = self.distance() < old_distance
            self.recent_progress = 1.0 if distance_decreased else 0.0
            if distance_decreased:
                self.progress_credit = 1
                if self.mode == "small_progress_reward":
                    reward += 0.03
                elif self.mode == "large_progress_reward":
                    reward += 0.15
            if self.pos == self.goal:
                reward = 1.0
                done = True
                event = "goal"
            elif self.pos == self.hazard:
                reward = -1.0
                done = True
                event = "hazard"
        elif action == 3:
            reward = self.button_value(distance_decreased)
            event = "good_button"
            self.recent_progress *= 0.4
        elif action == 4:
            reward = -1.0
            done = True
            event = "bad_button"

        if self.t >= self.max_steps:
            done = True

        self.last_reward = reward
        self.last_action = action
        return StepResult(self.observe(), reward, done, event, distance_decreased)


def run_episode(agent, env, train=True):
    obs = env.reset()
    h = agent.initial_state()
    logps, values, rewards, world_losses = [], [], [], []
    events = []

    for _ in range(env.max_steps):
        h, logits, value, _, _ = agent.forward_step(obs, h)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if train else torch.argmax(logits)
        pred_next = agent.predict_next_obs(h, action)
        result = env.step(action)

        logps.append(dist.log_prob(action))
        values.append(value)
        rewards.append(torch.tensor(result.reward))
        world_losses.append(F.mse_loss(pred_next, result.obs))
        events.append(result.event)
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
    loss = -(logps_t * advantage).mean() + 0.5 * F.mse_loss(values_t, returns) + 0.15 * torch.stack(world_losses).mean()
    return {
        "loss": loss,
        "reward": float(torch.stack(rewards).sum()),
        "events": events,
        "steps": len(events),
    }


def train_condition(mode, episodes=900):
    env = ShapingWorld(mode)
    agent = TinyRecurrentAgent(obs_dim=env.obs_dim, hidden_dim=32, actions=5)
    opt = torch.optim.Adam(agent.parameters(), lr=0.006)
    training = {"reward": [], "goal_rate": [], "button_rate": [], "steps": []}

    for _ in range(episodes):
        result = run_episode(agent, env, train=True)
        opt.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        training["reward"].append(result["reward"])
        training["goal_rate"].append(1.0 if "goal" in result["events"] else 0.0)
        training["button_rate"].append(result["events"].count("good_button") / max(result["steps"], 1))
        training["steps"].append(result["steps"])

    eval_runs = [run_episode(agent, env, train=False) for _ in range(96)]
    total_steps = sum(r["steps"] for r in eval_runs)
    return {
        "training": training,
        "eval_goal_rate": sum(1 for r in eval_runs if "goal" in r["events"]) / len(eval_runs),
        "eval_hazard_rate": sum(1 for r in eval_runs if "hazard" in r["events"]) / len(eval_runs),
        "eval_button_rate": sum(r["events"].count("good_button") for r in eval_runs) / max(total_steps, 1),
        "eval_mean_steps": float(np.mean([r["steps"] for r in eval_runs])),
        "eval_mean_reward": float(np.mean([r["reward"] for r in eval_runs])),
    }


def moving_average(values, window=35):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training(results, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for mode, data in results.items():
        axes[0].plot(moving_average(data["training"]["goal_rate"]), lw=2, label=mode)
        axes[1].plot(moving_average(data["training"]["button_rate"]), lw=2, label=mode)
        axes[2].plot(moving_average(data["training"]["reward"]), lw=2, label=mode)
    axes[0].set_title("Goal Completion During Training")
    axes[1].set_title("Good-Valence Button Use During Training")
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
    width = 0.22
    goal = [results[k]["eval_goal_rate"] for k in labels]
    button = [results[k]["eval_button_rate"] for k in labels]
    steps = [results[k]["eval_mean_steps"] / 32 for k in labels]
    reward = [results[k]["eval_mean_reward"] for k in labels]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].bar(x - width, goal, width, label="goal rate", color="#16a3a6")
    axes[0].bar(x, button, width, label="button action rate", color="#ff8a00")
    axes[0].bar(x + width, steps, width, label="mean steps / max", color="#7a7a7a")
    axes[0].set_ylabel("rate")
    axes[0].legend()
    axes[1].bar(x, reward, color="#4b6cff")
    axes[1].set_ylabel("mean reward")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    fig.suptitle("Valence Shaping Evaluation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(23)
    OUT.mkdir(exist_ok=True)
    modes = [
        "goal_only",
        "small_progress_reward",
        "large_progress_reward",
        "progress_gated",
        "decaying_positive",
        "direct_positive",
    ]
    results = {mode: train_condition(mode) for mode in modes}
    serializable = {
        mode: {
            "eval_goal_rate": data["eval_goal_rate"],
            "eval_hazard_rate": data["eval_hazard_rate"],
            "eval_button_rate": data["eval_button_rate"],
            "eval_mean_steps": data["eval_mean_steps"],
            "eval_mean_reward": data["eval_mean_reward"],
        }
        for mode, data in results.items()
    }
    serializable["note"] = (
        "This tests whether direct, decaying, negative, or progress-gated valence shaping helps or hurts external task completion."
    )
    (OUT / "valence_shaping_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_training(results, OUT / "valence_shaping_training.png")
    plot_eval(results, OUT / "valence_shaping_eval.png")
    print("Valence shaping lab complete")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
