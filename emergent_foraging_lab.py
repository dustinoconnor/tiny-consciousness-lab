#!/usr/bin/env python3
"""Reward-grounded foraging and latent trap-state emergence.

No policy rule approaches food, seeks a frontier, or labels an enclosure. Agents
receive local obstacle rays, food direction only when line-of-sight visible,
hunger, previous action, and previous reward. They learn from mushroom reward.

Three conditions are compared:

- feedforward_reward: no temporal hidden state, extrinsic reward only.
- recurrent_reward: GRU memory, extrinsic reward only.
- recurrent_curiosity: GRU memory plus observation-novelty information reward.

Policies train on open, L-wall, and offset-barrier layouts, then freeze for a
withheld U-detour. A post-hoc linear probe tests whether GRU hidden state carries
trap context that was never used as a training label.
"""

import json
import math
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_lab import OUT, set_seed


SIZE = 11
MAX_STEPS = 72
MOVES = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int64)
SENSOR_RANGE = 4
FOOD_SIGHT = 5.5
OBS_DIM = 4 + 1 + 2 + 1 + 4 + 1


def rectangle(x0, y0, width, height):
    return {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}


def reachable(blocked, start, goal):
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for move in MOVES:
            nxt = cell[0] + int(move[0]), cell[1] + int(move[1])
            if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def rotate_cell(cell, turns):
    x, y = cell
    for _ in range(turns % 4):
        x, y = SIZE - 1 - y, x
    return x, y


def make_layout(kind, rng):
    blocked = set()
    if kind == "open":
        for _ in range(7):
            blocked.add(tuple(rng.integers(1, SIZE - 1, size=2)))
    elif kind == "l_wall":
        x = int(rng.integers(3, 7))
        y = int(rng.integers(3, 7))
        blocked |= rectangle(x, y, 1, 4)
        blocked |= rectangle(x, y + 3, 4, 1)
    elif kind == "offset_barriers":
        blocked |= rectangle(3, 2, 1, 6)
        blocked |= rectangle(7, 4, 1, 6)
        blocked -= {(3, 3), (7, 8)}
    elif kind == "u_detour":
        blocked |= rectangle(3, 3, 1, 5)
        blocked |= rectangle(7, 3, 1, 5)
        blocked |= rectangle(3, 7, 5, 1)
        start = (5, 4)
        goal = (5, 9)
        trap_cells = {(x, y) for x in range(4, 7) for y in range(3, 7) if (x, y) not in blocked}
        turns = int(rng.integers(0, 4))
        return (
            {rotate_cell(cell, turns) for cell in blocked},
            rotate_cell(start, turns),
            rotate_cell(goal, turns),
            {rotate_cell(cell, turns) for cell in trap_cells},
        )
    else:
        raise ValueError(kind)

    free = [(x, y) for x in range(1, SIZE - 1) for y in range(1, SIZE - 1) if (x, y) not in blocked]
    for _ in range(100):
        start = free[int(rng.integers(len(free)))]
        goal = free[int(rng.integers(len(free)))]
        if math.dist(start, goal) >= 4.0 and reachable(blocked, start, goal):
            return blocked, start, goal, set()
    return blocked, free[0], free[-1], set()


def line_of_sight(start, goal, blocked):
    x0, y0 = start
    x1, y1 = goal
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        twice = 2 * error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy
        if (x, y) != goal and (x, y) in blocked:
            return False
    return True


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    ate: bool
    collision: bool


class ForagingWorld:
    def __init__(self, kind, seed, food_reward=2.0):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.blocked, self.start, self.food, self.trap_cells = make_layout(kind, self.rng)
        self.food_reward = food_reward
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.hunger = 0.15
        self.last_action = 0
        self.last_reward = 0.0
        self.visits = {self.pos: 1}
        return self.observe()

    def is_blocked(self, cell):
        return not (0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE) or cell in self.blocked

    def ray(self, move):
        for distance in range(1, SENSOR_RANGE + 1):
            cell = self.pos[0] + int(move[0]) * distance, self.pos[1] + int(move[1]) * distance
            if self.is_blocked(cell):
                return (distance - 1) / SENSOR_RANGE
        return 1.0

    def food_visible(self):
        return math.dist(self.pos, self.food) <= FOOD_SIGHT and line_of_sight(self.pos, self.food, self.blocked)

    def observe(self):
        rays = np.asarray([self.ray(move) for move in MOVES], dtype=np.float32)
        visible = self.food_visible()
        delta = (np.asarray(self.food) - np.asarray(self.pos)) / (SIZE - 1) if visible else np.zeros(2)
        action = np.zeros(len(MOVES), dtype=np.float32)
        action[self.last_action] = 1.0
        return np.concatenate(
            [rays, [float(visible)], delta, [self.hunger], action, [self.last_reward]]
        ).astype(np.float32)

    def novelty_reward(self):
        count = self.visits.get(self.pos, 0)
        return 1.0 / math.sqrt(max(1, count))

    def step(self, action):
        self.last_action = int(action)
        move = MOVES[self.last_action]
        candidate = self.pos[0] + int(move[0]), self.pos[1] + int(move[1])
        collision = self.is_blocked(candidate)
        if not collision:
            self.pos = candidate
        self.steps += 1
        self.hunger = min(1.0, self.hunger + 1.0 / MAX_STEPS)
        self.visits[self.pos] = self.visits.get(self.pos, 0) + 1

        ate = self.pos == self.food
        reward = self.food_reward if ate else -0.008 - 0.006 * self.hunger
        if collision:
            reward -= 0.045
        done = (ate and self.food_reward > 0.0) or self.steps >= MAX_STEPS
        if not ate and self.steps >= MAX_STEPS:
            reward -= 0.35
        self.last_reward = reward
        return StepResult(self.observe(), reward, done, ate, collision)

    def trap_label(self):
        return float(self.pos in self.trap_cells)


class Policy(nn.Module):
    def __init__(self, recurrent, hidden_dim=48):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim) if recurrent else None
        self.policy = nn.Linear(hidden_dim, len(MOVES))
        self.value = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch):
        return torch.zeros(batch, self.hidden_dim)

    def step(self, obs, hidden):
        encoded = self.encoder(obs)
        next_hidden = self.memory(encoded, hidden) if self.recurrent else encoded
        return self.policy(next_hidden), self.value(next_hidden).squeeze(-1), next_hidden


def make_training_env(index, seed, food_reward=2.0):
    kinds = ["open", "open", "l_wall", "offset_barriers"]
    return ForagingWorld(kinds[index % len(kinds)], seed + index * 17, food_reward=food_reward)


def train_condition(name, recurrent, curiosity_beta, food_reward=2.0, updates=420, env_count=24, rollout=28, seed=41):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = Policy(recurrent)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0018)
    envs = [make_training_env(i, seed * 1000 + i, food_reward=food_reward) for i in range(env_count)]
    obs = torch.tensor(np.stack([env.reset() for env in envs]))
    hidden = policy.initial_state(env_count)
    reward_curve = []
    success_curve = []
    recent_rewards = np.zeros(env_count, dtype=np.float32)
    recent_success = []

    for update in range(updates):
        log_probs = []
        values = []
        rewards = []
        entropies = []
        masks = []

        for _ in range(rollout):
            logits, value, hidden = policy.step(obs, hidden)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_obs = []
            step_rewards = []
            alive = []
            for index, env in enumerate(envs):
                result = env.step(int(action[index]))
                intrinsic = curiosity_beta * env.novelty_reward()
                combined_reward = result.reward + intrinsic
                recent_rewards[index] += result.reward
                if result.done:
                    recent_success.append(float(result.ate))
                    envs[index] = make_training_env(
                        index,
                        seed * 100000 + update * env_count + index,
                        food_reward=food_reward,
                    )
                    next_obs.append(envs[index].reset())
                    recent_rewards[index] = 0.0
                    alive.append(0.0)
                else:
                    next_obs.append(result.obs)
                    alive.append(1.0)
                step_rewards.append(combined_reward)

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
            entropies.append(dist.entropy())
            mask = torch.tensor(alive, dtype=torch.float32)
            masks.append(mask)
            obs = torch.tensor(np.stack(next_obs))
            hidden = hidden * mask.unsqueeze(-1)

        with torch.no_grad():
            _, bootstrap, _ = policy.step(obs, hidden)
        returns = []
        running = bootstrap
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running = reward + 0.97 * running * mask
            returns.append(running)
        returns.reverse()
        returns_t = torch.stack(returns)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)
        advantage = returns_t - values_t
        loss = -(log_probs_t * advantage.detach()).mean() + 0.45 * advantage.pow(2).mean() - 0.018 * entropy_t.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.8)
        optimizer.step()
        hidden = hidden.detach()

        reward_curve.append(float(np.mean(returns_t[0].detach().numpy())))
        success_curve.append(float(np.mean(recent_success[-100:])) if recent_success else 0.0)

    return policy.eval(), {"reward": reward_curve, "success": success_curve, "name": name}


def evaluate(
    policy,
    kind,
    runs=80,
    seed=7000,
    stochastic=False,
    collect_probe=False,
    reset_hidden_each_step=False,
):
    rows = []
    probe_hidden = []
    probe_obs = []
    probe_labels = []
    for run in range(runs):
        env = ForagingWorld(kind, seed + run)
        obs = torch.tensor(env.reset()).unsqueeze(0)
        hidden = policy.initial_state(1)
        collisions = 0
        away_steps = 0
        path = [env.pos]
        for step in range(MAX_STEPS):
            with torch.no_grad():
                logits, _, hidden = policy.step(obs, hidden)
            if stochastic:
                action = int(torch.distributions.Categorical(logits=logits).sample())
            else:
                action = int(torch.argmax(logits, dim=-1))
            old_distance = math.dist(env.pos, env.food)
            result = env.step(action)
            collisions += int(result.collision)
            away_steps += int(math.dist(env.pos, env.food) > old_distance + 1e-8)
            path.append(env.pos)
            if collect_probe:
                probe_hidden.append(hidden.squeeze(0).numpy().copy())
                probe_obs.append(obs.squeeze(0).numpy().copy())
                probe_labels.append(env.trap_label())
            obs = torch.tensor(result.obs).unsqueeze(0)
            if reset_hidden_each_step:
                hidden = policy.initial_state(1)
            if result.done:
                rows.append(
                    {
                        "success": float(result.ate),
                        "steps": step + 1,
                        "collisions": collisions,
                        "away_steps": away_steps,
                        "path": path,
                        "blocked": env.blocked,
                        "food": env.food,
                    }
                )
                break
    summary = {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_away_steps": float(np.mean([row["away_steps"] for row in rows])),
    }
    probe = (np.asarray(probe_hidden), np.asarray(probe_obs), np.asarray(probe_labels))
    return summary, rows, probe


def linear_probe_accuracy(features, labels, seed=9):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1.0)
    negative = np.flatnonzero(labels == 0.0)
    count = min(len(positive), len(negative))
    if count < 8:
        return 0.5, 0.5
    balanced = np.concatenate(
        [rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)]
    )
    order = rng.permutation(balanced)
    split = int(0.7 * len(order))
    train, test = order[:split], order[split:]
    mean = features[train].mean(axis=0, keepdims=True)
    std = features[train].std(axis=0, keepdims=True) + 1e-5
    x_train = np.concatenate([(features[train] - mean) / std, np.ones((len(train), 1))], axis=1)
    x_test = np.concatenate([(features[test] - mean) / std, np.ones((len(test), 1))], axis=1)
    target = labels[train] * 2.0 - 1.0
    ridge = 0.05 * np.eye(x_train.shape[1])
    weights = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ target)
    prediction = (x_test @ weights >= 0.0).astype(float)
    accuracy = float(np.mean(prediction == labels[test]))
    majority = float(max(np.mean(labels[test]), 1.0 - np.mean(labels[test])))
    return accuracy, majority


def plot_training(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for curve in curves:
        axes[0].plot(curve["reward"], label=curve["name"])
        axes[1].plot(curve["success"], label=curve["name"])
    axes[0].set_title("Training return")
    axes[1].set_title("Recent mushroom success")
    for ax in axes:
        ax.set_xlabel("update")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_eval(summary, path):
    names = list(summary)
    metrics = ["success_rate", "mean_collisions", "mean_away_steps"]
    labels = ["success", "collisions / 20", "away steps / 20"]
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        values = [summary[name][metric] / (20.0 if metric != "success_rate" else 1.0) for name in names]
        ax.bar(x + (index - 1) * width, values, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Frozen Withheld U-Detour Foraging")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(41)
    configs = [
        ("feedforward_reward", False, 0.0, 2.0),
        ("recurrent_no_food_reward", True, 0.0, 0.0),
        ("recurrent_reward", True, 0.0, 2.0),
        ("recurrent_curiosity", True, 0.018, 2.0),
    ]
    policies = {}
    curves = []
    for name, recurrent, curiosity_beta, food_reward in configs:
        policy, curve = train_condition(name, recurrent, curiosity_beta, food_reward=food_reward)
        policies[name] = policy
        curves.append(curve)

    evaluation = {}
    paths = {}
    probes = {}
    for name, policy in policies.items():
        summary, rows, probe = evaluate(policy, "u_detour", collect_probe=True)
        evaluation[name] = summary
        paths[name] = rows
        if policy.recurrent:
            reset_summary, _, _ = evaluate(policy, "u_detour", reset_hidden_each_step=True)
            evaluation[f"{name}_hidden_reset"] = reset_summary
            hidden_accuracy, hidden_majority = linear_probe_accuracy(probe[0], probe[2])
            obs_accuracy, obs_majority = linear_probe_accuracy(probe[1], probe[2])
            shuffled = probe[2].copy()
            np.random.default_rng(99).shuffle(shuffled)
            shuffled_accuracy, _ = linear_probe_accuracy(probe[0], shuffled)
            probes[name] = {
                "hidden_trap_accuracy": hidden_accuracy,
                "observation_trap_accuracy": obs_accuracy,
                "majority_baseline": max(hidden_majority, obs_majority),
                "shuffled_label_accuracy": shuffled_accuracy,
                "hidden_over_observation": hidden_accuracy - obs_accuracy,
            }

    payload = {
        "training_layouts": ["open", "l_wall", "offset_barriers"],
        "withheld_layout": "u_detour",
        "food_approach_rule_present": False,
        "trap_label_used_during_training": False,
        "evaluation": evaluation,
        "hidden_state_probes": probes,
        "claim_boundary": (
            "Food seeking counts as learned only if frozen reward-trained policies exceed controls without an approach-food rule. "
            "A hidden-state probe above observation and shuffled controls suggests decodable trap context; it does not prove "
            "the policy causally uses that representation or that a human-like enclosure concept exists."
        ),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "emergent_foraging_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_training(curves, OUT / "emergent_foraging_training.png")
    plot_eval(evaluation, OUT / "emergent_foraging_u_detour.png")
    print("Emergent foraging lab complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
