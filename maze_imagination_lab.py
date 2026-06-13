#!/usr/bin/env python3
"""2D maze test where instinct should fail and imagination should matter.

The 1D world was too simple: progress-shaped valence could solve it by reflex.
This maze contains walls and dead ends, so locally moving closer to the goal is
not always the right move.

The comparison:

- progress_reflex: recurrent agent with local Manhattan-progress valence.
- naive_imagination: same agent, but imagination can steer from the beginning.
- pretrained_imagination: world model is pretrained on random maze rollouts
  before imagination is allowed to steer action.

This is still a toy, but it tests whether imagination helps when the world
requires planning around obstacles.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tiny_lab import OUT, TinyRecurrentAgent, set_seed


MAZE = [
    "###########",
    "#S....#..G#",
    "#.....#...#",
    "#.....#...#",
    "#.....#...#",
    "#.........#",
    "###########",
]

MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@dataclass
class StepResult:
    obs: torch.Tensor
    reward: float
    done: bool
    event: str


class MazeWorld:
    def __init__(self, max_steps=54, progress_reward=0.08):
        self.grid = [list(row) for row in MAZE]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.cells = self.height * self.width
        self.max_steps = max_steps
        self.progress_reward = progress_reward
        self.start = self.find("S")
        self.goal = self.find("G")
        self.distance_vector = self.make_distance_vector()
        self.reset()

    @property
    def obs_dim(self):
        # position, goal, walls, last reward
        return self.cells * 3 + 1

    @property
    def open_cells(self):
        cells = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell != "#":
                    cells.append((x, y))
        return cells

    def find(self, char):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == char:
                    return (x, y)
        raise ValueError(char)

    def idx(self, xy):
        x, y = xy
        return y * self.width + x

    def xy(self, idx):
        return (idx % self.width, idx // self.width)

    def is_wall(self, xy):
        x, y = xy
        return self.grid[y][x] == "#"

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def shortest_distance(self, start=None):
        start = self.pos if start is None else start
        q = deque([(start, 0)])
        seen = {start}
        while q:
            cur, dist = q.popleft()
            if cur == self.goal:
                return dist
            for nxt in self.neighbors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, dist + 1))
        return 999

    def make_distance_vector(self):
        distances = []
        for idx in range(self.cells):
            xy = self.xy(idx)
            distances.append(999.0 if self.is_wall(xy) else float(self.shortest_distance(xy)))
        return torch.tensor(distances, dtype=torch.float32)

    def neighbors(self, xy):
        x, y = xy
        for dx, dy in MOVES:
            nxt = (x + dx, y + dy)
            if not self.is_wall(nxt):
                yield nxt

    def transition_from(self, xy, action):
        dx, dy = MOVES[int(action)]
        candidate = (xy[0] + dx, xy[1] + dy)
        if self.is_wall(candidate):
            return xy, "wall"
        if candidate == self.goal:
            return candidate, "goal"
        return candidate, "move"

    def reset(self):
        self.t = 0
        self.pos = self.start
        self.last_reward = 0.0
        return self.observe()

    def observe(self):
        v = torch.zeros(self.obs_dim)
        v[self.idx(self.pos)] = 1.0
        v[self.cells + self.idx(self.goal)] = 1.0
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "#":
                    v[self.cells * 2 + self.idx((x, y))] = 1.0
        v[-1] = self.last_reward
        return v

    def step(self, action):
        old = self.pos
        old_manhattan = self.manhattan(old, self.goal)
        candidate, event = self.transition_from(old, action)
        self.t += 1

        reward = -0.025
        if event == "wall":
            reward -= 0.08
        else:
            new_manhattan = self.manhattan(candidate, self.goal)
            if new_manhattan < old_manhattan:
                reward += self.progress_reward
            elif new_manhattan > old_manhattan:
                reward -= self.progress_reward * 0.65
            self.pos = candidate

        done = False
        if self.pos == self.goal:
            reward = 1.0
            event = "goal"
            done = True
        elif self.t >= self.max_steps:
            done = True

        self.last_reward = reward
        return StepResult(self.observe(), reward, done, event)


def imagined_maze_bonus(agent, h, obs, env):
    bonuses = []
    for action in range(agent.actions):
        pred = agent.predict_next_obs(h, action)
        predicted_idx = int(torch.argmax(pred[: env.cells]).item())
        predicted_distance = env.distance_vector[predicted_idx]
        wall_penalty = torch.tensor(3.0 if predicted_distance > 100.0 else 0.0)
        bonuses.append(-0.28 * predicted_distance - wall_penalty)
    return torch.stack(bonuses)


def prediction_accuracy(pred_next, actual_next):
    mse = F.mse_loss(pred_next, actual_next.detach())
    return torch.exp(-10.0 * mse), mse


def run_episode(agent, env, train=True, imagination=False, gated=False):
    obs = env.reset()
    h = agent.initial_state()
    logps, values, rewards, world_losses = [], [], [], []
    events = []
    confidence = torch.tensor(0.0)
    accuracies = []

    for _ in range(env.max_steps):
        h, logits, value, _, _ = agent.forward_step(obs, h)
        if imagination:
            bonus = imagined_maze_bonus(agent, h, obs, env)
            gate = confidence if gated else torch.tensor(1.0)
            logits = logits + gate * bonus
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if train else torch.argmax(logits)
        pred_next = agent.predict_next_obs(h, action)
        result = env.step(action)
        accuracy, pred_mse = prediction_accuracy(pred_next, result.obs)
        confidence = 0.92 * confidence + 0.08 * accuracy.detach()

        reward = torch.tensor(result.reward)
        if imagination:
            reward = reward + 0.04 * (accuracy.detach() - 0.55)

        logps.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)
        world_losses.append(pred_mse)
        events.append(result.event)
        accuracies.append(float(accuracy.detach()))
        obs = result.obs
        if result.done:
            break

    returns = []
    g = torch.tensor(0.0)
    for r in reversed(rewards):
        g = r + 0.95 * g
        returns.append(g)
    returns = torch.stack(list(reversed(returns)))
    values_t = torch.stack(values)
    logps_t = torch.stack(logps)
    advantage = returns - values_t.detach()
    loss = -(logps_t * advantage).mean() + 0.5 * F.mse_loss(values_t, returns) + 0.28 * torch.stack(world_losses).mean()
    return {
        "loss": loss,
        "reward": float(torch.stack(rewards).sum()),
        "events": events,
        "steps": len(events),
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
    }


def pretrain_world_model(agent, env, episodes=260):
    opt = torch.optim.Adam(agent.parameters(), lr=0.004)
    losses = []
    for _ in range(episodes):
        obs = env.reset()
        h = agent.initial_state()
        episode_losses = []
        for _ in range(env.max_steps):
            h, _, _, _, _ = agent.forward_step(obs, h)
            action = torch.randint(0, agent.actions, ()).item()
            pred_next = agent.predict_next_obs(h, action)
            result = env.step(action)
            episode_losses.append(F.mse_loss(pred_next, result.obs))
            obs = result.obs
            if result.done:
                break
        loss = torch.stack(episode_losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def pretrain_tabular_world_model(env):
    """A tiny exact world model learned from the environment's transition table.

    This is deliberately simple: the counter-example is about planning depth,
    not about whether a neural world model can memorize a seven-row maze.
    """
    transitions = {}
    for cell in env.open_cells:
        transitions[cell] = {}
        for action in range(4):
            transitions[cell][action] = env.transition_from(cell, action)
    return transitions


def myopic_progress_action(env, pos):
    """Choose only actions that immediately reduce Manhattan distance.

    If no move feels better right now, the reflex keeps pushing toward the goal
    and gets stuck against the wall. That is the local-minimum failure mode.
    """
    current = env.manhattan(pos, env.goal)
    candidates = []
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall":
            continue
        dist = env.manhattan(nxt, env.goal)
        if dist < current:
            candidates.append((dist, action))
    if candidates:
        return min(candidates)[1]

    # No locally positive move exists, so push horizontally toward the goal.
    if env.goal[0] > pos[0]:
        return 3
    if env.goal[0] < pos[0]:
        return 2
    if env.goal[1] > pos[1]:
        return 1
    return 0


def lookahead_score(env, world, pos, depth):
    if pos == env.goal:
        return 100.0
    if depth <= 0:
        return -env.shortest_distance(pos)
    best = -999.0
    for action in range(4):
        nxt, event = world[pos][action]
        penalty = -4.0 if event == "wall" else 0.0
        best = max(best, penalty + 0.92 * lookahead_score(env, world, nxt, depth - 1))
    return best


def pretrained_world_lookahead_action(env, world, pos, horizon=8):
    scored = []
    for action in range(4):
        nxt, event = world[pos][action]
        penalty = -4.0 if event == "wall" else 0.0
        score = penalty + 0.92 * lookahead_score(env, world, nxt, horizon - 1)
        scored.append((score, action))
    return max(scored)[1]


def evaluate_detour_policy(label, policy, max_steps=34):
    env = MazeWorld(max_steps=max_steps)
    world = pretrain_tabular_world_model(env)
    pos = env.start
    path = [pos]
    events = []
    away_steps = 0

    for _ in range(max_steps):
        old_distance = env.manhattan(pos, env.goal)
        if policy == "myopic":
            action = myopic_progress_action(env, pos)
        elif policy == "lookahead":
            action = pretrained_world_lookahead_action(env, world, pos)
        else:
            raise ValueError(policy)

        nxt, event = env.transition_from(pos, action)
        if env.manhattan(nxt, env.goal) > old_distance:
            away_steps += 1
        pos = nxt
        path.append(pos)
        events.append(event)
        if pos == env.goal:
            break

    return {
        "label": label,
        "policy": policy,
        "goal_reached": pos == env.goal,
        "steps": len(events),
        "wall_hits": events.count("wall"),
        "away_from_goal_steps": away_steps,
        "path": path,
        "events": events,
    }


def train_condition(label, imagination=False, gated=False, pretrain=False, episodes=520, eval_runs=48):
    env = MazeWorld()
    agent = TinyRecurrentAgent(obs_dim=env.obs_dim, hidden_dim=36, actions=4)
    pretrain_losses = pretrain_world_model(agent, env) if pretrain else []
    opt = torch.optim.Adam(agent.parameters(), lr=0.005)
    training = {"reward": [], "goal_rate": [], "steps": [], "accuracy": []}

    for _ in range(episodes):
        result = run_episode(agent, env, train=True, imagination=imagination, gated=gated)
        opt.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        training["reward"].append(result["reward"])
        training["goal_rate"].append(1.0 if "goal" in result["events"] else 0.0)
        training["steps"].append(result["steps"])
        training["accuracy"].append(result["mean_accuracy"])

    evals = [run_episode(agent, env, train=False, imagination=imagination, gated=gated) for _ in range(eval_runs)]
    return {
        "label": label,
        "imagination": imagination,
        "gated": gated,
        "pretrained_world_model": pretrain,
        "pretrain_final_loss": float(np.mean(pretrain_losses[-25:])) if pretrain_losses else None,
        "training": training,
        "eval_goal_rate": sum(1 for r in evals if "goal" in r["events"]) / len(evals),
        "eval_wall_rate": sum(r["events"].count("wall") for r in evals) / max(sum(r["steps"] for r in evals), 1),
        "eval_mean_steps": float(np.mean([r["steps"] for r in evals])),
        "eval_mean_reward": float(np.mean([r["reward"] for r in evals])),
        "eval_mean_accuracy": float(np.mean([r["mean_accuracy"] for r in evals])),
    }


def moving_average(values, window=45):
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
    axes[0].set_title("2D Maze Goal Completion During Training")
    axes[1].set_title("2D Maze Episode Length")
    axes[2].set_title("2D Maze Reward")
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
    steps = [results[k]["eval_mean_steps"] / 54 for k in labels]
    walls = [results[k]["eval_wall_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, goal, width, label="goal rate", color="#16a3a6")
    ax.bar(x, steps, width, label="mean steps / max", color="#7a7a7a")
    ax.bar(x + width, walls, width, label="wall hit rate", color="#ff8a00")
    ax.set_title("2D Maze Evaluation")
    ax.set_ylabel("rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_maze(path):
    env = MazeWorld()
    fig, ax = plt.subplots(figsize=(6, 5))
    img = np.zeros((env.height, env.width, 3), dtype=np.float32)
    for y, row in enumerate(env.grid):
        for x, cell in enumerate(row):
            img[y, x] = [0.05, 0.05, 0.05] if cell == "#" else [0.93, 0.95, 0.98]
    sx, sy = env.start
    gx, gy = env.goal
    img[sy, sx] = [0.1, 0.65, 0.7]
    img[gy, gx] = [1.0, 0.55, 0.0]
    ax.imshow(img)
    ax.set_title("2D Maze: Locally Closer Can Be Wrong")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_detour_paths(counterexample, path):
    env = MazeWorld()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "myopic_progress_reflex": "#e05a47",
        "pretrained_world_lookahead": "#16a3a6",
    }
    for ax, (label, data) in zip(axes, counterexample.items()):
        img = np.zeros((env.height, env.width, 3), dtype=np.float32)
        for y, row in enumerate(env.grid):
            for x, cell in enumerate(row):
                img[y, x] = [0.05, 0.05, 0.05] if cell == "#" else [0.93, 0.95, 0.98]
        sx, sy = env.start
        gx, gy = env.goal
        img[sy, sx] = [0.1, 0.65, 0.7]
        img[gy, gx] = [1.0, 0.55, 0.0]
        ax.imshow(img)
        xs = [p[0] for p in data["path"]]
        ys = [p[1] for p in data["path"]]
        ax.plot(xs, ys, color=colors[label], lw=3, marker="o", markersize=5)
        ax.set_title(
            f"{label}\n"
            f"goal={data['goal_reached']} steps={data['steps']} walls={data['wall_hits']}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Detour Maze: Local Valence Fails, Lookahead Escapes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(53)
    OUT.mkdir(exist_ok=True)
    counterexample = {
        "myopic_progress_reflex": evaluate_detour_policy("myopic_progress_reflex", "myopic"),
        "pretrained_world_lookahead": evaluate_detour_policy("pretrained_world_lookahead", "lookahead"),
    }
    configs = [
        ("progress_reflex", False, False, False),
        ("naive_imagination", True, False, False),
        ("gated_imagination", True, True, False),
        ("pretrained_gated_imagination", True, True, True),
    ]
    results = {
        label: train_condition(label, imagination=imagination, gated=gated, pretrain=pretrain)
        for label, imagination, gated, pretrain in configs
    }
    serializable = {
        label: {
            "imagination": data["imagination"],
            "gated": data["gated"],
            "pretrained_world_model": data["pretrained_world_model"],
            "pretrain_final_loss": data["pretrain_final_loss"],
            "eval_goal_rate": data["eval_goal_rate"],
            "eval_wall_rate": data["eval_wall_rate"],
            "eval_mean_steps": data["eval_mean_steps"],
            "eval_mean_reward": data["eval_mean_reward"],
            "eval_mean_accuracy": data["eval_mean_accuracy"],
        }
        for label, data in results.items()
    }
    serializable["note"] = (
        "2D detour maze where locally moving closer to the goal can be wrong. The deterministic counter-example isolates myopic progress-valence from pretrained world-model lookahead."
    )
    serializable["detour_counterexample"] = {
        label: {
            "policy": data["policy"],
            "goal_reached": data["goal_reached"],
            "steps": data["steps"],
            "wall_hits": data["wall_hits"],
            "away_from_goal_steps": data["away_from_goal_steps"],
            "path": [list(p) for p in data["path"]],
            "events": data["events"],
        }
        for label, data in counterexample.items()
    }
    (OUT / "maze_imagination_metrics.json").write_text(json.dumps(serializable, indent=2))
    plot_maze(OUT / "maze_layout.png")
    plot_detour_paths(counterexample, OUT / "maze_detour_counterexample.png")
    plot_training(results, OUT / "maze_imagination_training.png")
    plot_eval(results, OUT / "maze_imagination_eval.png")
    print("Maze imagination lab complete")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
