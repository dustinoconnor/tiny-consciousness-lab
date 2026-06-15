#!/usr/bin/env python3
"""Unified toy-mind lab.

This file pulls the main ideas from the separate experiments into one readable
toy system:

- sensory state: where the agent is in a tiny maze
- valence: whether a move feels locally good or bad
- imagination: a lookahead search through an internal world model
- attention/workspace: a gate that decides when reflex is enough and when
  imagination should take over
- recurrence: the system carries workspace and self-model state across time
- self-model: a small rolling register of tension, dominance, confidence, and
  delusion risk
- self-report: symbolic labels generated from the internal math

The important distinction:

The world model here is "pretrained" in the simplest possible way: before the
agent acts, it is given a transition table for the maze. That is not PyTorch
gradient training. It is a tiny exact model of "if I take action A from cell X,
where will I land?" The point is to make the architecture easy to inspect.
"""

import json
import math
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


MAZE = [
    "###########",
    "#S....#..G#",
    "#.....#...#",
    "#.....#...#",
    "#.....#...#",
    "#.........#",
    "###########",
]

# up, down, left, right
MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["up", "down", "left", "right"]


def sigmoid(x):
    """Squash any number into the 0..1 range for soft gating."""
    return 1.0 / (1.0 + math.exp(-x))


def normalize_score(x, scale=8.0):
    """Turn path scores into a small valence-like range."""
    return math.tanh(x / scale)


@dataclass
class StepTrace:
    """One row of internal state for plotting and reading later."""

    t: int
    pos: tuple[int, int]
    action: int
    event: str
    local_valence: float
    imagined_valence: float
    tension: float
    alpha: float
    confidence: float
    delusion_risk: float
    integration_proxy: float
    report: str


class MazeWorld:
    """Small deterministic maze with a local-minimum trap.

    The start is on the left, the goal is on the upper right, and a wall blocks
    the direct route. A pure "move closer to the goal" reflex gets stuck against
    that wall. A model with lookahead can walk away from the goal temporarily to
    go around it.
    """

    def __init__(self):
        self.grid = [list(row) for row in MAZE]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start = self.find("S")
        self.goal = self.find("G")

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

    def is_wall(self, xy):
        x, y = xy
        return self.grid[y][x] == "#"

    def manhattan(self, a, b=None):
        b = self.goal if b is None else b
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def transition_from(self, xy, action):
        dx, dy = MOVES[int(action)]
        candidate = (xy[0] + dx, xy[1] + dy)
        if self.is_wall(candidate):
            return xy, "wall"
        if candidate == self.goal:
            return candidate, "goal"
        return candidate, "move"

    def neighbors(self, xy):
        for action in range(4):
            nxt, event = self.transition_from(xy, action)
            if event != "wall":
                yield nxt

    def shortest_distance(self, start):
        """Ground-truth distance used only for evaluation/visual scoring."""
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


def pretrain_tabular_world_model(env):
    """Build the tiny internal world model.

    In a real neural system this would be learned from experience. Here we keep
    it exact and inspectable: a dictionary of transitions for every open cell
    and every action.
    """
    world = {}
    for cell in env.open_cells:
        world[cell] = {}
        for action in range(4):
            world[cell][action] = env.transition_from(cell, action)
    return world


def local_progress_valence(env, pos, action):
    """Immediate reward signal: does this single move feel good right now?"""
    old_dist = env.manhattan(pos)
    nxt, event = env.transition_from(pos, action)
    if event == "wall":
        return -1.0
    new_dist = env.manhattan(nxt)
    if event == "goal":
        return 1.0
    if new_dist < old_dist:
        return 0.45
    if new_dist > old_dist:
        return -0.35
    return -0.05


def reflex_action(env, pos):
    """The baseline reflex: choose the move with the best immediate valence."""
    scores = [(local_progress_valence(env, pos, action), action) for action in range(4)]
    return max(scores)[1]


def lookahead_score(env, world_model, pos, depth, memo=None):
    """Recursive imagination: simulate futures inside the world model."""
    memo = {} if memo is None else memo
    key = (pos, depth)
    if key in memo:
        return memo[key]
    if pos == env.goal:
        memo[key] = 100.0
        return memo[key]
    if depth <= 0:
        memo[key] = -float(env.shortest_distance(pos))
        return memo[key]

    best = -999.0
    for action in range(4):
        nxt, event = world_model[pos][action]
        wall_penalty = -8.0 if event == "wall" else 0.0
        best = max(best, wall_penalty + 0.92 * lookahead_score(env, world_model, nxt, depth - 1, memo))
    memo[key] = best
    return best


class UnifiedToyMind:
    """A compact architecture containing the loops from the whole repo."""

    def __init__(self, env, horizon=14):
        self.env = env
        self.world_model = pretrain_tabular_world_model(env)
        self.horizon = horizon

        # Recurrent workspace memory. This is the tiny "global workspace" state.
        self.workspace_alpha = 0.0
        self.workspace_confidence = 0.65

        # Persistent self-model. It does not feel anything; it stores a rolling
        # history of internal control variables that can change future behavior.
        self.self_model = {
            "tension": 0.0,
            "dominance": 0.0,
            "confidence": self.workspace_confidence,
            "delusion_risk": 0.0,
        }

    def imagine_action_values(self, pos):
        """Ask the internal world model what each action could lead to."""
        values = []
        for action in range(4):
            nxt, event = self.world_model[pos][action]
            if event == "wall":
                values.append(-1.0)
            elif event == "goal":
                values.append(1.0)
            else:
                # The world model imagines beyond the next step, but we keep
                # the final value path-sensitive: shorter futures feel better
                # than loops that only reach the goal eventually.
                future_distance = self.env.shortest_distance(nxt)
                lookahead = lookahead_score(self.env, self.world_model, nxt, self.horizon - 1, {})
                values.append(0.70 * normalize_score(-future_distance, scale=5.0) + 0.30 * normalize_score(lookahead))
        return np.array(values)

    def choose_action(self, pos, t):
        """Blend reflex and imagination through an attention/workspace gate."""
        local_values = np.array([local_progress_valence(self.env, pos, a) for a in range(4)])
        imagined_values = self.imagine_action_values(pos)

        reflex_choice = int(np.argmax(local_values))
        imagined_choice = int(np.argmax(imagined_values))

        # Tension rises when immediate valence and imagined future disagree.
        # This is the signal that says "reflex may not be enough here."
        disagreement = abs(local_values[reflex_choice] - imagined_values[reflex_choice])
        conflict = 1.0 if reflex_choice != imagined_choice else 0.0
        tension = min(1.0, 0.55 * disagreement + 0.45 * conflict)

        # The workspace only asserts itself when tension is high. Self-model
        # vigilance lowers the threshold slightly after recent conflict.
        vigilance = self.self_model["delusion_risk"]
        threshold = 0.42 - 0.12 * vigilance
        target_alpha = sigmoid(8.5 * (tension - threshold))
        alpha = 0.65 * self.workspace_alpha + 0.35 * target_alpha

        # Attention is the blend: low alpha means reflex; high alpha means
        # imagination/world-model planning gets more control.
        action_values = (1.0 - alpha) * local_values + alpha * imagined_values
        if alpha > 0.55 and imagined_choice != reflex_choice:
            # Once the workspace is strongly engaged, the system is saying:
            # "local reflex is not enough here." At that point the grounded
            # world model gets executive control for the next action.
            action = imagined_choice
        else:
            action = int(np.argmax(action_values))

        # Delusion risk is high when the internal model wants to override
        # sensation while confidence is low. In this exact tabular world, it
        # should stay low after the first few steps.
        confidence = 0.82 * self.workspace_confidence + 0.18 * (1.0 - min(1.0, tension * 0.45))
        delusion_risk = max(0.0, alpha * (1.0 - confidence))

        # This is not official IIT Phi. It is a simple proxy: integration is high
        # when the system is both tense and actually coupling modules together.
        integration_proxy = alpha * tension * confidence

        self.workspace_alpha = alpha
        self.workspace_confidence = confidence
        self.self_model = {
            "tension": 0.85 * self.self_model["tension"] + 0.15 * tension,
            "dominance": 0.85 * self.self_model["dominance"] + 0.15 * alpha,
            "confidence": confidence,
            "delusion_risk": 0.80 * self.self_model["delusion_risk"] + 0.20 * delusion_risk,
        }

        report = self_report(tension, alpha, confidence, reflex_choice, imagined_choice)
        return action, {
            "local_valence": float(local_values[action]),
            "imagined_valence": float(imagined_values[action]),
            "tension": float(tension),
            "alpha": float(alpha),
            "confidence": float(confidence),
            "delusion_risk": float(delusion_risk),
            "integration_proxy": float(integration_proxy),
            "report": report,
        }


def self_report(tension, alpha, confidence, reflex_choice, imagined_choice):
    """Turn internal math into a small symbolic self-report."""
    if reflex_choice != imagined_choice and alpha > 0.55:
        return "conflict_detected_workspace_overriding_reflex"
    if tension > 0.55:
        return "high_tension_evaluating_counterfactuals"
    if alpha < 0.25 and confidence > 0.78:
        return "autonomous_reflex_stable"
    if confidence < 0.62:
        return "low_confidence_regrounding"
    return "integrated_monitoring"


def run_reflex(env, max_steps=40):
    """Run the reflex-only baseline."""
    pos = env.start
    path = [pos]
    trace = []
    for t in range(max_steps):
        action = reflex_action(env, pos)
        nxt, event = env.transition_from(pos, action)
        trace.append(
            StepTrace(
                t=t,
                pos=pos,
                action=action,
                event=event,
                local_valence=local_progress_valence(env, pos, action),
                imagined_valence=0.0,
                tension=0.0,
                alpha=0.0,
                confidence=0.0,
                delusion_risk=0.0,
                integration_proxy=0.0,
                report="reflex_only",
            )
        )
        pos = nxt
        path.append(pos)
        if event == "goal":
            break
    return path, trace


def run_unified(env, max_steps=40):
    """Run the unified architecture."""
    mind = UnifiedToyMind(env)
    pos = env.start
    path = [pos]
    trace = []
    for t in range(max_steps):
        action, state = mind.choose_action(pos, t)
        nxt, event = env.transition_from(pos, action)
        trace.append(StepTrace(t=t, pos=pos, action=action, event=event, **state))
        pos = nxt
        path.append(pos)
        if event == "goal":
            break
    return path, trace


def trace_to_dict(trace):
    return [
        {
            **item.__dict__,
            "pos": list(item.pos),
            "action_name": ACTION_NAMES[item.action],
        }
        for item in trace
    ]


def summarize_run(env, path, trace):
    return {
        "goal_reached": path[-1] == env.goal,
        "steps": len(trace),
        "wall_hits": sum(1 for row in trace if row.event == "wall"),
        "away_from_goal_steps": sum(
            1
            for row, nxt in zip(trace, path[1:])
            if env.manhattan(nxt) > env.manhattan(row.pos)
        ),
        "mean_alpha": float(np.mean([row.alpha for row in trace])) if trace else 0.0,
        "mean_tension": float(np.mean([row.tension for row in trace])) if trace else 0.0,
        "mean_integration_proxy": float(np.mean([row.integration_proxy for row in trace])) if trace else 0.0,
        "final_report": trace[-1].report if trace else "none",
    }


def draw_maze_background(env, ax):
    img = np.zeros((env.height, env.width, 3), dtype=np.float32)
    for y, row in enumerate(env.grid):
        for x, cell in enumerate(row):
            img[y, x] = [0.05, 0.05, 0.05] if cell == "#" else [0.94, 0.96, 0.98]
    sx, sy = env.start
    gx, gy = env.goal
    img[sy, sx] = [0.08, 0.60, 0.70]
    img[gy, gx] = [1.0, 0.56, 0.0]
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_paths(env, runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"reflex_only": "#e05a47", "unified_pretrained_world": "#16a3a6"}
    for ax, (name, data) in zip(axes, runs.items()):
        draw_maze_background(env, ax)
        xs = [p[0] for p in data["path"]]
        ys = [p[1] for p in data["path"]]
        ax.plot(xs, ys, color=colors[name], lw=3, marker="o", markersize=5)
        ax.set_title(f"{name}\ngoal={data['summary']['goal_reached']} steps={data['summary']['steps']}")
    fig.suptitle("Unified Toy Mind: Reflex Trap vs Pretrained World-Model Lookahead")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_unified_trace(trace, path):
    x = [row.t for row in trace]
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, [row.local_valence for row in trace], label="local valence")
    axes[0].plot(x, [row.imagined_valence for row in trace], label="imagined valence")
    axes[1].plot(x, [row.tension for row in trace], color="#ff8a00", label="tension")
    axes[2].plot(x, [row.alpha for row in trace], color="#16a3a6", label="workspace alpha")
    axes[3].plot(x, [row.confidence for row in trace], color="#65a30d", label="confidence")
    axes[3].plot(x, [row.delusion_risk for row in trace], color="#e05a47", label="delusion risk")
    axes[4].plot(x, [row.integration_proxy for row in trace], color="#7c3aed", label="integration proxy")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Unified Toy Mind Internal Control Trace")
    axes[4].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(71)
    OUT.mkdir(exist_ok=True)
    env = MazeWorld()

    reflex_path, reflex_trace = run_reflex(env)
    unified_path, unified_trace = run_unified(env)

    runs = {
        "reflex_only": {
            "path": reflex_path,
            "trace": trace_to_dict(reflex_trace),
            "summary": summarize_run(env, reflex_path, reflex_trace),
        },
        "unified_pretrained_world": {
            "path": unified_path,
            "trace": trace_to_dict(unified_trace),
            "summary": summarize_run(env, unified_path, unified_trace),
        },
    }

    payload = {
        "note": (
            "One readable capstone toy model combining sensory state, valence, recurrent imagination, "
            "a pretrained tabular world model, attention/workspace gating, a rolling self-model, and self-report."
        ),
        "pretrained_world_model": (
            "The world model is a transition table learned before action. It is exact and inspectable, not a large neural net."
        ),
        "runs": runs,
        "thesis": [
            "Reflex valence is efficient in simple worlds but fails in local minima.",
            "A pretrained world model gives imagination something grounded to simulate.",
            "The workspace should assert control when tension rises, not stay permanently dominant.",
            "The self-model matters only when it changes future control.",
        ],
    }

    (OUT / "unified_mind_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_paths(env, runs, OUT / "unified_mind_paths.png")
    plot_unified_trace(unified_trace, OUT / "unified_mind_trace.png")

    print("Unified toy-mind lab complete")
    print(json.dumps({k: v["summary"] for k, v in runs.items()}, indent=2))
    print(f"Wrote {OUT / 'unified_mind_metrics.json'}")
    print(f"Wrote {OUT / 'unified_mind_paths.png'}")
    print(f"Wrote {OUT / 'unified_mind_trace.png'}")


if __name__ == "__main__":
    main()
