#!/usr/bin/env python3
"""Partial-observer social workspace experiment.

This lab creates a case where two agents can outperform either one alone.

The world has two routes through a wall:

- the short upper opening contains a hidden hazard
- the longer lower opening is safe

The agents are partial observers:

- map_agent sees walls and the goal, but cannot see hidden hazards
- safety_agent sees hazards, but has no goal-directed map
- combined_workspace blends goal direction with safety veto

This tests a stronger social-cognition claim than the previous lab:

Social cognition helps when agents have complementary, non-redundant access to
reality. It is not useful merely because there is another agent in the loop.
"""

import json
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed


GRID = [
    "#############",
    "#S....#....G#",
    "#.....#.....#",
    "#...........#",
    "#.....#.....#",
    "#...........#",
    "#############",
]

MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["up", "down", "left", "right"]
HIDDEN_HAZARD = (6, 3)


@dataclass
class PartialStep:
    t: int
    pos: tuple[int, int]
    action: int
    event: str
    map_choice: int
    safety_choice: int
    beta_safety: float
    tension: float
    safety_veto: float
    report: str


class HiddenHazardMaze:
    def __init__(self):
        self.grid = [list(row) for row in GRID]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start = self.find("S")
        self.goal = self.find("G")
        self.hazard = HIDDEN_HAZARD

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

    def is_wall(self, pos):
        x, y = pos
        return self.grid[y][x] == "#"

    def is_hazard(self, pos):
        return pos == self.hazard

    def transition_from(self, pos, action):
        dx, dy = MOVES[int(action)]
        nxt = (pos[0] + dx, pos[1] + dy)
        if self.is_wall(nxt):
            return pos, "wall"
        if self.is_hazard(nxt):
            return nxt, "hazard"
        if nxt == self.goal:
            return nxt, "goal"
        return nxt, "move"

    def neighbors(self, pos, avoid_hazard=False):
        for action in range(4):
            nxt, event = self.transition_from(pos, action)
            if event == "wall":
                continue
            if avoid_hazard and event == "hazard":
                continue
            yield nxt

    def shortest_distance(self, start, avoid_hazard=False):
        q = deque([(start, 0)])
        seen = {start}
        while q:
            cur, dist = q.popleft()
            if cur == self.goal:
                return dist
            for nxt in self.neighbors(cur, avoid_hazard=avoid_hazard):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, dist + 1))
        return 999


def norm_value(distance):
    if distance >= 999:
        return -1.0
    return float(np.tanh(-distance / 5.0))


def map_agent_values(env, pos):
    """Goal-directed map values, blind to the hidden hazard."""
    values = []
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall":
            values.append(-1.0)
        elif event == "goal":
            values.append(1.0)
        else:
            values.append(norm_value(env.shortest_distance(nxt, avoid_hazard=False)))
    return np.array(values)


def safety_agent_values(env, pos):
    """Safety values, blind to the goal except for local hazard avoidance."""
    values = []
    for action in range(4):
        _, event = env.transition_from(pos, action)
        if event == "wall":
            values.append(-0.6)
        elif event == "hazard":
            values.append(-1.0)
        else:
            values.append(0.15)
    return np.array(values)


def oracle_safe_values(env, pos):
    """For reference: full map plus hazard knowledge."""
    values = []
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall" or event == "hazard":
            values.append(-1.0)
        elif event == "goal":
            values.append(1.0)
        else:
            values.append(norm_value(env.shortest_distance(nxt, avoid_hazard=True)))
    return np.array(values)


def choose_action(env, pos, condition, hazard_known=False):
    map_values = map_agent_values(env, pos)
    safety_values = safety_agent_values(env, pos)
    oracle_values = oracle_safe_values(env, pos)

    map_choice = int(np.argmax(map_values))
    safety_choice = int(np.argmax(safety_values))
    safety_veto = 1.0 if safety_values[map_choice] <= -0.9 else 0.0
    tension = 0.65 * safety_veto + 0.35 * (map_choice != safety_choice)
    beta_safety = 1.0 / (1.0 + np.exp(-9.0 * (tension - 0.35)))

    if condition == "map_only":
        action = map_choice
        report = "map_blind_to_hazard"
    elif condition == "safety_only":
        action = safety_choice
        report = "safe_but_goal_blind"
    elif condition == "combined_workspace":
        combined = (1.0 - beta_safety) * map_values + beta_safety * (map_values + 1.4 * safety_values)
        if safety_veto or hazard_known:
            # Once safety identifies the hidden hazard, the workspace can
            # re-run goal-directed planning with that cell masked out. Neither
            # partial observer can do this alone: map lacks hazard knowledge,
            # safety lacks goal direction.
            combined = oracle_safe_values(env, pos)
        action = int(np.argmax(combined))
        report = "safety_veto_reroutes_map" if safety_veto else "map_guides_safe_motion"
    elif condition == "oracle_full_agent":
        action = int(np.argmax(oracle_values))
        report = "full_internal_world_model"
    else:
        raise ValueError(condition)

    return action, {
        "map_choice": map_choice,
        "safety_choice": safety_choice,
        "beta_safety": float(beta_safety if condition == "combined_workspace" else 0.0),
        "tension": float(tension),
        "safety_veto": float(safety_veto),
        "report": report,
    }


def run_condition(condition, max_steps=48):
    env = HiddenHazardMaze()
    pos = env.start
    path = [pos]
    trace = []
    hazard_known = False
    for t in range(max_steps):
        action, state = choose_action(env, pos, condition, hazard_known=hazard_known)
        if condition == "combined_workspace" and state["safety_veto"] > 0.5:
            hazard_known = True
        nxt, event = env.transition_from(pos, action)
        trace.append(PartialStep(t=t, pos=pos, action=action, event=event, **state))
        pos = nxt
        path.append(pos)
        if event in {"goal", "hazard"}:
            break
    return env, path, trace


def summarize(env, path, trace):
    return {
        "goal_reached": path[-1] == env.goal,
        "hit_hazard": path[-1] == env.hazard or any(row.event == "hazard" for row in trace),
        "steps": len(trace),
        "mean_beta_safety": float(np.mean([row.beta_safety for row in trace])) if trace else 0.0,
        "mean_tension": float(np.mean([row.tension for row in trace])) if trace else 0.0,
        "safety_veto_count": int(sum(row.safety_veto > 0.5 for row in trace)),
        "final_report": trace[-1].report if trace else "none",
    }


def trace_to_dict(trace):
    return [
        {
            **row.__dict__,
            "pos": list(row.pos),
            "action_name": ACTION_NAMES[row.action],
            "map_action_name": ACTION_NAMES[row.map_choice],
            "safety_action_name": ACTION_NAMES[row.safety_choice],
        }
        for row in trace
    ]


def draw_world(env, ax):
    img = np.zeros((env.height, env.width, 3), dtype=np.float32)
    for y, row in enumerate(env.grid):
        for x, cell in enumerate(row):
            img[y, x] = [0.05, 0.05, 0.05] if cell == "#" else [0.94, 0.96, 0.98]
    sx, sy = env.start
    gx, gy = env.goal
    hx, hy = env.hazard
    img[sy, sx] = [0.08, 0.60, 0.70]
    img[gy, gx] = [1.0, 0.56, 0.0]
    img[hy, hx] = [0.90, 0.10, 0.10]
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_paths(results, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = {
        "map_only": "#e05a47",
        "safety_only": "#7a7a7a",
        "combined_workspace": "#16a3a6",
        "oracle_full_agent": "#7c3aed",
    }
    for ax, (name, data) in zip(axes, results.items()):
        env = data["env"]
        draw_world(env, ax)
        xs = [p[0] for p in data["path"]]
        ys = [p[1] for p in data["path"]]
        ax.plot(xs, ys, color=colors[name], lw=3, marker="o", markersize=4)
        s = data["summary"]
        ax.set_title(f"{name}\ngoal={s['goal_reached']} hazard={s['hit_hazard']} steps={s['steps']}")
    fig.suptitle("Partial Observers: Complementary Agents Beat Either Alone")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trace(trace, path):
    x = [row.t for row in trace]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(x, [row.beta_safety for row in trace], label="safety beta", color="#16a3a6")
    axes[1].plot(x, [row.tension for row in trace], label="cross-agent tension", color="#ff8a00")
    axes[2].plot(x, [row.safety_veto for row in trace], label="safety veto", color="#e05a47")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Combined Workspace: Safety Gate Opens at Hidden Hazard")
    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(results, path):
    names = list(results)
    goal = [float(results[n]["summary"]["goal_reached"]) for n in names]
    hazard = [float(results[n]["summary"]["hit_hazard"]) for n in names]
    beta = [results[n]["summary"]["mean_beta_safety"] for n in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, goal, width, label="goal reached", color="#16a3a6")
    ax.bar(x, hazard, width, label="hit hazard", color="#e05a47")
    ax.bar(x + width, beta, width, label="mean safety beta", color="#7c3aed")
    ax.set_ylim(0, 1.05)
    ax.set_title("Partial Observability: When Two Agents Beat One")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(121)
    OUT.mkdir(exist_ok=True)
    conditions = ["map_only", "safety_only", "combined_workspace", "oracle_full_agent"]
    results = {}
    for condition in conditions:
        env, path, trace = run_condition(condition)
        results[condition] = {
            "summary": summarize(env, path, trace),
            "path": [list(p) for p in path],
            "trace": trace_to_dict(trace),
            "raw_trace": trace,
            "env": env,
        }

    serializable = {
        name: {k: v for k, v in data.items() if k not in {"env", "raw_trace"}}
        for name, data in results.items()
    }
    payload = {
        "note": "Tests complementary partial observers: map-only, safety-only, combined workspace, and oracle full agent.",
        "results": serializable,
        "thesis": (
            "Two agents outperform one when their information channels are complementary and the workspace can gate safety into goal-directed planning."
        ),
    }
    (OUT / "partial_observer_social_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_paths(results, OUT / "partial_observer_social_paths.png")
    plot_trace(results["combined_workspace"]["raw_trace"], OUT / "partial_observer_social_trace.png")
    plot_summary(results, OUT / "partial_observer_social_summary.png")

    print("Partial-observer social lab complete")
    print(json.dumps({name: data["summary"] for name, data in serializable.items()}, indent=2))


if __name__ == "__main__":
    main()
