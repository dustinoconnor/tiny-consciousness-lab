#!/usr/bin/env python3
"""Causal router learning / credit-assignment experiment.

Earlier labs showed that intelligence is not "more loops"; it is regulated
routing. This lab upgrades the router from a traffic cop into a small causal
credit-assignment system.

The router listens to three specialists:

- reflex specialist: fast local progress, hazard-blind
- map specialist: goal-directed shortest path, hazard-blind
- safety-map specialist: corrected route planning after local hazard discovery

When an action fails, the causal router asks a counterfactual question:

    "If I had routed to another specialist last step, would the outcome have
    been better?"

It then updates trust for each specialist. The comparison conditions are:

- static_router: fixed routing weights
- uniform_penalty_router: penalizes all specialists after failure
- causal_credit_router: assigns credit/blame to the specialist whose proposed
  action would have changed the outcome

This is a toy routing model, not a general learning algorithm. It isolates the
question: does backward-looking causal credit make routing more intelligent than
global punishment?
"""

import json
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from partial_observer_social_lab import ACTION_NAMES, MOVES
from tiny_lab import OUT, set_seed


GRID = [
    "###############",
    "#S....#.....G.#",
    "#.....#.......#",
    "#.............#",
    "#.....#.......#",
    "#.............#",
    "###############",
]

SPECIALISTS = ["reflex", "map", "safety_map"]


class DynamicHazardMaze:
    def __init__(self, hazard):
        self.grid = [list(row) for row in GRID]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start = self.find("S")
        self.goal = self.find("G")
        self.hazard = hazard

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

    def manhattan(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])


def norm_dist(distance):
    if distance >= 999:
        return -1.0
    return float(np.tanh(-distance / 5.0))


def reflex_values(env, pos):
    values = []
    current = env.manhattan(pos)
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall":
            values.append(-1.0)
        else:
            values.append(0.8 if env.manhattan(nxt) < current else -0.2)
    return np.array(values)


def map_values(env, pos):
    values = []
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall":
            values.append(-1.0)
        elif event == "goal":
            values.append(1.0)
        else:
            values.append(norm_dist(env.shortest_distance(nxt, avoid_hazard=False)))
    return np.array(values)


def safety_map_values(env, pos, known_hazards):
    """Goal-directed values with a small learned hazard mask.

    This specialist is not a full oracle. It only avoids hazards that have been
    discovered by local safety sensing. That makes the routing problem concrete:
    the master must learn when this slower, safer channel is worth trusting.
    """
    values = []
    original_hazard = env.hazard
    for action in range(4):
        nxt, event = env.transition_from(pos, action)
        if event == "wall" or event == "hazard" or nxt in known_hazards:
            values.append(-1.0)
        elif event == "goal":
            values.append(1.0)
        else:
            if known_hazards:
                # Temporarily treat the known hazard as blocked for route
                # planning. The toy maze has one hazard, so this is enough to
                # model a corrected local world model.
                env.hazard = next(iter(known_hazards))
                values.append(norm_dist(env.shortest_distance(nxt, avoid_hazard=True)))
                env.hazard = original_hazard
            else:
                values.append(norm_dist(env.shortest_distance(nxt, avoid_hazard=False)))
    env.hazard = original_hazard
    return np.array(values)


def softmax(x):
    x = np.asarray(x, dtype=float)
    x -= np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def score_outcome(env, pos, action):
    nxt, event = env.transition_from(pos, action)
    if event == "goal":
        return 1.0
    if event == "hazard":
        return -1.0
    if event == "wall":
        return -0.45
    progress = env.shortest_distance(pos, avoid_hazard=True) - env.shortest_distance(nxt, avoid_hazard=True)
    return 0.10 + 0.18 * np.sign(progress)


def local_danger(env, pos):
    return any(env.transition_from(pos, action)[1] == "hazard" for action in range(4))


def context_key(env, pos):
    return "danger" if local_danger(env, pos) else "ordinary"


def route_step(env, pos, trust, known_hazards):
    key = context_key(env, pos)
    proposals = {
        "reflex": reflex_values(env, pos),
        "map": map_values(env, pos),
        "safety_map": safety_map_values(env, pos, known_hazards),
    }
    choices = {name: int(np.argmax(values)) for name, values in proposals.items()}
    weights = softmax(np.array([trust[key][name] for name in SPECIALISTS]))
    routed_to = SPECIALISTS[int(np.argmax(weights))]
    action = choices[routed_to]
    return action, routed_to, proposals, choices, weights, key


def update_trust(condition, trust, key, chosen_action, proposals, env, pos, actual_score, lr=0.42):
    if condition == "static_router":
        return trust, {name: 0.0 for name in SPECIALISTS}

    counterfactual = {
        name: score_outcome(env, pos, int(np.argmax(values)))
        for name, values in proposals.items()
    }
    deltas = {name: counterfactual[name] - actual_score for name in SPECIALISTS}

    if condition == "uniform_penalty_router":
        if actual_score < 0:
            for bucket in trust.values():
                for name in SPECIALISTS:
                    bucket[name] -= 0.08
        else:
            for bucket in trust.values():
                for name in SPECIALISTS:
                    bucket[name] += 0.02
    elif condition == "causal_credit_router":
        # Causal credit is assigned only in the current context. A safety module
        # should not dominate the whole mind just because it was valuable at a
        # hazard boundary; the router learns a small conditional policy.
        for name in SPECIALISTS:
            trust[key][name] += lr * deltas[name]
        # Mild forgetting keeps one early lucky module from dominating forever.
        mean = np.mean([trust[key][name] for name in SPECIALISTS])
        for name in SPECIALISTS:
            trust[key][name] = 0.98 * trust[key][name] + 0.02 * mean
    else:
        raise ValueError(condition)

    for bucket in trust.values():
        for name in SPECIALISTS:
            bucket[name] = float(np.clip(bucket[name], -2.0, 2.0))
    return trust, deltas


def hazard_schedule(seed, episodes):
    rng = np.random.default_rng(seed)
    candidates = [(6, 3), (7, 3), (8, 3), (6, 1), (6, 5), (10, 3)]
    return [candidates[int(rng.integers(0, len(candidates)))] for _ in range(episodes)]


def run_condition(condition, episodes=80, max_steps=42, seed=931):
    hazards = hazard_schedule(seed, episodes)
    trust = {
        "ordinary": {"reflex": 0.20, "map": 0.85, "safety_map": -0.10},
        "danger": {"reflex": 0.10, "map": 0.85, "safety_map": -0.15},
    }
    episode_rows = []
    trace = []
    for ep, hazard in enumerate(hazards):
        env = DynamicHazardMaze(hazard)
        pos = env.start
        known_hazards = set()
        for t in range(max_steps):
            if local_danger(env, pos):
                known_hazards.add(env.hazard)
            action, routed_to, proposals, choices, weights, key = route_step(env, pos, trust, known_hazards)
            actual_score = score_outcome(env, pos, action)
            nxt, event = env.transition_from(pos, action)
            old_trust = {bucket: values.copy() for bucket, values in trust.items()}
            trust, deltas = update_trust(condition, trust, key, action, proposals, env, pos, actual_score)
            trace.append(
                {
                    "episode": ep,
                    "t": t,
                    "pos": list(pos),
                    "hazard": list(hazard),
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "event": event,
                    "context": key,
                    "routed_to": routed_to,
                    "actual_score": float(actual_score),
                    "reflex_choice": choices["reflex"],
                    "map_choice": choices["map"],
                    "safety_map_choice": choices["safety_map"],
                    "reflex_weight": float(weights[0]),
                    "map_weight": float(weights[1]),
                    "safety_map_weight": float(weights[2]),
                    "reflex_trust": float(old_trust[key]["reflex"]),
                    "map_trust": float(old_trust[key]["map"]),
                    "safety_map_trust": float(old_trust[key]["safety_map"]),
                    "reflex_delta": float(deltas["reflex"]),
                    "map_delta": float(deltas["map"]),
                    "safety_map_delta": float(deltas["safety_map"]),
                }
            )
            pos = nxt
            if event in {"goal", "hazard"}:
                break
        episode_rows.append(
            {
                "episode": ep,
                "goal_reached": pos == env.goal,
                "hit_hazard": pos == env.hazard,
                "steps": t + 1,
                "final_reflex_trust": trust["ordinary"]["reflex"],
                "final_map_trust": trust["ordinary"]["map"],
                "final_safety_map_trust": trust["ordinary"]["safety_map"],
                "danger_reflex_trust": trust["danger"]["reflex"],
                "danger_map_trust": trust["danger"]["map"],
                "danger_safety_map_trust": trust["danger"]["safety_map"],
            }
        )
    return episode_rows, trace


def summarize(episodes, trace):
    late = episodes[len(episodes) // 2 :]
    hazards = [e for e in episodes if e["hit_hazard"]]
    route_entropy = []
    for row in trace:
        weights = np.array([row["reflex_weight"], row["map_weight"], row["safety_map_weight"]])
        route_entropy.append(float(-np.sum(weights * np.log2(np.clip(weights, 1e-9, 1.0)))))
    return {
        "goal_rate": float(np.mean([e["goal_reached"] for e in episodes])),
        "late_goal_rate": float(np.mean([e["goal_reached"] for e in late])),
        "hazard_rate": float(len(hazards) / len(episodes)),
        "late_hazard_rate": float(np.mean([e["hit_hazard"] for e in late])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "mean_route_entropy": float(np.mean(route_entropy)),
        "final_reflex_trust": float(episodes[-1]["final_reflex_trust"]),
        "final_map_trust": float(episodes[-1]["final_map_trust"]),
        "final_safety_map_trust": float(episodes[-1]["final_safety_map_trust"]),
        "danger_map_trust": float(episodes[-1]["danger_map_trust"]),
        "danger_safety_map_trust": float(episodes[-1]["danger_safety_map_trust"]),
        "danger_safety_advantage": float(
            episodes[-1]["danger_safety_map_trust"] - episodes[-1]["danger_map_trust"]
        ),
    }


def moving_average(values, window=8):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="same")


def plot_summary(summary, path):
    names = list(summary)
    metrics = ["goal_rate", "late_goal_rate", "hazard_rate", "mean_route_entropy"]
    x = np.arange(len(names))
    width = 0.20
    colors = ["#16a3a6", "#7c3aed", "#e05a47", "#ff8a00"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [summary[n][metric] for n in names], width, label=metric, color=colors[i])
    ax.set_title("Causal Router Learning Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_learning(results, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for name, payload in results.items():
        episodes = payload["episodes"]
        x = [e["episode"] for e in episodes]
        axes[0].plot(x, moving_average([e["goal_reached"] for e in episodes]), label=name)
        axes[1].plot(x, moving_average([e["hit_hazard"] for e in episodes]), label=name)
        axes[2].plot(x, [e["final_map_trust"] for e in episodes], label=f"{name}: map")
        axes[2].plot(x, [e["final_safety_map_trust"] for e in episodes], ls="--", label=f"{name}: safety-map ordinary")
        axes[2].plot(x, [e["danger_safety_map_trust"] for e in episodes], ls=":", label=f"{name}: safety-map danger")
        axes[3].plot(x, [e["final_reflex_trust"] for e in episodes], label=f"{name}: reflex")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("goal rate")
    axes[1].set_ylabel("hazard rate")
    axes[2].set_ylabel("map/safety trust")
    axes[3].set_ylabel("reflex trust")
    axes[3].set_xlabel("episode")
    axes[0].set_title("Routing Credit Assignment Over Episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(931)
    OUT.mkdir(exist_ok=True)
    conditions = ["static_router", "uniform_penalty_router", "causal_credit_router"]
    results = {}
    for condition in conditions:
        episodes, trace = run_condition(condition)
        results[condition] = {
            "episodes": episodes,
            "trace": trace,
        }
    summary = {name: summarize(payload["episodes"], payload["trace"]) for name, payload in results.items()}
    payload = {
        "note": (
            "Routing credit-assignment toy. The causal router runs a backward-looking counterfactual update "
            "to learn which specialist would have improved the last outcome."
        ),
        "summary": summary,
        "thesis": (
            "Intelligent routing requires causal credit assignment. A master controller should not merely route signals; "
            "it should learn which internal source caused success or failure."
        ),
    }
    (OUT / "causal_router_learning_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(summary, OUT / "causal_router_learning_summary.png")
    plot_learning(results, OUT / "causal_router_learning_trust.png")
    print("Causal router learning lab complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
