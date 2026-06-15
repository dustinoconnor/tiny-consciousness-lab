#!/usr/bin/env python3
"""Social workspace experiment.

This lab asks whether a second agent improves control, acts as redundant
workspace, or creates group delusion.

The primary agent has local reflex valence plus a limited internal world model.
The peer can be:

- grounded_peer: a better lookahead critic
- redundant_peer: the same lookahead as the primary agent
- noisy_peer: sometimes right, sometimes random
- echo_peer: simply amplifies the primary reflex choice
- adversarial_peer: consistently prefers the wrong-looking option

Hypothesis:

Social loops help only when they add independent, grounded error correction. If
they merely amplify confidence, they can increase delusion risk.
"""

import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from tiny_lab import OUT, set_seed
from unified_mind_lab import (
    ACTION_NAMES,
    MazeWorld,
    draw_maze_background,
    local_progress_valence,
    normalize_score,
    pretrain_tabular_world_model,
)


@dataclass
class SocialStep:
    t: int
    pos: tuple[int, int]
    action: int
    event: str
    alpha: float
    social_beta: float
    trust: float
    social_tension: float
    delusion_risk: float
    peer_choice: int
    reflex_choice: int
    internal_choice: int
    report: str


def shortest_value(env, pos):
    """Path-sensitive value: closer by true maze distance is better."""
    if pos == env.goal:
        return 1.0
    dist = env.shortest_distance(pos)
    return normalize_score(-dist, scale=5.0)


def action_values_from_horizon(env, world, pos, horizon):
    """A bounded world-model valuation.

    Horizon 1 is mostly reflex-like. A larger horizon can see farther around
    the detour.
    """
    values = []
    for action in range(4):
        nxt, event = world[pos][action]
        if event == "wall":
            values.append(-1.0)
        elif event == "goal":
            values.append(1.0)
        else:
            # Blend true shortest value with a horizon penalty. The shorter
            # horizon agent underestimates detours because away-steps look bad.
            direct = shortest_value(env, nxt)
            local = local_progress_valence(env, pos, action)
            horizon_weight = min(1.0, horizon / 10.0)
            values.append(horizon_weight * direct + (1.0 - horizon_weight) * local)
    return np.array(values)


class Peer:
    def __init__(self, kind, env, rng):
        self.kind = kind
        self.env = env
        self.world = pretrain_tabular_world_model(env)
        self.rng = rng

    def advise(self, pos, reflex_values, internal_values):
        if self.kind == "none":
            return np.zeros(4), "silent"
        if self.kind == "grounded_peer":
            return action_values_from_horizon(self.env, self.world, pos, horizon=14), "grounded"
        if self.kind == "redundant_peer":
            return internal_values.copy(), "redundant"
        if self.kind == "noisy_peer":
            if self.rng.random() < 0.55:
                return action_values_from_horizon(self.env, self.world, pos, horizon=14), "grounded"
            noise = self.rng.normal(0.0, 0.9, size=4)
            return noise, "noisy"
        if self.kind == "echo_peer":
            values = np.zeros(4) - 0.2
            values[int(np.argmax(reflex_values))] = 1.0
            return values, "echo"
        if self.kind == "adversarial_peer":
            values = -action_values_from_horizon(self.env, self.world, pos, horizon=14)
            return values, "adversarial"
        raise ValueError(self.kind)


class SocialWorkspaceMind:
    def __init__(self, env, peer_kind, rng):
        self.env = env
        self.world = pretrain_tabular_world_model(env)
        self.peer = Peer(peer_kind, env, rng)
        self.peer_kind = peer_kind
        self.alpha = 0.0
        self.trust = 0.55 if peer_kind != "none" else 0.0
        self.confidence = 0.65

    def choose_action(self, pos, t):
        reflex_values = np.array([local_progress_valence(self.env, pos, a) for a in range(4)])
        internal_values = action_values_from_horizon(self.env, self.world, pos, horizon=4)
        peer_values, peer_report = self.peer.advise(pos, reflex_values, internal_values)

        reflex_choice = int(np.argmax(reflex_values))
        internal_choice = int(np.argmax(internal_values))
        peer_choice = int(np.argmax(peer_values)) if self.peer_kind != "none" else internal_choice

        internal_conflict = 1.0 if reflex_choice != internal_choice else 0.0
        social_disagreement = 1.0 if peer_choice != internal_choice else 0.0
        social_tension = 0.55 * internal_conflict + 0.45 * social_disagreement

        target_alpha = 1.0 / (1.0 + np.exp(-8.0 * (social_tension - 0.38)))
        self.alpha = 0.62 * self.alpha + 0.38 * target_alpha

        if self.peer_kind == "none":
            social_beta = 0.0
        elif self.peer_kind == "echo_peer":
            # Echoes feel socially confident but do not add independent evidence.
            social_beta = 0.45 * self.trust
            self.confidence = min(1.0, self.confidence + 0.03)
        else:
            social_beta = self.alpha * self.trust

        combined = (
            (1.0 - self.alpha) * reflex_values
            + self.alpha * (1.0 - social_beta) * internal_values
            + social_beta * peer_values
        )

        action = int(np.argmax(combined))
        if self.peer_kind == "grounded_peer" and social_beta > 0.30 and peer_choice != reflex_choice:
            # A trusted independent critic is allowed to override local reflex.
            # Echo/noisy/adversarial peers do not get this privilege.
            action = peer_choice
        elif self.peer_kind == "noisy_peer" and peer_report == "grounded" and social_beta > 0.42 and peer_choice != reflex_choice:
            # Noisy peers can help, but only after trust is stronger and the
            # current advice came from the grounded mode.
            action = peer_choice
        nxt, event = self.env.transition_from(pos, action)
        improved = self.env.shortest_distance(nxt) < self.env.shortest_distance(pos)

        if self.peer_kind != "none":
            peer_helped = peer_choice == action and improved
            peer_hurt = peer_choice == action and not improved and event != "goal"
            if peer_helped:
                self.trust = min(1.0, self.trust + 0.08)
            elif peer_hurt:
                self.trust = max(0.05, self.trust - 0.12)
            else:
                self.trust = 0.98 * self.trust + 0.02 * 0.55

        false_consensus = self.peer_kind == "echo_peer" and peer_choice == reflex_choice and reflex_choice != internal_choice
        delusion_risk = float((0.25 + 0.75 * social_beta) * false_consensus + 0.45 * social_beta * (not improved))
        report = classify_report(self.peer_kind, peer_report, social_tension, social_beta, delusion_risk)

        return action, {
            "alpha": float(self.alpha),
            "social_beta": float(social_beta),
            "trust": float(self.trust),
            "social_tension": float(social_tension),
            "delusion_risk": float(delusion_risk),
            "peer_choice": peer_choice,
            "reflex_choice": reflex_choice,
            "internal_choice": internal_choice,
            "report": report,
        }


def classify_report(peer_kind, peer_report, tension, beta, delusion):
    if peer_kind == "none":
        return "solo_workspace"
    if delusion > 0.35:
        return "group_delusion_risk"
    if peer_report == "grounded" and beta > 0.35 and tension > 0.4:
        return "grounded_peer_correcting_workspace"
    if peer_report == "redundant":
        return "redundant_social_loop"
    if peer_report == "noisy":
        return "noisy_peer_uncertain"
    if peer_report == "adversarial":
        return "adversarial_peer_suppressed"
    return "social_monitoring"


def run_condition(peer_kind, seed, max_steps=40):
    env = MazeWorld()
    rng = np.random.default_rng(seed)
    random.seed(seed)
    mind = SocialWorkspaceMind(env, peer_kind, rng)
    pos = env.start
    path = [pos]
    trace = []
    for t in range(max_steps):
        action, state = mind.choose_action(pos, t)
        nxt, event = env.transition_from(pos, action)
        trace.append(SocialStep(t=t, pos=pos, action=action, event=event, **state))
        pos = nxt
        path.append(pos)
        if event == "goal":
            break
    return env, path, trace


def summarize(env, path, trace):
    return {
        "goal_reached": path[-1] == env.goal,
        "steps": len(trace),
        "away_from_goal_steps": sum(
            1 for row, nxt in zip(trace, path[1:]) if env.manhattan(nxt) > env.manhattan(row.pos)
        ),
        "mean_alpha": float(np.mean([r.alpha for r in trace])) if trace else 0.0,
        "mean_social_beta": float(np.mean([r.social_beta for r in trace])) if trace else 0.0,
        "mean_trust": float(np.mean([r.trust for r in trace])) if trace else 0.0,
        "mean_social_tension": float(np.mean([r.social_tension for r in trace])) if trace else 0.0,
        "mean_delusion_risk": float(np.mean([r.delusion_risk for r in trace])) if trace else 0.0,
        "final_report": trace[-1].report if trace else "none",
    }


def aggregate(condition_runs):
    keys = [
        "goal_reached",
        "steps",
        "away_from_goal_steps",
        "mean_alpha",
        "mean_social_beta",
        "mean_trust",
        "mean_social_tension",
        "mean_delusion_risk",
    ]
    out = {}
    for key in keys:
        vals = [float(run["summary"][key]) for run in condition_runs]
        out[key if key != "goal_reached" else "goal_rate"] = float(np.mean(vals))
    out["failures"] = int(sum(not run["summary"]["goal_reached"] for run in condition_runs))
    return out


def trace_to_dict(trace):
    return [
        {
            **row.__dict__,
            "pos": list(row.pos),
            "action_name": ACTION_NAMES[row.action],
            "peer_action_name": ACTION_NAMES[row.peer_choice],
            "reflex_action_name": ACTION_NAMES[row.reflex_choice],
            "internal_action_name": ACTION_NAMES[row.internal_choice],
        }
        for row in trace
    ]


def plot_summary(aggregate_results, path):
    names = list(aggregate_results)
    goal = [aggregate_results[n]["goal_rate"] for n in names]
    delusion = [aggregate_results[n]["mean_delusion_risk"] for n in names]
    beta = [aggregate_results[n]["mean_social_beta"] for n in names]
    tension = [aggregate_results[n]["mean_social_tension"] for n in names]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5 * width, goal, width, label="goal rate", color="#16a3a6")
    ax.bar(x - 0.5 * width, beta, width, label="social beta", color="#7c3aed")
    ax.bar(x + 0.5 * width, tension, width, label="social tension", color="#ff8a00")
    ax.bar(x + 1.5 * width, delusion, width, label="delusion risk", color="#e05a47")
    ax.set_ylim(0, 1.05)
    ax.set_title("Social Workspace: Grounded Peer vs Echo Chamber")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paths(example_runs, path):
    names = list(example_runs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    colors = {
        "none": "#7a7a7a",
        "grounded_peer": "#16a3a6",
        "redundant_peer": "#7c3aed",
        "noisy_peer": "#ff8a00",
        "echo_peer": "#e05a47",
        "adversarial_peer": "#111111",
    }
    for ax, name in zip(axes, names):
        env = example_runs[name]["env"]
        path_points = example_runs[name]["path"]
        draw_maze_background(env, ax)
        xs = [p[0] for p in path_points]
        ys = [p[1] for p in path_points]
        ax.plot(xs, ys, color=colors[name], lw=3, marker="o", markersize=4)
        summary = example_runs[name]["summary"]
        ax.set_title(f"{name}\ngoal={summary['goal_reached']} steps={summary['steps']}")
    fig.suptitle("Example Social Workspace Paths")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trace(trace, path):
    x = [r.t for r in trace]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(x, [r.alpha for r in trace], label="workspace alpha", color="#16a3a6")
    axes[1].plot(x, [r.social_beta for r in trace], label="social beta", color="#7c3aed")
    axes[2].plot(x, [r.trust for r in trace], label="peer trust", color="#65a30d")
    axes[3].plot(x, [r.delusion_risk for r in trace], label="delusion risk", color="#e05a47")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Grounded Peer Social Control Trace")
    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed(101)
    OUT.mkdir(exist_ok=True)
    conditions = ["none", "grounded_peer", "redundant_peer", "noisy_peer", "echo_peer", "adversarial_peer"]
    runs = {name: [] for name in conditions}
    examples = {}
    for name in conditions:
        for seed in range(30):
            env, path, trace = run_condition(name, seed=1000 + seed)
            item = {
                "summary": summarize(env, path, trace),
                "path": [list(p) for p in path],
                "trace": trace_to_dict(trace),
            }
            runs[name].append(item)
            if seed == 0:
                examples[name] = {"env": env, "path": path, "trace": trace, "summary": item["summary"]}

    aggregate_results = {name: aggregate(items) for name, items in runs.items()}
    payload = {
        "note": (
            "Tests whether a second agent acts as grounded error correction, redundant workspace, or group-delusion amplifier."
        ),
        "aggregate": aggregate_results,
        "example_runs": {name: runs[name][0] for name in conditions},
        "thesis": (
            "Social loops help when they add independent grounded information; echo loops can increase confidence without improving reality contact."
        ),
    }
    (OUT / "social_workspace_metrics.json").write_text(json.dumps(payload, indent=2))
    plot_summary(aggregate_results, OUT / "social_workspace_summary.png")
    plot_paths(examples, OUT / "social_workspace_paths.png")
    plot_trace(examples["grounded_peer"]["trace"], OUT / "social_workspace_grounded_trace.png")
    print("Social workspace lab complete")
    print(json.dumps(aggregate_results, indent=2))


if __name__ == "__main__":
    main()
