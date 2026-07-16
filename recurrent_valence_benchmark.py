#!/usr/bin/env python3
"""Matched recurrence x grounded-valence navigation benchmark.

Four independently trained actor-critic policies receive the same environments,
actions, rewards, optimizer budget, and observation dimensions:

* feedforward
* feedforward_valence
* recurrent
* recurrent_valence

The feedforward control receives an explicit eight-frame observation window.
Valence channels contain grounded reward feedback and its moving average; those
same channels are zeroed in no-valence controls. This tests recurrence and
reward-feedback components, not every module in the functional-ego stack.
"""

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from emergent_foraging_lab import FOOD_SIGHT, MOVES, SENSOR_RANGE, SIZE, line_of_sight, make_layout


OUT = Path("outputs")
METRICS = OUT / "recurrent_valence_benchmark_metrics.json"
PLOT = OUT / "recurrent_valence_benchmark.png"
CHECKPOINT_DIR = Path("checkpoints/recurrent_valence_benchmark")
MAX_STEPS = 96
CONTEXT = 8
VALENCE_DIM = 2
OBS_DIM = 4 + 2 + 4 + 1 + 4 + VALENCE_DIM
TRAIN_FAMILIES = ("open", "l_wall", "offset_barriers")
WITHHELD_FAMILIES = ("u_detour", "c_shape")
CONDITIONS = (
    ("feedforward", False, False),
    ("feedforward_valence", False, True),
    ("recurrent", True, False),
    ("recurrent_valence", True, True),
)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def rectangle(x0, y0, width, height):
    return {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}


def make_benchmark_layout(family, rng):
    if family != "c_shape":
        blocked, start, _food, _trap = make_layout(family, rng)
        return blocked, start
    blocked = rectangle(2, 2, 7, 1) | rectangle(2, 8, 7, 1) | rectangle(2, 2, 1, 7)
    turns = int(rng.integers(0, 4))

    def rotate(cell):
        x, y = cell
        for _ in range(turns):
            x, y = SIZE - 1 - y, x
        return x, y

    return {rotate(cell) for cell in blocked}, rotate((4, 5))


@dataclass
class Transition:
    reward: float
    collision: bool
    pickup: bool
    preferred_pickup: bool
    done: bool


class PreferenceForagingWorld:
    """Partially observed two-food world with optional hidden preference switch."""

    def __init__(self, family, seed, task="stable", sensor_noise=0.0):
        self.family = family
        self.seed = int(seed)
        self.task = task
        self.sensor_noise = float(sensor_noise)
        self.rng = np.random.default_rng(seed)
        self.blocked, self.start = make_benchmark_layout(family, self.rng)
        self.free = [
            (x, y)
            for x in range(1, SIZE - 1)
            for y in range(1, SIZE - 1)
            if (x, y) not in self.blocked
        ]
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.hunger = 0.15
        self.last_action = 0
        self.last_outcome = 0.0
        self.valence_ema = 0.0
        self.preference = int(self.rng.integers(0, 2))
        self.switch_step = MAX_STEPS // 2 if self.task == "reversal" else None
        self.foods = {0: [], 1: []}
        for kind in (0, 1):
            for _ in range(4):
                self.foods[kind].append(self.sample_food(exclude={self.pos}))
        self.pickups = 0
        self.preferred_pickups = 0
        self.bad_pickups = 0
        self.collisions = 0
        self.visits = {self.pos}
        self.pre_switch = [0, 0]
        self.post_switch = [0, 0]

    def sample_food(self, exclude=()):
        candidates = [cell for cell in self.free if cell not in exclude]
        return candidates[int(self.rng.integers(len(candidates)))]

    def is_blocked(self, cell):
        return not (0 <= cell[0] < SIZE and 0 <= cell[1] < SIZE) or cell in self.blocked

    def ray(self, move):
        for distance in range(1, SENSOR_RANGE + 1):
            cell = self.pos[0] + int(move[0]) * distance, self.pos[1] + int(move[1]) * distance
            if self.is_blocked(cell):
                return (distance - 1) / SENSOR_RANGE
        return 1.0

    def sight_range(self):
        if self.task == "full_observable":
            return SIZE * 2.0
        if self.task == "deprivation":
            return 3.5
        return FOOD_SIGHT

    def nearest_visible(self, kind):
        visible = []
        for food in self.foods[kind]:
            if math.dist(self.pos, food) <= self.sight_range() and line_of_sight(self.pos, food, self.blocked):
                if self.sensor_noise <= 0.0 or self.rng.random() >= min(0.45, self.sensor_noise * 1.8):
                    visible.append(food)
        return min(visible, key=lambda food: math.dist(self.pos, food)) if visible else None

    def observe(self, valence_enabled, valence_mode="normal"):
        rays = np.asarray([self.ray(move) for move in MOVES], dtype=np.float32)
        if self.sensor_noise:
            rays = np.clip(rays + self.rng.normal(0.0, self.sensor_noise, size=4), 0.0, 1.0)
        visibility = np.zeros(2, dtype=np.float32)
        deltas = np.zeros(4, dtype=np.float32)
        for kind in (0, 1):
            food = self.nearest_visible(kind)
            if food is not None:
                visibility[kind] = 1.0
                delta = (np.asarray(food, dtype=np.float32) - np.asarray(self.pos, dtype=np.float32)) / (SIZE - 1)
                if self.sensor_noise:
                    delta += self.rng.normal(0.0, self.sensor_noise * 0.30, size=2)
                deltas[kind * 2 : kind * 2 + 2] = delta
        action = np.zeros(4, dtype=np.float32)
        action[self.last_action] = 1.0
        valence = np.zeros(2, dtype=np.float32)
        if valence_enabled:
            valence[:] = (self.last_outcome, self.valence_ema)
            if valence_mode == "zero":
                valence.fill(0.0)
            elif valence_mode == "sign_flip":
                valence *= -1.0
        return np.concatenate((rays, visibility, deltas, [self.hunger], action, valence)).astype(np.float32)

    def step(self, action):
        if self.switch_step is not None and self.steps == self.switch_step:
            self.preference = 1 - self.preference
        self.last_action = int(action)
        move = MOVES[self.last_action]
        candidate = self.pos[0] + int(move[0]), self.pos[1] + int(move[1])
        collision = self.is_blocked(candidate)
        if not collision:
            self.pos = candidate
        else:
            self.collisions += 1
        self.steps += 1
        self.visits.add(self.pos)
        self.hunger = min(1.0, self.hunger + 0.012)

        pickup_kind = None
        pickup_index = None
        for kind in (0, 1):
            for index, food in enumerate(self.foods[kind]):
                if food == self.pos:
                    pickup_kind, pickup_index = kind, index
                    break
            if pickup_kind is not None:
                break

        reward = -0.004 - 0.006 * self.hunger
        preferred = False
        if collision:
            reward -= 0.035
        if pickup_kind is not None:
            preferred = pickup_kind == self.preference
            reward += 1.0 if preferred else -1.0
            self.pickups += 1
            self.preferred_pickups += int(preferred)
            self.bad_pickups += int(not preferred)
            phase = self.pre_switch if self.switch_step is None or self.steps <= self.switch_step else self.post_switch
            phase[0] += int(preferred)
            phase[1] += 1
            self.hunger = max(0.05, self.hunger - 0.34)
            occupied = {self.pos}
            for foods in self.foods.values():
                occupied.update(foods)
            self.foods[pickup_kind][pickup_index] = self.sample_food(exclude=occupied)

        done = self.steps >= MAX_STEPS
        if done and self.pickups < 3:
            reward -= 1.0
        self.last_outcome = float(np.clip(reward, -1.0, 1.0))
        self.valence_ema = float(np.clip(0.82 * self.valence_ema + 0.18 * self.last_outcome, -1.0, 1.0))
        return Transition(reward, collision, pickup_kind is not None, preferred, done)


class MatchedPolicy(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = 48 if recurrent else 67
        if recurrent:
            self.core = nn.GRUCell(OBS_DIM, self.hidden_dim)
        else:
            self.core = nn.Sequential(nn.Linear(OBS_DIM * CONTEXT, self.hidden_dim), nn.Tanh())
        self.actor = nn.Linear(self.hidden_dim, len(MOVES))
        self.critic = nn.Linear(self.hidden_dim, 1)

    def initial_state(self, batch):
        if self.recurrent:
            return torch.zeros(batch, self.hidden_dim)
        return torch.zeros(batch, CONTEXT, OBS_DIM)

    def step(self, obs, state):
        if self.recurrent:
            state = self.core(obs, state)
            features = state
        else:
            state = torch.cat((state[:, 1:], obs.unsqueeze(1)), dim=1)
            features = self.core(state.reshape(obs.shape[0], -1))
        return self.actor(features), self.critic(features).squeeze(-1), state

    def mask_state(self, state, mask):
        shape = (len(mask),) + (1,) * (state.ndim - 1)
        return state * mask.reshape(shape)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def new_training_world(rng, update):
    if update < 90:
        family = "open"
        task = "full_observable"
    elif update < 210:
        families = ("open", "l_wall")
        family = families[int(rng.integers(len(families)))]
        task = rng.choice(("stable", "full_observable"), p=(0.65, 0.35))
    else:
        family = TRAIN_FAMILIES[int(rng.integers(len(TRAIN_FAMILIES)))]
        task = rng.choice(("stable", "reversal", "full_observable"), p=(0.40, 0.45, 0.15))
    return PreferenceForagingWorld(family, int(rng.integers(1, 10_000_000)), task=task)


def train_condition(recurrent, valence_enabled, seed, updates, env_count=28, rollout=24):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    model = MatchedPolicy(recurrent)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.7e-3)
    envs = [new_training_world(rng, 0) for _ in range(env_count)]
    obs = torch.tensor(np.stack([env.observe(valence_enabled) for env in envs]))
    state = model.initial_state(env_count)
    recent_rewards = deque(maxlen=200)
    recent_preference = deque(maxlen=200)
    curve = []

    for update in range(updates):
        log_probs, values, rewards, masks, entropies = [], [], [], [], []
        episode_returns = np.zeros(env_count, dtype=np.float32)
        for _ in range(rollout):
            logits, value, state = model.step(obs, state)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            next_obs, step_rewards, alive = [], [], []
            for index, env in enumerate(envs):
                transition = env.step(int(actions[index]))
                episode_returns[index] += transition.reward
                step_rewards.append(transition.reward)
                if transition.done:
                    recent_rewards.append(float(episode_returns[index]))
                    recent_preference.append(env.preferred_pickups / max(1, env.pickups))
                    episode_returns[index] = 0.0
                    envs[index] = new_training_world(rng, update)
                    next_obs.append(envs[index].observe(valence_enabled))
                    alive.append(0.0)
                else:
                    next_obs.append(env.observe(valence_enabled))
                    alive.append(1.0)
            log_probs.append(distribution.log_prob(actions))
            values.append(value)
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
            masks.append(torch.tensor(alive, dtype=torch.float32))
            entropies.append(distribution.entropy())
            obs = torch.tensor(np.stack(next_obs))
            state = model.mask_state(state, masks[-1])

        with torch.no_grad():
            _, bootstrap, _ = model.step(obs, state)
        returns = []
        running = bootstrap
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running = reward + 0.975 * running * mask
            returns.append(running)
        returns.reverse()
        returns_t = torch.stack(returns)
        values_t = torch.stack(values)
        advantage = returns_t - values_t
        loss = (
            -(torch.stack(log_probs) * advantage.detach()).mean()
            + 0.45 * advantage.pow(2).mean()
            - 0.018 * torch.stack(entropies).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
        optimizer.step()
        state = state.detach()
        if update % 20 == 0 or update == updates - 1:
            curve.append(
                {
                    "update": update + 1,
                    "reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                    "preference_accuracy": float(np.mean(recent_preference)) if recent_preference else 0.0,
                }
            )
    return model.eval(), curve


def evaluate(
    model,
    recurrent,
    valence_enabled,
    task,
    families,
    episodes,
    seed,
    sensor_noise=0.0,
    reset_memory=False,
    valence_mode="normal",
):
    rows = []
    for episode in range(episodes):
        family = families[episode % len(families)]
        env = PreferenceForagingWorld(family, seed + episode * 37, task=task, sensor_noise=sensor_noise)
        obs = torch.tensor(env.observe(valence_enabled, valence_mode)).unsqueeze(0)
        state = model.initial_state(1)
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                logits, _, state = model.step(obs, state)
            action = int(torch.argmax(logits, dim=-1))
            transition = env.step(action)
            total_reward += transition.reward
            obs = torch.tensor(env.observe(valence_enabled, valence_mode)).unsqueeze(0)
            if reset_memory:
                state = model.initial_state(1)
            if transition.done:
                break
        rows.append(
            {
                "reward": total_reward,
                "pickups": env.pickups,
                "preferred_pickups": env.preferred_pickups,
                "pre_preferred": env.pre_switch[0],
                "pre_pickups": env.pre_switch[1],
                "post_preferred": env.post_switch[0],
                "post_pickups": env.post_switch[1],
                "collisions": env.collisions,
                "coverage": len(env.visits),
            }
        )
    total_pickups = sum(row["pickups"] for row in rows)
    pre_pickups = sum(row["pre_pickups"] for row in rows)
    post_pickups = sum(row["post_pickups"] for row in rows)
    return {
        "reward": float(np.mean([row["reward"] for row in rows])),
        "pickups": float(np.mean([row["pickups"] for row in rows])),
        "preference_accuracy": float(sum(row["preferred_pickups"] for row in rows) / max(1, total_pickups)),
        "pre_switch_accuracy": float(sum(row["pre_preferred"] for row in rows) / max(1, pre_pickups)),
        "post_switch_accuracy": float(sum(row["post_preferred"] for row in rows) / max(1, post_pickups)),
        "post_pickups": float(post_pickups / len(rows)),
        "post_sampling_rate": float(np.mean([row["post_pickups"] > 0 for row in rows])),
        "collisions": float(np.mean([row["collisions"] for row in rows])),
        "coverage": float(np.mean([row["coverage"] for row in rows])),
    }


def aggregate_seed_rows(rows):
    keys = rows[0].keys()
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def plot_results(results):
    names = [name for name, _, _ in CONDITIONS]
    tasks = ("full_observable", "partial_detour", "reward_reversal", "sensor_noise", "deprivation")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(tasks))
    width = 0.19
    colors = ("#7A848C", "#4F9D75", "#D9822B", "#1976A3")
    for index, name in enumerate(names):
        rewards = [results[name][task]["reward"]["mean"] for task in tasks]
        axes[0].bar(x + (index - 1.5) * width, rewards, width, label=name.replace("_", " "), color=colors[index])
        preferences = [results[name][task]["preference_accuracy"]["mean"] for task in tasks]
        axes[1].bar(x + (index - 1.5) * width, preferences, width, color=colors[index])
    for ax, ylabel in zip(axes, ("Mean episode reward", "Preferred-pickup fraction")):
        ax.set_xticks(x, [task.replace("_", "\n") for task in tasks])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Matched Architectural Controls")
    axes[1].set_title("Hidden Preference Tracking")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)


def run(args):
    OUT.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {name: {} for name, _, _ in CONDITIONS}
    parameter_counts = {}
    curves = {}
    task_specs = {
        "full_observable": ("full_observable", ("open",), 0.0),
        "partial_detour": ("stable", WITHHELD_FAMILIES, 0.0),
        "reward_reversal": ("reversal", WITHHELD_FAMILIES, 0.0),
        "sensor_noise": ("reversal", WITHHELD_FAMILIES, 0.18),
        "deprivation": ("deprivation", WITHHELD_FAMILIES, 0.0),
    }
    best_models = {}
    for name, recurrent, valence_enabled in CONDITIONS:
        curves[name] = []
        models = []
        for seed_index in range(args.seeds):
            seed = args.seed + seed_index * 1009
            print(f"training {name} seed={seed}", flush=True)
            model, curve = train_condition(recurrent, valence_enabled, seed, args.updates, args.env_count, args.rollout)
            parameter_counts[name] = parameter_count(model)
            curves[name].append(curve)
            models.append(model)
            for task_name, (task, families, noise) in task_specs.items():
                row = evaluate(
                    model,
                    recurrent,
                    valence_enabled,
                    task,
                    families,
                    args.eval_episodes,
                    70_000 + seed_index * 10_000,
                    sensor_noise=noise,
                )
                raw[name].setdefault(task_name, []).append(row)
        best_models[name] = models[0]
        torch.save(
            {
                "state_dict": models[0].state_dict(),
                "recurrent": recurrent,
                "valence_enabled": valence_enabled,
                "obs_dim": OBS_DIM,
                "context": CONTEXT,
            },
            CHECKPOINT_DIR / f"{name}.pt",
        )

    results = {
        name: {task: aggregate_seed_rows(rows) for task, rows in task_rows.items()}
        for name, task_rows in raw.items()
    }
    recurrent_valence = best_models["recurrent_valence"]
    ablations = {
        "normal": evaluate(recurrent_valence, True, True, "reversal", WITHHELD_FAMILIES, args.eval_episodes, 190_000),
        "hidden_reset": evaluate(
            recurrent_valence, True, True, "reversal", WITHHELD_FAMILIES, args.eval_episodes, 190_000, reset_memory=True
        ),
        "valence_zero": evaluate(
            recurrent_valence, True, True, "reversal", WITHHELD_FAMILIES, args.eval_episodes, 190_000, valence_mode="zero"
        ),
        "valence_sign_flip": evaluate(
            recurrent_valence,
            True,
            True,
            "reversal",
            WITHHELD_FAMILIES,
            args.eval_episodes,
            190_000,
            valence_mode="sign_flip",
        ),
    }
    payload = {
        "design": {
            "conditions": [name for name, _, _ in CONDITIONS],
            "train_families": TRAIN_FAMILIES,
            "withheld_families": WITHHELD_FAMILIES,
            "seeds": args.seeds,
            "updates": args.updates,
            "environment_steps_per_condition_seed": args.updates * args.env_count * args.rollout,
            "feedforward_context_frames": CONTEXT,
            "parameter_counts": parameter_counts,
            "reward_design": {
                "preferred_pickup": 1.0,
                "nonpreferred_pickup": -1.0,
                "minimum_pickups": 3,
                "under_minimum_terminal_penalty": -1.0,
            },
            "claim_boundary": "Tests recurrence and grounded reward-feedback components in one navigation POMDP; it does not prove recurrent systems are universally superior or evaluate every functional-ego module.",
        },
        "results": results,
        "recurrent_valence_ablations": ablations,
        "training_curves": curves,
    }
    METRICS.write_text(json.dumps(payload, indent=2))
    plot_results(results)
    print("\nMatched recurrence x valence benchmark complete")
    for name, _, _ in CONDITIONS:
        reversal = results[name]["reward_reversal"]
        detour = results[name]["partial_detour"]
        print(
            f"{name:22s} params={parameter_counts[name]:5d} "
            f"detour_reward={detour['reward']['mean']:+.3f} "
            f"reversal_reward={reversal['reward']['mean']:+.3f} "
            f"post_switch={reversal['post_switch_accuracy']['mean']:.3f}"
        )
    print(f"Metrics: {METRICS}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17031)
    parser.add_argument("--updates", type=int, default=420)
    parser.add_argument("--env-count", type=int, default=28)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.updates = 35
        args.env_count = 12
        args.rollout = 16
        args.eval_episodes = 12
        args.seeds = 1
    return args


if __name__ == "__main__":
    run(parse_args())
