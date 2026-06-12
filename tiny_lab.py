#!/usr/bin/env python3
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class StepResult:
    obs: torch.Tensor
    reward: float
    done: bool


class LineWorld:
    """A tiny 1D world with a goal and a hazard.

    Observation = one-hot position + one-hot goal + one-hot hazard + last reward.
    The reward is our first functional valence signal: good/bad consequences
    that shape future action and hidden state.
    """

    def __init__(self, size=9, max_steps=24):
        self.size = size
        self.max_steps = max_steps
        self.reset()

    @property
    def obs_dim(self):
        return self.size * 3 + 1

    def reset(self):
        self.t = 0
        self.pos = self.size // 2
        self.goal = random.choice([0, self.size - 1])
        self.hazard = random.choice([i for i in range(self.size) if i not in {self.pos, self.goal}])
        self.last_reward = 0.0
        return self.observe()

    def observe(self):
        v = torch.zeros(self.obs_dim)
        v[self.pos] = 1.0
        v[self.size + self.goal] = 1.0
        v[self.size * 2 + self.hazard] = 1.0
        v[-1] = self.last_reward
        return v

    def step(self, action):
        move = [-1, 0, 1][int(action)]
        self.pos = max(0, min(self.size - 1, self.pos + move))
        self.t += 1

        reward = -0.02
        done = False
        if self.pos == self.goal:
            reward = 1.0
            done = True
        elif self.pos == self.hazard:
            reward = -1.0
            done = True
        elif self.t >= self.max_steps:
            done = True

        self.last_reward = reward
        return StepResult(self.observe(), reward, done)


class TinyRecurrentAgent(nn.Module):
    """A tiny agent with recurrence, attention/control, world model, and self model."""

    def __init__(self, obs_dim, hidden_dim=24, actions=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actions = actions
        self.attn = nn.Linear(obs_dim + hidden_dim, obs_dim)
        self.rnn = nn.GRUCell(obs_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, actions)
        self.valence = nn.Linear(hidden_dim, 1)
        self.world_model = nn.Linear(hidden_dim + actions, obs_dim)
        self.self_model = nn.Linear(hidden_dim, actions)

    def initial_state(self):
        return torch.zeros(self.hidden_dim)

    def forward_step(self, obs, h):
        gate = torch.sigmoid(self.attn(torch.cat([obs, h])))
        attended_obs = obs * gate
        h2 = torch.tanh(self.rnn(attended_obs, h))
        logits = self.policy(h2)
        value = self.valence(h2).squeeze(-1)
        self_logits = self.self_model(h2)
        return h2, logits, value, self_logits, gate

    def predict_next_obs(self, h, action):
        onehot = F.one_hot(torch.tensor(int(action)), self.actions).float()
        return self.world_model(torch.cat([h, onehot]))


def run_episode(agent, env, train=True):
    obs = env.reset()
    h = agent.initial_state()
    logps, values, rewards = [], [], []
    world_losses, self_losses = [], []
    hidden, actions, gates = [], [], []

    last_logits = None
    for _ in range(env.max_steps):
        h, logits, value, self_logits, gate = agent.forward_step(obs, h)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if train else torch.argmax(logits)
        pred_next = agent.predict_next_obs(h, action)

        result = env.step(action)
        world_losses.append(F.mse_loss(pred_next, result.obs))
        if last_logits is not None:
            self_losses.append(F.cross_entropy(last_logits.unsqueeze(0), action.view(1)))
        last_logits = self_logits

        logps.append(dist.log_prob(action))
        values.append(value)
        rewards.append(torch.tensor(result.reward))
        hidden.append(h.detach().numpy())
        actions.append(int(action))
        gates.append(gate.detach().numpy())
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
    policy_loss = -(logps_t * advantage).mean()
    value_loss = F.mse_loss(values_t, returns)
    world_loss = torch.stack(world_losses).mean()
    self_loss = torch.stack(self_losses).mean() if self_losses else torch.tensor(0.0)
    loss = policy_loss + 0.5 * value_loss + 0.25 * world_loss + 0.05 * self_loss

    return {
        "loss": loss,
        "reward": float(torch.stack(rewards).sum()),
        "hidden": np.array(hidden),
        "actions": actions,
        "gates": np.array(gates),
    }


def train(agent, episodes=900):
    env = LineWorld()
    opt = torch.optim.Adam(agent.parameters(), lr=0.006)
    rewards = []
    for ep in range(episodes):
        result = run_episode(agent, env, train=True)
        opt.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        rewards.append(result["reward"])
    return rewards


def pca3(x):
    x = np.asarray(x)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:3].T
    y = x @ basis
    if y.shape[1] < 3:
        y = np.pad(y, ((0, 0), (0, 3 - y.shape[1])))
    return y


def collect_trajectory(agent, ablate_unit=None, steps=120):
    env = LineWorld(max_steps=steps)
    obs = env.reset()
    h = agent.initial_state()
    hidden = []
    rewards = []
    predicted_valence = []
    positions = []
    for _ in range(steps):
        if ablate_unit is not None:
            h = h.clone()
            h[ablate_unit] = 0.0
        h, logits, value, _, _ = agent.forward_step(obs, h)
        if ablate_unit is not None:
            h = h.clone()
            h[ablate_unit] = 0.0
        action = torch.argmax(logits)
        result = env.step(action)
        hidden.append(h.detach().numpy())
        rewards.append(result.reward)
        predicted_valence.append(float(value.detach()))
        positions.append(env.pos)
        obs = result.obs
        if result.done:
            obs = env.reset()
    return np.array(hidden), rewards, predicted_valence, positions


def causal_influence(agent, trajectory):
    """Ablate each source hidden unit and measure next-hidden effects."""
    env = LineWorld()
    obs = env.reset()
    h = agent.initial_state()
    matrix = np.zeros((agent.hidden_dim, agent.hidden_dim), dtype=np.float32)
    count = 0

    for _ in range(min(80, len(trajectory))):
        h, logits, _, _, _ = agent.forward_step(obs, h)
        action = torch.argmax(logits)
        result = env.step(action)
        base_next, _, _, _, _ = agent.forward_step(result.obs, h)
        for src in range(agent.hidden_dim):
            damaged = h.clone()
            damaged[src] = 0.0
            alt_next, _, _, _, _ = agent.forward_step(result.obs, damaged)
            matrix[src] += torch.abs(base_next - alt_next).detach().numpy()
        obs = result.obs if not result.done else env.reset()
        count += 1
    return matrix / max(count, 1)


def integration_metrics(hidden, influence):
    h = np.asarray(hidden)
    corr = np.corrcoef(h.T)
    corr = np.nan_to_num(corr)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    temporal = np.mean(np.sum(h[:-1] * h[1:], axis=1) / (np.linalg.norm(h[:-1], axis=1) * np.linalg.norm(h[1:], axis=1) + 1e-8))
    threshold = float(np.percentile(influence, 75))
    causal_density = float((influence > threshold).mean())
    mean_cross_influence = float(influence[~np.eye(influence.shape[0], dtype=bool)].mean())
    return {
        "mean_abs_feature_correlation": float(np.mean(np.abs(off_diag))),
        "temporal_state_similarity": float(temporal),
        "causal_density_top_quartile": causal_density,
        "mean_cross_unit_causal_influence": mean_cross_influence,
        "note": "These are integration proxies, not formal IIT Phi.",
    }


def plot_trajectory(hidden, damaged_hidden, path):
    y = pca3(np.vstack([hidden, damaged_hidden]))
    a = y[: len(hidden)]
    b = y[len(hidden) :]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(a[:, 0], a[:, 1], a[:, 2], color="#16a3a6", lw=2, label="normal recurrent trajectory")
    ax.plot(b[:, 0], b[:, 1], b[:, 2], color="#ff8a00", lw=2, label="with one hidden unit ablated")
    ax.scatter(a[0, 0], a[0, 1], a[0, 2], color="white", edgecolor="black", s=60, label="start")
    ax.set_title("Hidden-State Trajectory Projected to 3D")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(hidden, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(hidden.T, aspect="auto", cmap="magma")
    ax.set_title("Hidden Unit Activations Over Time")
    ax.set_xlabel("time step")
    ax.set_ylabel("hidden unit")
    fig.colorbar(im, ax=ax, label="activation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_influence(influence, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(influence, cmap="viridis")
    ax.set_title("Causal Influence: Ablate Source Unit -> Change Target Unit")
    ax.set_xlabel("target hidden unit changed")
    ax.set_ylabel("source hidden unit ablated")
    fig.colorbar(im, ax=ax, label="mean absolute change")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def moving_average(values, window=25):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_valence(training_rewards, trajectory_rewards, predicted_valence, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(moving_average(training_rewards), color="#16a3a6", lw=2)
    axes[0].axhline(0, color="black", lw=1, alpha=0.4)
    axes[0].set_title("Valence-Shaped Learning: Reward Moving Average")
    axes[0].set_ylabel("episode reward")

    x = np.arange(len(trajectory_rewards))
    axes[1].plot(x, predicted_valence, color="#ff8a00", lw=2, label="agent valence/value prediction")
    axes[1].bar(x, trajectory_rewards, color="#16a3a6", alpha=0.35, label="actual reward")
    axes[1].axhline(0, color="black", lw=1, alpha=0.4)
    axes[1].set_title("Functional Valence Trace During One Recurrent Trajectory")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("good/bad signal")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    set_seed()
    OUT.mkdir(exist_ok=True)
    agent = TinyRecurrentAgent(obs_dim=LineWorld().obs_dim, hidden_dim=24)
    rewards = train(agent)
    hidden, trajectory_rewards, predicted_valence, _ = collect_trajectory(agent)
    influence = causal_influence(agent, hidden)
    ablate_unit = int(np.argmax(influence.mean(axis=1)))
    damaged_hidden, _, _, _ = collect_trajectory(agent, ablate_unit=ablate_unit)
    metrics = integration_metrics(hidden, influence)
    metrics["most_influential_unit_ablated_in_plot"] = ablate_unit
    metrics["mean_training_reward_last_100"] = float(np.mean(rewards[-100:]))
    metrics["mean_predicted_valence_on_trajectory"] = float(np.mean(predicted_valence))
    metrics["mean_actual_reward_on_trajectory"] = float(np.mean(trajectory_rewards))

    plot_trajectory(hidden, damaged_hidden, OUT / "hidden_trajectory_3d.png")
    plot_trajectory(hidden, damaged_hidden, OUT / "hidden_trajectory_ablation_3d.png")
    plot_heatmap(hidden, OUT / "activation_heatmap.png")
    plot_influence(influence, OUT / "causal_influence_graph.png")
    plot_valence(rewards, trajectory_rewards, predicted_valence, OUT / "valence_trace.png")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Tiny Consciousness Lab complete")
    print(f"outputs: {OUT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
