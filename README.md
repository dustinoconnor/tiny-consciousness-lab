# Tiny Consciousness Lab

Toy experiments for recurrence, functional valence, simple world modeling,
mechanistic interpretability, and a small exact Phi-like integration proxy.

This project is not a consciousness detector. It is a small experimental bench
for making questions about recurrent systems visible:

- Does recurrence produce different hidden-state geometry?
- Does a functional valence signal shape action and internal state?
- Which hidden units causally influence other hidden units over time?
- Does adding a valence-feedback node increase a tiny exact integration proxy?

The short answer from the first run is interesting:

```text
feedforward_chain:                 0.0971
recurrent_ring:                    0.0937
recurrent_with_valence_feedback:   0.1136
```

In this toy setup, recurrence alone did not automatically produce the highest
integration proxy. The recurrent system with a valence-feedback node scored
highest.

## Why This Is Not Official IIT Phi

Integrated Information Theory uses a specific mathematical framework for
measuring irreducible cause-effect structure. Official IIT Phi depends on
details such as system mechanisms, cause/effect repertoires, partitions, and
the formal version of IIT being used. Those calculations become extremely
expensive as the number of nodes grows.

This project uses a deliberately simpler exact proxy:

1. Build a tiny binary system.
2. Enumerate every possible binary state.
3. Compute the full next-state probability distribution.
4. Split the system across every bipartition.
5. Compare the full system to each partitioned approximation.
6. Use the minimum KL divergence as a small Phi-like score for that state.
7. Average across all states.

So the calculation is exact for this toy definition, but it is not official IIT
Phi. A better name is:

```text
exact tiny Phi proxy
```

That distinction matters. The result should be read as:

> Adding valence feedback increased irreducible transition structure in this
> toy binary system under this exact proxy.

Not:

> This proves valence causes consciousness.

## Recurrent Agent

`tiny_lab.py` builds a small recurrent PyTorch agent in a 1D world. The agent
has:

- recurrence / memory through a GRU cell
- attention/control over the observation
- a policy head for action
- a valence/value head predicting good/bad outcome
- a world-model head predicting the next observation
- a self-model head predicting its next action

The world is intentionally tiny: the agent moves left, right, or stays still.
One location is rewarding, one location is harmful. Reward is the first
functional valence signal.

### Hidden-State Trajectory

The recurrent hidden state is high-dimensional, so the script projects it into
3D with PCA. The orange path shows what happens when the most influential hidden
unit is ablated.

![Hidden-state trajectory](outputs/hidden_trajectory_3d.png)

### Functional Valence

The top plot shows reward learning over training. The bottom plot compares the
agent's internal valence/value prediction against actual reward during one
trajectory.

![Valence trace](outputs/valence_trace.png)

### Activation Heatmap

Each row is a hidden unit. Each column is a time step.

![Activation heatmap](outputs/activation_heatmap.png)

### Causal Influence Map

Each source hidden unit is ablated. The heatmap shows how much that damage
changes each target hidden unit on the next step.

![Causal influence graph](outputs/causal_influence_graph.png)

## Exact Tiny Phi Proxy

`exact_phi_lab.py` compares three 6-node binary systems:

1. A feedforward chain.
2. A recurrent ring.
3. A recurrent ring with a valence-feedback node.

### Result

![Exact Phi proxy bar graph](outputs/exact_phi_bar_graph.png)

### State-by-State Scores

The Phi proxy is not the same for every state. Some states are more
partitionable than others.

![Exact Phi proxy by state](outputs/exact_phi_by_state.png)

### Architectures

Feedforward chain:

![Feedforward chain](outputs/feedforward_chain_network.png)

Recurrent ring:

![Recurrent ring](outputs/recurrent_ring_network.png)

Recurrent with valence feedback:

![Recurrent with valence feedback](outputs/recurrent_with_valence_feedback_network.png)

## How To Run

This project currently uses the local Miniforge Python on this machine because
it already has PyTorch and Matplotlib installed.

```zsh
cd /Users/dustinoconnor/tiny_consciousness_lab
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 tiny_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 exact_phi_lab.py
```

Outputs land in:

```text
/Users/dustinoconnor/tiny_consciousness_lab/outputs
```

## Files

- `tiny_lab.py` - recurrent agent, valence trace, hidden-state trajectory, ablation map
- `exact_phi_lab.py` - exact tiny binary Phi proxy experiment
- `outputs/metrics.json` - recurrent agent metrics
- `outputs/exact_phi_metrics.json` - exact Phi proxy metrics

## Next Steps

Good next experiments:

- Compare recurrent agents with and without the valence/value head.
- Add a feedforward-only PyTorch baseline.
- Convert trained hidden-state dynamics into a tiny binary system and calculate
  Phi proxy on that binarized subnetwork.
- Animate the hidden-state trajectory as a video.
- Add a small web UI for changing ablated units and watching the trajectory
  update.

