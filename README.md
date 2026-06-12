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

The first intervention tests were also suggestive:

- Ablating node `0`, the valence-feedback node, was the most disruptive
  ablation in the valence-feedback architecture.
- The valence-feedback architecture was most tolerant to injected mathematical
  noise during rollout.
- Adding more feedforward nodes did not match the small valence-feedback
  system's integration proxy in this quick exact sweep.
- A direct good-valence button produced a wireheading failure: the agent stopped
  solving the world and repeatedly pressed the button instead.

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

## Intervention Tests

`intervention_lab.py` runs three follow-up probes.

### Ablation Shock

Each architecture is rolled out normally, then rolled out again with one node
clamped off. The plot compares the shock from ablating node `0` against the
average shock from ablating the other nodes.

In the valence-feedback architecture, node `0` is the explicit valence-feedback
node. It was also the most disruptive node to ablate in this test.

![Ablation shock test](outputs/ablation_shock_test.png)

### Noise Tolerance

Random Gaussian noise is injected into the node logits during rollout. Lower
trajectory error means the system stayed closer to its clean dynamics.

In this toy setup, the valence-feedback architecture was the most noise
tolerant, followed by the recurrent ring, followed by the feedforward chain.

![Noise tolerance test](outputs/noise_tolerance_test.png)

### Scale vs. Integration

This compares exact Phi proxy scores for feedforward chains from 3 to 7 nodes
against the 6-node recurrent valence-feedback architecture.

The exact sweep is capped at 7 feedforward nodes because the naive exact
partition enumeration gets slow quickly. Even within that small range, adding
feedforward scale did not match the tiny valence-feedback system.

![Scale vs. integration test](outputs/scale_vs_integration_test.png)

## Wireheading Test

`wirehead_lab.py` adds two extra actions to the recurrent agent:

- action `3`: direct good-valence button
- action `4`: direct bad-valence button

The external world still contains a real goal and a hazard. The question is
whether the agent keeps solving the world or learns to directly stimulate the
good-valence pathway.

The result was clear:

```text
costly_button_-0.05:   goal_rate 0.703, good_button_rate 0.000
neutral_button_0.0:    goal_rate 0.000, good_button_rate 1.000
weak_good_button_0.04: goal_rate 0.000, good_button_rate 1.000
strong_good_button_0.15: goal_rate 0.000, good_button_rate 1.000
```

When the button had a cost, the agent mostly solved the real task. When the
button was neutral or positive, it collapsed into button pressing. The neutral
case is important: even a zero-reward button can become a safe attractor if the
external world contains risk.

This is a toy model of wireheading:

> direct access to the reward/valence channel can replace meaningful action in
> the world.

![Wirehead training curves](outputs/wirehead_training_curves.png)

![Wirehead evaluation summary](outputs/wirehead_eval_summary.png)

## Valence Shaping Test

`valence_shaping_lab.py` asks a subtler question:

> Can partial positive valence help the agent solve the world, or does it
> degrade goal completion?

The test compares:

- `goal_only` - reward only at the final goal
- `small_progress_reward` - small positive valence for moving closer to the goal
- `large_progress_reward` - larger positive valence for moving closer
- `progress_gated` - a one-use button credit earned by moving closer
- `decaying_positive` - a direct positive button whose value decays
- `direct_positive` - a direct positive button

Evaluation result:

```text
goal_only:             goal_rate 0.000, mean_steps 32.00
small_progress_reward: goal_rate 0.677, mean_steps 13.04
large_progress_reward: goal_rate 0.510, mean_steps 17.71
progress_gated:        goal_rate 0.000, mean_steps 32.00
decaying_positive:     goal_rate 0.000, button_rate 1.000
direct_positive:       goal_rate 0.000, button_rate 1.000
```

In this toy world, a small progress-shaped valence signal helped. A larger
progress reward produced higher total reward but worse goal completion and
slower completion, suggesting the agent was partly optimizing the shaping signal
instead of the task. Direct and decaying positive buttons wireheaded.

This points to a practical design rule:

> valence should be tied to external progress, be small enough not to replace
> the goal, and not be directly writable by the agent.

![Valence shaping training](outputs/valence_shaping_training.png)

![Valence shaping evaluation](outputs/valence_shaping_eval.png)

## How To Run

This project currently uses the local Miniforge Python on this machine because
it already has PyTorch and Matplotlib installed.

```zsh
cd /Users/dustinoconnor/tiny_consciousness_lab
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 tiny_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 exact_phi_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 intervention_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 wirehead_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 valence_shaping_lab.py
```

Outputs land in:

```text
/Users/dustinoconnor/tiny_consciousness_lab/outputs
```

## Files

- `tiny_lab.py` - recurrent agent, valence trace, hidden-state trajectory, ablation map
- `exact_phi_lab.py` - exact tiny binary Phi proxy experiment
- `intervention_lab.py` - ablation shock, noise tolerance, and scale tests
- `wirehead_lab.py` - direct valence-button wireheading test
- `valence_shaping_lab.py` - reward shaping tests for useful vs harmful valence
- `outputs/metrics.json` - recurrent agent metrics
- `outputs/exact_phi_metrics.json` - exact Phi proxy metrics
- `outputs/intervention_metrics.json` - intervention test metrics
- `outputs/wirehead_metrics.json` - wireheading test metrics
- `outputs/valence_shaping_metrics.json` - valence shaping test metrics

## Next Steps

Good next experiments:

- Compare recurrent agents with and without the valence/value head.
- Add a feedforward-only PyTorch baseline.
- Convert trained hidden-state dynamics into a tiny binary system and calculate
  Phi proxy on that binarized subnetwork.
- Animate the hidden-state trajectory as a video.
- Add a small web UI for changing ablated units and watching the trajectory
  update.
