# Tiny Consciousness Lab

Toy experiments for recurrence, functional valence, simple world modeling,
mechanistic interpretability, and a small exact Phi-like integration proxy.

Keywords: AI consciousness, machine consciousness, recurrent neural networks,
functional valence, wireheading, Integrated Information Theory, IIT, Phi proxy,
mechanistic interpretability, reinforcement learning, world models, AI
alignment, attention, PyTorch.

This project is not a consciousness detector. It is a small experimental bench
for making questions about recurrent systems visible:

- Does recurrence produce different hidden-state geometry?
- Does a functional valence signal shape action and internal state?
- Which hidden units causally influence other hidden units over time?
- Does adding a valence-feedback node increase a tiny exact integration proxy?

The working thesis emerging from these toy runs:

> Capacity without grounded valence is unstable.  
> Valence without boundaries is exploitable.  
> Imagination without reality-checking is delusional.  
> Attention should be rewarded for staying grounded.  
> Useful intelligence requires co-tuned cognition, reward, attention, and world
> modeling.

## Substrate-Independence Angle

This project does not prove machine consciousness, and it should not be read as
a consciousness detector. The stronger, defensible claim is narrower:

> These experiments support substrate independence for several functional
> prerequisites often discussed in consciousness research, while stopping short
> of proving subjective experience.

The toy systems are not biological. They have no neurons, cells, hormones, or
embodied metabolism. Yet changing the informational architecture changes their
behavior in recognizable ways:

- Adding valence feedback increases irreducible transition structure under the
  exact tiny Phi proxy.
- Ablating the valence-feedback node is especially disruptive in that toy
  architecture.
- Direct access to positive valence causes wireheading.
- Grounded progress-valence improves goal completion.
- Ungrounded imagination degrades behavior.
- Accuracy-rewarded and gated imagination recover much of that loss.
- A pretrained world model enables lookahead in a detour maze where myopic
  progress-valence gets stuck.
- A valence-shaped attention filter improves task focus by rewarding
  prediction-aligned imagination and suppressing distractor fixation.
- When the environment changes, an adaptive attention-valence filter can use
  prediction error as surprise to rebuild its inner model.

The philosophical inference is not that silicon is conscious. It is that
integration, valence, confidence gating, world modeling, and lookahead can be
implemented as substrate-independent information-processing patterns.

In evolutionary language, the detour maze captures a functional pressure:

> Reflex works in simple worlds. Once the world contains traps, detours, and
> delayed reward, useful action needs internal simulation.

That is the bridge this repo is trying to make visible: not consciousness
itself, but a small ladder of functional prerequisites that can exist outside
carbon biology.

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

## Valence Scaling Without Phi

`valence_scaling_lab.py` asks what happens when hidden size grows but exact Phi
is no longer calculated. It compares hidden sizes `8`, `16`, `32`, `64`, and
`96` using behavioral metrics:

- goal completion
- task efficiency
- good-button/wireheading rate
- mean reward

This run compared:

- `goal_only`
- `small_progress_reward`
- `large_progress_reward`
- `direct_positive`

The result was mixed but useful. The weak progress signal did not reliably help
in this shorter scaling sweep. The larger progress signal produced the strongest
goal completion and task efficiency across most sizes. Direct positive valence
wireheaded at every size.

The practical lesson is not simply "less valence is always better." It is more
like:

> valence needs to be externally grounded and strong enough to guide learning,
> but not directly writable and not so strong that it replaces the task.

![Valence scaling behavior](outputs/valence_scaling_behavior.png)

## Pre-Action Imagination Test

`imagination_lab.py` adds a fast intuition-like loop before action:

1. Use the world model to imagine the next observation for each possible action.
2. Score each imagined future for expected progress and hazard risk.
3. Add that score as an action prior before the policy acts.

This was meant to test whether "intuition" improves learning by letting the
agent simulate before acting.

The first result was negative, then grounding improved it:

```text
baseline_goal_only:                    goal_rate 0.000, mean_steps 32.00
naive_imagination_goal_only:           goal_rate 0.000, mean_steps 32.00
baseline_progress_valence:             goal_rate 0.635, mean_steps 14.21
naive_imagination_progress_valence:    goal_rate 0.240, mean_steps 25.29
accuracy_rewarded_imagination:         goal_rate 0.510, mean_steps 17.71
gated_accuracy_rewarded_imagination:   goal_rate 0.531, mean_steps 17.13
pretrained_gated_imagination:          goal_rate 0.563, mean_steps 16.25
```

Naive imagination did not help. It hurt the progress-valence agent, likely
because its early world model was not reliable enough. The agent acted on bad
intuition.

Rewarding imagination for matching reality helped recover much of the lost
performance. Adding a confidence gate helped a little more. But neither grounded
imagination variant beat the plain progress-valence baseline in this run.

Pretraining the world model before reinforcement learning improved imagination
accuracy and goal completion compared with non-pretrained gated imagination, but
it still did not beat the plain progress-valence baseline. This suggests that
better world modeling helps, but the current imagination prior is still too
crude or too costly.

This suggests another useful boundary:

> Imagination should be rewarded for matching reality before it is trusted to
> guide action.

And the negative version:

> Imagination without an accurate world model is delusional. Ungrounded
> intuition corrupts the motivational signal.

![Imagination training](outputs/imagination_training.png)

![Imagination evaluation](outputs/imagination_eval.png)

## 2D Detour Maze Imagination Test

`maze_imagination_lab.py` moves the agent from a 1D line into a small 2D maze
with walls. The second version adds a real detour: the agent must temporarily
move away from the goal to get around a wall.

The clean counter-example:

```text
myopic_progress_reflex:       goal_reached false, wall_hits 30
pretrained_world_lookahead:   goal_reached true, steps 16, away_from_goal_steps 4
```

The myopic reflex only accepts immediately positive Manhattan progress. It walks
to the wall and keeps pushing into it because every useful detour step feels
locally worse. The pretrained world-model lookahead accepts four temporarily bad
steps, rounds the wall, and reaches the goal.

The learned recurrent agents are still messier than the hand-isolated
counter-example. They can sometimes learn the route through reinforcement, and
the pretrained neural imagination policy did not cleanly dominate in this tiny
setup. The important result here is narrower but useful: once a world contains a
true local minimum, pure progress-valence is not enough; some form of trusted
lookahead or world modeling is required.

![Maze layout](outputs/maze_layout.png)

![Maze detour counterexample](outputs/maze_detour_counterexample.png)

![Maze imagination training](outputs/maze_imagination_training.png)

![Maze imagination evaluation](outputs/maze_imagination_eval.png)

## Imagination Phi-Proxy Test

`imagination_phi_lab.py` asks whether imagination, self-modeling, and imagined
valence increase irreducible causal structure in the exact tiny Phi proxy. The
latest version builds eight-node binary circuits with the same node vocabulary:

```text
sense, memory, valence, imagination, confidence, self, imagined_valence, action
```

Result:

```text
reflex_only:                         0.0000
reflex_valence:                      0.0000
valence_memory:                      0.0000
valence_imagination:                 0.0289
gated_imagination:                   0.0276
recurrent_gated_imagination:         0.0233
self_model_loop:                     0.0281
counterfactual_self_imagination:     0.0253
counterfactual_imagined_valence:     0.0244
recursive_inner_world:               0.0214
attention_reconciled_inner_world:    0.0213
```

This expanded run changed the node vocabulary, so the absolute numbers should
not be compared directly to the earlier six-node run. Within the eight-node
vocabulary, the simple reflex/valence variants are easy for the proxy to
partition because several nodes are inactive. The architectures that actually
activate imagination and self-modeling become measurably less separable.

The strongest score in this wiring is still `valence_imagination`. Adding a
self-model loop stays close, but counterfactual self-imagination, imagined
valence, a recursive inner-world loop, and an attention-reconciled inner-world
loop do not automatically increase the proxy. That matters: a richer mind-like
vocabulary is not enough by itself. The loops have to be routed in a way that
makes the whole system harder to split.

The attention-reconciled circuit tested a stricter routing idea:

- confidence receives direct input from real valence, imagined valence, self,
  and sense
- action is weakened on raw sense and forced to depend more on confidence, real
  valence, imagined valence, and self-state
- imagination and self-state feed back through confidence before action

That did not raise the Phi proxy. It landed essentially tied with the recursive
inner-world circuit and below the simpler valence-imagination circuit. But the
node ablation map changed in a useful way. In the attention-reconciled circuit,
removing `confidence`, `action`, `imagination`, `valence`, or `self` causes much
larger causal distribution damage than in the looser self-model loop. In other
words, the gating made those nodes matter more to the system's transition
dynamics, even though this exact partition proxy still found the circuit easier
to split than the simpler imagination-valence wiring.

So the careful interpretation is:

> In this toy binary circuit, imagination and self-modeling can create
> irreducible causal structure, but counterfactual and imagined-valence loops do
> not automatically raise integration. More inner-world machinery is not
> automatically more unified mind-like structure. Tighter attention/confidence
> routing can make inner-world nodes more causally load-bearing without
> necessarily increasing this Phi proxy.

![Imagination Phi proxy bar graph](outputs/imagination_phi_bar_graph.png)

![Imagination Phi proxy by state](outputs/imagination_phi_by_state.png)

![Valence imagination network](outputs/valence_imagination_network.png)

![Self model loop network](outputs/self_model_loop_network.png)

![Counterfactual imagined valence network](outputs/counterfactual_imagined_valence_network.png)

![Attention reconciled inner world network](outputs/attention_reconciled_inner_world_network.png)

![Imagination Phi ablation damage](outputs/imagination_phi_ablation_damage.png)

## Delusional Integration Sweep

`delusional_integration_lab.py` tests a warning from the self/imagination
experiments: if internal self-model and imagination loops become too strong
relative to external sensory grounding, the system may start tracking its own
inner state more than the outside world.

The sweep increases internal self/imagination recurrence and measures:

- exact tiny Phi proxy
- external grounding ratio from the weight graph
- action sensitivity to the external `sense` node
- action sensitivity to internal `memory`, `valence`, `imagination`, and `self`
  nodes
- a simple delusion-risk index: internal action influence divided by external
  action influence

Result:

```text
internal_scale  phi_proxy  grounding  delusion_risk
0.0             0.0465     0.3900     0.5632
0.6             0.0363     0.2986     0.6655
1.2             0.0315     0.2419     0.7685
1.8             0.0317     0.2033     0.8705
2.4             0.0307     0.1754     0.9688
```

In this first wiring, internal dominance did **not** make Phi climb. Instead,
external grounding fell steadily while internal influence over action rose
toward parity with sensory influence.

That refines the warning:

> Delusion risk is not just "high integration." It is a mismatch between
> integration and grounding. A system can become more internally driven without
> becoming more irreducibly integrated under this proxy.

![Delusional integration sweep](outputs/delusional_integration_sweep.png)

## Attention-Valence Filter Test

`attention_valence_lab.py` tests the "Ritalin circuit" idea as a small
attention-control toy. This is not a biological ADHD model. The nickname is just
useful shorthand for the engineering question:

> Can valence keep attention locked onto task-relevant sensory reality while
> starving imagination loops that drift away from the world?

The toy world has three competing streams:

- a task-relevant sensory target
- a high-novelty distractor
- an internal imagination stream that predicts the next sensory target but can
  drift if it mostly listens to itself

The attention-valence filter uses a simple version of the idea:

```text
attention = softmax(query . key / sqrt(d)) * valence_signal
```

Here, the valence signal is high when imagination predicts the next sensory
state and low when imagination detaches from the sensory stream.

Result:

```text
condition                     accuracy  task_attention  distractor  delusion
ungated_attention             0.789     0.653           0.306       0.041
self_amplified_imagination    0.872     0.599           0.158       0.242
attention_valence_filter      0.983     0.922           0.076       0.002
overconstrained_filter        0.978     0.969           0.028       0.003
```

The self-amplified imagination condition looked more confident internally, but
its grounding fell and its delusion index rose. The attention-valence filter
kept imagination useful by tying its influence to prediction accuracy. In this
toy, that increased task accuracy, increased task attention, and sharply reduced
distractor fixation.

This adds a fifth line to the working thesis:

> Attention should be rewarded for staying grounded, not merely for becoming
> internally confident.

![Attention valence summary](outputs/attention_valence_summary.png)

![Attention valence timeseries](outputs/attention_valence_timeseries.png)

## Paradigm-Shift Attention Test

`attention_shift_lab.py` changes the world halfway through a run. The target
signal initially moves according to one hidden rule, then reverses direction and
changes speed. This asks whether a system merely protects an old inner model, or
whether it can use surprise to rebuild that model from sensory evidence.

Result:

```text
condition                         pre_acc  early_post  late_post  recovery
ungated_old_model                 0.727    0.314       0.267      never
static_attention_valence_filter   0.936    0.714       0.720      43 steps
adaptive_attention_valence_filter 0.936    0.914       0.947      0 steps
```

The ungated old model collapsed after the rule change and never recovered. The
static attention-valence filter stayed more grounded, but it still acted through
an obsolete internal prediction rule. The adaptive attention-valence filter used
prediction error as surprise, retuned its model angle from about `+0.13` to
about `-0.21`, and recovered immediately.

This sharpens the Phi lesson:

> Integration is not a scoreboard by itself. A useful inner world must be
> integrated enough to simulate, but plastic enough to update when sensory
> reality proves it wrong.

![Attention shift summary](outputs/attention_shift_summary.png)

![Attention shift timeseries](outputs/attention_shift_timeseries.png)

## Explainer Video Angle

This project could be turned into a short narrated explainer:

1. Start with the question: does useful intelligence need both capacity and
   tuned valence?
2. Show the exact tiny Phi proxy result: valence feedback scored higher than a
   feedforward chain or simple recurrent ring.
3. Show ablation and noise tests: the valence-feedback node became structurally
   important and improved robustness in the toy system.
4. Show wireheading: direct access to good valence made the agent stop solving
   the world.
5. Show valence shaping: progress-grounded reward helped, while direct reward
   hijacked behavior.
6. Show imagination: ungrounded imagination hurt performance, while
   accuracy-rewarded and gated imagination recovered much of the loss.
7. Show the detour maze: myopic progress-valence got stuck at the wall, while
   pretrained world-model lookahead accepted temporary negative valence and
   reached the goal.
8. Show the imagination/self Phi-proxy test: activating imagination and
   self-modeling made the circuit less separable, but counterfactual and
   imagined-valence loops did not automatically keep raising integration.
9. Show the delusional integration sweep: internal self/imagination dominance
   reduced external grounding and increased internal influence, even though Phi
   did not rise in that wiring.
10. Show the attention-valence filter: prediction-aligned imagination improves
   task focus, while self-amplified imagination can become confident but
   detached.
11. Show the paradigm-shift test: adaptive attention-valence can rebuild an
   obsolete inner model after environmental surprise.
12. End with the thesis: capacity without grounded valence is unstable; valence
   without boundaries is exploitable; imagination without reality-checking is
   delusional; attention should be rewarded for staying grounded; useful
   intelligence requires co-tuned cognition, reward, attention, and world
   modeling.

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
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 valence_scaling_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 imagination_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 maze_imagination_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 imagination_phi_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 delusional_integration_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 attention_valence_lab.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.13 attention_shift_lab.py
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
- `valence_scaling_lab.py` - behavioral scaling sweep without exact Phi
- `imagination_lab.py` - pre-action world-model/intuition test
- `maze_imagination_lab.py` - 2D maze imagination test
- `imagination_phi_lab.py` - exact tiny Phi-proxy test for imagination circuits
- `delusional_integration_lab.py` - internal-loop grounding and delusion-risk sweep
- `attention_valence_lab.py` - attention/relevance gate driven by valence and prediction alignment
- `attention_shift_lab.py` - dynamic environment shift test for adaptive re-grounding
- `outputs/metrics.json` - recurrent agent metrics
- `outputs/exact_phi_metrics.json` - exact Phi proxy metrics
- `outputs/intervention_metrics.json` - intervention test metrics
- `outputs/wirehead_metrics.json` - wireheading test metrics
- `outputs/valence_shaping_metrics.json` - valence shaping test metrics
- `outputs/valence_scaling_metrics.json` - behavioral scaling metrics
- `outputs/imagination_metrics.json` - pre-action imagination test metrics
- `outputs/maze_imagination_metrics.json` - 2D maze imagination metrics
- `outputs/imagination_phi_metrics.json` - imagination circuit Phi-proxy metrics
- `outputs/delusional_integration_metrics.json` - internal-loop grounding metrics
- `outputs/attention_valence_metrics.json` - attention-valence filter metrics
- `outputs/attention_shift_metrics.json` - paradigm-shift attention metrics

## Next Steps

Good next experiments:

- Convert trained hidden-state dynamics into a tiny binary system and calculate
  Phi proxy on that binarized subnetwork.
- Build a harder 2D world with irreversible dead ends where learned imagination
  has to outperform learned reflex, not just a hand-isolated myopic baseline.
- Compare recurrent agents with and without the valence/value head.
- Add a feedforward-only PyTorch baseline.
- Animate the hidden-state trajectory as a video.
- Add a small web UI for changing ablated units and watching the trajectory
  update.
