# Grounded Recurrent Control in Toy Partially Observable Worlds

## An audit-first study of memory, workspace routing, zero-shot detours, Python-to-Unity deployment, stochastic model-predictive control, and robustness

**Tiny Consciousness Lab**  
Research paper draft · 16 July 2026

### Abstract

This paper audits and synthesizes a local repository of small control experiments that combine recurrent policies, signed outcome feedback, workspace-like routing, learned forward models, and model-predictive control (MPC). The contribution is empirical and methodological rather than a claim about consciousness. We distinguish demonstrations supported by code and stored outputs from hypotheses and planned work. In a five-seed delayed-preference POMDP, a recurrent policy receiving a grounded signed outcome pulse retained preferred-action accuracy across delays up to 28 steps (0.820 ± 0.158 across seeds at delay 28), whereas an outcome-enabled feedforward policy with an eight-frame context fell to chance (0.499 ± 0.006). Resetting recurrent state, zeroing or shuffling the outcome pulse, or reversing its sign reduced accuracy to 0.500, 0.500, 0.500, and 0.180, respectively. In navigation, recurrent reward-trained policies reached a withheld rotated U-detour on 67.5% of trials; per-step hidden-state reset reduced their success to 0%. A larger five-seed continuous-ray pipeline achieved 98.25% mean success across strictly withheld C-shape and zigzag families, with a family-specific memory-reset drop from 98.0% to 23.5% on zigzag gates. A Python policy and calibrated predictive controller were then deployed into a Unity body through a UDP bridge. A recorded visibility-gated controller succeeded in all 12 completed course episodes, while offline calibration reduced held-out transition MAE by 41.7%. In a separate 36-episode Python benchmark, uncertainty-bounded adaptive stochastic MPC matched fixed MPC at 35/36 successes while reducing mean steps from 88.1 to 82.4. Engineered workspace and robustness assays show useful routing and recovery effects, but remain synthetic and largely hand-designed. Collectively, the results support a narrow thesis: temporal state, externally grounded outcome signals, selective information routing, and uncertainty-aware receding-horizon control can be composed into compact partially observable agents. They do not demonstrate phenomenal consciousness, sentience, AGI, or biological equivalence.

**Keywords:** recurrent control; partial observability; reward feedback; model-predictive control; global workspace; zero-shot navigation; uncertainty; robustness; Unity

## 1. Introduction

Partially observable control requires an agent to act from incomplete local evidence. Three practical problems recur across domains: information relevant to the current decision may have arrived many steps earlier; learned predictions can become unreliable when rolled forward; and globally shared representations can improve coordination while also propagating error. The Tiny Consciousness Lab repository explores these problems through deliberately small tasks. Its vocabulary includes “valence,” “workspace,” “imagination,” and “functional ego.” In this paper, those terms are operationalized as control components rather than treated as evidence about subjective experience.

The repository’s current working thesis is broad: recurrence supplies temporal coupling, grounded valence supplies functional orientation, workspace routing regulates which internal source controls action, and world models support detours that myopic progress signals cannot solve. The audit reported here narrows that thesis to claims the stored artifacts can support. It asks six questions:

1. Does recurrent state preserve a grounded outcome signal beyond a matched feedforward context window?
2. Does reward-trained behavior transfer to withheld detour topologies, and is memory causally necessary?
3. Do designed workspace representations route information in reportable, reusable, and causally active ways?
4. What transfers from Python training and evaluation into the Unity runtime?
5. Does uncertainty-bounded adaptive stochastic MPC improve the control trade-off over fixed MPC?
6. Which robustness claims survive controlled perturbations, and where do the tested governors fail?

These questions connect to established work on recurrent policies for POMDPs, reward-channel integrity, global workspace architectures, and uncertainty-aware model-based reinforcement learning. Recurrent networks are a standard way to condition policy on observation history in partially observed environments [1]. Reward shaping can accelerate learning but can also change the optimized behavior unless carefully constrained [2], and reward-channel tampering creates a distinct alignment failure [3]. Global workspace theories emphasize selective amplification and broad access among specialized processors [4,5]; here, “workspace” refers only to an engineered information-routing pattern inspired by that functional description. Finally, probabilistic model ensembles and receding-horizon replanning are established tools for limiting compounding model error [6].

The paper makes three contributions. First, it provides an evidence ledger that separates direct demonstrations from interpretation and planned work. Second, it organizes the strongest local experiments into a coherent control architecture without treating the repository name as a result. Third, it identifies provenance, statistical, and external-environment limitations that must be resolved before publication-quality claims are possible.

## 2. Audit protocol and claim discipline

### 2.1 Audited snapshot

The audit used the repository working tree on 16 July 2026. The Git baseline was `main` at commit `183ce36`, but multiple relevant files were modified or untracked. Accordingly, the unit of evidence is the local snapshot, not the Git commit or GitHub repository. We inspected the README, experiment scripts, stored JSON outputs, checkpoints, plots, and Unity telemetry. Full training runs were not repeated.

Each claim was assigned one of five labels: demonstrated; demonstrated with material qualification; hypothesis; planned; or unsupported. A result counted as demonstrated only when code and a direct stored metric or telemetry artifact were both present. README language alone did not qualify. The complete ledger appears in the companion `EVIDENCE_LEDGER.md`.

### 2.2 Terminology

**Recurrent memory** means a GRU or other state updated across timesteps. A hidden-state reset is treated as a causal intervention on temporal continuity.

**Grounded valence** means a signed scalar derived from task outcome, such as food contact or correct/incorrect choice. It is not a measurement of felt affect. Because “valence” risks importing a phenomenal interpretation, we also use the more precise term *grounded outcome feedback*.

**Workspace routing** means a shared, compressed representation or coupling pathway accessible to multiple downstream functions. It does not imply conscious access.

**Zero-shot detour transfer** means evaluation on obstacle topology families withheld from policy training. It does not imply broad zero-shot reasoning.

**Python-to-Unity transfer** means that a policy trained or calibrated in Python controls a Unity agent through a network bridge. It is cross-runtime deployment within simulation, not sim-to-real transfer.

### 2.3 Statistical reporting

Stored results vary in rigor. The delayed-preference and upgraded-foraging experiments use five training seeds. The foundational U-detour experiment uses a single training seed with multiple evaluation episodes. Conditional and hierarchical workspace studies store a single seeded synthetic sequence. Robustness sweeps use 80 or 100 seeded replicates per condition. Unity course evidence contains a small number of completed episodes and temporally correlated frames. Where per-episode paired data are absent, we report descriptive differences and do not infer statistical significance.

## 3. Architecture and experimental sequence

The strongest supported architecture is a division of labor rather than a monolithic “mind.” A recurrent policy integrates local rays, visible-goal direction, hunger, prior action, and prior reward. Grounded outcome pulses provide task feedback. A body-clearance mask removes physically invalid moves. When a goal is hidden, recurrent control supports exploration from local history. When the goal becomes sensor-visible, a learned ensemble forward model supports receding-horizon action evaluation. Workspace experiments separately test whether compressed conflict or obstruction signals can be promoted to shared control. Robustness experiments perturb the routing process and add engineered governors.

The experiments form a staged ladder:

1. a minimal delayed-outcome POMDP isolates temporal memory and feedback;
2. grid and continuous-ray navigation test withheld topologies and hidden-state interventions;
3. workspace tasks test selective coupling and packet intervention;
4. predictive heads are calibrated on Unity telemetry;
5. fixed and adaptive MPC are evaluated in Python continuous courses;
6. a visibility-gated controller is deployed in Unity; and
7. synthetic perturbation sweeps test failure and recovery modes.

The stages are related by design, but they are not a single end-to-end preregistered experiment. Results should therefore be read as converging engineering evidence, not as one confirmatory test.

## 4. Recurrent memory and grounded outcome feedback

### 4.1 Delayed hidden-preference assay

The cleanest matched assay contains two actions and a hidden preferred action. A decision earns +1 for the preferred action and −1 otherwise. The preference reverses halfway through each 12-decision episode without an explicit cue. The only evidence about preference is a signed outcome pulse immediately after a decision. Between decisions, Gaussian distractor inputs fill a variable delay. The feedforward policy receives an explicit eight-frame window; the recurrent policy uses a 32-unit GRU. Parameter counts are closely matched: 4,127 for feedforward variants and 4,194 for recurrent variants.

Four conditions were trained across five seeds: feedforward without outcome, feedforward with outcome, recurrent without outcome, and recurrent with outcome. Training sampled delays of 2, 4, 6, 8, 12, 16, and 20 steps; evaluation additionally included delays 10 and 28.

The result isolates complementarity. Neither recurrence alone nor a finite window without outcome information solved the task. The feedforward+outcome condition reached 0.813 mean accuracy through delay 8, then fell to 0.502 at delay 10 and remained at chance. The recurrent+outcome condition stayed between 0.817 and 0.838 across delays 2–20 and reached 0.820 ± 0.158 at delay 28. The wide across-seed spread is material: the effect is strong in mean behavior but training stability is not solved.

The ablations strengthen the mechanistic interpretation. At delay 28, resetting recurrent state every step yielded 0.500 accuracy; zeroing the outcome channel yielded 0.500; shuffling outcome pulses across the batch yielded 0.500; and flipping outcome sign yielded 0.180. Normal control reached 0.820. This establishes use of temporal state and signed task feedback within the assay. It does not establish that recurrence is uniquely optimal, that the learned state resembles biological affect, or that the same mechanism generalizes to all POMDPs.

![Figure 1. Preferred-action accuracy across temporal delays. Shading is ±1 across-seed standard deviation, not a confidence interval.](../outputs/delayed_preference_benchmark.png)

### 4.2 Negative and mixed results

An older matched navigation benchmark compares feedforward, feedforward+valence, recurrent, and recurrent+valence controllers. It uses one training seed and produces mixed performance. The recurrent+valence controller often reduces collisions, but neither reward nor preferred-pickup fraction consistently dominates across conditions. Because the output aggregates a single seed, its reported standard deviations are zero. This benchmark does not support a general claim that recurrence or valence improves navigation. Its proper role is diagnostic: adding architectural components can hurt, and task design determines whether memory or feedback is useful.

This negative evidence is important. The supported thesis is not “more loops plus valence equals better intelligence.” It is that memory and grounded feedback become useful when the task makes past outcomes both informative and inaccessible from the current observation.

## 5. Zero-shot detour foraging

### 5.1 Foundational emergence experiment

The foundational foraging task removes explicit `approach_food`, `escape_U`, map, frontier, and enclosure rules. Policies receive local obstacle rays, visible-food direction, hunger, previous action, previous reward, and food-contact reward. Training uses open, L-wall, and offset-barrier layouts. Evaluation uses randomly rotated U-detours withheld from training. No trap label is used during training.

The reward-trained recurrent and recurrent+curiosity policies each reached 67.5% success. A no-food-reward recurrent control reached 0%. However, a feedforward reward-trained policy also reached 67.5%. Thus, the experiment demonstrates reward-grounded food seeking and withheld U-detour transfer, but not a recurrence advantage in raw success.

The memory-reset intervention provides the stronger recurrent result. Resetting hidden state after every movement reduced both recurrent reward-trained conditions from 67.5% to 0%, holding policy weights and current observations fixed. This shows causal dependence on temporal continuity for those trained policies. Linear probes also decoded a binary trap-context label from recurrent state at 93.2–99.6% accuracy, exceeding observation-only probes by 5.7–10.3 points and shuffled-label controls by much more. That is evidence of distributed context information, not proof of a human-like enclosure concept.

### 5.2 Five-seed continuous-ray replication

The upgraded pipeline increases the action vocabulary to eight directions, uses eight continuous normalized rays, and trains on open, pocket, L-wall, offset-barrier, and U-trap families. C-shape and zigzag-gate families are strictly withheld. Curiosity strength is selected using familiar validation rather than withheld results. Five policies are trained with seeds 101, 211, 307, 419, and 523.

Across 400 withheld episodes—80 per seed—the selected group averaged 98.25% success. Individual seed success ranged from 95.0% to 100%. C-shape and zigzag success were 98.5% and 98.0%, respectively. No-reward controls averaged 1.0%. The collision count was zero, but every condition used an engineered capsule/body-clearance mask; zero collision is therefore a property of the composed controller, not a learned-policy claim.

Memory reset reduced aggregate success to 51.5%. The family breakdown is the more informative result: C-shape remained at 79.5%, while zigzag fell from 98.0% to 23.5%. The 74.5-point zigzag drop supports memory-dependent transfer in the topology where repeated turns make observation history more important. It also cautions against reporting the aggregate reset as a universal loss of competence.

![Figure 2. Five-seed training, familiar-only curiosity selection, and withheld evaluation gates.](../outputs/upgraded_foraging_pipeline_summary.png)

### 5.3 Mechanistic boundary

The repository’s strongest interpretation is distributed and context-dependent. Hidden-state probes show that trap direction or context can be decoded. Selective erasure can reduce real U-detour success. Yet injecting a decoded global direction does not reliably create retreat in clear or sensory-neutral corridors. The evidence therefore does not support a single portable “escape” vector. A more conservative model is that recurrent state modulates action jointly with current rays, hunger, and recent trajectory.

## 6. Workspace routing

### 6.1 Conditional coupling

The conditional-workspace experiment generates a 240-step rule-shift sequence. Specialist sensory, imagination, and valence channels can act alone or be altered by a workspace. Four conditions compare bypass, always-on workspace, hard-threshold routing, and soft tension-gating. The coupling coefficient α measures workspace influence.

Always-on control achieved the best late accuracy (0.976) but used α=1.0 at every step and obtained a hand-defined efficiency score of 0.763. Hard-threshold routing reached 0.953 late accuracy with mean α=0.067 and efficiency 0.891. Soft routing also reached 0.953 with α=0.117 and efficiency 0.878. Under high tension, soft α rose from 0.028 to 0.319, workspace rewrite rose from 0.0002 to 0.0315, and imagination rewrite rose from 0.0044 to 0.0650.

This is a demonstration of designed conditional routing. It is not evidence that a learned controller discovered a workspace or that the efficiency formula is externally valid. Only one seeded sequence is stored, so small differences between hard and soft routing are hypothesis-generating.

### 6.2 Reportable packet and causal intervention

The workspace-lift experiment compares reflex-only control, private modules, a globally promoted packet, and a forced packet intervention across tree, rock, dense-mushroom, and false-alarm scenarios. The packet contains intent, problem, strategy, feeling, and confidence fields. Forty seeded replicates are stored for each condition.

Private modules escaped the three obstruction scenarios in 19.4–22.2 steps. The global packet reduced escape to 11.0–11.5 steps, produced accurate reports, and reused a common `local_obstruction_cluster` strategy across obstruction labels. Forced packet injection reduced tree-pocket escape to 9.45 steps, saving 66.8 steps relative to reflex-only control. The intervention also forced breakout behavior in the false-alarm condition and produced zero report accuracy there. This paired success and failure is useful: the shared packet is causally active, but global availability does not guarantee truth.

### 6.3 Hierarchical routing

A hierarchical master experiment compares monolithic, flat multi-workspace, fast master, and slow “bureaucratic” routing on a rule shift. The fast master slightly exceeded the monolithic controller in early post-shift accuracy (0.714 vs 0.686), recovery (15 vs 16 steps), and efficiency (0.858 vs 0.851). The flat and slow hierarchies performed worse. Because the differences are small and single-seed, the result supports only an engineering hypothesis: compressed conflict signals may be useful when arbitration is fast enough, while hierarchy can add harmful latency.

The workspace experiments align with functional descriptions of global availability and routing [4,5], but they do not test the neural or phenomenal claims of global neuronal workspace theory. Their contribution is a set of falsifiable software criteria: shared representations should be accessible to multiple downstream functions, causally active under intervention, selectively gated, and vulnerable to false broadcast.

## 7. Python-to-Unity deployment and calibrated MPC

### 7.1 Transfer definition and bridge

The deployment bridge sends commands from Python to Unity on localhost and returns body/world telemetry to Python. The controller observes local rays, food visibility and direction, hunger, previous action, and previous reward. A learned recurrent policy proposes actions; a geometry-aware body-clearance mask removes invalid actions; MPC can evaluate candidate actions with learned predictive heads.

This is a meaningful systems step because the policy is no longer evaluated solely inside its training script. The original terrain project and its third-party art remain outside the repository, but the public artifact now includes `unity/TrapCourseLab`, a minimal asset-free Unity project that reconstructs the six primitive courses, capsule body, food target, sensors, UDP bridge, and camera. This closes the source-availability gap for the course benchmark while leaving the exploratory terrain scene externally dependent. We call the result Python-to-Unity or cross-runtime transfer, not sim-to-real transfer.

### 7.2 Predictive calibration

A 25.6-minute Unity recording contains 7,409 frames and 4,879 usable transitions. The policy and recurrent memory were frozen while only three forward-model heads were calibrated. A chronological split allocated 3,477 transitions to training, 671 to validation, and 731 to testing. Test MAE fell from 0.1267 to 0.0739, a 41.7% reduction. Constrained-region MAE also improved from 0.1294 to 0.0886.

The chronological split is stronger than a random frame split, but adjacent transitions remain temporally correlated and originate from one run. The effective independent sample size is therefore lower than 731, and transfer to other Unity scenes is not established.

### 7.3 Fixed MPC in matched Python courses

The calibrated controller evaluates eight root actions over a four-step horizon. Its score combines policy prior, predicted target progress, collision risk, ensemble disagreement, and angular jerk. It executes one action, then replans from the next observation. On 108 matched continuous courses, selected calibrated MPC achieved 95.37% success, compared with 90.74% for the baseline recurrent policy. Mean steps fell from 102.8 to 86.3 and mean path length from 45.2 to 38.0. Both used the body-clearance mask and reported zero collisions.

The stored summary does not contain paired episode-level outcomes, so the success difference cannot be subjected to an audited paired test. It is best treated as an engineering improvement on this suite.

### 7.4 Recorded Unity course run

The strongest live artifact is a visibility-gated recurrent/MPC course log. The gate leaves recurrent control in charge when food is hidden and engages MPC when food becomes visible, holding engagement briefly across telemetry updates. In the analyzed file, 12 completed episodes—two cycles of U-Trap, C-Trap, L-Wall, Zigzag, Offset Barriers, and Narrow Corridor—produced 12 successes, zero timeouts, zero reported stuck frames, mean completion time 29.74 s, and learned-control fraction 1.0.

Two additional episode fragments in the same analysis are marked unsuccessful but are not included in `completed_episodes`; they lasted 21.3 s and 1.0 s and appear to be aborted or incomplete course records. Consequently, “12/12” accurately describes the prespecified completed episodes but not every parsed episode fragment. With only 12 completions, uncertainty is substantial. The result should be replicated across sessions with a preregistered rule for aborted episodes.

The README contrasts this run with an earlier full-time-MPC run that completed 8/13 attempts and failed hidden-goal U-Trap and L-Wall cases. This sequential comparison supports the architectural division of labor but is not randomized or paired. Controller code and environment state may have changed between sessions.

## 8. Adaptive stochastic MPC

The adaptive stochastic controller changes inference without retraining the checkpoint. It defaults to recurrent control when planning demand is low. Planning is triggered by visible food, low proposed clearance, or prolonged lack of progress. Horizon increases from 4 to 6 or 8 with policy entropy, obstruction, hunger, or stagnation. For each root action, three ensemble-conditioned rollouts sample deviations around ensemble consensus. Rollouts terminate when cumulative ensemble disagreement exceeds 0.0015. Action scores combine mean return, a lower quartile, and spread to penalize downside and uncertainty.

In the stored Python continuous-course benchmark, recurrent control reached 34/36 successes; fixed MPC and both adaptive variants reached 35/36. Adaptive stochastic MPC reduced mean steps from 88.1 to 82.4 and path length from 38.8 to 36.3 relative to fixed MPC. Measured evaluation time fell from 24.27 s to 8.34 s, and planning was used on 70.4% of frames. The mean requested horizon was 6.20, while uncertainty stopping reduced realized depth to 5.03. The hunger-adaptive sensing variant produced essentially the same result: 35/36 success and 82.3 steps. It exercised longer sight radii but did not demonstrate a performance gain.

These results are provisional for three reasons. First, 36 episodes provide little resolution when controllers differ by at most one failure. Second, wall-clock timings are not a controlled latency benchmark. Third, the recorded 12/12 Unity course run predates the adaptive stochastic artifact.

A subsequent targeted Unity terrain diagnostic exercised the adaptive controller at a location implicated by repeated overnight starvation failures. With normal hunger the controller collected 18 mushrooms in 180 s. Forcing continuous critical-hunger MPC produced zero pickups, one stuck event, and 77.8 collision-seconds despite available food. A visibility-reanchored variant allowed recurrent exploration when no target was visible and used short stochastic MPC only for grounded food or obstacles; it collected 18 mushrooms, produced zero stuck events, and reduced hunger from 1.0 after 25.3 s. This before/after diagnostic isolates a plausible controller failure mechanism, but its short, sequential design does not establish long-duration reliability or a general adaptive-MPC advantage.

## 9. Robustness experiments

### 9.1 Perturbation sweep

The altered-state robustness task varies two synthetic controls: `noise_injection`, which perturbs internal salience, and `calcium_gate`, which adjusts promotion/excitability. The labels are metaphors; the simulator is not a biological or clinical model. Across a 4×4 grid, each cell runs 80 seeded 220-step episodes. The agent must eat, escape traps, and avoid an absorbing internal loop.

At zero noise, all promotion settings achieved 100% survival. At noise 0.35, survival remained 100% through calcium 0.75 but fell to 63.75% at calcium 1.0, where the false-promotion ratio reached 0.452. At noise 0.70, survival was 81.25% at calcium 0.15, 60% at 0.45, and 0% at 0.75 or 1.0. At maximum noise, only the lowest calcium setting produced nonzero survival (11.25%). The sweep demonstrates an interaction in the defined system: permissive promotion is useful under clean signals but destructive when noise is high.

### 9.2 Stabilizer interventions

The stabilizer experiment fixes noise and calcium at 1.0 and compares nine engineered governors over 100 replicates. Baseline, reality gate, meta-monitor, and homeostatic plasticity each produced 0% survival. A hunger anchor reached 1%; an earlier full stack reached 21%; predictive clamp reached 37%; a next-generation stack reached 74%; and sensory focus reached 94%.

No single scalar captures the trade-off. Sensory focus maximized survival and food intake but retained a false-promotion ratio of 0.585 and spent 82.1 steps in the internal loop. The next-generation stack reduced false promotions to zero and reached 74% survival, but retained fewer legitimate trap escapes than sensory focus (17.3 vs 30.3) and only partially restored performance. These results show that different governors optimize different robustness objectives. They do not support psychological diagnosis, claims about altered human states, or biological calcium mechanisms.

![Figure 3. Engineered stabilizers under maximum synthetic noise and promotion pressure.](../outputs/altered_state_stabilizer_summary.png)

## 10. Discussion

### 10.1 What the repository demonstrates

The strongest result is compositional. The delayed-preference assay shows that recurrent state can carry a grounded signed outcome beyond a fixed context window, and targeted corruptions abolish or reverse behavior. Navigation then shows that reward-trained control can transfer to withheld detour families, with recurrent memory becoming causally important in particular topologies. Workspace assays show that designed shared packets and conditional coupling can change downstream action, reduce cost, and propagate false states. Predictive calibration and receding-horizon control improve measured navigation in the Python course suite, while recorded telemetry shows that the composed controller can operate in Unity. Robustness sweeps expose regimes where routing fails and where engineered governors partially recover function.

These findings support a bounded control thesis:

> Compact partially observable agents can benefit from a regulated division of labor in which recurrent state preserves task-relevant history, externally grounded outcome signals orient learning, workspace-like routes selectively share conflict or obstruction information, and uncertainty-aware receding-horizon control is used when sensory grounding makes prediction actionable.

Every clause is functional and testable. None requires a claim about subjective experience.

### 10.2 What remains hypothetical

The repository does not establish that these modules are necessary or sufficient for consciousness. It does not show open-ended transfer, language-level reasoning, autonomous goal formation, or broad adaptation. It does not show that workspace packets correspond to conscious access, that signed reward is felt valence, or that recurrence creates phenomenology. “Substrate independence” is defensible only as a hypothesis that similar control computations can be implemented in different media; it is not evidence that phenomenology is substrate-independent.

The broader architectural claim—that useful intelligence is concentrated in regulated routing—remains plausible but under-tested. Most workspace rules are designed, not learned. Robustness tasks encode their own failure and recovery mechanisms. The Unity project is external to the audited repository. These limitations prevent treating the current stack as a unified cognitive architecture in the strong sense.

### 10.3 Negative results refine the thesis

Several failures are theoretically useful. Recurrence without informative feedback stays at chance in the delayed-preference task. Feedforward reward control matches recurrent success in the foundational U-detour. The older navigation benchmark does not show stable component superiority. A full-time MPC controller fails when the target is hidden. Hunger-adaptive sensing does not improve the small adaptive-MPC benchmark. Workspace injection improves escape but also creates false alarms. A reality gate can eliminate false promotions without restoring food pursuit. These results argue for conditional composition rather than component maximalism.

## 11. Limitations and threats to validity

**Snapshot provenance.** The working tree is not clean. Some output files were generated just before subsequent code edits. The delayed-preference script currently writes a contrast block absent from the stored JSON, and the Unity bridge was modified after the key live log. Exact code-to-output identity is therefore not guaranteed.

**Researcher degrees of freedom.** The repository contains many sequential experiments, tuned thresholds, named scores, and controller variants. There is no preregistration or held-out global test set across the project. Reported results should be treated as development evidence.

**Independence and uncertainty.** Episode counts can overstate effective sample size when trials share trained policies, layouts, or temporally adjacent telemetry. Several experiments use one training seed or one synthetic sequence. Confidence intervals and paired tests are often impossible from aggregate JSON.

**Engineered safety and routing.** The body-clearance mask directly prevents invalid motion. Workspace fields, routing thresholds, robustness failures, and governors are substantially hand-designed. Performance belongs to the complete engineered system, not solely to learned representations.

**External Unity dependency.** The minimal course source and settings are included, but the large exploratory terrain project and third-party art assets are not. Terrain telemetry therefore cannot be reproduced exactly from this repository alone.

**Semantic overreach.** Terms such as valence, delusion, revelation, ego, and calcium may be useful metaphors but can be mistaken for biological constructs. Publication should foreground operational definitions and consider neutral variable names.

## 12. Planned confirmatory program

A publication-quality next phase should freeze a versioned artifact containing Python and Unity sources, checkpoints, package versions, raw per-episode records, and run manifests. Headline hypotheses should be preregistered before new tuning. The delayed-preference result should be repeated with more seeds, alternative memory architectures, matched compute, and held-out delay distributions. Navigation should add irreversible dead ends, changed dynamics, sensor dropout, and morphology shifts. Mechanistic work should pair decoding with erasure, patching, dose response, and matched off-target interventions.

Unity comparisons should randomize controller order within session, predefine treatment of aborted episodes, and store paired trajectories. Adaptive stochastic MPC should be evaluated live against fixed MPC and raw recurrent control. Workspace routing should be learned from data and tested for intervention, reuse, reportability, and false-broadcast susceptibility without hand-coding the target packet. Robustness governors should be evaluated across different tasks and perturbation families rather than only the simulator that motivated them.

## 13. Conclusion

The audited repository supports a focused research program in grounded, recurrent, partially observable control. Its strongest demonstrations are: long-delay use of signed outcome feedback by recurrent policies; causal memory dependence in specific detour tasks; repeated-seed transfer to withheld obstacle families; improved held-out Unity prediction after predictive-head calibration; successful Python-to-Unity course control; and partial recovery from synthetic routing failures. Workspace and adaptive stochastic MPC results are promising engineering prototypes with clear next tests.

The evidence does not establish machine consciousness, phenomenal valence, sentience, AGI, or biological equivalence. Avoiding those overclaims makes the result stronger, not weaker: the repository already contains a falsifiable set of control experiments about memory, feedback integrity, selective routing, predictive grounding, and robustness.

## References

[1] M. Hausknecht and P. Stone, “Deep Recurrent Q-Learning for Partially Observable MDPs,” 2015. [arXiv:1507.06527](https://arxiv.org/abs/1507.06527).

[2] A. Y. Ng, D. Harada, and S. Russell, “Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping,” *Proceedings of ICML*, 1999. [Paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

[3] T. Everitt, M. Hutter, R. Kumar, and V. Krakovna, “Reward Tampering Problems and Solutions in Reinforcement Learning: A Causal Influence Diagram Perspective,” 2019. [arXiv:1908.04734](https://arxiv.org/abs/1908.04734).

[4] B. J. Baars, “Global workspace theory of consciousness: toward a cognitive neuroscience of human experience,” *Progress in Brain Research*, vol. 150, pp. 45–53, 2005. [doi:10.1016/S0079-6123(05)50004-9](https://pubmed.ncbi.nlm.nih.gov/16186014/).

[5] G. A. Mashour, P. Roelfsema, J.-P. Changeux, and S. Dehaene, “Conscious Processing and the Global Neuronal Workspace Hypothesis,” *Neuron*, vol. 105, no. 5, pp. 776–798, 2020. [doi:10.1016/j.neuron.2020.01.026](https://pubmed.ncbi.nlm.nih.gov/32135090/).

[6] K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models,” *Advances in Neural Information Processing Systems 31*, 2018. [Paper](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html).

## Data and code availability

The release includes experiment code, compact metric artifacts, small PyTorch checkpoints, the evidence ledger, and an asset-free Unity trap-course project under `unity/TrapCourseLab`. Large raw Unity JSONL recordings and the third-party terrain/art collection are intentionally excluded. Before formal submission, the project should additionally freeze package versions, generate a clean reproduction manifest, archive prespecified raw per-episode records, and rerun headline results from that immutable snapshot.
