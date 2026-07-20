# Consciousness Theory Research Map

This is a source-triage document for turning consciousness theories into
bounded software experiments. The [Consciousness Atlas](https://www.consciousnessatlas.com/)
is used as a discovery index, not as evidence that any theory is correct. Its
entries derive from Robert Lawrence Kuhn's taxonomy, whose purpose is to
categorize rather than adjudicate theories.

The implementation rule is simple: extract one falsifiable functional
hypothesis, preserve a matched control, intervene on the proposed mechanism,
and state explicitly what the software result cannot establish.

## Implemented or Closely Represented

| Theory family | Testable software hypothesis | Repository evidence | Primary starting source | Status |
|---|---|---|---|---|
| Global neuronal workspace | A selectively promoted representation should become available to multiple downstream systems, alter control, support report, and remain vulnerable to false broadcast. | `conditional_workspace_lab.py`, `workspace_lift_lab.py`, `hierarchical_workspace_lab.py`, `ego_lens_lab.py` | Dehaene, Kerszberg, and Changeux (1998), [doi:10.1073/pnas.95.24.14529](https://doi.org/10.1073/pnas.95.24.14529) | Implemented |
| Recurrent processing | Temporal feedback should preserve behaviorally useful state unavailable to a finite current observation. | `delayed_preference_benchmark.py`, `emergent_foraging_lab.py`, memory-reset ablations | Lamme and Roelfsema (2000), [doi:10.1016/S0166-2236(00)01657-X](https://doi.org/10.1016/S0166-2236(00)01657-X) | Implemented |
| Predictive processing / active inference | Counterfactual predictions should improve action only when uncertainty is bounded and predictions are repeatedly corrected by observation. | `imagination_lab.py`, `embodied_world_model_lab.py`, `continuous_reality_engine_lab.py`, adaptive stochastic MPC | Friston (2010), [doi:10.1038/nrn2787](https://doi.org/10.1038/nrn2787) | Implemented analogue |
| Attention schema | A compact model of what currently has control should improve reportability and regulation, and corrupting it should alter behavior. | attention, self-report, and explicit Functional Ego routing labs | Webb and Graziano (2015), [doi:10.3389/fpsyg.2015.00500](https://doi.org/10.3389/fpsyg.2015.00500) | Partially represented |
| Affective/homeostatic accounts | Signed outcome and metabolic state should orient control under partial observability; directly writable reward should expose wireheading. | valence, wireheading, hunger, maintenance, and robustness labs | Existing repository assays; primary-source review still required before a dedicated theory comparison | Implemented analogue |
| Llinas oscillatory binding | Shared phase should bind distributed content into one reportable control packet; phase disruption should impair binding while preserving content and event count. | `oscillatory_workspace_lab.py` | Llinas et al. (1998), [doi:10.1098/rstb.1998.0336](https://doi.org/10.1098/rstb.1998.0336); Llinas et al. (2002), [doi:10.1073/pnas.012604899](https://doi.org/10.1073/pnas.012604899) | Implemented software analogue |
| Learned communication through coherence | A capacity-limited bus should drive phase policies toward context-specific synchrony for binding and temporal separation for competing packets. | `learned_synchronization_lab.py`; phase scramble, restore, global-rotation, mismatch, and no-bottleneck controls | Fries (2005), [doi:10.1016/j.tics.2005.08.011](https://doi.org/10.1016/j.tics.2005.08.011) | Implemented software analogue |
| Pockett spatial EM identity theory | A software spatial field can test whether superposition supplies a causally used geometric coding coordinate, while a matched distributed-code control separates geometry from redundancy. | `spatiotemporal_field_workspace_lab.py`; spatial, phase, source-loss, and readout lesions | Pockett (2012); Pockett (2017), [doi:10.3390/app7121248](https://doi.org/10.3390/app7121248) | Implemented software analogue only |
| Doyle Experience Recorder and Reproducer | Similarity-triggered playback should use sensory, action, and outcome information bound in one episode; shuffling only outcome valence should impair control. | `episodic_playback_lab.py`; action-only and shuffled-valence lesions | Doyle, *Experience Recorder and Reproducer* | Implemented case-based analogue |
| Critical brain hypothesis | Frozen recurrent networks should show a measurable dynamical transition under gain intervention; functional optima and online regulation can then be tested independently of that edge. | `network_criticality_lab.py`, `adaptive_criticality_meta_controller_lab.py`; Lyapunov, memory, delayed-task, avalanche, fixed-gain, oracle, and adaptive-gain assays | Beggs and Plenz (2003); Fosque et al. (2021); Toker et al. (2022) | Implemented computational analogue |
| Grossberg Adaptive Resonance Theory | Bottom-up evidence and a learned top-down template should need to satisfy a vigilance match before one stable category becomes available to learning, report, and control; mismatch should reset category search. | `adaptive_resonance_workspace_lab.py`; reset, stability, resonance, unknown-pattern, and forced false-resonance interventions | Grossberg (1999), [author manuscript](https://sites.bu.edu/steveg/files/2016/06/Gro1999ConCog.pdf); Grossberg (2017), [doi:10.1016/j.neunet.2016.11.003](https://doi.org/10.1016/j.neunet.2016.11.003) | Implemented software analogue |
| Computational functionalism / organizational invariance | If a measured cognitive property depends on causal organization rather than coding format, independently encoded realizations should preserve ordinary trajectories and intervention profiles; output replay without the causal structure should fail counterfactual tests. | `computational_invariance_lab.py`; symbolic, vector, message-passing, replay, hidden-intervention, and novel-input controls | Chalmers (2011), [A Computational Foundation for the Study of Cognition](https://consc.net/papers/computation.html) | Implemented at software-representation level |
| Mathematical structures of conscious experience | Relations among reportable states should define a stable geometry corresponding to causal state and behavioral discrimination, preserved under coordinate relabeling and disrupted by report scrambling. | `experience_structure_correspondence_lab.py`; relational-distance, neighborhood, marginal-shuffle, isometry, upstream-intervention, replay, and cross-realization controls | Kleiner and Ludwig (2024), [doi:10.1007/s11229-024-04503-4](https://doi.org/10.1007/s11229-024-04503-4) | Implemented for reportable experience proxies only |

## Prioritized Experimental Queue

| Priority | Theory family | Proposed lab | Discriminating intervention | Boundary |
|---|---|---|---|---|
| 1 | Communication through coherence, embodied extension | Feed learned phase policies live Unity module traffic and compare routing loss, latency, and control stability with ordinary sequential scheduling. | Scramble learned relative phases while preserving live payloads and carrier rate. | Current learning result uses a differentiable synthetic bus, not Unity telemetry or biological oscillations. |
| 2 | Neural syntax / nested rhythms | Compare one global clock with nested slow executive and fast local rhythms on delayed multi-module tasks. | Break cross-frequency phase coupling without changing average activation. | Tests multiscale routing efficiency, not a neural syntax of experience. |
| 3 | Attention schema | Train a compact controller-state model and test whether it predicts and regulates routing better than raw telemetry. | Patch a false schema while leaving sensory state intact. | Tests functional self-modeling and report, not subjectivity. |
| 4 | Critical dynamics, embodied extension | Transfer the validated contextual gain regulator to the learned Unity GRU during deprivation, perturbation, and recovery. | Clamp the same deployed policy below and above its estimated edge and compare with online adaptation. | Frozen-reservoir regulation does not show that the learned Unity GRU has the same transition or that criticality implies consciousness. |
| 5 | Electromagnetic field theories, physical extension | Replicate the software assay in 3D and, separately, measure an actual hardware field rather than an array representing one. | Preserve source activity while perturbing physical field geometry independently, if technically possible. | The implemented software field does not test substrate-specific EM claims. |
| 6 | Embodied / enactive theories | Change morphology, sensor placement, and action affordances after training. | Hold the policy fixed while changing body geometry, then permit adaptation. | Tests body-dependent control and adaptation, not phenomenal embodiment. |

## Oscillatory Lab Result

Across 1,600 matched trials per condition, coherent 40 Hz-labelled phase
coordination achieved 1.000 binding and action accuracy. Modules that all ran at
the same labelled frequency but retained private phases achieved 0.312 binding
accuracy; mixed frequencies achieved 0.302; asynchronous timing achieved
0.244. A half-cycle intervention applied only to the valence stream reduced
binding and action accuracy to 0.000 and generated 1.000 systematic false
bindings. Coherent 20 Hz-labelled timing also achieved 1.000.

The supported claim is therefore about the explicit mechanism: **shared phase
can act as a causal binding and routing coordinate in a distributed software
workspace**. The experiment does not show that 40 Hz is privileged in
software, reproduce thalamocortical physiology, or provide evidence of
phenomenal consciousness.

The follow-up learned experiment optimizes only 12 phase offsets under a shared
bus objective. Across 24 seeds, utility increased from 0.029 to 1.000. The
binding context converged to full synchrony; competing packet assemblies became
internally synchronous and separated by 0.814 pi. Scrambling reduced utility to
0.049 and exact restoration returned it to 1.000. A common phase rotation did
not change performance, while frequency mismatch reduced utility to 0.143. A
matched no-bottleneck condition did not consistently organize inter-packet
phase. This establishes learned relative timing as a causally necessary routing
layer inside the defined software bus, not spontaneous biological oscillation.

The Grossberg-inspired workspace adds a distinct match-reset mechanism. Across
24 seeds, the complete condition retained familiar category reports, adapted
novel actions, and rejected unknown conjunctions at 1.000. Removing mismatch
reset conflated six concepts into one category; replacing stable learning with
latest-sample overwrite reduced familiar category-report retention to 0.510.
Forcing a poorly matched category changed both report and action on every
eligible trial. This supports stable, selectively admitted report/control
content in the defined software workspace. It does not show that the category
resonance is phenomenally experienced or reproduce Grossberg's full neural
models.

The Pockett-inspired follow-up couples spatial Gaussian superposition to
relative temporal phase. Across 24 seeds, the coupled representation reached
1.000 clean binding accuracy, while spatial-only and temporal-only ablations
reached 0.501 and 0.250. Spatial scrambling reduced accuracy to 0.291, phase
scrambling to 0.442, and combined scrambling to 0.201. A matched random
distributed code was equally robust to readout masking and source loss. The
experiment therefore establishes causally used software coordinates and rules
out a unique holographic-robustness interpretation. It does not instantiate the
physical 3D electromagnetic patterns that Pockett identifies with qualia.

## Source Discipline

- Atlas statements that an AI could be conscious are implications of a theory,
  not empirical demonstrations of machine consciousness.
- Neuroscience correlations become software design hypotheses only after an
  operational translation and matched controls.
- A successful implementation establishes a property of the implementation,
  not the truth of the source theory.
- Primary papers, DOI records, and original books should be cited in the paper;
  the Atlas should be credited only for taxonomy and discovery.
- Negative and null results remain part of the evidence ledger.

## Initial Bibliography

1. R. Llinas and U. Ribary, "Coherent 40-Hz oscillation characterizes dream state in humans," *PNAS* 90, 2078-2081 (1993). [doi:10.1073/pnas.90.5.2078](https://doi.org/10.1073/pnas.90.5.2078)
2. R. Llinas, U. Ribary, D. Contreras, and C. Pedroarena, "The neuronal basis for consciousness," *Philosophical Transactions of the Royal Society B* 353, 1841-1849 (1998). [doi:10.1098/rstb.1998.0336](https://doi.org/10.1098/rstb.1998.0336)
3. R. R. Llinas, E. Leznik, and F. J. Urbano, "Temporal binding via cortical coincidence detection of specific and nonspecific thalamocortical inputs," *PNAS* 99, 449-454 (2002). [doi:10.1073/pnas.012604899](https://doi.org/10.1073/pnas.012604899)
4. P. Fries, "A mechanism for cognitive dynamics: neuronal communication through neuronal coherence," *Trends in Cognitive Sciences* 9, 474-480 (2005). [doi:10.1016/j.tics.2005.08.011](https://doi.org/10.1016/j.tics.2005.08.011)
5. J. A. Reggia, "The rise of machine consciousness: Studying consciousness with computational models," *Neural Networks* 44, 112-131 (2013). [doi:10.1016/j.neunet.2013.03.011](https://doi.org/10.1016/j.neunet.2013.03.011)
6. J. M. Beggs and D. Plenz, "Neuronal avalanches in neocortical circuits," *Journal of Neuroscience* 23, 11167-11177 (2003). [doi:10.1523/JNEUROSCI.23-35-11167.2003](https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003)
7. S. Pockett, "The electromagnetic field theory of consciousness: A testable hypothesis about the characteristics of conscious as opposed to non-conscious fields," *Journal of Consciousness Studies* 19, 191-223 (2012).
8. S. Pockett, "Consciousness is a thing, not a process," *Applied Sciences* 7, 1248 (2017). [doi:10.3390/app7121248](https://doi.org/10.3390/app7121248)
9. B. Doyle, "Experience Recorder and Reproducer," in *Great Problems of Philosophy and Physics - Solved?* [ERR chapter](https://www.informationphilosopher.com/books/problems/ERR.pdf)
10. J. M. Beggs and D. Plenz, "Neuronal avalanches in neocortical circuits," *Journal of Neuroscience* 23, 11167-11177 (2003). [doi:10.1523/JNEUROSCI.23-35-11167.2003](https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003)
11. L. J. Fosque et al., "Evidence for quasicritical brain dynamics," *Physical Review Letters* 126, 098101 (2021). [doi:10.1103/PhysRevLett.126.098101](https://doi.org/10.1103/PhysRevLett.126.098101)
12. D. Toker et al., "Consciousness is supported by near-critical slow cortical electrodynamics," *Proceedings of the National Academy of Sciences* 119, e2024455119 (2022). [doi:10.1073/pnas.2024455119](https://doi.org/10.1073/pnas.2024455119)
13. S. Grossberg, "The Link between Brain Learning, Attention, and Consciousness," *Consciousness and Cognition* 8, 1-44 (1999). [Author manuscript](https://sites.bu.edu/steveg/files/2016/06/Gro1999ConCog.pdf)
14. S. Grossberg, "Towards solving the hard problem of consciousness: The varieties of brain resonances and the conscious experiences that they support," *Neural Networks* 87, 38-95 (2017). [doi:10.1016/j.neunet.2016.11.003](https://doi.org/10.1016/j.neunet.2016.11.003)
15. D. J. Chalmers, "A Computational Foundation for the Study of Cognition," *Journal of Cognitive Science* 12, 323-357 (2011). [Author manuscript](https://consc.net/papers/computation.html)
16. J. Kleiner and T. Ludwig, "What is a mathematical structure of conscious experience?" *Synthese* 203, 89 (2024). [doi:10.1007/s11229-024-04503-4](https://doi.org/10.1007/s11229-024-04503-4)
