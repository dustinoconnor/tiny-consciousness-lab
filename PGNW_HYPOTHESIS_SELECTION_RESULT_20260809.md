# Predictive-GNW Multiple-Hypothesis Result — 9 August 2026

## Outcome

The predictive workspace successfully selected informative experiments and
retained more pragmatic value than pure information seeking, but it did not
pass the preregistered overall criterion. The independent confirmation exposed
a real Pareto tradeoff: the fixed pragmatic weight bought food value at a small
but statistically resolved cost in final causal certainty.

## Passive causal interface

Before this audit, the delayed red effect was converted from an internal
`metabolic_pressure` intervention to a behaviorally disconnected
`causal_probe_signal`. Red schedules a `+0.34` signal after ten seconds; blue is
the negative control. The signal has zero prediction-error, crosstalk,
complexity, workspace, MPC, routing, and navigation influence. Historical
pressure fields remain read-only aliases for old recordings.

## Candidate pool and actions

The formal hypothesis set was:

1. red pickup causes the probe;
2. blue pickup causes the probe;
3. either pickup causes the probe;
4. the probe is spontaneous;
5. none of the tested events causes the probe.

The capacity-one selector chose observation of a red pickup, observation of a
blue pickup, or a no-pickup wait interval. Hidden worlds were counterbalanced
across all five models. The selector received model likelihoods but never the
hidden true-model index.

## Initial 20-seed audit

Across 100 counterbalanced worlds per condition, PGNW assigned mean posterior
`0.8950` to the true model and identified it in 94% of episodes. It beat
pragmatic-only and scrambled-broadcast controls, but the PGNW-over-random
interval crossed zero and the information-noninferiority interval was too wide.
No parameters or criteria were changed.

## Disjoint 100-seed confirmation

The confirmation used seed indices 20–119: 500 hidden worlds and 4,000
experiment steps per condition.

| Condition | Identification | True posterior | Reached 0.90 | Pragmatic value |
|---|---:|---:|---:|---:|
| PGNW epistemic + pragmatic | 90.8% | 0.8755 | 85.0% | 1.0790 |
| Epistemic only | 92.6% | 0.9053 | 88.2% | 0.7933 |
| Pragmatic only | 57.6% | 0.5194 | 11.2% | 1.2000 |
| Confirmation seeking | 69.6% | 0.6575 | 37.2% | 1.1443 |
| Random selection | 87.2% | 0.8227 | 70.8% | 0.7619 |
| Scrambled PGNW broadcast | 55.6% | 0.5520 | 14.2% | 0.6067 |

Paired confirmation contrasts were:

- PGNW over pragmatic-only true posterior: `+0.35608`, 95% interval
  `[+0.33451, +0.37766]`;
- PGNW over random: `+0.05279`, `[+0.02695, +0.07864]`;
- bound PGNW over scrambled broadcast: `+0.32356`,
  `[+0.30310, +0.34401]`;
- PGNW over confirmation seeking: `+0.21800`,
  `[+0.19735, +0.23866]`;
- PGNW pragmatic value over epistemic-only: `+0.28576`,
  `[+0.27286, +0.29866]`;
- PGNW true posterior versus epistemic-only: `-0.02976`,
  `[-0.04973, -0.00978]`.

The final contrast failed the preregistered `-0.03` noninferiority rule because
its lower interval bound was `-0.04973`. Four of five criteria passed; the
overall result remains a failure under the frozen decision rule.

## Interpretation

The epistemic calculation, pragmatic term, and broadcast binding all have
measurable functional consequences. The hybrid policy is much more scientific
than goal-only or confirmation-seeking selection, more informative than random
selection, and more useful than pure epistemic selection. It is not free: the
fixed `0.35` pragmatic weight measurably sacrifices causal certainty.

This does not yet use Gemma to invent the five hypotheses, select Unity motor
actions, or establish conscious access. A legitimate next experiment would
learn or select a point on the epistemic/pragmatic Pareto frontier using only
training worlds, then freeze it for untouched validation. The current weight
must not be retuned against these confirmation worlds.

Artifacts:

- `outputs/pgnw_hypothesis_selection_20260809.json`
- `outputs/pgnw_hypothesis_selection_confirmation_20260809.json`
