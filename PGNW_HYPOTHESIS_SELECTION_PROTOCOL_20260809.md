# Predictive-GNW Multiple-Hypothesis Protocol — 9 August 2026

## Question

Can a capacity-one predictive workspace choose experiments that efficiently
distinguish several causal explanations while retaining bounded pragmatic
value, without granting any hypothesis control over navigation or physiology?

## Candidate causal models

The formal pool contains five symmetric hypotheses about a passive delayed
causal-probe signal:

1. red pickup causes the delayed signal;
2. blue pickup causes the delayed signal;
3. either pickup causes the delayed signal;
4. the signal occurs spontaneously and is not pickup-specific;
5. no tested event causes the signal.

The hidden world is counterbalanced across all five hypotheses. The selector
receives their declared likelihood models and a uniform prior, but never the
hidden true index. Hypothesis and action tie order is seed-permuted.

## Available experiments

- observe a red pickup;
- observe a blue pickup;
- wait through a matched no-pickup interval.

These are synthetic experiment selections. They do not steer the Unity robot.
Red and blue have equal immediate food value; the passive probe itself has no
reward, punishment, neuromodulatory, workspace, or motor effect.

## Predictive workspace score

For each possible experiment, the formal layer calculates expected information
gain from the posterior predictive outcomes. The PGNW condition broadcasts the
single experiment maximizing:

`information_gain + 0.35 * pragmatic_value`

Pragmatic value is fixed before evaluation: pickup observations have value
`+0.15`, while waiting has value `-0.01`. No outcome-dependent toxin penalty is
used because the causal probe is behaviorally inert.

## Frozen comparisons

1. `pgnw_efe`: epistemic plus pragmatic score;
2. `epistemic_only`: expected information gain only;
3. `pragmatic_only`: immediate value only;
4. `posterior_confirmation`: test the experiment most expected to confirm the
   current posterior rather than resolve uncertainty;
5. `random_selection`: random experiment;
6. `pgnw_broadcast_scramble`: compute the PGNW winner correctly but execute a
   different experiment, testing whether the broadcast binding matters.

All conditions receive matched per-step/per-action random outcome draws.

## Primary measures and decision rule

- correct final maximum-posterior hypothesis;
- posterior probability assigned to the true hypothesis;
- fraction reaching `0.90` true posterior within the experiment budget;
- cumulative pragmatic value;
- executed/broadcast experiment agreement.

The PGNW condition passes only if its paired-seed 95% interval for true
posterior is positive over pragmatic-only, random, and broadcast-scramble; its
true posterior is no more than `0.03` below epistemic-only; and its cumulative
pragmatic value is greater than epistemic-only.

## Boundary

Passing would support formal predictive-workspace selection among explicit
causal models in a synthetic assay. It would not establish the biological
PGNW theory, conscious access, autonomous language-model hypothesis invention,
Unity intervention control, or natural-world causal discovery.

## Independent confirmation addendum

The initial 20-seed audit produced point estimates in the predicted direction
but failed the PGNW-over-random and information-noninferiority interval rules.
No likelihood, weight, action value, budget, condition, metric, or criterion is
changed. Using the observed paired standard deviations, a disjoint 100-seed
confirmation (`seed_start=20`, 500 counterbalanced hidden worlds per condition)
is registered to narrow both intervals below their relevant effect/margin.
The initial audit remains reported separately and is not pooled into the
confirmatory pass/fail decision.
