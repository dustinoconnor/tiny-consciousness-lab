# PGNW Committed-versus-Passive Batch Protocol — 12 August 2026

## Question

Does the manipulation-checked committed PGNW controller improve acquisition of
diagnostic red evidence and causal-learning reliability relative to passive
experiment logging under matched Unity starts?

## Conditions

- **Committed:** constrained target acquisition at 0.18 score regret; a
  ten-second food-repulsion isolation window; and a two-second immediate retreat
  at 0.35 regret.
- **Passive:** identical PGNW hypothesis selection, probe, posterior update, and
  contamination rejection, with structurally zero navigation guidance,
  isolation, or retreat authority.

All controller checkpoints, scene state, duration, safety gates, seeds, and
causal mechanisms remain fixed. Odd seeds run committed first and even seeds
passive first. Automated reset requires zero pickup counters and records both
conditions from the same floor-start protocol.

## Initial batch

Run three matched pairs, seeds 150 through 152, for 1,200 seconds per condition.
This is a six-run, approximately two-hour replication. Analyze before adding
more seeds.

## Frozen outcomes

Primary:

- paired time to `P(red_causes_probe) >= 0.95`, censored at 1,200 seconds;
- proportion of runs reaching the threshold.

Mechanism:

- clean red observations and time to first clean red;
- red-trial contamination rate;
- requested-color pickup compliance;
- pre-threshold acquisition, isolation, and retreat action changes.

Safety and cost:

- survival failures, stuck events, critical-hunger exposure;
- total pickup and clean-experiment rates;
- posterior-entropy area over time.

## Decision boundary

Three pairs are a replication block, not a definitive inferential sample.
Committed control is promising only if it increases clean red evidence or
threshold reliability with verified pre-evidence action influence and no
systematic safety degradation. Do not credit posterior improvements to PGNW in
runs where its authority did not change behavior before evidence acquisition.
