# PGNW Observation-Isolation Smoke — 12 August 2026

## Purpose

Test whether the committed Tiny Scientist can preserve a clean ten-second
causal observation after acquiring a requested mushroom. The adjacent red/blue
spawn pair is retained as a deliberate contamination stress test.

## Frozen target acquisition

Keep committed PGNW target acquisition unchanged: three-second commitments,
two-second cooldown, and a maximum MPC-score regret of 0.18. Keep the causal
probe, five-model posterior, ten-second delay, contamination rejection,
controller checkpoints, and Unity scene unchanged.

## Isolation authority

After any experimental pickup starts the ten-second observation window:

- suppress optional food-seeking and clear the ordinary foraging commitment;
- combine visible red and blue directions into an inverse-distance-weighted
  repulsion vector;
- choose the most food-averse action only among collision-safe MPC actions
  within the same 0.18 score-regret bound;
- stop isolation when the observation completes or is contaminated.

Isolation yields absolutely to stable fallback, stuck recovery, hunger at or
above 0.92, guided AIR, guided resource memory, and collision masking. It does
not activate MPC by itself.

## Ten-minute criterion

Pass if the run records:

- at least one red pickup requested by PGNW;
- at least one isolation decision after that pickup;
- at least one uncontaminated completed red observation;
- fewer intervening-pickup contaminations per red start than the seed-147
  smoke's 2/2;
- no survival failure attributable to isolation.

A pass establishes evidence-preservation feasibility only. A matched overnight
comparison remains necessary for any learning-efficiency claim.
