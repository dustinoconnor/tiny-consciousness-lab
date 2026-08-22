# PGNW Active-versus-Passive Learning-Speed Protocol — 10 August 2026

## Question

Does bounded epistemic guidance make the Unity agent identify the delayed
red-pickup causal rule faster than passive experiment selection and ordinary
wandering?

## Conditions

Hold the Unity scene, duration, controller checkpoints, seed, causal probe,
posterior update, and all safety systems fixed. Change only:

- **Passive:** PGNW selects and records experiments but has structurally zero
  navigation guidance.
- **Bounded:** the requested visible color may contribute at most `0.03` to an
  already-engaged ambiguous MPC choice under the registered safety gates.

The completed bounded seed-141 run is the first half of the pilot pair. Run the
passive condition with seed 141 next. A single pair is diagnostic only; any
learning-speed claim requires additional counterbalanced seed pairs.

## Frozen metrics

Primary metric:

- elapsed seconds to the first posterior update at which
  `red_causes_probe >= 0.95`, censored at 1,200 seconds if never reached.

Secondary metrics:

- posterior-entropy area under the time curve;
- elapsed time to the first uncontaminated red and blue observations;
- uncontaminated experiments per recorded hour;
- requested/encountered experiment agreement;
- discarded and protocol-mismatch rates;
- PGNW guidance decisions and changed MPC selections;
- pickups, survival failures, stuck events, and critical-hunger exposure.

## Interpretation boundary

Bounded guidance supports accelerated active learning only if it reaches the
correct posterior threshold earlier across matched seeds without degrading
safety. Equal learning speed supports passive sufficiency at the tested
`0.03` authority. Because the seed-141 bounded smoke changed only one MPC
selection, a null pilot result is plausible and should not be repaired by
raising authority after seeing the passive result.
