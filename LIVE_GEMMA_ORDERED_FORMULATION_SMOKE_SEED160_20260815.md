# Live Gemma Ordered-Formulation Smoke — Seed 160 — 2026-08-15

## Verdict

The bounded live smoke passed. Gemma formulated and the formal layer admitted
the yellow ordered-suppression rule asynchronously inside the active Unity
process, before a new post-cutoff observation reached its outcome deadline. The
new observation then independently agreed with the frozen hypothesis.

## Temporal audit

- Log: `outputs/unity_shadow/typed_yellow_live_formulation_seed160_20260815.jsonl`
- One continuous session, steps 0–816, 172.94 wall-clock seconds.
- Formulation began at step 0 from exactly three persisted discovery records.
- The controlled red and yellow pickups occurred during formulation, but their
  result was not yet known and neither was included in the frozen cutoff.
- Gemma completed at step 33 and selected raw role token `yellow`.
- The answer-blind compiler admitted at cutoff 3:
  `T1 i red s yellow r after e suppresses_probe`.
- The new red/yellow interval reached its step-309 deadline after admission,
  observed suppression, and became accepted observation 4.
- The Bayesian yellow-specific posterior rose from 92.7844% to 97.6319%.

## Runtime and safety

- Pickups: 1 red, 3 blue, and 1 yellow.
- Hidden outcomes: 1 cancellation, 0 completed probe rises, 0 pending.
- PGNW made 35 guidance decisions, changed 16 MPC actions, and began one bounded
  commitment.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation: 1 stuck event.
- Gemma/formal learner motor authority: 0.0.

## Formulation diagnostics

Content-free calibration again produced yellow evidence gain +0.307418 versus
blue -0.192582, a +0.500000 margin and two-choice confidence 0.622459. The live
result therefore reproduces the offline calibrated decoding outcome using the
same frozen evidence rather than silently consuming the later observation.

The recorded formulation status remains `admitted_unverified` because the first
live integration does not yet automatically promote status after held-out
agreement. The temporal log itself establishes one post-admission confirmation;
automatic held-out counters/status promotion are the next bookkeeping change.

## Scope

This establishes live asynchronous causal-role formulation, formal admission,
and one post-admission embodied confirmation. It remains an exploratory result:
the calibration was developed after earlier failures, and counterbalanced
color/order replications are still required for a generalization claim.
