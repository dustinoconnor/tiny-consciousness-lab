# Typed Yellow Online Learning — Seed 156 Result — 2026-08-15

## Verdict

The online passive learner operated correctly, but seed 156 supplied no yellow
opportunity and therefore did not identify the yellow-specific rule.

## Record

- Log: `outputs/unity_shadow/typed_yellow_learning_seed156_20260815.jsonl`
- One continuous session, steps 0–2818, 592.91 wall-clock seconds.
- Pickups: 3 red, 10 blue, 0 yellow.
- Yellow visible sensor frames: 0; yellow pickup count: 0.
- Probe outcomes: 2 completed, 0 cancelled, 1 still pending at termination.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation: 5 stuck events.

## Learner audit

- One initial red/blue episode was discarded after an additional pickup made it
  contaminated.
- One clean `red_wait` episode reached its deadline and correctly recorded
  `probe_observed`.
- A final clean red-wait episode began at step 2618 but its deadline was step
  2918, 100 steps beyond run termination, so it remained pending and caused no
  update.
- Accepted updates: 1; discarded episodes: 1; incomplete episodes: 1.
- All four posteriors remained exactly 0.25. This is expected because the
  preregistered candidate models assign the same likelihood to `red_wait`; that
  control verifies the common red effect but cannot distinguish the suppressor.
- Motor authority remained 0.0.

## Interpretation

This is not evidence against the yellow rule and not evidence that the learner
failed. Passive wandering simply provided no yellow observation. Repeating
unguided seeds will have high opportunity variance. The next efficient gate is
to let the typed experiment selector request `red_then_yellow` and
`red_then_blue` targets through safety-bounded PGNW guidance while leaving
posterior updates passive and deadline-based.
