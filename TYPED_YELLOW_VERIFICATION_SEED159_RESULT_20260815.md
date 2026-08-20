# Typed Yellow Verification — Seed 159 — 2026-08-15

## Verdict

The persistent embodied learner verified the ordered
`red_then_yellow_suppresses_probe` rule above the preregistered 0.95 threshold.
The threshold was crossed by the first clean seed-159 episode; a later accepted
positive raised the final posterior further but is not required for the claim.

## Run record

- Log: `outputs/unity_shadow/typed_yellow_guided_seed159_20260815.jsonl`
- One continuous session, steps 0–2835, 594.09 wall-clock seconds.
- Pickups: 2 red, 21 blue, and 6 yellow.
- Yellow visible frames: 144.
- Hidden-world accounting: 0 completed probe events, 2 cancelled events, and 0
  pending at termination.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation: 3 stuck events and 0 unstuck respawns.

## Threshold-crossing episode

- Persistent starting posterior: yellow-specific 92.7844%, null 5.0426%,
  any-pickup 2.1372%, blue-specific 0.0358%.
- PGNW requested `red_then_yellow`.
- Red was collected at step 12 and yellow at step 22.
- PGNW had changed 3 MPC actions by the yellow pickup, before the delayed outcome
  was available.
- No intervening pickup contaminated the interval through its step-312
  deadline. The learner observed no probe event and admitted `suppressed`.
- The yellow-specific posterior reached 97.6319%, clearing the frozen 95% gate.

## Additional evidence and final state

After verification, expected information gain requested `red_then_blue`. A later
clean episode actually executed red then yellow and was admitted under its
observed action rather than the requested action. This second positive raised
the final persisted posterior to:

- yellow-specific suppression: 98.0562%
- any-pickup suppression: 1.9280%
- no ordered suppression: 0.0157%
- blue-specific suppression: 0.0001%

Across persistent memory, the learner now contains five accepted episodes and
two conservative discards. The decisive evidence comprises two earlier clean
red/blue negative controls and at least two clean red/yellow positive controls.
The hidden cancellation counter was retained for audit only; posterior updates
were made from pickup order and the observed absence/presence of the delayed
probe at the deadline. Learner motor authority remained 0.0, while PGNW target
requests used the existing safety-bounded MPC path.

## Scope

This verifies the supplied symmetric typed candidate pool's online embodied
selection of the correct ordered rule. It does not yet establish that Gemma
generated this new interaction ontology or candidate pool dynamically. That is
the next discovery-layer gate.
