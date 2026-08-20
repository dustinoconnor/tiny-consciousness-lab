# Typed Yellow PGNW-Guided Result — Seed 158 — 2026-08-15

## Verdict

The persistent online learner acquired strong evidence for the yellow-specific
ordered rule. It reached 92.7844% posterior confidence from one newly accepted
red-then-yellow suppression episode, but remains below the preregistered 95%
verification threshold.

## Run record

- Log: `outputs/unity_shadow/typed_yellow_guided_seed158_20260815.jsonl`
- One continuous session, steps 0–2806, 592.76 wall-clock seconds.
- Pickups: 3 red, 6 blue, and 3 yellow.
- Yellow visible frames: 68; nearest recorded distance: 2.45 units.
- Hidden-world accounting: 1 completed probe event, 2 cancelled events, and 0
  pending at termination.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation: 4 stuck events and 0 unstuck respawns.

## Accepted causal episode

- The persistent memory loaded the two seed-157 negative controls and began at
  yellow-specific 49.2121%, null 49.2121%, any-pickup 1.2269%, and blue-specific
  0.3490%.
- PGNW requested `red_then_yellow`.
- Red was collected at step 12 and yellow at step 22, ten controller steps (2.0
  nominal seconds) later.
- PGNW had changed 6 MPC actions by the yellow pickup, establishing action
  influence before the causal outcome was known.
- No additional pickup contaminated the interval through the original deadline
  at step 312. The learner observed no probe event, admitted `suppressed`, and
  persisted the update.
- Posterior after the accepted observation:
  - yellow-specific suppression: 92.7844%
  - no ordered suppression: 5.0426%
  - any-pickup suppression: 2.1372%
  - blue-specific suppression: 0.0358%

## Conservative discard

A later red at step 2029 was followed by yellow at step 2101 and also cancelled
a hidden event. However, an additional red pickup occurred at step 2329, exactly
the first red's deadline. Because pickup processing precedes deadline admission,
the interval was conservatively marked contaminated and did not update the
posterior. This avoids claiming a second independent confirmation from an
order-boundary event.

## Next gate

One additional comparable clean red-then-yellow suppression from the persisted
92.7844% posterior is expected to raise yellow-specific confidence to about
97.6319%, clearing the 0.95 threshold. A fresh Unity Play reset with the same
controlled pair and persistent memory is sufficient; no code or scene change is
needed.
