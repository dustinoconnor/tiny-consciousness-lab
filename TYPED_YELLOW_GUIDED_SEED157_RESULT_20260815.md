# Typed Yellow PGNW-Guided Result — Seed 157 — 2026-08-15

## Verdict

The run established active PGNW causal influence and accumulated informative
negative-control evidence, but it did not complete a red-then-yellow episode.
The yellow-specific rule therefore remains tied with the null model.

## Run record

- Log: `outputs/unity_shadow/typed_yellow_guided_seed157_20260815.jsonl`
- One continuous session, steps 0–5686, 1193.07 wall-clock seconds.
- Pickups: 4 red, 32 blue, and 3 yellow.
- Yellow was visible on 72 recorded frames and approached to 2.47 units.
- Hidden outcomes: 4 probe events, 0 cancellations, 0 pending at termination.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation: 5 stuck events and 0 unstuck respawns.

## Causal and learning audit

- PGNW requested `red_then_yellow` throughout because that remained the most
  informative unavailable outcome.
- PGNW made 103 guidance decisions, changed 63 MPC actions, and began 3 bounded
  target commitments. Fifty-six of those changed actions occurred during
  post-pickup isolation. This is direct pre/post-evidence control influence,
  not passive-wandering equivalence.
- Two clean `red_then_blue -> probe_observed` episodes reached their deadlines
  and updated the learner. One additional episode was discarded for pickup
  contamination.
- No `red_then_yellow` episode completed. The first and last initiating red
  pickups were followed by blue only six controller steps later, before PGNW
  could redirect toward yellow. This is consistent with the manually colocated
  red/blue spawn pair left from earlier temporal-foresight tests.
- Final posterior:
  - yellow-specific suppression: 49.2121%
  - no ordered suppression: 49.2121%
  - any-pickup suppression: 1.2269%
  - blue-specific suppression: 0.3490%
- Thus the run strongly rejects blue-specific and generic any-pickup accounts,
  but cannot distinguish yellow-specific suppression from no suppression.

## Required next evidence

Move or remove the manually colocated blue pickup beside the spawn red pickup,
then collect clean red-then-yellow positive episodes. One positive update from
the current posterior would raise the yellow-specific model to about 92.78%; a
second comparable positive should clear 95%. Because the current posterior
exists only in this completed process/log, a persistent typed-interaction memory
must be added before the next run if these two negative controls are to carry
forward rather than restarting at uniform priors.
