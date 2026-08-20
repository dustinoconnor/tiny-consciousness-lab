# PGNW Dual-Need Conflict Calibration — Seed 172

Seed 172 was the first live conflict between the held-out verified yellow
protective rule and held-out verified blue metabolic rule. It did not pass the
preregistered physical sequence.

## Result

- Normal bounded completion: 1,133 rows, 239.406 seconds, steps 0–1,132.
- First red pickup: 2.971 seconds at hunger 0.934.
- PGNW immediately selected blue with full bounded authority for the next 80
  frames, but produced zero MPC action changes during that first conflict.
- The unguided trajectory physically intercepted yellow at 19.872 seconds.
  Yellow cancelled the pending hazard before blue was reached.
- First blue pickup: 23.650 seconds, 3.778 seconds after yellow.
- Therefore the required sequence `red -> blue relief -> authority switch ->
  yellow cancellation` did not occur.
- A second red pickup at 130.831 seconds created another pending hazard. As
  hunger rose, arbitration progressed from yellow authority through 29
  low-margin abstention frames to blue authority at 178.888 seconds. This later
  phase did influence MPC, but neither target was physically reached before the
  run ended and the hazard remained pending.
- Whole-run totals: 432 arbitration frames, 403 authority frames, 107
  arbitration-sourced MPC action changes, one recovered stuck event, zero
  hazard costs, survival failures, or respawns.

## Verdict

This is a disclosed calibration failure, not confirmation of dual-rule
behavior. It proves that the verified rule scores can disagree and transfer
authority, but the first blue commitment had no causal motor effect relative to
unguided MPC. The nearby yellow route intercepted the body before blue. A fresh
confirmation should strengthen only the already bounded committed-action regret
allowance or use counterbalanced geometry where neither target lies on the
other's route; seed 172 must remain excluded from confirmation.
