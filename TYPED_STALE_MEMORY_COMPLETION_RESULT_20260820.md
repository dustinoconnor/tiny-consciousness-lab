# Typed Stale-Memory Physical Completion — Seeds 167–168

## Purpose

Test whether verified protective recall can reject a deliberately false nearby
yellow memory and then physically complete the route to a genuine remembered
yellow, rather than succeeding through an opportunistic pickup.

Both runs used the frozen seed-163 acquisition map, started 8.04 units short of
the natural red at `(82.9424, -181.1689)`, injected a decoy yellow at
`(82.9424, -171.1689)`, and retained the genuine yellow at
`(116.8036, -174.3526)`.

## Seed 167 — calibration failure

Seed 167 ran 119.425 seconds. The robot collected red and received extensive
bounded protective guidance, but approached only within 4.868 units of the
decoy. That missed the original 4.0-unit stale-arrival threshold, so no stale
event fired and no yellow was collected. A later target change resulted only
from distance re-ranking. This run is retained as a failed calibration and is
not counted as confirmation.

The repair derives typed arrival tolerance from representation resolution:
`max(legacy_arrival_radius, cell_size / 2)`. With 12-unit memory cells, the
typed radius is 6 units. It is not fitted to the decoy answer or target color.

## Seed 168 — untouched confirmation

- Normal bounded completion: 566 rows, 119.389 seconds, steps 0–565.
- Red pickup: 2.96 seconds.
- Decoy stale detection: 6.36 seconds, 3.396 seconds after red.
- The decoy persisted exactly one failure and was suppressed.
- Genuine yellow `(116.8036, -174.3526)` selected in the same frame, while
  yellow remained invisible.
- From fallback selection to visibility: 54 frames, 53 memory-active frames,
  44 bounded-guidance frames, and 33 additional protective MPC action changes.
- Genuine yellow entered perception at 17.61 seconds and was collected at
  `(116.67, -175.24)` at 22.26 seconds.
- Red-to-antidote latency: 19.299 seconds.
- One pending hazard cancelled; zero hazard-cost events, survival failures, or
  respawns. Two stuck events recovered normally.

The frozen source memory remained unchanged at SHA-256
`a47ecc470aff22d6cbb736ebdbc4e32e6e21d8d601b3be8ca62beba7e8df38fe`.

## Verdict

The calibrated seed is a pass for the complete chain: false typed recall,
counterfactual rejection, persistent failure evidence, same-frame fallback,
pre-visibility PGNW/MPC influence, perceptual handoff, physical pickup at the
genuine remembered region, and metabolic cancellation. This is one untouched
confirmation after one explicitly reported calibration failure; broader
reliability remains unmeasured.
