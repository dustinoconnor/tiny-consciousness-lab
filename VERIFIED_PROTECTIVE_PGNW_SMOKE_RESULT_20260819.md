# Verified Protective PGNW Smoke Result — Seed 162 — 2026-08-19

## Verdict

The 180-second protective plumbing smoke passed. After red scheduled the
bounded hazard, the >=0.95 verified typed rule became active, selected yellow,
changed five committed-MPC choices, and reached yellow before the deadline.
Yellow cancelled the event and the configured 0.25 hunger cost did not fire.
This establishes rule-to-planner influence in one physical run. A matched
authority-zero counterpart is still required for a comparative efficacy claim.

## Sequence

- Telemetry contains one continuous 854-record session, steps 0-853, spanning
  179.724 seconds.
- The read-only source began at 97.6319% posterior confidence for
  `yellow_after_red_suppresses_probe`.
- Before red, protective guidance was inactive.
- Red was collected at step 13, 2.731 seconds after the first record, scheduling
  one delayed event.
- Protective PGNW was active for steps 13-22, exactly ten frames. Yellow was
  visible on all ten frames and was the sole protective target.
- The committed-MPC path made ten protective guidance decisions and changed
  five selections relative to unguided MPC.
- Yellow was collected at step 23, 2.144 seconds after red, and immediately
  cancelled the pending event.

## Outcome and safety

- Pending events: peak 1, final 0.
- Cancelled events: 1.
- Completed probe events: 0.
- Hazard-cost events and total hunger cost: 0 and 0.0.
- Final pickups: 1 red, 7 blue, and 2 yellow.
- The clean episode later updated the typed posterior to 98.0562% and remained
  classified as suppressed.
- Survival failures, critical-hunger seconds, and respawns: all 0.
- One stuck event occurred later in the run, outside the ten-frame protective
  interval.

## Boundary

The five pre-yellow MPC changes establish that the verified rule affected the
trajectory; the coincident cancellation establishes the desired physical
outcome. One run cannot estimate how often that influence is necessary or
beneficial. The next defensible comparison restarts the same Unity scene and
controller seed with identical hazard dynamics but passive typed-rule/PGNW
authority.

This opportunity was local and staged, not terrain-wide retrieval. Yellow was
already visible on the first telemetry frame at distance 13.65 and remained
visible when red was collected, at distance 7.67. Protective control reduced
that distance to 2.30 before pickup. A second yellow pickup at step 636 occurred
with protective control inactive. The run therefore demonstrates selection of
a visible antidote, not search for an unseen flower or recall of a distant
yellow resource location.
