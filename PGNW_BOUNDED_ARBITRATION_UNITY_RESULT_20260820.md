# Bounded PGNW Multi-Hypothesis Unity Authority — Seed 170

Seed 170 tested whether the live multi-hypothesis arbiter could safely own the
existing committed PGNW target after passing preregistered verification gates.
It used the same frozen start geometry, causal posterior, terrain memory,
240-second hazard deadline, and controller bounds as the passive seed-169 run.

## Result

- Normal bounded completion: 571 rows, 119.385 seconds, steps 0–570.
- Red pickup and first arbitration evaluation: 2.95 seconds.
- Arbitration active and authoritative: 90/90 frames, steps 14–103.
- Selected feature: yellow on 90/90 authoritative frames.
- Agreement with the existing verified-protective target: 90/90.
- Expected yellow suppression probability: 0.916019.
- Score margin over blue: 0.058344–0.081224, above the 0.02 gate.
- Arbitration-sourced MPC action changes: 58.
- Physical yellow pickup: 21.95 seconds, 19.00 seconds after red.
- Additional yellow pickups: 80.80 and 113.49 seconds.
- One pending hazard cancelled; zero hazard-cost events, stuck events, survival
  failures, or respawns.

All preregistered bounded-authority checks passed. The result demonstrates that
the arbiter can progress from passive recommendation to safety-gated target
ownership and materially influence MPC while retaining the verified physical
outcome and controller safety.

Post-run verification passed 77/77 focused arbitration, planner, and embodiment
tests; the full repository suite passed 301/301 in 6.832 seconds.

## Claim boundary

The arbitration winner agreed with the prior verified-protective controller on
every active frame. Therefore the 58 action changes are changes relative to
unguided MPC that were operationally sourced through the arbitration-owned
target; they are not evidence that arbitration chose a better action than the
old verified controller. A stronger test requires at least two independently
verified actionable rules that can recommend different safe targets.
