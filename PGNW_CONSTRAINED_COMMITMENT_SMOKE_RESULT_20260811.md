# PGNW Constrained-Commitment Smoke Result — 11 August 2026

## Verdict

**Partial manipulation success; preregistered smoke failure. Do not run an
overnight batch yet.** Constrained commitment changed two MPC selections and a
requested red pickup followed the second change by 0.21 seconds. This is the
first run in which PGNW experiment authority both reached motor selection and
was immediately followed by acquisition of the requested target.

The registered threshold required at least five changed selections, so two is
insufficient. More importantly, a blue pickup occurred 1.50 seconds after the
red pickup and contaminated its ten-second observation window. The remaining
bottleneck is no longer only target acquisition; the agent must preserve a
clean post-intervention observation interval.

## Run metrics

- Seed: 147
- Recorded duration: 593.14 seconds
- Start position: `(0, -0.008, 0)`
- Mode: `committed`
- Score-regret bound: 0.18
- Guidance decisions: 27
- Changed MPC selections: 2
- Pre-convergence changed selections: 2
- Commitment starts: 2
- Total pickups: 20
- Red pickups: 2
- Clean experiments: 5
- Discarded experiments: 7
- Stuck events: 2
- Critical-hunger exposure: 0 seconds
- Survival failures: 0

## Linked intervention

The second commitment began at 514.35 seconds with red requested and both
colors visible. PGNW changed the MPC selection at 514.35 and 517.14 seconds.
The robot picked up red at 517.35 seconds, 0.21 seconds after the second changed
selection. This temporal link is consistent with successful target steering,
but one event cannot establish causation.

At 518.85 seconds the robot picked up blue. The red observation was therefore
discarded as confounded. The earlier red pickup at 2.97 seconds was also
followed by blue at 4.03 seconds and was correctly discarded.

## Posterior and safety

Five clean blue-negative observations and zero clean red observations left
`no_tested_cause` as MAP with red-cause posterior 0.45412. That is the expected
bounded response to insufficient diagnostic evidence, not an inference defect.

No collision or survival failure was attributed to commitment. The
score-regret constraint and existing vetoes preserved safety in this smoke.

## Next manipulation

Freeze the constrained target-acquisition rule. Add a ten-second,
safety-gated observation-isolation phase after the experimental pickup:

- suppress optional food-seeking toward any visible mushroom;
- prefer safe movement away from visible food, or idle when safely clear;
- yield immediately to collision avoidance, fallback, stuck recovery, and
  critical hunger;
- log isolation decisions, changed actions, intervening pickups, and releases.

Run another short smoke and require both a request-linked pickup and one clean
completed red observation. Only then is a matched committed-versus-passive
overnight batch warranted.
