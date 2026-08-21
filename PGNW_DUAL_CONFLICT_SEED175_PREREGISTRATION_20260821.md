# PGNW Dual-Conflict Seed 175 Preregistration

Seed 175 is the geometry-controlled confirmation of the motor-path mechanism
validated in seed 174. It changes no causal rule, posterior, arbitration score,
guidance weight, MPC regret bound, metabolic effect, or safety gate.

## Frozen physical layout

The saved `ChallengeTerrain` scene has SHA-256
`64efbc407503150f707540e20f4b6ab90179a795229ee9221d19340765cd30dd`.
Its three root test objects are:

- red: `(-5.62, 4.53)` in the horizontal X/Z plane;
- blue: `(-12.0, 6.918891)`;
- yellow: `(-1.06, 9.86)`.

Blue and yellow are 6.81 m and 7.01 m from red, respectively, separated by
11.33 m and approximately 110 degrees. The resource fixture gives blue and
yellow one reward and zero failures each. It encodes location only and is
balanced in confidence; it contains no causal outcome or preferred answer.

The run starts from the saved robot pose and deliberately omits diagnostic
teleportation. Frozen held-out verified interaction and metabolic memories are
loaded exactly as before. The bound is 90 seconds with seed 175, initial hunger
0.92, 240-second hazard delay, and the original 0.30 MPC score-regret limit.

## Registered result

Success requires this complete temporal sequence:

1. red is physically collected and opens the pending hazard;
2. verified dual arbitration selects blue and causes at least one MPC action
   change before any yellow contact;
3. blue is physically collected before yellow and relieves hunger;
4. while the original hazard remains pending, verified yellow becomes the
   active target and causes at least one post-blue MPC action change;
5. yellow is physically collected and cancels that hazard before the deadline;
6. no hazard cost, survival failure, or respawn occurs.

Authority telemetry without the complete physical sequence is not a pass.
