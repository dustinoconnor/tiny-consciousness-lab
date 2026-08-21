# PGNW Counterbalanced Fork Protocol

## Question

Does bounded dual-rule PGNW authority improve the first post-red resource
choice when blue and yellow are presented at equal distance and their left/right
positions are counterbalanced?

This follows the 4/4-active versus 4/4-passive null pilot. That earlier scene is
not reused as evidence because its passive controller already chose blue in
every trial.

## Layout mechanism

`PGNWCounterbalancedForkTools.cs` binds the nearest standalone red, blue, and
yellow spawn-area pickups under a dedicated `PGNW Counterbalanced Fork` root.
It never moves objects under a `Scattered ...` population root. Four editor menu
actions create two calibration and two reserved layouts:

- calibration, blue left;
- calibration, yellow left;
- reserved, blue left;
- reserved, yellow left.

Within each layout pair, blue and yellow are exact mirrored positions at equal
horizontal distance from the robot. Red remains centered and closer. Reserved
layouts use a different radius and an 18-degree rotation and must not be run
while calibration parameters are being assessed. The validation menu reports
red, blue, and yellow distances and requires a blue/yellow distance difference
of at most 0.01 m.

## Calibration gate

Run passive dual-arbitration telemetry only on both calibration layouts. The
purpose is to reject a geometry that remains deterministically blue, not to
estimate an active effect. Use paired seeds and balance which color is left.

The corrected calibration order is `CBL1R, CYL1, CYL2, CBL2`, where CBL means blue
left and CYL means yellow left. CBL1/CYL1 use controller seed 185 and
CYL2/CBL2 use seed 186. Each run is bounded to 20 seconds and Unity Play mode is
stopped and restarted between runs. The saved CBL scene SHA-256 is
`9a05ffe2a692428a15480761a04e077e77d8c9ad15ee366379674a5ea58bffe6`.
The CYL hash is sealed until that mirror layout is applied, visually checked,
and saved. It is now frozen at
`7fcdfab55c16260638f1f5812a7b8ade563ec5c126c976a4274c25ed815a3897`,
with blue `(3.50, 5.43)` and yellow `(-3.50, 5.43)` in the matching resource
fixture.
The original CBL1 is retained as a rejected setup check because red was already
consumed on telemetry frame zero; CBL1R is its non-overwriting corrected run.

The calibration geometry is acceptable only if:

- both mirror assignments complete red followed by at least one competing
  pickup;
- passive does not choose blue first in every run or yellow first in every run;
- neither left nor right is selected on every run; and
- there are no systematic accessibility, collision, or pickup failures.

If this gate fails, revise only calibration geometry and repeat calibration.
Do not inspect or modify the reserved geometry based on reserved outcomes.

## Reserved comparison

After calibration passes, freeze the tool, controller parameters, causal and
metabolic memories, scoring, safety gates, duration, and analysis. Run active
and passive conditions on both reserved mirror assignments using paired seeds
and counterbalanced condition order.

- Active: `bounded_dual_verified`.
- Passive control: `passive`, which computes identical candidates and scores
  but gives dual arbitration zero authority.
- Primary outcome: blue is the first blue/yellow pickup after red.
- A missing red, missing competing pickup, or same-frame tie counts as a
  primary failure rather than being excluded.
- Secondary outcomes: pickup timing, active action influence before pickup,
  yellow protective influence, hazard cancellation, stuck events, costs,
  survival failures, and respawns.

Analyze paired discordance as well as condition rates and left/right balance.
Do not replace trials or search seeds after observing outcomes. A result can
support added PGNW choice benefit only if active improves the paired outcome
and its authority reaches the motor path; a layout with zero active action
influence remains nondiagnostic even if raw choices differ.

## Preserved control boundary

The word `passive` still appears in several current components because it means
zero behavioral authority, not obsolete code. The PGNW passive condition is the
necessary causal control and is retained. The legacy live synchrony and
criticality proxy observers are removed separately and play no role in this
protocol.
