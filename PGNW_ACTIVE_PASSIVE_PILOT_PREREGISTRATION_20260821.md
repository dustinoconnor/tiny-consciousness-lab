# PGNW Active-versus-Passive Pilot Preregistration

This pilot tests whether bounded dual-rule PGNW authority changes the first
post-red resource choice on the frozen spawn fork. It adds no cognitive module.

## Conditions

- **Active:** `bounded_dual_verified`; independently verified metabolic and
  protective rules may arbitrate and receive bounded MPC authority.
- **Passive:** `passive`; the same candidates and scores are observed, but dual
  arbitration receives zero motor authority. Existing verified yellow
  protection remains available in both conditions.

Thus this is specifically an assay of dual-arbitration authority, not a claim
that every PGNW-related process is disabled in the passive condition.

## Frozen design

- Four trials per condition, 20 seconds each.
- Paired controller seeds 181, 182, 183, and 184.
- Counterbalanced execution order: `A1, P1, P2, A2, A3, P3, P4, A4`.
- Saved scene SHA-256:
  `64efbc407503150f707540e20f4b6ab90179a795229ee9221d19340765cd30dd`.
- Same balanced fork fixture, verified memories, initial hunger 0.92, 0.45 MPC
  score-regret bound, 240-second hazard delay, controllers, and safety gates.
- Unity Play mode must be stopped and restarted between trials so physical
  objects and robot pose reset. No scene or parameter changes are allowed after
  the first trial until all eight terminate.

## Outcomes

The primary outcome is whether the first blue/yellow pickup after the first red
pickup is blue. Missing red or no subsequent competing pickup within the bound
counts as not blue-first rather than being excluded. A same-frame tie counts as
not blue-first.

Secondary outcomes are complete red -> blue -> yellow order, pickup latencies,
blue arbitration action influence, yellow protective action influence, hazard
cancellation, costs, stuck events, survival failures, and respawns.

The frozen analyzer reports condition counts and a two-sided Fisher exact test
for blue-first. With four trials per condition, a 4/4 active versus 0/4 passive
split yields `p = 0.028571`; weaker splits are descriptive pilot evidence. No
trial or seed will be replaced based on its result.
