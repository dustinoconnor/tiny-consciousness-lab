# Bounded Active-PGNW Unity Protocol — 10 August 2026

## Purpose

Test whether the formal PGNW selector can gather its own red, blue, and
no-pickup observations in Unity while leaving motor execution, survival, and
recovery authority with the existing controller stack.

## Authority contract

PGNW emits one capacity-one request:

- `observe_red_pickup`;
- `observe_blue_pickup`;
- `wait_no_pickup`.

It never emits a motor command. In `bounded` mode, a visible requested color
provides at most `0.03` alignment preference to already-engaged, ambiguous MPC
scores. The preference cannot engage MPC by itself and is disabled whenever:

- stable fallback is active;
- the robot is physically stuck;
- hunger is at least `0.92`;
- guided AIR routing is active;
- guided resource-memory routing is active;
- the requested target is not visible.

Collision masking remains absolute. Ordinary recurrent/MPC specialists,
systemic routing, adaptive GNW, AIR, resource memory, and stable recovery keep
their existing authority.

## Evidence contract

After a requested red or blue pickup, the planner waits ten seconds and reads
the passive `causal_probe_signal`. A no-pickup request begins a matched wait
window. Any intervening pickup marks the trial confounded and prevents a
posterior update. An encountered non-requested color is logged as a protocol
mismatch and analyzed according to the experiment actually performed.

The planner updates the same five-model posterior used in the synthetic PGNW
audit, then broadcasts the next experiment. The passive probe itself has zero
physiological, workspace, routing, MPC, and motor influence.

## Initial smoke criteria

A bounded Unity smoke is feasible if it completes at least two uncontaminated
experiments, performs the corresponding posterior updates, records all
guidance decisions and changed MPC choices, and produces no new safety or
survival failure attributable to scientific targeting. One run cannot
establish navigation benefit; a matched passive/bounded comparison is required
after feasibility.

## Boundary

This grants experiment-selection authority and a small navigational preference,
not direct motor control, general rule authority, conscious agency, or
autonomous natural-world science.
