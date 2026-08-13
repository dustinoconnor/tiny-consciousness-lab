# PGNW Stronger-Authority Manipulation Smoke — 11 August 2026

## Purpose

Determine whether stronger but still subordinate PGNW guidance actually changes
navigation before diagnostic evidence is acquired. This is a manipulation
check, not an active-versus-passive learning claim.

## Frozen components

Keep the five causal hypotheses, EFE experiment selector, ten-second delayed
probe, posterior update, contamination rejection, controller checkpoints,
systemic conductor, adaptive GNW, and all survival/recovery gates unchanged.

## Manipulation

- Increase maximum experiment-guidance weight from `0.03` to `0.12`.
- Increase the eligible MPC ambiguity margin from `0.08` to `0.25`.
- Continue targeting the selected red or blue experiment until pickup or
  experiment reselection; the planner already preserves this request across
  frames.
- Guidance remains restricted to already-engaged MPC and cannot override
  collision masking, stable fallback, stuck recovery, hunger at or above 0.92,
  guided AIR, or guided resource memory.

## Ten-minute smoke criterion

The manipulation is behaviorally present if the run logs at least five changed
MPC selections before causal convergence and at least one requested-color
pickup after a guidance interval, with no survival failure attributable to
PGNW. Failure means the authority mechanism must be redesigned before another
matched batch; posterior success alone is insufficient.

## Claim boundary

A passing smoke establishes only that the manipulation reaches motor selection
under safety gates. It does not establish faster learning, better navigation,
or autonomous scientific competence.
