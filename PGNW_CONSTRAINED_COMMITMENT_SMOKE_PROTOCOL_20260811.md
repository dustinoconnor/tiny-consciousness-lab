# PGNW Constrained-Commitment Smoke — 11 August 2026

## Purpose

Test whether PGNW can exert measurable experiment-seeking authority without
overriding ordinary MPC safety. This replaces the failed additive-weight
manipulation; causal inference and evidence validation remain frozen.

## Authority contract

When the requested color is visible and all existing PGNW safety gates allow
intervention:

1. ordinary MPC and all upstream controllers score actions normally;
2. collision-masked actions remain inadmissible;
3. retain only finite actions within `0.18` score units of the ordinary MPC
   optimum;
4. select the retained action most aligned with the requested target;
5. apply authority for at most three seconds, followed by a two-second cooldown.

Fallback, physical stuck state, hunger at or above 0.92, guided AIR, and guided
resource memory still veto PGNW. PGNW cannot activate MPC by itself.

## Ten-minute manipulation criterion

Pass only if the recording contains:

- at least five changed MPC selections before causal convergence;
- at least one requested-color pickup following a commitment interval;
- no survival failure attributable to PGNW;
- no collision-masked action selected by commitment.

Posterior success without changed actions does not pass. A passing smoke permits
a later matched committed-versus-passive batch; it does not itself establish a
learning or navigation benefit.
