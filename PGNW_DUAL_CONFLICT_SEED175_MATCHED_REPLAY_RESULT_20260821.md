# PGNW Dual-Conflict Seed 175 Matched Replay Result

The targeted seed-175 replay combined the 0.45 constrained MPC bound with
pickup-bound candidate retention on the frozen spawn fork.

## Outcome

- Normal completion: 212 rows over 44.375 seconds.
- Red pickup: 2.515 seconds at hunger 0.933.
- Blue remained selected until its pickup at 4.624 seconds; hunger fell to
  0.603 and control transferred to yellow on that frame.
- Blue-sourced counterfactual MPC changes: zero.
- Yellow-sourced post-blue MPC changes: six.
- Yellow pickup and hazard cancellation: 7.609 seconds, 2.985 seconds after
  blue.
- Zero hazard costs, stuck events, survival failures, or respawns.

## Verdict

`physical_and_transfer_pass_conjunction_fail`. The complete safe physical order
and pickup-bound transfer replicated, but the replay did not reproduce the
original seed-175 blue action-influence count. Controller seed alone therefore
does not freeze the live asynchronous Unity trajectory tightly enough to serve
as a deterministic matched replay.

Further seed search would select on the desired metric. The appropriate causal
test is now a preregistered active-versus-passive comparison over multiple runs,
using blue-first rate and trajectory outcomes rather than requiring every
individual active run to disagree with an already-aligned counterfactual action.
