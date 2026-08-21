# PGNW Dual-Conflict Seed 176 Result

Seed 176 tested the frozen spawn fork with committed MPC max-score regret 0.45.

## Outcome

- Normal completion: 215 rows over 44.908 seconds.
- Red pickup: 2.516 seconds at hunger 0.933.
- Blue held full verified arbitration authority for nine frames.
- Arbitration changed zero pre-blue MPC actions because the unguided MPC choice
  was already aligned with the blue route.
- Arbitration switched from blue to yellow at 4.409 seconds, when the blue
  memory candidate entered its arrival radius.
- Blue was physically collected at 4.621 seconds and relieved hunger to 0.603.
- Verified yellow protection then changed seven MPC actions.
- Yellow was collected at 7.585 seconds, 2.964 seconds after blue, and cancelled
  the same pending hazard.
- Zero hazard costs, stuck events, survival failures, or respawns.

## Verdict

`physical_sequence_pass_causal_preregistration_fail`. The desired safe physical
order red -> blue -> yellow occurred. It is not a complete preregistered pass
because blue produced no counterfactual motor changes and the authority switch
preceded measured blue relief by 0.212 seconds. The early switch is caused by
typed memory excluding a candidate inside its generic arrival radius before
physical pickup confirmation.

Across seeds 175 and 176, the evidence establishes that blue can win verified
arbitration, can affect MPC in a conflicting trajectory, and can precede yellow
in the controlled fork. No single run yet combines those facts with a strictly
post-relief authority transfer.
