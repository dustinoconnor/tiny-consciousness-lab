# PGNW Dual-Conflict Seed 174 Result

Seed 174 tested the critical-hunger guidance-gate repair with the original 0.30
MPC score-regret bound and otherwise frozen seed-172 protocol.

## Outcome

- Normal completion: 1,138 rows over 239.468 seconds.
- Red pickup: 2.773 seconds at hunger 0.934.
- Verified blue authority: 91 frames before the first competing pickup.
- Blue-sourced pre-pickup MPC action changes: 42.
- Yellow pickup and hazard cancellation: 21.903 seconds.
- Blue pickup: 24.874 seconds, 2.971 seconds after yellow.
- One recovered stuck event; zero hazard costs, survival failures, or respawns.

The repair succeeded at its narrow mechanistic target: verified blue authority
was no longer inert at critical hunger. It changed 42 MPC decisions, compared
with zero in both prior runs.

The complete preregistered sequence nevertheless failed. One frame before the
yellow contact, blue remained selected with full authority and was 9.45 m away,
while a visible yellow was only 2.58 m away. Contact cancelled the hazard before
blue relief, preventing the required subsequent transfer of authority to yellow.

## Claim boundary

Verdict: `partial_mechanism_pass_physical_sequence_fail`. The run demonstrates
causal motor influence from the verified metabolic rule, not successful
blue-then-yellow multi-hypothesis sequencing. A fresh confirmation needs
preregistered separated target headings so incidental contact cannot resolve the
conflict, while retaining the frozen rules and controller parameters.
