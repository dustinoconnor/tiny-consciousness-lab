# PGNW Dual-Conflict Seed 175 Result

Seed 175 used the preregistered spawn fork, balanced location fixture, frozen
verified rules, and original 0.30 MPC score-regret bound.

## Outcome

- Normal completion: 429 rows over 89.927 seconds.
- First red pickup: 2.519 seconds at hunger 0.933.
- Verified blue immediately won with full authority.
- Blue authority changed nine MPC decisions before the first competing pickup.
- Yellow was collected first at 5.481 seconds and cancelled the hazard.
- Blue followed at 8.835 seconds, 3.354 seconds after yellow.
- The same yellow-before-blue order repeated after a second red episode.
- Zero stuck events, hazard costs, survival failures, or respawns.

At the first red pickup, remembered blue was 9.03 m away and yellow was 8.57 m
away. The blue guidance vector required a substantial leftward component, but
the 0.30 constrained MPC selector chose forward actions. By step 20, blue was
almost directly left at 7.58 m while yellow was 4.30 m ahead/right; the selected
action remained forward. Blue authority was therefore causally operative but
insufficient to admit the turn needed to realize its selected target.

## Verdict

`motor_authority_calibration_fail`. The controlled geometry rules out the
seed-174 route-overlap explanation, but the complete blue-then-yellow sequence
still did not occur. The result supports symbolic selection plus bounded motor
influence, not successful multi-hypothesis execution.
