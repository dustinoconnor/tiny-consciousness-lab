# PGNW Dual-Conflict Seed 176 Preregistration

Seed 176 is a single-parameter motor calibration on the frozen seed-175 spawn
fork. It changes committed MPC max-score regret from 0.30 to 0.45, the value
registered for seed 173 but never behaviorally exercised because the upstream
critical-hunger gate blocked that run.

The saved scene, balanced coordinate fixture, held-out verified causal memories,
arbitration scores, initial hunger 0.92, controller, and all safety gates remain
unchanged. The shortened 45-second bound covers the first fork episode only.

The pass criterion remains the full physical sequence: red opens the hazard;
verified blue causes pre-pickup MPC influence and is collected before yellow;
hunger relief transfers the still-pending need to verified yellow; yellow causes
post-blue MPC influence and cancels the same hazard; and no hazard cost, survival
failure, or respawn occurs. Authority or eventual collection alone is not a pass.
