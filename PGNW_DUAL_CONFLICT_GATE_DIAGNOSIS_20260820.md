# PGNW Dual-Conflict Gate Diagnosis — Seed 173

Seed 173 repeated the live verified yellow-versus-blue conflict with committed
MPC max-score regret increased from 0.30 to 0.45. It failed identically to seed
172 and exposed the actual control-path defect.

## Result

- Normal completion: 1,137 rows, 239.452 seconds.
- Red pickup: 2.781 seconds at hunger 0.934.
- Verified blue held arbitration authority for 84 frames.
- Arbitration-sourced MPC action changes before yellow: 0.
- Yellow pickup and hazard cancellation: 20.541 seconds.
- Blue pickup: 24.344 seconds, 3.803 seconds after yellow.
- Zero stuck events, survival failures, or respawns.

Increasing the score-regret allowance had exactly zero effect. Inspection found
that `pgnw_experiment_guidance_allowed` rejected all PGNW motor guidance at
hunger >=0.92. The conflict protocol intentionally initialized hunger at 0.92,
so the verified blue rule could win symbolic arbitration while its committed
action was structurally blocked.

## Repair

Critical hunger may now bypass only the hunger gate when bounded dual arbitration
has authority, the selected feature exactly matches the held-out verified
nutrient, and the metabolic learner remains verified. Fallback, stuck, AIR,
resource-controller, collision-mask, and MPC-regret constraints remain active.
The confirmation returns max-score regret to the original 0.30 so the gate
repair—not the failed strength calibration—is the only operative change from
seed 172.
