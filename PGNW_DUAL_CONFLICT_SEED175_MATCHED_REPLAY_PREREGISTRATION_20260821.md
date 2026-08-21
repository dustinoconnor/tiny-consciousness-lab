# PGNW Dual-Conflict Seed 175 Matched Replay Preregistration

This is a targeted mechanistic replay of controller seed 175, not a new
held-out seed. It reuses the frozen spawn fork and seed-175 stochastic rollout
because the original seed produced nonzero blue-sourced MPC action changes.

Relative to the original seed-175 failure, the replay combines two changes that
were evaluated separately afterward:

1. committed MPC max-score regret is 0.45 rather than 0.30; seed 176 then
   achieved the safe red -> blue -> yellow physical order;
2. a visible selected target remains eligible until pickup telemetry; seed 177
   then confirmed pickup-bound blue-to-yellow transfer.

All causal memories, scoring, needs, effects, scene geometry, and safety gates
remain frozen. The 45-second replay uses fresh artifact paths.

A complete conjunction pass requires red -> blue -> yellow physical order;
nonzero blue-sourced MPC influence before blue; blue remains selected through
the last pre-pickup frame; transfer to yellow occurs no earlier than measured
blue pickup/relief; nonzero yellow-sourced post-blue MPC influence; cancellation
of the same hazard; and no hazard cost, survival failure, or respawn.

Even if it passes, the claim is mechanistic conjunction on a deliberately
replayed seed. Generalization still requires fresh counterbalanced seeds.
