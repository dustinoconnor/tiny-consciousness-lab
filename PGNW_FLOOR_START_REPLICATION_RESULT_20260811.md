# PGNW Floor-Start Replication Result — 11 August 2026

## Verdict

**The current bounded PGNW guidance does not reliably accelerate learning or
reduce its worst-case variability.** In the internally matched three-pair
floor-start replication, passive reached the registered posterior threshold in
two of three runs and bounded in one of three. Passive won two paired
threshold-time comparisons; bounded won one.

The useful result is a localized bottleneck: every condition that obtained at
least two uncontaminated red observations learned the correct rule, and every
condition with zero clean red observations failed. Online causal updating works
when diagnostic evidence arrives. The `0.03` guidance does not consistently
cause the agent to acquire that evidence.

## Batch validity

All six subprocesses returned zero and produced approximately 1,200 seconds of
telemetry. Each began with zero total and red pickup counters and the same
floor-level recorded position `(0, -2.856, 0)`. The three pairs are therefore
internally matched as a floor-start replication block, although they are not
equivalent to the earlier command-first Unity starts recorded near `y = 0`.

## Paired results

| Seed | Bounded time to 0.95 | Passive time to 0.95 | Faster condition |
|---|---:|---:|---|
| 143 | 550.61 s | Censored at 1,200 s | Bounded |
| 144 | Censored at 1,200 s | 909.53 s | Passive |
| 145 | Censored at 1,200 s | 639.14 s | Passive |

| Aggregate measure | Bounded | Passive |
|---|---:|---:|
| Threshold successes | 1/3 | 2/3 |
| Correct final MAP hypothesis | 1/3 | 2/3 |
| Restricted mean threshold time | 983.54 s | 916.22 s |
| Restricted median threshold time | 1,200.00 s | 909.53 s |
| Threshold-time standard deviation | 374.92 s | 280.49 s |
| Mean run-averaged posterior entropy | 1.11089 bits | 1.13435 bits |
| Mean clean experiments/hour | 42.01 | 36.00 |
| Total stuck events | 23 | 25 |
| Survival failures | 0 | 0 |

Bounded was descriptively 67.31 seconds slower on the 1,200-second-capped paired
mean and more variable in this replication. Its small entropy advantage did not
translate into more threshold successes.

## Evidence-acquisition diagnosis

| Seed/condition | Clean red | Clean blue | Threshold result |
|---|---:|---:|---|
| 143 bounded | 7 | 11 | Reached at 550.61 s |
| 143 passive | 0 | 9 | Not reached |
| 144 bounded | 0 | 10 | Not reached |
| 144 passive | 2 | 16 | Reached at 909.53 s |
| 145 bounded | 0 | 14 | Not reached |
| 145 passive | 3 | 6 | Reached at 639.14 s |

The separation is exact in this block: `clean red >= 2` predicts all three
successes, while `clean red = 0` predicts all three failures. Failed conditions
ended with `no_tested_cause` as the MAP hypothesis because blue-only evidence
cannot distinguish a red cause from no tested cause.

Bounded generated only two changed MPC selections across all three runs. Both
occurred in seed 144 near 967 seconds; that run still obtained zero clean red
observations and failed. Bounded seed 143's apparent 649-second win occurred
with zero changed MPC selections. It therefore cannot be attributed to the
registered discrete guidance intervention.

## Cross-block interpretation

The earlier four-pair interim analysis also split evenly, two wins per
condition. This independent floor-start replication now favors passive two to
one. Taken together, the evidence does not support raw acceleration, variance
regulation, or a worst-case floor guarantee at `0.03` authority. Both
conditions can learn, and both can fail when wandering does not yield clean red
evidence.

## Decision

Do not spend additional runs on this unchanged configuration. The next
experiment should target evidence acquisition directly and preregister a
meaningful manipulation: for example, stronger but still safety-gated
experiment authority, controlled red-target availability, or a planner that
can commit to a requested target for a bounded interval. Any new authority must
be compared against passive control with action influence verified before the
diagnostic observation. The current causal updater and contamination rejection
should remain frozen.
