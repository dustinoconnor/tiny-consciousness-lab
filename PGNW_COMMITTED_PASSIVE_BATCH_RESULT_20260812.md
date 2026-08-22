# PGNW Committed-versus-Passive Batch Result — 12 August 2026

## Registered comparison

Three matched 20-minute pairs used controller seeds 150-152. Order was
counterbalanced. Every run began at the same floor-state position
`[0.0, -2.8556, 0.0]` with zero pickup, red-pickup, probe-event, and metabolic-
challenge counters. All six processes returned successfully and supplied the
full requested duration.

The primary endpoint was elapsed time until the posterior probability of
`red_causes_probe` first reached 0.95. Runs that did not reach threshold were
right-censored at 1,200 seconds.

## Primary result

| Seed | Passive threshold | Committed threshold | Committed advantage |
|---:|---:|---:|---:|
| 150 | not reached | 930.8 s | at least 269.2 s |
| 151 | 1,179.8 s | 556.9 s | 623.0 s |
| 152 | not reached | 411.5 s | at least 788.5 s |

Committed guidance reached the registered threshold in 3/3 runs. Passive
guidance reached it in 1/3, only 20.2 seconds before the limit. Treating missed
runs as 1,200 seconds, mean threshold time was 633.1 seconds committed versus
1,193.3 seconds passive: a 560.2-second (46.9%) reduction in bounded time.

All three matched-pair differences favored committed guidance. With only three
pairs this is a strong manipulation check, not a stable population estimate or
a claim of statistical significance.

## Secondary evidence

| Metric, summed unless noted | Passive | Committed |
|---|---:|---:|
| Final red-cause posterior, mean | 0.617184 | 0.996598 |
| Clean red observations | 2 | 8 |
| Clean blue observations | 29 | 45 |
| Completed observations | 31 | 53 |
| Discarded observations | 30 | 20 |
| Discard fraction | 49.2% | 27.4% |
| Time-averaged posterior entropy, mean | 1.140 bits | 0.718 bits |
| PGNW action changes | 0 | 628 |
| Stuck events | 17 | 21 |
| Critical-hunger exposure | 0 s | 0 s |
| Survival failures | 0 | 0 |

The committed controller changed actions before the first clean red completion
in every committed run (86, 147, and 36 cumulative changes for seeds 150, 151,
and 152). This rules out the trivial explanation that the controller only began
influencing behavior after the causal rule had already been learned. It also
produced four times as many clean red observations and reduced the discarded-
observation fraction by 21.8 percentage points.

Protocol-mismatch rates were high in both conditions because ordinary foraging
could still encounter the non-requested color: 49/62 starts passive (79.0%) and
56/73 committed (76.7%). Commitment did not solve target compliance outright;
its benefit came from increasing usable evidence acquisition and protecting
observation windows.

## Safety and limitations

No run incurred critical hunger or a survival failure. Committed mode did have
four more stuck events across the batch (21 versus 17), so the intervention is
not behaviorally free even though the registered severe-safety outcomes stayed
at zero.

The result supports the bounded claim that safety-gated PGNW commitment made
causal learning faster and more reliable in this fixed Unity scene and seed
range. It does not yet establish generalization across scenes, mushroom layouts,
or a larger seed population. The immediate-retreat logic is bundled with the
committed condition, so this batch identifies the complete committed controller
as effective; it does not isolate commitment from retreat.

## Decision

The manipulation passed. A larger replication is warranted if the next goal is
publication-quality estimation. The most informative follow-up is a new-seed,
new-layout replication, ideally with commitment and retreat separated as a
small factorial ablation. No controller tuning should occur on seeds 150-152.
