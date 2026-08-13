# PGNW Active-versus-Passive Seed-141 Pilot — 10 August 2026

## Verdict

**The diagnostic pair favors bounded epistemic guidance, but one pair is not a
causal learning-speed result.** Bounded PGNW reached the registered 0.95
posterior threshold at 863.97 seconds. Passive PGNW did not reach the threshold
during its 1,194.35 recorded seconds and ended at 0.93265. The observed bounded
lead is therefore greater than 330.38 seconds (5.51 minutes), with the passive
time right-censored.

Only one bounded MPC selection changed, while the separately executed Unity
trajectories differed substantially. Real-time physics and trajectory
variability therefore remain a credible explanation for much of the apparent
effect. Additional counterbalanced seed pairs are required.

## Frozen-metric comparison

| Metric | Bounded | Passive | Pilot difference |
|---|---:|---:|---:|
| Time to `P(red cause) >= 0.95` | 863.97 s | Not reached by 1,194.35 s | Bounded >330.38 s earlier |
| Final red-cause posterior | 0.999134 | 0.932651 | +0.066483 |
| Final posterior entropy | 0.01014 bits | 0.36806 bits | -0.35792 bits |
| Mean entropy over run | 0.79257 bits | 1.20123 bits | 34.02% lower bounded |
| Clean experiments | 11 | 9 | +2 bounded |
| Clean experiments/hour | 33.18 | 27.13 | 22.31% higher bounded |
| First clean red result | 301.32 s | 283.26 s | Passive 18.07 s earlier |
| First clean blue result | 159.61 s | 416.38 s | Bounded 256.77 s earlier |
| Discarded/started | 10/21 (47.62%) | 9/18 (50.00%) | Similar |
| Request compliance | 8/21 (38.10%) | 6/18 (33.33%) | +4.77 points bounded |
| All pickups | 32 | 28 | +4 bounded |
| Red pickups | 6 | 3 | +3 bounded |
| Guidance decisions | 96 | 0 | As designed |
| Changed MPC selections | 1 | 0 | +1 bounded |
| Survival failures | 0 | 0 | Equal |
| Stuck events | 1 | 9 | -8 bounded |
| Critical-hunger exposure | 0 s | 0 s | Equal |

Both runs used controller seed 141, bounded recurrent MPC, bounded adaptive GNW,
the same systemic conductor configuration, the same 1,200-second requested
duration, and the same Unity scene. The experimental-control mode was the only
intended condition change. Telemetry confirms structurally zero guidance in
passive mode and a maximum effective bounded weight of 0.02961.

## What produced the posterior difference

Bounded obtained four clean red-positive and seven clean blue-negative results.
Passive obtained one clean red-positive and eight clean blue-negative results.
Both inferred `red_causes_probe`, but passive lacked enough clean red evidence
to cross the registered threshold or strongly eliminate `no_tested_cause`.

The first clean red result actually arrived earlier under passive control. The
bounded advantage arose later because it encountered additional red evidence,
not because every stage of learning was uniformly faster.

## Interpretation boundary

This pair demonstrates that passive wandering can discover the correct MAP
hypothesis, but it did not reach the registered confidence threshold within the
run. It provides a positive pilot signal for bounded active epistemic guidance.
It does not establish that guidance caused the difference: the active condition
changed only one discrete MPC choice, and the large stuck/pickup divergence
shows that independent Unity runs can follow very different trajectories.

Freeze the pilot and continue with counterbalanced pairs. To reverse the order
used for seed 141, run passive then bounded for seed 142; alternate order again
for later seeds. Evaluate the paired threshold times with censoring, entropy
area, clean evidence rate, request compliance, and safety. Do not increase the
0.03 authority based on this result.
