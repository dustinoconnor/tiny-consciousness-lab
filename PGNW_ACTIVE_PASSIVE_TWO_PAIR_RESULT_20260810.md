# PGNW Active-versus-Passive Two-Pair Result — 10 August 2026

## Verdict

**The causal-discovery loop replicated, but faster active learning did not.**
Bounded PGNW reached the correct 0.95 posterior threshold in both seeds and did
so at nearly identical times. Passive PGNW was highly variable: it failed to
reach threshold in seed 141 but beat bounded by 280.93 seconds in seed 142.

With one pair favoring each condition and only two pairs total, there is no
defensible general learning-speed conclusion. The most interesting descriptive
pattern is lower bounded-condition variability, which requires more seeds and
cannot yet be attributed to epistemic guidance.

## Paired threshold result

| Seed | Bounded time to 0.95 | Passive time to 0.95 | Pair result |
|---|---:|---:|---|
| 141 | 863.97 s | Not reached by 1,194.35 s | Bounded >330.38 s earlier |
| 142 | 867.42 s | 586.49 s | Passive 280.93 s earlier |

The two bounded threshold times differ by only 3.46 seconds. Using the frozen
1,200-second censoring horizon descriptively, bounded's mean restricted time is
865.70 seconds and passive's is 893.25 seconds, a 27.55-second bounded advantage.
This two-pair descriptive average is not an inferential result.

## Seed-142 comparison

| Metric | Bounded | Passive |
|---|---:|---:|
| Time to posterior 0.95 | 867.42 s | 586.49 s |
| Final red-cause posterior | 0.998370 | 0.999968 |
| Mean posterior entropy | 1.02632 bits | 0.89205 bits |
| Clean experiments | 10 | 17 |
| Clean experiments/hour | 30.15 | 51.17 |
| Clean red / blue results | 3 / 7 | 9 / 8 |
| Discarded / started | 7 / 17 | 7 / 24 |
| Request compliance | 47.06% | 50.00% |
| Pickups / red pickups | 24 / 4 | 35 / 10 |
| Stuck events | 12 | 5 |
| Critical-hunger exposure | 39.4 s | 17.0 s |
| Survival failures | 0 | 0 |

Bounded seed 142 made 56 effective guidance decisions and changed one MPC
selection. That changed selection occurred at 1,036.66 seconds, after the
posterior had already crossed 0.95. Consequently, its threshold result cannot
be credited to the one logged discrete intervention.

## Across-pair descriptive checks

- Bounded crossed threshold in 2/2 runs; passive crossed in 1/2.
- Mean run-averaged posterior entropy was 0.90944 bits bounded versus 1.04664
  passive, 13.11% lower bounded.
- Mean clean-experiment rate was 31.67/hour bounded versus 39.15/hour passive.
- Aggregate stuck events were similar: 13 bounded versus 14 passive.
- All four runs ended with zero survival failures.
- Each bounded run changed only one discrete MPC selection.

These mixed metrics do not support a general statement that bounded guidance
accelerates evidence acquisition. They do continue to support robust online
causal identification and safe posterior updating.

## Next test

Continue the frozen alternating order with bounded then passive on seed 143.
Additional pairs should test both central tendency and tail reliability. In
particular, record whether the apparent bounded threshold-time stability
persists and whether any changed MPC selection occurs before evidence is
acquired. Do not increase authority or reinterpret unchanged decisions as motor
interventions after seeing these results.
