# PGNW Active-versus-Passive Four-Pair Interim Result — 10 August 2026

## Verdict

**The embodied causal-discovery loop replicated, but bounded PGNW has not been
shown to accelerate learning or guarantee a worst-case floor.** Among four
valid matched pairs, bounded reached the posterior threshold in three runs and
passive in two. Each condition won two paired speed comparisons. With
1,200-second censoring, bounded's paired mean advantage was only 7.38 seconds.

Bounded retained a descriptive reliability signal: lower run-averaged entropy,
lower threshold-time dispersion, and one additional threshold success. Four
pairs are far too few to distinguish a real variance effect from trajectory
noise, especially because bounded changed very few MPC selections before
learning.

## Valid paired results

| Seed | Bounded time to 0.95 | Passive time to 0.95 | Faster condition |
|---|---:|---:|---|
| 141 | 863.97 s | Censored at 1,200 s | Bounded |
| 142 | 867.42 s | 586.49 s | Passive |
| 144 | Censored at 1,200 s | 968.16 s | Passive |
| 145 | 993.76 s | Censored at 1,200 s | Bounded |

| Aggregate measure | Bounded | Passive |
|---|---:|---:|
| Threshold successes | 3/4 | 2/4 |
| Restricted mean threshold time | 981.29 s | 988.66 s |
| Restricted median threshold time | 930.59 s | 1,084.08 s |
| Threshold-time standard deviation | 157.82 s | 289.53 s |
| Mean run-averaged posterior entropy | 0.96286 bits | 1.14088 bits |
| Mean clean experiments/hour | 27.83 | 36.83 |
| Total stuck events | 27 | 34 |
| Survival failures | 0 | 0 |

The paired restricted-time differences were `-336.03`, `+280.93`, `+231.84`,
and `-206.24` seconds, where negative favors bounded. Their mean is `-7.38`
seconds and median is `+12.80` seconds. The conditions are therefore tied for
practical speed in this small interim sample despite the lower descriptive
dispersion under bounded control.

Bounded's clean-experiment rate was lower, not higher. Across these four pairs,
only two changed bounded MPC choices occurred before posterior convergence.
This sparse intervention count makes it difficult to attribute condition-level
trajectory differences to active epistemic guidance.

## Excluded seed 143

Seed 143 is not included in the paired aggregate. Its bounded half was the
earlier manual Unity restart and began at `(0, -0.008, 0)`. Its passive half was
the first automated reset and began at `(0, -2.856, 0)`. The pickup counters
were correctly zero, but the starting heights were not matched.

Descriptively, bounded crossed threshold at 592.07 seconds and passive at
816.90 seconds. Bounded also incurred one `persistent_physics_wedge` survival
failure at 477.57 seconds. Effective PGNW guidance was zero at that event, so it
is not attributable to scientific targeting, but the mismatched start prevents
using this pair in the condition comparison.

## Automation and reset audit

All five scheduled unattended subprocesses returned zero and produced complete
recordings of approximately 1,200 seconds with zero initial pickup counters.
The batch runner therefore worked operationally. The reset implementation used
terrain height sampling, however, which moved automated starts below the
manual-start height. Seeds 144 and 145 remained internally matched because both
conditions used the same automated reset; seed 143 did not.

The reset now restores the exact scene-authored spawn transform instead of
resampling terrain height. The runner also requires reset acknowledgement within
0.25 Unity units of the preregistered `(0, -0.008, 0)` start. A future batch
will abort rather than record a mismatched starting position.

## Claim boundary

Current evidence supports safe online causal learning in both conditions. It
does not support raw active acceleration, a reliability guarantee, or a stable
14.4-minute bounded envelope: bounded ranged from 863.97 seconds to censoring in
the valid pairs. The lower bounded variance and 3/4 versus 2/4 threshold success
are hypotheses for further paired testing, not established effects.
