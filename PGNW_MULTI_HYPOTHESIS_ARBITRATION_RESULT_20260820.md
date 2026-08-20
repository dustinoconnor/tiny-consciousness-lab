# PGNW Actionable Multi-Hypothesis Arbitration — 2026-08-20

## Question

Can PGNW choose among several actionable causal candidates using the complete
typed posterior, metabolic value, epistemic value, route cost, and memory
reliability without leaking the correct suppressor or depending on color/order?

This differs from the earlier synthetic PGNW experiment-selection benchmark:
that assay selected which intervention would be informative. This assay selects
which already-grounded causal candidate should receive an actionable broadcast
after a red-triggered hazard.

## Registered scoring rule

For each safe candidate, the arbiter computes:

`score = pragmatic_value + epistemic_value - route_cost`

- Suppression probability is derived from the full four-hypothesis posterior
  and the formal likelihood matrix.
- Pragmatic value combines suppression probability, hazard cost, memory
  confidence, and probability of arriving before the deadline.
- Epistemic value is normalized expected information gain.
- Route cost uses conservative travel time divided by remaining deadline.
- Safety denial sets eligibility false and score to negative infinity.

No function receives the true suppressor, expected winner, or answer label.

## Counterbalanced matrix

The 2 × 2 × 2 matrix crossed:

- dominant causal identity: yellow / blue;
- route assignment: dominant nearer / dominant farther;
- candidate presentation order: yellow-first / blue-first.

All 8/8 cases selected the posterior-supported causal candidate. Score margins
ranged from 0.071292 to 0.083736, so the result was not produced by a numerical
tie. Swapping causal posterior roles swapped the selected color, while reversing
candidate order left records and selection unchanged.

## Replay-grounded embodied case

Using the seed-168 verified posterior and frozen seed-163 terrain memory at the
post-decoy position:

| Candidate | Score | Suppression probability | Information gain | Route cost |
|---|---:|---:|---:|---:|
| Yellow | 0.079840 | 0.916019 | 0.009896 bits | 0.003841 |
| Blue | 0.006319 | 0.066640 | 0.060137 bits | 0.005963 |

The arbiter selected yellow. This is not simply a nearest-memory rule: yellow
was approximately 36.87 units away and blue approximately 57.24 units away, and
both posterior, memory confidence, epistemic value, and route terms contributed
to their transparent scores.

In the safety ablation, an otherwise dominant yellow candidate was marked
unsafe and could not be selected; the eligible blue candidate won. Safety is an
absolute gate rather than another soft score.

## Verification and boundary

- Focused arbitration regressions: 5/5 passed.
- Full repository suite: 296/296 passed in 6.850 seconds.
- Machine-readable artifact:
  `outputs/pgnw_multi_hypothesis_arbitration_20260820.json`.

This establishes deterministic, answer-blind arbitration in a counterbalanced
offline and replay-grounded assay. It does not yet grant the arbiter Unity motor
authority, validate its fixed utility weights across environments, or show
online arbitration after a genuinely novel rule. The next defensible step is a
passive Unity shadow field that compares arbitration recommendations with the
existing verified-protective controller before any authority change.
