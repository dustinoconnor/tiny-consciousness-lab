# RPT-2 / HOT-4 Perceptual Intelligence Result — 22 August 2026

## Verdict

The frozen reserved evaluation did not support RPT-2, HOT-4, or the combined
added-intelligence claim under the preregistered thresholds.

The recurrent mechanism had a large causal dependence on temporal order, and
the combined model achieved the highest absolute reserved accuracy. However,
the matched flat control was already strong, the recurrent increment was too
small, the explicit identity probe remained near chance, and the learned HOT-4
quality geometry was weak.

## Reserved results

Each condition used four training initializations and all eight reserved data
seeds, producing 32 matched evaluations per condition.

| Condition | Validation accuracy | Reserved accuracy | Identity probe | Quality geometry | Active code |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.9423 | 0.9114 | 0.5020 | 0.0043 | 0.2314 |
| RPT-2 | 0.9685 | 0.9338 | 0.4901 | 0.0219 | 0.3854 |
| HOT-4 | 0.9370 | 0.9108 | 0.4995 | 0.2108 | 0.4584 |
| RPT-2 + HOT-4 | 0.9710 | **0.9438** | 0.4840 | **0.2213** | 0.4156 |
| Combined, permuted quality target | 0.9670 | 0.9401 | 0.4887 | 0.0668 | 0.2752 |

Frozen contrasts:

- RPT-2 minus flat reserved accuracy: `+0.02236`.
- RPT-2 time-shuffle loss: `-0.38764`.
- HOT-4 increment over RPT-2: `+0.01007`.
- Correct versus permuted HOT-4 geometry: `+0.15448`.
- Combined versus best single mechanism: `+0.01007`.
- Combined validation tax: `-0.00250` (no tax; combined was slightly better).

## Decision audit

### RPT-2

The time-shuffle requirement passed decisively: destroying temporal order
reduced recurrent accuracy from 0.9338 to 0.5461. This establishes causal use
of ordered recurrence. The primary improvement requirement failed because the
recurrent advantage over the parameter-matched flat control was 2.24 percentage
points, below 5 points. The linear identity-binding probe was also near chance,
so the result does not establish an explicitly organized identity scene code.

### HOT-4

The combined code was sparse enough (41.56% active coordinates), and correct
quality supervision produced more geometry than a permuted answer-blind control.
But distance correlation was only 0.2213 rather than 0.70, the control contrast
was 0.1545 rather than 0.20, and the accuracy increment was 1.01 rather than
3 points. HOT-4 therefore learned a weak quality trace, not a defensible smooth
quality space that added meaningful transfer intelligence.

### Combined claim

The combined model was the best condition at 94.38% and incurred no
in-distribution cost. Its one-point advantage over RPT-2 was below the frozen
three-point decision boundary. It is a promising descriptive observation, not
a positive result.

## Claim boundary and next action

This benchmark does not test phenomenal consciousness. It tests two narrow
functional properties derived from consciousness theories.

The present components should not be integrated into the Unity production
controller. A defensible follow-up would first remove the flat model's apparent
task shortcut and train an object-centric perceptual state whose identity is
independently decodable. HOT-4 should then operate over those explicit object
states rather than over one global scene vector. That is a new hypothesis and
requires a new protocol and untouched evaluation set.

Full frozen artifact:
`outputs/rpt2_hot4_perceptual_intelligence_20260822.json`.
