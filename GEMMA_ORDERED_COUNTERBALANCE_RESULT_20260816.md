# Gemma Ordered-Formulation Counterbalance — 2026-08-16

## Verdict

The calibrated Gemma 3 1B suppressor-slot decoder reproduced 6/8 conditions
(75%). The result does not support presentation-invariant generalization. Color
assignment and evidence-line order were individually balanced, but the temporal
surface form produced a clear interaction: `after` passed 4/4, while `then`
passed only 2/4.

## Frozen matrix

Each condition crossed three factors:

- causal suppressor assignment: blue or yellow;
- evidence presentation order: blue/yellow or yellow/blue;
- equivalent temporal wording: `red then color` or `color after red`.

The suppressor received 1/1 suppression evidence and the control received 0/2.
Each condition used a separately matched neutral baseline with 0/1 for both
colors, so calibration did not encode the answer.

| Suppressor | Presentation | Wording | Selected | Margin | Result |
|---|---|---|---|---:|---|
| blue | blue, yellow | then | yellow | 0.187500 | fail |
| blue | blue, yellow | after | blue | 0.500000 | pass |
| blue | yellow, blue | then | blue | 0.125000 | pass |
| blue | yellow, blue | after | blue | 1.687500 | pass |
| yellow | blue, yellow | then | yellow | 0.937500 | pass |
| yellow | blue, yellow | after | yellow | 3.062500 | pass |
| yellow | yellow, blue | then | blue | 0.312500 | fail |
| yellow | yellow, blue | after | yellow | 0.125000 | pass |

The deterministic rerun reproduced every selection and margin exactly.

## Factor summaries

- blue assigned as suppressor: 3/4;
- yellow assigned as suppressor: 3/4;
- blue/yellow presentation: 3/4;
- yellow/blue presentation: 3/4;
- `then` wording: 2/4;
- `after` wording: 4/4.

The two failures occurred when the true suppressor was presented first under
`then`; Gemma selected the second-listed color. This is consistent with a
wording-by-recency interaction, although eight conditions are too few to
identify a mechanism confidently. Incorrect selections had margins up to
0.312500, so they were not numerical ties.

## Held-out assessment and limitations

Under the counterfactual label mappings, the selected rules matched 12/16
designated held-out outcomes. These are systematic relabelings of the existing
Unity evidence, not sixteen new physical episodes. Therefore this experiment
tests decoder invariance and prompt-order artifacts; it does not establish that
the Unity world generalizes across physically reassigned colors or reversed
causal order.

The earlier seed-160 live result remains valid for its frozen yellow/`then`
condition, but calibrated success is not robust to all equivalent presentations.
The next defensible step is to freeze a wording-invariant evidence interface or
train/evaluate a small balanced role-binding adapter, then rerun this unchanged
8-condition matrix. No prompt repair should be judged on the same matrix without
a newly reserved replication set.

Artifact: `outputs/gemma_ordered_counterbalance_20260816.json`
