# Predictive Attention-Schema Result — 9 August 2026

## Outcome

The preregistered predictive attention schema did **not** pass the overall
behavioral decision rule. It learned temporally useful information and clearly
outperformed a shuffled-role schema, but its fixed influence produced too much
inertia at genuine context changes to outperform the existing adaptive GNW.

## Frozen audit

- 12 seed-separated conductor, governor, and schema fits
- 17,280 held-out steps per condition
- 60-parameter linear softmax schema
- fixed schema blend: 0.30
- conditions: adaptive GNW, correctly bound schema, shuffled-role schema, and
  disconnected post-training lesion

The predictor used only current/previous specialist bids, current broadcast
identity, bid entropy, and bid margin. It did not receive context labels,
optimal specialists, evaluation targets, or future observations.

## Main results

| Measure | Adaptive GNW | AST schema | AST shuffled |
|---|---:|---:|---:|
| Utility per step | 0.6413 | 0.5972 | 0.5205 |
| Optimal routing | 0.7980 | 0.7547 | 0.6718 |
| Boundary routing | 0.6442 | 0.4588 | 0.5188 |
| Distractor routing | 0.4810 | 0.6483 | 0.4994 |
| Unnecessary handoffs | 0.1112 | 0.0806 | 0.0718 |
| Next-focus prediction accuracy | — | 0.7206 | 0.7278 raw / 0.0963 applied |

Paired seed contrasts for the correctly bound schema were:

- utility versus adaptive GNW: `-0.04414`, 95% interval
  `[-0.07698, -0.01129]`;
- utility versus shuffled schema: `+0.07664`, 95% interval
  `[+0.02562, +0.12766]`;
- boundary routing versus adaptive GNW: `-0.18542`, 95% interval
  `[-0.26313, -0.10770]`;
- distractor routing versus adaptive GNW: `+0.16628`, 95% interval
  `[+0.06743, +0.26513]`;
- unnecessary handoffs versus adaptive GNW: `-0.03056`, 95% interval
  `[-0.05086, -0.01025]`.

The disconnected lesion exactly matched the no-schema adaptive GNW, confirming
that behavioral differences came from schema access rather than predictor
training or evaluation bookkeeping.

## Interpretation

The schema acts as a learned persistence prior. That prior is useful when a
single-frame distractor challenges a valid broadcast, and the role-shuffled
control shows that correctly bound predictions matter. The same prior is
harmful when the environment truly changes: its one-frame forecast favors the
previous focus and delays the already-capable adaptive governor.

This is not evidence that attention schemas are generally harmful or that AST
is false. It is evidence that an always-on, fixed-weight next-focus predictor is
redundant with—and can impair—this governor, which already explicitly tracks
active focus, challenger streaks, persistence, and reward. Prediction accuracy
alone did not establish control value.

## Next admissible step

A new experiment could preregister a change-point-aware schema that predicts
both next focus and the reliability of that prediction, allowing schema
influence only when calibrated confidence is high. That would be a new
hypothesis, not a reinterpretation or tuning of this frozen negative result.
Unity integration is not warranted until a revised schema passes a separate
synthetic held-out audit.

Full metrics: `outputs/attention_schema_registered_20260809.json`.
