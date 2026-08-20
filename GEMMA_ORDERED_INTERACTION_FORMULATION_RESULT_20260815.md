# Gemma Ordered-Interaction Formulation — 2026-08-15

## Result

Local Gemma 3 1B selected the newly observed yellow causal role from a frozen
three-episode discovery set. A formal, answer-blind compiler admitted:

`T1 i red s yellow r after e suppresses_probe`

The two seed-159 red/yellow observations were withheld until after admission and
both agreed with the compiled rule (2/2 held-out accuracy).

## Evidence boundary

Gemma saw only accepted observations from seed 157 and seed 158:

- `red_then_blue`: 0 of 2 probe suppressions
- `red_then_yellow`: 1 of 1 probe suppressions

It did not receive seed-159 observations, the hidden cancellation feature, the
98.06% Bayesian posterior, the words `antidote`, `toxin`, or `correct`, or the
name of the verified candidate. The compiler checked grounding, temporal
structure, schema, and whether the proposed action had actually been observed;
it did not check whether the selected color matched the hidden world.

## Decoder ablations

Three uncalibrated attempts failed and are retained as results rather than
discarded:

1. Free JSON generation restated the blue negative control and emitted an
   invalid relation field.
2. A clearer positive-suppression request produced malformed, ungrounded schema
   values.
3. Symmetric whole-record scoring admitted the incorrect blue-suppression rule.
   Even single-role constrained scoring preferred `blue` by 0.1875 mean log
   probability despite blue's observed 0/2 suppression rate.

This exposed a lexical/token prior toward `blue`. A content-free calibration
run used the identical blue/yellow choices under equal zero-suppression evidence
and subtracted that baseline preference. The evidence-induced gains were:

- yellow: +0.307418
- blue: -0.192582

The calibrated margin was +0.500000 for yellow, corresponding to a two-choice
softmax confidence of 0.622459. The formal layer then bound the shared red
initiator, `after` relation, and positive suppression effect around Gemma's
selected `yellow` role.

## Scope and next test

This is a successful frozen-log replay of neuro-symbolic dynamic formulation,
not yet live asynchronous formulation inside the Unity process. Gemma selected
a previously unknown causal role; it did not invent the supplied object types,
temporal operator, or effect vocabulary. Because calibration was introduced
after observing the uncalibrated failures, the result is exploratory. A frozen,
counterbalanced replication with color/order permutations is required before a
general reliability claim, followed by integration of this calibrated slot
decoder into the live PGNW candidate-admission path.

Artifacts:

- Successful result: `outputs/gemma_ordered_interaction_seed159_calibrated_20260815.json`
- Initial free failure: `outputs/gemma_ordered_interaction_seed159_20260815.json`
- Clarified free failure: `outputs/gemma_ordered_interaction_seed159_v2_20260815.json`
- Uncalibrated whole-record failure:
  `outputs/gemma_ordered_interaction_seed159_grounded_20260815.json`
