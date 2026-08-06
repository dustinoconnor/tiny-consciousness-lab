# Agent-Native Causal Representation: A Reliability–Token Pareto Result

**Date:** 4 August 2026
**Status:** Frozen local result with preregistered repair confirmation; not peer reviewed

## Question

Can a compact agent-native causal representation reduce Gemma 3 inference
tokens while preserving the exact cause/control/effect binding reliability of
JSON?

## Matched experiment

Gemma 3 1B JSON and labeled-DSL (L1) adapters were trained from the same base
model with the same seed-104 factorial curriculum: 192 examples, two epochs,
rank-8 q/v LoRA, and gradient accumulation of eight. Both adapters were frozen
before a seed-205 evaluation on 128 held-out causal tables.

| Representation | Exact matches | Accuracy | Mean visible tokens |
| --- | ---: | ---: | ---: |
| JSON | 128/128 | 100.000% | 282.367 |
| L1 labeled DSL | 121/128 | 94.531% | 184.430 |

L1 reduced visible input-plus-output tokens by **34.684%**, but lost **5.469
percentage points** of exact semantic accuracy. Paired outcomes were 121 both
correct, seven JSON-only correct, zero L1-only correct, and zero both wrong.
Failures included repeated cause/control atoms, inserted numeric fragments, and
malformed effect binding.

One final semantic-label variant (H1) was frozen and screened on a separate
seed-306 set of 64 examples. JSON scored 64/64; H1 scored 56/64 while reducing
visible tokens by 39.930%. Semantic words therefore did not remove the
reliability tax.

## Embodied diagnostic boundary

The corrected Unity recording contained only one bidirectionally isolated red
pickup, below the registered minimum of two. Each representation was also
tested in only canonical and reversed evidence order. Those two-item results
are diagnostic and exposure-ineligible; they are not estimates of embodied
accuracy and cannot establish robust transfer.

## Conclusion

The compact representations occupy a different point on a Pareto frontier:
lower visible token volume in exchange for lower exact reliability. They are
not strict improvements over JSON. Because a cause/control swap can compile
the wrong embodied intervention, JSON remains the production interchange
format. L1 and H1 remain research artifacts until a frozen replication
preserves JSON-level binding reliability.

This result does **not** support claims of a 50% inference-cost reduction,
equal-performance compression, robust embodied transfer, or production
readiness. It does show that shortening a structured language is not free:
grammar validity and semantic variable binding are separable constraints for a
small language model.

## Reproduction artifacts

- `causal_dsl_lora.py`
- `compare_causal_dsl_adapters.py`
- `outputs/causal_dsl_1b_summary_20260804.json`
- `outputs/causal_dsl_frozen_comparison_1b_20260804.json`
- `outputs/causal_dsl_h1_screen_1b_seed306_20260804.json`
- `outputs/causal_dsl_l1_unity_seed97_summary_20260804.json`

Adapter weights are intentionally excluded from the ordinary Git repository;
the training settings and frozen result artifacts are retained for audit and
reproduction.

## Mechanism-guided resolution (5–6 August 2026)

The original Pareto result remains the baseline, but a subsequent mechanistic
probe separated its seven errors into four comparison-role binding failures and
three grammar/serialization failures. A symmetric finite-state decoder was
then registered before new-seed evaluation. It follows Gemma's ordinary greedy
choice unless that token would make every valid L1 continuation impossible;
it enforces valid grammar and distinct observed cause/comparison roles without
encoding which causal assignment is correct.

On seed 309, ordinary L1 scored 123/128 and masked-greedy L1 scored 128/128,
repairing five errors with zero regressions. The repair added 2.55% elapsed time
relative to ordinary L1 while leaving visible-token volume effectively
unchanged. An independent, untouched seed-311 replication also scored 128/128.
Thus the frozen masked system produced **zero observed defects across 256
new-seed cases**.

The publication confirmation evaluated the frozen JSON adapter on those same
128 seed-309 cases:

| Representation | Exact matches | Mean visible tokens | Elapsed time |
| --- | ---: | ---: | ---: |
| JSON | 128/128 | 282.328 | 323.345 s |
| Masked-greedy L1 | 128/128 | 184.328 | 164.412 s |

Masked L1 therefore preserved matched point accuracy while using **34.711%
fewer visible tokens**. Its observed run was 49.15% faster than JSON, although
that timing is descriptive rather than a controlled hardware benchmark.

The defensible conclusion is that the observed reliability–token trade-off was
resolved on these frozen audits by combining statistical causal selection with
a non-answer-leaking formal grammar and role constraint. “Zero observed
defects” must not be shortened to universal “zero-defect precision”; broader
causal distributions and embodied replication remain future work.

Additional artifacts:

- `VARIABLE_BINDING_PROBE_PROTOCOL_20260805.md`
- `L1_MASKED_GREEDY_PROTOCOL_20260805.md`
- `L1_PUBLICATION_CONFIRMATION_PROTOCOL_20260806.md`
- `outputs/variable_binding_probe_summary_20260805.json`
- `outputs/l1_masked_greedy_summary_20260805.json`
- `outputs/l1_publication_confirmation_summary_20260806.json`
