# Agent-Native Causal Representation: A Reliability–Token Pareto Result

**Date:** 4 August 2026
**Status:** Frozen local result; not peer reviewed

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
