# L1 Publication Confirmation Protocol

Registered 6 August 2026 before evaluating seed 309 with the JSON adapter or
generating seed 311 with the masked-L1 adapter.

## Fixed system

- Base model: `google/gemma-3-1b-it`
- JSON adapter: `checkpoints/causal_dsl_json_lora_1b_20260804`
- L1 adapter: `checkpoints/causal_dsl_l1_lora_1b_20260804`
- Masked contract: `labeled_causal_ir_masked_greedy`
- No prompt, adapter, candidate-set, verifier, or decoder changes

## Audit A: matched representation comparison

Evaluate frozen JSON on the same seed-309 128-case curriculum already used by
the completed masked-L1 audit. Pair the resulting records by case with
`outputs/l1_masked_greedy_seed309_20260805.json`.

Primary endpoints:

1. Exact semantic accuracy for JSON and masked L1.
2. Mean visible tokens and masked-L1 reduction relative to JSON.
3. Wall-clock runtime, reported descriptively because adapters run
   sequentially and system load is not controlled.

## Audit B: independent repair replication

Evaluate masked L1 on untouched seed 311 with 128 held-out examples.

Primary endpoint: exact semantic accuracy. Report every failure without tuning.
The confirmatory target is 128/128. A lower result is a failed exact replication
and must be published as such.

## Claim boundary

Two 128/128 masked-L1 audits support zero observed defects across 256 new-seed
cases, not universal zero-defect performance. A matched seed-309 comparison may
support the measured token reduction and equal point accuracy on that audit;
it cannot prove permanent JSON parity or equivalent hardware-level cost.
