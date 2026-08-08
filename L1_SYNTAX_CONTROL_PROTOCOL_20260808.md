# L1 Syntax-Only Constrained-Decoding Control Protocol

Registered 8 August 2026 after implementing and unit-testing the syntax-only
candidate set, but before loading Gemma or observing any output from this new
decoder condition.

## Question

Did the successful L1 repair require the mechanism-motivated rule that cause
and comparison bind to distinct observed features, or would an ordinary
grammar-constrained greedy decoder have produced the same result?

## Fixed system

- Base model: `google/gemma-3-1b-it`
- Frozen adapter: `checkpoints/causal_dsl_l1_lora_1b_20260804`
- Prompt and tokenizer: unchanged from the completed L1 audits
- Validator and semantic exact-match criterion: unchanged
- Decoding remains one-path greedy in both masked conditions
- Existing role-bound contract: `labeled_causal_ir_masked_greedy`
- New syntax-only control: `labeled_causal_ir_syntax_masked_greedy`

The syntax-only control permits every grammatical L1 assignment over observed
feature tokens, including `cause == comparison`. It constrains the version and
literal role atoms, feature vocabulary, canonical effect atom, observed/zero
delay values, confidence value, and record termination. It does not enforce
the distinct-role rule. The unchanged validator rejects a repeated-role output
as semantically invalid.

## Stage A: matched diagnostic control

Run only the new syntax-only condition on the already frozen seed-309,
128-case curriculum. Compare it case-by-case with the preserved ordinary L1
greedy and role-bound masked-greedy records in
`outputs/l1_masked_greedy_seed309_20260805.json`.

This stage is a targeted post-hoc decomposition, not an independent
replication. Based on the already published failure classification, the fixed
prediction is that syntax-only masking may repair serialization corruption but
will continue to permit genuine repeated comparison-role bindings.

## Stage B: untouched replication

If Stage A executes correctly, evaluate untouched seed 313 with 128 held-out
examples under three frozen L1 conditions:

1. ordinary greedy L1;
2. syntax-only masked greedy;
3. syntax-plus-distinct-role masked greedy.

Run all conditions with the same model, adapter, prompt, tokenizer, examples,
and semantic scorer. Runtime is descriptive because conditions are sequential.
Seed 312 with eight cases is reserved for a smoke test and is not confirmatory.

## Primary interpretation gate

- If syntax-only and role-bound masking perform equally and repair the same
  cases, attribute the observed repair to generic grammar-constrained decoding;
  the present audit does not support mechanism-specific added value.
- If role-bound masking repairs repeated-role failures left by syntax-only,
  without regressions, the result supports the narrower claim that the
  mechanism-motivated semantic constraint adds value beyond syntax alone on
  this task.
- If role-bound masking introduces regressions or does not replicate, report
  the repair as unstable.

No condition can establish a universal decoder advantage, a complete
variable-binding circuit, or an end-to-end hardware efficiency improvement.
