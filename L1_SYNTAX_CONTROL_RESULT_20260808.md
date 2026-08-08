# Syntax-Only Control for the L1 Causal Decoder

## Question

Could ordinary grammar-constrained decoding explain the complete L1 repair, or
did the mechanism-motivated distinct-role rule add anything beyond valid
surface syntax?

The control was registered in `L1_SYNTAX_CONTROL_PROTOCOL_20260808.md` before
Gemma produced any output under the new syntax-only condition. The base model,
frozen L1 adapter, prompt, tokenizer, one-path greedy search, validator, and
semantic exact-match criterion were unchanged.

## Decoder conditions

1. **Ordinary greedy:** no token mask.
2. **Syntax-only masked greedy:** permits every grammatical L1 record over the
   observed feature vocabulary, including `cause == comparison`.
3. **Syntax plus distinct-role masked greedy:** uses the same grammar but
   requires cause and comparison to bind to different observed features.

The syntax-only and role-bound decoders had identical valid tokens everywhere
except assignments that repeated one observed feature in both semantic roles.
Neither mask identified the correct cause.

## Stage A: matched seed-309 diagnostic

| Condition | Exact matches | Accuracy | Mean visible tokens | Elapsed |
|---|---:|---:|---:|---:|
| Ordinary greedy | 123/128 | 96.094% | 184.344 | 160.323 s |
| Syntax-only mask | 124/128 | 96.875% | 184.328 | 158.072 s |
| Syntax + distinct role | 128/128 | 100.000% | 184.328 | 164.412 s |

Syntax-only masking repaired the one malformed numeric insertion and left all
four `gold`-as-cause / `gold`-as-comparison repetitions unchanged. Adding the
distinct-role constraint repaired those four cases without a regression. This
stage reuses previously analyzed seed-309 ordinary and role-bound records, so
it is a targeted post-hoc decomposition rather than independent confirmation.

## Stage B: untouched seed-313 replication

| Condition | Exact matches | Accuracy | Mean visible tokens | Elapsed |
|---|---:|---:|---:|---:|
| Ordinary greedy | 121/128 | 94.531% | 184.547 | 157.541 s |
| Syntax-only mask | 124/128 | 96.875% | 184.492 | 155.588 s |
| Syntax + distinct role | 128/128 | 100.000% | 184.492 | 159.159 s |

On the untouched audit, syntax-only masking repaired three surface corruptions:
two inserted numeric atoms between the comparison and effect fields and one
noncanonical `-0.2` effect atom. It left four repeated `gold/gold` role
assignments. The distinct-role mask repaired exactly those four remaining
cases, again with no regression.

The confirmatory syntax-to-role comparison contains four repairs and zero
regressions. Its exact two-sided McNemar value is 0.125, so this 128-case audit
alone does not establish a population-level advantage at the conventional
0.05 threshold. The identical four-case decomposition replicated the earlier
diagnostic set. Pooling both registered stages descriptively gives eight
role-specific repairs and zero regressions (exact two-sided McNemar 0.0078125),
but this pooled value includes the post-hoc diagnostic set and is not the
primary confirmatory statistic.

## Conclusion and boundary

The skeptical explanation is partly correct: a constrained-decoding harness
accounts for the serialization repairs. Generic syntax enforcement did **not**
repair the observed comparison-role binding failures. The mechanism-motivated
distinct-role constraint added a reproducible four-case improvement beyond
syntax alone on each 128-case set.

This remains an engineered neuro-symbolic repair. Code supplies the semantic
invariant that cause and comparison must be distinct; the language model is not
demonstrating unaided perfect variable binding. The result supports a narrow
division of labor—statistical selection inside a formally valid semantic state
space—not a universal decoder advantage, complete binding circuit, or general
solution to language-model reasoning.

Artifacts:

- `outputs/l1_syntax_masked_greedy_seed309_20260808.json`
- `outputs/l1_mask_controls_seed313_20260808.json`
- `outputs/l1_syntax_masked_greedy_smoke_seed312_20260808.json`
- `outputs/l1_role_masked_greedy_smoke_seed312_20260808.json`
