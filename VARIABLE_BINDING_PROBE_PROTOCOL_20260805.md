# Frozen Gemma 1B Variable-Binding Probe Protocol

**Registered before model execution:** 5 August 2026

## Motivation

The frozen Gemma 3 1B L1 adapter produced 121/128 exact causal records while
the matched JSON adapter produced 128/128. This protocol asks where the compact
adapter loses cause/control binding. It is motivated by Wu, Geiger, and
Millière (ICML 2025), but does not assume that Gemma implements their circuit.

## Frozen inputs

- Base: `google/gemma-3-1b-it`
- Adapter: `checkpoints/causal_dsl_l1_lora_1b_20260804`
- Audit: `outputs/causal_dsl_frozen_comparison_1b_20260804.json`
- Audit SHA-256:
  `0ad6e97ef9f93e60eec45c06e47b9cdf6ca38043c196e3209b0df0774f92f10c`
- No adapter, prompt, curriculum, or generated record may change.

## Cohort

Select 32 of the 128 frozen cases. Retain all seven L1 failures, then select
successful controls deterministically to balance feature pair, cause row,
evidence order, and effect. The frozen manifest contains eight cases from each
of four color pairs and sixteen cases for each cause-row label.

## Primary endpoint

At every residual-stream layer, read the activation immediately before the
teacher-forced cause feature token. Train a ridge probe to predict whether the
correct cause occurred in evidence row zero or one. Test with leave-one-feature-
pair-out cross-validation so a probe cannot succeed by memorizing color names.

Report the peak layer accuracy. Correct for searching across layers with 100
within-feature-pair label permutations, using the maximum null accuracy from
each permutation. This is a representation-decoding result, not yet a causal
circuit result.

## Attention discovery and causal confirmation

On even original case indices only, rank attention heads by attention mass from
the cause prediction position to the correct evidence feature minus the
comparison feature. Freeze the top positive head. In the same layer, freeze the
head with the smallest absolute score as the negative control.

On odd original case indices only, ablate each selected head before its output
projection. Measure the change in correct-versus-swapped first cause-token
logit margin. The causal gate passes only if the discovery-ranked head lowers
the margin and lowers it more than the fixed same-layer control.

Passing supports only: *the ranked head contributes to first-token causal-role
binding in this adapter*. It does not establish a complete variable-binding
circuit, explain later syntax errors, or show generality beyond this assay.

## Execution

Dry registration (no model load):

```bash
python3 variable_binding_probe.py --dry-run
```

Bounded live probe, only when MPS is available:

```bash
python3 variable_binding_probe.py \
  --device mps \
  --maximum-cases 32 \
  --permutations 100 \
  --output outputs/variable_binding_probe_1b_20260805.json \
  --activations outputs/variable_binding_probe_1b_20260805.npz
```
