# Dynamic Tiny Scientist Unity Smoke Protocol — 12 August 2026

## Purpose

Test plumbing, not learning speed: determine whether one uninterrupted Unity
embodiment can collect discovery evidence, formulate an L1 hypothesis in an
asynchronous local Gemma call, formally admit it into a previously null-only
candidate pool, and apply only later clean observations to its posterior.

## Frozen configuration

- New controller seed 153; terrain scene and validated floor-start protocol.
- 1,200-second bound.
- Existing committed target selection, collision masks, MPC regret limits,
  observation isolation, immediate retreat, and all survival vetoes unchanged.
- Dynamic pool begins with only `probe_is_spontaneous` and `no_tested_cause`.
- Before admission, a seed-balanced red/blue/no-pickup survey supplies discovery
  coverage. It does not encode which color is causal.
- Proposal eligibility requires at least four clean discovery observations and
  at least one clean red and one clean blue observation.
- Frozen model: `google/gemma-3-1b-it` with
  `checkpoints/causal_dsl_l1_lora_1b_20260804`.
- Masked-greedy candidate grammar remains symmetric across distinct red/blue
  role assignments, effects, and grounded delays.
- Model generation runs on one background worker. Unity control must continue
  while formulation is in progress.
- The admission cutoff is the exact discovery count copied into the model
  request. Every observation at or before that cutoff is ineligible for
  posterior verification.

## Pass criteria

The plumbing smoke passes only if:

1. telemetry records `dynamic_proposal_status = admitted_unverified` or a later
   verified state, a nonempty raw L1 record, and no admission error;
2. the dynamic pool contains exactly one admitted DSL candidate;
3. at least one clean observation strictly after the admission cutoff causes a
   held-out posterior update;
4. no discovery observation is reused as a candidate update;
5. the loop records no survival failure; and
6. telemetry continues during background generation without a control-stream
   pause large enough to trigger a physics wedge or respawn.

Posterior 0.95 is not required for this first plumbing smoke. If formulation
occurs but there is no later clean evidence within the bound, the result is
inconclusive and the run may be lengthened without changing the architecture.

## Failure interpretation

- No balanced discovery: environmental exposure failure, not LLM failure.
- Model rejection: formulation/interface failure; retain zero candidate
  authority.
- Candidate admitted but no held-out update: timing/exposure failure.
- Any pre-cutoff update: evidence leakage and invalid result.
- Control pause, wedge, or survival failure during generation: async integration
  failure even if the hypothesis is correct.
