# Sealed Calibrated Formulation Smoke — Seed 161 — 2026-08-17

## Verdict

The core bounded smoke passed. The live loop loaded an immutable three-record
discovery checkpoint, wrote state to a separate run memory, formulated and
admitted the yellow ordered-suppression rule asynchronously, and promoted it to
`verified_held_out` from one post-cutoff physical episode. Controller safety
also passed. Guided resource memory was configured and updated but did not cross
its hunger gate, so this run does not test resource-guidance action authority.

## Provenance and timing

- Telemetry: `outputs/unity_shadow/sealed_calibrated_formulation_seed161_20260817.jsonl`
- Read-only discovery: `checkpoints/typed_interaction/discovery_three_20260817.json`
- Writable run memory: `outputs/typed_interaction_formulation_memory_seed161_20260817.json`
- Resource memory: `outputs/resource_memory_live_smoke_seed161_20260817.json`
- One continuous session, steps 0–851, 852 records, and 179.966 recorded seconds.
- The discovery input contained exactly three observations and remained separate
  from the run-memory output. Formulation cutoff was exactly 3.

## Causal sequence

- At startup, committed PGNW requested `red_then_yellow`.
- Red was collected at step 8 (3.142 seconds).
- PGNW changed four MPC selections at steps 9, 13, 14, and 15.
- Yellow was collected at step 16 (4.982 seconds) and cancelled the pending red
  probe event. The interval remained uncontaminated.
- Gemma completed at step 24 (6.637 seconds), after the physical pickups but
  before their delayed outcome was available. It could use only the frozen
  three-record discovery summary.
- The formal layer admitted
  `T1 i red s yellow r after e suppresses_probe` at cutoff 3.
- At step 308 (66.178 seconds), the clean post-cutoff interval reached its
  deadline, was accepted as observation 4, and promoted formulation status from
  `admitted_unverified` to `verified_held_out` with 1/1 confirmation.

The formulation layer retained zero motor authority. PGNW selected and executed
the intervention from the pre-existing typed Bayesian pool; the later held-out
outcome verifies the separately admitted Gemma/formal rule rather than showing
that the new formulation caused its own evidence collection.

## Decoder and posterior

The answer-blind rate mask exposed blue 0/2 and yellow 1/1, leaving yellow as
the sole maximum-rate candidate. Gemma's content-free calibrated gains were
+0.307418 for yellow and -0.192582 for blue. The decoder emitted raw `yellow`
and logged the policy
`maximum_observed_rate_mask_then_calibrated_gemma_tiebreak`.

The accepted fourth observation raised the typed Bayesian posterior for
yellow-specific suppression from 92.7844% to 97.6319%. No discovery record was
added to the immutable source.

## Controller and resource memory

- PGNW: committed mode, 16 guidance decisions, 1 commitment start, and 4 action
  changes. All four changes occurred after red and before yellow; none were
  isolation changes.
- Pickups: 1 red, 1 yellow, and 4 blue.
- Hidden outcomes: 1 cancellation, 0 probe rises, 0 pending events, and 0
  ambiguous pickup frames.
- Resource memory loaded 52 regions and encoded all 6 new pickups. Hunger ranged
  from 0.001 to 0.429, below the fixed 0.70 retrieval gate, producing 0 queries,
  recommendations, guidance decisions, or action changes.
- Safety: stable throughout, with 0 survival failures, critical-hunger seconds,
  stuck events, and respawns.

## Limits

This is one bounded synthetic Unity confirmation, not a population-level
reliability result. The formal rate mask guarantees that a uniquely lower
observed rate cannot win from lexical bias; therefore the result supports the
neuro-symbolic decoder, not unaided Gemma arithmetic. The resource-memory path
was enabled and writable but received no opportunity to exercise guidance.
