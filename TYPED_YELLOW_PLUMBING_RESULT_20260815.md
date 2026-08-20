# Typed Yellow Unity Plumbing Result — 2026-08-15

## Verdict

The bounded Unity smoke passed the pickup and hidden-world plumbing gate, but it
did **not** constitute learned causal discovery. The live embodied loop does not
yet instantiate or update `TypedInteractionPool`; its four candidate posteriors
therefore remained an offline formal substrate rather than an online belief.

## Audited record

- Log: `outputs/unity_shadow/typed_yellow_plumbing_seed155_20260815.jsonl`
- The file contains two appended Unity sessions, identified by a step counter
  discontinuity. The second/current session ran from step 0 through step 2827.
- Current-session pickups: 6 red, 10 blue, and 1 yellow.
- Current-session delayed probe outcomes: 5 completed, 1 cancelled, 0 pending
  at termination.
- The yellow pickup occurred at step 1824 while one red-triggered event was
  pending. It changed the cancellation counter from 0 to 1 and the pending
  count from 1 to 0.
- No blue pickup occurred between the most recent initiating red pickup at step
  1632 and the yellow pickup at step 1824. The interval was 192 controller
  steps (38.4 nominal seconds), inside the frozen 60-second window.
- Same-frame/order-ambiguous red/yellow pickups: 0.
- Safety: 0 survival failures and 0 critical-hunger seconds.
- Navigation diagnostics: 5 stuck events in the current session and 0 unstuck
  respawns. These do not invalidate the plumbing encounter, but must remain a
  matched-condition metric in later behavioral tests.
- `causal_probe_action_influence` remained 0 throughout, so the hidden outcome
  did not directly steer control.

## What this establishes

Yellow is independently counted in Unity and can participate in the intended
ordered hidden transition. At least one naturally encountered red-then-yellow
sequence cleanly cancelled a pending probe event. This satisfies the bounded
pickup/world-dynamics portion of protocol gate 4.

## What it does not establish

- No posterior was updated and no hypothesis was learned or verified online.
- The cancellation counter is emitted by the hidden environment dynamics; it
  is ground truth for auditing, not evidence that the agent inferred the rule.
- One positive yellow sequence cannot distinguish yellow-specific suppression
  from the `any_pickup_after_red` alternative. Clean red-wait, red-blue, and
  yellow-before-red controls are still required.
- Yellow visibility, distance, and direction reached Python through the Unity
  bridge but were not persisted in the shadow recorder. Those fields must be
  added before claiming the full telemetry gate or auditing epistemic target
  selection.

## Next gate

Persist yellow sensing telemetry, then connect clean ordered-episode admission
to the symmetric `TypedInteractionPool`. The first online test should update all
four candidates only after completed/discarded windows and should retain the
predeclared contamination rules. Gemma formulation and PGNW experiment
selection should remain downstream of that auditable learner.
