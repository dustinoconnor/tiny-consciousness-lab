# Dynamic Tiny Scientist Unity Smoke Result — Seed 153

## Outcome

The 1,200-second live plumbing smoke failed the registered admission criterion.
It nevertheless isolated two concrete integration defects without a survival
failure.

The recording is complete: 5,110 rows spanning 1,199.384 seconds in
`dynamic_committed` mode. It contains 22 clean observations, four discarded
windows, 22 pickups including two red, 66 PGNW action changes, five stuck
events, zero critical-hunger exposure, zero respawns, and zero survival
failures.

## Formulation failure

Balanced discovery was initially blue/no-pickup heavy. The first clean red
positive completed at 798.267 seconds as discovery observation 16. Its frozen
summary contained one red delta `+0.340` and nine blue controls with mean delta
`-0.000000363`. Gemma generated:

```text
L1 c blue k red e - t 10.0 q 0.5
```

The formal verifier correctly rejected it as reversing the grounded causal
direction: blue was effectively unchanged, not a delayed decrease. No dynamic
candidate was admitted, no discovery observation updated a candidate, and no
held-out verification occurred.

## Retry defect

After rejection, the live planner immediately scheduled another identical
background proposal. This repeated for the remainder of the run and left the
last frame in `generating` with the previous rejection preserved in
`dynamic_admission_error`. The body continued receiving commands, but repeated
model loads are an invalid and wasteful behavior. Rejection is now terminal for
the run.

## Frozen diagnostic repair

The discovery summary had inherited feature insertion order from whichever
color was first encountered. It is now explicitly red then blue, matching the
registered canonical evidence order without selecting an answer.

An exact-evidence diagnostic localized the remaining sign failure to numerical
serialization below meaningful telemetry precision:

| Blue mean encoded in prompt | Frozen masked-greedy output |
|---:|---|
| `0.0` | correct `red ... e +` |
| `-0.000001` | rejected `red ... e -` |
| `-0.008` | correct `red ... e +` |

The live value was approximately `-3.63e-7`. Discovery means are now frozen to
three decimal places with magnitudes below `0.0005` normalized to `0.0`. This is
a measurement-precision rule applied symmetrically to every feature, not an
answer-specific correction. The strict semantic verifier remains unchanged.

## Decision

Seed 153 is a failed but informative plumbing smoke and must not be counted as
a successful autonomous formulation run. One new-seed repeat is warranted with
the terminal-rejection and numerical-precision repairs frozen. Posterior 0.95
remains unnecessary for that smoke; admission plus one strictly later update is
the gate.
