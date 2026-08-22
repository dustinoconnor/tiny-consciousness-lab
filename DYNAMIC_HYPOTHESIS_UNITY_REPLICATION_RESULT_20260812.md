# Dynamic Tiny Scientist Unity Replication Result — Seed 154

## Outcome

The registered live plumbing replication passed all six criteria. One
uninterrupted Unity embodiment collected discovery evidence, asynchronously
generated an L1 causal proposal with local Gemma, formally admitted it into a
null-only hypothesis pool, selected subsequent experiments, and verified the
new candidate using strictly later observations.

## Complete run

- 5,636 telemetry rows spanning 1,193.503 seconds.
- 37 clean observations and nine discarded confounded windows.
- 37 mushroom pickups, including 12 red.
- 147 PGNW action changes from 312 guidance decisions.
- Five stuck events, zero critical-hunger exposure, zero respawns, and zero
  survival failures.

## Discovery, formation, and admission

The four discovery observations were deliberately mixed and answer-neutral:

1. no-pickup, no probe rise;
2. red pickup, probe rise;
3. no-pickup, no probe rise; and
4. blue pickup, no probe rise.

At 171.232 seconds the planner froze observation cutoff 4 and started the
background proposal. At 178.113 seconds it admitted:

```text
L1 c red k blue e + t 10.0 q 0.5
```

The masked-greedy generation itself took 1.686 seconds, with 167 input and 20
output tokens. The formal compiler created
`dsl:red_causes_probe_rise@10s` at prior 0.20 against two 0.40 baselines,
`probe_is_spontaneous` and `no_tested_cause`. Admission produced zero posterior
updates and no production-rule authority.

## Strictly later verification

Observations 5-10 were all after the frozen cutoff. Four blue-negative controls
and one red-positive result preceded the second red-positive result. On clean
observation 10—the sixth held-out update—the dynamic candidate crossed 0.95 at
elapsed time 548.356 seconds with posterior 0.954535.

By the end, 33 held-out updates yielded:

| Candidate | Final posterior |
|---|---:|
| `dsl:red_causes_probe_rise@10s` | 0.999999999584 |
| `probe_is_spontaneous` | 0.000000000213 |
| `no_tested_cause` | 0.000000000203 |

The run recorded zero pre-admission evidence rejections because no update was
attempted at or before the cutoff. Thus discovery evidence selected the
candidate but was never counted as verification evidence.

## Asynchronous-control audit

Median telemetry spacing was 0.2120 seconds and the 99th percentile was 0.2212
seconds. The largest interval during background model loading/generation was
0.936 seconds. It caused no physics wedge, respawn, critical hunger, or survival
failure. The local proposal therefore ran concurrently enough for this bounded
Unity control loop, though the isolated sub-second pause remains measurable.

## Interpretation

This is the first successful live closure of:

`embodied discovery -> language-mediated hypothesis formation -> formal
admission -> PGNW experiment selection -> held-out embodied verification`.

The claim remains bounded. Gemma chose cause, comparison, direction, and delay,
but only within the existing red/blue/probe vocabulary and a symmetric finite
L1 grammar. This is dynamic hypothesis formation, not open-ended ontology
invention. One successful corrected smoke does not estimate reliability across
layouts or seeds.

## Next decision

Do not tune seed 154. The next scientific step is two or three new-seed,
new-layout confirmations with the formulation and verifier frozen. A separate
ablation should compare model-generated admission against a formally generated
candidate with identical downstream PGNW control if the goal is to quantify
the language model's unique contribution.
