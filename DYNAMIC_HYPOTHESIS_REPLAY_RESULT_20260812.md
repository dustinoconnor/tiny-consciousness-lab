# Dynamic Tiny Scientist Admission Replay — 12 August 2026

## Question

Can the local language model formulate a causal candidate that was absent from
the initial Bayesian pool, can the formal layer admit it without granting
authority, and can strictly later Unity evidence verify or reject it?

This is a post-development replay integration check. It was not preregistered
and is not an independent replication.

## Discovery and formulation

Discovery evidence came only from the completed seed-149 Unity smoke: one clean
red pickup with causal-probe delta `+0.339898` at the ten-second delay and six
clean blue controls with mean delta approximately `-0.007995`.

The frozen local Gemma 3 1B L1 adapter with mechanism-guided masked-greedy
decoding generated:

```text
L1 c red k blue e + t 10.0 q 0.5
```

Generation used 190 input tokens and 20 output tokens and took 2.538 seconds on
MPS. The existing verifier accepted the distinct red/blue role binding and
evidence-grounded direction and delay. The formal admission compiler produced
`dsl:red_causes_probe_rise@10s` with action-conditional rise probabilities
`[0.92, 0.08, 0.05]` for red, blue, and no-pickup observations.

The candidate was not among the pool's two starting explanations,
`probe_is_spontaneous` and `no_tested_cause`. It entered with prior 0.20 versus
0.40 for each baseline, zero held-out updates, and no production-rule or direct
motor authority. Its admission cutoff was discovery observation 7.

## Held-out chronological replay

The chronological clean observations from committed seed 150 were then applied
as observations 8 onward. The dynamic candidate did not receive or reuse any
seed-149 discovery update.

Nine initial clean blue-negative controls moved its posterior only from 0.20 to
0.272269. The first later clean red-positive observation raised it to 0.871836;
the second raised it above the verification threshold to 0.991050. After all 15
later clean observations, its posterior was 0.999877. The final competing
probabilities were:

| Candidate | Posterior |
|---|---:|
| `dsl:red_causes_probe_rise@10s` | 0.999877 |
| `probe_is_spontaneous` | 0.0000979 |
| `no_tested_cause` | 0.0000248 |

After admission, expected-information-gain selection requested red observations
while they remained maximally discriminative. Once the causal candidate was
nearly settled, the preferred unresolved test changed to no-pickup observation.

## Interpretation and boundary

This closes the loop in offline chronological replay:

`Unity discovery evidence -> Gemma L1 proposal -> formal admission -> PGNW test
selection -> later Unity verification`.

It is stronger than merely inserting a hand-written candidate, because the
actual local model selected the cause, comparison, direction, and delay. It is
not yet a live autonomous Unity loop: formulation occurred between preserved
recordings, not asynchronously during one running embodiment. It is also not
open-ended ontology construction. The first compiler deliberately supports only
the currently actionable red/blue pickup vocabulary and binary delayed probe
outcome.

The old verifier internally names `e +` as `pressure_increase`; the admission
compiler explicitly translates that legacy semantic name to a rise in the new
passive causal-probe signal. This should be made probe-native before extending
the vocabulary.

## Next gate

The next implementation should run proposal generation asynchronously after a
frozen discovery quota, then admit it into the live planner without pausing
Unity. One short live smoke is sufficient to test plumbing: it must log the raw
L1 proposal, cutoff, admission, pre-admission rejection count, subsequent
experiment choices, and at least one held-out posterior update. Only after that
passes are matched confirmation runs warranted.
