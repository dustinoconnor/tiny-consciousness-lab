# Dynamic Tiny Scientist Unity Replication Protocol — Seed 154

This is a one-run plumbing replication after the observed seed-153 failure.
Everything remains as registered in
`DYNAMIC_HYPOTHESIS_UNITY_SMOKE_PROTOCOL_20260812.md` except the two repairs
fixed before seed 154:

1. rejected proposal generation is terminal for the run; and
2. discovery feature rows are always red then blue, with mean deltas rounded to
   three decimals and sub-`0.0005` magnitudes normalized to zero.

The semantic verifier, model, adapter, candidate grammar, four-clean-observation
minimum, requirement for both colors, admission prior, held-out cutoff, PGNW
selection, committed control limits, isolation, retreat, safety vetoes,
1,200-second duration, and pass criteria are unchanged.

Seed 154 and output
`outputs/unity_shadow/dynamic_hypothesis_smoke_seed154_20260812.jsonl` are frozen
before execution. A rejected proposal is a formulation failure and will not be
retried or interpreted post hoc. A proposal admitted without a later clean
update remains inconclusive. Passing requires admission plus at least one
strictly post-cutoff update and zero survival failures.
