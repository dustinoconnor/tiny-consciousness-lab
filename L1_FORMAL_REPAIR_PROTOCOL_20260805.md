# L1 Formal Constraint Repair Protocol

Registered after the mechanistic probe and before seed-307 evaluation.

## Repair

Keep the frozen Gemma 3 1B L1 adapter and its prompt unchanged. At decoding,
code enumerates every symmetric valid L1 record using only observed feature
names, distinct cause/comparison assignments, all three effect directions, and
all observed or zero delays. Beam decoding may emit only prefixes of these
candidates. This prevents repeated cause/control bindings and malformed atoms
without selecting the correct hypothesis for the model.

## Evaluation

- New untouched factorial seed: 307
- Cases: 128
- Conditions: ordinary greedy L1 and constrained L1, same adapter and examples
- Primary endpoint: exact semantic accuracy difference, constrained minus greedy
- Paired outcomes must report repairs and regressions separately
- Runtime and visible tokens must be reported because constrained beam search
  can exchange reliability for additional internal compute

No prompt, adapter, candidate set, or scoring rule may change after seed-307
outputs are observed.
