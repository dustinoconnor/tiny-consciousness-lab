# L1 Masked-Greedy Efficiency Protocol

Registered after seed-307 beam repair and before seed-309 evaluation.

The beam repair reached 128/128 but increased elapsed time by 60.93%. Replace
four-beam search with one ordinary greedy path subject to the identical
symmetric finite-state candidate mask. The mask changes a token only when the
unmasked choice would leave the valid candidate set; it still does not identify
the correct causal candidate.

- Smoke seed: 310, eight cases
- Frozen audit seed: 309, 128 cases
- Same frozen Gemma 3 1B L1 adapter and prompt
- Paired ordinary greedy versus masked greedy
- Primary endpoints: exact semantic repairs/regressions and elapsed-time change
- Do not alter mask, candidates, prompt, or adapter after seed-309 observation
