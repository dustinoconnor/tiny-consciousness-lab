# Prompt-Normalized Ordered-Formulation Held-Out Result

## Question

Can a deterministic, answer-blind normalization layer remove the temporal-wording
and evidence-position sensitivity observed in the frozen 2 x 2 x 2 replication,
without tuning against those eight completed conditions?

## Frozen prior result

The earlier counterbalanced replication remains unchanged at 6/8. Its two misses
occurred under `then` wording when the true suppressor appeared first. Those eight
conditions were used only to motivate normalization; they were not used as an
evaluation set for this experiment, and the new wrapper was not tuned against
their answers.

## Deterministic normalization

`normalized_ordered_formulation.py` receives structured records rather than raw
prose. It applies the same answer-blind transformation to every condition:

1. Sort the two grounded second-action labels lexically and bind them to
   `role_0` and `role_1`.
2. Sort evidence records by that anonymous role.
3. Express the temporal relation with the fixed predicate `initiator=red` and
   `relation=before`.
4. Retain only observed suppression and episode counts.

The resulting grammar is:

```text
role_N|initiator=red|relation=before|suppressed=X|episodes=Y
```

The mapping never consults which color is the assigned suppressor, the observed
outcome, raw record order, or surface wording. A content-free neutral calibration
also treats both anonymous roles symmetrically.

## Newly reserved held-out matrix

The evaluation crossed four factors for 16 conditions:

| Factor | Levels |
|---|---|
| Assigned suppressor | blue, yellow |
| New count regime | 2/3 vs 1/4; 3/5 vs 0/4 |
| Input record order | forward, reverse |
| Unseen surface rendering | `first before second`; `second follows first` |

The two count regimes were not present in the earlier eight-condition matrix.
Surface rendering and input ordering were deliberately discarded before Gemma
saw the evidence. They remain in the artifact so invariance can be audited.

## Result

Gemma 3 1B scored **8/16 (50%)** conditions, corresponding to **16/32** held-out
synthetic episodes. Every wording/order rendering of the same semantic evidence
produced the same canonical prompt and the same decision, so normalization did
remove the measured surface and position variance.

However, Gemma selected `role_1`, mapped to yellow, in all 16 conditions:

| True suppressor | Correct | Result |
|---|---:|---|
| blue | 0/8 | selected yellow in every condition |
| yellow | 8/8 | selected yellow in every condition |

The minimum selected calibrated margin was 0.03125. There were four unique
canonical semantic prompts because the order and wording variants intentionally
collapse to identical inputs.

## Interpretation and limitation

This is a useful negative result. The wrapper succeeds as a representation
invariance mechanism, but it does not establish answer generalization: removing
surface variation exposes a persistent anonymous-role/color preference. The
held-out score therefore cannot support a claim that prompt normalization repairs
causal selection.

This matrix is synthetic and answer-balanced; it is not a set of new physical
Unity episodes. Its repeated wording/order cells test invariance, while only four
unique semantic prompts test selection. This held-out set is now frozen and must
not be used to tune a repair. A defensible follow-up would develop any role-bias
repair on a separate calibration set, then evaluate once on another reserved set
with new labels, new count regimes, and ideally three or more candidates or new
Unity observations.

## Verification

- Focused normalization regressions: 3/3 passed.
- Combined ordered-formulation and typed-domain regressions: 19/19 passed.
- Full repository suite: 251/251 passed.
- Raw artifact: `outputs/gemma_ordered_normalized_holdout_20260816.json`.
