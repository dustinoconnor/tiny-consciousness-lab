# Calibrated Anonymous-Role Decoder: Reserved Evaluation

## Purpose

The frozen prompt-normalization holdout scored 8/16 because Gemma selected the
anonymous role mapped to yellow in every condition. This experiment repairs that
role bias on a separate calibration set and evaluates the frozen repair once on
a disjoint reserve. The previous 16-condition artifact was not modified, read as
training data, or used to select parameters.

## Calibration design

The calibrator uses three candidates so the repair cannot specialize to one
binary blue/yellow contrast. Grounded calibration labels are `amber`, `jade`, and
`violet`; lexical normalization removes those labels before Gemma sees evidence
and maps them to `role_0`, `role_1`, and `role_2`.

The nine calibration conditions cross all three high-rate role assignments with
three count regimes:

- 3/4 versus 1/5 controls
- 5/7 versus 1/6 controls
- 7/9 versus 2/8 controls

Every anonymous role is the high-rate role exactly three times. The frozen score
equation is identical for every role:

```text
score(role) = neutral_adjusted_gemma_gain(role)
              - balanced_mean_role_offset(role)
              + 0.1387319566 * observed_suppression_rate(role)
```

The role offsets are means over the fully balanced matrix. One shared rate weight
is the smallest value that gives the unique maximum observed rate a 0.01 margin
on calibration. It is inferred from rates, not a supplied color answer, and there
are no per-color or per-condition exceptions. Offset-only calibration scored 3/9;
the frozen fused calibrator scored 9/9. Its SHA-256 identifier is
`4e5b75819920a9fa81f8f5791c310eaabbecef95f2d5b7f9e760eaf1898fc64c`.

## Untouched reserved design

After the calibration artifact was frozen, it was evaluated once on nine
conditions using disjoint grounded labels (`cobalt`, `saffron`, `umber`) and
three unseen count regimes:

- 5/8 versus 1/6 controls
- 4/7 versus 0/5 controls
- 7/10 versus 2/9 controls

Each of the three anonymous roles was the true high-rate role once per regime.
The evaluation command refuses to overwrite an existing reserved artifact, and
the frozen calibration hash is verified before model loading.

## Reserved result

The calibrated decoder scored **9/9 (100%)**. Each role assignment scored 3/3,
and every count regime scored 3/3. The minimum selected margin was **0.0133032**.

Post-evaluation ablations calculated from the same stored logits were:

| Decoder | Reserved accuracy |
|---|---:|
| Neutral-adjusted Gemma only | 5/9 |
| Mean role-offset correction only | 3/9 |
| Frozen offset + shared rate fusion | 9/9 |

No additional model calls or parameter changes were made for these ablations.

## Interpretation and limits

This supports a narrow conclusion: a role-balanced calibration set plus one
symmetric evidence-strength feature repaired the observed anonymous-role bias on
this reserved synthetic matrix. It does not show that Gemma independently learned
fraction comparison. The formal rate feature directly supplies task-relevant
numeric evidence, so this is properly described as a neuro-symbolic calibrated
decoder, not unconstrained language-model reasoning.

The reserve contains only nine synthetic conditions, grounded labels disappear
during canonicalization, and no new physical Unity episodes were collected.
Further claims require a larger preregistered reserve with closer rate gaps,
unequal control rates, additional candidate counts, and live telemetry. This
reserved artifact is now frozen and must not be used to tune those extensions.

## Verification

- New focused regression suite: 7/7 passed.
- Combined ordered-formulation/typed-domain suite: 26/26 passed.
- Full repository suite: 258/258 passed in 5.38 seconds.
- Calibration artifact: `outputs/gemma_role_bias_calibration_20260816.json`.
- One-time reserve artifact: `outputs/gemma_role_bias_reserved_20260816.json`.
