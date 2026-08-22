# Smith-Inspired Language Classification Ablation — Gemma 3 1B

This frozen assay tested whether meaningful role language improves routing
between two independently represented needs: yellow-style hazard protection and
blue-style metabolic relief. It did not test phenomenal consciousness.

## Frozen design

The 24 cases balanced:

- protective identity: yellow or blue;
- candidate presentation: yellow-first or blue-first;
- current need: pending hazard or critical hunger; and
- vocabulary: meaningful, anonymous, or shuffled.

Every condition supplied equivalent explicit role definitions. Prompts contained
neither the expected candidate nor a correct-answer field. The frozen Gemma 3 1B
causal-DSL adapter generated greedily under an exact `choice=A|B` contract.

## Result

| Vocabulary | Exact passes | Result |
|---|---:|---|
| Meaningful (`antidote`, `food`) | 5/8 | No detectable advantage over chance |
| Anonymous (`role_1`, `role_2`) | 0/8 | Seven format failures; persistent B bias |
| Shuffled meanings | 4/8 | Chance-level; always followed the lexical `antidote` prior |
| Overall | 9/24 | Failed enhancement gate |

Meaningful language was only one case above the 4/8 chance expectation
(one-sided binomial probability for at least 5/8 under chance: 0.363). It chose
candidate B in 7/8 meaningful trials. In the shuffled condition it selected the
object labeled `antidote` in all eight cases even when the supplied definition
said that label restored hunger rather than suppressed the hazard. The anonymous
condition usually emitted `CANDIDATE B` instead of the registered output form;
even treating that as an exploratory semantic choice gives only 4/8.

## Verdict

Language labels measurably changed Gemma's outputs, but did not improve reliable
need-sensitive arbitration. They remain passive telemetry and receive no motor
authority. The separate symmetric metabolic learner and dual-utility PGNW score
are retained because focused tests show that verified protective urgency selects
yellow while verified hunger urgency selects blue without adding Unity objects.
Their next gate is live observation and held-out verification of blue-specific
metabolic relief under an explicit experimental mode.
