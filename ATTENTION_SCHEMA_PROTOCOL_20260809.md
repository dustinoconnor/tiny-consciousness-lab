# Predictive Attention-Schema Protocol — 9 August 2026

## Question

Does a compact learned prediction of the GNW router's next attentional focus
improve control during rapid context changes and transient distractors, beyond
the existing adaptive ignition governor?

## Representation

The attention schema is a small softmax predictor trained on separate synthetic
router trajectories. It receives only quantities available inside the router:
the current and preceding specialist-bid distributions, the currently broadcast
specialist, bid entropy, and the leading bid margin. Its target is the next
frame's local specialist focus. It never receives the hidden context label,
optimal specialist, sampled outcome, or evaluation target.

At evaluation, the prediction made on frame `t-1` is blended with the observed
bids on frame `t` before the unchanged adaptive ignition governor decides
whether to hold or release the broadcast. The one-frame lag prevents future
observation leakage.

## Frozen conditions

1. `adaptive_gnw`: existing governor with no schema input.
2. `ast_schema`: same governor and sensory bids, plus the correctly bound
   one-frame-ahead schema prediction.
3. `ast_schema_shuffled`: same trained predictor and compute, but its output
   roles are deterministically permuted at evaluation.
4. `ast_schema_lesion`: the predictor is trained and evaluated for diagnostic
   accuracy but disconnected from the governor.

The schema blend is fixed at `0.30`; no evaluation-seed tuning is permitted.
Conductor, governor, and schema training use seed-separated episodes from the
held-out evaluation blocks.

## Primary measures

- utility per step;
- optimal routing in the first two frames after a context change;
- resistance to one-frame distractors;
- unnecessary handoff rate;
- next-focus prediction accuracy, including boundary and distractor subsets.

## Decision rule

The AST condition is considered supported on this benchmark only if the
paired-seed 95% interval for utility over the ordinary adaptive governor is
positive, the interval over the shuffled-schema control is positive, and
neither boundary routing nor distractor routing is reduced by more than 0.01.
Prediction accuracy alone is insufficient.

## Claim boundary

A positive result would show that an explicit predictive summary of internal
router allocation improves this synthetic controller. It would not establish
the biological Attention Schema Theory, phenomenal consciousness, awareness,
or benefit in Unity until separately replicated there. A null result would be
informative because the existing adaptive governor already explicitly tracks
its active broadcast, challengers, persistence, and reward.
