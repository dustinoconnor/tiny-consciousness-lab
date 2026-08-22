# RPT-2 / HOT-4 Perceptual Intelligence Protocol — 22 August 2026

## Question

Do an organized recurrent perceptual representation (RPT-2) and a sparse,
smooth perceptual quality space (HOT-4) improve held-out object tracking and
choice, rather than merely adding consciousness-themed machinery?

This is a functional benchmark. Passing it would not demonstrate phenomenal
consciousness or validate recurrent processing theory or higher-order theory.

## Task

Each episode contains two continuously parameterized objects crossing in front
of the agent. Observations arrive as a short sequence of left/right detections.
Object qualities are visible early, partially masked during the crossing, and
absent at the final choice frame. The agent receives a continuous need vector
and must choose the final side occupied by the object whose qualities best
match that need.

Initial object identities, slot order, visibility masks, crossing speeds,
continuous qualities, and observation noise vary independently. Reserved
evaluation adds unseen quality combinations, stronger occlusion, and shifted
crossing dynamics. Chance choice accuracy is 50%.

## Matched conditions

1. `flat`: frame encoder plus order-insensitive temporal pooling and an
   unconstrained bottleneck.
2. `rpt2`: a parameter-matched recurrent scene encoder plus the unconstrained
   bottleneck.
3. `hot4`: the flat encoder plus a low-dimensional bottleneck trained to
   reconstruct continuous object qualities, preserve their pairwise geometry,
   and remain sparse.
4. `rpt2_hot4`: the recurrent encoder plus the same HOT-4 bottleneck.

Every action head receives the matched scene state, bottleneck, and need vector.
The bottleneck is therefore an auxiliary higher-order perceptual representation,
not an information choke point that can erase identity tracking. HOT-4 can only
help action by shaping the shared scene state or adding useful quality geometry.

The HOT-4 target contains object qualities only. It does not contain the
correct final side, action label, or need-match answer. All conditions receive
identical observations, need vectors, optimizer budgets, and action labels.
HOT-4 conditions use a fixed two-stage curriculum within that same budget:
action learning occupies the first half, after which answer-blind quality
reconstruction, geometry, and sparsity losses become active. Model selection
for HOT-4 begins only after those losses are active.

## Frozen data separation

- Development/training seeds: 3101--3104.
- Model-selection seeds: 3201--3202.
- Reserved evaluation seeds: 3301--3308.
- Reserved episodes are generated and evaluated only after implementation and
  regression tests pass. Reserved outcomes are never used to tune architecture,
  thresholds, losses, or training duration.

## Measures and prospective decisions

Primary intelligence measure: reserved final-side accuracy.

Mechanism measures:

- identity-binding accuracy from a linear probe of the perceptual state;
- organized-scene reconstruction error;
- Spearman correlation between bottleneck and true quality-space distances;
- active-coordinate fraction as a sparsity measure;
- a time-shuffled sequence lesion for recurrent conditions;
- a quality-target permutation lesion for HOT-4 conditions.

RPT-2 earns functional support only if `rpt2` exceeds `flat` reserved accuracy
by at least 0.05 and sequence shuffling removes at least 0.03.

HOT-4 earns functional support only if its held-out quality-distance correlation
is at least 0.70, its code uses no more than 60% active coordinates on average,
and `rpt2_hot4` exceeds `rpt2` reserved accuracy by at least 0.03. The permuted
quality-target control must reduce geometry correlation by at least 0.20.

The combined mechanism counts as added intelligence only if it beats both
single-mechanism conditions by at least 0.03 on reserved accuracy without an
in-distribution accuracy tax larger than 0.03. Otherwise the result is partial,
null, or harmful even if individual representation metrics look favorable.

Eight reserved seeds provide a bounded replication, not a population-level
claim. No failed seed is replaced and no threshold is changed after reserved
evaluation begins.
