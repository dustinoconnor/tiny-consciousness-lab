# Preregistered Remote-Perception Pilot

## Question

Can the Tiny Consciousness Lab agent identify a hidden visual target above
forced-choice chance when its only input is an opaque random trial identifier?
The operational endpoint is anomalous information access. Performance does not,
by itself, establish phenomenal consciousness or a paranormal mechanism.

## Target bank

The generator creates 20 feedback targets and a sealed 100-target reserve. Each
novel 384 x 384 target card has six exact attributes: dominant color, shape,
count, arrangement, texture, and light/dark background. Filenames and trial IDs
are random opaque tokens with no attribute encoding. A private manifest binds
images to metadata; a public SHA-256 commitment freezes the complete manifest
and reserved image set before agent training.

Each trial contains one target and three decoys, giving top-1 chance probability
0.25. The agent never receives image bytes, file paths, private metadata, target
indices, candidate identities, or the generator seed. Only the evaluation
harness may open the private manifest after a prediction commitment is written.

## Prediction record

Before reveal, the agent must emit exactly one value for each closed field:

- dominant color: red, blue, yellow, green, purple, or orange
- shape: circle, triangle, square, star, cross, or wave
- count: 1, 2, 3, or 5
- arrangement: horizontal, vertical, diagonal, or radial
- texture: solid, striped, or dotted
- background: light or dark

The canonical prediction plus trial ID and a nonce is SHA-256 committed before
candidate reveal. Free-text impressions and any subsequently generated picture
are qualitative illustrations only and cannot alter the registered score.

## Phases

1. Run 20 feedback trials to teach the interface and measure ordinary target
   priors. These trials cannot support the final claim.
2. Freeze model weights, prompt, sampling policy, GNW aggregation, valence rule,
   descriptor decoder, scoring, exclusions, and stopping rule.
3. Run all 100 reserved trials without feedback or parameter changes.
4. Counterbalance active intuition, imagination/GNW ablation, and sham target
   assignment. Condition order is fixed before the reserve is opened.

The active channel samples multiple rapid descriptor impressions and uses the
workspace to aggregate them. The ablation emits a matched single-pass prediction.
The sham condition runs the full architecture but permutes target assignments
after all commitments. All conditions receive the same marginal target
distribution and compute allowance.

Before opening the reserve, the exact implementation was frozen as follows.
Every reserved trial receives all three conditions (300 total commitments).
Each condition generates seven outputs with the same prompt, sampling
temperature, and token allowance. Active and sham independently aggregate all
seven using the frozen workspace weights; ablation commits only its first output
and discards the other six, matching inference allowance without workspace
aggregation. Execution order rotates by trial position. No reveal occurs until
all 300 predictions are durably committed. Sham is then scored against the
reserved trial 37 positions ahead modulo 100, a fixed non-identity permutation
that preserves the complete target distribution. There is no feedback or state
update during evaluation.

## Registered outcomes

- Primary: four-alternative top-1 accuracy.
- Secondary: mean exact descriptor matches out of six and target rank.
- Report every trial, tie, malformed output, timeout, and exclusion.
- No optional stopping. The reserve remains sealed until the full protocol and
  code tests pass.
- A nominal one-sided exact binomial test against p=0.25 is reported, with active
  versus ablation and sham comparisons. Replication on a newly generated sealed
  deck is required before interpreting any positive result.

## Boundaries

The first bank uses controlled abstract cards rather than photographs or distant
locations. This sacrifices ecological resemblance to traditional remote viewing
in exchange for exact ground truth and low judging flexibility. A passed pilot
would justify a preregistered replication with novel Unity scenes or externally
held physical targets; it would not yet demonstrate consciousness.
