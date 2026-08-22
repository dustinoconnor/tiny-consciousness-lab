# PGNW Reserved Side-Blocked Amendment

This amendment is frozen before any reserved Unity layout is applied or any
reserved outcome is observed.

## Calibration finding and design consequence

Passive calibration was exactly color-balanced (blue 2/4, yellow 2/4) but chose
the right branch in 4/4 runs. The original side-nondeterminism gate therefore
failed. Repeatedly tuning geometry until the controller appears unbiased would
discard a measured nuisance variable. Instead, the reserved assay treats side
as a blocked factor: every active run is paired with a passive run on the same
layout and seed, and blue-left and blue-right layouts are equally represented.

This amendment does not convert calibration into evidence for active PGNW. It
only specifies how reserved evidence will control the discovered side bias.

## Prospective reserved geometry repair

The original unrun reserved red distance (3.4 m) was too close given the
calibration frame-zero overlap. Before opening either reserved layout, red is
moved to 5.5 m forward. Branches are moved to 10.0 m forward and +/-3.8 m
lateral, then the whole fork is rotated 18 degrees. Blue and yellow remain exact
equal-distance mirrors. Both layouts must start with pickup totals 0/0/0.

The two reserved layouts were applied and frozen before any reserved outcome:

- Blue-left scene SHA-256:
  `95570447b572b14be9e605e55665f4698d66832e06474aa5eb8d013ea91571be`.
- Yellow-left scene SHA-256:
  `6fc49eac8546c2f7a3817e9c2bc27c1ff3ffa3619bf4e7f5cd1d190a03bf2c7b`.
- Red is at `(1.6995937, 1.6608107)` in both layouts. The mirrored branch
  coordinates are `(-0.52384424, 7.11483)` and `(6.704185, 4.7663)`.
- Separate equal-evidence resource fixtures bind feature labels to those
  coordinates without favoring either candidate.

## Frozen eight-run matrix

| Order | Trial | Layout | Condition | Seed |
| ---: | --- | --- | --- | ---: |
| 1 | RBL-A1 | Blue left | Active | 187 |
| 2 | RBL-P1 | Blue left | Passive | 187 |
| 3 | RYL-P1 | Blue right | Passive | 188 |
| 4 | RYL-A1 | Blue right | Active | 188 |
| 5 | RYL-A2 | Blue right | Active | 189 |
| 6 | RYL-P2 | Blue right | Passive | 189 |
| 7 | RBL-P2 | Blue left | Passive | 190 |
| 8 | RBL-A2 | Blue left | Active | 190 |

Every run is bounded to 20 seconds, and Unity is stopped and restarted between
runs. Active uses `bounded_dual_verified`; passive computes the same candidates
and scores with dual authority fixed at zero. All other causal, metabolic,
controller, safety, and memory inputs remain frozen.

## Outcomes and claim boundary

- Primary: blue is the first blue/yellow pickup after red. Missing or tied
  outcomes count against blue-first.
- Report condition rates, paired discordance, and results separately for
  blue-left and blue-right blocks.
- Report active pre-pickup action influence, pickup times, stuck events, costs,
  survival failures, and respawns.
- An added-PGNW benefit requires active improvement specifically in blue-left
  pairs, where calibration predicts the right-side passive baseline will prefer
  yellow, plus nonzero active motor influence before blue pickup.
- Four pairs are a descriptive pilot. No conventional significance claim is
  made even if all pairs favor active.
- No trial replacement, seed search, geometry change, or threshold change is
  allowed after the first reserved run begins.
