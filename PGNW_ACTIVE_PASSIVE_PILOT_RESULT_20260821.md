# PGNW Active-versus-Passive Pilot Result

The frozen eight-trial pilot produced a clean null result. Active dual-rule
PGNW authority did not improve the first post-red resource choice over the
passive control on this saved fork.

## Primary result

| Condition | Blue first after red | Total | Rate |
| --- | ---: | ---: | ---: |
| Active | 4 | 4 | 100% |
| Passive | 4 | 4 | 100% |

- Absolute rate difference: `0.0`
- Two-sided Fisher exact test: `p = 1.0`

Every trial completed the same safe physical sequence: red -> blue -> yellow.
There were no costs, stuck events, survival failures, or respawns in either
condition. Each trial cancelled one pending hazard.

## Trial matrix

| Order | Trial | Condition | Seed | Red (s) | Blue (s) | Yellow (s) | Blue influence | Yellow influence |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A1 | Active | 181 | 2.529 | 4.633 | 7.615 | 0 | 6 |
| 2 | P1 | Passive | 181 | 2.525 | 4.632 | 7.620 | 0 | 7 |
| 3 | P2 | Passive | 182 | 2.527 | 4.638 | 7.628 | 0 | 6 |
| 4 | A2 | Active | 182 | 2.557 | 4.689 | 7.721 | 0 | 8 |
| 5 | A3 | Active | 183 | 2.541 | 4.681 | 7.672 | 0 | 6 |
| 6 | P3 | Passive | 183 | 2.530 | 4.641 | 7.644 | 0 | 6 |
| 7 | P4 | Passive | 184 | 2.507 | 4.662 | 7.659 | 0 | 7 |
| 8 | A4 | Active | 184 | 2.538 | 4.650 | 7.637 | 0 | 6 |

Mean pickup times were nearly identical. Active trials reached red, blue, and
yellow at 2.541, 4.663, and 7.661 seconds; passive trials reached them at 2.522,
4.643, and 7.638 seconds.

## Interpretation and boundary

This fork is not a discriminating assay of added blue PGNW authority. The
passive controller already chose blue first in every paired seed, and blue
authority changed zero MPC actions in every active trial. The result therefore
does not show that active PGNW is ineffective in general. It shows that the
present geometry and controller baseline leave no observable choice advantage
for active PGNW to create.

Do not repeat more seeds on this unchanged fork or describe the pilot as a
positive causal result. A subsequent assay should use separately calibrated,
counterbalanced geometries where passive choice is not already deterministically
blue, then evaluate active authority once on untouched layouts.

## Operational note

The first A2 launch stopped before Python imported the experiment because the
shell resolved an interpreter without NumPy. It produced no telemetry and no
behavioral outcome. Its copied resource fixture was archived under
`tmp/failed_pgnw_pilot_launches/`; Unity was reset, and A2 was launched once
with the intended environment. No trial was replaced or repeated after an
outcome was observed.
