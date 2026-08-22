# PGNW Reserved Side-Blocked Pilot Result

## Outcome

The frozen eight-trial reserved pilot is a clean null for added PGNW choice
benefit. Active and passive each selected blue first in 2/4 trials. All four
matched active/passive pairs were concordant, so there were zero active wins
and zero passive wins (exact paired McNemar p = 1.0).

The result is not an authority failure. In both difficult blue-left active
trials, bounded dual-rule PGNW authority accumulated 21 action changes before
the first branch pickup (42 total). Passive accumulated zero. Despite that
physical intervention, active and passive both selected the right branch:
yellow when blue was left, and blue when blue was right.

## Frozen matrix

| Trial | Blue side | Mode | Seed | Red (s) | First branch | Blue (s) | Yellow (s) | Pre-choice influence |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| RBL-A1 | left | active | 187 | 1.269 | yellow | 7.370 | 5.700 | 21 |
| RBL-P1 | left | passive | 187 | 1.265 | yellow | 5.465 | 3.166 | 0 |
| RYL-P1 | right | passive | 188 | 1.266 | blue | 2.741 | 5.080 | 0 |
| RYL-A1 | right | active | 188 | 1.278 | blue | 2.772 | 5.101 | 0 |
| RYL-A2 | right | active | 189 | 1.280 | blue | 2.767 | 5.109 | 0 |
| RYL-P2 | right | passive | 189 | 1.276 | blue | 2.766 | 5.099 | 0 |
| RBL-P2 | left | passive | 190 | 1.263 | yellow | 5.465 | 3.152 | 0 |
| RBL-A2 | left | active | 190 | 1.267 | yellow | 7.415 | 5.734 | 21 |

Block results were active 0/2 versus passive 0/2 with blue left, and active
2/2 versus passive 2/2 with blue right. The active blue-left intervention
delayed both branch pickups by roughly two seconds but did not change their
order.

## Validity and safety

All eight trials ran once in the preregistered order and lasted approximately
20 seconds. Every first telemetry frame contained red/blue/yellow totals
0/0/0. Each trial acquired red, yellow, and blue; each recorded one successful
yellow cancellation and zero hazard-cost events. There were zero stuck events,
survival failures, or respawns.

The conclusion is bounded: the current `bounded_dual_verified` steering signal
does not overcome this layout's deterministic right-side attraction. The run
does not show that PGNW generally lacks behavioral utility; it shows that merely
raising action authority is insufficient when the target commitment and local
food-seeking controller disagree. A next experiment should inspect and repair
that arbitration interface before collecting more seeds or layouts.
