# PGNW Immediate-Retreat Smoke Result — 12 August 2026

## Verdict

**The preregistered evidence-preservation smoke passed, with an important
limitation.** One requested red pickup completed an uncontaminated ten-second
observation and produced the expected delayed probe rise. Red-trial
contamination fell from 2/2 in seed 147 to 1/2 in seed 149. Thirty immediate
retreat decisions changed the selected MPC action, and no survival failure
occurred.

The deliberately adjacent spawn pair was not completely solved: its blue
mushroom was still acquired 4.26 seconds after red. This was an improvement over
2.14 seconds in seed 148, but short of the ten-second requirement. The clean red
trial occurred later elsewhere in the scene and did not itself require visible
food repulsion. The result therefore establishes system-level clean-red
feasibility, not that retreat alone caused the clean trial.

## Metrics

- Recorded duration: 593.82 seconds
- Experiments started/completed/discarded: 10 / 7 / 3
- Red pickups: 2
- Clean red-positive observations: 1
- Clean blue-negative observations: 6
- Immediate-retreat decisions/action changes: 30 / 30
- Isolation decisions/action changes: 73 / 54
- Total PGNW guidance decisions/action changes: 130 / 56
- Intervening pickups during observations: 3
- Final MAP: `red_causes_probe`
- Final red posterior: 0.93103
- Stuck events: 4
- Critical-hunger exposure: 0 seconds
- Survival failures: 0

The posterior did not cross 0.95 within ten minutes, but it correctly identified
the rule. The registered smoke criterion concerned preservation of a clean red
experiment rather than final convergence, so this is not a failure.

## Next step

The manipulation is now sufficiently present to justify a matched batch. Run
three counterbalanced committed-versus-passive pairs at 1,200 seconds each,
then measure whether clean-red yield and threshold reliability improve. Keep
the current authority and inference parameters frozen.
