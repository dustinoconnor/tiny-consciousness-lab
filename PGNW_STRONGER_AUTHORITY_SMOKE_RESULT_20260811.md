# PGNW Stronger-Authority Smoke Result — 11 August 2026

## Verdict

**The manipulation failed; do not run an overnight comparison.** Raising the
maximum guidance weight to `0.12` and the eligible MPC margin to `0.25` produced
13 effective guidance decisions but changed zero MPC selections. The
preregistered requirement was at least five pre-convergence action changes plus
a requested-color pickup following guidance.

The failure is specific to motor authority, not safety or posterior integrity.
The run had zero stuck events, zero critical-hunger exposure, and zero survival
failures. The causal updater correctly refused to learn from contaminated red
trials.

## Run

- Seed: 146
- Requested duration: 600 seconds
- Recorded duration: 594.28 seconds
- Initial position: `(0, -0.008, 0)`
- Initial pickups: 0
- Maximum effective PGNW weight: 0.09299
- Guidance decisions: 13
- Changed MPC selections: 0

## Evidence outcome

- Total pickups: 10
- Red pickups: 2
- Experiments started: 8
- Clean experiments completed: 6
- Confounded experiments discarded: 2
- Clean red observations: 0
- Clean blue observations: 6
- Final MAP: `no_tested_cause`
- Final red-cause posterior: 0.44923

The red pickups occurred at 2.96 and 112.87 seconds while red was requested,
but each ten-second observation window contained an additional blue pickup.
Both were therefore discarded exactly as preregistered. The six uncontaminated
blue-negative observations cannot distinguish a red-only cause from no tested
cause, so failure to cross the posterior threshold is expected.

Neither red pickup followed a changed PGNW-guided action because the guidance
never changed an action. They are opportunistic request matches, not evidence
of active experimental control.

## Decision

Do not increase sample size for this configuration. Additive score weighting is
still too weak to alter the selected action, even after increasing both weight
and ambiguity range. The next manipulation should be a constrained target
commitment rather than another larger scalar:

1. preserve collision masking and every existing survival veto;
2. admit only safe MPC actions within a preregistered score-regret bound of the
   ordinary best action;
3. among those admissible actions, choose the action most aligned with the
   requested visible target for a short bounded commitment;
4. verify at least five pre-evidence action changes in another short smoke;
5. run matched overnight trials only after that manipulation check passes.

This would grant experiment-selection authority a real but explicitly bounded
behavioral channel without handing PGNW unrestricted motor control.
