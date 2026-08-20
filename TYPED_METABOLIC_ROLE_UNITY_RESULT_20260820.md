# Typed Metabolic Role Unity Verification — Seed 171

Seed 171 tested whether the passive typed learner could discover and verify
blue-specific nutritional relief from live Unity telemetry without being given
the correct color. Language-role labels retained zero motor authority.

## Result

- Normal bounded completion: 851 rows, 179.429 seconds, steps 0–850.
- Accepted observations: 7 clean single-pickup hunger transitions.
- Calibration sequence: red/no relief, yellow/no relief, blue/relief,
  blue/relief.
- Admission: `blue_restores_metabolic_reserve` after observation 4 at 32.341
  seconds, posterior confidence 0.975865.
- Held-out negative: yellow/no relief at 93.207 seconds.
- Held-out positive: blue/relief at 129.360 seconds.
- Promotion: `verified_held_out` after observation 6.
- Final evidence: 3 held-out confirmations, 0 contradictions, including 2
  positive blue observations and 1 negative non-blue observation.
- Final posterior confidence: 0.997130 for blue.
- Existing protective rule remained functional: red at 2.999 seconds, yellow at
  21.502 seconds, and one pending hazard cancellation with zero hazard cost.
- Controller safety: 2 recovered stuck events, zero survival failures, and zero
  respawns.

All preregistered live-learning checks passed. This establishes a second
independently admitted and post-cutoff verified actionable rule using the
existing objects. It does not yet establish behavioral arbitration between the
yellow protective rule and blue metabolic rule: the metabolic learner remained
passive during this run, and the language classifier remained observational.
