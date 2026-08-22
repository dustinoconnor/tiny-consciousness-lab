# Bounded Active-PGNW Unity Smoke Result — 10 August 2026

## Verdict

**The preregistered feasibility smoke passed.** The planner completed 11
uncontaminated experiments, performed 11 posterior updates, logged 96 bounded
guidance decisions and one changed MPC selection, and produced no survival or
safety failure attributable to scientific targeting.

This is evidence that the closed Unity loop can select experiments, observe the
result of the passive delayed probe, reject contaminated windows, and update a
five-model causal posterior online. It is not yet evidence that PGNW guidance
improves navigation or experiment-acquisition efficiency.

## Run

- Seed: `141`
- Recording: `outputs/unity_shadow/pgnw_active_science_smoke_20260810.jsonl`
- Recorded duration: 1,193.508 seconds (19 minutes 53.5 seconds)
- Telemetry rows: 5,674
- Control: terrain shadow policy, recurrent MPC/AIR systemic conductor,
  bounded adaptive GNW, and bounded PGNW experiment guidance at a maximum
  requested weight of `0.03`

## Scientific evidence

| Measure | Result |
|---|---:|
| Experiments selected | 22 |
| Experiments started | 21 |
| Experiments completed uncontaminated | 11 |
| Experiments discarded as confounded | 10 |
| Posterior updates | 11 |
| Completed red observations | 4/4 probe rise |
| Completed blue observations | 7/7 no probe rise |
| Final MAP hypothesis | `red_causes_probe` |
| Final MAP posterior | 0.999134 (99.9134%) |
| Final red-versus-all odds | 1,153.99:1 |
| Posterior entropy | 2.32193 to 0.01014 bits |
| Information gained | 2.31179 bits |

The uncontaminated evidence was internally perfect for the implemented world:
every completed red pickup was followed by the delayed passive signal and no
completed blue pickup was. The information-gain sum exactly matches the
reduction from the uniform five-model prior to the final posterior.

The robot consumed 32 mushrooms, including six red mushrooms, and generated six
delayed causal-probe events. Ten observation windows were excluded because an
additional pickup occurred during the ten-second delay; those windows did not
update the posterior.

## Authority and protocol audit

- The planner made 96 effective guidance decisions across 96 frames.
- Effective guidance never exceeded `0.02961`, below the registered `0.03`
  ceiling.
- Guidance changed the selected MPC action once, at 727.50 seconds, while both
  colors were visible and the requested experiment was red.
- MPC was already engaged; the planner did not activate it.
- Thirteen of 21 encountered pickups did not match the requested color. They
  were logged as protocol mismatches and interpreted according to the action
  actually observed, as preregistered.
- The planner requested red or blue observations in this run and did not select
  a no-pickup wait.

The high mismatch count and single changed action show that most evidence was
collected opportunistically by the existing controller rather than caused by
strong PGNW steering. That does not invalidate the online causal updates, but
it sharply limits any active-navigation claim.

## Safety audit

- Survival failures: 0
- Trap or escape-teacher failures: 0
- Critical-hunger exposure: 0 seconds
- Final survival state: `stable`
- Stuck events: 1
- Unstuck respawns: 0

The single stuck event occurred at 583.05 seconds while effective PGNW guidance
was zero. Stable fallback was active for 1,180 recorded frames, during which the
PGNW safety gate prevented guidance. This run therefore contains no observed
safety failure attributable to scientific targeting.

## Claim boundary and next test

The result establishes bounded-loop feasibility for autonomous experiment
selection and posterior updating in this Unity scene. It does not establish a
navigation benefit, causal superiority over passive observation, generalization
to new causal rules, or natural-world scientific reasoning.

The next registered comparison should be matched passive versus bounded PGNW
over multiple counterbalanced seeds, measuring clean-experiment yield,
time-to-correct posterior, request compliance, pickup rate, survival, and stuck
events. Because this smoke changed only one MPC selection, a null navigation
difference would be unsurprising; increasing authority or target availability
would require a separate protocol rather than a post-hoc reinterpretation of
this run.
