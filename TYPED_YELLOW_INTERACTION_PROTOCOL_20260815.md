# Typed Yellow Interaction Protocol — 2026-08-15

## Question

Can the embodied Tiny Scientist compose a newly grounded object concept with an
existing causal concept, choose ordered interventions, and verify a two-action
rule without receiving the semantic label or correct interaction in its prompt?

This is a bounded engineering test inspired by Mao et al.'s concept-centric
neuro-symbolic representation and PDSketch's separation of typed domain
structure from learned transition details. It is not a reproduction claim for
either paper.

## Typed vocabulary supplied to the agent

- `red_mushroom : pickup`
- `blue_mushroom : pickup`
- `yellow_flower : pickup`
- `consume(x) : pickup -> action`
- `before(a, b) : action x action -> ordered_relation`
- `probe_rise : delayed_observation`
- `suppresses(x, y) : ordered_relation x delayed_observation -> candidate_rule`

The words `antidote`, `toxin`, `cure`, and the correct suppressor identity are
not supplied to hypothesis generation, candidate scoring, or navigation.

## Hidden world transition

For this protocol only, consuming red schedules one passive causal-probe rise
60 seconds later. Consuming yellow after red while that event remains pending
cancels one pending red event. Yellow before red, yellow alone, blue after red,
and blue alone do not cancel or create probe events.

The 60-second interval replaces the historical 10-second delay only when the
new protocol flag is enabled. It was frozen before the first Unity run because
the initial 30-second proposal was unlikely to permit unguided travel between
the separately scattered red and yellow populations. The causal probe remains
excluded from survival, reward, workspace, GNW, conductor, and ordinary MPC
utility.

The first scene population is also frozen before execution at 100 yellow
flowers, matching the intended 100-red population while retaining the larger
blue population as the abundant control. This change is an opportunity-rate
adjustment, not evidence-dependent tuning; no Unity record had yet been made.

## Symmetric formal candidate set

The initial formal pool must contain at least:

1. `yellow_after_red_suppresses_probe`
2. `blue_after_red_suppresses_probe`
3. `any_pickup_after_red_suppresses_probe`
4. `no_ordered_suppression`

No candidate receives motor or production-memory authority merely by existing.
The experiment selector may request ordered interventions by expected
information gain, but safety vetoes and collision-filtered MPC remain superior.

## Required intervention classes

- red alone / red then wait
- red then yellow before the due time
- red then blue before the due time
- yellow alone or yellow before red

An episode is discarded if an unrequested pickup occurs between the initiating
red event and the observation deadline, if more than one unresolved red event
is present, or if reset/counter discontinuity occurs.

## Discovery and verification boundary

Discovery episodes may be used to formulate one typed ordered hypothesis.
Their observation indices are frozen at admission. Only later clean episodes
may update the admitted candidate. A posterior at or above 0.95 must therefore
come from held-out embodied interventions, not reused prompt evidence.

## Staged gates

1. **Telemetry gate:** Unity reports independent red, blue, and yellow totals
   plus per-feature visibility and direction. Non-red is never inferred to mean
   blue.
2. **World-dynamics gate:** deterministic unit tests establish ordered yellow
   cancellation, non-cancellation by blue, and no effect for yellow-before-red.
3. **Domain-sketch gate:** typed actions reject invalid argument types and the
   symmetric pool selects an informative ordered experiment.
4. **Unity plumbing smoke:** at least one yellow pickup is separately observed,
   one red event is scheduled, and one clean ordered episode completes or is
   explicitly discarded. No inference claim is made at this gate.
5. **Dynamic discovery:** only after the first four gates pass may Gemma propose
   an ordered rule and held-out Unity episodes attempt verification.

## Safety and falsification

- Zero survival failures and zero critical-hunger exposure remain required.
- Yellow sensing or cancellation must have zero direct motor influence outside
  an explicitly logged PGNW experimental request.
- A result fails if yellow is collapsed into blue, if prompt text names the
  correct suppressor, if discovery evidence updates the admitted posterior, or
  if contaminated episodes are counted as clean.
- One successful run establishes plumbing only. Matched seeds are required for
  a behavioral or generalization claim.
