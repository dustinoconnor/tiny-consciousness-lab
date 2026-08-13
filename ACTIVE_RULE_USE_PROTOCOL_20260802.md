# Active Rule-Use Protocol: Verified Pressure Rule

## Purpose

Test whether a frozen, prospectively verified declarative rule can improve
bounded homeostatic planning. This is a rule-use experiment, not a new
language-model discovery test.

## Critical semantic boundary

The committed rule is `red pickup -> delayed metabolic-pressure increase`.
In the present controller, metabolic pressure is a cost/stress signal: it
increases body error and is not an energy reserve. Therefore this protocol
must not reward seeking red to prevent a hypothetical pressure drop. That
would reverse the existing signal's meaning and introduce an unverified new
causal claim.

## Registered intervention

Create matched food-choice opportunities containing one reachable red and one
reachable blue food item with comparable distance and ordinary pickup reward.
Add a registered high-pressure safety threshold. No new red effect is added.

Two otherwise matched conditions use the same frozen navigation stack, Unity
layout family, seeds, and committed production-memory file:

1. `memory_authority=0.0`: rule is logged only.
2. `memory_authority=planning_read`: the planner may apply the frozen rule as
   a bounded cost forecast when choosing between visible food targets.

Gemma, GNW, and production memory do not emit actions. The existing
collision-masked MPC remains the only motor selector.

## Bounded planner utility

For each visible candidate food `f`, retain the existing grounded MPC score
and add only this planning term:

    U(f) = 0.35 - 0.01 * distance(f)
           - predicted_pressure_risk(f)

where:

    predicted_pressure_risk(f) =
        max(0, current_metabolic_pressure + 0.34 - 0.70)
        if f.feature == red and planning_read is authorized;
        0 otherwise.

The red increment (0.34) is the registered/observed causal magnitude and
`0.70` is a fixed safety threshold. The comparison runs only for simultaneous
red/blue visibility with no more than 2 m distance difference. When blue has
higher utility, it supplies a bounded candidate-level MPC term: a maximum
0.25 alignment bonus toward blue and equal penalty toward red. It cannot
override collision masking, obstacle avoidance, a mandatory MPC recovery, or
the existing survival rules. The planner telemetry logs both candidate
utilities, rule-derived risk, effective weight, chosen target, and whether the
rule changed a choice.

## Primary outcome

On matched red-versus-blue visible choice episodes beginning near the pressure
threshold, the planning-read condition should select blue more often than the
authority-0 condition, produce fewer threshold crossings, and never increase
collisions or survival failures. The expected behavior is selective *avoidance*
of red under pressure urgency, not indiscriminate red aversion.

## Separate future experiment: beneficial reserve

If the desired question is proactive red seeking, introduce a separate
`metabolic_reserve` state and a new, programmed red-to-delayed-reserve effect.
That effect needs its own discovery, formal falsifier, prospective verification,
and Rule Commit before any planner may use it. It cannot inherit authority from
the pressure rule.
