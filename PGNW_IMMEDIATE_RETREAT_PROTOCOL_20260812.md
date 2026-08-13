# PGNW Immediate-Retreat Smoke — 12 August 2026

## Rationale

The seed-148 observation-isolation smoke changed 46 MPC selections but the
spawn-adjacent blue mushroom was still acquired 2.14 seconds after red. The
ordinary `0.18` MPC regret bound could not reverse the existing approach
momentum quickly enough. This protocol changes only the first two seconds after
an experimental pickup.

## Manipulation

- Keep target acquisition, posterior inference, contamination rejection, and
  the ten-second isolation phase unchanged.
- During the first two seconds of isolation, admit collision-safe MPC actions
  within `0.35` of the ordinary optimum and select maximum food repulsion.
- After two seconds, restore the existing `0.18` regret bound for the remainder
  of the observation window.
- Stop isolation immediately if an intervening pickup already confounds the
  trial.

Fallback, stuck recovery, hunger at or above 0.92, guided AIR, guided resource
memory, collision masking, and the requirement that MPC already be active remain
absolute vetoes.

## Ten-minute criterion

Pass if at least one requested red pickup completes an uncontaminated ten-second
observation, with immediate-retreat decisions recorded and no survival failure
attributable to retreat. The adjacent spawn pair remains unchanged as a stress
test. Only a pass permits a matched overnight comparison.
