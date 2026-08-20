# Typed Episodic Protective Recall — Two-Pair Result

## Question

Can a verified causal production use typed episodic memory to navigate toward
an initially invisible yellow antidote after a red pickup creates a delayed
metabolic hazard?

## Frozen architecture

- Verified rule: `yellow_after_red_suppresses_probe`, posterior 0.9763185.
- Red schedules a delayed metabolic event; yellow cancels it and blue does not.
- Typed memory stores red, blue, and yellow coordinates without selecting which
  type will later matter.
- The verified rule requests yellow only while the metabolic event is pending.
- PGNW supplies bounded committed guidance to MPC. Collision, stuck, fallback,
  hunger, AIR, and score-regret safety gates retain priority.
- Blue remains edible. It may contaminate a learning episode but cannot cancel
  the hazard or terminate already verified protective execution.

## Counterbalanced pairs

| Pair | Run order | Start red | Memory red-to-yellow | Empty control red-to-yellow | Paired difference |
|---|---|---|---:|---:|---:|
| Seed 164 | memory then control | `(72, -82)` | 43.07 s | 38.55 s | memory 4.52 s slower |
| Seed 165 | control then memory | `(83, -181)` | 18.96 s | 54.56 s | memory 35.60 s faster |
| Mean | counterbalanced | two starts | **31.02 s** | **46.56 s** | **memory 15.54 s faster (33.4%)** |

Pair 1's empty controller encountered a closer previously unknown yellow. The
memory treatment nevertheless reached its selected known target in 43.07 seconds
versus 92.65 seconds for that same target under control. In pair 2, the control
did not collect the treatment's selected target during the bounded run.

## Causal action evidence

- Seed 164 memory treatment: yellow began 104.46 units away and invisible.
  Memory remained active for 180 pre-visibility frames, changed 67 MPC actions,
  and closed to 13.55 units before visual handoff. A blue pickup did not erase
  the yellow objective.
- Seed 165 memory treatment: yellow began 36.85 units away and invisible.
  Memory remained active for 66 pre-visibility frames, changed 49 MPC actions,
  and closed to 14.59 units before visual handoff.
- Empty controls began with zero typed regions and produced zero typed-memory
  recommendations or pre-visibility memory influence.
- Both memory treatments cancelled their hazards without metabolic cost or
  survival failure and later completed a second protective cycle.

## Interpretation and limitation

The runs establish functional pre-visibility typed recall, causal influence on
MPC decisions, visual handoff, and metabolic resolution. The two-pair mean favors
memory, but `n=2` is descriptive and cannot establish a population-level speed
advantage. Pair 1 demonstrates the expected boundary: incomplete memory can
commit to a farther known target while exploration luckily finds an unknown
closer target. Chance encounters therefore remain permitted; episodic memory is
retained as a reliability mechanism, not claimed as an oracle for the globally
nearest resource.

## Retained local raw evidence

The four final paired JSONL recordings remain local and are excluded from Git:

- Seed-164 memory SHA-256: `3eb1925bf938fa2643bbfea17d489ffc76e89981b54a0e6bd1b100ac92025b3f`
- Seed-164 control SHA-256: `1199358893ada3deced37b718efd4c6dfc7122d95f056e1b41edf47abab0f00a`
- Seed-165 memory SHA-256: `a2a24eb342a23be6e3f7da44d0d2d7493bd96e74bc531b3163ffe784ce0cad53`
- Seed-165 control SHA-256: `52d18130682b2ed7d5931b5e8a4faa009d9326bb93dc58434d12b84d630c0186`
