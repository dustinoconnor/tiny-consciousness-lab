# Typed Stale-Memory Fallback — Seed 166

## Question

Can verified protective recall reject an absent remembered yellow resource and
fall through to another typed memory without changing the frozen acquisition
artifact?

## Frozen intervention

- The seed-163 acquisition map was copied for the run.
- One answer-neutral decoy yellow memory was added at `(72.4295, -72.3327)`,
  10 units north of the known red pickup.
- All genuine yellow memories, the verified rule, PGNW/MPC bounds, metabolic
  deadline, and terrain controller remained unchanged.
- Raw telemetry: 1,426 rows over 299.553 seconds, steps 0 through 1,425.

## Result

- Red pickup: 3.39 seconds.
- Initial typed recommendation: decoy yellow `(72.4295, -72.3327)`.
- Empty arrival/stale detection: 4.87 seconds, 1.48 seconds after red.
- The decoy was persisted with `failures: 1` and suppressed for the bounded
  refractory interval.
- In the same telemetry frame, recall switched to the genuine yellow memory at
  `(116.8036, -174.3526)` while the protective rule remained active.
- From stale rejection until direct perception, memory guidance was active for
  15/15 frames and accumulated 13 additional protective MPC action changes.
- A closer unmemorized yellow entered the 16-unit sensor radius at 7.85 seconds
  and correctly replaced memory guidance through direct perceptual handoff.
- Yellow pickup occurred at 12.94 seconds, 9.55 seconds after red. The pending
  hazard was cancelled with zero hunger-cost events.

The frozen source memory remained unchanged at SHA-256
`a47ecc470aff22d6cbb736ebdbc4e32e6e21d8d601b3be8ca62beba7e8df38fe`.

## Safety and limitation

The run had zero survival failures and zero respawns, with three recovered stuck
events. It establishes stale typed-memory detection, persistent failure evidence,
same-frame fallback selection, bounded control influence, and opportunistic
perceptual handoff. It does not establish physical completion of the second
remembered route during the protective episode, because the closer chance yellow
validly resolved the metabolic need first.
