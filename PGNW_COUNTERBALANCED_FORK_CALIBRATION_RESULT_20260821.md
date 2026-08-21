# PGNW Counterbalanced Fork Calibration Result

## Outcome

The corrected four-run passive calibration eliminated the earlier uniform-blue
confound but revealed a deterministic right-side preference.

| Trial | Seed | Blue position | First after red | Red (s) | First (s) | Other (s) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| CBL1R | 185 | Left | Yellow (right) | 0.850 | 4.451 | Blue 6.356 |
| CYL1 | 185 | Right | Blue (right) | 0.887 | 2.722 | Yellow 5.593 |
| CYL2 | 186 | Right | Blue (right) | 1.045 | 2.934 | Yellow 5.772 |
| CBL2 | 186 | Left | Yellow (right) | 1.048 | 4.406 | Blue 6.513 |

- Blue first: 2/4.
- Yellow first: 2/4.
- Right branch first: 4/4.
- All four runs began with red/blue/yellow pickup totals at zero.
- All four completed red and both branch pickups.
- Stuck events and respawns: zero.

The original CBL1 setup attempt is excluded for a prospective setup reason:
red pickup total was already one on telemetry frame zero. Its recording remains
preserved. Calibration geometry was then moved farther away, and the corrected
run used the distinct CBL1R output ID.

## Verdict

`color_balance_pass_original_side_gate_fail`

The calibration succeeds at its main practical purpose: passive behavior is no
longer uniformly blue, and mirrored color assignment produces a balanced 2/4
color baseline. It fails the original stricter requirement that neither side be
chosen in every run.

No active-PGNW conclusion follows from calibration. Before reserved data are
opened, the protocol must either retain the strict gate and redesign calibration
geometry, or transparently amend the design to treat side as a blocked nuisance
factor. Under the latter design, every controller seed and layout must be paired
active/passive, both mirror assignments must be equally represented, and the
analysis must report within-layout discordance. That would turn the stable side
bias into a controlled baseline rather than pretending it does not exist.
