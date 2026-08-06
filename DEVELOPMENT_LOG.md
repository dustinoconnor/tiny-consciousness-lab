# Development Log

This log measures how much substantive Codex work fits inside the user's
reported daily agentic-usage budget. It excludes casual discussion, time spent
waiting for Unity runs, and user-operated simulation time. Post-run analysis,
implementation, debugging, tests, and publication work are included.

Usage percentages are copied from the Codex usage HUD when the user reports
them; the repository cannot read that meter directly.

Context-remaining percentages are also copied from the HUD when reported.
They measure active-chat capacity, not account usage. Unattended runtime does
not count as work, and casual discussion does not start a measured session.

## 2026-07-30

Daily target: spend at most 14 percentage points of weekly usage.

### Session 1 - Route-Bound Resource Memory

- Start: 10:08 EDT
- End: 10:21 EDT
- Usage remaining at start: 85%
- Scope: strengthen Python episodic food memory by binding rewarded regions to
  successful approach routes, with shuffled and memory-reset controls.
- Work completed: added a frozen-policy five-condition benchmark, bounded
  two-stage region/approach recall, route-entry gating, shuffled spatial
  controls, reset control, metrics export, and focused unit tests.
- Verification: four route-memory unit tests passed; a four-seed smoke test
  exposed premature route engagement; after one principled correction, the
  full matched 20-seed benchmark completed successfully.
- Outcome: route-bound memory raised mean pickups from 8.25 to 9.35 (+13.3%),
  exceeded coordinate-only memory (9.00), remained collision-free, and beat
  shuffled memory (7.80). The mean worst meal gap improved only 2.3%, route
  binding did not beat coordinate memory on that metric, and favorable pickup
  differences occurred in only 5/20 layouts. Promising partial result; not
  approved for Unity integration.
- Usage remaining at end: 82%
- Measured usage: 3 percentage points

### Session 2 - Prime-Then-Test Resource Recall

- Start: 10:27 EDT
- End: 10:38 EDT
- Usage remaining at start: 82%
- Scope: isolate episodic resource recall from chance discovery, replace full
  path replay with resource centroids and sustained-visibility approach
  anchors, and add stale-memory suppression plus escape behavior.
- Work completed: added a controlled two-phase benchmark with autonomous
  near-resource priming, distant food-invisible recall starts, hunger/visibility
  gates, compact resource centroids, sustained-visibility approach anchors,
  stale-memory confidence decay, bounded escape, and matched controls.
- Verification: three focused tests passed; four-seed smoke and matched
  20-seed evaluations completed; paired seed-level differences and approximate
  confidence intervals were calculated.
- Outcome: coordinate memory increased pickups from 11.15 to 18.30 (+64.1%),
  reduced the worst meal gap from 1030.35 to 477.90 (-53.6%), raised
  first-pickup success from 70% to 100%, and retained zero collisions. Reset
  exactly matched baseline and shuffled locations failed to produce a reliable
  pickup gain. Approach anchors underperformed coordinate-only memory, so
  coordinate memory is approved for passive Unity integration and anchors are
  not.
- Usage remaining at end: 79%
- Measured usage: 3 percentage points

### Session 3 - Passive Unity Resource Memory

- Start: 10:40 EDT
- End: 10:46 EDT
- Usage remaining at start: 79%
- Scope: add coordinate resource-memory encoding and hunger-gated passive
  recommendations to the Unity UDP loop without changing motor control.
- Work completed: added an opt-in persistent resource-region observer to the
  Unity UDP loop, strict hunger and visible-food gates, passive world-space
  recommendations, reward encoding, stale-arrival telemetry, CLI flags,
  recorder fields, compact command telemetry, and an offline analyzer.
- Verification: five focused resource-memory/analyzer tests passed; 37
  existing embodied-loop and terrain-AIR regression tests passed; Python
  compilation, CLI help, and diff hygiene checks passed.
- Outcome: ready for a 30-minute passive scarcity-terrain run. The observer's
  action influence is structurally fixed at zero, so this stage can validate
  Unity memory formation and recommendation timing without changing movement.
- Usage remaining at end: not sampled separately

### Session 4 - Passive Resource-Memory Run Analysis

- Start: 11:26 EDT
- End: 11:29 EDT
- Usage remaining at start: not sampled separately
- Scope: analyze the completed 30-minute Unity scarcity run, validate passive
  coordinate-memory formation and recommendation timing, and inspect stale or
  false-confidence behavior.
- Work completed: analyzed 8,426 recorded frames, added recommendation-transition
  outcome tracing, identified a post-meal re-query calibration fault, and added
  reward/stale-region refractory handling without enabling motor influence.
- Verification: seven focused resource-memory/analyzer tests and 37 embodied
  regression tests passed after the correction; Python compilation passed.
- Outcome: the 29.9-minute run encoded 35 pickups into 15 regions and issued
  passive recommendations during 87.0% of eligible frames with zero action
  influence. Of 14 recommendation transitions, 8 were followed by movement
  toward the target and 4 by a pickup within 12 m of it. Late distant-target
  switching exposed an immediate post-meal query bug; the observer now waits
  60 seconds after reward and temporarily suppresses an empty arrived region.
  A short repeat validation is required before bounded control.
- Usage remaining at end: 76%
- Combined measured usage for Sessions 3-4: 3 percentage points (79% to 76%)

### Session 5 - Resource-Memory Refractory Validation

- Start: 12:33 EDT
- End: 12:35 EDT
- Usage remaining at start: 76%
- Scope: compare the corrected 20-minute passive Unity run against the first
  run, emphasizing post-meal suppression, recommendation stability, target
  approach, and causal separation.
- Work completed: analyzed the corrected 5,541-frame run, compared it with the
  first passive run, strengthened Unity-scale distance cost, raised the recall
  gate from 0.55 to the established 0.70 hunger threshold, and added a
  reproducible offline policy-replay tool.
- Verification: seven focused tests and 37 embodied regression tests passed;
  the revised policy was replayed over both real Unity recordings without
  changing their trajectories.
- Outcome: the live refractory run collected 51 pickups and expanded memory
  from 15 to 23 regions with zero action influence. Recommendation transitions
  fell from 14 to 1, confirming that post-meal chatter was fixed. Offline replay
  with the 0.70/distance-calibrated rule selected nearer 115-160 m regions
  instead of the 239 m favorite, but subsequent passive pickups occurred
  elsewhere. Passive integration is valid; motor benefit remains unproven and
  requires a tightly bounded MPC-prior intervention test.
- Usage remaining at end: pending
- Usage remaining at end: 75%
- Measured usage: 1 percentage point

### Session 6 - Bounded Resource Prior

- Start: 12:37 EDT
- End: 12:38 EDT
- Usage remaining at start: 75%
- Context remaining after implementation: 22%
- Scope: allow validated resource memory to provide a small hunger-gated
  directional prior inside grounded MPC while preserving sensor vetoes,
  fallback priority, and visible-food control.
- Work completed: added opt-in guided resource recall as a small
  uncertainty-gated directional bonus inside stochastic MPC, with 0.70 hunger
  gating, distance-calibrated recall, visible-food release, meal refractory,
  AIR priority, fallback priority, and existing collision masks.
- Verification: seven focused tests and 37 embodied regression tests passed;
  compilation and diff hygiene passed; a real-checkpoint stochastic-MPC smoke
  test applied an effective 0.0047 weight under the 0.03 ceiling without
  changing the clear-field action.
- Outcome: ready for a 20-minute bounded Unity smoke run. Resource memory can
  influence only ambiguous MPC rankings; it cannot authorize blocked actions
  or override visible food, AIR escape guidance, or stable fallback.
- Follow-up overhead before analysis: re-read this log, confirmed that the
  user-started Unity run already had the intended bounded command, briefly
  inspected partial telemetry and started a read-only progress watcher, then
  stopped the watcher when the user clarified that no activity was needed
  during the run. No completed-run analysis was performed.
- Usage remaining at end: 72%
- Measured usage: 3 percentage points (includes implementation and the
  unnecessary pre-analysis follow-up overhead)
- Context remaining after the new context window: 81%
- Context note: the increase from 22% after implementation to 81% reflects a
  fresh/expanded context window, not recovered weekly usage. The user reported
  that the three usage points were an avoidable cost. Further recording access
  and analysis are paused until the user explicitly authorizes them.

### Session 7 - Bounded Resource-Memory Unity Smoke Analysis

- Start: 13:05 EDT
- End: 13:07 EDT
- Usage remaining at start: 72%
- Context remaining at start: 81%
- Scope: analyze the completed 20-minute guided resource-memory Unity smoke
  run within the final one percentage point of the reported daily target.
- Work completed: ran the existing resource-memory outcome analyzer, audited
  the guided/unguided action counterfactual recorded on every frame, checked
  the effective-weight ceiling and all authority gates, compared safety and
  collection telemetry with the preceding passive run as non-causal context,
  and saved a guided-control audit.
- Verification: the recording contains 5,601 frames over 19.88 minutes. The
  resource prior was nonzero on 12 frames, never exceeded 0.013865 under its
  0.03 cap, and was applied only while hungry, with food hidden, AIR inactive,
  and fallback inactive. It changed the selected MPC action on 0 frames.
- Outcome: the run collected 24 mushrooms, encoded 24 rewards, and expanded
  memory from 23 to 28 regions. Three recommendation episodes occurred; one
  target was approached and one pickup landed 4.43 m from its recommended
  region. There were no survival failures or forced respawns. Contact and stuck
  telemetry were worse than in the preceding passive run, but cannot be
  attributed to resource guidance because the recorded guided and unguided
  actions were identical throughout and the runs were not a reset matched
  pair. The authority bounds are validated; motor benefit remains unproven,
  so the guidance weight should not be increased from this result.
- Artifacts:
  `outputs/resource_memory_guided_smoke_analysis_20260730.json`;
  `outputs/resource_memory_guided_control_audit_20260730.json`.
- Usage remaining at end: 71%
- Context remaining at end: 75%
- Measured usage: 1 percentage point
- Measured work time: 2 minutes

### Daily measured-work summary

- Sessions 1-7: 38 minutes of substantive Codex work.
- This total includes implementation, tests, debugging, and run analysis; it
  excludes the Unity runtime itself and user-operated simulation time.
- Usage moved from 85% remaining at the start of Session 1 to 71% at the end
  of Session 7: exactly 14 percentage points, matching the daily target.
- Final reported context remaining: 75%. Context-window resets or compaction
  can increase the displayed context available during one logical workday;
  these readings are therefore tracked separately from account usage and are
  not summed as consumption.

## 2026-07-31

Daily target: stop at 58% weekly usage remaining. The user reported 70%
remaining at the start, providing a 12-percentage-point budget after exceeding
the prior day's target by two points.

### Session 8 - Grounded Tiny Scientist Design

- Start: 09:28 EDT
- Usage remaining at start: 70%
- Scope: introduce one additional metabolic outcome, use passive adaptive GNW
  to select grounded anomalies, and build a bounded Tiny Scientist that can
  propose falsifiable hypotheses without motor authority.
- Initial decision: begin with a visually distinct red mushroom and a delayed,
  context-dependent metabolic cost rather than lakes, water assets, or moving
  traps. Reusing the validated mushroom pickup path changes one causal factor
  at a time; adding new navigation physics and animated hazards concurrently
  would make failed hypotheses uninterpretable.
- Work completed so far: added deterministic red/green observable mushroom
  profiles without new assets, pickup-feature UDP telemetry, an equal immediate
  food reward with a ten-second delayed internal-pressure response for red
  pickups, passive scientific-GNW episode selection, a zero-authority JSON
  hypothesis contract, and semantic Rule Commit checks. Seven focused tests
  pass, Python compilation passes, and diff hygiene is clean.
- Model-path correction: "tiny language model" was initially interpreted as a
  request to select a local pretrained model before the experiment contract
  was fully settled. Qwen3-0.6B and Qwen3-1.7B were downloaded for smoke tests,
  consuming 1.4 GB and 3.8 GB respectively (5.2 GB total). Both produced
  semantically invalid hypotheses: the smaller model mangled the evidence and
  the larger model reversed the observed red/green causal direction. With the
  user's authorization, both exact Hugging Face cache directories were
  deleted. The cache now occupies 614 MB and the data volume reports 15 GiB
  free. No Qwen model remains as a project dependency.
- Revised model decision: a purpose-built language model is unnecessary for
  the initial claim. The local hypothesis backend now targets Google's
  instruction-tuned `google/gemma-3-270m-it`, while all causal grounding and
  Rule Commit authority remain outside the model. It has not been downloaded:
  the official repository is gated by the Gemma usage license, so acceptance
  and authentication are a separate user step. This keeps today's work from
  consuming storage on another unapproved download.
- Red asset integration: confirmed the user's 1.7 MB
  `Assets/mushroomRed.glb` import in the active Unity project. Removed runtime
  tinting and automatic red-label assignment so the observed geometry/color
  cannot silently disagree with telemetry. Patched the active project's
  mushroom and UDP scripts in place, preserving its newer conductor and course
  reset code. Added a Unity menu command,
  `Tiny Consciousness > Food > Create Red Food Prefab`, which creates
  `Assets/RedFoodMushroom.prefab` with explicit red observable metadata. The
  command was not invoked automatically; scale and collider need visual
  inspection in Unity before red instances are placed.
- Verifier hardening: hypotheses must now identify the empirically
  higher-pressure feature as the proposed cause, use the lower-pressure
  feature as the comparison, predict a pressure increase, stay within four
  seconds of the observed delay, and state a genuinely contradictory
  falsifier. This closes the directional-reversal failure seen with Qwen3-1.7B.
- Verification: nine focused Tiny Scientist/metabolic tests pass, Python
  compilation passes, diff hygiene passes, and no Qwen-specific prompt or
  runtime auto-profile code remains. The Unity editor log contained no current
  C# compilation error, but prefab appearance/collider verification remains a
  manual editor check.
- Work checkpoint: 09:56 EDT (28 minutes elapsed since session start).
- Status: implementation ready for Unity prefab creation and visual QA; no
  embodied hypothesis claim has been made yet.
- Usage checkpoint reported by user: 64% remaining. Six percentage points
  have been used since today's 70% start; the user estimates roughly ten
  minutes remain based on yesterday's measured work rate.
- Prefab clarification: `RedFoodMushroom.prefab` has not yet been created.
  `RedMushroomPrefabTools.cs` adds the editor command that will generate it at
  `Assets/RedFoodMushroom.prefab`; creating and placing the asset were kept as
  separate gates so its scale and collider can be visually checked before it
  enters the experiment.
- Gemma access check: the logged-in Hugging Face account can read the public
  repository metadata (575,485,707 bytes total; 536,223,056-byte weight file),
  but a real model load returned HTTP 403 because this account has not accepted
  the Gemma usage license. No Gemma files were downloaded and the Hugging Face
  cache remains 614 MB. Official hosted Gemini API access is not an equivalent
  test of the 270M local model, so its structured-hypothesis performance is
  still unknown rather than presumed successful.
- Voice-path recovery: located the existing local MLX Kokoro installation in
  `/Users/dustinoconnor/dream_loop/.venv_mlx_tts`, confirmed `mlx_audio`
  imports successfully, found the cached 609 MB
  `mlx-community/Kokoro-82M-4bit` model, and confirmed the preferred British
  male voice is `bm_george`. This can become a presentation-only output layer
  for verified hypothesis reports; it must not increase hypothesis or motor
  authority. Microphone speech recognition would be a separate input layer.
- Work checkpoint: 10:33 EDT. Session remains open pending the user's usage
  reading after this inspection pass.
- Unity prefab checkpoint (10:41 EDT): the user ran the editor command and
  placed the generated `Assets/RedFoodMushroom.prefab` in the scene. Read-only
  inspection confirmed a BoxCollider, Rigidbody, FoodMushroom component, and
  serialized `observableProfile: 1` (Red), with no current C# compilation
  errors in the Unity editor log. Inspection also caught Unity's default
  serialized physics settings. Corrected both the generated prefab and its
  builder so the collider is a trigger and the Rigidbody is kinematic with
  gravity disabled before runtime, matching `FoodMushroom.Awake()` and
  preventing the mushroom from falling over.
- Usage checkpoint after prefab verification: 63% remaining. Seven percentage
  points have been used since today's 70% start, leaving five points before the
  58% stop target.
- Grounding correction: the user clarified that the 250 ordinary mushrooms are
  visually blue, not green. Renamed the default observable profile and all
  experiment/test labels from Green/`green` to Blue/`blue` before collecting
  data. This prevents the language layer from receiving a symbolic color label
  that does not match the embodied scene. The existing default enum value stays
  numeric zero, so the 250 ordinary prefab instances remain the control class;
  the red prefab remains numeric one.
- Gemma 3 270M test (10:51 EDT): after the user obtained license access,
  downloaded `google/gemma-3-270m-it` successfully. Its local cache occupies
  544 MB; the full Hugging Face cache is now 1.1 GB. Nine focused tests still
  pass and Unity reports no current C# compilation errors after the blue-label
  correction.
- Model outcome: Gemma 3 270M loaded and generated text, but failed both allowed
  attempts on balanced synthetic evidence (six isolated blue controls with
  mean pressure delta 0.01 versus six red episodes with delta 0.34 at ten
  seconds). It ignored the outcome contrast and proposed the passive-GNW
  broadcast count as both cause and comparison. The semantic verifier rejected
  it for lacking a distinct comparison. Therefore the 270M zero-shot backend
  is operational but not competent for this hypothesis contract; no Rule
  Commit occurred and authority remained zero. The verifier was not weakened
  to manufacture a success.
- Usage checkpoint after the first Gemma test: 62% remaining. Eight percentage
  points have been used since today's 70% start, leaving four points before the
  58% stop target.
- Interpretation boundary: this is not yet a failure of the paper's full
  embodied abduction claim. It is a narrower failure of zero-shot,
  language-mediated hypothesis formulation from a clean synthetic evidence
  summary: the 270M model confused GNW salience metadata with the physical
  cause. The embodied sequence of novel observation, hypothesis, deduction,
  intervention, and verification has not yet been run.
- Run preparation: clarified that Gemma 270M emitted syntactically valid JSON
  but selected semantically invalid causal variables. A tiny language model is
  therefore not ruled out; the result rules out this zero-shot 270M prompt as
  sufficient. Prepared a 20-minute red/blue evidence-collection command using
  the validated terrain MPC controller with adaptive GNW observational only,
  no resource-memory guidance, and an explicit recording path:
  `outputs/unity_shadow/red_blue_tiny_scientist_20260731.jsonl`.
- Launch correction: the first command failed before opening UDP or writing
  experimental data because `--adaptive-gnw-passive` requires a systemic
  conductor checkpoint. This was a command-preparation error, not an experiment
  failure. Removed that live adaptive-GNW flag: the experiment-specific
  `PassiveScientificGNW` is intentionally applied offline to metabolic episodes
  by `tiny_scientist.py`, so adding the systemic routing stack would be an
  unnecessary causal variable. No partial run needs cleanup.
- Completed red/blue baseline analysis (12:15 EDT): the recording contains
  5,667 frames over 1,192.68 seconds (19.88 minutes), 16 mushroom pickups, no
  survival failure, 455 visible-food frames, and a final durable count of two
  red pickups. Red was visible in exactly two contiguous sighting episodes.
  In both, the controller acquired the target from about 20.5 m away,
  approached to about 3.8 m, and collected it. Thus the red mushrooms it saw
  were not placed too far apart; the other placed red mushrooms were simply
  outside the route/line-of-sight sample during this run. The controller did
  not learn a red preference online.
- Attribution repair: the first red pickup's transient `mushroom_feature`
  string had reset to `none` by the frame Python observed the pickup, even
  though the durable red-total counter incremented. Updated
  `metabolic_episodes` to treat red/total counter deltas as authoritative and
  use blue as the complementary class in this two-profile experiment. Added a
  regression test; ten focused Tiny Scientist/metabolic tests pass.
- Corrected causal evidence: 14 blue episodes (11 isolated) had mean pressure
  delta 0.0 and zero positive responses. Both red episodes were isolated; both
  produced a 0.34 pressure increase after a mean 10.464 seconds. Passive
  Scientific GNW broadcast both red surprises and had action influence zero.
  This validates the sensing, delayed metabolism, attribution repair, and
  passive salience path, but two red replications are preliminary evidence,
  not a strong causal sample.
- Real-data model outcome: Gemma 3 270M again emitted syntactically valid but
  semantically invalid JSON on both allowed attempts. It proposed `red:2` as
  both cause and comparison and described GNW broadcasts as the predicted
  physical effect. The verifier rejected it for lacking a distinct comparison;
  no hypothesis or rule was committed and authority remained zero. Saved the
  evidence summary to
  `outputs/red_blue_tiny_scientist_evidence_20260731.json` and the rejected
  model output to
  `outputs/red_blue_tiny_scientist_gemma270m_failure_20260731.json`.
- Interpretation: this replicates the 270M model's zero-shot semantic failure
  on both synthetic and embodied evidence. It does not establish that all tiny
  models lack abductive capacity: prompt scaffolding, task-specific tuning, and
  a larger-model comparison remain untested.
- Usage checkpoint after completed-run analysis: 61% remaining. Nine
  percentage points have been used since today's 70% start, leaving three
  points before the 58% stop target. No 1B download or further model experiment
  was started at this checkpoint.
- Storage-only 1B planning check: official Hugging Face metadata reports
  `google/gemma-3-1b-it` at 2,039,069,601 bytes total, including a
  1,999,811,208-byte weight file. The data volume currently reports 15 GiB
  available and the existing Hugging Face cache occupies 1.1 GB, so the full
  1B repository fits without deleting ComfyUI models. Expected free space
  after download is roughly 13 GiB, excluding temporary transfer overhead.
- End-of-day model preparation: the user successfully downloaded
  `google/gemma-3-1b-it`; execution is deferred until the matched 270M-versus-1B
  comparison tomorrow.
- Session 8 end status: final reported usage is 61% remaining, down nine points
  from the 70% start and three points above the 58% stop target. Approximate
  substantive Codex work time is 55 minutes. This includes implementation,
  focused tests, Unity integration fixes, two model evaluations, completed-run
  analysis, and documentation. It excludes the 20-minute autonomous Unity run,
  lunch/waiting time, user-operated prefab placement, and background model
  downloads.
- Post-session literature/design note: reviewed the proposal to use Pankseppian
  affect as an abductive constraint. The source framework identifies seven
  mammalian primary-process systems: SEEKING, RAGE, FEAR, LUST, CARE,
  PANIC/GRIEF, and PLAY. Treating them as software utility filters is an
  engineering reinterpretation, not a result established by Panksepp. Much of
  the proposed functionality already exists here as hunger/fatigue/body error,
  valence/arousal, trap pressure, neuromodulation, passive GNW salience,
  conductor routing, and MPC costs. The useful new experiment is therefore not
  to add an unvalidated seven-float emotion array. It is to test whether a
  pre-language homeostatic relevance filter improves hypothesis validity under
  frozen evidence without naming the causal color or changing motor control.
  This is deferred; no implementation or model run was performed.
- Post-session GCML design note: reviewed the Generative Cognitive Map Learner
  described in *Neural sampling from cognitive maps enables goal-directed
  imagination and planning*. GCML learns a goal-independent joint embedding of
  observations and actions through next-state prediction, learns an inverse
  state-difference-to-action map, and samples imagined trajectories toward
  known or novel goals. Its generativity is trajectory sampling, not language
  or pixel generation. It is a promising missing long-horizon/counterfactual
  layer, but not a replacement for validated components: ART retains stable
  categories/episodes, AIR/GNW selects relevant goals or anomalies, the
  conductor decides when to query specialists, and collision-masked MPC
  verifies and executes short-horizon actions. A future bounded pilot should
  train a GCML-like transition map offline from recorded
  `(state, action, next_state)` tuples and compare MPC alone, AIR/ART+MPC, and
  GCML+MPC on frozen held-out goals before granting any control. For the
  red/blue causal task, spatial GCML alone is insufficient; observable color,
  internal metabolic state, elapsed time, and wait/intervention transitions
  must be represented before it can support causal counterfactuals. No code or
  model change was made.
- End-of-day GitHub publication audit: local `main` and `origin/main` both point
  to commit `6f9d3f0` (`Publish attention-gated terrain memory (#15)`), so no
  unpublished commit is waiting to be pushed. The working tree instead contains
  several days of interleaved work: 11 modified tracked files (including 696
  added/changed lines in the shared embodied loop) and many untracked source,
  test, checkpoint, and result files. The 49 MB raw red/blue JSONL recording is
  correctly ignored. Recommendation: do not upload the dirty tree under the
  final two-percent budget. Next session, curate a branch and draft PR around
  one evidence-backed slice, starting with bounded terrain resource memory and
  its analyzer/audit, then publish Tiny Scientist/metabolic embodiment as a
  separate PR after the 270M-versus-1B result. No files were staged, committed,
  pushed, or published during this audit.

## 2026-08-01

Daily target: stop at 86% weekly usage remaining. The user reported an
unexpected reset to 100% at the start, providing a 14-percentage-point budget.
Displayed context will be tracked separately if reported; account reset timing
is treated as external and is not inferred from these measurements.

### Session 9 - Frozen-Evidence Gemma 2x2

- Start: 09:54 EDT
- Usage remaining at start: 100%
- Scope: compare Gemma 3 270M and Gemma 3 1B on the same frozen embodied
  red/blue evidence, each under the original evidence interface and a
  pre-language homeostatic relevance filter.
- Pre-registered constraint: filtering may remove GNW bookkeeping and frame the
  homeostatic concern as delayed internal pressure, but may not name red as the
  cause, remove blue counterevidence, change the verifier, or affect motor
  control. All four cells use deterministic generation and the same two-attempt
  verifier feedback limit.
- Model availability: confirmed local caches for both candidates (270M: 544 MB;
  1B: 1.9 GB).
- Implementation: added an explicit `homeostatic_filtered` evidence interface
  while preserving the original prompt. The filter retains both red and blue
  outcomes and frames metabolic integrity as the selected problem, but removes
  GNW broadcast counts and does not identify the causal color. CLI runs now
  preserve rejected model outputs as JSON artifacts. Thirteen focused
  Tiny Scientist/metabolic tests pass; Python compilation and diff hygiene pass.
- Verifier repair: the initial 1B-filtered output correctly named red, blue,
  increased pressure, and the 10.464-second delay, but its proposed falsifier
  was a decrease after a blue pickup. That result would not contradict a claim
  about red. Because all hypothesis authority was still zero, no rule was
  committed. Added a regression test and required the falsifying observation
  to target the proposed-cause feature in the same clause as the contradictory
  outcome. Reran all four deterministic cells under this final verifier.
- 2x2 outcome (10:00 EDT): zero of four cells passed the final semantic
  verifier. 270M/original again used `red:2` GNW bookkeeping as both cause and
  comparison. 270M/filtered copied prompt text and emitted a nonnumeric delay.
  1B/original reversed the observed causal direction and emitted `None` as the
  delay. 1B/filtered was substantially closer: cause red, comparison blue,
  pressure increase, and 10.464-second delay were all correct, but it repeated
  the control-feature falsifier on both attempts and was rejected.
- Interpretation: homeostatic filtering materially improved the 1B causal
  formulation, but neither model completed the full falsifiable hypothesis
  contract. This supports an interaction between evidence interface and model
  capacity, not a successful abductive jump. The next bounded comparison should
  test identical non-answer-leaking schema/falsifier scaffolding for both model
  sizes, or task-specific tuning, before considering a still larger model.
- Artifacts: `outputs/tiny_scientist_2x2_270m_original_20260801.json`;
  `outputs/tiny_scientist_2x2_270m_filtered_20260801.json`;
  `outputs/tiny_scientist_2x2_1b_original_20260801.json`;
  `outputs/tiny_scientist_2x2_1b_filtered_20260801.json`;
  `outputs/tiny_scientist_2x2_summary_20260801.json`.
- Codex task-timer checkpoint after the 2x2: 7 minutes 27 seconds.
- Status: 2x2 complete; next intervention not yet started.
- Usage checkpoint: 98% weekly usage remaining, down two points from today's
  100% start and leaving 12 points before the 86% stop target.
- Context remaining: 30%. The user chose to continue in the current task and
  allow automatic context compaction instead of opening another task. Context
  is tracked separately from weekly usage; this log and the saved 2x2 artifacts
  are the durable resume source.
- Non-leaking schema follow-up: implemented a separate `bound_schema` contract
  in which the model emits `target_cause` once. The formal layer binds
  `action_to_retest` to that exact token and derives the contradictory outcome
  from `observed_effect`; it cannot substitute the control feature. The model
  still chooses cause, comparison, effect direction, and latency. Post-hoc
  validation rejected both free generators: 270M copied a verifier error as
  the cause, while 1B selected `positive-pressure fraction` rather than either
  observed color.
- Actual grammar constraint: because Pydantic only validates after generation
  and Outlines/Guidance/LM Format Enforcer are not installed, added a
  dependency-free Transformers prefix constraint. Its candidate set is
  symmetric: either color may be cause, the other is formally bound as
  comparison, every effect direction is available, and both 0.0 and 10.464
  seconds are available to either color. Seventeen focused tests pass,
  including symmetry and non-leakage checks.
- Constrained outcome: schema enforcement produced valid JSON and eliminated
  variable substitution but did not produce a valid causal selection. 270M
  chose blue/no-pressure-change/10.464 seconds; 1B chose
  blue/pressure-decrease/10.464 seconds. The verifier rejected both for
  reversing the observed evidence. This separates structural validity from
  causal competence: formal grammar can guarantee bindings but cannot supply
  the abductive choice.
- Validity warning: blue is listed before red in the evidence, and both
  constrained models chose blue. A counterbalanced red-first prompt is needed
  to test positional/candidate-scoring bias before attributing this constrained
  failure solely to capacity. No counterbalanced run has started.
- Artifacts: `outputs/tiny_scientist_bound_270m_filtered_20260801.json`;
  `outputs/tiny_scientist_bound_1b_filtered_20260801.json`;
  `outputs/tiny_scientist_constrained_270m_filtered_20260801.json`;
  `outputs/tiny_scientist_constrained_1b_filtered_20260801.json`;
  `outputs/tiny_scientist_bound_schema_summary_20260801.json`.
- Cumulative Codex task-timer checkpoint after bound-schema work: 15 minutes
  24 seconds.
- Usage checkpoint before order audit: 95% weekly usage remaining, down five
  points from today's 100% start and leaving nine points before the 86% target.
- Context remaining: 20%. Counterbalanced audit authorized; no larger model
  download is authorized or needed before resolving order sensitivity.
- Counterbalanced constrained audit (10:47 EDT): added an evidence-order switch
  that reverses only the red/blue evidence lines; model weights, frozen data,
  candidate grammar/order, decoding, and verifier remain fixed. Eighteen
  focused tests pass. With red listed first, 270M remained unchanged and wrong:
  blue/no-pressure-change/10.464 seconds. The 1B target flipped from blue to
  red, demonstrating order sensitivity, but its effect remained
  `pressure_decrease`, opposite the observed +0.34 change. It was rejected.
- Interpretation: order is a real confound for 1B target binding, but does not
  explain the complete failure; this audit does not establish embodied
  abduction. A larger model remains premature. The next clean intervention is
  to reuse the original filtered freeform interface, where 1B already selected
  the correct cause, comparison, effect, and latency, while compiling only the
  falsifier from that single cause binding. Success would be a claim about the
  neuro-symbolic Tiny Scientist system, not the unaided 1B model.
- Artifacts: `outputs/tiny_scientist_constrained_270m_redfirst_20260801.json`;
  `outputs/tiny_scientist_constrained_1b_redfirst_20260801.json`;
  `outputs/tiny_scientist_order_audit_20260801.json`.
- Cumulative Codex task-timer checkpoint after the order audit: 18 minutes
  26 seconds.
- Neuro-symbolic falsifier compilation: added the
  `freeform_compiled_falsifier` contract. The language model selects one
  observed cause, comparison, effect direction, and latency from the unchanged
  homeostatically filtered evidence. A deterministic formal layer then binds
  the retest to that same cause and derives the logical contradiction of the
  selected effect. The model's own falsifier text is retained for audit but
  ignored. Any accepted hypothesis remains `proposed_unverified` with authority
  0.0. Eighteen focused Tiny Scientist tests pass, including exact
  cause binding and rejection of unbound causes.
- Matched neuro-symbolic outcome (10:54 EDT): 270M failed before compilation
  because it did not bind exactly one real observed feature as the cause. Gemma
  3 1B selected red as cause, blue as comparison, increased internal pressure
  as effect, and 10.464 seconds as latency. Its own falsifier again substituted
  blue, but the formal compiler correctly produced: `Matched red pickups do
  not precede an increase in internal pressure at the predicted delay.` The
  complete 1B-plus-compiler result passed the semantic verifier as
  `accepted_unverified`; no rule was committed and it has no action authority.
- Interpretation: this is a successful bounded neuro-symbolic hypothesis-
  formulation result, not proof that the unaided 1B model made the full
  abductive jump. GNW/analyzer supplied grounded summaries, 1B selected the
  causal candidate, and code performed exact deduction. It remains a
  retrospective association based on two isolated red episodes. The next
  scientific gate is a pre-registered held-out or intervention Unity run that
  can falsify the red-to-delayed-pressure prediction; only a surviving result
  may advance toward staged Rule Commit.
- Artifacts: `outputs/tiny_scientist_neurosymbolic_270m_20260801.json`;
  `outputs/tiny_scientist_neurosymbolic_1b_20260801.json`;
  `outputs/tiny_scientist_neurosymbolic_summary_20260801.json`.
- Cumulative Codex task-timer checkpoint after the neuro-symbolic result:
  22 minutes 52 seconds. This is reconstructed from Codex's per-turn
  `durationMs` metadata rather than wall-clock time or manual rounding.
- Usage checkpoint after neuro-symbolic analysis: 92% weekly usage remaining,
  down eight points from today's 100% start and leaving six points before the
  planned 86% stop target. User recalled the refreshed context level as
  approximately 85% but was not certain; record this as an estimate, not an
  exact reading.
- Time-accounting correction: use the Codex task history's exact per-turn
  `durationMs` values—the same work durations displayed by the app—and sum all
  completed turns for the day. The user does not need to count or round them.
  Do not substitute wall-clock time. A user-operated autonomous Unity run is
  outside a Codex turn and therefore contributes no Codex task time; work done
  in a later analysis turn is included automatically.
- Reconstructed current total through the last completed turn: 24 minutes
  52 seconds today. The current in-progress turn will become countable only
  after completion.
- Historical task-timer audit: the earlier 38- and 55-minute figures were
  human/assistant estimates, not sums of Codex's displayed turn timers. Exact
  `durationMs` totals across the two relevant project tasks are 49 minutes
  0 seconds on 2026-07-30 and 40 minutes 16 seconds on 2026-07-31. Today's
  incomplete 24 minutes 52 seconds produces a provisional exact three-day
  average of approximately 38 minutes 3 seconds. Use these timer-derived totals
  for the ongoing average; retain the older estimates only as historical notes.
- Pre-registered held-out verification plan: collect a new 20-minute Unity
  recording with seed 92, the same frozen terrain geometry policy and MPC
  controller, no resource-memory guidance, no live adaptive-GNW routing, and
  no language-model or compiled-rule action influence. This tests the accepted
  candidate prospectively on new encounters rather than asking the model to
  reformulate it.
- Eligibility requires at least two isolated red pickups and six isolated blue
  pickups. Fewer observations make the trial inconclusive, not failed. A
  preliminary verification pass requires every eligible isolated red pickup to
  precede a positive pressure response, mean red pressure delta at least 0.30,
  mean red response latency between 9 and 11 seconds, isolated-blue positive
  response fraction no greater than 0.10, and no red counterexample. Any
  eligible red pickup without the predicted response is a falsification for
  this bounded trial. The rule remains authority 0.0 during collection.
- Planned artifact:
  `outputs/unity_shadow/red_blue_rule_verification_20260801.jsonl`.
- Held-out seed-92 run outcome (11:41 EDT completion): the bounded command
  stopped normally after 1,199.793 seconds and recorded 5,720 frames. It
  collected 12 blue mushrooms, including eight isolated analyzable episodes;
  all had pressure delta 0.0 and the blue positive-response fraction was 0.0.
  However, it recorded zero red-visible frames, zero red pickups, zero
  metabolic challenges, and no pickup-counter reset that could have hidden an
  earlier red event. The blue eligibility threshold passed, but the required
  two isolated red pickups did not occur.
- Pre-registered verdict: `inconclusive_insufficient_red_exposure`, not failed
  and not verified. The seed-92 trajectory never sampled a red mushroom, so it
  did not exercise the compiled falsifier. The candidate remains
  `proposed_unverified` with authority 0.0; no Rule Commit is permitted. One
  survival failure and one unstuck respawn occurred, but neither produced red
  exposure or a counter reset.
- Verification artifacts:
  `outputs/tiny_scientist_rule_verification_evidence_20260801.json` and
  `outputs/tiny_scientist_rule_verification_summary_20260801.json`.
- Exposure correction implementation: the existing red-prefab editor utility
  only created `Assets/RedFoodMushroom.prefab`; it contained no scatter action.
  Added `Tiny Consciousness/Food/Scatter 40 Red Food Mushrooms`. It uses a fixed
  seed, samples across the scene's blue-mushroom population bounds, snaps to
  the active terrain, enforces 14 m from existing foods and 24 m between newly
  placed reds, groups results under `Scattered Red Food Mushrooms`, marks the
  scene dirty, supports Undo, and refuses to run twice while that root exists.
  Forty red mushrooms are 16% of the reported 250-blue population: uncommon
  enough to preserve novelty but substantially more likely to provide the two
  isolated red encounters required by the registered verification gate.
- Live-project path correction: Unity was actually running
  `/Users/dustinoconnor/My project`, while the first scatter implementation was
  written only to the repository mirror under `unity/TrapCourseLab`; therefore
  the menu could not appear. Applied the same additive red-scatter editor code
  to the live project's `Assets/Editor/RedMushroomPrefabTools.cs`. Do not reuse
  `MushroomPlacementTools` for the red population: its scatter path first clears
  the shared `Generated Food Mushrooms` parent and would replace the existing
  blue scatter. The dedicated red command preserves the blue population.
- Live-script compile hygiene: removed an unnecessary `using System` import
  from both copies so the existing `Object.DestroyImmediate` call remains
  unambiguously `UnityEngine.Object`; the deterministic generator continues to
  reference `System.Random` explicitly.
- Matched exposure-repeat plan: after the user confirms the additive red group
  exists in the hierarchy, save the scene and repeat the 20-minute collection
  with controller seed 92. Reusing the seed, frozen geometry policy, terrain
  MPC settings, and eligibility/verdict thresholds holds the controller
  condition fixed; the intended experimental change is increased red spatial
  coverage. Record to
  `outputs/unity_shadow/red_blue_rule_verification_scattered_20260801.jsonl`
  without overwriting the zero-red trial.
- Running repeat limitation: the matched seed-92 command intentionally enables
  only the frozen Unity geometry recurrent policy, terrain takeover, and
  four-step shadow MPC. It does not load the orbit-exit adapter, L-wall episodic
  hidden-goal adapter, AIR terrain route library, resource memory, conductor,
  systemic conductor, or adaptive GNW. The user observed a sustained rock-wall
  orbit with critical hunger. This behavior is therefore not evidence that the
  previously developed episodic/adapter stack failed; those modules are opt-in
  and absent from this command. Holding the weak controller fixed made the
  scene comparison cleaner, but it reduced the chance of collecting eligible
  red exposures. If the run remains exposure-ineligible, the next registered
  collection should restore the natural-terrain-relevant orbit-exit, AIR route,
  and resource-memory components while keeping the proposed red rule at zero
  action authority. The L-wall-specific hidden-goal adapter should remain off
  outside its validated context.
- Live repeat observation: the user reports one failure already, labeled
  `persistent_physics_wedge`, during the scattered-red seed-92 run. Treat this
  as a controller/navigation failure under the deliberately reduced stack, not
  as a red-rule counterexample. A scientifically eligible metabolic episode
  still requires an actual isolated red pickup followed by the registered
  7-14-second response window.
- Scattered-red repeat outcome (12:48 EDT completion): 5,713 frames over
  1,199.979 seconds, 24 pickups, and no pickup-counter resets. The controller
  saw red during one 23-frame contiguous episode, approached to 3.487 m, and
  collected it at step 3,068. That pickup was isolated and prospectively
  matched the compiled hypothesis: pressure delta +0.34 at 10.416 seconds
  versus the proposed 10.464-second delay. Passive Scientific GNW broadcast
  the red anomaly once with action influence 0. Fifteen isolated blue controls
  had mean pressure delta -0.0001 and zero positive responses.
- Pre-registered verdict: `supportive_but_inconclusive_insufficient_red_replication`.
  The blue threshold passed, and the one red observation was a clean predictive
  success, but only one isolated red pickup occurred versus the required two.
  Therefore this is not a verification pass, falsification, or Rule Commit;
  the candidate remains `proposed_unverified` with authority 0.0. The run also
  recorded one `persistent_physics_wedge` survival failure and one unstuck
  respawn under the reduced controller stack.
- Scattered-repeat artifacts:
  `outputs/tiny_scientist_rule_verification_scattered_evidence_20260801.json`
  and
  `outputs/tiny_scientist_rule_verification_scattered_summary_20260801.json`.
- Placement audit: Unity's live editor log confirms the tool placed all 40/40
  red mushrooms after 50 candidate attempts. The active
  `ChallengeTerrain.unity` file still has its pre-scatter 12:23 modification
  time, so the additive population is present in the open editor session but
  has not yet been saved to disk. Save the scene before closing Unity. The low
  exposure was therefore route sampling—one red among 24 collected foods—not
  a partial scatter operation.
- Placement-height correction: the first additive red tool used terrain height
  plus 0.05 m, which left much of the imported red mesh underground because of
  its pivot. The user reports that the established blue placement workflow
  needs a 1.5 m vertical offset for visible stems. Updated both live and mirror
  editor utilities to `Repair and Expand to 100 Red Food Mushrooms`: it finds
  the existing red root, lifts all contained reds to terrain +1.5 m, preserves
  their X/Z positions and red metadata, and deterministically adds enough new
  separated instances to reach 100 total. It does not clear or modify the blue
  scatter. The scene must be saved after running the repair.
- Robust-stack replication plan: after the user visually checked an initial
  sample of the repaired population, register a new 20-minute seed-92 run with
  a separate recording. This is not another reduced-controller matched cell:
  restore the natural-terrain orbit-exit adapter, passive terrain AIR memory,
  guided AIR route library, hunger-gated guided resource memory, frozen
  four-context systemic conductor, and bounded adaptive GNW routing. Keep the
  L-wall-only hidden-goal adapter disabled and keep the proposed red rule at
  authority 0.0. Apply the same eligibility threshold (at least two isolated
  red and six isolated blue episodes) and metabolic pass/falsification rules.
  Planned artifact:
  `outputs/unity_shadow/red_blue_rule_verification_fullstack_20260801.jsonl`.
  Some reds were reported on steep/inaccessible slopes, but the user judged the
  remainder acceptable; telemetry exposure counts, not nominal scene count,
  determine eligibility.
- Full-stack verification outcome (13:20 EDT completion): 5,551 frames over
  1,199.858 seconds, 30 pickups, 10 red pickups, zero survival failures, zero
  unstuck respawns, and one stuck event. All ten red episodes were isolated;
  every one produced a positive pressure response. Mean red delta was +0.3373
  (range +0.3256 to +0.3400), and mean delay was 10.890 seconds (range 10.731
  to 10.963). Fifteen isolated blue controls had mean delta -0.0150 and zero
  positive responses. There were no red counterexamples.
- Pre-registered verdict: `preregistered_verification_pass`. Every registered
  gate passed: at least two isolated reds (10), at least six isolated blues
  (15), all reds positive, red mean delta at least +0.30, red mean latency from
  9-11 seconds, blue positive fraction at most 0.10, and no red
  counterexample. Passive Scientific GNW broadcast nine red anomalies and had
  action influence 0. The candidate is now eligible to advance from
  `proposed_unverified` into verified shadow production memory, but no Rule
  Commit or authority change was performed during analysis.
- Restored-stack audit: MPC engaged for 4,127 frames; the orbit adapter was
  active for 296 frames across 19 events; systemic routing influenced 324
  frames, made 192 handoffs, and recorded 1,433 safety overrides; bounded
  adaptive GNW ignited 107 times and substituted 50 routing recommendations
  while retaining zero direct motor influence; guided AIR ran 21 interventions
  and changed seven action decisions; resource memory influenced three frames
  but changed zero actions. This confirms the requested stack was actually
  present. Compared with the reduced run, navigation and exposure improved,
  but the stack, red count, and vertical placement were changed together, so
  their individual causal contributions cannot be separated.
- Scientific boundary: this prospectively verifies the neuro-symbolic Tiny
  Scientist candidate inside the programmed synthetic Unity metabolism. It
  does not show unaided Gemma abduction, autonomous selection of the embodied
  intervention, or natural-world causal discovery. A staged Rule Commit should
  append the exact verified declarative rule to passive production memory
  first; it should not immediately modify Gemma weights, MPC weights, or GNW
  routing authority.
- Full-stack verification artifacts:
  `outputs/tiny_scientist_rule_verification_fullstack_evidence_20260801.json`
  and
  `outputs/tiny_scientist_rule_verification_fullstack_summary_20260801.json`.
- Session 9 end checkpoint: user reports 87% weekly usage remaining, down 13
  points from today's 100% start and one point above the planned 86% stop
  target. Stop work for the day after this administrative close.
- Exact Codex task-timer audit across the two relevant project tasks, using
  completed-turn `durationMs` values by America/New_York date: 49 minutes
  0 seconds on 2026-07-30; 40 minutes 16 seconds on 2026-07-31; and 44 minutes
  35 seconds on 2026-08-01 through the last completed substantive turn. The
  three-day total is 133 minutes 51 seconds (2 hours 13 minutes 51 seconds),
  and the arithmetic mean is 44 minutes 37 seconds per day. The in-progress
  administrative timing/logging turn is excluded to avoid recursively adding
  the act of counting time to the coding-work total; it can be reconciled from
  task history later if a full all-turn audit is desired.
- Session 9 technical endpoint: neuro-symbolic hypothesis formulation passed
  with Gemma 3 1B plus the formal falsifier compiler, and the subsequent
  full-stack prospective Unity run passed every pre-registered metabolic
  verification gate. The verified rule is eligible for a future passive Rule
  Commit but remains uncommitted with authority 0.0 at shutdown.

## Session 10 — Publication checkpoint (2026-08-02)

- User-reported budget/context checkpoint: 86% weekly budget remaining at
  session start (14% available). Scope for this short session: publish the
  successful neuro-symbolic Tiny Scientist result before selecting new
  experiments. Private usage/context logs remain excluded from publication.
- GitHub publication: created and pushed branch
  `codex/tiny-scientist-neurosymbolic` at commit `a5a906c` and opened draft
  PR #16, “Add neuro-symbolic Tiny Scientist verification.” The focused,
  coherent commit includes Tiny Scientist implementation/tests, the full
  embodied stack needed to reproduce the run, Unity red-food integration,
  compact result artifacts, checkpoints, and root reproducibility docs. It
  intentionally excludes raw Unity recordings, `DEVELOPMENT_LOG.md`, and
  unrelated in-progress experiments already present in the dirty worktree.
- Validation before publication: `git diff --cached --check` passed; the
  focused Python suite passed 86 tests (`test_tiny_scientist`, systemic
  conductor, resource memory, AIR route controller, escape teacher, adaptive
  GNW, embodied Unity loop, and the systemic analyzers).
- Website publication: updated the existing Tiny Consciousness Lab site with
  a bounded-claim Tiny Scientist section (10/10 isolated red confirmations,
  15/15 negative blue controls, 10.890 s mean latency, 0 red
  counterexamples), linked compact artifacts/branch, and stated the no-motor-
  authority and programmed-synthetic-metabolism boundary. Website source
  commit `37bb31f` passed `pnpm run build` and rendered-page tests (2/2), was
  saved as site version 15, and was deployed successfully at
  `https://tiny-consciousness-lab.pix3ldust418476.chatgpt.site`.
- Recommended next experimental step (not begun in this session): a staged,
  authority-0 Rule Commit into passive production memory, followed by a
  preregistered held-out replication on a new Unity seed/placement layout with
  the current navigation stack frozen. This distinguishes retention and
  generalization of the verified declarative rule from the original discovery
  and verification episode, without granting the language model motor or
  online rule-editing authority.

## Session 10 — Passive Rule Commit (2026-08-02)

- User-reported budget checkpoint after publication: 84% weekly budget
  remaining. User selected the conservative next step before attempting active
  behavioral exploitation; no Tegmark/IIT measurement is started.
- Implemented `tiny_scientist_production_memory.py`: a strict production-memory
  format that accepts only a `preregistered_verification_pass` with the
  registered eligible-rule status, validates the exact red-pickup/increased-
  internal-pressure/delayed contract, and rejects nonzero authority or any
  control permission. `embodied_unity_loop.py` can load it solely for passive
  telemetry via `--tiny-scientist-production-memory`; it has no MPC, GNW,
  routing, or motor call path.
- Committed the verified full-stack summary to
  `checkpoints/tiny_scientist/verified_production_memory_20260802.json`.
  Audit: one rule (`red_pickup_delayed_pressure_increase_v1`), authority 0.0,
  empty control permissions, and zero action influence. The commit operation
  is idempotent and conflict-checked.
- Validation: Tiny Scientist plus embodied-loop tests passed (49 tests), and
  both modified modules passed Python compilation. Next registered collection
  should use a new controller seed and a distinct JSONL recording, with this
  passive memory loaded, before any planning authority is considered.
- Held-out passive-memory run outcome: seed 93 ran for 1,199.808 seconds
  (5,618 frames), with 32 total pickups and 14 red pickups. Thirteen red
  pickups had enough post-pickup time for an isolated metabolic assay; all 13
  were positive (mean pressure delta +0.3302, mean delay 10.723 s), with zero
  counterexamples. Eleven isolated blue controls had mean delta -0.0125 and
  zero positive responses. Every registered red/blue criterion therefore
  passed again. The final red pickup occurred less than a full observation
  window before shutdown and is excluded rather than treated as a failed assay.
- Passive-memory audit: the committed one-rule store was loaded from first to
  last telemetry frame with authority 0.0, no control permissions, and zero
  action influence. This run has zero survival failures and zero unstuck
  respawns, though it recorded five stuck events. Verdict:
  `held_out_passive_rule_commit_replication_pass`. Artifact:
  `outputs/tiny_scientist_rule_commit_holdout_evidence_20260802.json` and
  `outputs/tiny_scientist_rule_commit_holdout_summary_20260802.json`.
- Interpretation and next boundary: the system now demonstrates verified-rule
  retention and observational availability across a new controller seed; it
  does not yet demonstrate behavioral use. The next distinct experiment may
  grant the *planner* (not Gemma) narrow, audited read access to this frozen
  rule under a preregistered pressure-depletion challenge. No direct language
  model, GNW, or production-memory motor authority is warranted.
- Active-use design checkpoint: before staging code, wrote
  `ACTIVE_RULE_USE_PROTOCOL_20260802.md`. It identifies a semantic mismatch in
  the suggested “seek red before a pressure crash” design: current metabolic
  pressure is a stress/cost signal, while the verified red effect increases
  it. The valid immediate rule-use experiment is therefore bounded,
  pressure-aware *avoidance* of red in matched red/blue choices near a fixed
  safety threshold. The documented utility keeps existing MPC as motor owner
  and adds only a capped forecast penalty for a red candidate. A future
  beneficial `metabolic_reserve` experiment would be a distinct programmed
  contingency requiring its own discovery and verification; it cannot inherit
  authority from the verified pressure rule.
- Active-use implementation in progress: added Unity bridge telemetry for the
  nearest visible/reachable red food and nearest visible/reachable blue food,
  in addition to the old generic nearest-food sensor. Added a new
  `--tiny-scientist-rule-control pressure_avoidance` mode in the Python loop.
  It can activate only when the frozen committed rule exists, red and blue are
  both visible, their distances differ by no more than 2 m, and the verified
  +0.34 red increment would cross a fixed 0.70 pressure threshold. Its only
  permitted effect is a capped 0.05 ambiguity-gated MPC heading prior toward
  blue; collision masking, recovery, and all motor selection remain intact.
  JSONL telemetry logs activation, predicted risk, target feature, effective
  weight, and actual action changes.
- Interim validation: Python compilation, focused Tiny Scientist/embodied-loop
  tests (49), CLI help, and diff whitespace checks pass. The Unity editor
  binary is not discoverable from this shell, so a live editor recompilation
  and short telemetry smoke test remain required before registering an active
  behavioral sweep. No active-rule sweep has run yet.
- Unity path correction: inspection of the running editor process found that
  Unity is open on `/Users/dustinoconnor/My project`, not the repository mirror
  at `unity/TrapCourseLab`. The initial bridge edit therefore could not import.
  Compared rather than overwrote the live bridge, then applied the same
  red/blue-candidate telemetry addition to the live project's
  `Assets/Scripts/RobotUdpBridge.cs`. Unity's AssetImportWorker log confirms
  that file imported after the edit, with no `RobotUdpBridge` C# error logged.
- First 120-second terrain smoke (seed 94): 526 JSONL frames, active
  `pressure_avoidance` configuration, committed memory loaded at authority 0,
  no rule activation/action influence, and no survival failure. This does not
  test behavioral avoidance. Initial analysis exposed a Python recorder
  omission: Unity's new per-feature fields were not copied into JSONL, so the
  recording cannot establish their arrival. Added red/blue visibility,
  distance, and world-direction fields to `ShadowRecorder`; Python compilation
  and embodied-loop tests passed (30). One short replacement smoke recording
  is required before any behavioral sweep.
- Replacement smoke outcome: 563 frames across 119.907 seconds. All required
  red/blue candidate and rule-control telemetry fields were present; committed
  memory remained one read-only authority-0 rule. The terrain path saw blue on
  77 frames but saw no red and therefore no dual-visible matched choice. Rule
  activation and action influence correctly stayed zero, and there were no
  survival failures. Verdict: telemetry wiring pass, behavioral-exposure
  ineligible. The next implementation requirement is a controlled matched
  red/blue terrain choice opportunity near the registered pressure threshold;
  another unstructured foraging sweep would not answer the active-rule-use
  question.
- Natural-terrain exposure run (seed 95, 1,192.476 seconds, 5,539 frames):
  27 total pickups, including six red. All six red episodes were isolated and
  positive (mean pressure delta +0.3376; mean delay 10.828 s); twelve isolated
  blue controls had mean delta -0.0037 and zero positive responses. This is a
  further observational replication of the existing rule, with no survival
  failures or respawns and four stuck events.
- Active-use eligibility: red was visible on 115 frames and blue on 442, but
  there were zero simultaneous red/blue visible frames. There were also zero
  forecast threshold-risk states. Consequently the bounded pressure-avoidance
  gate had zero active frames and zero action influence, exactly as designed.
  Verdict: natural exposure is sufficient to replicate the metabolic effect
  but insufficient to test a deliberate planner choice. A controlled paired
  terrain opportunity and an independently induced, registered pressure state
  remain necessary before an active-rule-use claim.
- Paired-choice setup: user placed one additive red/blue terrain pair near the
  terrain spawn, approximately two Unity meters apart and to the same side of
  a nearby tree. Exact world distances are not assumed; the next telemetry
  smoke must confirm simultaneous line-of-sight and the registered matched-
  distance gate. Existing natural scatter remains unchanged.
- Added `--initial-metabolic-pressure` as the registered diagnostic challenge:
  it sets only the initial normalized internal pressure and does not alter
  food, Unity physics, rule content, or motor authority. Use 0.75 for the
  paired-choice smoke, which makes the frozen +0.34 red forecast exceed the
  fixed 0.70 safety threshold at start. Python compilation, embodied-loop
  tests (30), and CLI argument validation pass.
- Paired-choice smoke (seed 96, 119.947 seconds, 565 frames): blue was visible
  for 42 frames (closest observed distance 3.130 m), but red was never visible.
  There were therefore zero dual-visible frames, zero forecast-risk states,
  zero rule-gate active frames, and zero rule-driven MPC action changes. Two
  ordinary pickups occurred, none red; no survival failure occurred. No pickup
  was required for temporal forecasting—the gate could have activated before
  eating—but a visible red candidate is required. Verdict: placement/line-of-
  sight failure, not a temporal-foresight or planner failure. Move the red
  paired item beside the visible blue item on the same unobstructed side of the
  tree, then repeat the short smoke.
- Paired-choice smoke v2 (seed 96, 108.807 seconds, 509 frames): red and blue
  were simultaneously visible for 12 initial frames, and the 0.75 initial
  pressure challenge was active. However, red was 3.59–3.84 m nearer than blue
  (red 11.26→3.88 m; blue 14.85→7.91 m), exceeding the preregistered 2 m
  matched-distance gate. The bounded rule correctly stayed inactive and did
  not override the objectively nearer red target; red was picked first at step
  12, then blue at step 19. This is not a failed temporal forecast. The pair
  must be repositioned on a lateral/equal-distance arc from the spawn—rather
  than one item farther along the robot's approach direction—before another
  paired-choice run can count.
- Paired-choice smoke v3 (seed 96, 112.767 seconds, 526 frames): placement
  now met the geometry gate. Red and blue were simultaneously visible for 12
  frames, with distance differences 0.395–0.944 m (within the registered 2 m
  limit). With initial pressure 0.75, the frozen red +0.34 forecast yielded
  positive predicted risk (maximum 0.3788), and the bounded rule gate was
  active for all 12 matched-choice frames. This is the first successful
  perception-to-forecast activation under the committed rule.
- Behavioral result: the ambiguity-gated MPC prior was intentionally very
  small (maximum effective weight 0.00713 against a 0.05 cap) and did not
  alter the base MPC action; red was still collected first at step 12, then
  blue at step 15. Therefore the run does not yet demonstrate active
  behavioral exploitation or avoidance. It validates the placement and
  forecast gate, while showing that the current soft heading tie-breaker is
  too weak/late for a decisive paired-target test. Do not move the pair again;
  the next code change should make the registered candidate-level utility
  comparison explicit while retaining collision masking and bounded authority.
- Candidate-level planner revision: replaced the weak direction-only
  ambiguity tie-breaker with an explicit, logged utility comparison on the
  already registered matched-choice gate. For each simultaneous candidate,
  `U = 0.35 - 0.01 * distance - predicted_pressure_risk`; red alone receives
  the frozen +0.34 forecast cost. Only if blue utility is higher does the
  planner add a bounded (maximum 0.25) blue-alignment / red-opposition term to
  collision-masked MPC. It remains inactive outside a matched red/blue pair,
  cannot bypass recovery or safety masks, and still has no Gemma/GNW direct
  motor path. Added utility telemetry and two focused tests for activation and
  no-activation conditions. Python compilation, 51 focused tests, and diff
  whitespace checks pass. The Unity scene/pair is unchanged.
- Paired-choice smoke v4 (seed 96, 112.287 seconds, 528 frames): the explicit
  candidate-level utility path activated on all 12 matched dual-visible frames.
  It selected blue as higher utility (first frame: red -0.1413 versus blue
  +0.2335), applied the full bounded 0.25 weight, entered
  `tiny_scientist_pressure_avoidance_mpc`, and changed two MPC selections.
  This is the first observed planning influence from the frozen verified rule.
- Outcome boundary: red was nevertheless collected first at step 12, followed
  by blue at step 14. Red remained 0.4–0.9 m nearer and its large pickup
  volume lay in the route corridor, so the planned blue preference did not win
  the final physical pickup race. This is evidence of rule-to-planner influence
  but not evidence of successful blue-over-red behavioral avoidance. For the
  definitive behavioral test, keep the equal-distance constraint but increase
  lateral separation/clearance so a route to blue cannot trigger red's pickup
  collider; then compare authority-0 and planning-read conditions over matched
  repeated trials.
- Authority-0 paired baseline after wider lateral placement (seed 96, 113.489
  seconds, 534 frames): red and blue were jointly visible for 12 frames. Their
  distance difference ranged 1.062–2.268 m; the latter is outside the strict
  matched gate and will be ignored automatically by planning-read mode. Rule
  control was passive throughout, with zero active frames and zero action
  influence. Red was collected first at step 12 and blue at step 17; no
  survival failure occurred. This establishes the expected no-authority
  baseline, though the pair remains slightly red-nearer and needs per-frame
  eligibility filtering in the planning counterpart.
- Matched planning-read counterpart (seed 96, 112.628 seconds, 526 frames):
  exposure matched the authority-0 run exactly at 12 dual-visible frames and
  10 strict eligible frames. The committed rule activated on all 10 eligible
  frames and reached the full 0.25 bounded candidate utility weight, but it
  changed zero MPC selections. Both authority-0 and planning-read conditions
  collected red first at step 12, with no survival failure. Verdict: clean
  negative behavioral result for this particular bounded soft-prior design.
  It demonstrates forecast availability but not improved avoidance; do not
  claim successful temporal behavioral exploitation. A future revision would
  need a preregistered target-lock/target-selection interface whose effect is
  assessed in a fresh counterbalanced protocol, rather than repeatedly tuning
  this soft-prior cell after observing its outcome.
- Work-time audit (2026-08-02): summing every completed Codex task work-time
  label in this thread from the 04:00 EDT day boundary through the last
  completed turn gives 1,808,077 ms = 30 minutes 8 seconds. This is the exact
  completed-turn task total available at audit time; the in-progress audit/log
  turn is excluded until it completes, and Unity run/wait time is not counted
  because it is outside Codex task work-time labels.

## Session 11 — Agent-Native Causal Representation (2026-08-04)

- Usage/time protocol: user reported 70% weekly budget remaining at session
  start and selected 44% as today's stopping target, treating the 26-point
  allowance as approximately two ordinary workdays. Continue to count Codex
  completed-turn task time rather than Unity wait time or conversational wall
  time.
- Research question: test whether replacing the Tiny Scientist's verbose JSON
  hypothesis boundary with a compact agent-native causal representation can
  reduce retry-adjusted Gemma inference tokens without reducing grounded
  causal accuracy. The proposed 50% reduction is a hypothesis, not a result.
- Implemented causal IR v1 with the strict grammar
  `C1 cause comparison effect delay confidence`, where effect is one of
  `+`, `0`, or `-`. A strict parser compiles the six atoms into the existing
  canonical bound schema; the same evidence-grounding validator, formal
  falsifier compiler, authority 0.0 boundary, and evidence-order audit remain
  in force. Python remains the human-readable implementation language; only
  the model interchange representation changes.
- Added `benchmark_causal_ir.py` for matched JSON-versus-C1 trials using the
  same embodied recording, homeostatic evidence, greedy decoding, canonical
  and reversed evidence order, and already cached Gemma 3 270M/1B models. It
  records exact tokenizer input/output tokens for every attempt, elapsed
  inference time, rejection reasons, success rate, and retry-adjusted tokens
  per accepted hypothesis. This prevents character-count claims and charges
  malformed outputs/retries to the representation that caused them.
- Initial validation: Python compilation and 23 focused Tiny Scientist/causal-
  IR benchmark tests pass. Live matched model trials and result analysis are
  next; no efficiency or publication claim has yet been made.
- Static tokenizer result: the canonical compact JSON hypothesis is 39 Gemma
  tokens and the equivalent C1 record is 17, a 56.41% output reduction. The
  unscaffolded chat prompt fell from 249 to 152 tokens (38.96%). These are
  representation measurements only, not discovery-cost results.
- Free-generation audit: Gemma 270M accepted neither JSON nor C1 in two
  counterbalanced orders; C1 used 441 versus JSON's 760 mean retry-adjusted
  tokens per trial (41.97% less) while failing. Gemma 1B also accepted neither;
  C1 used 414.5 versus 743 tokens (44.21% less) while failing. Both small models
  commonly ignored the unfamiliar C1 grammar, while the 1B JSON baseline bound
  an aggregate label rather than an observed feature.
- Strict symmetric candidate decoding removed syntax errors but did not repair
  semantic selection. Unscaffolded JSON and C1 both scored 0/2 with Gemma 1B.
  A single abstract negative example produced a 1/2 JSON result but 0/2 C1 and
  visibly biased/copying outputs, so it was rejected as a design endpoint.
- Balanced-scaffold audit: added two unrelated examples with opposite effect
  directions and opposite causal-feature row positions. With identical
  symmetric candidate spaces, JSON passed canonical order but failed reversed
  order (1/2); C1 failed both (0/2). C1 still used fewer visible tokens (254
  versus 415 per trial), but its retry-adjusted cost per accepted hypothesis is
  undefined rather than cheaper.
- Dependency-order experiment: C2 changed the grammar to
  `C2 effect delay cause comparison confidence`, requiring the model to select
  the measured effect before feature binding. In canonical order it correctly
  selected red and pressure increase but selected a zero-second delay, which
  the unchanged grounding verifier rejected. In reversed order it selected the
  wrong effect. C2 therefore scored 0/2; it used 167.5 versus constrained
  JSON's 288 tokens per trial (41.84% less) while failing.
- Final bounded verdict: `token_reduction_without_success_preservation`. The
  compact representations repeatedly reduce visible input/output tokens by
  roughly 39–44%, and the isolated canonical output by 56%, but no compact-IR
  condition produced an accepted hypothesis. Therefore the proposed 50%
  inference-cost-per-success improvement is not supported, and this is not yet
  a positive publishable discovery. The important finding is that unfamiliar
  grammar and atom order change small-model causal binding; grammar constraints
  guarantee syntax but not semantics.
- Next clean experiment (not begun): generate synthetic causal tables spanning
  feature identities, row orders, positive/negative/no-change effects, and
  delays; train or adapter-tune C1/C2 on a training split; freeze the grammar
  and model before evaluating success-at-budget on held-out tables and the
  untouched Unity rule. Do not continue prompt-tuning against this red/blue
  recording. Summary artifact:
  `outputs/tiny_scientist_causal_ir_summary_20260804.json`.
- Final validation for this checkpoint: Python compilation, 26 focused tests,
  and the full 173-test repository suite pass. Unity was not modified or run;
  the experiment is entirely at the passive authority-0 hypothesis boundary.
- Usage checkpoint after completing and analyzing the initial causal-IR phase:
  user reported 67% weekly budget remaining, down three percentage points from
  the 70% session start and still 23 points above today's 44% stopping target.
  Interpret the result as approximately 39–44% fewer live visible tokens (and
  56.41% for the isolated canonical output), not yet lower cost per successful
  hypothesis because all compact-IR hypotheses failed verification. User
  selected a LoRA/adaptor post-training experiment as the likely next phase.
- LoRA phase authorization: user reported 67% remaining and authorized matched
  post-training while the daily budget was available. Installed `peft` 0.20.0
  (775.8 kB package) and reused the cached Gemma 3 270M checkpoint; no new base
  model was downloaded. Each rank-8 q/v adapter trains 368,640 parameters and
  occupies approximately 33 MB including tokenizer/configuration files.
- Implemented `causal_dsl_lora.py`: deterministic synthetic causal-table
  generation, assistant-only supervised loss, MPS LoRA training, strict syntax
  parsing, exact semantic scoring, and visible input/output token accounting.
  Training features exclude red and blue. Held-out pairs include red/blue plus
  silver/gold, teal/pink, and copper/indigo. JSON and DSL conditions use the
  same evidence and examples; only their output instructions and targets
  differ.
- The first eight-example smoke exposed and repaired a Transformers 5.9
  `BatchEncoding` compatibility error before any model update completed. The
  replacement smoke trained and saved normally; its zero accuracy was expected
  at that deliberately tiny size.
- Initial 192-example adapters revealed an index-based curriculum confound:
  feature identity, target position, direction, and evidence order were
  partially correlated. C1 reached 12/32 exact (37.5%, 29/32 syntactically
  valid) while JSON reached 23/32 (71.875%, 32/32 syntactically valid). Error
  decomposition showed primarily cause/control swaps. These v1 adapters remain
  as audit artifacts but are not used for the final comparison.
- Replaced the generator with a factorial curriculum spanning all pairwise
  combinations of eight training features, both targets, both pressure
  directions, both evidence orders, independent magnitudes, and independent
  delays. Retrained both formats from the same base and seed. JSON v2 reached
  30/32 exact (93.75%) at 282.31 mean visible tokens. Unlabeled positional C1
  v2 reached only 9/32 (28.125%) at 163.19 tokens. This establishes that maximum
  positional compression teaches syntax but seriously damages held-out role
  binding in Gemma 270M.
- Designed labeled DSL L1 as the minimal middle ground:
  `L1 c cause k comparison e effect t delay q confidence`. Its canonical output
  is 21 Gemma tokens versus 38 for JSON while retaining one-token semantic role
  markers. The matched factorial L1 adapter reached 29/32 exact (90.625%) at
  184.28 mean visible tokens, close to JSON's accuracy with substantially less
  inference text.
- Frozen replication: after all three adapters and prompts were frozen, added
  `compare_causal_dsl_adapters.py` and evaluated JSON v2 versus L1 on 128 new-
  seed held-out examples. Both scored exactly 116/128 = 90.625%. Paired outcomes
  were 104 both correct, zero both wrong, 12 JSON-only correct, and 12 L1-only
  correct. Mean visible tokens were 282.367 for JSON and 184.367 for L1: L1
  reduced them by 34.7066% with zero point-estimate accuracy loss. Because no
  non-inferiority margin was preregistered, treat this as preliminary parity,
  not formal proof of non-inferiority.
- Untouched Unity transfer audit used
  `outputs/unity_shadow/red_blue_tiny_scientist_20260731.jsonl` in canonical and
  reversed evidence order. JSON failed both orders; L1 passed reversed order
  only. All outputs bound red, blue, and the 10.464-second delay correctly, but
  canonical order selected pressure decrease. Therefore robust embodied
  transfer is not demonstrated and no production-memory or control authority
  changes are permitted.
- Bounded result: the labeled agent-native DSL is the first condition to retain
  the JSON adapter's frozen synthetic held-out point accuracy while reducing
  visible tokens 34.7%. The stronger original 50% goal is not met, positional
  C1 is rejected, and a publishable embodied-efficiency claim requires a
  preregistered multi-seed adapter replication plus Unity-order robustness.
  Artifacts: `outputs/causal_dsl_frozen_comparison_20260804.json`,
  `outputs/causal_dsl_lora_summary_20260804.json`, and the three v2 adapters in
  `checkpoints/causal_dsl_{json,c1,l1}_lora_v2_20260804`.
- Validation after the frozen comparison: Python compilation, focused tests,
  diff whitespace checks, and the full repository suite pass (177 tests).
- Usage checkpoint after LoRA training/frozen comparison: user reported 63%
  remaining, four points below the 67% checkpoint and 19 points above today's
  44% stopping target. User selected live Tiny Scientist integration next and
  asked about model-size scope. Clarification: the successful labeled-DSL
  adapter is for Gemma 3 270M only; it scored 116/128 rather than perfect
  binding and passed only one of two original-Unity evidence orders. No Gemma
  3 1B DSL LoRA has been trained yet; the earlier 1B success used the verbose
  neuro-symbolic/freeform contract.
- Integrated frozen adapters into the production Tiny Scientist CLI via
  `--adapter` and added the `labeled_causal_ir` hypothesis contract. Adapter
  loading wraps the named base model with PEFT, uses the adapter's matching
  tokenizer, compiles L1 into the unchanged canonical semantic verifier and
  formal falsifier, and records the adapter path in the artifact. This remains
  offline/passive: Unity collection, MPC, GNW routing, production memory, and
  motor authority have no adapter call path.
- Production-path counterbalance on the original embodied recording: canonical
  evidence order was rejected after retry (`L1 c tiny red k blue - e - t
  10.464 q 0.8`, malformed and wrong direction); reversed order emitted exactly
  `L1 c red k blue e + t 10.464 q 0.8` and compiled successfully to red cause,
  blue comparison, pressure increase, 10.464-second delay, formal red
  falsifier, status `proposed_unverified`, and authority 0.0. Artifacts:
  `outputs/tiny_scientist_l1_unity_canonical_20260804.json` and
  `outputs/tiny_scientist_l1_unity_reversed_20260804.json`. This confirms real
  CLI integration but also replicates evidence-order fragility; neither order
  may be selected post hoc as the sole result.
- Prepared the next clean embodied collection as seed 97, 1,200 seconds, using
  the validated terrain recurrent/MPC controller, orbit-exit adapter, systemic
  conductor with bounded GNW, passive AIR memory, guided ART route library,
  guided resource memory, and the frozen production rule in passive mode.
  Planned recording:
  `outputs/unity_shadow/causal_dsl_l1_unity_transfer_seed97_20260804.jsonl`.
  The L1 adapter runs only after collection in both canonical and reversed
  evidence order; keeping it out of the live motor loop preserves causal and
  timing boundaries.
- Integration validation: Python compilation, CLI help, diff hygiene, and the
  full repository suite pass (177 tests).
- Seed-97 collection completed: 5,557 frames over 1,188.739 measured seconds,
  33 pickups including four red, zero survival failures, zero respawns, and two
  stuck events. Frozen production memory remained loaded at authority 0.0 with
  no permissions/action influence; Tiny Scientist rule control stayed passive.
- Initial one-sided-isolation summary appeared eligible with two isolated reds
  and 19 isolated blues, but showed two anomalous positive blue controls. Both
  blue pickups occurred only 1.068 and 0.870 seconds after red pickups and
  inherited those reds' delayed pressure rise. `metabolic_episodes` previously
  labeled isolation using only the *next* pickup, so a control immediately
  following a cause could be falsely treated as isolated.
- Corrected the assay to bidirectional isolation: an episode is confounded if
  another pickup occurs anywhere within the 14-second window before or after
  it. Added a regression test reproducing a blue pickup one second after red.
  The full suite now passes 178 tests.
- Corrected seed-97 evidence: one isolated red with delta +0.3400 and delay
  10.724 s; ten isolated blues with mean delta -0.0046, zero positive fraction,
  and no positive-delay statistic. The run is therefore
  `inconclusive_insufficient_bidirectionally_isolated_red_exposure` against the
  registered minimum of two reds—not a causal failure or verification pass.
- Frozen diagnostic on corrected evidence (not an eligible preregistered
  endpoint): L1 canonical remained malformed/wrong-direction; L1 reversed
  emitted the correct red/blue/+10.724 rule. Frozen JSON rejected both orders.
  L1 therefore remains order-sensitive and robust Unity transfer is not yet
  demonstrated.
- Historical impact audit: recomputing the original discovery, full-stack
  verification, and passive Rule Commit holdout with bidirectional isolation
  leaves their criteria intact. Counts are respectively red/blue 2/8, 10/12,
  and 13/7; all retained red responses and blue controls satisfy the registered
  thresholds. Thus the isolation fix changes some control counts but does not
  overturn the verified red-pressure rule or passive Rule Commit.
- Seed-97 audit artifact:
  `outputs/causal_dsl_l1_unity_seed97_summary_20260804.json`. Preserve both the
  pre-fix diagnostic artifacts and new `*_bidirectional_*` artifacts for an
  explicit audit trail rather than overwriting observed failures.
- Usage checkpoint after seed-97 analysis/isolation repair: user reported 60%
  remaining, three points below the 63% checkpoint and 16 points above today's
  44% stopping target. User selected matched Gemma 3 1B LoRA training next due
  concern that 270M is not robust enough for embodied transfer.
- Scaling clarification: the 270M frozen synthetic audit showed equal point
  accuracy for JSON and L1 (116/128 each) with 34.7% fewer L1 visible tokens;
  it did not show error-free binding. The Unity audit was only two evidence-
  order trials, not another 128-case accuracy estimate, and became exposure-
  ineligible after the assay fix. Real Unity evidence also contained temporal
  confounding/noise absent from the clean synthetic curriculum. The 270M phase
  remains useful as a capacity-floor/representation result and found a pipeline
  bug that would have contaminated a 1B test, but it does not establish robust
  embodied efficiency.
- Registered next comparison: train both 1B L1 and 1B JSON rank-8 adapters from
  the same cached `google/gemma-3-1b-it` base, seed-104 factorial curriculum,
  192 examples, two epochs, and identical q/v LoRA targets. Freeze both before
  a new-seed synthetic comparison and corrected Unity counterbalance. Training
  L1 alone would confound model scale with representation, so JSON remains the
  required matched control.
- Completed the registered matched 1B training. Both adapters used the same
  seed-104 factorial 192-example curriculum, two epochs, rank-8 q/v LoRA, and
  gradient accumulation of eight. L1 trained in 338.068 s and its initial
  32-case check scored 30/32 exact (93.75%) at 184.375 mean visible tokens.
  JSON trained in 524.964 s and scored 32/32 exact at 282.313 tokens. These
  initial checks are development diagnostics, not the final frozen comparison.
- Frozen seed-205 evaluation on 128 new examples: JSON scored 128/128 (100%);
  L1 scored 121/128 (94.53125%). Paired outcomes were 121 both correct, seven
  JSON-only correct, zero L1-only correct, and zero both wrong. L1 reduced mean
  visible tokens from 282.367 to 184.430 (34.6844%) but lost 5.46875 percentage
  points of exact semantic accuracy. Its failures included repeated cause/
  control atoms, inserted numeric fragments, and malformed effect binding.
- Corrected seed-97 Unity evidence remained an exposure-ineligible diagnostic:
  there was only one bidirectionally isolated red pickup and two evidence-order
  trials. JSON passed canonical and failed reversed (1/2); L1 failed both
  (0/2). This does not estimate embodied accuracy or overturn the verified
  red-pressure rule, but it supplies no evidence that L1 is ready for embodied
  production use.
- Tested one final compact representation to determine whether meaningful role
  words repaired the one-letter binding problem: H1 uses `cause`, `control`,
  `effect`, `delay`, and `confidence` while retaining a flat strict grammar.
  Added its prompt, parser, training target, comparison option, and regression
  coverage. Its rank-8 1B adapter trained in 379.568 s; the initial 32-case
  development check scored 28/32 exact (87.5%) at 170.063 mean visible tokens.
- Final untouched seed-306 H1 screen on 64 new cases: JSON again scored 64/64;
  H1 scored 56/64 (87.5%). H1 reduced mean visible tokens from 282.328 to
  169.594 (39.9303%) but lost 12.5 accuracy points. All eight paired errors were
  JSON-only successes; H1 inserted stray numeric atoms or swapped cause and
  control. Its two-item Unity diagnostic scored 2/2 versus JSON 1/2, but this
  tiny, exposure-ineligible result cannot outweigh the controlled 64-case test.
- Final 1B verdict: scaling from 270M improved L1 point accuracy from 90.625%
  to 94.531% on the seed-205 128-case audit, but the matched 1B JSON adapter
  improved to 100%. Compact L1/H1 outputs save approximately 35-40% of visible
  tokens, yet neither preserves JSON reliability. Do not claim a 50% cost cut,
  equal performance, robust embodied transfer, or production readiness. Keep
  JSON as the reliable interface; retain compact DSLs as research artifacts.
  Summary: `outputs/causal_dsl_1b_summary_20260804.json`.
- Post-result validation: the full repository suite passes all 178 tests and
  `git diff --check` reports no whitespace errors. No Unity scene, GNW/MPC
  authority, or production-memory authority was changed during this phase.
- Usage checkpoint after completing the 1B analysis: user reported 51% weekly
  budget remaining, down nine percentage points from the 60% checkpoint and
  seven points above today's 44% stopping target.
- Decision/claim framing: the 1B L1 result is a Pareto tradeoff, not a strict
  improvement—34.6844% fewer visible tokens in exchange for a 5.46875-point
  exact-reliability loss versus matched JSON. Use JSON as the production causal
  interchange because exact cause/control/effect binding is more important than
  the current inference-token reduction. Retain L1/H1 only as experimental
  representations pending a frozen test that preserves JSON-level reliability.
- Publication phase authorized: stop representation tuning against the observed
  evaluation sets and publish the qualified result, including the negative
  outcome. Added `CAUSAL_DSL_PARETO_RESULT_20260804.md` and a website subsection
  reporting JSON 128/128, L1 121/128, the 34.684% token reduction, 5.469-point
  reliability tax, H1 replication, and the exposure-ineligible Unity boundary.
- GitHub publication completed on `codex/tiny-scientist-neurosymbolic` at commit
  `4ad3484` and pushed to draft PR #16. Published only the causal-language code,
  tests, frozen evaluation artifacts, qualified report, and development log;
  excluded approximately 105 MB of duplicate adapter/tokenizer directories and
  unrelated dirty-worktree experiments. The repository, focused, site-render,
  build, and whitespace validations pass.
- Website source was built, validated, pushed to its existing Sites source
  repository, and saved as version 16. The existing site is public, so the
  final production deployment is paused for the explicit public-deployment
  confirmation required by the hosting workflow.
- User explicitly approved public deployment. Sites version 16 deployed
  successfully to the existing Tiny Consciousness Lab production URL. No MPS
  inference or training was run during publication because the user reserved
  the GPU for a Comfy render.
- Final usage checkpoint: user reported 5% remaining and reserved it for Comfy
  prompts/dream-loop work. Codex timing records for 4 August contain 102 min
  21 s across completed work turns plus approximately 1 min 36 s for the final
  deployment/counting turn: 103 min 57 s total, rounded to **104 minutes**.
  Per user instruction, divide this two-day-equivalent budget by two: **52
  minutes of effective coding/research work per day**.
- 5 August starting checkpoint: user reported 44% remaining, a 30% nominal
  stopping target, and plans to reserve roughly four percentage points for
  dream-loop work, leaving approximately ten points for this lab. Recommended
  no additional LoRA or MPS run today: yesterday's frozen Pareto result is a
  clean stopping point, and tuning again against its observed failures would
  weaken the evaluation. Highest-value future repair is a preregistered formal
  candidate-selection assay: code enumerates all symmetric valid causal JSON
  candidates, the model selects one candidate ID, and the formal layer maps the
  ID back to canonical JSON. This tests whether serialization can be made cheap
  without sacrificing semantic binding or leaking the answer. JSON remains the
  production interface meanwhile.
- Paper triage for the bounded 5 August phase: verified all three suggested
  publications. Selected Wu, Geiger, and Millière's ICML 2025 variable-binding
  work as the only direction directly diagnostic of yesterday's cause/control
  failures. Its actual experiment trained a small Transformer on synthetic
  variable-dereferencing programs and used linear probes plus causal
  interventions to identify residual-stream/addressable-memory and specialized
  attention-routing mechanisms. A Gemma hook alone cannot establish the same
  circuit claim; our clean extension would require paired correct/error cases,
  layerwise role decodability, and causal activation patching or head ablation.
- Rejected Mixture-of-Depths as today's task: the published result trains MoD
  architectures with learned per-block routers, fixed token capacities, and an
  autoregressive routing predictor. It is not a drop-in inference wrapper for a
  frozen Gemma LoRA, and local MPS top-k/gather overhead could erase theoretical
  FLOP reductions. Also rejected `Beyond Markov` as an immediate implementation
  recipe: it is a conceptual account of transformers as non-Markovian
  generative models using memory and attention, not the claimed discrete-state
  variational-free-energy Unity algorithm. That algorithm would be a new design
  inspired by the paper rather than a reproduction.
- Implemented and preregistered `variable_binding_probe.py` without loading the
  model. The deterministic cohort contains 32 frozen seed-205 cases, retains
  all seven L1 failures, assigns eight cases to each held-out feature pair, and
  balances the correct cause between evidence rows 16/16. The primary endpoint
  is cause-row decoding at every residual layer with leave-one-feature-pair-out
  ridge probes and 100 within-pair permutations corrected for choosing the
  maximum layer.
- Added discovery/confirmation separation for the mechanistic claim. Even
  original indices rank heads by cause-minus-control evidence attention; odd
  indices test pre-output-projection ablation of the frozen top head against a
  same-layer low-signal head. The registered causal endpoint is loss of correct-
  versus-swapped first-cause-token logit margin. Even a pass supports only that
  the head contributes to this binding decision, not a complete circuit.
  Protocol: `VARIABLE_BINDING_PROBE_PROTOCOL_20260805.md`.
- Non-MPS validation passes: six new probe unit tests, actual cached-tokenizer
  span validation for all 32 cases (181–188 tokens), Python compilation, dry
  manifest generation, and whitespace checks. No Gemma weights have been loaded
  and no MPS computation has run; live execution awaits user confirmation that
  MPS is free.
- User reported 42% remaining and confirmed Comfy rendering was complete, so
  the registered 32-case Gemma 3 1B MPS probe ran without training or prompt
  changes. Cause-row identity was chance at the embedding state (0.500), rose
  to 1.000 at hidden-state indices 4–17, and declined to 0.84375 at the final
  state. The peak survived 100 within-pair, max-layer-corrected permutations
  (`p=0.00990099`; maximum null peak 0.84375).
- Even-index discovery selected zero-based block 17, head 0: mean attention to
  correct cause evidence minus control evidence was 0.5836, versus 0.00448 for
  the fixed same-layer head-1 control. On 18 untouched odd-index confirmation
  cases, ablating head 0 reduced correct-versus-swapped cause-token margin from
  10.376 to 7.624 (drop 2.752); control ablation raised it to 11.967. The
  preregistered causal gate passed. Defensible claim: this head contributes to
  the first cause-token binding decision in this adapter; it is not a complete
  variable-binding circuit.
- Added an explicitly post-registration comparison-role audit because the first
  output could not explain later L1 failures. Comparison-row decoding peaked at
  1.000 at hidden-state index 11 and declined to 0.8125 finally; its separate
  max-layer-corrected permutation result was also `p=0.00990099`. The same block
  17/head 0 was selected (comparison-minus-cause attention 0.3389). Confirmation
  ablation reduced comparison-token margin from 14.091 to 8.762 (drop 5.330),
  while head-1 control ablation raised it to 16.021.
- Mechanistic failure decomposition: all four genuine role-binding failures
  were the frozen `gold` cause incorrectly repeated as the `gold` comparison
  instead of `silver`; all four had negative teacher-forced comparison margins,
  while zero successful cases did. The other three of seven L1 failures had
  strong correct cause/comparison margins and failed by inserting numeric atoms
  or corrupting the effect field. Therefore yesterday's 5.47-point reliability
  tax combines four localized comparison-binding failures with three distinct
  serialization/grammar failures. Summary artifact:
  `outputs/variable_binding_probe_summary_20260805.json`.
- Robustness/validation: peak leave-pair-out decoding remained 1.000 for both
  cause and comparison across ridge penalties 0.01, 0.1, 1, 10, and 100. Final
  cause accuracy remained 0.844–0.875 and final comparison 0.781–0.813, so the
  middle-layer peak and late decline are not artifacts of the registered ridge
  value. The full repository suite passes all 184 tests, summary JSON parses,
  Python compilation passes, and diff whitespace checks pass.
- Mechanism-guided repair: added `labeled_causal_ir_constrained`, which keeps
  the frozen L1 adapter and prompt but restricts beam decoding to a symmetric
  code-generated set of valid L1 records. Candidate causes/comparisons include
  every distinct observed-feature assignment, all three canonical effects, and
  every observed or zero delay; the candidate space does not identify the
  correct answer. Registered seed 307, 128 cases, paired greedy/constrained
  accuracy, repairs versus regressions, visible tokens, and runtime before the
  seed was observed (`L1_FORMAL_REPAIR_PROTOCOL_20260805.md`).
- Seed-308 eight-case smoke initially appeared to fall from 8/8 greedy to 6/8
  constrained, but both rejected outputs were semantically exact negative-
  pressure records. This exposed a pre-existing positive-only verifier bug:
  `validate_bound_hypothesis` always required the largest-valued feature and
  `pressure_increase`, incorrectly rejecting the largest absolute negative
  departure and `pressure_decrease`. Fixed the verifier generically and added a
  negative-effect regression test. The preserved v2 smoke then scored 8/8 in
  both conditions; seed 307 remained untouched throughout diagnosis.
- Frozen seed-307 repair audit: ordinary L1 scored 118/128 = 92.1875%; symmetric
  constrained L1 scored 128/128 = 100%. Paired outcomes were 118 both correct,
  ten constrained-only, zero greedy-only, and zero both wrong: +7.8125 points,
  exact two-sided McNemar `p=0.001953125`. The repair removed four replicated
  `gold`-for-`silver` comparison bindings plus six numeric/effect serialization
  errors without a regression.
- Repair cost boundary: mean visible tokens were essentially unchanged
  (184.633 greedy versus 184.531 constrained; -0.055%), while elapsed MPS time
  rose from 162.865 to 262.101 seconds (+60.93%). The mechanism-guided formal
  constraint therefore repairs reliability by spending additional internal
  beam-search compute; it does not yet solve the original inference-efficiency
  goal. A matched seed-307 JSON control and independent replication remain
  future work. Summary: `outputs/l1_constraint_repair_summary_20260805.json`.
- Efficiency repair was registered before observation in
  `L1_MASKED_GREEDY_PROTOCOL_20260805.md`: replace four-beam search with a
  one-path finite-state mask over the same symmetric candidate set. The model
  follows its ordinary greedy choice unless that token would make every valid
  L1 continuation impossible. The mask enforces grammar and distinct observed
  cause/comparison roles but does not identify the correct causal assignment.
- Seed-310 smoke scored 8/8 under both ordinary and masked greedy, with 10.472
  versus 10.019 seconds elapsed. The untouched seed-309 128-case audit then
  scored 123/128 = 96.09375% for ordinary greedy and 128/128 = 100% for masked
  greedy. Paired outcomes were 123 both correct, five masked-only, zero
  greedy-only, and zero both wrong: +3.90625 points with no regression. The
  five repairs comprised four repeated `gold` cause-as-comparison errors and
  one invalid inserted numeric atom. Exact two-sided McNemar `p=0.0625`, so
  this single audit is encouraging but does not by itself establish a
  population-level accuracy improvement.
- Masked-greedy cost was nearly neutral: mean visible tokens changed from
  184.344 to 184.328 (-0.0085%), and elapsed MPS time changed from 160.323 to
  164.412 seconds (+2.55%). This repairs the observed L1 failures without the
  earlier beam method's +60.93% latency, while retaining the original compact
  representation. Claim boundary: independent replication and a matched JSON
  audit are still needed before claiming general JSON-level reliability or an
  end-to-end inference-cost win. Summary:
  `outputs/l1_masked_greedy_summary_20260805.json`.
- Post-repair validation passes all 186 repository tests, Python compilation,
  both final JSON parse checks, and `git diff --check`. The MPS audit exited
  normally; no model process remains running.
- 6 August starting checkpoint: user reported 32% remaining with a 16% stopping
  target and requested a confirmation sweep suitable for publication and a
  LessWrong mechanistic-interpretability post. Registered
  `L1_PUBLICATION_CONFIRMATION_PROTOCOL_20260806.md` before new inference. To
  conserve budget and MPS time, the confirmation omits a redundant ordinary-L1
  pass: frozen JSON will run on seed 309 for a case-matched accuracy/token
  comparison, and masked L1 will run on untouched seed 311 for independent
  128-case replication. The public wording is explicitly bounded to zero
  observed defects, not universal zero-defect performance.
- Publication confirmation passed both registered audits without modification.
  Frozen JSON scored 128/128 on the same seed-309 cases where masked L1 had
  scored 128/128. Mean visible tokens were 282.328 for JSON and 184.328 for
  masked L1, confirming a matched **34.711% reduction** at equal point
  accuracy. Sequential elapsed times were 323.345 and 164.412 seconds,
  respectively; the observed 49.15% L1 speed advantage is descriptive because
  machine load was not experimentally controlled.
- Untouched seed-311 masked-L1 replication scored 128/128 in 163.012 seconds,
  with 184.422 mean visible tokens. Across the independent seed-309 and
  seed-311 confirmation sets, the frozen repaired system therefore produced
  256/256 exact matches with zero observed failures. Updated the qualified
  result report and wrote
  `outputs/l1_publication_confirmation_summary_20260806.json`. Public wording
  must say “zero observed defects across 256 cases,” not universal “100%
  zero-defect precision.”
- Confirmation validation passes all 186 repository tests, Python compilation,
  JSON parsing, and whitespace checks. Updated the existing research website's
  Tiny Scientist section with the mechanism-guided resolution, matched token
  result, independent replication, and explicit claim boundary. The rendered
  site build and both content tests passed; Sites version 17 was saved and
  deployed successfully to the existing public Tiny Consciousness Lab URL.
- GitHub publication completed on `codex/tiny-scientist-neurosymbolic`: commit
  `4d4599c` adds the mechanistic probe, causal intervention, finite-state repair,
  preregistered protocols, full frozen audit artifacts, tests, and qualified
  report, while excluding adapter weights and unrelated dirty-worktree work.
  The existing draft PR #16 was pushed and retitled “Resolve compact causal
  binding tradeoff”; its description now reports the root cause, 256/256
  confirmation, matched 34.711% token reduction, validation, and claim limits.
