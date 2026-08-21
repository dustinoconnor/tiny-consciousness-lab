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
- Post-publication discoverability review: the two papers directly framing the
  Tiny Scientist/repair sequence were Tom Zahavy's 2026 position paper `LLMs
  can't jump` (abductive hypothesis generation from grounded simulation) and
  Wu, Geiger, and Millière's ICML 2025 `How Do Transformers Learn Variable
  Binding in Symbolic Programs?` (residual-stream probes plus causal attention
  interventions). LessWrong's March 2026 policy allows human-written text whose
  facts, arguments, experiments, or code were developed with LLM assistance,
  but treats first-time users as near-zero-LLM for submitted prose; any post
  should therefore be written in the user's own words and limited to claims the
  user can explain and defend.
- GitHub metadata audit found 14 of the 20 permitted repository topics already
  populated, including `mechanistic-interpretability`. Acronyms alone are weak
  discovery terms. A future explicit metadata update should preferentially add
  expanded topics such as `model-predictive-control`, `embodied-ai`,
  `neuro-symbolic-ai`, `causal-reasoning`, `global-workspace-theory`, and
  `adaptive-resonance-theory`; no public topic mutation was made during this
  read-only review.
- Audited the user's first human-written LessWrong draft for factual accuracy
  without rewriting it. The compact-DSL result section is substantially
  accurate, but publication corrections are required: distinguish the DSL
  synthetic-table audit from Unity embodiment; describe block 17/head 0 as a
  causal contributor rather than the primary or complete binding circuit; and
  report the negative comparison-margin diagnostic against the 25 successful
  cases in the selected 32-case probe cohort, not all 121 successes in the
  original 128-case L1 audit. Recommended adding the held-out case definition,
  frozen/new-seed chronology, exact ablation margins, and the internal-
  replication boundary.
- Fixed the public research site's broken hash deep links. A direct load of
  `#tiny-scientist` preserved the URL hash but remained at the top because the
  Vinext client tree completed after the browser's native anchor-navigation
  pass. Added a client hash-scroll manager that re-aligns the requested target
  after hydration, fonts, and pending images settle, and on later hash changes.
  Removed the duplicate section `scroll-margin` offset so the global sticky-
  header padding is applied only once. The full site build and two rendered-
  content tests pass. Browser verification on both local production output and
  the deployed public URL placed “Neuro-symbolic Tiny Scientist in Unity” at
  92 px from the viewport top (68 px sticky header), rather than at the page
  top or Access Consciousness. Sites version 19 deployed successfully from
  source commit `025bf08`.
- 8 August starting checkpoint: user reported a fresh 100% daily allowance;
  the previous day's unused 19% did not carry over. No inference or new
  experiment has run yet. After reviewing the Tiny Scientist/compact-DSL
  evidence, the highest-value next question is whether the successful repair
  comes from generic syntax enforcement or from the mechanism-motivated
  distinct-role constraint. Proposed a frozen four-condition comparison:
  ordinary L1 greedy, syntax-only finite-state masking that still permits
  cause/comparison repetition, the existing syntax-plus-distinct-role masked
  greedy decoder, and matched JSON. This would establish the correct generic
  constrained-decoding baseline before making a broader novelty claim.
- Implemented the preregistered syntax-only L1 control without changing the
  frozen Gemma 3 1B adapter, prompt, tokenizer, validator, semantic scorer, or
  existing role-bound decoder. The new one-path finite-state mask permits all
  grammatical observed-feature assignments, including repeated cause and
  comparison roles. Added focused candidate-space and paired-comparison tests;
  the full non-MPS suite now passes 188 tests.
- Seed-312 smoke separated the controls as intended: ordinary and syntax-only
  both scored 7/8 and emitted the same invalid `gold/gold` role assignment;
  the role-bound decoder scored 8/8 by selecting `gold/silver`.
- Matched seed-309 diagnostic: ordinary greedy scored 123/128, syntax-only
  scored 124/128, and syntax-plus-distinct-role scored 128/128. Syntax masking
  repaired the one numeric insertion but left all four repeated-role failures;
  the semantic constraint repaired those four with no regression.
- Untouched seed-313 confirmation reproduced the decomposition: ordinary
  greedy scored 121/128, syntax-only 124/128, and role-bound 128/128. Generic
  syntax repaired three malformed records; the distinct-role rule repaired the
  four remaining `gold/gold` bindings with zero regressions. The confirmatory
  role-over-syntax comparison is four versus zero discordant cases (exact
  two-sided McNemar `p=0.125`), so the single new audit is directionally exact
  but underpowered at 0.05. Across the diagnostic and confirmation stages the
  registered pattern is eight role-specific repairs and zero regressions
  (`p=0.0078125` descriptively), with the explicit boundary that the pooled
  statistic includes the post-hoc diagnostic set. Full interpretation:
  `L1_SYNTAX_CONTROL_RESULT_20260808.md`.
- Published the syntax-only control implementation, frozen artifacts, tests,
  preregistered protocol, and bounded result report to GitHub on
  `codex/tiny-scientist-neurosymbolic` in commit `99c333f`; updated draft PR
  #16 to foreground the grammar-versus-binding decomposition. Expanded the
  repository's discoverability metadata to include `causal-reasoning`,
  `constrained-decoding`, `embodied-ai`, `gemma`,
  `model-predictive-control`, and `neuro-symbolic-ai` while preserving the
  existing research topics.
- Updated the public Tiny Scientist section with the untouched seed-313
  three-way control table (ordinary 121/128, syntax-only 124/128, and
  syntax-plus-distinct-role 128/128), the underpowered confirmatory
  `p=0.125` boundary, and a direct link to the complete result report. The
  website build and both rendered-content tests passed; Sites version 20 was
  saved from source commit `f939310` and deployed successfully to the existing
  Tiny Consciousness Lab public URL.
- 9 August AST starting point: reviewed the existing adaptive GNW governor and
  determined that its active-broadcast, challenger, persistence, and reward
  fields already constitute substantial explicit self-state. Preregistered a
  stricter test in `ATTENTION_SCHEMA_PROTOCOL_20260809.md`: a 60-parameter
  softmax predictor forecasts the router's next local focus from current and
  previous bid distributions plus broadcast state, with no context labels,
  optimal-specialist labels, outcomes, or future observations. Frozen
  conditions were no-schema adaptive GNW, a correctly bound schema at fixed
  0.30 influence, a role-shuffled parameter-matched control, and a disconnected
  post-training lesion.
- The registered 12-seed AST audit evaluated 17,280 held-out steps per
  condition and did not pass its overall decision rule. Correct schema access
  beat shuffled schema in utility by `+0.07664` (95% interval
  `[+0.02562, +0.12766]`), improved transient-distractor routing over adaptive
  GNW by `+0.16628` (`[+0.06743, +0.26513]`), and reduced unnecessary handoffs
  by `0.03056`. However, it reduced genuine boundary routing by `0.18542` and
  utility by `0.04414` versus adaptive GNW; both intervals excluded zero. The
  lesion exactly matched the baseline. Interpretation: the learned schema
  carried correctly bound temporal information but behaved as an always-on
  persistence prior, duplicating the existing governor and delaying real rule
  changes. Full bounded result: `ATTENTION_SCHEMA_RESULT_20260809.md`; metrics:
  `outputs/attention_schema_registered_20260809.json`. No Unity integration is
  warranted unless a separately preregistered change-point-aware schema first
  passes held-out synthetic evaluation.
- Replaced the delayed red-mushroom `metabolic_pressure` intervention with a
  passive `causal_probe_signal`. Red still schedules a `+0.34` observable event
  after ten seconds and blue remains the negative control, but the signal no
  longer enters body prediction error, crosstalk, complexity, workspace
  selection, MPC scoring, routing, or navigation. Retired the optional
  `pressure_avoidance` Tiny Scientist control path; committed rules are now
  structurally read-only. New recordings expose explicit causal-probe fields
  plus deprecated read-only pressure aliases for historical analyzer
  compatibility, and `tiny_scientist.py` prefers the new signal. A paired
  red/blue functional-state test confirms identical prediction error,
  crosstalk, complexity, hunger, dopamine, action, and workspace state despite
  the red-only telemetry event. All 193 tests pass after the conversion.
- Implemented the preregistered multiple-hypothesis predictive-workspace assay
  in `pgnw_hypothesis_selection_lab.py`. The formal pool contains red-cause,
  blue-cause, either-pickup, spontaneous, and no-cause models; available tests
  are red observation, blue observation, and a no-pickup wait. Hidden worlds
  are counterbalanced, all policies receive matched per-action outcome draws,
  and no hypothesis receives Unity, motor, physiology, or navigation authority.
  Comparisons include epistemic-only, pragmatic-only, confirmation seeking,
  random selection, and a scrambled broadcast binding.
- The initial 20-seed PGNW audit passed three causal comparisons and the
  pragmatic-value criterion but left PGNW-over-random and epistemic
  noninferiority intervals unresolved. Without changing any model, weight,
  budget, metric, or criterion, registered and ran a disjoint 100-seed
  confirmation (500 counterbalanced worlds per condition). PGNW identified the
  true model in 90.8% of worlds with mean true posterior `0.8755`; it beat
  pragmatic-only by `+0.35608`, random by `+0.05279`, confirmation seeking by
  `+0.21800`, and scrambled broadcast by `+0.32356`, with all 95% intervals
  above zero. It also gained `+0.28576` pragmatic value over epistemic-only.
  However, its true posterior was `0.02976` below epistemic-only with interval
  `[-0.04973, -0.00978]`, failing the frozen `-0.03` noninferiority boundary.
  Four of five criteria passed; overall result remains a bounded failure and a
  measured epistemic/pragmatic Pareto tradeoff. Full report:
  `PGNW_HYPOTHESIS_SELECTION_RESULT_20260809.md`.
- 10 August bounded-authority implementation: added
  `EmbodiedPGNWExperimentPlanner`, which carries the five-model posterior into
  the Unity loop and broadcasts `observe_red_pickup`, `observe_blue_pickup`, or
  `wait_no_pickup`. It starts a ten-second passive-probe observation after the
  experiment actually encountered, discards windows contaminated by additional
  pickups, logs requested/executed mismatches, updates the posterior only from
  uncontaminated outcomes, and selects the next experiment. Passive mode has
  structurally zero guidance.
- Added `--tiny-scientist-experiment-control {disabled,passive,bounded}`. In
  bounded mode a requested visible color supplies at most `0.03` alignment
  preference to an already-engaged ambiguous MPC decision; it cannot activate
  MPC. The preference is disabled during fallback, physical wedges, critical
  hunger, guided AIR, guided resource memory, no target visibility, or any
  collision-masked direction. Telemetry records the full posterior, request,
  phase, outcome, completed/discarded trials, effective guidance, decisions,
  and changed MPC selections. `ACTIVE_PGNW_UNITY_PROTOCOL_20260810.md`
  preregisters feasibility and the boundary that this is experiment-selection
  authority rather than direct motor or general production-rule authority.
  All 206 repository tests, Python compilation, and whitespace validation pass;
  live Unity feasibility remains pending a user-run smoke.
- Completed and analyzed the seed-141 bounded Active-PGNW Unity smoke
  (`outputs/unity_shadow/pgnw_active_science_smoke_20260810.jsonl`; 1,193.508
  recorded seconds, 5,674 rows). The preregistered feasibility criterion passed:
  21 experiments began, 11 uncontaminated experiments completed and caused 11
  posterior updates, while ten contaminated delay windows were discarded.
  Completed evidence separated perfectly into four red observations with a
  delayed probe rise and seven blue observations with no rise. The posterior
  moved from a uniform five-model prior to `red_causes_probe = 0.999134`, an
  entropy reduction of 2.31179 bits and final red-versus-all odds of 1,153.99:1.
- The authority audit found 96 effective bounded-guidance decisions, a maximum
  effective weight of `0.02961`, and one changed MPC selection while both
  colors were visible. Thirteen of 21 pickups mismatched the current request and
  were correctly analyzed by the experiment actually encountered. This means
  the online science loop worked, but most evidence remained opportunistic and
  the run does not establish a navigation benefit. Safety remained intact: zero
  survival, trap, or escape failures and zero critical-hunger exposure. One
  stuck event occurred while PGNW guidance was zero. Full bounded result and
  matched passive-versus-bounded next-test boundary:
  `ACTIVE_PGNW_UNITY_RESULT_20260810.md`.
- Preregistered the next comparison in
  `PGNW_ACTIVE_PASSIVE_PROTOCOL_20260810.md`. A different active seed alone
  cannot distinguish epistemic guidance from passive wandering, so the next
  run is the matched seed-141 passive half of the existing bounded pilot. The
  frozen primary metric is elapsed time to `red_causes_probe >= 0.95`, with
  posterior-entropy area, clean-experiment rate, request compliance, and safety
  as secondary measures. One pair remains diagnostic; a learning-speed claim
  requires additional counterbalanced passive/bounded seed pairs.
- Completed the matched seed-141 passive half and audited it against the frozen
  bounded run. The diagnostic pair favors bounded PGNW: bounded crossed
  `P(red_causes_probe) >= 0.95` at 863.97 seconds, whereas passive remained
  below threshold through 1,194.35 seconds and ended at 0.932651. This is a
  right-censored bounded lead greater than 330.38 seconds. Bounded also had
  34.02% lower mean posterior entropy and 22.31% more clean experiments per
  hour, ending with four clean red-positive and seven blue-negative trials;
  passive obtained one red-positive and eight blue-negative trials.
- The result remains a positive pilot rather than a causal claim. Passive found
  the first clean red result 18.07 seconds earlier, bounded changed only one MPC
  selection, and independent Unity trajectories diverged strongly (32 versus
  28 pickups and one versus nine stuck events). Both had zero survival failures
  and zero critical-hunger exposure. Real-time trajectory variation could
  explain much of the gap, so the next step is counterbalanced passive/bounded
  seed pairs with unchanged `0.03` authority. Full result:
  `PGNW_ACTIVE_PASSIVE_PILOT_RESULT_20260810.md`.
- Completed the passive-first half of counterbalanced seed 142. Passive crossed
  `P(red_causes_probe) >= 0.95` at 586.49 seconds, completed 17 clean
  experiments from 24 starts (nine red-positive and eight blue-negative), and
  ended at posterior 0.999968 with mean entropy 0.89205 bits. It collected 35
  mushrooms including ten red, logged seven discarded windows, 12 protocol
  mismatches, five stuck events, 17 seconds of critical hunger, and zero
  survival failures. Guidance and action influence were structurally zero.
  This strong result relative to passive seed 141 demonstrates substantial
  between-run/seed variability; no active-versus-passive conclusion is drawn
  until the bounded seed-142 half is complete.
- Completed bounded seed 142 and closed the second active/passive pair. Bounded
  again inferred the correct rule, crossing posterior 0.95 at 867.42 seconds
  and ending at 0.998370 with zero survival failures. It completed ten clean
  experiments (three red-positive, seven blue-negative), compared with
  passive's 17, and was 280.93 seconds slower than passive on the primary
  threshold metric. Bounded also had higher mean entropy, fewer pickups, more
  stuck events, and more critical-hunger exposure in this pair.
- Across seeds 141 and 142, one pair favors each condition. Bounded threshold
  times were remarkably consistent at 863.97 and 867.42 seconds, while passive
  ranged from 586.49 seconds to not reaching threshold by 1,194.35 seconds.
  With the registered 1,200-second censoring horizon, the descriptive
  restricted mean favors bounded by only 27.55 seconds; two pairs do not support
  an inferential speed claim. Bounded changed one MPC selection in each run,
  and the seed-142 change occurred at 1,036.66 seconds, after learning crossed
  threshold. The replicated result is safe online causal discovery, not yet
  active acceleration. Full comparison:
  `PGNW_ACTIVE_PASSIVE_TWO_PAIR_RESULT_20260810.md`.
- Pre-seed-143 storage audit: the data volume reports 12 GiB available at 97%
  capacity. `outputs/` occupies 2.8 GiB and `outputs/unity_shadow/` occupies
  2.7 GiB across 155 JSONL recordings. The four completed PGNW logs total about
  235 MiB; another bounded/passive pair should add roughly 118 MiB. Chose a
  staged design rather than committing immediately to ten pairs: complete the
  counterbalanced seed-143 pair, reassess after three pairs, and reserve a
  larger sample for any formal variance or tail-reliability claim.
- Storage cleanup authorized by the user: permanently removed 59 completed
  non-PGNW raw JSONL recordings from `outputs/unity_shadow`, freeing 2.08 GiB.
  Preserved the four completed PGNW comparison recordings and the in-progress
  seed-143 bounded recording, along with derived metrics and result reports.
  `outputs/` fell from 2.8 GiB to 731 MiB and available disk space increased
  from 12 GiB to 14 GiB. The raw deleted recordings are not recoverable from
  the project directory.
- Added resumable unattended PGNW batching in
  `run_pgnw_active_passive_batch.py`. It counterbalances odd seeds
  bounded-first and even seeds passive-first, skips only recordings verified as
  complete, refuses to overwrite partial data, checks for at least 5 GiB free,
  applies a per-run timeout, records a batch manifest, and aborts on any Unity,
  subprocess, or recording-validation failure. The seed-143 bounded recording
  is complete (1,194.69 seconds, posterior 0.999570, 12 clean experiments), so
  a `143-145` batch will skip it and execute five remaining runs.
- Back-to-back Python processes alone would not reproduce the zero-position,
  zero-pickup Unity starts seen in all valid runs. Added an acknowledged
  `experiment_reset` command to both the repository Unity bridge and the actual
  `/Users/dustinoconnor/My project` scripts. Before every run it restores the
  terrain spawn, clears movement and pickup telemetry, and makes all mushrooms
  available. The batch waits for Unity telemetry acknowledging zero red and
  total pickups before launching the condition. Added focused batch tests; all
  209 repository tests, Python compilation, and whitespace validation pass.
- The unattended seed-143-through-145 batch completed all five scheduled runs
  with zero subprocess errors and approximately 1,200 recorded seconds per
  condition. Analysis uncovered a reset-height mismatch: automated starts were
  `(0, -2.856, 0)` while manual Unity restarts were `(0, -0.008, 0)`. Seed 143
  mixed those start methods and was excluded from the paired aggregate. Seeds
  144 and 145 remained internally matched because both halves used the same
  automated reset. The one seed-143 bounded survival failure was a persistent
  physics wedge while effective PGNW guidance was zero.
- Across the four valid pairs (141, 142, 144, 145), bounded and passive each won
  two threshold-time comparisons. Bounded crossed posterior 0.95 in 3/4 runs
  versus passive 2/4, but the 1,200-second-capped paired mean advantage was only
  7.38 seconds. Bounded had lower descriptive threshold-time dispersion and
  mean posterior entropy, but a lower clean-experiment rate and only two changed
  MPC choices before convergence. This does not replicate raw acceleration or
  establish a worst-case guarantee; it leaves variance reduction as an
  underpowered hypothesis. Full interim result:
  `PGNW_ACTIVE_PASSIVE_FOUR_PAIR_INTERIM_RESULT_20260810.md`.
- Corrected unattended resets to use the exact scene-authored terrain spawn
  rather than resampling terrain height. The batch runner now also rejects a
  reset acknowledgement unless position is within 0.25 Unity units of
  `(0, -0.008, 0)` in addition to zero pickup counters, preventing silent
  start-state mismatch in future data.
- Added `--max-new-runs` to the resumable batch runner for a one-run validation
  checkpoint. The corrected seed-143-through-145 replication can execute one
  bounded seed-143 run, stop for inspection, then resume the identical schedule
  overnight while skipping the verified completed recording.
- Completed the six-run seed-143-through-145 floor-start replication under the
  resumable batch. All subprocesses returned zero, all recordings reached about
  1,200 seconds, and every condition began with zero pickup counters at the same
  recorded `(0, -2.856, 0)` floor position. This forms an internally matched
  replication block, separate from the earlier command-first starts near
  `y = 0`.
- The active reliability hypothesis did not replicate. Bounded crossed the
  0.95 posterior threshold in 1/3 runs and passive in 2/3; passive won two
  paired comparisons, bounded one. The 1,200-second-capped paired mean was
  67.31 seconds slower for bounded, whose threshold times were also more
  variable. Both conditions had zero survival failures. Across the three
  bounded runs, guidance changed only two MPC selections, both in the seed-144
  failure; the large seed-143 bounded win occurred with zero changed actions
  and cannot be attributed to active guidance.
- Localized the failure to diagnostic evidence acquisition rather than causal
  updating. All three conditions with at least two uncontaminated red
  observations reached the correct threshold; all three conditions with zero
  clean red observations failed and retained `no_tested_cause` as MAP. More
  unchanged runs are not warranted. The next registered manipulation must make
  experiment authority behaviorally meaningful before the diagnostic event
  while freezing posterior inference and contamination rejection. Full result:
  `PGNW_FLOOR_START_REPLICATION_RESULT_20260811.md`.
- Hardened batch provenance after the replication: reset acknowledgement near
  `y = 0` is no longer treated as proof of recording start, because model load
  time lets the body settle before telemetry begins. The runner now separately
  validates the first recorded position and defaults to the explicitly labeled
  floor-start protocol `(0, -2.856, 0)`. A requested airborne-start protocol
  will fail rather than being silently mixed with floor-start data.
- Preregistered a ten-minute stronger-authority manipulation check in
  `PGNW_STRONGER_AUTHORITY_SMOKE_PROTOCOL_20260811.md`. The causal updater and
  safety architecture remain frozen. The new explicit configuration raises the
  PGNW score weight from 0.03 to 0.12 and the eligible ambiguous-MPC margin from
  0.08 to 0.25, while retaining the existing persistent experiment request and
  every collision, fallback, stuck, hunger, AIR, and resource-memory gate. The
  smoke must produce at least five changed MPC selections before convergence
  and a requested-color pickup following guidance; posterior success alone does
  not pass the manipulation check.
- Completed the seed-146 stronger-weight manipulation smoke. It failed the
  registered behavioral criterion: 13 eligible guidance decisions and maximum
  effective weight 0.09299 produced zero changed MPC selections. Two red
  pickups were opportunistic rather than guidance-caused, and both ten-second
  red windows were contaminated by an additional blue pickup. The updater
  correctly discarded them; six clean blue-negative trials left
  `no_tested_cause` as MAP and red-cause posterior 0.44923.
- Safety remained intact with zero stuck events, zero critical-hunger exposure,
  and zero survival failures. No overnight comparison is warranted for this
  configuration. The next manipulation must replace larger additive weighting
  with a preregistered constrained commitment: retain every safety veto, admit
  only actions within a bounded score regret of ordinary MPC, and choose the
  safest admissible action most aligned to the requested target. Full result:
  `PGNW_STRONGER_AUTHORITY_SMOKE_RESULT_20260811.md`.
- Implemented the preregistered constrained-commitment controller for a second
  short manipulation check. New `committed` PGNW mode preserves the persistent
  experiment request but, only while the requested color is visible, chooses
  target alignment from collision-safe actions no more than 0.18 MPC-score
  units below the ordinary optimum. Authority lasts at most three seconds with
  a two-second cooldown. Fallback, stuck, critical hunger, guided AIR, guided
  resource memory, and the requirement that MPC already be engaged remain
  absolute vetoes. Telemetry records commitment starts, expirations, remaining
  duration, regret bound, decisions, and action influence. Protocol:
  `PGNW_CONSTRAINED_COMMITMENT_SMOKE_PROTOCOL_20260811.md`.
- Completed the seed-147 constrained-commitment smoke. It was the first partial
  behavioral success: 27 committed decisions changed two MPC selections, and a
  requested red pickup followed the second change by 0.21 seconds. Safety
  remained intact with zero critical-hunger exposure and zero survival
  failures. However, the registered requirement was five changes, so the smoke
  did not pass.
- A blue pickup 1.50 seconds after the guidance-linked red pickup contaminated
  the ten-second observation; an earlier red pickup was similarly followed by
  blue after 1.06 seconds. Seven trials were discarded, five clean blue-only
  trials completed, and zero clean red evidence left `no_tested_cause` as MAP
  with red posterior 0.45412. The next bottleneck is post-intervention evidence
  isolation. Freeze target acquisition and add a safety-gated ten-second phase
  that suppresses optional food seeking and prefers movement away from visible
  mushrooms. Do not schedule an overnight comparison until a short smoke
  produces both a request-linked pickup and a clean red completion. Full
  result: `PGNW_CONSTRAINED_COMMITMENT_SMOKE_RESULT_20260811.md`.
- 12 August observation-isolation implementation: retained the adjacent red and
  blue spawn mushrooms as a deliberate stress test. In committed mode, any
  experimental pickup now opens the existing ten-second observation delay and
  activates a safety-gated isolation phase. Optional food seeking is suppressed
  and its commitment cleared; visible red/blue directions form an
  inverse-distance-weighted repulsion vector; constrained MPC chooses the most
  food-averse collision-safe action within the frozen 0.18 regret bound.
  Fallback, stuck recovery, critical hunger, guided AIR, guided resource
  memory, collision masking, and the requirement that MPC already be active
  remain absolute. Telemetry now separates isolation decisions, action changes,
  food-suppression frames, and intervening pickups. Preregistered smoke:
  `PGNW_OBSERVATION_ISOLATION_PROTOCOL_20260812.md`.
- Completed the seed-148 isolation smoke. Isolation clearly reached behavior:
  63 isolation decisions suppressed food seeking for 63 frames and changed 46
  MPC selections, with zero survival failures. It nevertheless failed the
  evidence criterion. The single red pickup at 2.96 seconds was followed by the
  adjacent blue at 5.11 seconds, contaminating the trial after 2.14 seconds.
  Overall, 4/6 trials were discarded, only two blue-negative trials completed,
  and the final red posterior was 0.43091. Six stuck events and 63 seconds of
  critical hunger also rule out calling the behavior harmless beyond the
  absence of survival failure.
- Preregistered and implemented a narrow immediate-retreat refinement in
  `PGNW_IMMEDIATE_RETREAT_PROTOCOL_20260812.md`. For only the first two seconds
  after pickup, food repulsion may choose collision-safe MPC actions within a
  0.35 score-regret bound to overcome approach momentum; the remaining delay
  returns to the frozen 0.18 bound. Isolation now stops once a trial is already
  confounded. Target acquisition, inference, evidence rejection, and every
  safety veto remain unchanged.
- Completed the seed-149 immediate-retreat smoke. It passed the registered
  system-level evidence criterion: one requested red pickup completed a clean
  ten-second window and produced the delayed probe rise; six clean blue controls
  followed, leaving `red_causes_probe` as MAP at 0.93103. Red contamination
  improved from 2/2 in seed 147 to 1/2. Thirty retreat decisions changed 30 MPC
  selections; total isolation changed 54 selections; safety ended with zero
  critical-hunger exposure and zero survival failures.
- The adjacent spawn pair itself remained unsolved: blue followed its red after
  4.26 seconds, improved from 2.14 seconds but below ten. The clean red occurred
  later and did not itself require visible repulsion, so causation is not
  assigned to retreat from this single run. Full qualified result:
  `PGNW_IMMEDIATE_RETREAT_SMOKE_RESULT_20260812.md`.
- Preregistered a three-pair committed-versus-passive batch on seeds 150-152 in
  `PGNW_COMMITTED_PASSIVE_BATCH_PROTOCOL_20260812.md`. Extended the resumable
  batch runner with `--active-mode committed`, preserving alternating order,
  reset validation, partial-file refusal, disk checks, and fail-fast behavior.
  The six-run batch will measure threshold time, clean-red acquisition,
  contamination, verified pre-evidence influence, entropy, and safety before
  deciding whether a larger sample is warranted.
- Completed and analyzed the six-run seed-150-152 committed-versus-passive
  batch. All runs began from the same validated floor state with cleared Unity
  counters and completed the full 20-minute bound. Committed guidance reached
  posterior `P(red_causes_probe) >= 0.95` in 3/3 runs at 930.8, 556.9, and
  411.5 seconds. Passive reached it in only 1/3 runs, at 1,179.8 seconds.
  Right-censoring misses at 1,200 seconds gives means of 633.1 seconds committed
  versus 1,193.3 seconds passive, a bounded 560.2-second (46.9%) reduction.
- The behavioral chain preceded inference: committed action changes before the
  first clean red completion numbered 86, 147, and 36 across the three seeds.
  Committed mode produced 8 clean red observations versus 2 passive, completed
  53 total observations versus 31, reduced the discard fraction from 49.2% to
  27.4%, and reduced mean time-averaged posterior entropy from 1.140 to 0.718
  bits. Final mean red-cause posterior was 0.996598 versus 0.617184.
- Severe safety remained intact: zero critical-hunger seconds and zero survival
  failures in both conditions. Committed mode incurred 21 stuck events versus
  17 passive, and target-protocol mismatch remained high in both conditions, so
  the intervention is effective but not cost-free and does not guarantee target
  compliance. With only three pairs, this is a successful manipulation check,
  not a population-level significance claim. Full qualified result:
  `PGNW_COMMITTED_PASSIVE_BATCH_RESULT_20260812.md`.
- Began the dynamic Tiny Scientist loop with an explicit discovery/verification
  boundary. The architecture will preserve Bayesian updating as the evidence
  accounting layer while removing the frozen five-hypothesis ontology: Gemma
  proposes a compact L1 causal record from discovery telemetry, the existing
  semantic verifier and formal layer compile it into an executable likelihood
  model at zero production-rule authority, and PGNW may select future tests.
  Evidence at or before admission is ineligible to verify the proposal.
- Added `dynamic_hypothesis_pool.py` and focused tests. The first bounded
  compiler accepts a grounded red/blue directional L1 rule, derives its action-
  conditional probe likelihoods, records the exact source DSL and admission
  cutoff, rejects duplicates, and supports variable-size posterior and expected-
  information-gain calculation. This is an offline admission substrate, not yet
  live Unity integration or a claim of open-ended ontology formation.
- Exercised the admission path with the actual frozen Gemma 3 1B L1 adapter and
  mechanism-guided masked-greedy decoder. From seed-149 discovery evidence it
  generated `L1 c red k blue e + t 10.0 q 0.5` in 2.538 seconds using 190 input
  and 20 output tokens. The verifier accepted it and the compiler admitted
  `dsl:red_causes_probe_rise@10s` at prior 0.20 against only spontaneous and
  no-tested-cause baselines; it began with zero held-out updates and no rule or
  direct motor authority.
- In chronological post-development replay, all seed-149 discovery observations
  were excluded from verification. Fifteen later clean seed-150 observations
  raised the dynamically admitted candidate above 0.95 on the second clean red
  result and to 0.999877 finally. Nine earlier blue-negative controls alone did
  not verify it. This closes formulation, formal admission, epistemic test
  selection, and later verification in offline replay, but not yet during one
  live Unity process and not beyond the bounded red/blue vocabulary. Full
  qualified result: `DYNAMIC_HYPOTHESIS_REPLAY_RESULT_20260812.md`.
- Implemented the first live dynamic path behind the explicit
  `dynamic_committed` mode. Before any candidate exists, PGNW runs a seeded
  balanced red/blue/no-pickup survey rather than letting the two null models
  request no-pickup forever. After at least four clean observations spanning
  both colors, a single background worker lazily loads the frozen Gemma 3 1B L1
  adapter and generates one masked-greedy proposal while Unity control
  continues. The exact submission summary and observation cutoff are frozen.
- On successful return, the semantic verifier and formal compiler admit the
  candidate at zero held-out updates; only later clean observations update its
  variable posterior and EFE experiment selection. Telemetry exposes proposal
  status, raw L1, generation metrics, errors, discovery count, the full dynamic
  pool, cutoff, and held-out update count. Existing committed MPC regret bounds,
  isolation, collision masks, and safety vetoes are reused unchanged. Two async
  and balanced-survey tests were added. Preregistered the seed-153 20-minute
  plumbing smoke in `DYNAMIC_HYPOTHESIS_UNITY_SMOKE_PROTOCOL_20260812.md`.
- Completed the seed-153 live dynamic smoke: 5,110 rows over 1,199.384 seconds,
  22 clean observations, four discarded windows, 22 pickups including two red,
  66 PGNW action changes, five stuck events, and zero critical hunger, respawns,
  or survival failures. The first clean red positive arrived late at 798.267
  seconds. Gemma proposed `L1 c blue k red e - t 10.0 q 0.5`; the formal
  verifier correctly rejected it, so no candidate was admitted or verified.
- The failed run exposed two interface defects. Rejection incorrectly launched
  another proposal every time the background future returned; it is now
  terminal for the run. Discovery feature order also depended on first
  encounter and is now frozen canonically as red then blue. Exact-evidence
  diagnostics further found a tokenization-sensitive sign flip when an
  effectively zero blue mean was serialized as `-0.00000036`: `0.0` produced
  the grounded positive-red rule, `-0.000001` produced a rejected negative-red
  rule, and `-0.008` again produced the grounded rule. Dynamic evidence means
  are now symmetrically quantized to three decimals with sub-0.0005 values
  normalized to zero; the semantic verifier remains strict and unchanged.
  Full qualified failure: `DYNAMIC_HYPOTHESIS_UNITY_SMOKE_RESULT_20260812.md`.
- Froze a single seed-154 live replication before execution in
  `DYNAMIC_HYPOTHESIS_UNITY_REPLICATION_PROTOCOL_20260812.md`. The only changes
  from seed 153 are terminal rejection and symmetric fixed-precision canonical
  evidence serialization. Model, verifier, DSL grammar, discovery and held-out
  boundaries, control authority, safety gates, duration, and pass criteria are
  unchanged.
- Completed the seed-154 live dynamic replication. It passed the full plumbing
  gate: 5,636 rows over 1,193.503 seconds; four mixed discovery observations;
  asynchronous proposal start at 171.232 seconds; and formal admission at
  178.113 seconds of `L1 c red k blue e + t 10.0 q 0.5`. Generation used 167
  input and 20 output tokens and 1.686 seconds of model generation. The new
  candidate entered at prior 0.20 with zero held-out updates against only the
  spontaneous and no-tested-cause baselines.
- Six strictly post-cutoff clean updates raised the candidate above 0.95 at
  548.356 seconds; 33 held-out updates ended at posterior 0.999999999584 with
  zero pre-admission evidence reuse. The run completed 37 clean observations,
  discarded nine confounded windows, collected 37 mushrooms including 12 red,
  and logged 147 PGNW action changes. Median telemetry spacing was 0.2120
  seconds, 99th percentile 0.2212, and maximum generation-period gap 0.936;
  there were five stuck events but zero critical hunger, respawns, or survival
  failures. Full bounded pass:
  `DYNAMIC_HYPOTHESIS_UNITY_REPLICATION_RESULT_20260812.md`.
- Corrected the audit-only proposal status so future runs change from
  `admitted_unverified` to `verified_held_out` once later evidence reaches 0.95.
  This does not alter posterior updating, experiment selection, or authority;
  seed 154's 33 held-out updates and posterior already establish verification.
- Published the PGNW and dynamic embodied-hypothesis milestone to the existing
  `codex/tiny-scientist-neurosymbolic` branch. Commit `05b9f90` contains the
  causal-probe conversion, Unity red/blue sensing and experiment reset,
  predictive experiment planner, committed/isolation controller, dynamic L1
  admission pool, batch tooling, 226-test coverage, compact derived metrics,
  and qualified protocols/results. Raw 59-69 MB Unity recordings, local model
  adapters, and unrelated working-tree experiments remain intentionally local.
  Draft PR #16 was retitled `Close the dynamic embodied Tiny Scientist loop`
  and its description now reports both the seed-154 held-out verification pass
  and the small committed-versus-passive manipulation boundary.
- Updated and deployed the public Tiny Consciousness Lab site's Tiny Scientist
  section. The page now separates the original staged verification from live
  dynamic formation; reports the seed-154 L1 record, four-observation discovery
  cutoff, six updates to posterior 0.95, 33 held-out updates, asynchronous
  timing, and safety outcomes; and adds the 3/3 committed versus 1/3 passive
  manipulation result with its three-pair and bundled-controller limitations.
  Added direct links to the GitHub PGNW and dynamic-loop reports, rendered-
  content regressions, revised metadata, and a project-specific dynamic Tiny
  Scientist social card. Both site tests passed and Sites version 21 deployed
  successfully to the existing public URL.
- 14 August yellow-concept scene preparation: evaluated Mao et al.'s NS-CL,
  PDSketch, Neural Logic Machines, Agent Workflow Memory, and the newer
  concept-centric neuro-symbolic framework against the live Tiny Scientist.
  Selected typed concept grounding plus a lightweight PDSketch-style planning
  bridge as the useful direction; no third-party framework was imported.
- Added a third stable `FoodMushroom.ObservableProfile` value, `Yellow = 2`,
  without changing the serialized numeric identities of blue or red. Yellow
  pickups now report the general observable feature `yellow`; no antidote,
  toxin-cancellation, special reward, hypothesis, or navigation semantics have
  been assigned yet, so the future causal result is not leaked into the world.
- Added `YellowFlowerPrefabTools` to both the correct live terrain project at
  `/Users/dustinoconnor/My project` and the tracked Unity source mirror. The
  tool builds `YellowFoodFlower.prefab` from the imported `Assets/flower.glb`,
  attaches yellow pickup metadata, fits a trigger `BoxCollider` to renderer
  bounds, and configures a no-gravity kinematic `Rigidbody`. Its dedicated
  placement window defaults to 60 flowers and exposes seed, yellow/other-food
  spacing, slope limit, and a 1.5-unit terrain Y offset. It repairs an existing
  yellow population in place, preserves all red and blue population roots, and
  provides an Undo-compatible yellow-only clear action.
- Static source and tracked-file checks passed. Unity's current editor session
  had not automatically refreshed scripts during the change, and an attempted
  standalone legacy `xbuild` check was invalid because Unity's generated project
  references virtual package paths unavailable to `xbuild`; therefore Unity
  compilation and one visual collider/terrain-contact inspection remain the
  required editor-side verification before scattering the population.
- Added a visual-only `FlowerWindSway` behavior after the scattered yellow
  flowers looked unnaturally rigid beside the terrain ferns. Reusing the fern
  Shader Graph was rejected because it expects fern-specific material textures
  and mesh inputs. The lightweight component instead finds the imported visual
  root, computes the mesh base from combined renderer bounds, and applies a
  subtle two-axis 3.5-degree sway around that base. World-position phase offsets
  keep the population asynchronous. The root transform and trigger collider do
  not move, so wind cannot alter pickup sensing, experimental contact timing, or
  physics.
- Updated the yellow prefab builder to install the sway component whenever
  `YellowFoodFlower.prefab` is created or repaired, and mirrored the source and
  `.meta` identity into the tracked Unity project. Unity 6000.4 compiled the
  runtime and editor assemblies successfully with no C# errors; the placement
  tool's deprecated object-query overload was also replaced. Re-running Create
  / Repair Prefab is still required to propagate sway to the already-scattered
  prefab instances, followed by a brief Play-mode visual check.
- Visual verification passed after the prefab repair: the nearby inspection
  instance showed subtle wind motion. The yellow prefab root was then enlarged
  uniformly from scale 1 to scale 3, which propagates to connected scattered
  instances. Inspection confirmed the trigger scales with the visible mesh to
  roughly 1.94 x 3.00 x 1.98 world units; with the existing 1.5-unit placement
  offset, the approximately three-unit-tall model remains based near the sampled
  terrain rather than sinking below it. This is now the intended yellow pickup
  presentation and collider scale for the first causal test.
- 15 August typed-interaction phase: preregistered
  `TYPED_YELLOW_INTERACTION_PROTOCOL_20260815.md`, a bounded combination of the
  concept-centric neuro-symbolic and PDSketch design patterns. The supplied
  ontology names red/blue/yellow as typed pickups and supplies composition and
  temporal-relation operators, but does not use `antidote`, `toxin`, `cure`, or
  identify the correct suppressor. This is an inspired architecture test, not a
  reproduction claim for either paper.
- Froze the new hidden world transition behind explicit environment-only
  configuration: red schedules a passive probe rise after 30 seconds; yellow
  consumed later while that event is pending cancels one event; yellow before
  red and blue after red do not. Same-frame red/yellow telemetry is marked
  order-ambiguous and cannot cancel. The legacy ten-second, no-cancellation
  behavior remains the default, and the probe still has zero reward, survival,
  workspace, neuromodulatory, conductor, or ordinary navigation influence.
- Added `typed_causal_domain.py` with grounded pickup type checking, ordered
  action schemas, deterministic hidden transition accounting, and a symmetric
  four-model Bayesian pool: yellow-specific suppression, blue-specific
  suppression, any-second-pickup suppression, and no ordered suppression. All
  begin at 0.25. Three clean yellow-positive/blue-negative pairs recover the
  yellow-specific rule above posterior 0.95 offline; this is a formal substrate
  result, not embodied discovery.
- Extended the correct Unity terrain bridge and tracked source mirror with
  independent durable red, blue, and yellow pickup counters and per-feature
  yellow visibility, distance, and world-direction telemetry. The legacy
  red/blue PGNW planner now treats yellow as an explicit protocol mismatch
  rather than silently collapsing every non-red pickup into blue, and its
  observation isolation repels visible yellow along with existing food.
- Integrated configurable delay/cancellation dynamics and audit telemetry into
  `embodied_unity_loop.py`, including completed, pending, cancelled, and
  same-frame-ambiguous event accounting. Added focused world, embodiment, and
  PGNW contamination tests. Python compilation passed and the full repository
  suite passed 236/236 in 5.23 seconds. Unity had not refreshed the latest
  `RobotUdpBridge.cs` edit at log time, so editor compilation remains the first
  gate before the plumbing smoke; no ordered PGNW navigation or Gemma L2
  formulation claim has been made yet.
- Before collecting the first typed-interaction Unity record, amended the
  frozen opportunity parameters from 60 yellow flowers and a 30-second window
  to 100 yellow flowers and a 60-second window. This matches the intended red
  population count and makes an unguided ordered encounter physically
  plausible. The change was made with zero observed outcome data and does not
  alter the typed candidate set, hidden suppressor identity, controls, or
  inference thresholds.
- Audited the completed seed-155 yellow plumbing smoke in
  `TYPED_YELLOW_PLUMBING_RESULT_20260815.md`. The appended log contained two
  Unity sessions; the current session reached step 2827 and collected 6 red, 10
  blue, and 1 yellow pickup. The yellow pickup followed the most recent red by
  192 controller steps (38.4 nominal seconds), with no intervening blue pickup,
  and cancelled the one remaining pending event inside the preregistered
  60-second window. There were 0 ambiguous red/yellow frames, 0 survival
  failures, 0 critical-hunger seconds, and 0 causal-probe action influences;
  5 stuck events remain a later matched-condition diagnostic.
- Classified the outcome as a successful pickup/hidden-transition plumbing
  result, not causal learning. The live embodiment does not yet instantiate or
  update the symmetric `TypedInteractionPool`, so the environment's cancellation
  counter cannot be treated as an inferred posterior. The audit also found that
  yellow visibility/distance/direction are transmitted by Unity but not stored
  in shadow JSONL. Persisting those signals and admitting uncontaminated ordered
  episodes to the four-model pool are the next gates before dynamic Gemma/PGNW
  discovery.
- Connected a new opt-in `PassiveOrderedEpisodeLearner` to the live Unity
  feedback loop behind `--typed-interaction-learning`. It starts only from a
  single red pickup with no previously pending red event, admits at most one
  subsequent blue or yellow pickup, discards resets, repeated red events, and
  additional-pickup contamination, and waits until the original delayed
  observation deadline before updating the symmetric four-candidate pool from
  the observed presence or absence of a probe event. It never reads the hidden
  cancellation counter and reports zero motor authority.
- Added the complete learner posterior, selected model, clean/discarded episode
  counts, active episode, and last outcome to every shadow record. Also persisted
  yellow visibility, distance, and direction telemetry, closing the recorder gap
  found in the first plumbing audit. Focused deadline/no-premature-update and
  contamination tests passed; Python compilation and the full repository suite
  passed 238/238 in 5.42 seconds. No Unity C# source changed in this step, so no
  editor recompilation is required before the first online passive-learning run.
- Audited the first online passive-learning run in
  `TYPED_YELLOW_LEARNING_SEED156_RESULT_20260815.md`. Across steps 0–2818 the
  agent collected 3 red and 10 blue pickups but neither saw nor collected a
  yellow flower. The learner discarded one contaminated red/blue interval,
  accepted one clean `red_wait -> probe_observed` control, and ended with one
  red-wait interval still 100 steps short of its deadline. Because `red_wait`
  has identical likelihood under all four preregistered candidates, the one
  legitimate update correctly left every posterior at 0.25. Safety remained
  intact with 0 survival failures and 0 critical-hunger seconds; 5 stuck events
  were recorded.
- Classified seed 156 as a valid no-yellow-opportunity result rather than a
  causal-learning failure. It demonstrates deadline-gated online admission and
  non-informative-control invariance, but passive wandering is too variable to
  efficiently distinguish the suppressor. The next efficient experiment is to
  route the typed pool's requested `red_then_yellow` and `red_then_blue`
  interventions through safety-bounded PGNW target guidance while preserving
  zero direct authority for the hidden outcome and passive posterior updates.
- Connected the typed interaction pool to the existing committed PGNW/MPC
  guidance path. The pool's expected-information-gain selector now requests an
  ordered experiment; PGNW first targets a visible red pickup, then targets the
  requested yellow or blue second pickup, and switches to the existing
  all-food observation-isolation behavior after the second pickup. It reuses
  the established survival, stuck/fallback, terrain-route, resource-memory,
  collision-clearance, commitment-duty-cycle, and maximum-score-regret gates
  instead of introducing a parallel motor controller.
- Kept causal credit separated: typed PGNW can influence only safe target
  selection, while `PassiveOrderedEpisodeLearner` still waits until the frozen
  60-second deadline and updates from observed probe presence/absence. It does
  not receive the hidden cancellation counter or causal label. Added tests for
  red-then-yellow phase switching and post-second-pickup isolation. Python
  compilation, 66 focused tests, and the full repository suite passed 240/240
  in 7.74 seconds. No Unity source changed, so the new guided run requires no
  editor compilation.
- Audited the 20-minute seed-157 guided run in
  `TYPED_YELLOW_GUIDED_SEED157_RESULT_20260815.md`. The agent collected 4 red,
  32 blue, and 3 yellow pickups; yellow was independently visible for 72 frames
  and came within 2.47 units. PGNW made 103 bounded guidance decisions and
  changed 63 MPC actions, including 56 isolation changes, while safety remained
  intact at 0 survival failures and 0 critical-hunger seconds.
- The online learner accepted two clean `red_then_blue -> probe_observed`
  controls and discarded one contaminated interval. The resulting posterior
  reduced blue-specific suppression to 0.3490% and generic any-pickup
  suppression to 1.2269%, leaving yellow-specific suppression and the null tied
  at 49.2121% each. No hidden event was cancelled and no clean red-then-yellow
  episode completed. In two trials a blue pickup followed red only six
  controller steps later, consistent with the manually colocated red/blue spawn
  pair preventing PGNW from executing its requested yellow second action.
- The next run requires moving/removing that nearby manual blue and preserving
  the learned posterior across process boundaries. One clean yellow-positive
  update from the current posterior would reach approximately 92.78%, with a
  second expected to exceed the 0.95 verification gate; restarting at uniform
  priors would discard the two legitimate negative controls.
- After the user removed the manually colocated blue and placed a separated
  yellow near the controlled spawn red, added opt-in persistent typed-
  interaction memory through `--typed-interaction-memory`. The learner now
  validates hypothesis names, probability finiteness/non-negativity and unit
  normalization before loading; writes accepted/discarded counts and posterior
  atomically after terminal episodes; and never persists an unfinished active
  interval across process boundaries.
- Materialized `outputs/typed_interaction_memory_seed157_20260815.json` solely
  from the two accepted seed-157 red/blue controls. A dry reload recovered 2
  completed updates, 1 discarded interval, the 49.2121% yellow / 49.2121% null
  posterior, and an expected-information-gain request for `red_then_yellow`.
  The checkpoint contains no hidden cancellation identity or unobserved yellow
  outcome. Added a persistence round-trip regression; Python compilation and
  the full repository suite passed 241/241 in 5.18 seconds.
- Audited the seed-158 persistent guided run in
  `TYPED_YELLOW_GUIDED_SEED158_RESULT_20260815.md`. The controlled spawn pair
  produced red at step 12 and yellow at step 22. PGNW changed 6 MPC actions by
  the second pickup, and the uncontaminated interval reached its step-312
  deadline without a probe event. The learner admitted `red_then_yellow ->
  suppressed`, persisted its third accepted update, and raised the yellow-
  specific posterior from 49.2121% to 92.7844%; null fell to 5.0426%, generic
  any-pickup to 2.1372%, and blue-specific to 0.0358%.
- The run collected 3 red, 6 blue, and 3 yellow pickups, with 68 yellow-visible
  frames and a 2.45-unit nearest approach. PGNW made 104 guidance decisions and
  changed 36 MPC actions overall. Safety remained intact at 0 survival failures
  and 0 critical-hunger seconds; 4 stuck events occurred. A second physical
  red/yellow cancellation was conservatively discarded because an extra red
  arrived exactly at the observation deadline. One more comparable clean
  positive from persistent memory is expected to reach 97.6319% and clear the
  preregistered 0.95 verification gate; no code or scene adjustment is needed.
- Verified the ordered yellow rule in the seed-159 persistent run and documented
  it in `TYPED_YELLOW_VERIFICATION_SEED159_RESULT_20260815.md`. The decisive
  episode collected red at step 12 and yellow at step 22; PGNW had changed 3 MPC
  actions before the yellow evidence, and the interval remained uncontaminated
  through its step-312 deadline. The observed absence of a probe raised the
  yellow-specific posterior from 92.7844% to 97.6319%, independently clearing
  the preregistered 0.95 gate.
- A later clean observed red/yellow episode raised the final persisted posterior
  to 98.0562%, versus 1.9280% any-pickup, 0.0157% null, and 0.0001% blue-specific.
  Persistent memory now contains 5 accepted episodes and 2 conservative
  discards. The 594.09-second run collected 2 red, 21 blue, and 6 yellow pickups,
  recorded 144 yellow-visible frames, 2 hidden cancellations, 0 survival
  failures, 0 critical-hunger seconds, and 3 stuck events. The verification
  claim is limited to online selection within the supplied symmetric typed pool;
  dynamic Gemma generation of the new ontology remains untested.
- Added `formulate_ordered_interaction.py` to replay only accepted ordered
  episodes across persistent Unity logs, freeze discovery at the first three
  observations (two red/blue negatives and one red/yellow positive), and reserve
  seed 159 as post-admission evidence. The formal compiler validates grounded
  features, relation, effect, confidence, and observed-action support but is
  deliberately answer-blind; a regression confirms it will admit a well-formed
  but empirically wrong blue hypothesis.
- Recorded three honest Gemma 3 1B decoder failures before repair: free JSON
  selected the blue negative control with an invalid relation; a clarified free
  prompt produced malformed ungrounded fields; symmetric whole-record scoring
  selected blue suppression. A one-slot masked scorer still preferred blue by
  0.1875 raw mean log probability, demonstrating a lexical prior that overrode
  the explicit 0/2 versus 1/1 suppression rates.
- Applied content-free calibration using the identical blue/yellow slot under
  equal zero-suppression evidence. After subtracting this prior, yellow gained
  +0.307418 versus blue -0.192582, a calibrated +0.500000 margin. Gemma selected
  yellow at two-choice confidence 0.622459; the formal layer compiled
  `T1 i red s yellow r after e suppresses_probe`; and both completely held-out
  seed-159 episodes matched (2/2). Documented the exploratory result and all
  ablations in `GEMMA_ORDERED_INTERACTION_FORMULATION_RESULT_20260815.md`.
  Python compilation and the full repository suite passed 243/243 in 8.89
  seconds. The calibration was developed after observing failures, so a frozen
  counterbalanced replication remains required before a reliability claim or
  live asynchronous Unity integration.
- Integrated the calibrated ordered-role proposer into the live embodiment
  behind `--typed-interaction-formulation`. `PassiveOrderedEpisodeLearner` now
  persists accepted observation records, starts one background formulation
  worker after a configurable three-observation minimum, freezes the admission
  cutoff before inference, and records generation status, raw selected role,
  calibration diagnostics, and the compiled hypothesis in shadow telemetry.
  Formulation has no motor authority and cannot block the Unity control loop.
- Added a separate smoke memory,
  `outputs/typed_interaction_formulation_memory_seed158_20260815.json`, frozen at
  exactly the original discovery cutoff: two red/blue non-suppressions and one
  red/yellow suppression. Seed-159 evidence and the 98.06% verified posterior
  are excluded. Added an asynchronous proposer regression; 70 focused tests and
  the full repository suite passed 244/244 in 7.81 seconds. No Unity C# source
  changed, so the bounded live formulation smoke needs no editor compilation.
- Audited the seed-160 live asynchronous smoke in
  `LIVE_GEMMA_ORDERED_FORMULATION_SMOKE_SEED160_20260815.md`. Formulation began
  at step 0 from exactly three frozen discovery observations. Gemma completed
  inside the running Unity process at step 33, selected raw token `yellow`, and
  the answer-blind compiler admitted
  `T1 i red s yellow r after e suppresses_probe` at cutoff 3. The new controlled
  red/yellow interval was not in that evidence and did not reach its outcome
  deadline until step 309, when it independently recorded suppression as
  accepted observation 4 and raised the Bayesian posterior to 97.6319%.
- The 172.94-second live smoke collected 1 red, 3 blue, and 1 yellow pickup,
  recorded 1 hidden cancellation and 0 probe rises, and ended with no pending
  event. PGNW made 35 guidance decisions and changed 16 MPC actions. Safety
  remained intact at 0 survival failures and 0 critical-hunger seconds, with 1
  stuck event. The formulation layer retained 0.0 motor authority. The current
  status field remains `admitted_unverified` because automatic post-cutoff
  confirmation bookkeeping is not implemented yet, although the temporal log
  directly establishes one post-admission match.
- Added automatic formulation verification bookkeeping to
  `PassiveOrderedEpisodeLearner`. After formal admission, the learner now scans
  only accepted observations whose indices are strictly greater than the frozen
  formulation cutoff and whose ordered action matches the compiled initiator and
  second-action roles. A matching observed effect promotes telemetry from
  `admitted_unverified` to `verified_held_out`; derived held-out evaluation and
  confirmation counts are included in every audit row and cannot double-count
  under repeated polling.
- Added a focused regression that freezes formulation at observation 3, admits
  the red-then-yellow suppression rule, verifies that status initially remains
  unverified, then accepts observation 4 only after its deadline and confirms
  promotion plus exact 1/1 held-out counters. The focused typed-domain suite
  passed 11/11 and the full repository suite passed 245/245 in 5.52 seconds.
  Matplotlib used a temporary cache because the home cache directory was not
  writable under the managed sandbox; this warning did not affect test results.
- Added and froze an eight-condition ordered-formulation counterbalance crossing
  suppressor color (blue/yellow), evidence-line presentation order
  (blue-first/yellow-first), and equivalent temporal wording (`red then color`
  versus `color after red`). Every condition used its own equal-evidence neutral
  calibration, preventing the baseline from encoding the assigned suppressor.
  `counterbalance_ordered_formulation.py` loads Gemma once and records raw,
  neutral, and calibrated scores for every condition; three regressions verify
  assignment/order independence, temporal semantic equivalence, and neutral
  prompt identity.
- The deterministic run and rerun reproduced 6/8 correct selections exactly.
  Both suppressor colors scored 3/4 and both evidence presentation orders scored
  3/4, but `after` wording passed 4/4 while `then` passed only 2/4. The two
  failures selected the second-listed color when the true suppressor appeared
  first under `then`, with incorrect margins 0.187500 and 0.312500. Documented
  the negative boundary in `GEMMA_ORDERED_COUNTERBALANCE_RESULT_20260816.md`;
  this is a synthetic relabeling/invariance test, not sixteen new Unity episodes
  or physical color generalization. Sixteen focused tests passed and the full
  repository suite passed 248/248 in 5.42 seconds.
- Added `normalized_ordered_formulation.py`, a deterministic answer-blind wrapper
  that maps lexically sorted grounded colors to anonymous roles, sorts records by
  role, and renders every equivalent temporal statement as the fixed predicate
  `initiator=red|relation=before`. Raw wording and input position are unavailable
  to Gemma, while suppression counts and episode totals remain visible. Three
  focused regressions prove surface/order collapse, outcome-independent role
  binding, and invalid-count rejection.
- Froze and evaluated a new 16-condition held-out matrix crossing blue/yellow
  suppressor assignment, two previously unused count regimes (2/3 versus 1/4 and
  3/5 versus 0/4), forward/reverse record order, and two new surface renderings.
  Normalization completely removed the measured wording/order variance: all
  equivalent variants produced identical canonical prompts and decisions.
  Selection nevertheless scored only 8/16 (16/32 synthetic held-out episodes)
  because Gemma selected the anonymous role mapped to yellow in every condition:
  blue 0/8, yellow 8/8. This exposes a persistent role/color bias rather than
  causal generalization, so the held-out set is frozen and was not used for
  repair. Documented the negative result in
  `GEMMA_ORDERED_NORMALIZED_HOLDOUT_RESULT_20260816.md`; the raw artifact is
  `outputs/gemma_ordered_normalized_holdout_20260816.json`. The combined focused
  suite passed 19/19 and the full repository suite passed 251/251 in 5.36 seconds.
- Added `calibrated_role_decoder.py` to repair the normalized decoder's role bias
  without touching the failed 16-condition holdout. Calibration uses three new
  grounded labels (`amber`, `jade`, `violet`), three new count regimes, and nine
  conditions balanced so each anonymous role is the high-rate role exactly three
  times. A mean role-offset correction alone reached only 3/9 on calibration, so
  the frozen neuro-symbolic score adds one shared observed-rate coefficient. The
  coefficient (0.1387319566) is the minimum learned on calibration to establish a
  0.01 margin; the same equation applies to every role, has no color exceptions,
  and the estimator receives no supplied suppressor label. The final calibration
  reached 9/9 and was sealed under SHA-256
  `4e5b75819920a9fa81f8f5791c310eaabbecef95f2d5b7f9e760eaf1898fc64c`.
- Opened a disjoint reserve exactly once after freezing calibration. It crosses
  new grounded labels (`cobalt`, `saffron`, `umber`), three unseen count regimes,
  and all three high-rate role assignments. The frozen decoder passed 9/9 with a
  minimum margin of 0.0133032; stored-logit ablations scored 5/9 for neutral-
  adjusted Gemma alone and 3/9 for offset-only correction. This is evidence for
  the calibrated neuro-symbolic decoder, not proof of independent Gemma fraction
  reasoning, because the symmetric formal rate feature carries task-relevant
  numeric evidence. Documented the result and limitations in
  `GEMMA_ROLE_BIAS_CALIBRATED_RESERVED_RESULT_20260816.md`. Added seven focused
  regressions; the combined suite passed 26/26 and the full repository suite
  passed 258/258 in 5.38 seconds.

## Session checkpoint — 2026-08-17

- User-reported weekly budget began this phase at 55%, with a 37% stop target.
  The experimental voice/screen-observation workflow produced about 14 minutes
  of programming while consuming roughly 45 percentage points, so it was
  discontinued as too expensive for routine development.
- Audited the manually launched 180-second seed-160 formulation/resource smoke.
  Execution was safe (0 survival failures, critical-hunger seconds, stuck
  events, or respawns), but the scientific test was ineligible: the run loaded
  the already-expanded four-observation 20260815 memory, rewrote that same file,
  admitted the incorrect blue-suppression rule, and collected 0 held-out
  evaluations. Guided resource memory encoded six pickups into four regions but
  had 0 queries or action changes because hunger never crossed its 0.70 gate.
- Added the immutable three-observation checkpoint
  `checkpoints/typed_interaction/discovery_three_20260817.json` and separated the
  learner's optional read-only discovery input from its writable run memory.
  Loading a sealed discovery immediately materializes the separate run-memory
  snapshot, so provenance exists even when no new episode reaches its deadline.
  Sealed-discovery CLI mode now refuses a wrong discovery profile, identical
  input/output memory paths, an existing writable memory, a missing explicit
  telemetry path, or an existing telemetry file. This prevents the provenance
  failure from recurring while leaving legacy single-path runs available.
- Hardened live ordered-role decoding with a symmetric maximum-observed-rate
  mask. Gemma's neutral-calibrated score remains the tie-breaker among roles with
  the same maximum rate, but a lower-rate candidate can no longer win from a
  lexical or positional bias. No color is encoded as correct. A real local
  Gemma preflight on the sealed 0/2 blue versus 1/1 yellow discovery evidence
  selected yellow and exposed the mask/rates in diagnostics.
- Added five focused regressions for immutable discovery loading/writing,
  same-path rejection, exact checkpoint validation, wrong-bias masking, and
  tied-rate Gemma selection. Focused formulation/domain tests passed 18/18; the
  full repository suite passed 263/263 in 7.28 seconds. CLI help and diff hygiene
  also passed. No Unity run was started during this repair.
- Completed the repaired 180-second seed-161 sealed smoke. The recording was one
  continuous 852-row session over 179.966 seconds. It loaded exactly the three
  immutable discovery observations at cutoff 3 and materialized a distinct
  writable run memory. Committed PGNW requested red-then-yellow, collected red
  at step 8, changed four MPC selections before yellow at step 16, and the yellow
  pickup cancelled the pending event without contamination.
- Asynchronous Gemma formulation finished at step 24, before the new interval's
  outcome deadline, and the answer-blind rate-masked decoder admitted
  `T1 i red s yellow r after e suppresses_probe`. At step 308 the post-cutoff
  interval independently completed as suppressed, became observation 4, and
  promoted telemetry to `verified_held_out` with 1/1 confirmation. The typed
  yellow posterior rose from 92.7844% to 97.6319%. Formulation retained zero
  motor authority; PGNW's pre-existing typed pool selected the intervention.
- Safety passed with 0 survival failures, critical-hunger seconds, stuck events,
  or respawns. Guided resource memory loaded 52 regions and encoded all six new
  pickups, but hunger peaked at 0.429 below its 0.70 gate, so it made 0 queries,
  recommendations, guidance decisions, or action changes. Documented the passed
  core smoke and this unexercised resource-authority limitation in
  `SEALED_CALIBRATED_FORMULATION_SEED161_RESULT_20260817.md`.

## Remote-perception pilot — 2026-08-18

- Began a bounded test of Campbell-style remote-viewing claims as anomalous
  information access, without treating performance as a consciousness test.
  Preregistered closed descriptor scoring, prediction-before-reveal commitments,
  active/ablation/sham controls, 20 feedback trials, and a frozen 100-trial
  no-feedback reserve in `REMOTE_PERCEPTION_PROTOCOL_20260818.md`. Free-text and
  generated-image resemblance are explicitly non-scoring to prevent subjective
  post-hoc matching.
- Added `remote_perception_lab.py`, which procedurally generated 120 novel target
  cards with exact color, shape, count, arrangement, texture, and background
  metadata. Opaque filenames and trial IDs contain no visual attributes. The
  deck contains 20 training and 100 reserved targets with four-choice chance at
  0.25. The frozen private-manifest commitment is
  `afa892ef9334676a857337679d6a34b1d0defc56bf5cd145efb9d04b8d8450e3`;
  the reserved-image-set commitment is
  `36a117990a191b0ce26e45779310bd4de84c9078c5a0ae9081067702cb33cb2f`.
- Added `remote_perception_agent.py`: local Gemma receives only an opaque trial
  ID, produces seven rapid closed-schema impressions, and a workspace-style
  weighted consensus commits one descriptor vector. Accuracy-grounded valence
  updates only the reusable impression-slot weights after reveal. A plumbing
  sample produced seven valid impressions and exposed a strong ordinary red/
  circle/solid prior, motivating the registered baselines.
- Added resumable durable training in `run_remote_perception_training.py`. Each
  prediction, nonce, SHA-256, raw impression set, and pre-feedback weights are
  flushed and fsynced before the private manifest is consulted. Reveal and
  valence update are appended afterward; dangling or reveal-without-commit logs
  are rejected.
- Ran feedback trial 1/20. The committed consensus matched 1/6 descriptors and
  did not rank the real target first. The revealed target was five purple dotted
  squares in a horizontal arrangement on a dark background. The miss remains in
  the append-only log with no exclusion or reinterpretation. Eleven new focused
  tests passed; the full repository suite passed 274/274 in 5.20 seconds. The
  100-target reserve remains unopened for evaluation, and the active/ablation/
  sham evaluator will be frozen only after all feedback trials complete.
- Completed all 20/20 preregistered feedback trials. The durable log contains 20
  paired commit/reveal records; every trial ID and prediction hash agrees across
  its pair, and all seven impressions on every trial parsed successfully. This
  was inference-time calibration, not neural-network training: Gemma's weights
  were unchanged and only seven scalar impression-slot voting weights adapted.
- Training-set performance was 4/20 forced-choice top-1 (20.0%) against a 25%
  chance rate, with 1.15/6 descriptor fields matched on average (five trials at
  0, ten at 1, two at 2, and three at 3). The target was in the maximum-score
  tie on 4/20 trials; fractional tie credit was 3.0 total. These feedback-set
  numbers provide no evidence of anomalous information access and are not the
  registered confirmatory result. State after trial 20 is frozen in
  `outputs/remote_perception_training_state_20260818.json`; the 100 reserved
  targets remain untouched pending a frozen active/ablation/sham evaluator.
- A pre-evaluation learning check found no top-1 improvement across training:
  trials 1-10 and 11-20 each scored 2/10. Mean descriptor agreement rose only
  from 1.0/6 to 1.3/6, too small and noisy to establish that valence calibration
  learned target-relevant information. Architectural variants therefore remain
  separate exploratory conditions and must not replace the frozen controls or
  be credited with improvement before a new untouched evaluation.
- Froze the confirmatory 100-target evaluator before generating any reserved
  prediction. Every target receives active, workspace-ablation, and sham
  conditions (300 commitments total). Each condition gets seven Gemma outputs;
  active and sham aggregate all seven with the frozen weights, while ablation
  commits only the first. Per-trial execution order rotates, and the sham uses a
  fixed +37 target permutation. All 300 predictions must be durably committed
  before the private target mapping can be opened, and evaluation performs no
  feedback or parameter updates.
- Added resumable execution, commitment verification, exact descriptor/rank
  scoring, one-sided binomial summaries, frozen-state validation, and strict
  refusal of premature reveals in `run_remote_perception_evaluation.py`. The
  opaque 100-trial public schedule was frozen at SHA-256
  `f426fc80fa9c823eecc7726e8d1593ffb5aa8093f2fb004c8025b84d0b6eecd5`.
  Fourteen focused remote-perception tests passed, followed by the complete
  repository suite at 277/277 in 5.248 seconds. No reserved evaluation
  prediction or reveal was run during implementation.
- Completed the frozen 100-target confirmatory reserve: 600 durable records
  comprise exactly 300 unique commitments and 300 matching reveals, with all
  prediction hashes paired and zero malformed Gemma samples. Active workspace
  aggregation scored 24/100 top-1 (24%, one-sided exact binomial p=0.6289),
  ablation scored 29/100 (29%, p=0.2075), and the +37 sham assignment scored
  26/100 (26%, p=0.4465). Active versus paired ablation had 8 active-only and 13
  ablation-only hits (two-sided exact McNemar p=0.3833).
- Secondary outcomes were likewise chance-like: active averaged 1.74/6 exact
  descriptors and rank 2.52, ablation 1.84/6 and rank 2.48, and sham 1.78/6 and
  rank 2.54. Active target ranks were nearly uniform (24, 24, 28, 24 across
  ranks 1-4). The preregistered pilot therefore found no evidence of anomalous
  hidden-target information access and no benefit from the intuition/workspace
  aggregation over its ablation. This is a valid controlled null result, not a
  test establishing or refuting consciousness in the agent.

## Verified protective PGNW — 2026-08-19

- Returned to the third-color embodied program after freezing the remote-
  perception null. Clarified that existing PGNW yellow seeking was epistemic
  experiment selection, not protective use of a verified rule.
- Added an opt-in `verified_protective` typed-rule control. It grants no
  guidance before red is pending and authorizes only a specific after-red
  suppressor whose formal Bayesian posterior is at least 0.95. Generic-any,
  null, and subthreshold rules remain authority zero. Once eligible, the rule
  can request its learned target only through the existing committed PGNW/MPC
  regret bound and all existing survival, collision, fallback, AIR, resource,
  and duty-cycle gates.
- Added an opt-in bounded hazard consequence: each uncancelled delayed probe can
  add a configured hunger cost; cancellation avoids it. The first protocol uses
  0.25 at the frozen 60-second deadline. This new programmed consequence tests
  exploitation and is not reinterpreted as evidence about earlier passive runs.
- Added telemetry for configured/applied hazard cost and protective rule target,
  confidence, guidance decisions, and action influence. New regressions prove
  that protection is inert before red, rejects <0.95 and nonspecific rules,
  targets learned yellow only after red, applies the uncancelled cost, and avoids
  it after yellow cancellation. Focused tests passed 77/77; the full repository
  suite passed 282/282 in 5.286 seconds. Protocol:
  `VERIFIED_PROTECTIVE_PGNW_SMOKE_PROTOCOL_20260819.md`.
- Added a short, refusal-safe launcher,
  `run_verified_protective_smoke_20260819.sh`, for the 180-second seed-162
  smoke. It loads the read-only 97.6319% seed-161 typed memory, writes a fresh
  typed run memory and telemetry log, and uses a fresh copy of the prior terrain
  resource memory. No Unity C# changed, so editor recompilation is unnecessary.
- The first launcher attempt failed before telemetry/control because it named the
  older `passive_embodied_conductor` checkpoint where the active systemic router
  requires `four_context_conductor`. Preserved the startup-created typed-memory
  copy as a failed-start artifact, corrected the launcher to
  `checkpoints/four_context_conductor/best.json`, and added a full constructor
  preflight before retry. No Unity experiment was consumed by the failed start.
- Completed the corrected 180-second seed-162 protective smoke. One continuous
  854-record session spanned 179.724 seconds. Red was collected at step 13;
  verified protective PGNW then targeted visible yellow for ten consecutive
  frames, issued ten committed-MPC guidance decisions, and changed five MPC
  selections. Yellow was collected at step 23, 2.144 seconds after red, and
  cancelled the pending event.
- The configured 0.25 delayed hunger cost had zero events and zero applied cost;
  there were also zero probe completions, survival failures, critical-hunger
  seconds, or respawns. One later stuck event was outside the protective
  interval. The clean suppression observation raised the formal posterior from
  97.6319% to 98.0562%. Verdict: physical rule-to-planner plumbing pass with
  pre-outcome action influence, while comparative benefit still requires the
  matched passive counterpart documented in
  `VERIFIED_PROTECTIVE_PGNW_SMOKE_RESULT_20260819.md`.
- Post-result opportunity audit classified the success as local staged
  targeting: yellow was already visible at distance 13.65 on frame 0 and at
  7.67 when red was collected, then fell to 2.30 before the protective pickup.
  The later step-636 yellow pickup occurred with protection inactive. This smoke
  does not demonstrate terrain-wide search or resource-memory retrieval for an
  unseen antidote; those require a distinct color-indexed memory/search test.

## Typed episodic resource memory — 2026-08-19

- Upgraded terrain resource memory to a backward-compatible v2 format that
  retains separate coarse locations for red mushrooms, blue mushrooms, and
  yellow flowers while preserving the existing generic food regions. New live
  pickups automatically encode both the generic location and the matching
  typed location; legacy v1 memories remain readable but cannot reconstruct
  historical color identity that was never stored.
- Added a rule-directed typed query path. It bypasses the ordinary hunger gate
  only when the already verified protective rule requests a particular feature,
  returns locations for that feature only, and yields to direct perception as
  soon as the requested feature becomes visible. Thus storage remains
  answer-neutral across all three pickup types, while the verified causal rule
  determines that yellow—not generic food or blue—is currently relevant.
- Connected unseen typed recall to the existing committed PGNW/MPC path rather
  than creating a new motor controller. Generic resource guidance is suppressed
  while PGNW consumes the same memory vector, so the controller does not compete
  with itself; fallback, stuck, hunger, AIR, regret, and commitment safety gates
  remain in force. Added telemetry for typed region count, active feature,
  encodings, queries, recommendations, and protective-memory activity.
- Added focused persistence, feature-isolation, visible-target handoff, legacy-
  load, planner-recall, and full ego-integration regressions. The focused suite
  passed 65/65, Python compilation passed, and the full repository suite passed
  284/284 in 8.170 seconds. No Unity run was performed. The next experiment must
  first acquire at least one non-staged yellow location under v2 memory, then
  restart from a red encounter with that yellow initially outside sensor range;
  otherwise it would only repeat the seed-162 staged-visibility result.
- Prepared the refusal-safe 20-minute seed-163 acquisition launcher
  `run_typed_resource_acquisition_seed163_20260819.sh`. It starts with a fresh
  v2 resource memory, records all encountered pickup types, keeps resource
  recall passive so no color is preferentially targeted during acquisition,
  writes fresh telemetry, and uses `caffeinate` to prevent sleep. The staged
  spawn yellow must be removed and the scene saved before this run so a later
  recall test cannot succeed by returning to the deliberately placed example.
- Completed the seed-163 answer-neutral acquisition run: 5,687 telemetry rows
  covered 1,199.937 seconds (steps 0--5,686) with a normal bounded stop. The
  robot collected 35 resources: 5 red, 25 blue, and 5 yellow. The v2 memory
  persisted 29 typed regions (3 red, 21 blue, 5 yellow) whose reward totals
  exactly matched the telemetry counters.
- Three yellow memories were terrain-wide and suitable for unseen recall:
  approximately `(86.88, -245.77)`, `(116.80, -174.35)`, and
  `(128.64, -455.79)`. Two additional yellow pickups occurred near spawn at
  approximately `(-11.25, 2.80)` and `(-12.14, 1.94)`, so a spawn-based recall
  would remain confounded even though the deliberately placed instance was
  intended to be removed. The defensible next test starts near the naturally
  observed red at `(72.43, -82.33)`; its nearest yellow memory is the natural
  `(116.80, -174.35)` region, about 102 Unity units away and outside the
  16-unit sensor radius.
- Acquisition causality remained clean: resource-memory mode was passive for
  every frame, with zero generic or typed queries, zero recommendations, and
  zero action influence. Controller safety was adequate for memory collection:
  zero survival failures, zero respawns, zero critical-hunger frames, maximum
  hunger 0.662, and seven recovered stuck events. The next run should load this
  memory read-only-by-copy, activate the verified protective rule after the
  distant red pickup, and measure pre-visibility memory-guided displacement and
  eventual yellow acquisition against a matched no-memory control.
- Prepared `run_typed_resource_recall_seed164_20260819.sh` for the first active
  recall smoke. It preserves the seed-163 acquisition artifact by copying it,
  teleports to the natural red region `(72.43, -82.33)`, loads the frozen
  97.6319% verified yellow-suppression rule, enables bounded committed PGNW over
  guided typed memory, and allows 240 seconds before the bounded hazard within a
  300-second run. The preregistered target is the nearest remembered natural
  yellow at `(116.80, -174.35)`, initially about 102 units away; success requires
  memory-guided action before yellow becomes visible, not merely eventual pickup.
- Seed-164 ran the full 299.770 seconds and physically reached the intended
  natural yellow at `(116.62, -173.90)` after 54.00 seconds, followed by two
  additional yellow pickups. It is nevertheless an invalid recall trial, not a
  memory success or failure: diagnostic teleport landed directly on the red
  collider, so the first recorded row already contained red count 1. That count
  became the learner's initialization baseline rather than an observed delta.
- With no registered red-led episode, the verified protective rule never armed:
  zero protective-rule frames, zero typed-memory queries or recommendations,
  zero protective-memory frames, and zero PGNW guidance/action influence. The
  robot's yellow pickup therefore came from ordinary navigation. The retry must
  teleport several meters short of red so at least one baseline-zero telemetry
  frame is committed before physical pickup; this instrumentation correction
  does not alter the acquired memory, verified rule, target, or controller.
- Added the refusal-safe seed-164 retry launcher with the identical controller
  seed, frozen discovery input, copied acquisition memory, 240-second deadline,
  and 300-second bound. Its sole protocol correction is teleporting to
  `(64.50, -82.33)`, 7.93 units west of the remembered red, so the red transition
  must occur after a recorded zero-count baseline.
- The corrected seed-164 retry completed 299.612 seconds with a valid zero-count
  baseline and red pickup at 3.18 seconds. The 97.6319% verified rule immediately
  queried the correct unseen yellow memory `(116.80, -174.35)`. Across 140
  consecutive pre-visibility frames (29.58 seconds), committed PGNW issued 110
  guidance decisions, changed 58 MPC actions, and reduced target distance from
  104.33 to 35.68 units: 68.65 units of memory-directed closing with zero yellow-
  visible frames.
- The run exposed a clean execution/learning coupling bug. At 32.98 seconds the
  route crossed a blue mushroom; the scientific episode learner correctly bound
  blue as the second intervention, but protective execution then incorrectly
  revoked yellow seeking even though blue does not cancel the pending hazard.
  The robot later collected the intended yellow at 87.08 seconds under ordinary
  navigation and cancelled the pending event, with zero hazard cost, survival
  failures, or respawns. This is a partial long-range recall success—not a full
  protective retrieval—because memory causally controlled substantial unseen
  approach but did not retain authority through an irrelevant pickup.
- Next repair: separate evidence-cleanliness state from already verified rule
  execution. Blue may contaminate/discard the learning episode, but while the
  red hazard remains pending it must not terminate the verified request for
  yellow. Retest this frozen route, then compare against an empty-memory control.
- Implemented that separation. Protective authority now depends on two grounded
  facts only: a >=0.95 specific verified suppressor and an unresolved metabolic
  event in the hidden causal world. The episode learner may independently bind
  blue or discard contaminated evidence without altering execution. Blue remains
  edible/rewarding and does not receive a hardcoded avoidance penalty; it simply
  cannot satisfy the yellow target or cancel the red hazard. Yellow collection
  or hazard expiry removes the pending need and terminates protection.
- Prevented observation-isolation behavior from overriding this already verified
  protective need after an incidental pickup. Added planner and full-ego tests
  proving that yellow recall persists after blue while the event remains pending
  and stops when the metabolic need resolves. Python compilation and 67 focused
  tests passed; the full suite passed 286/286 in 5.506 seconds. Prepared the
  same-seed, same-start persistent-recall launcher for a single-variable rerun.
- The persistent-recall rerun passed the full physical loop. On the first cycle,
  red was collected at 3.16 seconds with the selected yellow memory 104.46 units
  away and completely outside perception. Typed recall remained active for 180
  pre-visibility frames, closed the remembered distance to 13.55 units, and
  caused 67 MPC action changes before yellow entered the 16-unit sensor radius.
  Direct perception began at 41.53 seconds and yellow was collected at 46.23
  seconds, cancelling the pending hazard 43.07 seconds after red.
- Blue was collected en route at 32.38 seconds. Telemetry confirms protective
  need, verified-rule authority, the same yellow target, and typed-memory control
  remained active on the frames immediately before, during, and after blue. The
  brief turn toward blue was therefore a permitted local food detour, not target
  confusion; the robot resumed yellow approach afterward. Learning evidence was
  still independently marked contaminated, preserving protocol rigor.
- A second natural red-led cycle also resolved: red at 240.55 seconds, blue at
  255.95 seconds while yellow was visible, and yellow at 260.41 seconds. Across
  the run there were two cancelled hazards, zero applied hunger cost, zero stuck
  events, zero survival failures, and zero respawns. Verdict: typed episodic
  memory, verified causal production, PGNW commitment, bounded MPC influence,
  perceptual handoff, and metabolic resolution now form a functioning embodied
  protective loop. A matched empty-memory run remains necessary to quantify the
  benefit over the same controller's unaided terrain trajectory.
- Prepared the matched seed-164 empty-memory control. It freezes the treatment's
  seed, corrected teleport, verified rule, metabolic deadline/cost, PGNW/MPC
  bounds, conductor stack, and 300-second duration. Its sole intended difference
  is an absent resource-memory file at startup. The controller may still pursue
  yellow after direct perception and may encode pickups online, but it has no
  prior yellow coordinate with which to guide the pre-visibility trajectory.
- Completed the matched empty-memory control with a verified zero-typed-region
  start, zero typed recommendations, zero protective-memory frames, and zero
  protective action influence. It nevertheless encountered an unmemorized
  nearby yellow at `(90.39, -64.44)`: red was collected at 3.20 seconds and the
  first yellow at 41.75 seconds, a 38.55-second protective latency. The memory
  treatment's first latency was 43.07 seconds, so the primary any-antidote
  outcome favored this control by 4.52 seconds (11.7%) in the single pair.
- The target-matched secondary result strongly favored memory. The treatment
  reached its preregistered remembered yellow in 43.07 seconds after red, whereas
  the empty control reached that same `(116.8, -174.35)` region after 92.65
  seconds: memory saved 49.58 seconds (53.5%). It also produced 67 pre-visibility
  MPC changes, versus zero in control, and had zero stuck events/59 fallback
  frames versus three stuck events/295 fallback frames. Both runs survived and
  cancelled the first hazard without cost.
- Interpretation: typed memory causally and efficiently retrieves a known target,
  but this acquisition map was incomplete. Committing to a known target 104 units
  away excluded a lucky/local discovery of an unknown yellow roughly 25 units
  from red. This pair does not support a raw first-antidote speed advantage;
  instead it exposes the next planning problem: arbitrate reliable known-target
  exploitation against local epistemic search when memory coverage is sparse.
  Do not tune a distance threshold to this one pair; replicate or preregister an
  EFE-style utility comparison before changing the controller.
- Preregistered pair 2 with both start and run order changed: seed 165 begins
  8.04 units short of the independently acquired red region
  `(82.94, -181.17)`, whose nearest stored yellow is `(116.80, -174.35)` at
  34.54 units from red and initially outside the 16-unit sensor radius. To
  counterbalance pair 1's treatment-first order, pair 2 runs the empty-memory
  control first and the frozen-memory treatment second. All other controller,
  verified-rule, metabolic, and duration settings remain fixed.
- Pair-2 empty-memory control completed with zero initial typed regions, zero
  typed recommendations, and zero protective-memory frames. Red was collected
  at 2.96 seconds and an unaided yellow at `(87.67, -246.21)` at 57.52 seconds,
  giving a 54.56-second red-to-antidote latency. Direct visual guidance occurred
  only after chance discovery; the run had three stuck events, zero survival
  failures, one cancelled hazard, and zero applied hazard cost.
- Prepared the matching seed-165 treatment with the frozen seed-163 acquisition
  memory restored and every other setting unchanged. This completes the planned
  second pair; further unaided seeds are deferred unless later replication is
  scientifically necessary.
- Seed-165 memory treatment completed successfully. Red was collected at 2.96
  seconds with the nearest remembered yellow invisible and 36.85 units away.
  Typed recall remained active for 66 pre-visibility frames, changed 49 MPC
  actions, and closed to 14.59 units before direct visual handoff. Yellow was
  collected at 21.92 seconds: 18.96 seconds after red versus 54.56 seconds in the
  matched empty-memory control, a 35.60-second (65.2%) improvement. The run later
  completed a second protected red-to-yellow cycle; total outcome was two hazard
  cancellations, zero metabolic cost, one stuck event, zero survival failures,
  and zero respawns.
- Two-pair counterbalanced descriptive summary: treatment latencies were 43.07
  and 18.96 seconds (mean 31.02); empty controls were 38.55 and 54.56 seconds
  (mean 46.56). Mean paired reduction was 15.54 seconds, or 33.4% relative to
  control. Treatment won the first-antidote metric in one of two pairs because
  pair 1's empty controller luckily found a closer unmemorized yellow. Memory
  reached its selected known target substantially sooner in both pairs (pair 1:
  43.07 versus 92.65 seconds; pair 2 control never collected the selected target
  within the bounded run).
- This small sample establishes functioning pre-visibility typed recall and a
  promising descriptive efficiency/reliability effect, not population-level
  statistical significance. Per user direction, retain episodic memory as the
  active architecture and stop additional unaided runs for now; chance yellow
  encounters remain valid opportunistic successes rather than controller errors.
- Consolidated the milestone in
  `TYPED_EPISODIC_PROTECTIVE_RECALL_RESULT_20260819.md` and the compact machine-
  readable `outputs/typed_resource_recall_paired_summary_20260819.json`. Retained
  the four final paired raw logs locally with recorded SHA-256 hashes. Deleted
  45 obsolete ignored Unity JSONL/JSONL.GZ recordings after their outcomes had
  been summarized, reducing `outputs/` from approximately 2.5 GB to 519 MB and
  reclaiming about 2.0 GB. No code, checkpoints, compact results, Unity assets,
  frozen memories, or final paired telemetry were removed.
- Published the accumulated embodied Tiny Scientist, typed-interaction, and
  protective-recall milestone to GitHub branch
  `codex/tiny-scientist-neurosymbolic` as commit `ae034f6`. The publication
  included 204 bounded code, protocol, checkpoint, and compact-result files;
  large LoRA caches, raw recordings, the local website tree, and temporary
  files remained excluded.
- Began the next adaptive-memory step without tuning against the two recall
  pairs. Audit found that generic resource recall suppressed a remembered
  location after an empty arrival, but typed protective recall did not record
  that counterfactual failure. Typed recall now suppresses an absent requested
  resource for the existing bounded refractory interval, increments and
  persists its failure evidence, and immediately falls through to the next
  matching typed memory when one exists. A pickup-frame guard prevents a newly
  collected requested resource from being misclassified as stale.
- Added explicit typed-stale telemetry and three discoverable regression tests:
  empty typed arrival/failure persistence, same-frame fallback to a second
  typed location, and pickup-frame non-penalization. The focused planner/memory/
  embodiment suite passed 70/70; the full Miniforge suite passed 289/289 in
  7.053 seconds. No Unity run was required for this bookkeeping/controller
  invariant; the next physical smoke should deliberately remove one remembered
  yellow and verify abandonment plus fallback to another memory.
- Prepared the bounded seed-166 stale-memory fallback launcher. It copies the
  frozen seed-163 acquisition map, adds a single answer-neutral decoy yellow
  memory 10 units north of the known red pickup, and preserves all real yellow
  memories. After red, the decoy must win by proximity; the preregistered result
  is one typed stale-arrival event followed by selection of a different yellow
  memory, with the original acquisition artifact left unchanged. The run is
  bounded to 300 seconds and writes fresh 20260820 memory and telemetry files.
- Seed-166 completed its normal 299.553-second bound with 1,426 telemetry rows.
  Red was collected at 3.39 seconds and immediately selected the injected decoy
  yellow memory. At 4.87 seconds the agent arrived without seeing yellow,
  recorded exactly one typed stale arrival, persisted `failures: 1` on the
  decoy, and selected the genuine `(116.8036, -174.3526)` yellow memory in the
  same frame while verified protection remained active.
- The fallback memory controlled all 15 frames until a closer, unmemorized
  yellow entered direct perception at 7.85 seconds, including 13 additional
  protective MPC action changes. Perception correctly superseded memory and the
  agent collected that opportunistic yellow at 12.94 seconds, 9.55 seconds after
  red. The hazard was cancelled with zero hunger-cost events, zero survival
  failures, and zero respawns; three stuck events recovered normally.
- Verdict: stale typed recall, persistent counterfactual failure evidence,
  same-frame fallback selection, and memory-to-perception handoff passed. The
  run does not prove completion of the second remembered route during the active
  hazard because the closer chance antidote validly resolved the need first.
  The frozen acquisition memory remained unchanged at SHA-256
  `a47ecc470aff22d6cbb736ebdbc4e32e6e21d8d601b3be8ca62beba7e8df38fe`.
  Detailed and compact records are in
  `TYPED_STALE_MEMORY_FALLBACK_RESULT_20260820.md` and
  `outputs/typed_resource_stale_fallback_summary_20260820.json`.
- Prepared the closing seed-167 physical fallback-completion run. It uses the
  counterbalanced seed-165 start 8.04 units short of the natural red at
  `(82.9424, -181.1689)`, injects a decoy yellow 10 units north at
  `(82.9424, -171.1689)`, and retains the genuine remembered yellow at
  `(116.8036, -174.3526)`, about 34.54 units from red. The run is bounded to
  120 seconds with the existing 240-second pending hazard, so no deadline cost
  can confound route completion. Success requires decoy rejection, selection of
  the genuine memory, pre-visibility PGNW/MPC influence, and physical pickup at
  the genuine region before any opportunistic yellow resolves the need.
- Seed-167 completed its normal 119.425-second bound but failed the preregistered
  completion criterion. Red was collected at 3.01 seconds and the decoy was
  selected with verified protection, but no yellow was collected and no stale
  event fired. The robot's closest decoy approach was 4.868 units at 29.01
  seconds while PGNW guidance and commitment were both active—just outside the
  fixed 4.0-unit arrival radius. The target later changed at 97.31 seconds only
  because the genuine memory became nearer than the decoy, not because the
  decoy was rejected. The run had 420 protective guidance decisions and 185
  action changes, zero stuck events, failures, or respawns, but it is a failed
  calibration run and must not be counted as stale-fallback confirmation.
- Replaced the typed path's exact-coordinate tolerance with an answer-blind
  resolution rule: typed arrival radius is the greater of the legacy radius and
  half the coarse memory cell. With the existing 12-unit cells this is 6 units.
  This parameter is derived from representation resolution rather than the
  observed 4.868-unit miss; seed-167 remains excluded from confirmation and a
  fresh seed is required. Added telemetry for the effective typed radius and
  boundary regressions at 5.5 and 6.5 units.
- The resolution repair passed 72/72 focused memory/planner/embodiment tests and
  the full suite passed 291/291 in 8.705 seconds. Prepared untouched seed 168
  with the same frozen geometry and two-minute bound as the failed calibration;
  it is the first admissible confirmation of the six-unit typed arrival rule.
- Seed-168 passed the complete physical fallback criterion over its normal
  119.389-second bound (566 rows, steps 0--565). Red was collected at 2.96
  seconds; the decoy was detected stale at 6.36 seconds and persisted exactly
  one failure. In that same frame the still-invisible genuine yellow memory at
  `(116.8036, -174.3526)` became the target while verified protection remained
  active.
- Across the 54 frames from fallback selection to direct visibility, typed
  memory remained active for 53 frames, PGNW issued bounded guidance for 44,
  and changed 33 additional MPC actions. The intended yellow entered perception
  at 17.61 seconds and was physically collected at `(116.67, -175.24)` at 22.26
  seconds, 19.299 seconds after red. The pending hazard was cancelled with zero
  cost events, survival failures, or respawns; two stuck events recovered.
- Verdict: one disclosed calibration failure followed by one untouched pass for
  decoy rejection, failure persistence, same-frame fallback, pre-visibility
  control, perceptual handoff, genuine remembered pickup, and metabolic
  resolution. This closes the physical-completion limitation but does not
  estimate population reliability. Detailed and compact records are in
  `TYPED_STALE_MEMORY_COMPLETION_RESULT_20260820.md` and
  `outputs/typed_resource_stale_completion_summary_20260820.json`.

## PGNW actionable multi-hypothesis arbitration — 2026-08-20

- Confirmed that the prior `pgnw_hypothesis_selection_lab.py` already tests
  multi-model experimental design, so the new milestone targets the missing
  layer: competition among actionable typed causal candidates after a pending
  hazard. Implemented a pure answer-blind arbiter over the complete typed
  posterior, formal suppression likelihood, expected information gain,
  metabolic hazard value, route time, memory confidence, and an absolute safety
  gate. The scoring function receives no true suppressor or expected winner.
- Registered a 2 x 2 x 2 counterbalance over dominant causal color, near/far
  route assignment, and candidate presentation order. All 8/8 cases selected
  the posterior-supported candidate; score margins ranged from 0.071292 to
  0.083736. Color-swapping the posterior swapped the decision, order reversal
  left the result unchanged, and an unsafe dominant candidate could not outvote
  the safety gate.
- Replayed the verified seed-168 posterior against the frozen seed-163 terrain
  memories at the post-decoy position. Yellow scored 0.079840 from 0.916019
  expected suppression probability; blue scored 0.006319 from 0.066640 despite
  its larger epistemic value. The result selected yellow using all explicit
  terms rather than MAP text, first position, or a hardcoded color answer.
- Five focused arbitration regressions passed and the full suite passed 296/296
  in 6.850 seconds. Detailed and machine-readable records are in
  `PGNW_MULTI_HYPOTHESIS_ARBITRATION_RESULT_20260820.md` and
  `outputs/pgnw_multi_hypothesis_arbitration_20260820.json`. Claim boundary:
  this is an offline and replay-grounded pass with zero Unity motor authority.
  Next add passive Unity telemetry and compare arbitration recommendations with
  the existing verified-protective controller before considering authority.
- Added an explicitly passive multi-hypothesis shadow to the embodied PGNW
  planner. When a red hazard is pending it scores all yellow/blue typed memories
  from the full live posterior, current memory confidence, route distance, and
  remaining deadline. Telemetry records the selected feature, complete score
  terms, evaluation count, and agreement with the existing verified-protective
  target. Its declared authority is 0.0 and action influence is structurally
  fixed at zero; the existing verified rule remains the sole PGNW target source.
- Added CLI flag `--pgnw-multi-hypothesis-arbitration passive`, deadline plumbing
  from the hidden pending event, and planner/full-ego regressions showing that
  passive arbitration leaves the original guidance vector and weight unchanged.
  A boundary test also confirmed that the same posterior selects yellow with a
  feasible 240-second route but nearer blue with an infeasible 10-second route,
  demonstrating utility arbitration rather than MAP relabeling. The focused
  suite passed 75/75 and the full suite passed 299/299 in 6.808 seconds.
- Prepared a two-minute seed-169 Unity shadow run from the seed-165 red geometry
  with frozen acquisition memory, the 97%+ verified typed posterior, existing
  committed protective control, and passive multi-hypothesis telemetry. The
  preregistered checks are: arbitration activates only after red; it emits both
  candidate records; passive authority/action influence stay exactly zero; its
  selected feature agrees with the existing yellow target while the 240-second
  deadline remains feasible; and controller safety remains bounded.
- Seed-169 passed the passive Unity manipulation check over its normal
  119.406-second bound (567 rows). Arbitration began with red at 2.97 seconds,
  emitted both yellow and blue records for 89 active frames, selected yellow on
  89/89, and agreed with the existing verified target on 89/89. Authority stayed
  exactly 0.0 and action influence exactly zero. Initial yellow score was
  0.079824 versus blue 0.006451. Yellow was collected 18.92 seconds after red;
  the hazard cancelled with zero cost, stuck events, failures, or respawns.
- Implemented `bounded_verified` authority as the only active arbitration mode.
  A winner may own the existing committed PGNW target only when it agrees with
  the >=0.95 specific verified production, expected suppression is >=0.50, the
  score advantage is >=0.02, and downstream MPC safety gates continue to pass.
  Otherwise it abstains with an explicit denial reason. Added telemetry for
  score margin, suppression probability, authority, control frames, denials,
  and arbitration-sourced MPC action changes.
- Prepared the two-minute seed-170 bounded-authority smoke with the same frozen
  start geometry, posterior, memory, deadline, and controller bounds as the
  passive manipulation check. Preregistered success requires nonzero bounded
  arbitration authority/control frames after red, yellow selection with all
  confidence gates satisfied, arbitration-sourced MPC changes, physical yellow
  pickup and hazard cancellation, and zero survival failures or respawns.
- Seed-170 passed every preregistered bounded-authority check over a normal
  119.385-second run (571 rows). Red was collected at 2.95 seconds. Arbitration
  selected yellow, cleared every verification gate, and held authority for all
  90 active frames; its score margin was 0.058344--0.081224 and its expected
  suppression probability was 0.916019. It produced 58 arbitration-sourced MPC
  action changes before yellow was physically collected at 21.95 seconds,
  19.00 seconds after red. The pending hazard cancelled with zero cost events,
  stuck events, survival failures, or respawns.
- Verdict: bounded multi-hypothesis authority is operational and safe in this
  smoke. Because it agreed with the previous verified-protective controller on
  all 90 frames, this validates authority handoff rather than a superior choice
  under controller disagreement. The next discriminating milestone needs two
  independently verified actionable rules capable of recommending different
  safe targets. Detailed records are in
  `PGNW_BOUNDED_ARBITRATION_UNITY_RESULT_20260820.md` and
  `outputs/pgnw_bounded_arbitration_seed170_summary_20260820.json`.
- Post-run verification passed 77/77 focused arbitration/planner/embodiment
  regressions and the full suite passed 301/301 in 6.832 seconds. The first
  attempted test invocation used Apple's bare `python3` and could not import
  NumPy; rerunning with the project's Miniforge `python` completed cleanly.

## Language classification and a second verified rule — 2026-08-20

- Reviewed Barry Smith's language-as-classifier proposal as a functional design
  hypothesis rather than a consciousness test. The implementation target is to
  classify stabilized causal content as reportable roles such as `antidote` and
  `food`, while keeping labels observational until an ablation demonstrates a
  benefit. Jaynes-style split control was deferred because it would add another
  controller without a preregistered performance advantage.
- Found a necessary environmental confound before live integration: the current
  Python metabolism reduces hunger after every pickup, so telemetry cannot yet
  independently identify blue as nutritional. Added a symmetric Bayesian
  metabolic-role learner with separate admission and post-cutoff held-out
  verification. The learner receives only pickup identity and observed relief;
  it is not given the correct nutrient.
- Extended the offline PGNW arbiter to combine independently computed protective
  and metabolic values. Meaningful, anonymous, and shuffled language labels are
  attached only after scoring and cannot alter selection or authority. This
  establishes the non-leaking scaffold needed to test whether language labels
  improve a downstream language-mediated decision rather than granting an
  improvement by construction.
- Froze and ran a 24-case Gemma 3 1B language-role ablation balancing protective
  color, candidate order, current need, and meaningful/anonymous/shuffled role
  vocabulary. All modes carried equivalent explicit definitions and no prompt
  contained its expected answer. Meaningful labels scored 5/8, anonymous labels
  scored 0/8 under the exact output contract, and shuffled labels scored 4/8;
  overall accuracy was 9/24 in 18.821 seconds.
- The meaningful result is not distinguishable from the 4/8 chance expectation
  (one-sided binomial probability for >=5/8: 0.363). Gemma selected B in 7/8
  meaningful cases. Seven anonymous outputs violated the frozen contract by
  emitting `CANDIDATE B`; interpreting those post hoc still yields only 4/8.
  Shuffled trials selected the lexical label `antidote` in all 8 cases despite
  its reversed supplied definition. Verdict: language labels affect the model
  but did not improve arbitration, so they remain passive and receive no control
  authority. Detailed results are in
  `SMITH_LANGUAGE_CLASSIFIER_ABLATION_RESULT_20260820.md` and
  `outputs/smith_language_classifier_ablation_20260820.json`.
- The useful independent result is retained: the answer-blind metabolic learner
  admits a color-specific nutrient only above 0.95 posterior confidence and
  promotes it only after post-cutoff positive and negative confirmations. In
  focused dual-rule arbitration, hazard urgency selects the verified protective
  target while hunger urgency selects the independently learned nutrient, using
  only existing yellow and blue objects. Fourteen focused tests and the full
  306-test suite passed; live metabolic observation remains the next gate.
- Added an explicitly opt-in live metabolic-role world. With
  `--typed-metabolic-role-learning`, only blue produces immediate hunger relief;
  red and yellow retain their dopamine and established causal roles but do not
  reset nutritional urgency. Unambiguous pickup identity and the observed
  before/after hunger delta are sent to the symmetric learner. The hidden correct
  feature is never passed to its posterior update, and its JSON audit is persisted
  independently from the established ordered-interaction memory.
- Added telemetry and CLI plumbing plus an embodiment regression reproducing the
  preregistered sequence: four calibration observations admit blue above 0.95,
  then a post-cutoff blue relief and non-blue non-relief promote the rule to
  `verified_held_out`. Twenty focused metabolic/embodiment tests passed and the
  full suite passed 310/310 in 7.147 seconds.
- Prepared a fresh three-minute seed-171 Unity run using the seed-170 start area,
  frozen resource acquisition memory, existing verified yellow protection, and
  passive metabolic-role learning. Success requires blue admission above 0.95,
  post-cutoff positive and negative confirmation with zero contradictions, and
  bounded controller safety. The language labels remain passive and are not a
  tested source of motor influence.
- Seed-171 passed the live metabolic-role gate over a normal 179.429-second run
  (851 rows). Seven clean single-pickup transitions were accepted. Red and
  yellow produced no hunger relief, while blue produced relief. The sequence
  admitted `blue_restores_metabolic_reserve` after observation 4 at 32.341
  seconds with 0.975865 posterior confidence.
- Post-cutoff yellow/no-relief at 93.207 seconds supplied the negative control;
  blue/relief at 129.360 seconds supplied the positive confirmation and promoted
  the rule to `verified_held_out`. A later blue relief added a third confirmation.
  Final status was 3 held-out confirmations, 0 contradictions, 2 positive blue
  observations, 1 negative non-blue observation, and 0.997130 confidence.
- Existing protection remained intact: red was collected at 2.999 seconds,
  yellow at 21.502 seconds, and the pending hazard cancelled without cost. Two
  stuck events recovered with zero survival failures or respawns. Verdict: the
  system now has a second independently admitted and held-out verified rule over
  existing objects. Both the metabolic rule and failed Smith-inspired language
  labels remained passive, so real yellow-versus-blue need conflict is the next
  test. Detailed records are in `TYPED_METABOLIC_ROLE_UNITY_RESULT_20260820.md`
  and `outputs/typed_metabolic_role_seed171_summary_20260820.json`.

## Verified yellow-versus-blue PGNW conflict — 2026-08-20

- Added read-only loading of the seed-171 held-out verified metabolic checkpoint
  into a distinct fresh writable run memory. Invalid posterior mass, mismatched
  hypotheses, non-verified status, or reuse of the discovery path is rejected.
- Added `bounded_dual_verified` arbitration. It is eligible only while a verified
  red hazard is pending and normalized hunger urgency is above zero. Candidate
  score combines the existing yellow suppression posterior with the independently
  verified blue relief posterior, memory confidence, route time, hazard value,
  and hunger urgency. A winning target receives authority only if its own causal
  production is verified and the common score-margin/safety gates pass.
- Registered hunger urgency as `clamp((hunger - 0.65) / 0.35)`. Thus critical
  hunger can make blue outvote yellow while the 240-second hazard deadline is
  generous; one blue relief should drop urgency below the gate, automatically
  returning the still-pending hazard to verified yellow protection. Frozen-memory
  replay at the seed-172 start selected blue at hunger 0.85--0.95 and yellow at
  hunger <=0.65 without changing either causal posterior.
- Added direct typed-memory targeting for an arbitration winner, avoiding a
  hidden dependency on the resource controller's prior yellow request. Focused
  tests cover verified checkpoint loading and the blue-then-yellow authority
  transition. The full suite passed 312/312 in 7.138 seconds.
- Prepared a four-minute seed-172 Unity conflict smoke with initial hunger 0.92,
  the existing red start, frozen yellow and blue terrain memories, both verified
  causal productions, and bounded dual authority. Preregistered success requires
  red to create the pending hazard; dual arbitration to select and physically
  reach blue first under high hunger; hunger relief to end blue urgency; control
  to return to yellow; physical yellow cancellation before the deadline; and no
  survival failure or respawn. Language labels remain absent from scoring.
- Seed-172 completed its normal 239.406-second bound (1,133 rows) but failed the
  preregistered physical conflict sequence. Red was collected at 2.971 seconds
  with hunger 0.934. PGNW immediately selected verified blue with full authority
  for 80 frames, but changed zero MPC actions during that first conflict. The
  unguided trajectory intercepted yellow at 19.872 seconds, cancelling the
  hazard, before reaching blue at 23.650 seconds.
- A second red at 130.831 seconds exposed genuine score competition later in the
  run: rising hunger moved arbitration from yellow authority through 29
  low-margin abstention frames to blue authority at 178.888 seconds. This phase
  produced action changes, but no further target was physically reached before
  the bound and the second hazard remained pending. Whole-run totals were 432
  arbitration frames, 403 authority frames, and 107 arbitration-sourced changes,
  with one recovered stuck event and zero costs, failures, or respawns.
- Verdict: disclosed calibration failure. The rules disagreed and authority
  transferred safely, but the first selected blue target had no causal motor
  influence and nearby yellow intercepted the body. Seed 172 is excluded from
  confirmation. Detailed records are in
  `PGNW_DUAL_CONFLICT_CALIBRATION_RESULT_20260820.md` and
  `outputs/pgnw_dual_conflict_seed172_summary_20260820.json`.
- Registered seed 173 as the single-parameter motor calibration after seed 172.
  It preserves the same start geometry, initial hunger 0.92, frozen protective
  and metabolic memories, scoring function, target distances, 240-second hazard,
  and four-minute bound. The only change is committed MPC max-score regret from
  0.30 to 0.45, still inside the existing collision-masked action set.
- Seed-173 success remains the original physical sequence: red creates both
  needs; verified blue wins and produces nonzero pre-pickup MPC influence; blue
  is physically reached before yellow; relief closes hunger urgency; verified
  yellow then owns guidance and physically cancels the same pending hazard; no
  survival failure or respawn occurs. Merely increasing authority telemetry or
  eventually eating both colors is not sufficient.
- Seed-173 completed normally (239.452 seconds, 1,137 rows) but repeated the
  seed-172 failure exactly. Red arrived at 2.781 seconds with hunger 0.934;
  verified blue held authority for 84 frames yet produced zero MPC changes.
  Yellow cancelled the hazard at 20.541 seconds and blue followed at 24.344.
  Raising max-score regret to 0.45 therefore had no observable effect.
- Diagnosed the upstream cause: `pgnw_experiment_guidance_allowed` categorically
  blocked PGNW motor guidance at hunger >=0.92. The protocol intentionally
  creates critical hunger to make the verified metabolic rule relevant, so the
  safety gate made symbolic blue authority behaviorally inert. This explains
  both zero-influence first conflicts without invoking insufficient weights.
- Added a narrow verified-rescue exception: the hunger gate alone may yield when
  bounded dual arbitration owns the target, that target exactly matches the
  held-out verified nutrient, and the metabolic learner is still verified. All
  fallback, stuck, AIR, resource, collision-mask, and MPC-regret gates remain.
  Tests confirm critical-hunger rescue is allowed while every other safety gate
  still vetoes it; the full suite passed 312/312 in 6.506 seconds.
- Prepared fresh seed 174 with the same frozen geometry, needs, memories, and
  original 0.30 regret bound. Thus the verified-rescue gate repair is the only
  operative change from seed 172. The physical preregistration is unchanged.
- Seed 174 completed normally (239.468 seconds, 1,138 rows) and validated the
  repaired motor path, but did not pass the complete preregistered physical
  sequence. Red was collected at 2.773 seconds with hunger 0.934. Verified blue
  immediately held full arbitration authority for 91 frames and changed 42 MPC
  actions before the first competing pickup, versus zero changes in seeds 172
  and 173. This confirms that the critical-hunger gate repair made the verified
  metabolic production behaviorally operative.
- The fixed start geometry still caused yellow to be contacted first at 21.903
  seconds. Immediately before contact, the selected blue target was 9.45 m away
  while a visible yellow was only 2.58 m away. Yellow cancelled the pending
  hazard; blue followed at 24.874 seconds, 2.971 seconds later. Because the
  hazard was already closed, the required blue-relief-to-yellow authority
  transfer was not exercised. One later stuck event recovered, with zero hazard
  costs, survival failures, or respawns.
- Verdict: `partial_mechanism_pass_physical_sequence_fail`. Seed 174 establishes
  causal pre-pickup motor influence from the verified blue rule, but it does not
  establish successful multi-hypothesis sequencing. A future confirmation must
  control target geometry so neither pickup lies on the route to the other,
  preregister that geometry independently of the outcome, and retain the same
  frozen rules and controller parameters.

## Geometry-controlled dual-conflict confirmation — 2026-08-21

- The user arranged a saved spawn-area fork with red at `(-5.62, 4.53)`, blue
  at `(-12.0, 6.918891)`, and yellow at `(-1.06, 9.86)` in X/Z. Independent
  scene inspection confirmed matched red-to-target distances (6.81 m blue and
  7.01 m yellow), 11.33 m target separation, and an approximately 110-degree
  fork. The saved scene hash is frozen in the seed-175 launcher.
- Added an answer-blind resource fixture with only those blue and yellow
  coordinates, equal one-observation confidence, and no causal outcomes. This
  prevents older scattered terrain memories from silently selecting a different
  target. Seed 175 starts from the saved pose without diagnostic teleportation.
- Preregistered a 90-second confirmation with the same held-out verified rules,
  arbitration calculation, 0.30 MPC regret bound, critical hunger, controller,
  and safety gates. A pass requires physical red -> blue -> yellow order, causal
  MPC influence before blue and after its hunger relief, cancellation of the
  same pending hazard, and no cost, survival failure, or respawn. Symbolic
  authority alone remains insufficient.
- Seed 175 completed normally (89.927 seconds, 429 rows) but failed the physical
  sequence. Red arrived at 2.519 seconds with hunger 0.933. Verified blue won
  immediately with full authority and changed nine MPC decisions, yet yellow
  was collected first at 5.481 seconds; blue followed at 8.835. The same order
  repeated after a second red. There were zero costs, stuck events, survival
  failures, or respawns.
- The controlled fork rules out the seed-174 route-overlap explanation. At red,
  blue was 9.03 m away and yellow 8.57 m away. Blue guidance required a strong
  leftward turn, but the 0.30 constrained selector continued forward. At step
  20, blue was nearly directly left at 7.58 m while yellow was 4.30 m ahead/right,
  and the selected action remained forward. Verdict:
  `motor_authority_calibration_fail`; blue selection was causally operative but
  insufficient to realize its target.
- Preregistered seed 176 as a single-parameter calibration on the identical
  saved scene and memories: max-score regret 0.30 -> 0.45. This is the value
  attempted in seed 173 but never exercised because its upstream hunger gate
  blocked motor guidance. The 45-second bound covers the first fork episode;
  the complete physical success criterion remains unchanged.
- Seed 176 completed normally (44.908 seconds, 215 rows) and achieved the safe
  physical order for the first time: red at 2.516 seconds, blue at 4.621, and
  yellow at 7.585. Blue relief reduced hunger to 0.603; verified yellow then
  changed seven MPC actions and cancelled the same hazard. There were zero
  costs, stuck events, survival failures, or respawns.
- The strict preregistration nevertheless failed. Blue held full verified
  authority for nine frames but produced zero counterfactual action changes
  because the unguided MPC path was already aligned. More importantly,
  arbitration switched to yellow at 4.409 seconds, 0.212 seconds before the
  physical blue pickup and relief. The typed resource-memory selector drops a
  candidate inside its generic four-meter arrival radius, so proximity—not the
  registered metabolic outcome—caused that transfer.
- Verdict: `physical_sequence_pass_causal_preregistration_fail`. Across seeds
  175 and 176, blue has separately shown verified selection, causal MPC influence,
  and physical precedence over yellow, but no single run yet demonstrates the
  full causal chain with a strictly post-relief transfer. The next narrow repair
  is to retain a visible, selected arbitration target until pickup confirmation
  instead of treating memory-radius arrival as completion.
