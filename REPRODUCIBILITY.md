# Reproducibility Guide

This guide separates fast stored-result inspection from experiments that train
new checkpoints or require the external Unity runtime.

## Environment

- Python 3.11 or newer
- Unity 6.4 for the minimal trap-course replication
- macOS is required only for the optional IAC MIDI application

Create an isolated Python environment and install the research dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Headline Python experiments

Run the delayed hidden-preference benchmark:

```bash
python delayed_preference_benchmark.py
```

Run the five-seed withheld-topology pipeline:

```bash
python upgraded_foraging_pipeline.py
```

Run the adaptive stochastic MPC comparison from the frozen Unity-calibrated
checkpoint:

```bash
python adaptive_stochastic_mpc_lab.py
```

Each experiment writes a JSON metric artifact under `outputs/`. Training runs
can replace checkpoint files, so preserve the committed checkpoints when doing
an exact stored-result audit.

## Minimal Unity replication

1. Add `unity/TrapCourseLab` to Unity Hub.
2. Open `Assets/Scenes/TrapCourseLab.unity`.
3. Press Play.
4. In a terminal at the repository root, run:

```bash
python embodied_unity_loop.py \
  --shadow-policy checkpoints/unity_mpc/best.pt \
  --shadow-control course \
  --shadow-control-confidence 0.0 \
  --shadow-mpc
```

The Unity project uses generated primitives and contains no NavMesh, terrain
pack, or external art dependency. Raw JSONL telemetry is ignored because long
runs can exceed hundreds of megabytes. Preserve a run explicitly by copying its
compact analyzer output into `outputs/` with a descriptive filename.

## Evidence boundaries

- Read `paper/EVIDENCE_LEDGER.md` before quoting a result.
- The 98.25% result is a five-seed Python withheld-topology benchmark.
- The 12/12 result describes completed episodes in one recorded Unity course
  session, with qualifications documented in the ledger.
- Zero simulator collisions in several experiments include an engineered body-
  clearance mask.
- The project does not establish consciousness, sentience, AGI, or biological
  equivalence.

## Tiny Scientist synthetic-Unity verification

`tiny_scientist.py` analyzes a JSONL Unity recording into color-conditioned
metabolic episodes. In the verified condition, Gemma 3 1B selected a candidate
from a grounded summary while code compiled the falsifier by binding the same
cause; the language model did not gain motor authority.

The compact full-stack result is
`outputs/tiny_scientist_rule_verification_fullstack_summary_20260801.json`.
It reports 10 isolated red pickups, 15 isolated blue controls, a +0.3373 mean
red pressure change, a 10.890-second mean red delay, no blue positive response,
and no red counterexample. The raw recording is intentionally ignored; rerun
the analyzer on a locally collected recording with:

```bash
python3 tiny_scientist.py <recording.jsonl> \
  --backend none \
  --evidence-interface homeostatic_filtered \
  --output outputs/tiny_scientist_evidence.json
```

This verifies a programmed causal contingency within the synthetic Unity
metabolism. It does not establish unaided LLM abduction, autonomous
intervention selection, natural-world causal discovery, consciousness, or live
control authority for the rule.
