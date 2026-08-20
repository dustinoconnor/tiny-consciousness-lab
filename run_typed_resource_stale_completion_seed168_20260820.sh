#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

source_memory="outputs/resource_memory_typed_acquisition_seed163_20260819.json"
run_memory="outputs/resource_memory_stale_completion_seed168_20260820.json"
typed_memory="outputs/typed_interaction_stale_completion_seed168_20260820.json"
run_log="outputs/unity_shadow/typed_resource_stale_completion_seed168_20260820.jsonl"

if [[ ! -f "$source_memory" ]]; then
  print -u2 "Missing frozen seed-163 acquisition memory."
  exit 2
fi
if [[ -e "$run_memory" || -e "$typed_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: seed-168 stale-completion output exists."
  exit 2
fi

cp "$source_memory" "$run_memory"
STALE_MEMORY_PATH="$run_memory" python3 -c '
import json, os
from pathlib import Path

path = Path(os.environ["STALE_MEMORY_PATH"])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["typed_regions"].append({
    "feature": "yellow",
    "x": 82.9424,
    "z": -171.1689,
    "rewards": 1,
    "failures": 0,
})
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
'

exec caffeinate -dimsu python3 embodied_unity_loop.py \
  --duration 120 \
  --seed 168 \
  --diagnostic-teleport 74.90 -181.17 \
  --shadow-policy checkpoints/unity_geometry_posttrained/best.pt \
  --shadow-control terrain \
  --shadow-control-confidence 0.0 \
  --shadow-mpc \
  --orbit-exit-adapter checkpoints/orbit_exit_adapter/best.pt \
  --systemic-conductor-checkpoint checkpoints/four_context_conductor/best.json \
  --systemic-conductor-control recurrent_mpc_air \
  --adaptive-gnw-control bounded \
  --tiny-scientist-experiment-control committed \
  --tiny-scientist-experiment-commit-seconds 6.0 \
  --tiny-scientist-experiment-max-score-regret 0.30 \
  --causal-probe-delay-seconds 240 \
  --causal-probe-cancellation-feature yellow \
  --causal-probe-hunger-cost 0.25 \
  --typed-interaction-learning \
  --typed-interaction-discovery-memory outputs/typed_interaction_formulation_memory_seed161_20260817.json \
  --typed-interaction-memory "$typed_memory" \
  --typed-interaction-rule-control verified_protective \
  --terrain-resource-memory "$run_memory" \
  --terrain-resource-control guided \
  --shadow-log "$run_log"
