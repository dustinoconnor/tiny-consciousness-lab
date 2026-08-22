#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

source_memory="outputs/resource_memory_typed_acquisition_seed163_20260819.json"
run_memory="outputs/resource_memory_recall_seed164_persistent_20260819.json"
typed_memory="outputs/typed_interaction_recall_seed164_persistent_20260819.json"
run_log="outputs/unity_shadow/typed_resource_recall_seed164_persistent_20260819.jsonl"

if [[ ! -f "$source_memory" ]]; then
  print -u2 "Missing frozen seed-163 acquisition memory."
  exit 2
fi
if [[ -e "$run_memory" || -e "$typed_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: persistent-recall output already exists."
  exit 2
fi

cp "$source_memory" "$run_memory"

exec caffeinate -dimsu python embodied_unity_loop.py \
  --duration 300 \
  --seed 164 \
  --diagnostic-teleport 64.50 -82.33 \
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
