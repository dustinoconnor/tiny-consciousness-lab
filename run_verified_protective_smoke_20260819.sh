#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

run_memory="outputs/typed_interaction_protective_seed162_20260819.json"
run_log="outputs/unity_shadow/verified_protective_seed162_20260819.jsonl"

if [[ -e "$run_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: seed-162 protective output already exists."
  exit 2
fi

exec python embodied_unity_loop.py \
  --duration 180 \
  --seed 162 \
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
  --causal-probe-delay-seconds 60 \
  --causal-probe-cancellation-feature yellow \
  --causal-probe-hunger-cost 0.25 \
  --typed-interaction-learning \
  --typed-interaction-discovery-memory outputs/typed_interaction_formulation_memory_seed161_20260817.json \
  --typed-interaction-memory "$run_memory" \
  --typed-interaction-rule-control verified_protective \
  --terrain-resource-memory outputs/resource_memory_protective_seed162_20260819.json \
  --terrain-resource-control guided \
  --shadow-log "$run_log"
