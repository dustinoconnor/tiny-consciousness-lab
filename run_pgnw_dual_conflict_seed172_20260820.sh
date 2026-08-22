#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

source_resource="outputs/resource_memory_typed_acquisition_seed163_20260819.json"
source_metabolic="outputs/typed_metabolic_role_seed171_20260820.json"
run_resource="outputs/resource_memory_pgnw_dual_conflict_seed172_20260820.json"
typed_memory="outputs/typed_interaction_pgnw_dual_conflict_seed172_20260820.json"
metabolic_memory="outputs/typed_metabolic_role_pgnw_dual_conflict_seed172_20260820.json"
run_log="outputs/unity_shadow/pgnw_dual_conflict_seed172_20260820.jsonl"

for required in "$source_resource" "$source_metabolic"; do
  if [[ ! -f "$required" ]]; then
    print -u2 "Missing frozen source: $required"
    exit 2
  fi
done
if [[ -e "$run_resource" || -e "$typed_memory" || -e "$metabolic_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: seed-172 dual-conflict output exists."
  exit 2
fi

cp "$source_resource" "$run_resource"

exec caffeinate -dimsu python3 embodied_unity_loop.py \
  --duration 240 \
  --seed 172 \
  --initial-hunger 0.92 \
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
  --typed-metabolic-role-learning \
  --typed-metabolic-role-discovery-memory "$source_metabolic" \
  --typed-metabolic-role-memory "$metabolic_memory" \
  --pgnw-multi-hypothesis-arbitration bounded_dual_verified \
  --terrain-resource-memory "$run_resource" \
  --terrain-resource-control guided \
  --shadow-log "$run_log"
