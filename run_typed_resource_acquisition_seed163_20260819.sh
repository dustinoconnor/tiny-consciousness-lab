#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

resource_memory="outputs/resource_memory_typed_acquisition_seed163_20260819.json"
run_log="outputs/unity_shadow/typed_resource_acquisition_seed163_20260819.jsonl"

if [[ -e "$resource_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: seed-163 acquisition output already exists."
  exit 2
fi

# This is an answer-neutral acquisition phase: all observed pickup types are
# encoded, while resource-memory control remains passive.
exec caffeinate -dimsu python embodied_unity_loop.py \
  --duration 1200 \
  --seed 163 \
  --shadow-policy checkpoints/unity_geometry_posttrained/best.pt \
  --shadow-control terrain \
  --shadow-control-confidence 0.0 \
  --shadow-mpc \
  --orbit-exit-adapter checkpoints/orbit_exit_adapter/best.pt \
  --systemic-conductor-checkpoint checkpoints/four_context_conductor/best.json \
  --systemic-conductor-control recurrent_mpc_air \
  --adaptive-gnw-control bounded \
  --terrain-resource-memory "$resource_memory" \
  --terrain-resource-control passive \
  --shadow-log "$run_log"
