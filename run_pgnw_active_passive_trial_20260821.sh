#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

trial_id="${1:-}"
case "$trial_id" in
  a1) condition="active"; arbitration="bounded_dual_verified"; controller_seed=181 ;;
  p1) condition="passive"; arbitration="passive"; controller_seed=181 ;;
  p2) condition="passive"; arbitration="passive"; controller_seed=182 ;;
  a2) condition="active"; arbitration="bounded_dual_verified"; controller_seed=182 ;;
  a3) condition="active"; arbitration="bounded_dual_verified"; controller_seed=183 ;;
  p3) condition="passive"; arbitration="passive"; controller_seed=183 ;;
  p4) condition="passive"; arbitration="passive"; controller_seed=184 ;;
  a4) condition="active"; arbitration="bounded_dual_verified"; controller_seed=184 ;;
  *)
    print -u2 "Usage: $0 {a1|p1|p2|a2|a3|p3|p4|a4}"
    exit 2
    ;;
esac

scene_file="/Users/dustinoconnor/My project/Assets/Scenes/ChallengeTerrain.unity"
scene_sha="64efbc407503150f707540e20f4b6ab90179a795229ee9221d19340765cd30dd"
source_resource="pgnw_dual_conflict_spawn_fork_resource_fixture_20260821.json"
source_metabolic="outputs/typed_metabolic_role_seed171_20260820.json"
stem="pgnw_active_passive_${trial_id}_seed${controller_seed}_20260821"
run_resource="outputs/resource_memory_${stem}.json"
typed_memory="outputs/typed_interaction_${stem}.json"
metabolic_memory="outputs/typed_metabolic_role_${stem}.json"
run_log="outputs/unity_shadow/${stem}.jsonl"

for required_file in "$scene_file" "$source_resource" "$source_metabolic"; do
  if [[ ! -f "$required_file" ]]; then
    print -u2 "Missing frozen source: $required_file"
    exit 2
  fi
done
actual_scene_sha="$(shasum -a 256 "$scene_file" | awk '{print $1}')"
if [[ "$actual_scene_sha" != "$scene_sha" ]]; then
  print -u2 "Refusing changed scene: expected $scene_sha, got $actual_scene_sha"
  exit 2
fi
if [[ -e "$run_resource" || -e "$typed_memory" || -e "$metabolic_memory" || -e "$run_log" ]]; then
  print -u2 "Refusing to append: trial $trial_id output exists."
  exit 2
fi

print "PGNW pilot trial=$trial_id condition=$condition seed=$controller_seed duration=20s"
cp "$source_resource" "$run_resource"

exec caffeinate -dimsu python3 embodied_unity_loop.py \
  --duration 20 \
  --seed "$controller_seed" \
  --initial-hunger 0.92 \
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
  --tiny-scientist-experiment-max-score-regret 0.45 \
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
  --pgnw-multi-hypothesis-arbitration "$arbitration" \
  --terrain-resource-memory "$run_resource" \
  --terrain-resource-control guided \
  --shadow-log "$run_log"
