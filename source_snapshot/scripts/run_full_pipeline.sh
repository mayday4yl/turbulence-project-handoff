#!/usr/bin/env bash
set -euo pipefail

project=/data0/yanglin/physical_bridge
venv=/data0/yanglin/envs/physical_bridge
dataset=/data2/yanglin/physical_bridge/data/places_p2s_bridge_v01
mini=/data2/yanglin/physical_bridge/runs/preflight/mini32
run_id=bridge_v01_seed20260826
run_dir=/data2/yanglin/physical_bridge/runs/$run_id
source_archives=/data2/yanglin/physical_bridge/data/source_archives
pipeline_log="$run_dir/pipeline.log"
stage=initializing

mkdir -p "$run_dir"
exec 9>"$run_dir/pipeline.lock"
if ! flock -n 9; then
  printf '%s pipeline already active\n' "$(date -Is)" >>"$pipeline_log"
  exit 0
fi

if [[ -f "$run_dir/PIPELINE_FAILED" ]]; then
  mv "$run_dir/PIPELINE_FAILED" \
    "$run_dir/PIPELINE_FAILED.$(date +%Y%m%dT%H%M%S)"
fi

on_exit() {
  status=$?
  if [[ "$status" -ne 0 ]]; then
    printf '%s stage=%s exit=%s\n' "$(date -Is)" "$stage" "$status" \
      >"$run_dir/PIPELINE_FAILED.tmp"
    mv "$run_dir/PIPELINE_FAILED.tmp" "$run_dir/PIPELINE_FAILED"
    printf '%s FAILED stage=%s exit=%s\n' "$(date -Is)" "$stage" "$status" \
      >>"$pipeline_log"
  fi
}
trap on_exit EXIT

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source /usr/local/Ascend/cann-8.5.2/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_HOME=/data1/yanglin/physical_bridge/cache/torch
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

log_stage() {
  stage="$1"
  printf '%s stage=%s\n' "$(date -Is)" "$stage" >>"$pipeline_log"
}

check_storage() {
  root_free=$(df --output=avail -B1 / | tail -n 1 | tr -d ' ')
  data0_free=$(df --output=avail -B1 /data0 | tail -n 1 | tr -d ' ')
  data1_free=$(df --output=avail -B1 /data1 | tail -n 1 | tr -d ' ')
  data2_free=$(df --output=avail -B1 /data2 | tail -n 1 | tr -d ' ')
  if [[ "$root_free" -lt 107374182400 ]]; then
    printf 'Safety stop: root free bytes %s\n' "$root_free" >&2
    return 1
  fi
  for value in "$data0_free" "$data1_free" "$data2_free"; do
    if [[ "$value" -lt 1099511627776 ]]; then
      printf 'Safety stop: data disk free bytes %s\n' "$value" >&2
      return 1
    fi
  done
  project_bytes=$(du -sb /data0/yanglin/physical_bridge /data1/yanglin/physical_bridge \
    /data2/yanglin/physical_bridge 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [[ "$project_bytes" -ge 2199023255552 ]]; then
    printf 'Safety stop: project bytes %s\n' "$project_bytes" >&2
    return 1
  fi
}

cards_free() {
  snapshot=$(npu-smi info)
  printf '%s\n' "$snapshot" >"$run_dir/npu_snapshot_latest.txt.tmp"
  mv "$run_dir/npu_snapshot_latest.txt.tmp" "$run_dir/npu_snapshot_latest.txt"
  for card in 4 5 6 7; do
    if ! grep -q "No running processes found in NPU $card" <<<"$snapshot"; then
      return 1
    fi
  done
}

wait_for_cards() {
  while ! cards_free; do
    printf '%s waiting_for_free_npu_4_7\n' "$(date -Is)" >>"$pipeline_log"
    sleep 30
  done
}

ensure_download() {
  artifact="$1"
  filename="$2"
  expected="$3"
  while true; do
    if [[ -f "$source_archives/$filename" ]] \
      && [[ "$(stat -c %s "$source_archives/$filename")" -eq "$expected" ]] \
      && [[ -f "$source_archives/$filename.complete" ]]; then
      break
    fi
    if ! pgrep -f "download_places365_parallel.py $artifact " >/dev/null; then
      "$venv/bin/python" "$project/scripts/download_places365_parallel.py" "$artifact" \
        --destination "$source_archives" --workers 8 \
        >>"$run_dir/download_${artifact}.log" 2>&1 &
    fi
    sleep 30
  done
}

run_generation_split() {
  split_name="$1"
  manifest_path="$2"
  output_root="$3"
  temporary_root="$4"
  shard_size="$5"
  check_storage
  wait_for_cards
  worker_pids=()
  for rank in 0 1 2 3; do
    physical_card=$((rank + 4))
    ASCEND_RT_VISIBLE_DEVICES="$physical_card" "$venv/bin/python" \
      "$project/scripts/generate_dataset.py" \
      --dataset-config "$project/configs/dataset_v01.yaml" \
      --simulator-config "$project/configs/simulator_v01.yaml" \
      --manifest "$manifest_path" --split "$split_name" \
      --output-root "$output_root" --temporary-root "$temporary_root" \
      --shard-size "$shard_size" --rank "$rank" --world-size 4 --device npu:0 \
      --simulator-source /data0/yanglin/vendor/PiRN/assets/simulator_new.py \
      --p2s-checkpoint /data0/yanglin/vendor/PiRN/assets/utils/P2S_model.pt \
      >>"$run_dir/generate_${split_name}_rank${rank}.log" 2>&1 &
    worker_pids+=("$!")
  done
  worker_status=0
  for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
      worker_status=1
    fi
  done
  return "$worker_status"
}

cd "$project"
log_stage download_official_places365
mkdir -p "$source_archives"
ensure_download filelist filelist_places365-standard.tar 67498496
ensure_download images val_256.tar 525158400

log_stage freeze_source_and_trajectory_manifests
if [[ ! -f "$dataset/manifests/SPLIT_FROZEN" ]]; then
  "$venv/bin/python" scripts/build_dataset_manifest.py \
    --config configs/dataset_v01.yaml >>"$run_dir/build_manifest.log" 2>&1
fi

log_stage mini32_generation
if [[ ! -f "$mini/manifests/planned_train.jsonl" ]]; then
  "$venv/bin/python" scripts/build_preflight_plan.py \
    --planned-train "$dataset/manifests/planned_train.jsonl" --output-root "$mini" \
    >>"$run_dir/build_mini_plan.log" 2>&1
fi
if [[ ! -f "$mini/DATASET_COMPLETE" ]]; then
  run_generation_split train "$mini/manifests/planned_train.jsonl" "$mini" \
    /data0/yanglin/tmp/physical_bridge/mini32 8
  "$venv/bin/python" scripts/finalize_dataset.py \
    --dataset-config configs/dataset_v01.yaml --output-root "$mini" \
    --planned-root "$mini/manifests" --splits train --shard-size 8 \
    >>"$run_dir/finalize_mini.log" 2>&1
fi

log_stage mini32_replay_and_model_preflight
if [[ ! -f "$mini/PREFLIGHT_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4 "$venv/bin/python" scripts/dataset_preflight.py \
    --dataset-root "$mini" --split train --simulator-config configs/simulator_v01.yaml \
    --expected-count 32 --device npu:0 --output "$mini/preflight_report.json" \
    >>"$run_dir/mini_preflight.log" 2>&1
fi

log_stage full_dataset_generation
if [[ ! -f "$dataset/DATASET_COMPLETE" ]]; then
  for split_name in train val test; do
    run_generation_split "$split_name" "$dataset/manifests/planned_${split_name}.jsonl" \
      "$dataset" "/data0/yanglin/tmp/physical_bridge/generation" 128
  done
  "$venv/bin/python" scripts/finalize_dataset.py \
    --dataset-config configs/dataset_v01.yaml --output-root "$dataset" \
    --planned-root "$dataset/manifests" --splits train val test --shard-size 128 \
    >>"$run_dir/finalize_dataset.log" 2>&1
fi

log_stage lpips_metric_ready
"$venv/bin/python" scripts/lpips_smoke.py >>"$run_dir/lpips_setup.log" 2>&1

log_stage full_dataset_statistics
statistics="$dataset/statistics"
if [[ ! -f "$statistics/DATASET_STATISTICS_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/dataset_metrics.py --dataset-root "$dataset" \
    --output-dir "$statistics" --preflight-report "$mini/preflight_report.json" \
    >>"$run_dir/dataset_statistics.log" 2>&1
fi

log_stage formal_training
if [[ ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/train.py --train-config configs/train_v01.yaml \
    --simulator-config configs/simulator_v01.yaml --run-id "$run_id" \
    --microbatch 4 --resume auto >>"$run_dir/training.log" 2>&1
fi

log_stage freeze_checkpoint
if [[ ! -f "$run_dir/frozen_checkpoint.json" ]]; then
  "$venv/bin/python" scripts/freeze_selections.py checkpoint --run-dir "$run_dir" \
    >>"$run_dir/freeze_checkpoint.log" 2>&1
fi
checkpoint="$run_dir/checkpoints/frozen_checkpoint.pt"

log_stage validation_refinement_learning_rate
if [[ ! -f "$run_dir/val/refinement_lr/PHASE_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/evaluate_rollouts.py --phase lr --split val \
    --dataset-root "$dataset" --checkpoint "$checkpoint" \
    --train-config configs/train_v01.yaml --simulator-config configs/simulator_v01.yaml \
    --inference-config configs/inference_v01.yaml \
    --output-dir "$run_dir/val/refinement_lr" >>"$run_dir/val_refinement_lr.log" 2>&1
fi
"$venv/bin/python" scripts/freeze_selections.py lr --run-dir "$run_dir" \
  --summary "$run_dir/val/refinement_lr/summary.json" >>"$run_dir/freeze_lr.log" 2>&1

log_stage validation_k
if [[ ! -f "$run_dir/val/k_selection/PHASE_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/evaluate_rollouts.py --phase k --split val \
    --dataset-root "$dataset" --checkpoint "$checkpoint" \
    --train-config configs/train_v01.yaml --simulator-config configs/simulator_v01.yaml \
    --inference-config configs/inference_v01.yaml \
    --frozen-inference "$run_dir/frozen_inference.json" \
    --output-dir "$run_dir/val/k_selection" >>"$run_dir/val_k.log" 2>&1
fi
"$venv/bin/python" scripts/freeze_selections.py k --run-dir "$run_dir" \
  --summary "$run_dir/val/k_selection/summary.json" >>"$run_dir/freeze_k.log" 2>&1

log_stage frozen_test_rollouts
if [[ ! -f "$run_dir/test/rollouts/PHASE_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/evaluate_rollouts.py --phase test --split test \
    --dataset-root "$dataset" --checkpoint "$checkpoint" \
    --train-config configs/train_v01.yaml --simulator-config configs/simulator_v01.yaml \
    --inference-config configs/inference_v01.yaml \
    --frozen-inference "$run_dir/frozen_inference.json" \
    --output-dir "$run_dir/test/rollouts" >>"$run_dir/test_rollouts.log" 2>&1
fi

log_stage frozen_test_teacher_forced_branch
if [[ ! -f "$run_dir/test/teacher_forced_branch/TEACHER_FORCED_COMPLETE" ]]; then
  check_storage
  wait_for_cards
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/teacher_forced_eval.py --split test \
    --dataset-root "$dataset" --checkpoint "$checkpoint" \
    --train-config configs/train_v01.yaml --simulator-config configs/simulator_v01.yaml \
    --output-dir "$run_dir/test/teacher_forced_branch" \
    >>"$run_dir/test_teacher_forced.log" 2>&1
fi

log_stage statistics_and_report
if [[ ! -f "$run_dir/FINAL_REPORT_COMPLETE" ]]; then
  "$venv/bin/python" scripts/analyze_test.py --run-dir "$run_dir" \
    --dataset-root "$dataset" --test-rollout-dir "$run_dir/test/rollouts" \
    --teacher-forced-dir "$run_dir/test/teacher_forced_branch" --seed 20260826 \
    >>"$run_dir/final_analysis.log" 2>&1
fi

stage=complete
printf '%s complete\n' "$(date -Is)" >"$run_dir/PIPELINE_COMPLETE.tmp"
mv "$run_dir/PIPELINE_COMPLETE.tmp" "$run_dir/PIPELINE_COMPLETE"
printf '%s COMPLETE\n' "$(date -Is)" >>"$pipeline_log"
