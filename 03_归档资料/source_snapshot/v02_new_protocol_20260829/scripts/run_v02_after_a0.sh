#!/usr/bin/env bash
set -euo pipefail

source_root=/data0/yanglin/physical_bridge/v02_new_protocol_20260829
run_root=/data2/yanglin/physical_bridge/runs/v02_new_protocol_20260829
venv=/data0/yanglin/envs/physical_bridge
authorized_cards=0,2,4,6
pipeline_log="$run_root/pipeline.log"
acceleration_patch=/data0/yanglin/physical_bridge/v02_a1_acceleration_patch_20260829

mkdir -p "$run_root"
source /usr/local/Ascend/cann-8.5.2/set_env.sh >/dev/null 2>&1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_HOME=/data1/yanglin/physical_bridge/cache/torch
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

log_stage() {
  printf '%s stage=%s\n' "$(date -Is)" "$1" >>"$pipeline_log"
}

cards_free() {
  for card in 0 2 4 6; do
    if npu-smi info -t proc-mem -i "$card" -c 0 2>/dev/null | grep -q 'Process id:'; then
      return 1
    fi
  done
}

log_stage waiting_for_a0
while [[ ! -f "$run_root/a0/A0_PASS" && ! -f "$run_root/a0/A0_STOP" ]]; do
  if ! pgrep -f 'scripts/v02_gradient_audit.py' >/dev/null; then
    log_stage a0_execution_failed
    printf 'A0 process exited without a scientific marker\n' >"$run_root/a0/A0_EXECUTION_FAILED"
    exit 21
  fi
  sleep 30
done

if [[ -f "$run_root/a0/A0_STOP" ]]; then
  log_stage scientific_stop_a0
  exit 20
fi

log_stage a0_pass_waiting_for_cards
while ! cards_free; do
  sleep 15
done

if [[ ! -f "$acceleration_patch/PATCH_READY" ]]; then
  log_stage a1_acceleration_patch_missing
  printf 'A1 acceleration patch was not frozen before the A0 gate completed.\n' \
    >"$run_root/A1_ACCELERATION_PATCH_MISSING"
  exit 24
fi

log_stage applying_a1_semantics_preserving_acceleration
cp "$acceleration_patch/physical_bridge/v02_losses.py" \
  "$source_root/physical_bridge/v02_losses.py"
cp "$acceleration_patch/physical_bridge/v02_training.py" \
  "$source_root/physical_bridge/v02_training.py"
cp "$acceleration_patch/scripts/v02_train_pilot.py" \
  "$source_root/scripts/v02_train_pilot.py"
cp "$acceleration_patch/tests/test_v02_loss_equivalence.py" \
  "$source_root/tests/test_v02_loss_equivalence.py"
cp "$acceleration_patch/docs/EXECUTION_AMENDMENT_20260829.md" \
  "$source_root/docs/EXECUTION_AMENDMENT_20260829.md"

cd "$source_root"
"$venv/bin/python" -m py_compile \
  physical_bridge/v02_losses.py \
  physical_bridge/v02_training.py \
  scripts/v02_train_pilot.py
"$venv/bin/python" -m pytest -q tests/test_v02_protocol.py tests/test_v02_loss_equivalence.py \
  >"$run_root/a1_acceleration_tests.log" 2>&1

mkdir -p "$run_root/engineering"
selected_microbatch=
for candidate in 4 2 1; do
  probe_dir="$run_root/engineering/a1_microbatch_${candidate}"
  mkdir -p "$probe_dir"
  if [[ -f "$probe_dir/ENGINEERING_PROBE_PASS" ]]; then
    selected_microbatch=$candidate
    break
  fi
  log_stage "a1_microbatch_${candidate}_probe"
  set +e
  ASCEND_RT_VISIBLE_DEVICES="$authorized_cards" "$venv/bin/torchrun" --standalone \
    --nproc-per-node=4 scripts/v02_train_pilot.py \
    --config configs/v02.yaml \
    --manifest "$run_root/manifests/a1/train/batches.jsonl" \
    --a0-report "$run_root/a0/v02_gradient_audit.json" \
    --output-dir "$probe_dir" \
    --microbatch "$candidate" --resume none --engineering-probe-only \
    >>"$probe_dir/probe.log" 2>&1
  probe_status=$?
  set -e
  if [[ $probe_status -eq 0 && -f "$probe_dir/ENGINEERING_PROBE_PASS" ]]; then
    selected_microbatch=$candidate
    break
  fi
  if ! grep -Eiq \
    'out of memory|OutOfMemory|memory allocation|ACL_ERROR_RT_MEMORY_ALLOCATION' \
    "$probe_dir/probe.log"; then
    log_stage "a1_microbatch_${candidate}_probe_non_memory_failure"
    printf 'A1 engineering probe failed for a non-memory reason; see %s\n' "$probe_dir/probe.log" \
      >"$run_root/A1_ENGINEERING_PROBE_FAILED"
    exit 25
  fi
  log_stage "a1_microbatch_${candidate}_probe_oom"
  while ! cards_free; do
    sleep 15
  done
done

if [[ -z "$selected_microbatch" ]]; then
  log_stage a1_all_microbatch_probes_failed
  printf 'No A1 microbatch candidate in 4,2,1 completed one update.\n' \
    >"$run_root/A1_ENGINEERING_PROBE_FAILED"
  exit 26
fi
printf '%s\n' "$selected_microbatch" >"$run_root/engineering/selected_microbatch.txt"
log_stage "a1_selected_microbatch_${selected_microbatch}"

mkdir -p "$run_root/a1"
sha256sum \
  configs/v02.yaml \
  configs/simulator_v01.yaml \
  docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_METHOD.md \
  docs/KNOWN_G_PHYSICAL_TURBULENCE_BRIDGE_V02_TRAINING_PLAN_CODEX.md \
  docs/EXECUTION_AMENDMENT_20260829.md \
  physical_bridge/model.py \
  physical_bridge/simulator.py \
  physical_bridge/v02_protocol.py \
  physical_bridge/v02_data.py \
  physical_bridge/v02_losses.py \
  physical_bridge/v02_training.py \
  scripts/v02_train_pilot.py \
  scripts/v02_evaluate_pilot.py \
  "$run_root/manifests/a1/train/batches.jsonl" \
  "$run_root/manifests/a1/val_subset.json" \
  "$run_root/engineering/selected_microbatch.txt" \
  >"$run_root/a1/source_snapshot.sha256"

log_stage a1_training
ASCEND_RT_VISIBLE_DEVICES="$authorized_cards" "$venv/bin/torchrun" --standalone \
  --nproc-per-node=4 scripts/v02_train_pilot.py \
  --config configs/v02.yaml \
  --manifest "$run_root/manifests/a1/train/batches.jsonl" \
  --a0-report "$run_root/a0/v02_gradient_audit.json" \
  --output-dir "$run_root/a1" \
  --microbatch "$selected_microbatch" --resume auto \
  >>"$run_root/a1/training.log" 2>&1

if [[ ! -f "$run_root/a1/TRAINING_COMPLETE" ]]; then
  log_stage a1_training_failed
  exit 22
fi

log_stage a1_evaluation
ASCEND_RT_VISIBLE_DEVICES="$authorized_cards" "$venv/bin/torchrun" --standalone \
  --nproc-per-node=4 scripts/v02_evaluate_pilot.py \
  --config configs/v02.yaml \
  --pilot-dir "$run_root/a1" \
  --val-subset "$run_root/manifests/a1/val_subset.json" \
  --output-dir "$run_root/a1/eval" \
  >>"$run_root/a1/evaluation.log" 2>&1

if [[ -f "$run_root/a1/eval/A1_PASS" ]]; then
  log_stage a1_pass
  printf 'A1 PASS; Stage B implementation/execution is the next gated action.\n' \
    >"$run_root/A1_PASS_WAITING_FOR_STAGE_B"
else
  log_stage scientific_stop_a1
  printf 'A1 scientific STOP; do not enter Stage B.\n' >"$run_root/SCIENTIFIC_STOP_A1"
fi
