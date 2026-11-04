#!/usr/bin/env bash
# Invoke as: bash scripts/run_env.sh -m torch.distributed.run ...
# One explicitly authorized physical card per independent evaluation worker.
set -euo pipefail
task_v03_root=/data2/yanglin/v031_overfit100_g4321_n4
task_v03_base=/data0/yanglin/physical_bridge
task_v03_python=/data0/yanglin/envs/physical_bridge/bin/python
task_v03_threads="${V03_CPU_THREADS:-4}"
if [[ ! "$task_v03_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo 'V03_CPU_THREADS must be a positive integer' >&2
  exit 2
fi
mkdir -p "$task_v03_root/cache/ascend" "$task_v03_root/cache/work" \
  "$task_v03_root/cache/log" "$task_v03_root/cache/torch" "$task_v03_root/tmp"
task_v03_env_found=0
for task_v03_env_file in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
  if [ -f "$task_v03_env_file" ]; then
    set +u
    source "$task_v03_env_file"
    set -u
    task_v03_env_found=1
    break
  fi
done
if [ "$task_v03_env_found" != 1 ]; then
  echo 'CANN set_env.sh was not found; worker not started' >&2
  exit 2
fi
# CANN contributes the tbe module through PYTHONPATH. Keep that contribution.
export PYTHONPATH="$task_v03_root/code:$task_v03_base:${PYTHONPATH:-}"
task_v031_card="${V031_EVAL_CARD:?A physical evaluation card is required}"
if [[ ! "$task_v031_card" =~ ^[0-3]$ ]]; then
  echo 'Only physical evaluation cards 0 through 3 are authorized' >&2
  exit 2
fi
export ASCEND_RT_VISIBLE_DEVICES="$task_v031_card"
# Keep per-op JIT compilation options on one host dispatch thread. Device
# execution remains asynchronous; this is not global launch-blocking mode.
export TASK_QUEUE_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export TMPDIR="$task_v03_root/tmp" TMP="$task_v03_root/tmp" TEMP="$task_v03_root/tmp"
export XDG_CACHE_HOME="$task_v03_root/cache" TORCH_HOME="$task_v03_root/cache/torch"
export ASCEND_CACHE_PATH="$task_v03_root/cache/eval4_card${task_v031_card}/ascend"
mkdir -p "$ASCEND_CACHE_PATH"
export ASCEND_WORK_PATH="$task_v03_root/cache/work"
export ASCEND_PROCESS_LOG_PATH="$task_v03_root/cache/log"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
unset PYTHONOPTIMIZE
export OMP_NUM_THREADS="$task_v03_threads" MKL_NUM_THREADS="$task_v03_threads"
export OPENBLAS_NUM_THREADS="$task_v03_threads"
cd "$task_v03_root"
exec "$task_v03_python" -B "$@"
