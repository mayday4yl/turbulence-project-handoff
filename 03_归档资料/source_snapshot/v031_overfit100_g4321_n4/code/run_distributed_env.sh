#!/usr/bin/env bash
# Invoke as: bash code/run_distributed_env.sh -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 ...
# Only physical de36 NPUs 0--3 are authorized for this distributed worker.
set -euo pipefail
if [[ "${1:-}" != '-m' || "${2:-}" != 'torch.distributed.run' ]]; then
  echo 'Use the torch.distributed.run parent CLI: -m torch.distributed.run ...' >&2
  exit 2
fi
task_v031_visible="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
if [[ "$task_v031_visible" != '0,1,2,3' ]]; then
  echo 'Only ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 is authorized' >&2
  exit 2
fi
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
export ASCEND_RT_VISIBLE_DEVICES="$task_v031_visible"
# Keep per-op JIT compilation options on one host dispatch thread. Device
# execution remains asynchronous; this is not global launch-blocking mode.
export TASK_QUEUE_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
# Mixed-G cold compilation can bring ranks to their first collective at very
# different times. This changes only socket connection waiting, not arithmetic.
# Every disposable run still has its separate 7200-second resource guard.
export HCCL_CONNECT_TIMEOUT=3600
export TMPDIR="$task_v03_root/tmp" TMP="$task_v03_root/tmp" TEMP="$task_v03_root/tmp"
export XDG_CACHE_HOME="$task_v03_root/cache" TORCH_HOME="$task_v03_root/cache/torch"
export ASCEND_CACHE_PATH="$task_v03_root/cache/ascend"
export ASCEND_WORK_PATH="$task_v03_root/cache/work"
export ASCEND_PROCESS_LOG_PATH="$task_v03_root/cache/log"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
unset PYTHONOPTIMIZE
export OMP_NUM_THREADS="$task_v03_threads" MKL_NUM_THREADS="$task_v03_threads"
export OPENBLAS_NUM_THREADS="$task_v03_threads"
cd "$task_v03_root"
exec "$task_v03_python" -B "$@"
