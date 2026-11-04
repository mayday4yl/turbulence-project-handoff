#!/usr/bin/env bash
set -euo pipefail
followup_root=/data2/yanglin/v031_followup_20260918
followup_python=/data0/yanglin/envs/physical_bridge/bin/python
mkdir -p "$followup_root/cache/ascend" "$followup_root/cache/work" "$followup_root/cache/log" "$followup_root/tmp"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONPATH="$followup_root/code:/data0/yanglin/physical_bridge:${PYTHONPATH:-}"
export TASK_QUEUE_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export TMPDIR="$followup_root/tmp" TMP="$followup_root/tmp" TEMP="$followup_root/tmp"
export XDG_CACHE_HOME="$followup_root/cache" TORCH_HOME="$followup_root/cache/torch"
export ASCEND_CACHE_PATH="$followup_root/cache/ascend" ASCEND_WORK_PATH="$followup_root/cache/work"
export ASCEND_PROCESS_LOG_PATH="$followup_root/cache/log"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$followup_root"
exec "$followup_python" -B "$@"
