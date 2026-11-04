#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TASK_QUEUE_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export V4_NPU_KERNEL_CACHE="$PWD/cache/npu_kernels"
exec /data0/yanglin/envs/physical_bridge/bin/python "$@"
