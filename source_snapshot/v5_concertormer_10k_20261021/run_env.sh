#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TASK_QUEUE_ENABLE=0
export V5_SIMULATOR_ROOT="${V5_SIMULATOR_ROOT:-/data2/yanglin/v5_v1_simulator_20261020}"
export V5_NPU_KERNEL_CACHE="$PWD/cache/npu_kernels"
export PYTHONPATH="$PWD:$V5_SIMULATOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /data0/yanglin/envs/physical_bridge/bin/python -u "$@"
