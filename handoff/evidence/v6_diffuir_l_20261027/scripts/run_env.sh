#!/usr/bin/env bash
set -eo pipefail
V6_ROOT=/data2/yanglin/v6_diffuir_l_20261027
STAGE_ROOT=/data2/yanglin/v5_diffuir_l_stage1_20261026
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${V6_DEVICES:-2}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TASK_QUEUE_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$V6_ROOT:$STAGE_ROOT/upstream:$STAGE_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$V6_ROOT/cache"
export TORCH_HOME="$V6_ROOT/cache/torch"
export ASCEND_CACHE_PATH="$V6_ROOT/cache/ascend"
export PIP_CACHE_DIR="$V6_ROOT/cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$ASCEND_CACHE_PATH" "$PIP_CACHE_DIR"
exec "$V6_ROOT/env/bin/python" -u "$@"
