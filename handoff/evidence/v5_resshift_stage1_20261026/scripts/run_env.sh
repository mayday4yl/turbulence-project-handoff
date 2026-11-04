#!/usr/bin/env bash
set -eo pipefail
STAGE_ROOT=/data2/yanglin/v5_resshift_stage1_20261026
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TASK_QUEUE_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$STAGE_ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$STAGE_ROOT/cache"
export TORCH_HOME="$STAGE_ROOT/cache/torch"
export ASCEND_CACHE_PATH="$STAGE_ROOT/cache/ascend"
export PIP_CACHE_DIR="$STAGE_ROOT/cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$ASCEND_CACHE_PATH" "$PIP_CACHE_DIR"
exec "$STAGE_ROOT/env/bin/python" -u "$@"
