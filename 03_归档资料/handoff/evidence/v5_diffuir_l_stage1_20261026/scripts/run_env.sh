#!/usr/bin/env bash
set -eo pipefail
DIFFUIR_ROOT=/data2/yanglin/v5_diffuir_l_stage1_20261026
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TASK_QUEUE_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DIFFUIR_ROOT/upstream:$DIFFUIR_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$DIFFUIR_ROOT/cache"
export TORCH_HOME="$DIFFUIR_ROOT/cache/torch"
export ASCEND_CACHE_PATH="$DIFFUIR_ROOT/cache/ascend"
export PIP_CACHE_DIR="$DIFFUIR_ROOT/cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$ASCEND_CACHE_PATH" "$PIP_CACHE_DIR"
exec "$DIFFUIR_ROOT/env/bin/python" -u "$@"
