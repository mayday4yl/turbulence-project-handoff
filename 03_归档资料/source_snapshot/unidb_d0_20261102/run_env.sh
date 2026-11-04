#!/usr/bin/env bash
set -eo pipefail
UNIDB_ROOT=/data2/yanglin/unidb_d0_20261102
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${UNIDB_DEVICES:-0}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TASK_QUEUE_ENABLE=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$UNIDB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$UNIDB_ROOT/cache"
export TORCH_HOME=/data2/yanglin/v6_diffuir_l_20261027/cache/torch
export ASCEND_CACHE_PATH="$UNIDB_ROOT/cache/ascend"
export PIP_CACHE_DIR="$UNIDB_ROOT/cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH" "$PIP_CACHE_DIR"
exec /data2/yanglin/v6_diffuir_l_20261027/env/bin/python -u "$@"
