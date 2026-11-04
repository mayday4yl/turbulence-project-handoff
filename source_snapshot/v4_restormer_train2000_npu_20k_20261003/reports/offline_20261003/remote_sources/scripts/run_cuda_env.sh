#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$ROOT_DIR/assets/torch"
cd "$ROOT_DIR"
exec /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python "$@"
