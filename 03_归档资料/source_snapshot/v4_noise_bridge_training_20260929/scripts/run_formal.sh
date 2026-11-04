#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
V4_PY=/home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
bash scripts/launch_phase.sh train
CUDA_VISIBLE_DEVICES= "$V4_PY" scripts/summarize_comparison.py
