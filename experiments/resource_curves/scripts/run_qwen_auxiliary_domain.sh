#!/usr/bin/env bash
# Qwen3 epoch 1, auxiliary-KD draft; one experimental point per GPU.
set -euo pipefail
RESOURCE_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$RESOURCE_SCRIPT_DIR/../../.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -B -u -m experiments.resource_curves.qwen_matrix domain "$@"
