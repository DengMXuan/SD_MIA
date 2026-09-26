#!/usr/bin/env bash
# 7 MIMIR domains + mixed Pile; seeds 1919/1949/1978. Default: dry-run.
set -euo pipefail
PRETRAINING_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PRETRAINING_SCRIPT_DIR/../../.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -B -u -m experiments.pretraining.matrix mimir "$@"
