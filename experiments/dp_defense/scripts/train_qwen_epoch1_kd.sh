#!/usr/bin/env bash
# DP training only: Qwen, epoch 1, KD; 3 datasets x 3 seeds x epsilon 1/4/8.
set -euo pipefail
DP_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$DP_SCRIPT_DIR/../../.."
DP_COMMAND="${1:-dry-run}"
if (( $# )); then shift; fi
case "$DP_COMMAND" in
  dry-run) DP_STAGE=dry-run ;;
  run) DP_STAGE=train ;;
  -h|--help)
    echo 'Usage: train_qwen_epoch1_kd.sh [dry-run|run] [--gpu N | --gpus ...] [--workers N] [--log-root PATH] [--seeds ...] [--benchmarks ...] [--epsilons ...] [--model-root PATH] [--accumulator-device cpu|cuda]'
    echo 'Only run starts training. It never starts an audit. No argument means dry-run.'
    exit 0 ;;
  *) echo 'Expected dry-run or run.' >&2; exit 2 ;;
esac
for DP_ARG in "$@"; do
  case "$DP_ARG" in
    --accumulator-device|--accumulator-device=*) ;;
    --gpu|--gpu=*|--gpus|--gpus=*|--workers|--workers=*|--log-root|--log-root=*|--seeds|--seeds=*|--benchmarks|--benchmarks=*|--epsilons|--epsilons=*|--model-root|--model-root=*|--audit-root|--audit-root=*|--reference-root|--reference-root=*|-h|--help) ;;
    -*) echo 'Unsupported option: use full option names. Qwen, epoch 1 and KD-only are fixed.' >&2; exit 2 ;;
  esac
done
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -B -u -m experiments.dp_defense.sweep "$DP_STAGE" \
  --model-pairs qwen3 --epochs 1 --draft-variants kd "$@"
