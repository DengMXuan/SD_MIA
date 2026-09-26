#!/usr/bin/env bash
# Main-method audit only; requires matching completed DP target/KD checkpoints.
set -euo pipefail
DP_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$DP_SCRIPT_DIR/../../.."
DP_COMMAND="${1:-dry-run}"
if (( $# )); then shift; fi
case "$DP_COMMAND" in
  dry-run) DP_STAGE=dry-run ;;
  run) DP_STAGE=audit ;;
  -h|--help)
    echo 'Usage: run_qwen_epoch1_kd_main.sh [dry-run|run] [--gpu N | --gpus ...] [--workers N] [--log-root PATH] [--seeds ...] [--benchmarks ...] [--epsilons ...] [--model-root PATH] [--audit-root PATH]'
    echo 'Only run starts main-method audits. It never trains DP models or runs baselines.'
    exit 0 ;;
  *) echo 'Expected dry-run or run.' >&2; exit 2 ;;
esac
for DP_ARG in "$@"; do
  case "$DP_ARG" in
    --gpu|--gpu=*|--gpus|--gpus=*|--workers|--workers=*|--log-root|--log-root=*|--seeds|--seeds=*|--benchmarks|--benchmarks=*|--epsilons|--epsilons=*|--model-root|--model-root=*|--audit-root|--audit-root=*|--reference-root|--reference-root=*|-h|--help) ;;
    -*) echo 'Unsupported option: use full option names. Qwen, epoch 1 and KD-only are fixed.' >&2; exit 2 ;;
  esac
done
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -B -u -m experiments.dp_defense.sweep "$DP_STAGE" \
  --model-pairs qwen3 --epochs 1 --draft-variants kd "$@"
