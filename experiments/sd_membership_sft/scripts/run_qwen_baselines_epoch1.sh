#!/usr/bin/env bash
# Qwen3-8B-Base epoch 1; Wiki/News/Arxiv x seeds 1919/1949/1978.
# Methods: Loss, Min-K, Min-K++, SEAD, PETAL, ReCaLL, ICP-MIA.
# Usage (from any directory):
#   bash /path/to/run_qwen_baselines_epoch1.sh dry-run
#   bash /path/to/run_qwen_baselines_epoch1.sh status
#   bash /path/to/run_qwen_baselines_epoch1.sh run --gpus 0 1
#   bash /path/to/run_qwen_baselines_epoch1.sh summarize
# Only "run" starts inference; completed methods are checked before reuse.
# Set SD_AUDIT_PYTHON to select an existing Python environment.
set -euo pipefail
QWEN_BASELINES_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$QWEN_BASELINES_SCRIPT_DIR/../../.."
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -u -m experiments.sd_membership_sft.audit.qwen_baselines_epoch1 "$@"
