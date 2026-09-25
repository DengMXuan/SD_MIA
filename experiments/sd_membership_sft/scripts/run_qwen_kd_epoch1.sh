#!/usr/bin/env bash
set -euo pipefail
QWEN_KD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$QWEN_KD_SCRIPT_DIR/../../.."
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -u -m experiments.sd_membership_sft.audit.qwen_kd_epoch1 "$@"
