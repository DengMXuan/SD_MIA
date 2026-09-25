#!/usr/bin/env bash
set -euo pipefail
FOUR_MODEL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$FOUR_MODEL_SCRIPT_DIR/../.."
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -u -m experiments.main_method_validity.four_model_epoch1 "$@"
