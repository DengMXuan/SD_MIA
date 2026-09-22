#!/usr/bin/env bash
set -euo pipefail
CROSS_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$CROSS_SCRIPT_DIR/../.."
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -u -m experiments.cross_model_audit.cli "$@"
