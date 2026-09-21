#!/usr/bin/env bash
set -euo pipefail
SD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SD_SCRIPT_DIR/../../.."
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -u -m experiments.sd_membership_sft.audit.cli "$@"
