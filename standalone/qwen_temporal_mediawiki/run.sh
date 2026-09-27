#!/usr/bin/env bash
set -euo pipefail
WIKI_TEMPORAL_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$WIKI_TEMPORAL_SCRIPT_DIR/../.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${SD_AUDIT_PYTHON:-.venv/bin/python}" -B -u -m standalone.qwen_temporal_mediawiki.prepare "$@"
