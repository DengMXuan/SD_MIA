#!/usr/bin/env bash
set -euo pipefail
SD_LAUNCHER=$(readlink -f -- "${BASH_SOURCE[0]}")
source "$(dirname -- "$SD_LAUNCHER")/../common.sh"
if (( $# == 0 )); then set -- dry-run; fi
exec "$SD_PYTHON" -B -u "$SD_REPO_ROOT/standalone/qwen_temporal_clean/run.py" "$@"
