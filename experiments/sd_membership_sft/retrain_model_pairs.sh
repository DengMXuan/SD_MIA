#!/usr/bin/env bash
set -euo pipefail
SD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SD_SCRIPT_DIR/scripts/retrain_model_pairs.sh" "$@"
