#!/usr/bin/env bash
set -euo pipefail
SD_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$SD_REPO_ROOT/experiments/scripts/common.sh"
exec "$SD_PYTHON" -B -u -m standalone.pretraining_baselines.run "$@"
