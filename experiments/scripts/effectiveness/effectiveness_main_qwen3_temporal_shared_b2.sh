#!/usr/bin/env bash
set -euo pipefail
SD_LAUNCHER=$(readlink -f -- "${BASH_SOURCE[0]}")
source "$(dirname -- "$SD_LAUNCHER")/../common.sh"
exec "$SD_PYTHON" -B -u -m experiments.pretraining.matrix temporal "$@"
