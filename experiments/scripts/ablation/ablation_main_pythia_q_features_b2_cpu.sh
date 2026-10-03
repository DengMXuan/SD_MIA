#!/usr/bin/env bash
set -euo pipefail
SD_LAUNCHER=$(readlink -f -- "${BASH_SOURCE[0]}")
source "$(dirname -- "$SD_LAUNCHER")/../common.sh"
export CUDA_VISIBLE_DEVICES=''
exec "$SD_PYTHON" -B -u -m experiments.pretraining.q_feature_exploration "$@"
