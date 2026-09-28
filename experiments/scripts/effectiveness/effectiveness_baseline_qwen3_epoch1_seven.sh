#!/usr/bin/env bash
set -euo pipefail
SD_LAUNCHER=$(readlink -f -- "${BASH_SOURCE[0]}")
source "$(dirname -- "$SD_LAUNCHER")/../common.sh"
if (( $# == 0 )); then set -- dry-run; fi
exec "$SD_PYTHON" -B -u -m experiments.sd_membership_sft.audit.qwen_baselines_epoch1 "$@"
