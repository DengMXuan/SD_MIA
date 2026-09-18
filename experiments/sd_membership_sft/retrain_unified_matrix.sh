#!/usr/bin/env bash
# Run every controlled-SFT model-pair and frozen-target head condition.
#
# The two child matrices intentionally run one after the other. Each child
# already schedules four independent single-GPU workers, so overlapping them
# would place two training processes on every configured GPU.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT" || exit 2

PYTHON=${PYTHON:-.venv/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-experiments/results/sft_runs/unified_matrix_v1}
SPLIT_ROOT=${SPLIT_ROOT:-$RESULTS_ROOT/shared_splits}
MODEL_PAIRS_ROOT=${MODEL_PAIRS_ROOT:-$RESULTS_ROOT/model_pairs}
SPECULATOR_ROOT=${SPECULATOR_ROOT:-$RESULTS_ROOT/speculator_matrix}
MTP_SOURCE_HEAD=${MTP_SOURCE_HEAD:-$SPECULATOR_ROOT/_internal/qwen35_native_mtp_source}
MODEL_REVISIONS_ENV=${MODEL_REVISIONS_ENV:-$SCRIPT_DIR/model_pair_revisions.env}
read -r -a GPUS <<< "${MATRIX_GPUS:-3 4 5 6}"
MODE=run

usage() {
  cat <<'EOF'
Usage: retrain_unified_matrix.sh [--dry-run | --preflight-only | --status]

  --dry-run         Print all 90 conditions without creating files or training.
  --preflight-only  Check four GPUs and all pinned offline models, then create
                    and audit nine shared splits under all five tokenizers.
  --status          Report completion for both child matrices and the total.

The normal run performs one shared preflight, then runs the 36-condition plain
model-pair matrix and the 54-condition EAGLE-3/MTP matrix sequentially. Each
child matrix uses four single-GPU workers. A child failure does not prevent the
other child matrix from running, but the unified launcher exits nonzero unless
all 90 conditions and 270 artifacts are complete.

Environment overrides:
  MATRIX_GPUS="3 4 5 6"  Four exclusive physical GPU indices.
  RESULTS_ROOT=...        Unified output root, isolated from earlier runs.
  SPLIT_ROOT=...          Nine shared split manifests and audit attestations.
  MODEL_PAIRS_ROOT=...    Plain Qwen3/Gemma4 matrix output root.
  SPECULATOR_ROOT=...     EAGLE-3/native-MTP matrix output root.
  MTP_SOURCE_HEAD=...     Immutable converted native-MTP source checkpoint.
  MODEL_REVISIONS_ENV=... Pinned plain-pair model revisions.
  PYTHON=...              Python interpreter (default .venv/bin/python).
  MATRIX_SKIP_GPU_CHECK=1 Skip GPU existence and busy-process checks.
  MATRIX_SKIP_GPU_BUSY_CHECK=1
                          Check GPU existence but allow current processes.
EOF
}

case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry-run ;;
  --preflight-only) MODE=preflight ;;
  --status) MODE=status ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
if (( ${#GPUS[@]} != 4 )); then
  echo "MATRIX_GPUS must contain exactly four distinct GPU indices" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -v "SEEN_GPUS[$gpu]" ]]; then
    echo "MATRIX_GPUS must contain exactly four distinct GPU indices" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done
if [ ! -x "$PYTHON" ]; then
  echo "Python interpreter is not executable: $PYTHON" >&2
  exit 2
fi
if [ ! -f "$MODEL_REVISIONS_ENV" ]; then
  echo "missing model revision file: $MODEL_REVISIONS_ENV" >&2
  exit 2
fi

run_preflight() {
  local -a command=(
    "$PYTHON" -u -m experiments.sd_membership_sft.unified_matrix_preflight
    --split-root "$SPLIT_ROOT"
    --model-revisions-env "$MODEL_REVISIONS_ENV"
    --gpus "${GPUS[@]}"
  )
  if [ "${MATRIX_SKIP_GPU_CHECK:-0}" = 1 ]; then
    command+=(--skip-gpu-check)
  elif [ "${MATRIX_SKIP_GPU_BUSY_CHECK:-0}" = 1 ]; then
    command+=(--skip-gpu-busy-check)
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${command[@]}"
}

run_model_pairs() {
  env \
    RESULTS_ROOT="$MODEL_PAIRS_ROOT" \
    SPLIT_ROOT="$SPLIT_ROOT" \
    MODEL_REVISIONS_ENV="$MODEL_REVISIONS_ENV" \
    MATRIX_GPUS="${GPUS[*]}" \
    MATRIX_SKIP_PREFLIGHT=1 \
    PYTHON="$PYTHON" \
    bash "$SCRIPT_DIR/retrain_model_pairs.sh" "$@"
}

run_speculators() {
  env \
    RESULTS_ROOT="$SPECULATOR_ROOT" \
    SPLIT_ROOT="$SPLIT_ROOT" \
    MTP_SOURCE_HEAD="$MTP_SOURCE_HEAD" \
    MATRIX_GPUS="${GPUS[*]}" \
    MATRIX_SKIP_PREFLIGHT=1 \
    PYTHON="$PYTHON" \
    bash "$SCRIPT_DIR/retrain_speculator_matrix.sh" "$@"
}

completed_from_status() {
  local output=$1
  if [[ "$output" =~ completed_conditions=([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "could not parse child matrix status: $output" >&2
  return 2
}

report_status() {
  local plain_status head_status plain_completed head_completed
  plain_status=$(run_model_pairs --status) || return
  head_status=$(run_speculators --status) || return
  echo "[model-pairs] $plain_status"
  echo "[speculators] $head_status"
  plain_completed=$(completed_from_status "$plain_status") || return
  head_completed=$(completed_from_status "$head_status") || return
  echo "[unified-status] completed_conditions=$((plain_completed + head_completed)) expected_conditions=90 completed_artifacts=$((plain_completed * 3 + head_completed * 3)) expected_artifacts=270"
}

if [ "$MODE" = status ]; then
  report_status
  exit $?
fi

if [ "$MODE" = dry-run ]; then
  echo "[unified-plan] matrix=model_pairs conditions=36 artifacts=108"
  run_model_pairs --dry-run || exit
  echo "[unified-plan] matrix=speculators conditions=54 artifacts=162"
  run_speculators --dry-run || exit
  echo "[unified-plan-ok] conditions=90 artifacts=270 workers=4 sequential_matrices=2"
  exit 0
fi

run_preflight || exit
if [ "$MODE" = preflight ]; then
  exit 0
fi

plain_status=0
head_status=0
echo "[unified-start] matrix=model_pairs conditions=36 artifacts=108"
run_model_pairs || plain_status=$?
if (( plain_status != 0 )); then
  echo "[unified-child-failed] matrix=model_pairs exit=$plain_status" >&2
fi

echo "[unified-start] matrix=speculators conditions=54 artifacts=162"
run_speculators || head_status=$?
if (( head_status != 0 )); then
  echo "[unified-child-failed] matrix=speculators exit=$head_status" >&2
fi

status_output=$(report_status) || exit
printf '%s\n' "$status_output"
if (( plain_status != 0 || head_status != 0 )) || \
  [[ "$status_output" != *"completed_conditions=90 expected_conditions=90"* ]]; then
  echo "[unified-matrix-incomplete] inspect child logs and rerun the same command" >&2
  exit 1
fi
echo "UNIFIED_MATRIX_COMPLETE conditions=90 artifacts=270"
