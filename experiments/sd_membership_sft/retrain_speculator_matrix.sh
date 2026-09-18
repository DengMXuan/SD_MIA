#!/usr/bin/env bash
# Train the 54-condition frozen-target EAGLE-3 / native-MTP matrix.
#
# Each condition saves one full target plus independently initialized
# auxiliary and member heads. Four workers run one single-GPU condition each.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT" || exit 2

PYTHON=${PYTHON:-.venv/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-experiments/results/sft_runs/speculator_matrix}
SPLIT_ROOT=${SPLIT_ROOT:-$RESULTS_ROOT/shared_splits}
MTP_SOURCE_HEAD=${MTP_SOURCE_HEAD:-$RESULTS_ROOT/_internal/qwen35_native_mtp_source}
read -r -a GPUS <<< "${MATRIX_GPUS:-3 4 5 6}"
PAIRS=(qwen3_8b_eagle3 llama31_8b_eagle3 qwen35_9b_mtp)
BENCHMARKS=(wikitection newstection arxivtection)
EPOCHS=(1 3)
SEEDS=(1919 1949 1978)
MODE=run

usage() {
  cat <<'EOF'
Usage: retrain_speculator_matrix.sh [--dry-run | --preflight-only | --status]

  --dry-run         Print every condition and stage command; do not write files.
  --preflight-only  Check GPUs and pinned offline model caches, then create and
                    audit the nine shared raw-split manifests. No training.
  --status          Count complete conditions without checking GPUs or models.

Environment overrides:
  MATRIX_GPUS="3 4 5 6"       Four exclusive physical GPU indices.
  RESULTS_ROOT=...             Matrix artifact root.
  SPLIT_ROOT=...               Shared raw-split manifest root.
  MTP_SOURCE_HEAD=...          One immutable native-MTP conversion cache.
  PYTHON=...                   Python interpreter (default .venv/bin/python).
  MATRIX_SKIP_GPU_CHECK=1      Skip GPU existence and busy-process checks.
  MATRIX_SKIP_GPU_BUSY_CHECK=1 Check GPU existence but allow existing processes.
  MATRIX_SKIP_PREFLIGHT=1      Internal: reuse a supervising launcher's preflight.
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
  echo "MATRIX_GPUS must contain exactly four GPU indices" >&2
  exit 2
fi
if [ ! -x "$PYTHON" ]; then
  echo "Python interpreter is not executable: $PYTHON" >&2
  exit 2
fi

checkpoint_complete() {
  local path=$1
  [ -f "$path/_COMPLETE.json" ] && [ -f "$path/config.json" ] && {
    [ -f "$path/model.safetensors" ] || \
      [ -f "$path/model.safetensors.index.json" ] || \
      [ -f "$path/pytorch_model.bin" ] || \
      [ -f "$path/pytorch_model.bin.index.json" ]
  }
}

condition_dir() {
  printf '%s/%s/%s/epoch%s/seed%s\n' "$RESULTS_ROOT" "$1" "$2" "$3" "$4"
}

condition_complete() {
  local output_dir=$1
  [ -f "$output_dir/CONDITION_COMPLETE.json" ] && \
    checkpoint_complete "$output_dir/checkpoints/target" && \
    checkpoint_complete "$output_dir/heads/auxiliary_head" && \
    checkpoint_complete "$output_dir/heads/member_head"
}

write_condition_marker() {
  local output_dir=$1 pair=$2 benchmark=$3 epoch=$4 seed=$5
  local temporary="$output_dir/.CONDITION_COMPLETE.json.tmp.$$"
  printf '{\n  "status": "complete",\n  "pair": "%s",\n  "benchmark": "%s",\n  "target_epochs": %s,\n  "seed": %s\n}\n' \
    "$pair" "$benchmark" "$epoch" "$seed" > "$temporary"
  mv "$temporary" "$output_dir/CONDITION_COMPLETE.json"
}

stage_checkpoint() {
  local pair=$1 output_dir=$2 stage=$3
  case "$stage" in
    target) printf '%s/checkpoints/target\n' "$output_dir" ;;
    source)
      if [ "$pair" != qwen35_9b_mtp ]; then
        return 2
      fi
      printf '%s\n' "$MTP_SOURCE_HEAD"
      ;;
    auxiliary) printf '%s/heads/auxiliary_head\n' "$output_dir" ;;
    member) printf '%s/heads/member_head\n' "$output_dir" ;;
    *) return 2 ;;
  esac
}

build_stage_command() {
  local pair=$1 benchmark=$2 epoch=$3 seed=$4 output_dir=$5 stage=$6
  local batch_size=2 grad_accum=8
  local split_manifest="$SPLIT_ROOT/$benchmark/seed${seed}.json"
  local module=experiments.sd_membership_sft.drafts.eagle3
  if [ "$pair" = qwen35_9b_mtp ]; then
    module=experiments.sd_membership_sft.drafts.mtp
  fi
  STAGE_COMMAND=(
    "$PYTHON" -u -m "$module"
    --gpu 0 --pair "$pair" --benchmark "$benchmark" --epochs "$epoch"
    --seed "$seed" --data-seed "$seed"
    --split-manifest "$split_manifest"
    --n-per-class 2000 --n-aux 2000
    --batch-size "$batch_size" --grad-accum "$grad_accum" --lr 2e-5
    --head-updates 384 --head-lr 2e-5
    --head-batch-size "$batch_size" --head-grad-accum "$grad_accum"
    --kd-temperature 2.0 --output-dir "$output_dir"
  )
  if [ "$pair" = qwen35_9b_mtp ]; then
    STAGE_COMMAND+=(--source-head "$MTP_SOURCE_HEAD")
    case "$stage" in
      source) STAGE_COMMAND+=(mtp-source) ;;
      target) STAGE_COMMAND+=(mtp-target) ;;
      auxiliary) STAGE_COMMAND+=(mtp-head --variant aux) ;;
      member) STAGE_COMMAND+=(mtp-head --variant member) ;;
      *) return 2 ;;
    esac
  else
    case "$stage" in
      target) STAGE_COMMAND+=(eagle-target) ;;
      auxiliary) STAGE_COMMAND+=(eagle-head --variant aux) ;;
      member) STAGE_COMMAND+=(eagle-head --variant member) ;;
      *) return 2 ;;
    esac
  fi
}

run_stage() {
  local pair=$1 benchmark=$2 epoch=$3 seed=$4 gpu=$5 output_dir=$6 stage=$7
  local checkpoint log status timestamp
  checkpoint=$(stage_checkpoint "$pair" "$output_dir" "$stage") || return 2
  if [ "$MODE" != dry-run ] && checkpoint_complete "$checkpoint"; then
    echo "[skip-stage] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed stage=$stage"
    return 0
  fi
  build_stage_command "$pair" "$benchmark" "$epoch" "$seed" "$output_dir" "$stage" || return
  if [ "$MODE" = dry-run ]; then
    printf '[stage] pair=%s benchmark=%s epoch=%s seed=%s stage=%s gpu=%s command=' \
      "$pair" "$benchmark" "$epoch" "$seed" "$stage" "$gpu"
    printf '%q ' env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      PYTHONHASHSEED="$seed" "${STAGE_COMMAND[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$output_dir/logs"
  log="$output_dir/logs/${stage}.log"
  if [ -f "$log" ]; then
    timestamp=$(date +%Y%m%dT%H%M%S)
    mv "$log" "$log.previous.$timestamp.$$"
  fi
  echo "[start-stage] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed stage=$stage gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONHASHSEED="$seed" \
    "${STAGE_COMMAND[@]}" > "$log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "[failed-stage] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed stage=$stage gpu=$gpu exit=$status log=$log" >&2
    return "$status"
  fi
  if ! checkpoint_complete "$checkpoint"; then
    echo "[failed-stage] command exited successfully without a complete checkpoint: $checkpoint" >&2
    return 1
  fi
  echo "[complete-stage] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed stage=$stage gpu=$gpu"
}

run_condition() {
  local pair=$1 benchmark=$2 epoch=$3 seed=$4 gpu=$5
  local output_dir failed=0
  output_dir=$(condition_dir "$pair" "$benchmark" "$epoch" "$seed")
  echo "[condition] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed gpu=$gpu"
  if [ "$MODE" != dry-run ] && condition_complete "$output_dir"; then
    echo "[skip-condition] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed"
    return 0
  fi

  if [ "$MODE" != dry-run ] && [ "$pair" = qwen35_9b_mtp ] && \
    ! checkpoint_complete "$MTP_SOURCE_HEAD"; then
    echo "[failed-condition] immutable MTP source is incomplete: $MTP_SOURCE_HEAD" >&2
    return 1
  fi
  run_stage "$pair" "$benchmark" "$epoch" "$seed" "$gpu" "$output_dir" target || return
  run_stage "$pair" "$benchmark" "$epoch" "$seed" "$gpu" "$output_dir" auxiliary || failed=1
  run_stage "$pair" "$benchmark" "$epoch" "$seed" "$gpu" "$output_dir" member || failed=1
  if (( failed )); then
    return 1
  fi
  if [ "$MODE" != dry-run ]; then
    write_condition_marker "$output_dir" "$pair" "$benchmark" "$epoch" "$seed"
  fi
  echo "[complete-condition] pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed gpu=$gpu"
}

run_worker() {
  local worker_index=$1 gpu=$2 failure_file=$3
  local index=0 pair benchmark epoch seed
  for pair in "${PAIRS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if (( index % ${#GPUS[@]} == worker_index )); then
            if ! run_condition "$pair" "$benchmark" "$epoch" "$seed" "$gpu"; then
              printf '%s\t%s\t%s\t%s\t%s\n' \
                "$pair" "$benchmark" "$epoch" "$seed" "$gpu" >> "$failure_file"
            fi
          fi
          index=$((index + 1))
        done
      done
    done
  done
}

count_complete_conditions() {
  local completed=0 pair benchmark epoch seed output_dir
  for pair in "${PAIRS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          output_dir=$(condition_dir "$pair" "$benchmark" "$epoch" "$seed")
          if condition_complete "$output_dir"; then
            completed=$((completed + 1))
          fi
        done
      done
    done
  done
  printf '%s\n' "$completed"
}

write_matrix_manifest() {
  mkdir -p "$RESULTS_ROOT"
  local temporary="$RESULTS_ROOT/.MATRIX.tsv.tmp.$$"
  printf 'pair\tbenchmark\ttarget_epochs\tseed\ttarget_checkpoint\tauxiliary_head\tmember_head\n' > "$temporary"
  local pair benchmark epoch seed output_dir
  for pair in "${PAIRS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          output_dir=$(condition_dir "$pair" "$benchmark" "$epoch" "$seed")
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$pair" "$benchmark" "$epoch" "$seed" \
            "$output_dir/checkpoints/target" \
            "$output_dir/heads/auxiliary_head" \
            "$output_dir/heads/member_head" >> "$temporary"
        done
      done
    done
  done
  mv "$temporary" "$RESULTS_ROOT/MATRIX.tsv"
}

run_preflight() {
  local -a command=(
    "$PYTHON" -u -m experiments.sd_membership_sft.head_matrix_preflight
    --split-root "$SPLIT_ROOT" --gpus "${GPUS[@]}"
  )
  if [ "${MATRIX_SKIP_GPU_CHECK:-0}" = 1 ]; then
    command+=(--skip-gpu-check)
  elif [ "${MATRIX_SKIP_GPU_BUSY_CHECK:-0}" = 1 ]; then
    command+=(--skip-gpu-busy-check)
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${command[@]}"
}

if [ "$MODE" = status ]; then
  completed=$(count_complete_conditions)
  echo "[status] completed_conditions=$completed expected_conditions=54 completed_checkpoints=$((completed * 3)) expected_checkpoints=162"
  exit 0
fi

if [ "$MODE" = dry-run ]; then
  dry_failure_file=/dev/null
  run_stage qwen35_9b_mtp newstection 1 1919 "${GPUS[0]}" \
    "$RESULTS_ROOT/_internal/mtp_source_stage" source || exit
  for worker_index in "${!GPUS[@]}"; do
    run_worker "$worker_index" "${GPUS[$worker_index]}" "$dry_failure_file"
  done
  echo "[plan-ok] conditions=54 checkpoints=162 workers=4"
  exit 0
fi

if [ "${MATRIX_SKIP_PREFLIGHT:-0}" = 1 ]; then
  echo "[preflight-reused] supervised launcher supplied shared manifests and audits"
else
  run_preflight || exit
fi
if [ "$MODE" = preflight ]; then
  exit 0
fi
write_matrix_manifest || exit

# The native MTP conversion is immutable and shared by all 18 MTP conditions.
# If it fails, EAGLE conditions still run; each MTP condition records a failure.
run_stage qwen35_9b_mtp newstection 1 1919 "${GPUS[0]}" \
  "$RESULTS_ROOT/_internal/mtp_source_stage" source || true

failure_root="$RESULTS_ROOT/.worker_failures"
mkdir -p "$failure_root"
pids=()
failure_files=()
for worker_index in "${!GPUS[@]}"; do
  failure_file="$failure_root/worker${worker_index}.tsv"
  : > "$failure_file"
  failure_files+=("$failure_file")
  run_worker "$worker_index" "${GPUS[$worker_index]}" "$failure_file" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

failures=0
for failure_file in "${failure_files[@]}"; do
  while IFS= read -r failure; do
    if [ -n "$failure" ]; then
      echo "[failed-condition] $failure" >&2
      failures=$((failures + 1))
    fi
  done < "$failure_file"
done
completed=$(count_complete_conditions)
if (( failures > 0 || completed != 54 )); then
  echo "[matrix-incomplete] completed_conditions=$completed expected_conditions=54 failures=$failures" >&2
  exit 1
fi
echo "MATRIX_COMPLETE completed_conditions=54 completed_checkpoints=162"
