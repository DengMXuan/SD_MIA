#!/usr/bin/env bash
# Build the 108-checkpoint Qwen3/Gemma 4 controlled-SFT matrix.
#
# A condition produces three full checkpoints: target SFT, auxiliary-data KD
# draft, and member-data SFT draft. The Gemma smoke condition is completed
# before any of the other 35 conditions are allowed to start.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT" || exit 2

RESULTS_ROOT=${RESULTS_ROOT:-experiments/results/sft_runs/model_pairs}
MODEL_REVISIONS_ENV=${MODEL_REVISIONS_ENV:-$SCRIPT_DIR/model_pair_revisions.env}
PYTHON=${PYTHON:-.venv/bin/python}
read -r -a GPUS <<< "${MATRIX_GPUS:-3 4 5 6}"
BENCHMARKS=(wikitection newstection arxivtection)
EPOCHS=(1 3)
SEEDS=(1919 1949 1978)
MODE=run

usage() {
  cat <<'EOF'
Usage: retrain_model_pairs.sh [--dry-run | --preflight-only]

  --dry-run         Print the smoke condition and remaining matrix commands.
                    No directories are created and no training is started.
  --preflight-only  Validate cached model revisions, tokenizers, datasets, and
                    configured GPUs. No training is started.

Environment overrides:
  MATRIX_GPUS="3 4 5 6"  Exclusive worker GPU indices.
  RESULTS_ROOT=...        Matrix output directory.
  MODEL_REVISIONS_ENV=... File containing the four pinned revisions.
  PYTHON=...              Python interpreter used for preflight and training.
EOF
}

case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry-run ;;
  --preflight-only) MODE=preflight ;;
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
if [ ! -f "$MODEL_REVISIONS_ENV" ]; then
  echo "missing model revision file: $MODEL_REVISIONS_ENV" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$MODEL_REVISIONS_ENV"

for variable in \
  QWEN_TARGET_REVISION QWEN_DRAFT_REVISION \
  GEMMA_TARGET_REVISION GEMMA_DRAFT_REVISION; do
  value=${!variable:-}
  if [[ ! "$value" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$variable must be a pinned 40-character commit revision" >&2
    exit 2
  fi
done

set_pair_models() {
  case "$1" in
    qwen3)
      target_model=Qwen/Qwen3-8B-Base
      draft_model=Qwen/Qwen3-1.7B-Base
      target_revision=$QWEN_TARGET_REVISION
      draft_revision=$QWEN_DRAFT_REVISION
      ;;
    gemma4)
      target_model=google/gemma-4-12B
      draft_model=google/gemma-4-E2B
      target_revision=$GEMMA_TARGET_REVISION
      draft_revision=$GEMMA_DRAFT_REVISION
      ;;
    *)
      echo "unknown model pair: $1" >&2
      return 2
      ;;
  esac
}

checkpoint_complete() {
  local path=$1
  [ -f "$path/config.json" ] && {
    [ -f "$path/model.safetensors" ] || \
      [ -f "$path/model.safetensors.index.json" ] || \
      [ -f "$path/pytorch_model.bin" ] || \
      [ -f "$path/pytorch_model.bin.index.json" ]
  }
}

condition_complete() {
  local output_dir=$1
  [ -f "$output_dir/results.json" ] && \
    checkpoint_complete "$output_dir/checkpoints/target" && \
    checkpoint_complete "$output_dir/checkpoints/draft_auxiliary_distilled" && \
    checkpoint_complete "$output_dir/checkpoints/draft_member_sft"
}

preflight() {
  local pool
  for benchmark in "${BENCHMARKS[@]}"; do
    pool="experiments/data/pools/$benchmark/pool.jsonl"
    if [ ! -f "$pool" ]; then
      echo "missing frozen dataset pool: $pool" >&2
      return 2
    fi
    if (( $(wc -l < "$pool") < 6000 )); then
      echo "dataset pool has fewer than 6000 records: $pool" >&2
      return 2
    fi
  done

  HF_HUB_OFFLINE=1 "$PYTHON" - \
    "${GPUS[*]}" \
    "Qwen/Qwen3-8B-Base@$QWEN_TARGET_REVISION" \
    "Qwen/Qwen3-1.7B-Base@$QWEN_DRAFT_REVISION" \
    "google/gemma-4-12B@$GEMMA_TARGET_REVISION" \
    "google/gemma-4-E2B@$GEMMA_DRAFT_REVISION" <<'PY'
import os
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

gpu_indices = [int(value) for value in sys.argv[1].split()]
if os.environ.get("MATRIX_SKIP_GPU_CHECK") != "1":
    visible_devices = torch.cuda.device_count()
    missing = [index for index in gpu_indices if index >= visible_devices]
    if missing:
        raise SystemExit(
            f"configured GPUs {missing} are unavailable; torch sees {visible_devices} devices"
        )
    print(f"[gpu-ok] configured={gpu_indices} visible_devices={visible_devices}")
else:
    print("[gpu-check-skipped]")

tokenizers = {}
for item in sys.argv[2:]:
    model_id, revision = item.rsplit("@", 1)
    snapshot = Path(
        snapshot_download(model_id, revision=revision, local_files_only=True)
    )
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"cached snapshot lacks config.json: {model_id}@{revision}")
    if not any(snapshot.glob("*.safetensors")):
        raise SystemExit(f"cached snapshot lacks model weights: {model_id}@{revision}")
    config = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    model_class = AutoModelForCausalLM._model_mapping[type(config)]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    tokenizers[model_id] = tokenizer
    print(
        f"[model-ok] {model_id}@{revision} "
        f"class={model_class.__name__} vocab={len(tokenizer)}"
    )

for target_id, draft_id in (
    ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-1.7B-Base"),
    ("google/gemma-4-12B", "google/gemma-4-E2B"),
):
    target = tokenizers[target_id]
    draft = tokenizers[draft_id]
    special_ids_match = (
        target.bos_token_id,
        target.eos_token_id,
        target.pad_token_id,
    ) == (
        draft.bos_token_id,
        draft.eos_token_id,
        draft.pad_token_id,
    )
    if target.get_vocab() != draft.get_vocab() or not special_ids_match:
        raise SystemExit(
            f"incompatible pair tokenizers: {target_id} and {draft_id}"
        )
    print(f"[pair-ok] {target_id} + {draft_id}")
PY
}

write_manifest() {
  mkdir -p "$RESULTS_ROOT"
  local temporary_manifest
  temporary_manifest=$(mktemp "$RESULTS_ROOT/.MATRIX.tsv.XXXXXX") || return 2
  printf 'pair\tbenchmark\tepoch\tseed\ttarget_checkpoint\tdraft_kd_checkpoint\tdraft_member_checkpoint\n' \
    > "$temporary_manifest"
  local pair benchmark epoch seed output_dir
  for pair in qwen3 gemma4; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          output_dir="$RESULTS_ROOT/$pair/$benchmark/epoch${epoch}/seed${seed}"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$pair" "$benchmark" "$epoch" "$seed" \
            "$output_dir/checkpoints/target" \
            "$output_dir/checkpoints/draft_auxiliary_distilled" \
            "$output_dir/checkpoints/draft_member_sft" \
            >> "$temporary_manifest"
        done
      done
    done
  done
  mv "$temporary_manifest" "$RESULTS_ROOT/MATRIX.tsv"
}

run_condition() {
  local pair=$1
  local benchmark=$2
  local epoch=$3
  local seed=$4
  local gpu=$5
  local phase=$6
  local output_dir="$RESULTS_ROOT/$pair/$benchmark/epoch${epoch}/seed${seed}"
  local status
  local -a resume=()
  local -a command

  set_pair_models "$pair" || return
  if [ "$MODE" != dry-run ] && condition_complete "$output_dir"; then
    echo "[skip] condition pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed"
    return 0
  fi
  if checkpoint_complete "$output_dir/checkpoints/target"; then
    resume=(--resume)
  fi
  command=(
    "$PYTHON" -u -m experiments.sd_membership_sft.drafts.plain
    --gpu 0
    --target-model "$target_model" --draft-model "$draft_model"
    --target-revision "$target_revision" --draft-revision "$draft_revision"
    --benchmark "$benchmark" --target-epochs "$epoch"
    --trainer full --optimizer adamw8bit
    --target-lr 2e-5 --draft-lr 2e-5
    --target-batch-size 2 --target-grad-accum 8
    --draft-batch-size 2 --draft-grad-accum 8
    --n-per-class 2000 --n-aux 2000
    --distill-steps 384 --seed "$seed" --data-seed "$seed"
    --output-dir "$output_dir" "${resume[@]}"
  )

  if [ "$MODE" = dry-run ]; then
    printf '[%s] condition pair=%s benchmark=%s epoch=%s seed=%s gpu=%s command=' \
      "$phase" "$pair" "$benchmark" "$epoch" "$seed" "$gpu"
    printf '%q ' env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false "${command[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$output_dir"
  echo "[start] condition pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${command[@]}" > "$output_dir/train.log" 2>&1
  status=$?
  if (( status != 0 )); then
    echo "[failed] condition pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed gpu=$gpu exit=$status log=$output_dir/train.log" >&2
    return "$status"
  fi
  if ! condition_complete "$output_dir"; then
    echo "[failed] condition returned success without all three checkpoints: $output_dir" >&2
    return 1
  fi
  echo "[complete] condition pair=$pair benchmark=$benchmark epoch=$epoch seed=$seed gpu=$gpu"
}

run_worker() {
  local worker_index=$1
  local gpu=$2
  local index=0
  local failures=0
  local pair benchmark epoch seed
  for pair in qwen3 gemma4; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [ "$pair" = gemma4 ] && [ "$benchmark" = newstection ] && \
            [ "$epoch" = 1 ] && [ "$seed" = 1919 ]; then
            continue
          fi
          if (( index % ${#GPUS[@]} == worker_index )); then
            run_condition "$pair" "$benchmark" "$epoch" "$seed" "$gpu" matrix || failures=1
          fi
          index=$((index + 1))
        done
      done
    done
  done
  return "$failures"
}

count_complete_conditions() {
  local completed=0
  local pair benchmark epoch seed output_dir
  for pair in qwen3 gemma4; do
    for benchmark in "${BENCHMARKS[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          output_dir="$RESULTS_ROOT/$pair/$benchmark/epoch${epoch}/seed${seed}"
          if condition_complete "$output_dir"; then
            completed=$((completed + 1))
          fi
        done
      done
    done
  done
  printf '%s\n' "$completed"
}

if [ "$MODE" = dry-run ]; then
  run_condition gemma4 newstection 1 1919 "${GPUS[0]}" smoke || exit
  for worker_index in "${!GPUS[@]}"; do
    run_worker "$worker_index" "${GPUS[$worker_index]}" || exit
  done
  echo "[plan-ok] conditions=36 checkpoints=108 workers=4"
  exit 0
fi

preflight || exit
echo "[preflight-ok] cached models, pair tokenizers, and datasets are ready"
if [ "$MODE" = preflight ]; then
  exit 0
fi

write_manifest || exit

# The smoke condition is an explicit gate. No matrix worker starts unless this
# exact Gemma condition has produced all three checkpoints.
run_condition gemma4 newstection 1 1919 "${GPUS[0]}" smoke || exit

pids=()
for worker_index in "${!GPUS[@]}"; do
  run_worker "$worker_index" "${GPUS[$worker_index]}" &
  pids+=("$!")
done

worker_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || worker_failed=1
done
if (( worker_failed )); then
  echo "matrix stopped with one or more failed conditions" >&2
  exit 1
fi

completed=$(count_complete_conditions)
if (( completed != 36 )); then
  echo "matrix incomplete: completed_conditions=$completed expected_conditions=36" >&2
  exit 1
fi
echo "MATRIX_COMPLETE completed_conditions=36 completed_checkpoints=108"
