#!/usr/bin/env bash
# Retrain the pool benchmarks (wikitection / newstection / arxivtection)
# with Qwen3-8B target + Qwen3-1.7B drafts, one benchmark per GPU, epoch 1 then
# 3 per benchmark. Hyperparameters replicate the original runs exactly
# (full-parameter 8-bit AdamW, lr 2e-5, effective batch 16, 384 distill steps).
set -uo pipefail
cd /home/mxd/lib/SD_MIA

RESULTS=${RESULTS:-artifacts/runs/training/four_role_v1}
COMMON=(
  --trainer full --optimizer adamw8bit
  --target-lr 2e-5 --draft-lr 2e-5
  --target-batch-size 2 --target-grad-accum 8
  --draft-batch-size 2 --draft-grad-accum 8
  --n-per-class 2000 --n-aux 2000 --n-audit-aux 600
  --seed 20260824 --data-seed 20260824
  --distill-steps 384
)

run_bench() {
  local BENCH=$1 GPU=$2
  for E in 1 3; do
    local OUT="$RESULTS/${BENCH}_qwen3_8b_epoch${E}"
    if [ -f "$OUT/results.json" ]; then
      echo "[skip] $OUT (results.json exists)"
      continue
    fi
    echo "[run ] bench=$BENCH epoch=$E gpu=$GPU"
    .venv/bin/python -m experiments.sd_membership_sft.drafts.plain \
      --gpu "$GPU" --benchmark "$BENCH" --target-epochs "$E" \
      --output-dir "$OUT" "${COMMON[@]}" \
      > "/tmp/retrain_${BENCH}_e${E}.log" 2>&1
    echo "[done] bench=$BENCH epoch=$E exit=$?"
  done
}

run_bench wikitection 1 &
PID1=$!
run_bench newstection 2 &
PID2=$!
run_bench arxivtection 3 &
PID3=$!
wait $PID1 $PID2 $PID3
echo "ALL_RETRAIN_DONE"
