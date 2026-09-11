#!/usr/bin/env bash
# Durable, CPU-only preparation. Re-running reuses successful API responses.
set -uo pipefail
cd "$(dirname "$0")/../.."
output_dir=experiments/data/pools/wiki2023
mkdir -p "$output_dir"
exec >> "$output_dir/build.log" 2>&1
printf '\nStarted at %s\n' "$(date -Is)"
printf '%s\n' "$$" > "$output_dir/collector.pid"
trap 'result=$?; printf "%s\n" "$result" > "$output_dir/exit_code"; printf "Finished at %s exit=%s\n" "$(date -Is)" "$result"' EXIT
rm -f "$output_dir/exit_code"
set -e
./.venv/bin/python -u -m experiments.sd_membership_sft.pools wiki \
  --window-start 2023-01-01T00:00:00Z \
  --window-end 2023-12-31T23:59:59Z \
  --snapshot-at 2023-12-31T23:59:59Z \
  --seed 20260824 --records 2000 --candidate-limit 15000 --parallel 3 \
  --selection-tokenizer Qwen/Qwen3-8B-Base \
  --output-path "$output_dir/pool.jsonl" "$@"
./.venv/bin/python -u -m experiments.baseline.prepare_temporal_wiki \
  --historical-pools "$output_dir/pool.jsonl" \
  --nonmember-run experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --output-dir experiments/data/audits/wiki_temporal_qwen3_8b
