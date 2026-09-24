# Qwen main-method and baseline evaluation matrix

The main-method identifier remains `main_fixed_sparse_positive`. The matrix
runs that method and 11 target-only baselines. Natural-generation implementations,
serial collectors, GRU detectors and compatibility aliases have been removed.

## Conditions and observation contract

- Qwen3-8B/1.7B; Wiki/News/ArXiv; target epochs 1/3; condition seeds
  1919/1949/1978; auxiliary-KD and member-SFT drafts.
- The 18 target conditions produce 36 draft audit configurations, 54 worker
  tasks and 234 independent method outputs. Displaying target-only baselines
  for both drafts produces 432 report rows, without adding independent samples.
- The main method verifies the complete candidate response under its original
  prefixes with B=2. Frozen model adapters support independent drafts and
  EAGLE-3/MTP heads; head validation checks prefix consistency and alignment.
- Four disjoint roles remain 2,000 members, 2,000 nonmembers, 2,000 draft
  auxiliaries and 600 audit auxiliaries. The detector uses 320 fit / 80
  validation / 200 independent calibration records; all 4,000 test records
  remain excluded from fitting and selection.
- Fit the established difficulty-conditioned count TCN on nonmembers only.
  Use positive sparse evidence with priors 0.05/0.10/0.25 and tilts 0.5/1/2.
  The method name, scoring formula, partitions and output directory suffix
  `<draft_role>/fixed/main_fixed_sparse_positive` are unchanged.

## Entrypoints and recovery

Both `audit.cli` and the direct `audit.qwen_audit_matrix` entry create the same
main-method/baseline task set. Collection defaults to the supported `fixed`
protocol; unsupported protocol requests fail before collecting or fitting.
The former serial collector and its old module aliases no longer import.

The Qwen CLI retains `--starts` and `--rounds-per-start` solely as legacy
request-identity fields. They do not affect observation collection. Keeping
these fields avoids an unrelated change to the surviving task dictionaries.
The standalone collector no longer accepts starting-prefix or round-budget
options. Its archive layout and per-record random streams remain unchanged.

Shared scheduling, locking and cancellation live in
`experiments/shared/audit/scheduler.py`; checked aggregation lives in
`experiments/shared/audit/reporting.py`. The Qwen worker retains its established
model/data checks. Cross-model and DP audits reuse the same detector and scoring
helpers. An unavailable model remains pending; independent ready tasks can run.

Reports, observations and detector caches require matching requests and sources.
Source hashes remain strict: deleting implementation branches changes source
files, so historical reports are not re-signed or silently reused with the new
code. Existing artifacts are untouched; use a new output batch for new code.
Status and dry-run create no experiment outputs and launch no GPU workers.

## Metrics and physical costs

Report AUC, raw pAUC over FPR [0, 0.10], pAUC divided by 0.10, empirical ROC
TPR at 1%/10% FPR, and independently calibrated TPR with actual test FPR.
ROC thresholds are descriptive, not deployed calibration thresholds. Baselines
that need reference data use at most the 400 fit/validation nonmembers, leaving
the same 200 calibration nonmembers untouched.

Measure preparation/fitting, calibration and test phases, amortized over 4,000
test records. Retain throughput, target/draft forward and token counts, and
peak allocated GPU memory. Synchronize CUDA timing boundaries; exclude model
loading, warmup and serialization from headline method time. Worker wall time
includes loading/retries and is reported separately from measured method work.

WS/RS/BT share greedy reference generation. The first method includes that cost;
later methods expose physical incremental fields and leave standalone cost
columns empty. Target-only rows repeated for both drafts are billed once per
execution group. Missing historical costs remain missing. Token counts are not
FLOPs, and full-context reference costs do not establish production SD speedup.

## Validation

Tests cover immutable main-method identities, both draft roles, collection and
detector recovery, independent calibration, baseline sharing, seed aggregation,
model-family/DP routing, and rejection of retired protocols and entrypoints.
Run from the repository root:

```bash
.venv/bin/python -m pytest -q
```

Read-only preflight and CPU tests do not establish real GPU inference capacity
or membership-inference effectiveness. See the current [README](../README.md)
for commands and the [structure guide](../../../docs/code_structure.md) for
model/draft extension points.
