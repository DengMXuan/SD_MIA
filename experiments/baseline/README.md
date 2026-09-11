# Target-only MIA baselines

This package evaluates either a saved controlled-SFT target or a frozen
pretrained Pythia/MIMIR target. For Pythia, use `--pretraining-manifest` as
described in [the pretraining guide](../pretraining/README.md).

For controlled SFT, the runner evaluates the saved fine-tuned target only. It
never loads a draft checkpoint, a reference model, or a separately scored
pre-SFT target. For a LoRA run, Transformers/PEFT must instantiate the
adapter's base weights to reconstruct the fine-tuned target; the base is not
scored as a second model and is not used as a reference. In pretraining mode,
the manifest's pretrained target is intentionally the model being evaluated;
the Pythia draft is still not loaded by this baseline runner.

## Run one condition

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.baseline.run \
  --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch1 \
  --gpu 0 --methods all \
  --output-dir experiments/results/baseline/newstection_qwen3_8b_epoch1
```

`CUDA_VISIBLE_DEVICES=1` is only an example: it maps physical GPU 1 to logical
`cuda:0` for this process. Use a free GPU and keep `--gpu 0` after pinning it.
Use a comma-separated subset such as `--methods loss,min_k_prob,min_k_pp`
when generation-based methods are not needed. The default runner uses eager
attention, generation batch size 8, one untimed warmup record per method and
seed `20260824`; all of these are recorded in the protocol. To select SDPA
explicitly, add `--attn-implementation sdpa` after validating it for the target
checkpoint.

For controlled SFT, the complete matrix is:

```text
experiments/results/baseline/
├── wikitection_qwen3_8b_epoch1/
├── wikitection_qwen3_8b_epoch3/
├── newstection_qwen3_8b_epoch1/
├── newstection_qwen3_8b_epoch3/
├── arxivtection_qwen3_8b_epoch1/
└── arxivtection_qwen3_8b_epoch3/
```

Each condition is 2,000 members + 2,000 nonmembers, with 2,000 disjoint
auxiliary records. Run one process per free GPU; do not reuse an output
directory that already contains a completed report.

## Implemented methods

All output scores are oriented so larger means “more likely member”.

| Method | Target-only implementation in this repository |
|---|---|
| Loss | Mean teacher-forced log-likelihood over the SFT response (prompt masked). |
| Min-K% Prob | Mean log-probability of the lowest `--k-percent` response tokens. |
| Min-K%++ | Lowest-K% standardized token log-probabilities using the target distribution's own mean and variance. |
| ReCaLL | Relative LL after prepending `--recall-shots` held-out auxiliary records; no reference model. |
| ICP-MIA | Optimization-gap proxy against the top-K auxiliary non-member probes, ranked in the fine-tuned target's own input-embedding space. |
| PETAL | PETAL's similarity-to-log-probability regression, recalibrated on auxiliary records scored by the fine-tuned target instead of a surrogate. Similarity uses target input-embedding cosine. This is a target-only adaptation, not the official surrogate-based threat model. |
| SEAD | Official surrogate-free frequency estimator: Monte Carlo next-token density from the fine-tuned target at `--sead-samples`. |
| WS / RS / BT | Label-only robustness probes. WS performs target-only lexical replacement, RS random adjacent swaps, and BT uses the target itself to generate a paraphrase before re-querying it. |
| SaMIA | Target-only sampled continuation overlap with the held-out response suffix, using the official ROUGE-1 recall statistic. |

WS/RS/BT and SaMIA are deliberately target-only adaptations for this SFT
protocol: they do not download a translator, surrogate, or original model.
Their output-channel scores should not be described as exact reproductions of
the original external-model label-only threat model.

## Source implementations read

The implementation was checked against the following first-party/public
artifacts:

- [ReCaLL official repository](https://github.com/ruoyuxie/recall):
  `src/run.py`, especially `get_ll`, `get_conditional_ll`, and the
  `base_ll - conditional_ll` score.
- [Min-K% Prob official repository](https://github.com/swj0419/detect-pretrain-code):
  `src/run.py`, including lowest-token probability aggregation and its
  reference-model baseline (the latter is intentionally omitted here).
- [Min-K%++ official repository](https://github.com/zjysteven/mink-plus-plus):
  `run.py`, including vocabulary expectation/variance normalization and
  lowest-K% aggregation.
- [ICP-MIA official repository](https://github.com/RPI-DSPlab/ICP-MIA):
  `icp_mia_attack.py` and `baseline/attacks/loss.py`; its reference-data and
  precomputed perturbation branches are replaced here with the run's
  auxiliary non-member records.
- [SaMIA official repository](https://github.com/nlp-titech/samia):
  `src/sampling.py` and `src/eval_samia.py`.
- [PETAL official repository](https://github.com/QingHuan-6/AIModelSecurityEvaluationPlatform/tree/main/PETAL)
  and [public artifact](https://zenodo.org/records/14725819): `run.py`,
  `utils.py`, and `vectors.py`. The artifact uses a surrogate regression;
  the target-only recalibration above is an explicit adaptation and must not
  be reported as exact PETAL.
- [SEAD author-associated repository](https://github.com/clearloveclearlove/SEAD):
  `run_sead.py` implements the target-only frequency estimator; equations
  (4)--(7) of the [paper](https://aclanthology.org/2026.findings-acl.337/)
  define its Monte Carlo density and optional semantic-aware extension. This
  runner uses frequency mode and does not load the optional NLI classifier.

The exact repository status, formulas, and unavailable official-code cases
are recorded in the companion research note under `research/`.

## Progress and completed-method recovery

New runs print timestamped stage progress (record/batch counts, rate and stage
ETA), with `--progress-interval 30` controlling updates **between** records or
batches. A long model call can exceed this interval. The same events are flushed
to `OUTPUT/executions/<UTC-time>-<unique-id>/progress.jsonl`; `status.json` holds
the latest event. Python exceptions and interrupts include their traceback and
completed-method list. SIGKILL, host failure and kernel OOM cannot execute a
Python exception handler; an unfinished status then requires checking the process
and system logs. This does not retrofit already-running Python processes.

Each method is saved immediately after its final score as an atomic `<method>.npz`
in that execution directory, containing labels, ordered record IDs, scores and
`protocol_json` and `cost_json`. Methods execute independently and each is saved
as soon as scoring and cost measurement finish. In-progress methods are not checkpointed,
and this is not automatic resume. Normal completion also retains the original
combined `baseline_metrics.json`, `baseline_scores.npz` and report output.

Recover completed methods after a later failure, without loading a model:

```bash
./.venv/bin/python -m experiments.baseline.export_completed \
  --execution-dir experiments/results/baseline/YOUR_RUN/executions/EXECUTION_ID \
  --output-dir experiments/results/baseline/YOUR_RUN/recovered
```

Use a fresh recovery directory. To capture library warnings and all stderr too,
launch from Bash with `set -o pipefail` and `2>&1 | tee /path/to/unique-run.log`.
Stage ETA is operational monitoring. The method-level cost table below is the
measurement to use for comparisons.

While a run is active, inspect the newest execution directory:

```bash
RUN=experiments/results/baseline/newstection_qwen3_8b_epoch1
cat "$RUN"/executions/*/status.json
tail -f experiments/results/baseline/logs/newstection_qwen3_8b_epoch1.log
```

The status file is operational only: a long model call can exceed the
configured progress interval. Python exceptions write `failed` plus a
traceback; SIGKILL, host failure and kernel OOM must be checked through the
process/system logs.

## Cost and efficiency (recorded automatically)

Each run records exactly three headline metrics, also shown in
`BASELINE_RESULTS.md` and `BASELINE_COSTS.md`:

| Metric | Definition |
|---|---|
| `amortized_ms_per_record` | Method preparation/calibration + scoring + postprocessing time, divided by **all** audited member and nonmember records. CUDA is synchronized at both timing boundaries. |
| `target_sequences_per_record` | Teacher-forced sequences plus generated sequences (expanded over batches and repeated samples), divided by audit records. These are logical sequence queries, not API requests or autoregressive decoder steps. |
| `tokens_per_record` | Nonpadding input tokens plus actual generated tokens, divided by audit records. Input prompts count once per returned generation sequence. Generation counts through the first EOS inclusive, excluding padding afterwards; even tokens later trimmed for scoring still cost work. |

Lower is better. Token counts are a workload proxy, not FLOPs: input/prefill and
autoregressive generation can have different time costs. Batch-amortized time is
not single-request latency. Raw totals and separate input/generated token counts
are retained to make the three metrics auditable and permit weighted aggregation.

For fair method comparisons, `--methods all` now loads the model **once** and
runs each method independently, with its own scorer/cache. Shared teacher-forced
statistics and WS/RS/BT reference generations are recomputed for each dependent
method. Therefore costs do not become zero or change merely because another
method was requested first. This takes more total time than the former shared
execution. A single-method command follows the same path.

One untimed teacher-forced warmup record precedes each method by default
(`--cost-warmup-records 1`; use 0 to disable). Model/data loading and result-file
serialization are excluded; method-specific setup/calibration and progress
bookkeeping are included. Warmup resets the sampling seed before measurement.
Generation-specific initialization is included in the method's amortized time.
Compare on the same records, auxiliary pool, model, precision, hardware, batch
size and sampling settings, with similar GPU contention. These settings and the
device/dtype are recorded in `protocol.cost_measurement`.

Examples of accounting:

- Loss, Min-K and Min-K++ each need one scoring sequence per audit record.
- ReCaLL also queries the context-conditioned record; ICP includes every selected
  probe query, plus its index/retrieval time.
- PETAL includes its auxiliary calibration forwards, amortized across audit N.
- SEAD's local logits sampling is **not** counted as repeated model queries.
- WS/RS each include their own reference generation; BT additionally includes
  rewriting; SaMIA includes all `--samia-samples` returned sequences.

`baseline_metrics.json` contains a `costs` entry; `baseline_costs.json` provides
the same comparison separately. Each completed-method NPZ stores `cost_json`,
and the execution directory's cost JSON/Markdown refresh after each method.
`export_completed` restores costs together with scores after a later failure.
Old archives lacking measurements do not receive invented zero-cost values.

## Matrix aggregation and SaMIA shards

`aggregate_report.py` validates a complete six-condition matrix and writes a
cross-condition Markdown/JSON report. Its current contract expects the
historical parallel layout `<benchmark>_epoch{1,3}_parallel/`, with complete
`baseline_metrics.json` and `baseline_scores.npz` containing all 11 methods:

```bash
./.venv/bin/python -m experiments.baseline.aggregate_report \
  --results-root experiments/results/baseline \
  --report experiments/results/baseline/QWEN3_BASELINE_RESULTS.md \
  --json experiments/results/baseline/qwen3_baseline_report.json
```

The ordinary single-process layout above intentionally omits the `_parallel`
suffix. Do not point this validator at a partially completed condition; use
the per-condition reports until all six inputs satisfy its contract.
`merge_samia.py` is a separate utility for contiguous `part*` SaMIA shards;
it requires complete ranges 0..4000 and validates the 2,000/2,000 labels.

## Pretraining experiments and historical Wiki preparation

The cost definitions and Pythia 6.9B / MIMIR protocol are documented in
`research/预训练成员评估与成本指标设计_2026-09-10.md`. The runner accepts a frozen MIMIR `--pretraining-manifest` as an alternative
to `--run-dir`; see `../pretraining/README.md` for the Pythia target/draft workflow.
MIMIR labels are preserved and no SFT template or EOS is inserted.

For historical Wiki preparation (CPU/API only):

```bash
bash experiments/baseline/collect_wiki2023.sh
```

It uses a cached Qwen3-8B-Base tokenizer, the existing text and token gates,
a separate 2023 pool and the original WikiTection epoch1 nonmember split.
The final `experiments/data/audits/wiki_temporal_qwen3_8b/audit.jsonl` contains
2000 historical positive proxies + 2000 preserved negative records. Creation
and main-page revision times must be in 2023. The manifest anchors provenance,
counts and the cross-split overlap check. Historical rendering may expand current
templates; dates do not prove inclusion in Qwen training.

The default request rate is 8/minute with concurrency <=3, honoring global
Retry-After cooldowns. This takes hours. Successful API responses survive in
`experiments/data/pools/wiki2023/api_cache/`; re-running reuses them. The wrapper
writes `build.log`, `collector.pid`, and a final `exit_code` in that directory.
Exit code 0 means collection and audit preparation both finished successfully.
For a real publicly usable contact, pass `--contact 'YOUR_CONTACT'` and a chosen
`--request-interval`; do not invent a contact or evade rate limits. Logs exclude
headers and credentials. A completed audit will not be silently overwritten.
