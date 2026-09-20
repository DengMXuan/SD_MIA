# Experiments

This directory contains two independent membership-audit tracks: controlled
post-cutoff instruction-SFT experiments across plain-model, EAGLE-3, and MTP
pairs, and pretrained Pythia 6.9B / 1.4B evaluation using frozen MIMIR labels.
It also contains the target-only baseline suite and the offline SD p/q, PQ-gap,
directional and M1 analyses. Generated pools, checkpoints, caches and reports
live under
`experiments/data/` and `experiments/results/`; both are intentionally ignored
by Git.

The opt-in [DP defense extension](dp_defense/README.md) adds full-parameter
document-level private Qwen training, auxiliary-KD/member-SFT draft variants,
and matched fixed-candidate audits. It uses separate entry points and result
directories so existing non-DP training and audit artifacts remain compatible.

All commands below run from the repository root. When
`CUDA_VISIBLE_DEVICES=<n>` is set, that physical GPU becomes logical `cuda:0`
inside the process, so use `--gpu 0` in the command.

The controlled-SFT source layout is:

```text
experiments/
├── sd_membership_sft/
│   ├── config.py                # shared experiment Config dataclass
│   ├── data.py                  # SFTRecord, prompt masking, collate, split metadata
│   ├── pools.py                 # collect and freeze the WikiTection/NewsTection/ArXivTection pools
│   ├── splits.py                # load a frozen pool into the three-class controlled split
│   ├── training.py              # model loading, LoRA/full SFT, auxiliary distillation, saving
│   ├── generalization.py        # generation-quality generalization check for a finished run
│   ├── retrain_sft.sh           # batch retrain script for the benchmark matrix
│   ├── retrain_unified_matrix.sh # all five pairs with one preflight and shared splits
│   ├── retrain_model_pairs.sh    # two plain-model pairs (standalone child launcher)
│   ├── retrain_speculator_matrix.sh # three EAGLE-3/MTP pairs (child launcher)
│   └── drafts/                  # the three parallel SD draft-adaptation approaches
│       ├── plain.py             # 1/3 plain small causal-LM draft (e.g. Qwen3-1.7B-Base)
│       ├── eagle3.py            # 2/3 published EAGLE-3 speculator head
│       ├── mtp.py               # 3/3 checkpoint's native MTP head
│       ├── common.py            # pair registry, CLI scaffolding, run-config writer
│       └── heads.py             # drafter-head loading for EAGLE-3 / MTP
├── baseline/
│   ├── run.py                    # target-only 11-method MIA runner
│   ├── methods.py                # score/statistic implementations
│   ├── costs.py, runtime.py      # cost accounting and durable progress
│   ├── export_completed.py       # recovery without model inference
│   ├── aggregate_report.py       # completed parallel-matrix report
│   └── README.md                 # methods, provenance, recovery and costs
├── pretraining/
│   ├── prepare.py                # freeze MIMIR records and manifest
│   ├── extract.py                # Pythia p/q and M1 Q/H caches
│   ├── evaluate_baselines.py     # baseline metrics on M1 partitions
│   └── README.md                 # Pythia/MIMIR workflow
├── data/pools/                  # frozen pools (pool.jsonl + SHA-256 manifest each)
└── results/
    ├── sft_runs/                # plain-draft run directories
    └── protocol_ft/             # EAGLE-3 / MTP run directories (historical root name)
```

The original single-pair controlled-SFT matrix is three pools × target epochs
1 and 3:

| Benchmark | Response token band | Target output |
|---|---:|---|
| WikiTection | 128–512 | `results/sft_runs/wikitection_qwen3_8b_epoch{1,3}` |
| NewsTection | 128–512 | `results/sft_runs/newstection_qwen3_8b_epoch{1,3}` |
| ArXivTection | 1024–2048 | `results/sft_runs/arxivtection_qwen3_8b_epoch{1,3}` |

The standard split has 2,000 members, 2,000 nonmembers and 2,000 auxiliary
records per condition. The target-only baseline matrix evaluates those six
saved Qwen3-8B targets with 11 methods; its commands and output contract are
documented in [`baseline/README.md`](baseline/README.md).

Membership is defined by the controlled data construction: member,
nonmember, and auxiliary records are drawn from the same frozen
post-cutoff pool by shuffled, hash-deduplicated assignment. The
target-only baseline suite lives in [`baseline/README.md`](baseline/README.md).
It loads only the saved
fine-tuned target (with adapter base weights when reconstruction is
necessary), never scores an unfine-tuned target or draft model, and reports
all scalar scores in a common member-positive direction.

## 1. Collect and freeze the benchmark pools

The three pools are post-cutoff document collections: pages first created
(Wikipedia), captured (CC-NEWS), or submitted (arXiv) inside a window that
postdates every target model's cutoff. Raw text is persisted so each target
model can re-tokenize; each pool ships a SHA-256 manifest.

```bash
uv run --no-sync python -m experiments.sd_membership_sft.pools wiki --records 6600
uv run --no-sync python -m experiments.sd_membership_sft.pools news --records 6600
uv run --no-sync python -m experiments.sd_membership_sft.pools arxiv --records 6600 --min-chars 4200
```

Each command writes `experiments/data/pools/<pool>/pool.jsonl`
plus `pool.manifest.json`. A `dedupe` subcommand removes near-duplicate
records from an already-frozen pool. The checked-in pools are already
frozen; re-running is only needed to rebuild them from source.

## 2. Draft approaches

An edge-cloud speculative-decoding deployment adapts one of three draft
forms to the fine-tuned target. They are parallel, equally
deployment-shaped options — none is the default or the legacy path:

| Approach | Draft form | Module | Results root |
|---|---|---|---|
| Plain small model | small causal LM (Qwen3-1.7B-Base) | `drafts.plain` | `results/sft_runs/` |
| EAGLE-3 head | published EAGLE-3 speculator head | `drafts.eagle3` | `results/protocol_ft/` |
| Native MTP head | checkpoint's own MTP layer | `drafts.mtp` | `results/protocol_ft/` |

Every approach replays the same frozen pools and writes a
`results.json` whose `config` block is compatible with
`generalization.py`, so fine-tuned targets can be reloaded and scored
the same way regardless of approach.

### 2a. Unified five-pair matrix (recommended)

`retrain_unified_matrix.sh` is the single supported entry point for the full
rerun. It covers 5 model pairs × 3 datasets × target epochs 1/3 × seeds
1919/1949/1978 = 90 conditions. Every condition produces a target plus two
draft variants, for 270 experiment artifacts:

| Child matrix | Pairs | Conditions | Artifacts |
|---|---:|---:|---:|
| Plain Qwen3-8B/1.7B and Gemma4-12B/E2B | 2 | 36 | 108 |
| Qwen3 EAGLE-3, Llama EAGLE-3, Qwen3.5 native MTP | 3 | 54 | 162 |
| Total | 5 | 90 | 270 |

No separate config file is needed. The launcher pins learning rate `2e-5`,
effective batch size 16, and KD temperature 2.0. Auxiliary plain-draft KD and
every EAGLE-3/MTP head branch use 384 optimizer updates; the plain member-data
draft retains its matched 1/3-epoch SFT definition. All three datasets use the
empirically validated micro-batch 2 × accumulation 8. For a condition, `seed`,
`data_seed`, `PYTHONHASHSEED`, split construction, target SFT, and draft/head
training all receive the same numeric seed.

The unified preflight checks every pinned offline model and builds only nine
schema-v2 split manifests: one per dataset and seed, audited under all five
actual training tokenizers. Consequently, all five pairs use exactly the same
raw member/nonmember/auxiliary document IDs for a fixed dataset and seed. An
existing split with a different tokenizer set, pool hash, seed, count, or audit
is rejected rather than silently reused.

Inspect the full command plan without loading models, touching GPUs, or
creating result directories:

```bash
MATRIX_GPUS="3 4 5 6" \
  bash experiments/sd_membership_sft/retrain_unified_matrix.sh --dry-run
```

Before the long run, perform the offline cache/GPU check and CPU-heavy shared
split audit:

```bash
MATRIX_GPUS="3 4 5 6" \
  bash experiments/sd_membership_sft/retrain_unified_matrix.sh --preflight-only
```

When that succeeds, launch the entire experiment:

```bash
nohup env MATRIX_GPUS="3 4 5 6" \
  bash experiments/sd_membership_sft/retrain_unified_matrix.sh \
  > unified_matrix_v1.launch.log 2>&1 &
```

The two child matrices run sequentially because each one already fills all
four GPUs with one single-GPU worker per device. Conditions are never split
across GPUs. If a child matrix fails, the other child matrix still runs; the
unified command finally exits nonzero unless all 90 conditions are complete.
Rerunning the identical command resumes/skips completed artifacts and retries
missing work.

```bash
bash experiments/sd_membership_sft/retrain_unified_matrix.sh --status
tail -f unified_matrix_v1.launch.log
```

The new default root is `experiments/results/sft_runs/unified_matrix_v1/`,
with shared splits in `shared_splits/`, plain-pair outputs in `model_pairs/`,
and EAGLE-3/MTP outputs in `speculator_matrix/`. It does not resume from or
overwrite either earlier `model_pairs/`, `model_pairs_shared_v2/`, or
`speculator_matrix/` roots. Change physical GPUs with `MATRIX_GPUS`; provide
exactly four distinct indices. If sharing busy devices is deliberate,
`MATRIX_SKIP_GPU_BUSY_CHECK=1` disables only the busy-process gate.

### 2b. Plain small-model draft (`drafts.plain`)

Builds the controlled split, fine-tunes the target on member records,
distills an auxiliary-only draft (member-blind, deployment-aligned),
optionally fine-tunes a member-data draft (boundary condition), and
saves everything under `--output-dir`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.drafts.plain \
  --gpu 0 \
  --benchmark wikitection \
  --trainer full \
  --optimizer adamw8bit \
  --target-epochs 3 \
  --target-lr 2e-5 --draft-lr 2e-5 \
  --n-per-class 2000 --n-aux 2000 \
  --target-batch-size 2 --target-grad-accum 8 \
  --draft-batch-size 2 --draft-grad-accum 8 \
  --distill-steps 384 \
  --output-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch3
```

Key flags:

- `--trainer full` fine-tunes every parameter (mainline settings: lr
  `2e-5`, effective batch 16, bf16); `--trainer lora` keeps the adapter
  path. `--optimizer adamw8bit` swaps in a bitsandbytes paged 8-bit AdamW
  so an 8B target fits on one A100-80GB.
- `--benchmark` selects the frozen pool: `wikitection`, `newstection`,
  or `arxivtection`. The controlled matrix uses micro-batch 2 × accumulation
  8 for all three; ArXiv retains its longer 2048-token response band.
- `--n-per-class` member/nonmember records; `--n-aux` auxiliary records
  for draft distillation. Records are selected at load time under the
  target tokenizer's token band (128..512 tokens for Wiki/News,
  1024..2048 for ArXiv) with the fixed instruction prompt; only the
  document continuation contributes loss (prompt masked with `-100`).
- `--split-manifest` loads a preflight-audited shared raw document-ID
  assignment. Matrix launchers require this mode so model pairs receive the
  same member/nonmember/auxiliary documents for a dataset and seed.
- `--skip-trained-drafts` trains the target only; `--skip-training`
  resumes from saved checkpoints.

Every run writes `results.json` (config, data passport, per-record
metadata, training losses) and `RESULTS.md` (human-readable run report)
into the output directory, plus `checkpoints/` (full runs) or `adapters/`
(LoRA runs) for the target and each trained draft variant.

`retrain_sft.sh` reproduces the full benchmark matrix (three pools x
epochs 1 and 3, one benchmark per GPU) and skips conditions whose
`results.json` already exists.

For a standalone two-pair Qwen3-8B/1.7B and Gemma4-12B/E2B run, use the child
shared-split launcher. Its first preflight deterministically filters truncated
token near-duplicates under both pair tokenizers, backfills rejected candidates
from the frozen pool, and writes nine schema-v2 manifests plus audit files:

```bash
bash experiments/sd_membership_sft/retrain_model_pairs.sh --preflight-only

nohup bash experiments/sd_membership_sft/retrain_model_pairs.sh \
  > model_pairs_shared_v2.launch.log 2>&1 &
```

New results default to
`experiments/results/sft_runs/model_pairs_shared_v2/`; the earlier
tokenizer-specific `model_pairs/` artifacts are preserved and are never resumed
into the shared-split rerun. Override physical devices with
`MATRIX_GPUS="3 4 5 6"` as needed.

### 2c. Frozen-target EAGLE-3 and native-MTP matrix

`retrain_speculator_matrix.sh` is the standalone child launcher for the three
pinned target–head pairs:

- `Qwen/Qwen3-8B` + `RedHatAI/Qwen3-8B-speculator.eagle3`
- `unsloth/Meta-Llama-3.1-8B-Instruct` + its RedHatAI EAGLE-3 head
- `Qwen/Qwen3.5-9B-Base` + its original native MTP head

The launcher covers 3 pairs × 3 datasets × target epochs 1/3 × 3 seeds = 54
conditions. Each condition saves exactly three experiment artifacts: a full
target checkpoint, an auxiliary-data head, and a member-data head (162 total).
The target is frozen before either head starts. EAGLE-3 uses KD for both head
branches; MTP uses KD for the auxiliary branch and native MTP cross-entropy for
the member branch. Both branches reload the same immutable original head and
never inherit each other's updates.

No separate config file is required. The launcher fixes full-parameter BF16
target SFT with paged 8-bit AdamW, learning rate `2e-5`, effective batch 16,
and 2,000 member documents. Head training uses 384 optimizer updates (not 384
micro-batches), effective batch 16, learning rate `2e-5`, and KD temperature
2.0 where applicable. All three datasets use the same empirically validated
micro-batch 2 × accumulation 8 configuration.

Inspect the complete plan without touching GPUs or creating files:

```bash
bash experiments/sd_membership_sft/retrain_speculator_matrix.sh --dry-run
```

Then run the offline preflight. It checks that physical GPUs 3–6 are present
and idle, validates every pinned cached revision, and creates nine shared raw
document-ID manifests (three datasets × three seeds). All three model families
therefore receive the same member/nonmember/auxiliary documents for a dataset
and seed. Candidate documents are checked under every tokenizer in seeded
order; exact-token or near-duplicate candidates are rejected and
deterministically backfilled before the shared assignment is frozen. A final
cross-split 13-gram audit fails closed if any tokenizer still exceeds the 80%
threshold. The first preflight is a CPU-heavy tokenization/audit pass; later
launches verify and reuse its immutable manifests and audit attestations.

```bash
bash experiments/sd_membership_sft/retrain_speculator_matrix.sh --preflight-only
```

Start all four single-GPU workers:

```bash
nohup bash experiments/sd_membership_sft/retrain_speculator_matrix.sh \
  > speculator_matrix.launch.log 2>&1 &
```

Each worker exposes one physical GPU through `CUDA_VISIBLE_DEVICES` and the
Python process uses `cuda:0`; a condition itself is not split across GPUs. A
failure is recorded but does not stop unrelated conditions. Relaunching the
same command validates completion markers, reuses a complete frozen target,
and runs only missing head stages. The final launcher exit status is nonzero
if any condition is still incomplete.

```bash
bash experiments/sd_membership_sft/retrain_speculator_matrix.sh --status
tail -f speculator_matrix.launch.log
```

Outputs are under
`experiments/results/sft_runs/speculator_matrix/<pair>/<dataset>/epoch<E>/seed<S>/`;
per-stage logs are in each condition's `logs/` directory. Override the four
exclusive devices with `MATRIX_GPUS="3 4 5 6"`. If GPU sharing is intentional,
`MATRIX_SKIP_GPU_BUSY_CHECK=1` disables only the busy-process gate.

## 3. Generalization check

After fine-tuning (any draft approach), score generation quality on the
run's own training split: greedy 128-token continuations from the first
256 tokens of each document, compared against the true continuations
with BLEU-4 / ROUGE-1 / ROUGE-L.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.generalization \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch3 \
  --gpu 0 --samples 500 --batch-size 8
```

It reports (1) member vs nonmember quality on the fine-tuned model with
bootstrap CIs and a soft no-overfitting gate at |gap| < 0.03, and (2)
base vs fine-tuned quality on the same samples, quantifying what the
fine-tuning cost in generality. `--include-drafts` additionally scores
the draft variants, which matters for speculative-decoding acceptance
quality. Results are written to `GENERALIZATION.md` /
`generalization.json` inside the run directory. `generalization.py` also
provides the model loaders (`load_run_config`, `load_finetuned_model`,
`load_draft_model`) reused by all three draft approaches.

## 4. Target-only baseline matrix

The baseline runner scores only the saved fine-tuned target for each condition;
it does not load a draft, reference model or separately scored pre-SFT target.
The six standard output directories are:

```text
experiments/results/baseline/
├── wikitection_qwen3_8b_epoch1/
├── wikitection_qwen3_8b_epoch3/
├── newstection_qwen3_8b_epoch1/
├── newstection_qwen3_8b_epoch3/
├── arxivtection_qwen3_8b_epoch1/
└── arxivtection_qwen3_8b_epoch3/
```

Run one process per free GPU, using `--gpu 0` after pinning the physical GPU:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.baseline.run \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --gpu 0 --methods all \
  --output-dir experiments/results/baseline/wikitection_qwen3_8b_epoch1
```

`--methods all` runs Loss, Min-K% Prob, Min-K%++, ReCaLL, ICP-MIA, PETAL,
SEAD, WS, RS, BT and SaMIA. Each method is timed independently while one
target model remains resident. A completed condition contains
`baseline_metrics.json`, `baseline_scores.npz`, `BASELINE_RESULTS.md`,
`baseline_costs.json` and `BASELINE_COSTS.md`; the execution directory also
keeps per-method recovery artifacts and `status.json`/`progress.jsonl`.
See [`baseline/README.md`](baseline/README.md) for score definitions,
monitoring, recovery and matrix aggregation.

## 5. SD p/q and M1 analyses

The SD analysis is separate from target-only baselines. It compares the saved
target distribution `p` with the saved draft distribution `q` under the fixed
SFT prompt and response-token contract. `score_role.py` scores one role per
process, `merge_role_logs.py` verifies record/checkpoint provenance and builds
`pq_gap_token_logps.npz`, and `pq_gap_mia.py` computes gap and acceptance
scores. `directional_mia.py` then computes signed, negative-part and fixed
window scores offline; `aggregate_directional.py` combines the six conditions.

For M1, `m1_extract.py` verifies a fresh draft `q` pass against the cached
probabilities before releasing Q/H features. Qwen3-1.7B uses decoder blocks
7/14/21/28 (one-based); Pythia uses 6/12/18/24. `m1_fit.py` performs the
conditional fit and detector selection on the frozen partitions, while
`m1_evaluate.py` aggregates reports and can run the CPU-only B0/B1/B2
probability baselines. These analyses write under
`experiments/results/sft_runs/` and never change the SFT checkpoints.

For the exact role-scoring and merge commands, see the examples in the module
docstrings and the generated `RESULTS.md` files under the corresponding
result root.

## 6. Pretraining/MIMIR track

The Pythia workflow is independent of controlled SFT: MIMIR train/test labels
are externally supplied, the token contract is raw completion without an
instruction template or appended EOS, and the Pythia draft is not an
auxiliary-distilled draft. Follow [`pretraining/README.md`](pretraining/README.md)
for freezing the manifest, running all baselines, extracting p/q/Q/H and
evaluating M1 on the same calibration/test records.

## 7. Accept-only key-token and q-corrected diagnostics

The retained workflow tests the paper's accept-only mechanism in four stages:
q-coordinate invariance (E0), exact-delta token anatomy (E1), draft-q-only
position selection (E1b), and equal-decision active-q replay (E3). Start with
the synthetic q-correction check:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.q_corrected_accept_only
```

Run E1 once per benchmark/epoch condition and then aggregate the six reports:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.token_signal_anatomy \
  --benchmark wikitection --epoch 1
uv run --no-sync python \
  -m experiments.sd_membership_sft.aggregate_token_signal_anatomy
uv run --no-sync python \
  -m experiments.sd_membership_sft.q_only_position_anatomy
```

E3 is a position-locked offline verifier replay, not a natural serial
speculative-decoding trajectory. Run every condition for each frozen replay
seed; the default output path includes the seed so runs cannot overwrite one
another:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.active_importance_replay \
  --benchmark wikitection --epoch 1 --replay-seed 20260914
uv run --no-sync python \
  -m experiments.sd_membership_sft.aggregate_active_importance_replay
```

The low-budget deployable score uses normal-q accept bits at the lowest-q
10%/20%/50% positions, standardizes all three scores on `N_ref`, and
calibrates their maximum on `N_cal`. Active-q results must use the joint
q-corrected estimate in the canonical `log(p)-log(q0)` coordinate. Exact-delta
and oracle-position variants are diagnostics only. The older
`accept_only_mia` pipeline is retained solely as the project's own historical
fixed-q baseline.

The exploratory nonmember-sample, learned-window, EVT-threshold, and
coverage-constrained active-query ablations are kept in one module. A single
condition can run independently so the six conditions can be scheduled in
parallel, then aggregated from their persisted outputs:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.adaptive_window_accept_only \
  --benchmark wikitection --epoch 1
uv run --no-sync python \
  -m experiments.sd_membership_sft.adaptive_window_accept_only \
  --aggregate-existing
```

The nonmember-only neural follow-up trains a multi-scale TCN on real trusted
nonmembers and synthetic positive-delta alternatives. Its token outputs are
also evaluated as adaptive probe priorities. The measurement-value variant
uses only high-query trusted-nonmember trajectories as its teacher:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.neural_adaptive_accept_only \
  --benchmark wikitection --epoch 1
uv run --no-sync python \
  -m experiments.sd_membership_sft.neural_adaptive_accept_only \
  --aggregate-existing
uv run --no-sync python \
  -m experiments.sd_membership_sft.aggregate_neural_adaptive_accept_only
```

The follow-up mechanism checks separate the neural residual, scale choice,
and query allocation claims.  `neural_residual_mechanism` measures pairwise
repairs and regressions rather than treating the neural blend as a black box;
`interpretable_scale_gate` compares fixed scales with q-only and
accept-aware learned gates:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.neural_residual_mechanism
uv run --no-sync python \
  -m experiments.sd_membership_sft.interpretable_scale_gate \
  --benchmark wikitection --epoch 1
uv run --no-sync python \
  -m experiments.sd_membership_sft.interpretable_scale_gate \
  --aggregate-existing
```

A legitimate local-shadow gate is trained only from 400 trusted target
nonmembers: 200 become members of the local 1.7B shadow and 200 remain shadow
nonmembers.  It never consumes a target-member label.  Build one cache per
benchmark, run all six target conditions, and aggregate:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.build_local_shadow_cache \
  --benchmark wikitection --gpu 0
uv run --no-sync python \
  -m experiments.sd_membership_sft.shadow_scale_gate \
  --benchmark wikitection --epoch 1 --device cuda:0
uv run --no-sync python \
  -m experiments.sd_membership_sft.shadow_scale_gate \
  --aggregate-existing
```

The equal-budget allocation check gives every token two pilot queries and
then compares uniform, 6/10 hybrid, one-shot full-AI, sequential full-AI, and
an unavailable-delta oracle at exactly mean K=8.  The post-hoc command uses
paired record resamples for method deltas:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.full_ai_query_allocation \
  --benchmark wikitection --epoch 1 --device cuda:0
uv run --no-sync python \
  -m experiments.sd_membership_sft.full_ai_query_allocation \
  --aggregate-existing
uv run --no-sync python \
  -m experiments.sd_membership_sft.analyze_full_ai_allocation
uv run --no-sync python \
  -m experiments.sd_membership_sft.combine_shadow_active
```

The closed-loop follow-up removes the fixed top-50% rule.  A marginal-value
MLP trained only on local-shadow trajectories recomputes token utility after
each query round, and diminishing-return water filling assigns an unequal
integer number of probes.  It evaluates both a one-query pilot with mean
total K=2 and a two-query pilot with mean total K=8:

```bash
uv run --no-sync python \
  -m experiments.sd_membership_sft.dynamic_marginal_query \
  --benchmark wikitection --epochs 1 3 --device cuda:0
uv run --no-sync python \
  -m experiments.sd_membership_sft.dynamic_marginal_query \
  --aggregate-existing
uv run --no-sync python \
  -m experiments.sd_membership_sft.analyze_dynamic_marginal_query
```

## 8. Conditional nonmember likelihood and counterfactual accept-only probes

`conditional_accept_only` learns a conditional distribution of acceptance
counts directly from local log-q sequences. It uses a small TCN and a finite
binomial mixture including an all-accept atom, without reconstructing target
p or delta. The mixture is a predictive model, not an identifiable estimate
of the target probability or its saturated mass. Only 320 trusted nonmembers
train it; 80 disjoint nonmembers select the checkpoint by count NLL. Another
200 nonmembers calibrate thresholds. The registered T partition is used only
after fitting. No real or synthetic members train/select the predictor.

The member-positive scores mix fixed positive count tilts globally or through
a two-state span prior (enter=1/64, leave=1/8). Those alternatives are explicit
research assumptions. The factorized conditional null is approximate; scores
are not claimed to be e-values. Ordinary NLL is reported as a two-sided
diagnostic. A fixed 0.25 residual fusion and q-only control are also reported.

Run the original-context cached replay on CPU or CUDA:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.conditional_accept_only \
  --benchmark wikitection --epoch 1 --budget 2 --device cpu
```

Outputs include model checkpoints, training history, exact record partitions,
source cache hashes, score arrays, query counts, and `REPORT.md` / `REPORT.json`
under `results/sft_runs/conditional_accept_only/<condition>/b2_seed20260914/`.
The default 30 epochs / 5-epoch patience use nonmember NLL only. Change
`--seed` for independent cached replay/training replicates; `--epoch` denotes target
SFT epochs while `--epochs` denotes detector training epochs.

The counterfactual collector scores the same final 64 response tokens under
the original prefix and a truncated prefix retaining the instruction plus
the immediate 32 response-context tokens. It omits appended EOS and fails
if there is insufficient distant context to remove. Both target p and draft q
are recomputed under each view. Target values stay inside the offline verifier
simulator; only q, independent accept bits and record metadata are exported:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.collect_counterfactual_accept_only \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --output experiments/results/sft_runs/counterfactual_observations/wiki_e1.npz \
  --device cuda:0 --repeats 2
uv run --no-sync python -m experiments.sd_membership_sft.conditional_accept_only \
  --observations experiments/results/sft_runs/counterfactual_observations/wiki_e1.npz \
  --budget 2 --device cpu
```

With `--observations`, `--seed` changes detector training only; the supplied
verifier bits stay frozen. Recollect with a different collector seed for an
independent paired verifier replay.

Paired mode predicts original counts conditioned on both q sequences and the
truncated-view count. At total budget B, original-only receives B original
decisions; paired receives B/2 original plus B/2 truncated decisions. Its
fusion reuses only its original half-budget. The collector stores additional
bits for the original-only control; report collection cost separately from
each detector's used queries. Compare these methods on the paired archive's
identical suffix tokens, not against historical whole-document metrics.
Truncation also changes context length and absolute positions, so it cannot
alone establish that a response difference is caused by memorization.
Both workflows remain position-locked verifier simulations, not live serial
SD trajectories. GPU model inference is needed for practical collection of
the saved 8B/1.7B checkpoints; cached detector fitting works on CPU.

Aggregate frozen runs and optionally compare the existing K=2 neural fusion:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.analyze_conditional_accept_only \
  --legacy-root experiments/results/sft_runs/accept_only_active_v2/neural_adaptive/conditions
```

Legacy comparisons require identical record IDs, labels and low-q scores.
Paired archives never reuse whole-document legacy scores. Bootstrap intervals
resample the same test records across methods and seeds within each condition;
they condition on fitted models and do not quantify retraining uncertainty.
The analyzer defaults to cached whole-document runs. Use `--scope paired`
to analyze supplied counterfactual archives separately; the two candidate
scopes are never pooled into one macro average.

## 9. Frozen-checkpoint validation of the four directions

The scope, fixed alternatives, splits and query budgets are recorded in
[`DIRECTIONS_VALIDATION.md`](DIRECTIONS_VALIDATION.md). Target and draft
parameters stay frozen; all new training is restricted to small nonmember
detectors. No shadow members or synthetic membership labels are used.

Resume the original-context three-seed matrix, or train paired detectors
after collecting frozen-checkpoint observations:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.run_direction_matrix conditional
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 uv run --no-sync python \
  -m experiments.sd_membership_sft.collect_counterfactual_accept_only \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --output experiments/results/sft_runs/counterfactual_observations/wikitection_epoch1.npz \
  --batch-size 8 --repeats 8 --replay-seeds 20260914,20260915,20260916
uv run --no-sync python -m experiments.sd_membership_sft.run_direction_matrix paired
```

Repeat collection for the three datasets and two target epochs. Multi-seed
collection reuses the same frozen-model forward passes, with independent bit
randomness. `run_direction_matrix paired` evaluates B=2 and B=8; it refuses
missing archives unless `--ready-only` explicitly permits a partial batch.
Completed detector runs are skipped only when their report and source
manifest both exist.

`active_protocol_design` learns a q-conditioned latent probability mixture
through nonmember multi-q accept-count likelihoods. It has no exact-p/delta
regression target. The fixed alternatives increase latent log probability
by .5/1/2. A Jensen–Shannon utility selects proposal levels and/or positions.
The five policies share one normal-q pilot, the same observable null model,
coupled verifier random streams, and exact B=1/2/4/8 decision checkpoints.
Training/selection uses ten decisions per nonmember token, reported separately.
Early positive stopping is calibrated on nonmember path maxima, accounting
for the registered multiple looks rather than repeatedly testing an ordinary
single-look threshold. This remains position-locked offline replay.

```bash
uv run --no-sync python -m experiments.sd_membership_sft.run_direction_matrix active
```

`serial_accept_only` performs actual sampled speculative decoding with the
saved target and draft. It uses the residual correction distribution after
rejection, a target bonus token after full acceptance, and crops both KV caches
before continuing. EOS can terminate early. The archive includes only reached
accept/reject bits, local draft logq/entropy, round/position features and costs;
target probabilities and correction-token identities never enter the detector.
A causal GRU predicts nonmember acceptance hazards. Fixed positive, negative,
and two-sided logit-tilt evidence are compared with acceptance rate, rejection
rate, and absolute deviation from nonmember validation acceptance rate on
identical transcripts. These directions were fixed before inspecting natural
SD metrics, since sampled-proposal acceptance averages 1-TV(p,q) and need not
increase for members. All directions are reported without member-based selection. This natural
generation experiment must not be pooled with fixed-candidate replay.

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 uv run --no-sync python \
  -m experiments.sd_membership_sft.serial_accept_only collect \
  --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch1 \
  --output-dir experiments/results/sft_runs/directions_validation/serial/wikitection_epoch1
uv run --no-sync python -m experiments.sd_membership_sft.serial_accept_only evaluate \
  --output-dir experiments/results/sft_runs/directions_validation/serial/wikitection_epoch1
```

Collection resumes per-record atomic checkpoints. For multiple GPUs, run
non-overlapping `--shard-count 2 --shard-index 0` / `--shard-index 1` workers
with the same output directory. The last worker merges all records. The
natural-SD pilot covers Wiki epochs 1/3, seed 20260914, gamma=4, at most eight
verification rounds and 1,400 records per checkpoint. The detector fit uses
the same 320/80/200 nonmember train/validation/calibration roles and 800-record
test set; no LM weights are updated.

After all registered runs finish, build the paired comparison report:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.summarize_direction_validation
```

`DIRECTIONS_REPORT.md/json` reports per-condition outcomes, paired bootstrap
intervals, nominal TPR and actual FPR. The same test-record draws are reused
across seeds and target checkpoints that share a test set. Intervals condition
on fitted models; they do not establish robustness to new training datasets.

The completed local matrix includes 18 conditional runs, 18 active-policy runs,
36 paired-prefix runs and two natural-SD runs. See the
[Chinese findings](results/sft_runs/directions_validation/FINDINGS_ZH.md) and
[full comparison report](results/sft_runs/directions_validation/DIRECTIONS_REPORT.md)
for all outcomes, including negative findings and actual false-positive rates.

## 10. Priority-ordered conditional-model follow-ups

The fixed scope is in [PRIORITY_VALIDATION.md](PRIORITY_VALIDATION.md).
`priority_accept_only` tests a causal accept-count history branch, extra frozen
local-draft difficulty features, three-initialization PMF averaging and a fixed
uncertainty discount, expanded/grouped nonmember calibration, sparse evidence,
and a pilot-plus-second-query allocation policy. Target and draft weights are
never updated. No real or synthetic member labels train/select any component.

Historical M1 caches supply only their raw draft q/entropy/rank/margin features;
none of their fitted membership detectors or target-score features are used.
Their q differs numerically from historical replay q, so the feature ablation
regenerates bits with that q and refits its own matched q-only baseline. Feature
results must not be compared to the historical baseline as if inputs matched.
The missing News/ArXiv epoch-3 caches can be collected with:

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python \
  -m experiments.sd_membership_sft.collect_draft_difficulty \
  --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch3 \
  --output-dir experiments/results/sft_runs/priority_validation/features/newstection_epoch3
```

Use the corresponding ArXiv run/output paths for its epoch-3 cache. Extraction
resumes from a memmapped feature array and a completed-record bitmap. All
feature collections retain record identity, exclude appended EOS and record
frozen-checkpoint provenance.

```bash
uv run --no-sync python -m experiments.sd_membership_sft.run_priority_matrix sequence --gpus 4,5,6 --jobs 3
uv run --no-sync python -m experiments.sd_membership_sft.run_priority_matrix features --gpus 4,5,6 --jobs 3
uv run --no-sync python -m experiments.sd_membership_sft.run_priority_matrix posthoc --jobs 3
uv run --no-sync python -m experiments.sd_membership_sft.summarize_priority_validation
```

Omit `--gpus` to fit the small detectors on CPU. A runner skips completed
reports. `features --ready-only` explicitly allows a partial batch while a
collector is running; strict summarization requires all 18 condition/seed
reports in each of the three phases. The causal branch is separate from the
bidirectional static-q branch and sees only shifted past counts. Its likelihood
must not be interpreted as recovering target p or as an exact e-value.

Calibration expands the original 200 to 1200 nonmembers disjoint from training,
validation and test. Both fixed binary partitions (length and mean logq) use
reference medians. Report actual FPR alongside TPR and group-level errors.
Allocation comparisons both spend L+floor(L/2) decisions, use the same pilot,
and score every observed decision with a shared latent-mixture model.

The complete 54-run follow-up is summarized in the
[Chinese findings](results/sft_runs/priority_validation/FINDINGS_ZH.md) and
[paired comparison report](results/sft_runs/priority_validation/PRIORITY_REPORT.md).
These report every registered variant, validation likelihoods, calibration
changes and condition-specific errors. The combination follow-up is below.

## 11. Feature, sparse-score and calibration combinations

[COMBINATION_VALIDATION.md](COMBINATION_VALIDATION.md) fixes a 2×2×3 factorial:
q/position versus added draft difficulty features; global versus sparse
evidence; pooled 200 versus pooled 1200 versus difficulty-grouped 1200
nonmember calibration. All twelve configurations use identical
feature-consistent q, B=2 feedback and test records within each run. Both
saved small detectors are frozen and reused, with no new fitting.

```bash
uv run --no-sync python -m experiments.sd_membership_sft.combined_accept_only matrix --gpus 4,5,6 --jobs 3
uv run --no-sync python -m experiments.sd_membership_sft.summarize_combination_validation
```

Omit `--gpus` for CPU inference. The runner checks record/split alignment and
reproduces saved global scores before evaluating sparse combinations. Each
score is calibrated independently. The summary requires all six conditions
and three seeds, reports 216 factorial cells, paired ranking/decision
differences, feature×sparsity interaction and conditional false-positive rates.
The primary full combination is fixed in advance; other predefined cells
remain visible even when they show better operating-point tradeoffs.

See [Chinese findings](results/sft_runs/combination_validation/FINDINGS_ZH.md)
and the [complete report](results/sft_runs/combination_validation/COMBINATION_REPORT.md).

## Tests

```bash
uv run --no-sync python -m pytest tests/ -q
```

The suite covers frozen-pool construction and generalization scoring together
with q-coordinate correction, exact-delta anatomy, draft-q-only position
selection, equal-budget active replay, and the retained historical fixed-q
baseline.
