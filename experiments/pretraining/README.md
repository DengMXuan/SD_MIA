# Pythia / MIMIR pretraining membership evaluation

This path evaluates a pretrained **EleutherAI/pythia-6.9b** target and a
pretrained **EleutherAI/pythia-1.4b** draft, without fine-tuning either model.
The exact model revisions and the MIMIR dataset revision are pinned in
`pretraining/data.py`; do not mix Pythia's standard and `-deduped` variants.
The draft also trained on The Pile, so it is not the member-blind
auxiliary-distilled draft used in the controlled-SFT experiment.

This workflow is a raw-completion protocol, not the instruction-SFT protocol:
there is no chat/instruction template, no added BOS/EOS, the first text token
is context only, and positions 1..L-1 are scored. The Pythia LM heads may have
padded rows; scoring and generation restrict the distribution to the real
shared tokenizer vocabulary and record that contract in every manifest.

Generated data and reports use this layout:

```text
experiments/data/pretraining/<name>/
├── records.jsonl
└── manifest.json

experiments/results/pretraining/<name>/
├── baseline/
│   ├── baseline_metrics.json
│   ├── baseline_scores.npz
│   └── BASELINE_RESULTS.md
└── m1/
    ├── probabilities/
    └── features/
```

The generated directories are ignored by Git. A frozen manifest is never
silently overwritten.

All commands run from the repository root. Use a free GPU; existing
experiments are not stopped or automatically rescheduled by these commands.
When pinning a physical GPU with `CUDA_VISIBLE_DEVICES=<n>`, use
`--device cuda:0` or `--gpu 0` inside the process.

## 1. Freeze official MIMIR labels and text

```bash
./.venv/bin/python -m experiments.pretraining.prepare \
  --source 'wikipedia_(en)' --split ngram_13_0.8 \
  --n-per-class 800 --n-aux 100 \
  --output-dir experiments/data/pretraining/mimir_wikipedia
```

The official per-domain caches contain 1000 samples per class. The default
uses 800 members, 800 nonmembers and 100 auxiliary records; the auxiliary
records are held out of both audit classes and come from the nonmember side of
the frozen source. `train` remains member and `test` remains nonmember.
Shuffling only selects/orders records within their original class. Too few
surviving records or contradictory cross-label token duplicates fail
explicitly.

The downloader pins MIMIR revision `02500d3b7cece0cb7628e939ba9fc93fdb6362ae`
and reads official JSONL caches without executing a Hugging Face dataset script.
If downloading is unavailable, import the same official cache files:

```bash
./.venv/bin/python -m experiments.pretraining.prepare \
  --source 'wikipedia_(en)' --split ngram_13_0.8 \
  --member-file /absolute/path/to/train/wikipedia_\(en\)_ngram_13_0.8.jsonl \
  --nonmember-file /absolute/path/to/test/wikipedia_\(en\)_ngram_13_0.8.jsonl \
  --output-dir experiments/data/pretraining/mimir_wikipedia
```

Official cache rows are JSON strings. Objects with a `text` field are accepted
for local exports. Input-file SHA-256 hashes, source row IDs, original text,
selected token IDs, labels, tokenizer fingerprint and exact model revisions are
frozen in `records.jsonl` / `manifest.json`. Existing frozen data is not
overwritten. Local imports are recorded as user-supplied caches, not falsely
asserted to have been downloaded from the pinned revision.

Supported Pile sources: arxiv, dm_mathematics, github, hackernews, pile_cc,
pubmed_central, wikipedia_(en), full_pile. The official 10k cache is available
for `--source full_pile --split none --cache-size 10000`; use
`--n-per-class 2000 --n-aux 2000` for a larger mixed-domain audit. Its mixed-domain
result is not a per-source or macro-average result. Unknown cache combinations
fail in the downloader instead of silently substituting another split.

`prepare.py` is an import/freeze step only: it does not load a language model.
Use a new `--output-dir` for a different count, source or split.

Token contract: raw completion, no instruction/chat template, no added BOS/EOS.
The first real text token is context only; tokens 1..L-1 are scored. Default
truncation is 512 text tokens. Pythia's real tokenizer vocabulary is shared but
its two LM heads have different padded sizes. Scoring excludes padded rows and
renormalizes over real tokenizer IDs; generation suppresses padded rows too.
This is explicitly recorded and is not claimed to reproduce an implementation
that includes padded rows in its normalizer.

## 2. Run all target-only baselines

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m experiments.baseline.run \
  --pretraining-manifest experiments/data/pretraining/mimir_wikipedia/manifest.json \
  --gpu 0 --methods all --attn-implementation sdpa --generation-batch-size 8 \
  --output-dir experiments/results/pretraining/mimir_wikipedia/baseline
```

All 11 existing methods use the same scoring functions, progress logging and
per-method saving as SFT runs. The three cost metrics are recorded automatically;
methods run independently for comparable costs (see the baseline guide).
Only the target model is loaded here. The existing
PETAL/SEAD and label-only target-only adaptation caveats still apply. ReCaLL,
ICP-MIA and PETAL require disjoint auxiliary records; use a method subset when
preparing with `--n-aux 0`.

The baseline output is not complete until `baseline_metrics.json` and
`baseline_scores.npz` exist. During a long run, monitor the execution
directory's `status.json` and `progress.jsonl`; a completed method is saved as
an atomic NPZ before the next method starts. See the recovery instructions in
[`../baseline/README.md`](../baseline/README.md).

## 3. Extract p/q and M1 Q/H features

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m experiments.pretraining.extract \
  --manifest experiments/data/pretraining/mimir_wikipedia/manifest.json \
  --device cuda:0 --batch-size 1 --attn-implementation sdpa \
  --output-dir experiments/results/pretraining/mimir_wikipedia/m1
```

Models are loaded sequentially, not simultaneously. Completed target/draft
probabilities are saved separately, then merged into the existing p/q cache
format under `probabilities/`. `--stage probabilities` stops after the p/q
cache; `--stage features` reuses an already validated probability cache and
only extracts Q/H. A validated completed probability cache can be reused;
partial role archives are saved for inspection, not automatically resumed.

The feature pass creates `q.npy`, `h.npy`, position metadata, a feature
manifest and a frozen `partition_manifest.json` under `features/`. It checks
fresh draft q values against the saved probability cache before atomically
releasing the feature directory. A failed or interrupted pass leaves partial
files for diagnosis, but those files are not a valid M1 input.

For Pythia 1.4B, M1 hooks the residual output of blocks **6, 12, 18, 24**
(one-based; zero-based 5,11,17,23), before the final GPT-NeoX LayerNorm. These
quarter-depth positions replace Qwen's 7,14,21,28. The numerical definitions
are unchanged: six Q features plus ten activation statistics at each of four
blocks. Fresh Q log-probabilities must match the independently extracted draft
cache.
Dataset/model provenance, arrays and record order are verified before fitting.

The extraction writes `features/partition_manifest.json` before any fitting:
stratified 40% train / 20% validation / 20% calibration / 20% test. Half the train
nonmembers fit the nuisance model, split 75/25 for location/scale; the other half
and all train members fit detectors. At 800 per class this gives Nμ=120, Ns=40,
D=320 members+160 nonmembers, and V/C/T each 160+160. Calibrated 1% FPR resolution
is limited by 160 calibration nonmembers (p-value spacing 1/161).
Small datasets of at least 40 per class are supported for smoke tests, not
meaningful low-FPR claims. These scaled splits are separate from the original
2000-per-class Qwen preregistration.

## 4. Fit the existing M1 method

```bash
./.venv/bin/python -u -m experiments.sd_membership_sft.m1_fit \
  --feature-dir experiments/results/pretraining/mimir_wikipedia/m1/features \
  --probability-dir experiments/results/pretraining/mimir_wikipedia/m1/probabilities \
  --role draft_pretrained --conditional-family linear --detector-families logistic \
  --device cpu \
  --output-dir experiments/results/pretraining/mimir_wikipedia/m1/fit_linear_logistic
```

This reuses the existing conditional Q/QH calibration, detectors, validation-only
selection, conformal thresholds and test reporting. Existing MLP and activation
ablation flags remain available. No MIMIR test labels train the target/draft or
fit the nuisance/detector models. Reports state `training_regime=pretraining`
and record the pretrained role and model identities. The primary outputs are
`m1_metrics.json` and `M1_RESULTS.md`; `m1_evaluate.py` aggregates completed
per-condition reports into `aggregate_metrics.json`.

For an aligned comparison, evaluate baselines on the **same M1 C/T records**:

```bash
./.venv/bin/python -m experiments.pretraining.evaluate_baselines \
  --manifest experiments/data/pretraining/mimir_wikipedia/manifest.json \
  --baseline-dir experiments/results/pretraining/mimir_wikipedia/baseline \
  --partition-manifest experiments/results/pretraining/mimir_wikipedia/m1/features/partition_manifest.json \
  --output experiments/results/pretraining/mimir_wikipedia/baseline/m1_matched_metrics.json
```

Do not directly compare the baseline full-audit AUC with M1's test-subset AUC.
Run each source separately and report its actual calibration/test counts.

## Validation

`tests/test_pretraining.py` uses local, randomly initialized tiny GPT-NeoX models
with different padded vocab sizes. It runs all baselines, extracts p/q + Q/H,
fits M1, checks original labels/no-EOS masks, disjoint partitions and detects
cache tampering. It requires no network or large checkpoint. It verifies the
integration, not Pythia 6.9B accuracy/performance. During implementation the
official dataset file endpoint returned HTTP 401; a full real-model MIMIR run
was not performed.

Sources: [Pythia](https://github.com/EleutherAI/pythia),
[MIMIR code](https://github.com/iamgroot42/mimir),
[MIMIR data](https://huggingface.co/datasets/iamgroot42/mimir).
