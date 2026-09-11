# Experiments

Pretrained Pythia 6.9B / 1.4B with MIMIR is supported by
[the pretraining workflow](pretraining/README.md), including all baselines and M1.
The controlled-SFT workflow below remains separate.

Controlled post-cutoff instruction-SFT experiments: dataset collection,
processing, model download/loading, fine-tuning, and post-training quality
checks. Experiment code is separated from generated artifacts:

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
│   └── drafts/                  # the three parallel SD draft-adaptation approaches
│       ├── plain.py             # 1/3 plain small causal-LM draft (e.g. Qwen3-1.7B-Base)
│       ├── eagle3.py            # 2/3 published EAGLE-3 speculator head
│       ├── mtp.py               # 3/3 checkpoint's native MTP head
│       ├── common.py            # pair registry, CLI scaffolding, run-config writer
│       └── heads.py             # drafter-head loading for EAGLE-3 / MTP
├── baseline/
│   ├── run.py                    # target-only MIA baseline runner
│   ├── methods.py                # pure score/statistic implementations
│   └── README.md                 # methods, provenance, and usage
├── data/pools/                  # frozen pools (pool.jsonl + SHA-256 manifest each)
└── results/
    ├── sft_runs/                # plain-draft run directories
    └── protocol_ft/             # EAGLE-3 / MTP run directories (historical root name)
```

Membership is defined by the controlled data construction: member,
nonmember, and auxiliary records are drawn from the same frozen
post-cutoff pool by shuffled, hash-deduplicated assignment. The
membership-scoring and attack-method code that used to live in this
package has been removed; the target-only baseline suite now lives in
[`baseline/README.md`](baseline/README.md). It loads only the saved
fine-tuned target (with adapter base weights when reconstruction is
necessary), never scores an unfine-tuned target or draft model, and reports
all scalar scores in a common member-positive direction.

All commands below are run from the repository root. Prefix with
`CUDA_VISIBLE_DEVICES=<n>` to pin a physical GPU (its logical index is
then `cuda:0`, selected via `--gpu 0`).

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

### 2a. Plain small-model draft (`drafts.plain`)

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
  or `arxivtection` (ArXiv keeps its 2048-token band; use
  `--target-batch-size 1 --target-grad-accum 16` there, likewise for the
  draft flags).
- `--n-per-class` member/nonmember records; `--n-aux` auxiliary records
  for draft distillation. Records are selected at load time under the
  target tokenizer's token band (128..512 tokens for Wiki/News,
  1024..2048 for ArXiv) with the fixed instruction prompt; only the
  document continuation contributes loss (prompt masked with `-100`).
- `--skip-trained-drafts` trains the target only; `--skip-training`
  resumes from saved checkpoints.

Every run writes `results.json` (config, data passport, per-record
metadata, training losses) and `RESULTS.md` (human-readable run report)
into the output directory, plus `checkpoints/` (full runs) or `adapters/`
(LoRA runs) for the target and each trained draft variant.

`retrain_sft.sh` reproduces the full benchmark matrix (three pools x
epochs 1 and 3, one benchmark per GPU) and skips conditions whose
`results.json` already exists.

### 2b. EAGLE-3 speculator head (`drafts.eagle3`)

Pairs: `qwen3_8b_eagle3` (Qwen/Qwen3-8B +
RedHatAI/Qwen3-8B-speculator.eagle3) and `llama31_8b_eagle3`
(unsloth/Meta-Llama-3.1-8B-Instruct + its EAGLE-3 speculator). Stages,
run in order:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.drafts.eagle3 \
  --gpu 0 --pair qwen3_8b_eagle3 --epochs 3 eagle-target
uv run --no-sync python -m experiments.sd_membership_sft.drafts.eagle3 \
  --gpu 0 --pair qwen3_8b_eagle3 --epochs 3 eagle-head --variant aux
```

- `eagle-target`: full-parameter member SFT of the EAGLE-line target
  (trunk hyperparameters mirror the mainline settings: lr 2e-5,
  effective batch 16, PagedAdamW8bit, bf16).
- `eagle-head --variant aux|member`: KD continue-training of the EAGLE-3
  head against the fine-tuned target (`aux` uses only document-disjoint
  auxiliary records; `member` is the same-member boundary condition).

### 2c. Native MTP head (`drafts.mtp`)

Pair: `qwen35_9b_mtp` (Qwen/Qwen3.5-9B-Base). Stages, run in order:

```bash
uv run --no-sync python -m experiments.sd_membership_sft.drafts.mtp \
  --gpu 0 --pair qwen35_9b_mtp mtp-prehead
uv run --no-sync python -m experiments.sd_membership_sft.drafts.mtp \
  --gpu 0 --pair qwen35_9b_mtp mtp-joint
uv run --no-sync python -m experiments.sd_membership_sft.drafts.mtp \
  --gpu 0 --pair qwen35_9b_mtp mtp-adapt
```

- `mtp-prehead`: convert the checkpoint's native depth-1 MTP layer into
  a speculators model (the "pre-head"; no extra training).
- `mtp-joint`: joint fine-tune trunk + native MTP head on member
  documents with `LM CE + lambda_mtp * MTP CE` (the "MTP trains
  together" condition).
- `mtp-adapt`: member-blind KD adaptation of the joint head to the
  joint target on auxiliary documents.

Outputs land under `experiments/results/protocol_ft/<pair>/` (the root
keeps its historical name).

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

## Tests

```bash
uv run --no-sync python -m pytest tests/ -q
```

The suite covers the frozen-pool loader (three-class disjointness, token
band, 13-gram cross-split gate) and the generalization-check scoring
logic.
