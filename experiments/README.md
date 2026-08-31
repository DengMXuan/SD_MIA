# Experiments

The experiment code is separated from generated artifacts:

```text
experiments/
├── sd_membership_pilot.py       # legacy GPT-2 pilot, kept for provenance
├── sd_membership_sft/           # Qwen3 instruction-SFT implementation
│   ├── audit.py                  # membership scores and transcript simulation
│   ├── config.py                 # reproducible experiment configuration
│   ├── data.py                   # source extraction and controlled SFT split
│   ├── draver_activation.py      # verifier-conditioned all-layer draft features
│   ├── p0_matrix.py              # seed/epoch stability and large-audit matrix
│   ├── p1_passive_l2.py          # natural draft + min(1,p/q) passive L2 protocol
│   ├── p2_activation.py           # NART-inspired DraVer-Act evaluation
│   ├── public_benchmark.py        # deployment-aligned public FineWeb benchmark
│   ├── public_budget_sweep.py     # nested verifier-feedback budget scan
│   ├── public_data.py             # manifest-checked public split loader
│   ├── public_fineweb.py          # bounded FineWeb snapshot constructor
│   ├── build_sft_public_datasets.py # WikiText/XSum/CNN-DM frozen pools
│   ├── independent_adapter_reanalysis.py # fair transcript adapter remeasurement
│   ├── protocol_features.py       # shared EAGLE/MTP/DSpark feature boundary
│   ├── eagle3_protocol_benchmark.py # hidden-conditioned EAGLE-3 validation
│   ├── mtp_protocol_benchmark.py  # exported native-MTP validation
│   ├── dspark_protocol_benchmark.py # DSpark block validation
│   ├── training.py               # LoRA SFT, distillation, and feature extraction
│   └── runner.py                 # command-line experiment entry point
├── configs/                      # checked-in experiment configurations
└── results/
    ├── pilot/                    # legacy GPT-2 high-memory result
    ├── pilot_epoch1/             # legacy GPT-2 weak-memory result
    └── qwen3_sft/                # Qwen3 SFT outputs, P0 and P1 reports
```

Run the new pipeline from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.sd_membership_sft.runner --gpu 0
```

`CUDA_VISIBLE_DEVICES=1` makes the physical GPU 1 appear as logical `cuda:0`; this
keeps the other GPUs out of the process. The target is Qwen3-8B-Base and the
draft is Qwen3-1.7B-Base by default.

The target and the two trained draft variants use LoRA instruction SFT. The
member response is the only part contributing to the language-model loss; the
user prompt is masked with `-100`. The raw PDF text is reconstructed at runtime
and is not written to result artifacts.

## Completed P0/P1 runs

P0 uses a fixed data split and audit split, five training seeds, target epochs
`0/1/2/4`, a 320/class stability matrix, and 1,048/class large-audit
conditions. Its report is in
[`results/qwen3_sft/p0_matrix/SUMMARY.md`](results/qwen3_sft/p0_matrix/SUMMARY.md).

P1 uses the actual passive L2 protocol: Qwen3-1.7B-Base naturally samples a
four-token block, Qwen3-8B-Base applies `min(1,p/q)`, and rejection samples a
correction from `(p-q)+`. Results for the epoch1 and epoch4 adapters are in
[`results/qwen3_sft/p1_passive_l2/SUMMARY.md`](results/qwen3_sft/p1_passive_l2/SUMMARY.md).

Run the P0 matrix with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.p0_matrix
```

Run one P1 condition with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.p1_passive_l2 \
  --gpu 0 \
  --target-adapter experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch1/adapters/target \
  --output-dir experiments/results/qwen3_sft/p1_passive_l2/epoch1
```

P2 reuses a checked target adapter and evaluates four separated signals:
draft-only NART-style all-layer activations, verifier transcript only, naive
concatenation, and the proposed verifier-conditioned activation trajectory. It
also includes a q-stratified shuffled-transcript control and explicit
representation-train/support/test separation. Run the epoch-1 pilot with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.p2_activation \
  --gpu 0 \
  --source-results experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch1/results.json \
  --output-dir experiments/results/qwen3_sft/p2_draver_act_epoch1
```

## Public controlled-SFT validation

The current fair comparison defines `transcript-only` as cloud verification
feedback plus protocol-public position/depth/visibility only. It excludes local
`q_logp`, private confidence heads, speculator states, and connector hidden
states. Unit tests enforce this information boundary. Older FineWeb v2 and
Granite artifacts predate this correction and are retained as historical
mechanism/deployment controls; the fair main endpoints are the corrected Qwen3
three-dataset matrix and the protocol results linked below.

The public validation follows the controlled fine-tuning structure of NART but
changes the measured object for edge–cloud speculative decoding. The current
benchmark keeps the old base-draft/adapted-target condition as an explicit
deployment-mismatch control, then evaluates two deployment-aligned drafts:

- `aux_distill`: the adapted target distills the draft only on document-disjoint
  auxiliary records, so the draft does not directly read members;
- `member_sft`: target and draft receive the same member SFT, reported as a
  realistic boundary condition with a potentially strong draft-only signal.

Every pair is diagnosed with the exact teacher-forced speculative-sampling
acceptance `sum_v min(p(v), q(v)) = 1 - TV(p,q)`, top-1 agreement, and candidate
logp RMSE. Draft/target vocabulary mappings and special-token IDs must match
exactly before training starts.

The checked FineWeb snapshot contains 600 filtered documents from
`CC-MAIN-2025-26`. Its SHA-256 is
`7855bc5485ea6892ecbdc13157162a636b34ccec5bf65435f551bf262a5f869a`.
The historical v2 protocol used 160 records per class and left the draft
untouched; its cross-model results remain useful as target-only mismatch
controls. Model revisions, data provenance, and all uncertainty estimates are
in [`results/public_sft/SUMMARY.md`](results/public_sft/SUMMARY.md).

Main v2 AUCs are:

| Model pair | DraVer-Act | Draft NART-style | Transcript-only | q-bin shuffle |
|---|---:|---:|---:|---:|
| Qwen3-1.7B → Qwen3-8B | 0.9995 | 0.6392 | 0.9960 | 0.4887 |
| Pythia-410M → Pythia-1.4B | 0.8177 | 0.4452 | 0.8239 | 0.4469 |
| GPT-2 → GPT2-XL | 0.6374 | 0.5235 | 0.6336 | 0.5435 |

These runs support an edge–cloud alignment effect relative to directly applying
NART-style features to the draft. They do not establish a stable activation
advantage over transcript-only at the same 624-bit budget; all three paired
confidence intervals for that stronger comparison include zero.

### Deployment-aligned Granite 4.0 validation

The new-model validation uses `granite-4.0-350m-base` as the draft and
`granite-4.0-1b-base` as the target. Both are revision-pinned Apache-2.0 Base
models released by IBM in October 2025. The completed single-seed run uses
96/96/96 member/nonmember/auxiliary records, 32 calibration records per class,
64 response tokens, three target/draft SFT epochs, and 120 auxiliary
distillation steps.

| Pair endpoint | Exact acceptance | Verifier mean AUC | DraVer-Act AUC | Draft NART-style AUC |
|---|---:|---:|---:|---:|
| base/base | 0.705 | 0.477 | 0.469 | 0.477 |
| base/adapted target | 0.588 | 0.919 | 0.923 | 0.419 |
| auxiliary-distilled/adapted target | 0.641 | **0.938** | 0.878 | 0.616 |
| member-SFT/adapted target | 0.669 | 0.702 | 0.605 | 0.562 |

The result file is
[`deployment_aligned/granite4_350m_to_1b_epoch3_seed20260828/RESULTS.md`](results/public_sft/deployment_aligned/granite4_350m_to_1b_epoch3_seed20260828/RESULTS.md).
The auxiliary-only condition restores 45.4% of the acceptance loss caused by
target-only adaptation. DraVer-Act remains well above draft-only and the
q-bin-shuffled control, but does not beat transcript-only; the protocol-native
q-aware acceptance score is therefore the most robust current method.

The Granite transcript comparison above used the earlier q-aware transcript
definition. It must not be used as the final pure transcript-only comparison;
the deployment-alignment and draft-only findings remain useful, while fair
incremental claims use the corrected runs below.

### Complete Qwen3 and protocol matrix (2026-08-28)

The completed main matrix uses WikiText-103 raw, XSum, and CNN/DailyMail with
512 member / 512 nonmember / 512 auxiliary records per dataset. Qwen3-8B-Base
receives three epochs of member SFT, and Qwen3-1.7B-Base is aligned with 384
steps of auxiliary-only KD; same-member draft SFT is a separate boundary
condition. Saved adapters were then remeasured with the pure transcript
baseline.

| Dataset | Proposed | Pure transcript | Δ proposed−transcript (95% CI) |
|---|---:|---:|---:|
| WikiText-103 | 0.9944 | 0.9985 | -0.0041 [-0.0074, -0.0014] |
| XSum | 0.9985 | 0.9996 | -0.0011 [-0.0026, -0.0001] |
| CNN/DailyMail | 0.9992 | 0.9996 | -0.0004 [-0.0014, 0.0003] |

The method does not improve over transcript-only in this matrix; the baseline
is nearly saturated. Protocol-specific extensions were also completed:

| Protocol | Proposed | Pure transcript | Δ proposed−transcript (95% CI) |
|---|---:|---:|---:|
| EAGLE-3 / Qwen3-8B | 0.6319 | 0.5677 | +0.0642 [+0.0107, +0.1151] |
| Native MTP / Qwen3.5-4B | 0.5252 | 0.5088 | +0.0164 [-0.0541, +0.0855] |
| DSpark / Qwen3.6-35B-A3B | 0.5056 | 0.4901 | +0.0156 [-0.1021, +0.1169] |

EAGLE-3 is positive relative to transcript but not relative to the stratified
shuffle control, so it is limited evidence rather than a resolved alignment
effect. Its current transcript is a mapped-token `p/q` block simulation, not
native EAGLE tree decoding. MTP and DSpark do not separate from transcript and
currently use teacher-forced greedy block matches; production generated-chain
validation remains future work. See the
[`protocols/RESULTS.md`](results/public_sft/protocols/RESULTS.md) report and
[`protocols/VALIDATION.md`](results/public_sft/protocols/VALIDATION.md) for all
controls, limitations, and the 11/11 statistical fallacy scan.

Run the deployment-aligned matrix with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.public_benchmark \
  --gpu 0 \
  --draft-model ibm-granite/granite-4.0-350m-base \
  --draft-revision a50b46cef21c8a86b15f0496cb794487a78a910b \
  --target-model ibm-granite/granite-4.0-1b-base \
  --target-revision 15148e99f8ca689325e2067ba282b23dec6670c7 \
  --output-dir experiments/results/public_sft/deployment_aligned/granite4_reproduction \
  --seed 20260828 --response-tokens 64 \
  --n-per-class 96 --n-aux 96 --audit-train-per-class 32 \
  --target-epochs 3 --draft-epochs 3 --distill-steps 120 \
  --draft-adaptation both --pair-diagnostic-batch-size 4 \
  --transcript-repeats 24 --bootstrap-repeats 500 \
  --detector-seeds 3 --cpu-threads 8
```

To scan nested verifier-feedback budgets without retraining the target, reuse a
v2 adapter as follows. The script defaults to eight CPU threads and evaluates
only the three preregistered budget endpoints.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.public_budget_sweep \
  --gpu 0 \
  --source-results experiments/results/public_sft/qwen3_1p7b_to_8b_epoch3_v2/results.json \
  --output-dir experiments/results/public_sft/qwen3_1p7b_to_8b_epoch3_v2/budget_sweep \
  --draft-endpoint auto \
  --repeats 1 4 24 \
  --bootstrap-repeats 500 \
  --detector-seeds 3 \
  --cpu-threads 8
```

The budget sweep uses the exact prefix of one Bernoulli transcript at 1, 4,
and 24 repetitions (26, 104, and 624 bits per record). It is a fixed-candidate
L3 semantic simulation, not a passive L2 production trace. For new aligned
results, `auto` selects the primary auxiliary-distilled adapter; old results
without a draft adapter fall back to the base draft. The sweep fails closed if
any selected target/draft log-probability is non-finite.

## NART-style post-cutoff track (full-parameter fine-tuning)

The NART benchmark track replaces the private PDF data with the three
post-cutoff pools of NART (Tan et al., NDSS 2026): **WikiTection** (English
Wikipedia pages first created in the window), **NewsTection** (CC-NEWS article
pages captured in the window), and **ArXivTection** (arXiv papers submitted in
the window, full HTML text). The default window is `2026-05-01 .. 2026-08-29`,
which postdates the Qwen3-8B 2025-07 cutoff and every repo target model
(including Qwen3.6, released 2026-04-15, assumed cutoff <= 2026-03 — see the
provenance note in each pool manifest).

Freeze the pools once (raw text is persisted so every target model can
re-tokenize; each pool carries a SHA-256 manifest):

```bash
uv run --no-sync python -m experiments.sd_membership_sft.nart_benchmarks wiki --records 6600
uv run --no-sync python -m experiments.sd_membership_sft.nart_benchmarks news --records 6600
uv run --no-sync python -m experiments.sd_membership_sft.nart_benchmarks arxiv --records 6600 --min-chars 4200
```

Each run then uses 2,000 member / 2,000 nonmember / 2,000 auxiliary documents
selected at load time under the target tokenizer's NART token band
(128..512 tokens for Wiki/News, 1024..2048 for ArXiv), the fixed NART Figure-3
prompt with the document as a loss-masked continuation, and NART Table IX
hyperparameters: lr `2e-5`, effective batch `16` (micro-batch 2 x accum 8;
ArXiv uses 1 x 16 at sequence length 2048), epochs `3` (NewsTection `4`),
bf16. `--trainer full` fine-tunes every parameter with a bitsandbytes paged
8-bit AdamW so the 8B target fits on one A100-80GB; `--trainer lora` keeps the
legacy adapter path. Transcript probing is capped at 26 selected positions
(26 x 24 = 624 bits per record) via `--selected-token-cap`, a no-op for the
legacy 64-token records.

Run the Qwen3-8B WikiTection condition with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.runner \
  --gpu 0 \
  --benchmark wikitection \
  --trainer full \
  --optimizer adamw8bit \
  --target-epochs 3 \
  --target-lr 2e-5 --draft-lr 2e-5 \
  --n-per-class 2000 --n-aux 2000 --audit-train-per-class 128 \
  --target-batch-size 2 --target-grad-accum 8 \
  --draft-batch-size 2 --draft-grad-accum 8 \
  --distill-steps 384 \
  --output-dir experiments/results/nart_sft/wikitection_qwen3_8b_epoch3
```

The main benchmark configuration is unified across the three pools:
**lr 2e-5, 3 epochs** (the earlier NewsTection-specific 4-epoch setting is
retained in git history as a training-length ablation; its generalization
gate marginally fails at ROUGE-1, see
`newstection_qwen3_8b_epoch4/GENERALIZATION.md`). A weak-memorization
variant uses `--target-epochs 1` (e.g.
`newstection_qwen3_8b_epoch1`). ArXivTection keeps its 2048-token sequence
with micro-batch 1 x accumulation 16:
`--benchmark arxivtection --target-batch-size 1 --target-grad-accum 16`
(similarly for the draft batch flags).

### Direct-verifier baseline comparison

Every NART-track run automatically evaluates three score-based MIA baselines
that attack the fine-tuned verifier directly, with no draft or protocol
signal, and compares them against the SD-scenario signals on the same audit
test split (paired bootstrap deltas included):

- **Min-K% Prob** (Shi et al., ICLR 2024; official code
  `swj0419/detect-pretrain-code`): mean log-probability of the k=20%
  least-likely tokens, reimplemented per the paper.
- **WBC** (Chen et al., USENIX Security 2026; official code `Stry233/WBC`):
  sliding-window sign test over per-token loss differences between the
  fine-tuned target and the pre-fine-tuning reference model, geometric
  ensemble with w_min=2, w_max=40, |W|=10 per the paper's ablation.
- **Reference loss-diff**: the global-average per-token loss difference the
  WBC paper compares against (reference-model-calibrated loss attack).

The SD signals are the 624-bit verifier acceptance transcript, acceptance
tomography, the transcript-only triplet detector, and DraVer-Act. For
DraVer-Act the auxiliary-distilled draft's all-layer token-aligned
activations are extracted during the run (NART track only) and the triplet
detector is trained on the shared audit calibration split. The comparison
table and paired deltas are written to `RESULTS.md`; run it exactly as in
the command above — no extra flags.

### Generalization check (NART Table X protocol)

After fine-tuning, run the NART generalization check on the saved run
directory. It replays the training split (same pool, tokenizer, sizes, seed),
has the fine-tuned target generate 128-token continuations for the first
256-token context of member and nonmember documents under greedy decoding,
and scores BLEU-4 / ROUGE-1 / ROUGE-L against the true continuations
(500 samples per class by default):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.generalization \
  --run-dir experiments/results/nart_sft/wikitection_qwen3_8b_epoch3 \
  --gpu 0 --samples 500 --batch-size 8
```

Two comparisons are reported in `GENERALIZATION.md` / `generalization.json`:
(1) the NART no-overfitting protocol — member vs nonmember quality gap on the
fine-tuned model, with bootstrap CIs and a soft gate at |gap| < 0.03 (NART
reports differences below 0.03 as stable generation quality); (2) base vs
fine-tuned quality on the same samples, quantifying how much generality the
full-parameter fine-tuning cost (reported with CIs and relative drop, no hard
gate). `--include-drafts` additionally scores the base, auxiliary-distilled,
and member-SFT draft variants, which matters for speculative-decoding
acceptance quality.
