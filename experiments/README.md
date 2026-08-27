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
│   ├── public_benchmark.py        # public FineWeb controlled-SFT benchmark
│   ├── public_budget_sweep.py     # nested verifier-feedback budget scan
│   ├── public_data.py             # manifest-checked public split loader
│   ├── public_fineweb.py          # bounded FineWeb snapshot constructor
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

The public validation follows the controlled fine-tuning structure of NART but
changes the measured object for edge–cloud speculative decoding. The on-device
draft remains an untouched white-box checkpoint; only member documents update
the cloud target. DraVer-Act then conditions token-aligned, all-layer draft
statistics on the verifier transcript and compares them with draft NART-style,
transcript-only, naive-concatenation, and q-bin-shuffled controls.

The checked FineWeb snapshot contains 600 filtered documents from
`CC-MAIN-2025-26`. Its SHA-256 is
`7855bc5485ea6892ecbdc13157162a636b34ccec5bf65435f551bf262a5f869a`.
The main v2 protocol uses 160 records per member/nonmember class, 160 auxiliary
records, 48 calibration examples per class, 112 test examples per class, 128
response tokens, 3 target LoRA epochs, and 24 verifier repetitions on 26 q-min
tokens. Model revisions, data provenance, and all uncertainty estimates are in
[`results/public_sft/SUMMARY.md`](results/public_sft/SUMMARY.md).

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

To scan nested verifier-feedback budgets without retraining the target, reuse a
v2 adapter as follows. The script defaults to eight CPU threads and evaluates
only the three preregistered budget endpoints.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m experiments.sd_membership_sft.public_budget_sweep \
  --gpu 0 \
  --source-results experiments/results/public_sft/qwen3_1p7b_to_8b_epoch3_v2/results.json \
  --output-dir experiments/results/public_sft/qwen3_1p7b_to_8b_epoch3_v2/budget_sweep \
  --repeats 1 4 24 \
  --bootstrap-repeats 500 \
  --detector-seeds 3 \
  --cpu-threads 8
```

The budget sweep uses the exact prefix of one Bernoulli transcript at 1, 4,
and 24 repetitions (26, 104, and 624 bits per record). It is a fixed-candidate
L3 semantic simulation, not a passive L2 production trace.
