# Experiments

The experiment code is separated from generated artifacts:

```text
experiments/
├── sd_membership_pilot.py       # legacy GPT-2 pilot, kept for provenance
├── sd_membership_sft/           # Qwen3 instruction-SFT implementation
│   ├── audit.py                  # membership scores and transcript simulation
│   ├── config.py                 # reproducible experiment configuration
│   ├── data.py                   # source extraction and controlled SFT split
│   ├── p0_matrix.py              # seed/epoch stability and large-audit matrix
│   ├── p1_passive_l2.py          # natural draft + min(1,p/q) passive L2 protocol
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
