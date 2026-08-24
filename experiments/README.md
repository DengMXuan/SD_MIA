# Experiments

The experiment code is separated from generated artifacts:

```text
experiments/
├── sd_membership_pilot.py       # legacy GPT-2 pilot, kept for provenance
├── sd_membership_sft/           # Qwen3 instruction-SFT implementation
│   ├── audit.py                  # membership scores and transcript simulation
│   ├── config.py                 # reproducible experiment configuration
│   ├── data.py                   # source extraction and controlled SFT split
│   ├── training.py               # LoRA SFT, distillation, and feature extraction
│   └── runner.py                 # command-line experiment entry point
├── configs/                      # checked-in experiment configurations
└── results/
    ├── pilot/                    # legacy GPT-2 high-memory result
    ├── pilot_epoch1/             # legacy GPT-2 weak-memory result
    └── qwen3_sft/                # Qwen3 SFT outputs and adapters
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
