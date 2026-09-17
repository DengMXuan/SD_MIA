from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    seed: int = 20260824
    data_seed: int = 20260824
    gpu: int = 0
    target_model: str = "Qwen/Qwen3-8B-Base"
    draft_model: str = "Qwen/Qwen3-1.7B-Base"
    target_revision: str | None = None
    draft_revision: str | None = None
    # Benchmark pool to run; each loads a frozen post-cutoff pool
    # (see splits.BENCHMARK_TOKEN_BANDS)..
    benchmark: str = "newstection"
    # Optional explicit pool override; defaults to
    # experiments/data/pools/<benchmark>/pool.jsonl.
    pool_path: Path | None = None
    # "lora" fine-tunes adapters; "full" fine-tunes every parameter with mainline
    # hyperparameters (lr 2e-5, effective batch 16, 3 epochs, bf16).
    trainer: str = "lora"
    # "adamw" is the standard fp32-state optimizer; "adamw8bit" swaps in
    # bitsandbytes PagedAdamW8bit so an 8B target fits on one A100-80GB.
    optimizer: str = "adamw"
    n_per_class: int = 2000
    n_aux: int = 2000
    target_epochs: int = 1
    target_batch_size: int = 2
    target_grad_accum: int = 8
    target_lr: float = 2e-5
    draft_batch_size: int = 2
    draft_grad_accum: int = 8
    draft_lr: float = 2e-5
    distill_steps: int = 384
    distill_temperature: float = 2.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    run_auxiliary_draft: bool = True
    run_member_draft: bool = True
    save_adapters: bool = True
    output_dir: Path = Path(
        "experiments/results/sft_runs/newstection_qwen3_8b_epoch1"
    )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        if self.pool_path is not None:
            value["pool_path"] = str(self.pool_path)
        return value
