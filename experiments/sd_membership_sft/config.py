from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    seed: int = 20260824
    data_seed: int = 20260824
    audit_seed: int = 20260824
    gpu: int = 0
    target_model: str = "Qwen/Qwen3-8B-Base"
    draft_model: str = "Qwen/Qwen3-1.7B-Base"
    response_tokens: int = 64
    n_per_class: int = 160
    n_aux: int = 160
    audit_train_per_class: int = 48
    target_epochs: int = 1
    target_batch_size: int = 2
    target_grad_accum: int = 4
    target_lr: float = 2e-4
    draft_batch_size: int = 2
    draft_grad_accum: int = 4
    draft_lr: float = 2e-4
    distill_steps: int = 80
    distill_temperature: float = 2.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    min_k_fraction: float = 0.20
    transcript_repeats: int = 24
    transcript_levels: int = 5
    bootstrap_repeats: int = 500
    run_auxiliary_draft: bool = True
    run_member_draft: bool = True
    save_adapters: bool = True
    output_dir: Path = Path("experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch1")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        return value
