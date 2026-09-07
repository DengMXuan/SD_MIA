from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedModel

from .data import SFTRecord, collate_sft, make_sft_example


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_causal_lm(
    model_id: str,
    device: torch.device,
    revision: str | None = None,
    local_files_only: bool = False,
) -> PreTrainedModel:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except ValueError as error:
        # wrapper architectures (e.g. qwen3_5 ConditionalGeneration) are not in
        # the causal-LM map; load the full multimodal wrapper for text-only use
        if "Unrecognized" not in str(error) and "does not appear to have" not in str(error):
            raise
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    model.to(device)
    model.config.use_cache = False
    return model


def add_lora(model: PreTrainedModel, r: int, alpha: int, dropout: float) -> PeftModel:
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type == "gpt_neox":
        target_modules = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    elif model_type == "opt":
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif model_type == "gpt2":
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif model_type == "granitemoehybrid":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "input_linear",
            "output_linear",
        ]
    elif model_type == "lfm2":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ]
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, config)
    peft_model.config.use_cache = False
    peft_model.print_trainable_parameters()
    return peft_model


def _enable_checkpointing(model: torch.nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _make_optimizer(
    model: torch.nn.Module, lr: float, optimizer_name: str = "adamw"
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw8bit":
        import bitsandbytes as bnb

        try:
            return bnb.optim.PagedAdamW8bit(parameters, lr=lr)
        except (TypeError, RuntimeError):
            return bnb.optim.AdamW8bit(parameters, lr=lr)
    try:
        return torch.optim.AdamW(parameters, lr=lr, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(parameters, lr=lr)


def _loader(
    records: list[SFTRecord], tokenizer: Any, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    examples = [make_sft_example(record, tokenizer) for record in records]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=lambda features: collate_sft(features, int(tokenizer.pad_token_id)),
    )


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def sft_train(
    model: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    label: str,
    optimizer_name: str = "adamw",
) -> list[float]:
    if epochs <= 0:
        return []
    _enable_checkpointing(model)
    optimizer = _make_optimizer(model, lr, optimizer_name)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        loader = _loader(records, tokenizer, batch_size, shuffle=True, seed=seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        pending = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(device):
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                ).loss
            epoch_losses.append(float(loss.detach().cpu()))
            (loss / grad_accum).backward()
            pending += 1
            if pending == grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        print(
            f"{label} epoch {epoch + 1}/{epochs}: loss={mean_loss:.5f}",
            flush=True,
        )
    del optimizer
    model.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def distill_on_auxiliary(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    temperature: float,
    seed: int,
    optimizer_name: str = "adamw",
) -> list[float]:
    if steps <= 0:
        return []
    _enable_checkpointing(draft)
    target.eval()
    draft.train()
    optimizer = _make_optimizer(draft, lr, optimizer_name)
    rng = np.random.default_rng(seed)
    examples = [make_sft_example(record, tokenizer) for record in records]
    losses: list[float] = []
    for step in range(steps):
        indices = rng.integers(0, len(examples), size=batch_size)
        batch = collate_sft([examples[int(index)] for index in indices], int(tokenizer.pad_token_id))
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad(), _autocast(device):
            teacher_logits = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, :-1].float()
        with _autocast(device):
            student_logits = draft(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, :-1].float()
            labels = batch["labels"][:, 1:]
            valid = labels.ne(-100)
            student_selected = student_logits[valid]
            teacher_selected = teacher_logits[valid]
            labels_selected = labels[valid]
            ce = F.cross_entropy(student_selected, labels_selected)
            kl = F.kl_div(
                F.log_softmax(student_selected / temperature, dim=-1),
                F.softmax(teacher_selected / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            loss = 0.20 * ce + 0.80 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 4) == 0:
            print(f"draft distill step {step + 1}/{steps}: loss={losses[-1]:.5f}", flush=True)
        del teacher_logits, student_logits
    del optimizer
    draft.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return losses


def save_adapter(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, PeftModel):
        raise TypeError("Expected a PEFT model when saving an adapter")
    model.save_pretrained(path)


def save_trained_model(model: torch.nn.Module, path: Path) -> None:
    """Save either a LoRA adapter or a fully fine-tuned checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
