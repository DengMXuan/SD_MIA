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


def load_causal_lm(model_id: str, device: torch.device) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.config.use_cache = False
    return model


def add_lora(model: PreTrainedModel, r: int, alpha: int, dropout: float) -> PeftModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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


def _make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
) -> list[float]:
    if epochs <= 0:
        return []
    _enable_checkpointing(model)
    optimizer = _make_optimizer(model, lr)
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


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    loader = _loader(records, tokenizer, batch_size, shuffle=False, seed=0)
    token_logp_batches: list[np.ndarray] = []
    entropy_batches: list[np.ndarray] = []
    top1_batches: list[np.ndarray] = []
    grad_batches: list[np.ndarray] = []
    hidden_batches: list[np.ndarray] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        safe_labels = labels.clamp_min(0)
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        ent = -(probs * log_probs).sum(dim=-1)
        match = logits.argmax(dim=-1).eq(safe_labels)
        hidden = output.hidden_states[-1][:, :-1].float()
        prob_sq = probs.square().sum(dim=-1)
        py = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        simplex_grad_norm = (prob_sq - 2.0 * py + 1.0).clamp_min(0).sqrt()
        grad_proxy = hidden.norm(dim=-1) * simplex_grad_norm

        max_response = int(valid.sum(dim=1).max().item())
        token_logp = np.full((len(batch["input_ids"]), max_response), np.nan, dtype=np.float32)
        entropy = np.full_like(token_logp, np.nan)
        top1 = np.full_like(token_logp, np.nan)
        grad = np.full_like(token_logp, np.nan)
        hidden_mean = np.zeros((len(batch["input_ids"]), hidden.shape[-1]), dtype=np.float32)
        for row in range(len(batch["input_ids"])):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            count = len(positions)
            token_logp[row, :count] = lp[row, positions].cpu().numpy()
            entropy[row, :count] = ent[row, positions].cpu().numpy()
            top1[row, :count] = match[row, positions].float().cpu().numpy()
            grad[row, :count] = grad_proxy[row, positions].cpu().numpy()
            hidden_mean[row] = hidden[row, positions].mean(dim=0).cpu().numpy()

        token_logp_batches.append(token_logp)
        entropy_batches.append(entropy)
        top1_batches.append(top1)
        grad_batches.append(grad)
        hidden_batches.append(hidden_mean)
        del output, logits, log_probs, probs, hidden

    width = max(batch.shape[1] for batch in token_logp_batches)

    def pad_batches(batches: list[np.ndarray], fill: float) -> np.ndarray:
        result = np.full((sum(batch.shape[0] for batch in batches), width), fill, dtype=np.float32)
        start = 0
        for batch in batches:
            result[start : start + len(batch), : batch.shape[1]] = batch
            start += len(batch)
        return result

    return {
        "token_logp": pad_batches(token_logp_batches, np.nan),
        "entropy": pad_batches(entropy_batches, np.nan),
        "top1_match": pad_batches(top1_batches, np.nan),
        "grad_proxy": pad_batches(grad_batches, np.nan),
        "hidden_mean": np.concatenate(hidden_batches, axis=0),
    }


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
) -> list[float]:
    if steps <= 0:
        return []
    _enable_checkpointing(draft)
    target.eval()
    draft.train()
    optimizer = _make_optimizer(draft, lr)
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
