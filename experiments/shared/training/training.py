from __future__ import annotations

import gc
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from experiments.shared.data.data import SFTRecord, collate_sft, make_sft_example


_BNB_8BIT_BLOCK_SIZE = 256
# bitsandbytes passes a tensor's element count to its CUDA optimizer kernel as
# a signed int32. Keep individual launches well below that limit and align all
# boundaries to its 256-element quantization blocks.
_BNB_8BIT_CHUNK_ELEMENTS = 1 << 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_causal_lm(
    model_id: str,
    device: torch.device,
    revision: str | None = None,
    local_files_only: bool = False,
    attn_implementation: str = "eager",
) -> PreTrainedModel:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
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
            attn_implementation=attn_implementation,
        )
    model.to(device)
    model.config.use_cache = False
    return model


def load_tokenizer(
    model_id: str,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load a tokenizer from a native Transformers model repository."""
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )


def _backward_chunked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    chunk_size: int = 32,
    loss_scale: float = 1.0,
) -> float:
    """Backpropagate the exact CE/KL objective without full-vocabulary FP32 copies.

    Gemma 4 has a 262k-token vocabulary. Materializing both models' complete
    logits and both softmax operands in FP32 can exhaust an 80 GiB worker for
    long ArXiv examples. Compute the loss gradient a few response tokens at a
    time, store that gradient in the logits dtype, then traverse the student
    model graph once.
    """
    valid_positions = labels.ne(-100).nonzero(as_tuple=False)
    valid_tokens = int(valid_positions.shape[0])
    if valid_tokens == 0:
        raise ValueError("distillation batch contains no response tokens")

    student_gradient = torch.zeros_like(student_logits)
    loss_value = 0.0
    for start in range(0, valid_tokens, chunk_size):
        positions = valid_positions[start : start + chunk_size]
        batch_indices = positions[:, 0]
        token_indices = positions[:, 1]
        student_chunk = (
            student_logits[batch_indices, token_indices]
            .detach()
            .float()
            .requires_grad_(True)
        )
        teacher_chunk = teacher_logits[batch_indices, token_indices].float()
        label_chunk = labels[batch_indices, token_indices]
        ce = F.cross_entropy(student_chunk, label_chunk, reduction="sum")
        kl = F.kl_div(
            F.log_softmax(student_chunk / temperature, dim=-1),
            F.softmax(teacher_chunk / temperature, dim=-1),
            reduction="sum",
        ) * (temperature**2)
        chunk_loss = (0.20 * ce + 0.80 * kl) / valid_tokens
        (chunk_gradient,) = torch.autograd.grad(chunk_loss, student_chunk)
        student_gradient[batch_indices, token_indices] = chunk_gradient.to(
            student_gradient.dtype
        )
        loss_value += float(chunk_loss.detach())

    student_logits.backward(student_gradient * loss_scale)
    return loss_value


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


def _update_bnb_8bit_parameter_in_chunks(
    optimizer: Any,
    group: dict[str, Any],
    parameter: torch.nn.Parameter,
    group_index: int,
    parameter_index: int,
    *,
    functional: Any | None = None,
    chunk_elements: int = _BNB_8BIT_CHUNK_ELEMENTS,
) -> None:
    """Apply one bitsandbytes 8-bit update without overflowing its int32 size.

    Gemma 4 E2B packs all Per-Layer Embeddings into one 2,348,810,240-element
    parameter. bitsandbytes 0.50 passes ``numel`` to CUDA as ``c_int32``, so a
    normal optimizer step overflows. Its optimizer state is blockwise, making
    aligned launches over views mathematically equivalent to one launch.
    """
    if functional is None:
        import bitsandbytes.functional as functional

    if chunk_elements <= 0 or chunk_elements % _BNB_8BIT_BLOCK_SIZE:
        raise ValueError(
            f"chunk_elements must be a positive multiple of {_BNB_8BIT_BLOCK_SIZE}"
        )
    if parameter.grad is None:
        return

    parameter.data = parameter.data.contiguous()
    parameter.grad = parameter.grad.contiguous()
    state = optimizer.state[parameter]
    if state["state1"].dtype != torch.uint8:
        raise RuntimeError(
            "chunked bitsandbytes updates require 8-bit optimizer state"
        )

    config = optimizer.get_config(group_index, parameter_index, group)
    state["step"] += 1
    step = state["step"]
    betas = config["betas"]

    flat_parameter = parameter.data.view(-1)
    flat_gradient = parameter.grad.view(-1)
    flat_state1 = state["state1"].view(-1)
    flat_state2 = (
        state["state2"].view(-1) if state.get("state2") is not None else None
    )
    if getattr(state["state1"], "is_paged", False):
        flat_state1.is_paged = True
    if flat_state2 is not None and getattr(state["state2"], "is_paged", False):
        flat_state2.is_paged = True
    total_elements = parameter.numel()

    def state_slice(tensor: torch.Tensor | None, start: int, end: int):
        if tensor is None:
            return None
        view = tensor[start:end]
        # Unified-memory tensors report as CPU and bitsandbytes exempts them
        # from its same-device check via this marker. Tensor views do not carry
        # arbitrary Python attributes, so restore the marker on every slice.
        if getattr(tensor, "is_paged", False):
            view.is_paged = True
        return view

    for start in range(0, total_elements, chunk_elements):
        end = min(start + chunk_elements, total_elements)
        block_start = start // _BNB_8BIT_BLOCK_SIZE
        block_end = (end + _BNB_8BIT_BLOCK_SIZE - 1) // _BNB_8BIT_BLOCK_SIZE
        functional.optimizer_update_8bit_blockwise(
            optimizer.optimizer_name,
            flat_gradient[start:end],
            flat_parameter[start:end],
            state_slice(flat_state1, start, end),
            state_slice(flat_state2, start, end),
            betas[0],
            betas[1],
            betas[2] if len(betas) >= 3 else 0.0,
            config.get("alpha", 0.0),
            config["eps"],
            step,
            config["lr"],
            state["qmap1"],
            state.get("qmap2"),
            state["absmax1"][block_start:block_end],
            (
                state["absmax2"][block_start:block_end]
                if state.get("absmax2") is not None
                else None
            ),
            config["weight_decay"],
            gnorm_scale=1.0,
            skip_zeros=config["skip_zeros"],
        )


class _LargeTensorSafeAdamW8bitMixin:
    """Split only tensors that exceed bitsandbytes' safe CUDA launch size."""

    @torch.no_grad()
    def update_step(self, group, parameter, group_index, parameter_index):
        if parameter.numel() <= _BNB_8BIT_CHUNK_ELEMENTS:
            return super().update_step(
                group, parameter, group_index, parameter_index
            )
        return _update_bnb_8bit_parameter_in_chunks(
            self,
            group,
            parameter,
            group_index,
            parameter_index,
        )


def _make_optimizer(
    model: torch.nn.Module, lr: float, optimizer_name: str = "adamw"
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw8bit":
        import bitsandbytes as bnb

        class LargeTensorSafePagedAdamW8bit(
            _LargeTensorSafeAdamW8bitMixin, bnb.optim.PagedAdamW8bit
        ):
            pass

        class LargeTensorSafeAdamW8bit(
            _LargeTensorSafeAdamW8bitMixin, bnb.optim.AdamW8bit
        ):
            pass

        try:
            return LargeTensorSafePagedAdamW8bit(parameters, lr=lr)
        except (TypeError, RuntimeError):
            return LargeTensorSafeAdamW8bit(parameters, lr=lr)
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
    # Reset immediately before optimization so model/tokenizer loading cannot
    # advance a condition's training RNG differently across architectures.
    set_seed(seed)
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
    grad_accum: int,
    lr: float,
    temperature: float,
    seed: int,
    optimizer_name: str = "adamw",
) -> list[float]:
    if steps <= 0:
        return []
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    set_seed(seed)
    _enable_checkpointing(draft)
    target.eval()
    draft.train()
    optimizer = _make_optimizer(draft, lr, optimizer_name)
    rng = np.random.default_rng(seed)
    examples = [make_sft_example(record, tokenizer) for record in records]
    losses: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _micro_step in range(grad_accum):
            indices = rng.integers(0, len(examples), size=batch_size)
            batch = collate_sft(
                [examples[int(index)] for index in indices],
                int(tokenizer.pad_token_id),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad(), _autocast(device):
                teacher_logits = target(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits[:, :-1]
            with _autocast(device):
                student_logits = draft(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits[:, :-1]
            micro_losses.append(
                _backward_chunked_distillation_loss(
                    student_logits,
                    teacher_logits,
                    batch["labels"][:, 1:],
                    temperature,
                    loss_scale=1.0 / grad_accum,
                )
            )
            del teacher_logits, student_logits
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        losses.append(float(np.mean(micro_losses)))
        if (step + 1) % max(1, steps // 4) == 0:
            print(
                f"draft distill step {step + 1}/{steps}: loss={losses[-1]:.5f}",
                flush=True,
            )
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
