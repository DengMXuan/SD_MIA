"""Token-aligned FP32 probability extraction without report or CLI dependencies."""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from experiments.shared.data.data import SFTRecord, collate_sft, make_sft_example

LOGSUMEXP_SEQUENCE_CHUNK = 64

def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sequence_chunk: int = LOGSUMEXP_SEQUENCE_CHUNK,
) -> torch.Tensor:
    """Compute selected-token log-probabilities with an FP32 normalizer.

    Qwen checkpoints are normally loaded in BF16.  Calling ``logsumexp`` on
    BF16 logits makes the vocabulary normalizer itself low precision, which is
    especially harmful for the p-q difference used by this experiment.  The
    full vocabulary is converted in sequence chunks so the FP32 operation does
    not require an additional full-length FP32 logits tensor.
    """
    if sequence_chunk <= 0:
        raise ValueError("sequence_chunk must be positive")
    selected = logits.gather(
        -1, labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1).float()
    normalizer = torch.empty_like(selected, dtype=torch.float32)
    for start in range(0, logits.shape[1], sequence_chunk):
        end = min(start + sequence_chunk, logits.shape[1])
        normalizer[:, start:end] = torch.logsumexp(
            logits[:, start:end, :].float(), dim=-1
        )
    return selected - normalizer



@torch.inference_mode()
def record_logprobabilities(
    model: Any,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    empty_cache_each_batch: bool = False,
    progress: Any = None,
) -> list[np.ndarray]:
    """Per-record log-probability arrays over response tokens (+ EOS)."""
    model.eval()
    examples = [make_sft_example(record, tokenizer) for record in records]
    order = sorted(
        range(len(examples)), key=lambda index: len(examples[index]["input_ids"])
    )
    outputs: list[np.ndarray | None] = [None] * len(examples)
    starts = list(range(0, len(order), batch_size))
    if progress is not None:
        starts = progress.track(starts, "token probability scoring", unit="batches")
    for start in starts:
        indices = order[start : start + batch_size]
        rows = [examples[index] for index in indices]
        batch = {
            key: value.to(device)
            for key, value in collate_sft(rows, int(tokenizer.pad_token_id)).items()
        }
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        # Keep the vocabulary normalizer in FP32.  It is computed in sequence
        # chunks to avoid materializing a full FP32 vocabulary tensor.
        logp = selected_token_logprobs(logits, labels)
        for row, index in enumerate(indices):
            outputs[index] = logp[row][valid[row]].to(torch.float32).cpu().numpy()
        del logits, logp, labels, valid, batch
        if empty_cache_each_batch and device.type == "cuda":
            torch.cuda.empty_cache()
    if any(output is None for output in outputs):
        raise RuntimeError("Forward pass left records unscored")
    return [output for output in outputs if output is not None]
