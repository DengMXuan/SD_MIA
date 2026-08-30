from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .data import SFTRecord, collate_sft, make_sft_example, records_metadata
from .draver_activation import make_audit_split
from .protocol_features import (
    ProtocolObservations,
    evaluate_protocol_audit,
    first_rejection_visibility,
)
from .public_data import build_public_snapshot_split
from .training import save_adapter, set_seed, sft_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSpark hidden-connector membership-safety validation."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--target-revision", required=True)
    parser.add_argument(
        "--speculator-model",
        default="RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
    )
    parser.add_argument("--speculator-revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-adapter", type=Path)
    parser.add_argument("--speculator-adapter", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--n-per-class", type=int, default=128)
    parser.add_argument("--n-aux", type=int, default=128)
    parser.add_argument("--audit-train-per-class", type=int, default=32)
    parser.add_argument("--target-epochs", type=int, default=2)
    parser.add_argument("--target-batch-size", type=int, default=1)
    parser.add_argument("--target-grad-accum", type=int, default=8)
    parser.add_argument("--target-lr", type=float, default=2e-4)
    parser.add_argument("--speculator-adapt-steps", type=int, default=128)
    parser.add_argument("--speculator-lr", type=float, default=2e-4)
    parser.add_argument("--adapt-anchors", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--probe-tokens", type=int, default=24)
    parser.add_argument("--candidate-blocks", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--detector-seeds", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--compatibility-only", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _loader(
    records: list[SFTRecord], tokenizer: Any, batch_size: int, seed: int = 0
) -> DataLoader:
    examples = [make_sft_example(record, tokenizer) for record in records]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=seed != 0,
        generator=generator,
        num_workers=0,
        collate_fn=lambda rows: collate_sft(rows, int(tokenizer.pad_token_id)),
    )


def _state_summary(state: torch.Tensor) -> torch.Tensor:
    values = state.float()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.25, 0.5, 0.75], device=values.device),
        dim=-1,
    )
    return torch.cat(
        [
            values.mean(dim=-1, keepdim=True),
            values.std(dim=-1, unbiased=False, keepdim=True),
            values.amin(dim=-1, keepdim=True),
            quantiles.permute(1, 2, 0),
            values.amax(dim=-1, keepdim=True),
            values.square().mean(dim=-1, keepdim=True).sqrt(),
        ],
        dim=-1,
    )


def _load_target(snapshot: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model


def _add_target_lora(
    model: torch.nn.Module, r: int, alpha: int, dropout: float
) -> torch.nn.Module:
    """Use attention-only LoRA so the 256-expert MoE stays memory bounded."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
        ],
    )
    return get_peft_model(model, config)


def _base_speculator(model: torch.nn.Module) -> torch.nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _draft_to_target_ids(
    draft_ids: torch.Tensor, d2t_offsets: torch.Tensor | None
) -> torch.Tensor:
    if d2t_offsets is None:
        return draft_ids
    return draft_ids + d2t_offsets[draft_ids]


def _load_speculator(
    snapshot: str, target_snapshot: str, device: torch.device
) -> torch.nn.Module:
    from speculators import SpeculatorModel, SpeculatorModelConfig

    config = SpeculatorModelConfig.from_pretrained(snapshot, local_files_only=True)
    config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
    config.speculators_config.verifier.name_or_path = target_snapshot
    model = SpeculatorModel.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    return model.to(device)


def _add_speculator_lora(
    model: torch.nn.Module, r: int, alpha: int, dropout: float
) -> torch.nn.Module:
    config = LoraConfig(
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
            "fc",
            "markov_w2",
            "proj",
        ],
    )
    return get_peft_model(model, config)


def _connector(
    target_output: Any, layer_ids: list[int]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    layers = [target_output.hidden_states[index] for index in layer_ids]
    return torch.cat(layers, dim=-1), layers


def adapt_speculator_on_auxiliary(
    speculator: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    steps: int,
    anchors: int,
    lr: float,
    seed: int,
) -> list[float]:
    target.eval()
    speculator.train()
    base = _base_speculator(speculator)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in speculator.parameters() if parameter.requires_grad],
        lr=lr,
    )
    examples = [make_sft_example(record, tokenizer) for record in records]
    rng = np.random.default_rng(seed)
    history: list[float] = []
    for step in range(steps):
        example = examples[int(rng.integers(0, len(examples)))]
        batch = collate_sft([example], int(tokenizer.pad_token_id))
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.no_grad(), _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
            connector, _ = _connector(target_output, list(base.target_layer_ids))
        with _autocast(device):
            _, loss, _ = speculator(
                hidden_states=connector,
                input_ids=batch["input_ids"],
                loss_mask=batch["labels"].ne(-100),
                verifier_last_hidden_states=target_output.hidden_states[-1],
                document_ids=torch.zeros_like(batch["input_ids"]),
                max_anchors=anchors,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(speculator.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 4) == 0:
            print(
                f"DSpark auxiliary adaptation step {step + 1}/{steps}: "
                f"loss={history[-1]:.5f}",
                flush=True,
            )
        del target_output, connector, loss
    speculator.eval()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return history


def _dspark_heads(
    base: torch.nn.Module,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    anchored_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = len(anchored_indices) // base.block_size
    block_tokens = input_ids[0, anchored_indices].view(blocks, base.block_size)
    if base.config.sample_from_anchor:
        previous = block_tokens
    else:
        previous = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    hidden_blocks = hidden.view(blocks, base.block_size, -1)
    previous_embedding = None
    if base.markov_head is not None:
        previous_embedding = base.markov_head.prev_embeddings(previous)
        bias = base.markov_head.block_bias(
            prev_token_ids=previous,
            hidden_states=hidden_blocks,
            prev_emb=previous_embedding,
        )
        logits = logits.view(blocks, base.block_size, -1) + bias
    else:
        logits = logits.view(blocks, base.block_size, -1)
    if base.confidence_head is None:
        confidence = logits.new_zeros((blocks, base.block_size))
    else:
        if base.config.confidence_head_with_markov:
            if previous_embedding is None:
                raise RuntimeError("DSpark confidence head requires Markov embedding")
            features = torch.cat(
                [hidden_blocks, previous_embedding.to(hidden_blocks.dtype)], dim=-1
            )
        else:
            features = hidden_blocks
        confidence = base.confidence_head(features).sigmoid()
    return logits, confidence


@torch.no_grad()
def extract_observations(
    speculator: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    probe_tokens: int,
    candidate_blocks: int,
) -> tuple[ProtocolObservations, dict[str, Any]]:
    target.eval()
    speculator.eval()
    base = _base_speculator(speculator)
    block_size = int(base.block_size)
    selected_blocks = probe_tokens // block_size
    if candidate_blocks < selected_blocks:
        raise ValueError("candidate-blocks must cover all selected probe blocks")

    local_rows: list[np.ndarray] = []
    connector_rows: list[np.ndarray] = []
    acceptance_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    for batch in _loader(records, tokenizer, batch_size=1):
        batch = {name: value.to(device) for name, value in batch.items()}
        with _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            connector, connector_layers = _connector(
                target_output, list(base.target_layer_ids)
            )
            hidden, logits, _, _, anchored_indices = base._backbone_forward(
                hidden_states=connector,
                input_ids=batch["input_ids"],
                loss_mask=batch["labels"].ne(-100),
                verifier_last_hidden_states=target_output.hidden_states[-1],
                document_ids=torch.zeros_like(batch["input_ids"]),
                max_anchors=candidate_blocks,
            )
            logits, confidence = _dspark_heads(
                base, hidden, logits, batch["input_ids"], anchored_indices
            )
        q_top = logits.float().argmax(dim=-1)
        # Speculators stores an offset mapping, not the target token ID:
        # target_token_id = draft_index + d2t[draft_index].
        mapped_top = _draft_to_target_ids(q_top, base.d2t)
        teacher_top = target_output.logits[0, anchored_indices].float().argmax(dim=-1)
        teacher_top = teacher_top.view(candidate_blocks, block_size)
        acceptance = mapped_top.eq(teacher_top)
        q_logp = logits.float().amax(dim=-1) - logits.float().logsumexp(dim=-1)
        choose = torch.argsort(confidence.mean(dim=1))[:selected_blocks]
        local = _state_summary(hidden).view(candidate_blocks, block_size, -1)[choose]
        connector_feature = torch.cat(
            [_state_summary(layer)[0, anchored_indices] for layer in connector_layers],
            dim=-1,
        ).view(candidate_blocks, block_size, -1)[choose]
        local_rows.append(local.reshape(probe_tokens, -1).cpu().numpy())
        connector_rows.append(
            connector_feature.reshape(probe_tokens, -1).cpu().numpy()
        )
        acceptance_rows.append(acceptance[choose].reshape(-1).float().cpu().numpy())
        q_rows.append(q_logp[choose].reshape(-1).cpu().numpy())
        confidence_rows.append(confidence[choose].reshape(-1).float().cpu().numpy())
        del target_output, connector, hidden, logits

    acceptance = np.asarray(acceptance_rows, dtype=np.float32)
    visible = first_rejection_visibility(acceptance >= 0.5, block_size)
    observations = ProtocolObservations(
        local_state=np.asarray(local_rows, dtype=np.float32),
        connector_state=np.asarray(connector_rows, dtype=np.float32),
        acceptance=acceptance,
        q_logp=np.asarray(q_rows, dtype=np.float32),
        confidence=np.asarray(confidence_rows, dtype=np.float32),
        visible=visible,
        depth=np.tile(np.arange(block_size), selected_blocks),
    )
    accepted = acceptance.reshape(len(acceptance), -1, block_size)
    diagnostics = {
        "verification_semantics": "teacher_forced_greedy_block_match",
        "target_layer_ids": list(base.target_layer_ids),
        "draft_vocab_size": int(base.draft_vocab_size),
        "verifier_vocab_size": int(base.verifier_vocab_size),
        "mean_position_match": acceptance.reshape(-1, block_size).mean(axis=0).tolist(),
        "mean_match": float(acceptance.mean()),
        "mean_confidence": float(observations.confidence.mean()),
        "visible_fraction": float(visible.mean()),
        "mean_accepted_prefix": float(
            np.mean(
                [
                    np.flatnonzero(block < 0.5)[0]
                    if np.any(block < 0.5)
                    else block_size
                    for row in accepted
                    for block in row
                ]
            )
        ),
    }
    return observations, diagnostics


@torch.no_grad()
def compatibility_check(
    target_snapshot: str,
    speculator_snapshot: str,
    tokenizer: Any,
    device: torch.device,
    anchors: int,
) -> None:
    target = _load_target(target_snapshot, device)
    speculator = _load_speculator(speculator_snapshot, target_snapshot, device)
    base = _base_speculator(speculator)
    encoded = tokenizer(
        "DSpark compatibility check with enough tokens for an anchored block.",
        return_tensors="pt",
    )
    batch = {name: value.to(device) for name, value in encoded.items()}
    loss_mask = torch.ones_like(batch["input_ids"], dtype=torch.bool)
    with _autocast(device):
        target_output = target(
            **batch, output_hidden_states=True, use_cache=False, logits_to_keep=1
        )
        connector, _ = _connector(target_output, list(base.target_layer_ids))
        _, loss, metrics = speculator(
            hidden_states=connector,
            input_ids=batch["input_ids"],
            loss_mask=loss_mask,
            verifier_last_hidden_states=target_output.hidden_states[-1],
            document_ids=torch.zeros_like(batch["input_ids"]),
            max_anchors=anchors,
        )
    print(
        json.dumps(
            {
                "gate": "PASS",
                "target_architecture": type(target).__name__,
                "target_layer_ids": list(base.target_layer_ids),
                "connector_shape": list(connector.shape),
                "block_size": int(base.block_size),
                "draft_vocab_size": int(base.draft_vocab_size),
                "verifier_vocab_size": int(base.verifier_vocab_size),
                "markov_head": base.markov_head is not None,
                "confidence_head": base.confidence_head is not None,
                "loss_finite": bool(torch.isfinite(loss).item()),
                "metric_keys": sorted(metrics),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    dataset = _resolve(root, args.dataset)
    output_dir = _resolve(root, args.output_dir)
    target_adapter = (
        _resolve(root, args.target_adapter) if args.target_adapter is not None else None
    )
    speculator_adapter = (
        _resolve(root, args.speculator_adapter)
        if args.speculator_adapter is not None
        else None
    )
    if (target_adapter is None) != (speculator_adapter is None):
        raise ValueError(
            "target-adapter and speculator-adapter must be supplied together"
        )
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)

    target_snapshot = snapshot_download(
        repo_id=args.target_model,
        revision=args.target_revision,
        local_files_only=True,
    )
    speculator_snapshot = snapshot_download(
        repo_id=args.speculator_model,
        revision=args.speculator_revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(target_snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if args.compatibility_only:
        compatibility_check(
            target_snapshot,
            speculator_snapshot,
            tokenizer,
            device,
            args.adapt_anchors,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    members, nonmembers, auxiliary, data_metadata = build_public_snapshot_split(
        dataset,
        tokenizer,
        args.response_tokens,
        args.n_per_class,
        args.n_aux,
        args.split_seed,
    )
    candidates = members + nonmembers
    labels = np.concatenate(
        [
            np.ones(len(members), dtype=np.int64),
            np.zeros(len(nonmembers), dtype=np.int64),
        ]
    )
    calibration, test = make_audit_split(
        len(members),
        len(nonmembers),
        args.audit_train_per_class,
        args.split_seed + 30,
    )
    started = time.time()

    if target_adapter is None:
        target = _add_target_lora(
            _load_target(target_snapshot, device),
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        target_loss = sft_train(
            target,
            members,
            tokenizer,
            device,
            args.target_epochs,
            args.target_batch_size,
            args.target_grad_accum,
            args.target_lr,
            args.seed,
            "DSpark verifier member SFT",
        )
        save_adapter(target, output_dir / "adapter_target")
        speculator = _add_speculator_lora(
            _load_speculator(speculator_snapshot, target_snapshot, device),
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        speculator_loss = adapt_speculator_on_auxiliary(
            speculator,
            target,
            auxiliary,
            tokenizer,
            device,
            args.speculator_adapt_steps,
            args.adapt_anchors,
            args.speculator_lr,
            args.seed + 20,
        )
        speculator.save_pretrained(output_dir / "adapter_speculator_aux_adapted")
        reused_adapters = False
    else:
        target = PeftModel.from_pretrained(
            _load_target(target_snapshot, device), str(target_adapter)
        ).to(device)
        speculator = PeftModel.from_pretrained(
            _load_speculator(speculator_snapshot, target_snapshot, device),
            str(speculator_adapter),
        ).to(device)
        target_loss = []
        speculator_loss = []
        reused_adapters = True
    observations, diagnostics = extract_observations(
        speculator,
        target,
        candidates,
        tokenizer,
        device,
        args.probe_tokens,
        args.candidate_blocks,
    )
    block_size = int(_base_speculator(speculator).block_size)
    if args.probe_tokens % block_size:
        raise ValueError("probe-tokens must be divisible by DSpark block size")
    audit = evaluate_protocol_audit(
        observations,
        labels,
        calibration,
        test,
        block_size,
        args.bootstrap_repeats,
        args.detector_seeds,
        args.seed + 100,
    )
    artifact = {
        "material_passport": {
            "experiment_id": f"dspark-qwen3.6-35b-a3b-{args.seed}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_HIDDEN_CONNECTOR_SFT",
        },
        "config": {
            **vars(args),
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
        },
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": {
            "target_sft_loss": target_loss,
            "speculator_auxiliary_adaptation_loss": speculator_loss,
            "reused_adapters_for_corrected_measurement": reused_adapters,
            "source_target_adapter": str(target_adapter) if target_adapter else None,
            "source_speculator_adapter": (
                str(speculator_adapter) if speculator_adapter else None
            ),
        },
        "protocol_diagnostics": diagnostics,
        "audit": audit,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runtime": artifact["runtime"],
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
