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
from peft import LoraConfig, get_peft_model
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
        description="Native-MTP hidden-connector membership-safety validation."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--converted-speculator", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--n-per-class", type=int, default=256)
    parser.add_argument("--n-aux", type=int, default=256)
    parser.add_argument("--audit-train-per-class", type=int, default=64)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--target-batch-size", type=int, default=1)
    parser.add_argument("--target-grad-accum", type=int, default=8)
    parser.add_argument("--target-lr", type=float, default=2e-4)
    parser.add_argument("--mtp-adapt-steps", type=int, default=256)
    parser.add_argument("--mtp-lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-speculative-steps", type=int, default=3)
    parser.add_argument("--probe-tokens", type=int, default=24)
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
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


def _base_speculator(model: torch.nn.Module) -> torch.nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _ensure_converted_speculator(
    target_snapshot: str,
    converted: Path,
    num_speculative_steps: int,
) -> None:
    if (converted / "config.json").is_file() and any(
        converted.glob("*.safetensors")
    ):
        return
    if converted.exists():
        raise RuntimeError(
            f"Converted MTP path exists but is incomplete: {converted}"
        )
    from speculators.convert.mtp import MTPConverter

    MTPConverter().convert(
        input_path=target_snapshot,
        output_path=converted,
        base_model=target_snapshot,
        num_speculative_steps=num_speculative_steps,
        validate=True,
    )


def _load_mtp(converted: Path, device: torch.device) -> torch.nn.Module:
    from speculators import SpeculatorModel

    model = SpeculatorModel.from_pretrained(
        str(converted), local_files_only=True, torch_dtype=torch.bfloat16
    )
    return model.to(device)


def _add_mtp_lora(
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
            "input_proj",
        ],
    )
    return get_peft_model(model, config)


def adapt_mtp_on_auxiliary(
    speculator: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    steps: int,
    lr: float,
    seed: int,
) -> list[float]:
    """Adapt MTP on auxiliary records using the adapted verifier connector.

    The native MTP objective is retained.  Unlike target-only SFT, every MTP
    update receives hidden states from the already adapted verifier, but no
    member record is used for the MTP update.
    """
    target.eval()
    speculator.train()
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
            )
            connector = target_output.hidden_states[-1]
        with _autocast(device):
            _, loss, _ = speculator(
                input_ids=batch["input_ids"],
                hidden_states=connector,
                # Speculators 0.7.0.1 passes an unsliced full-length mask to a
                # shorter MTP query (seq_len - steps - 1).  Batch size is one
                # and these examples have no padding, so omitting the mask is
                # equivalent while avoiding the incompatible Q/KV lengths.
                attention_mask=None,
                loss_mask=batch["labels"].ne(-100),
                return_dict=True,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(speculator.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 4) == 0:
            print(
                f"MTP auxiliary adaptation step {step + 1}/{steps}: "
                f"loss={history[-1]:.5f}",
                flush=True,
            )
        del target_output, connector, loss
    speculator.eval()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return history


def _valid_anchor_positions(
    labels: torch.Tensor, valid_len: int, steps: int
) -> torch.Tensor:
    candidates = torch.arange(valid_len, device=labels.device)
    keep = torch.ones(valid_len, dtype=torch.bool, device=labels.device)
    for depth in range(steps):
        keep &= labels[depth + 2 : depth + 2 + valid_len].ne(-100)
    return candidates[keep]


@torch.no_grad()
def extract_observations(
    speculator: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    probe_tokens: int,
    steps: int,
) -> tuple[ProtocolObservations, dict[str, Any]]:
    target.eval()
    speculator.eval()
    base = _base_speculator(speculator)
    blocks_per_record = probe_tokens // steps
    local_rows: list[np.ndarray] = []
    connector_rows: list[np.ndarray] = []
    acceptance_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []

    for batch in _loader(records, tokenizer, batch_size=1):
        batch = {name: value.to(device) for name, value in batch.items()}
        with _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        captured: list[torch.Tensor] = []
        handle = base.mtp_layers[0].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        with _autocast(device):
            logits_list, _, _ = speculator(
                input_ids=batch["input_ids"],
                hidden_states=target_output.hidden_states[-1],
                attention_mask=None,
                loss_mask=batch["labels"].ne(-100),
                return_dict=True,
            )
        handle.remove()
        if len(logits_list) != steps or len(captured) != steps:
            raise RuntimeError(
                f"MTP returned {len(logits_list)} logits and {len(captured)} states; "
                f"expected {steps}"
            )
        valid_len = int(logits_list[0].shape[1])
        anchors = _valid_anchor_positions(batch["labels"][0], valid_len, steps)
        if len(anchors) < blocks_per_record:
            raise RuntimeError(
                f"only {len(anchors)} complete MTP blocks, need {blocks_per_record}"
            )
        q_logp = []
        q_top = []
        p_top = []
        local = []
        connector = _state_summary(target_output.hidden_states[-1])
        for depth, (logits, state) in enumerate(zip(logits_list, captured, strict=True)):
            candidate_logits = logits[0, anchors].float()
            q_top.append(candidate_logits.argmax(dim=-1))
            q_logp.append(
                candidate_logits.amax(dim=-1) - candidate_logits.logsumexp(dim=-1)
            )
            target_positions = anchors + depth + 2
            teacher_logits = target_output.logits[0, target_positions - 1].float()
            p_top.append(teacher_logits.argmax(dim=-1))
            local.append(_state_summary(state)[0, anchors])
        block_q = torch.stack(q_logp, dim=1)
        selected = torch.argsort(block_q.mean(dim=1))[:blocks_per_record]
        block_acceptance = torch.stack(
            [draft.eq(teacher) for draft, teacher in zip(q_top, p_top, strict=True)],
            dim=1,
        )[selected]
        block_local = torch.stack(local, dim=1)[selected]
        block_connector = torch.stack(
            [connector[0, anchors] for _ in range(steps)], dim=1
        )[selected]
        local_rows.append(block_local.reshape(probe_tokens, -1).cpu().numpy())
        connector_rows.append(
            block_connector.reshape(probe_tokens, -1).cpu().numpy()
        )
        acceptance_rows.append(block_acceptance.reshape(-1).float().cpu().numpy())
        q_rows.append(block_q[selected].reshape(-1).cpu().numpy())
        del target_output, logits_list, captured

    acceptance = np.asarray(acceptance_rows, dtype=np.float32)
    visible = first_rejection_visibility(acceptance >= 0.5, steps)
    observations = ProtocolObservations(
        local_state=np.asarray(local_rows, dtype=np.float32),
        connector_state=np.asarray(connector_rows, dtype=np.float32),
        acceptance=acceptance,
        q_logp=np.asarray(q_rows, dtype=np.float32),
        confidence=None,
        visible=visible,
        depth=np.tile(np.arange(steps), blocks_per_record),
    )
    accepted = acceptance.reshape(len(acceptance), -1, steps)
    diagnostics = {
        "verification_semantics": "teacher_forced_greedy_block_match",
        "mean_position_match": acceptance.reshape(-1, steps).mean(axis=0).tolist(),
        "mean_match": float(acceptance.mean()),
        "visible_fraction": float(visible.mean()),
        "mean_accepted_prefix": float(
            np.mean(
                [
                    np.flatnonzero(block < 0.5)[0]
                    if np.any(block < 0.5)
                    else steps
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
    converted: Path,
    tokenizer: Any,
    device: torch.device,
    steps: int,
) -> None:
    target = _load_target(target_snapshot, device)
    speculator = _load_mtp(converted, device)
    encoded = tokenizer("MTP compatibility check.", return_tensors="pt")
    batch = {name: value.to(device) for name, value in encoded.items()}
    with _autocast(device):
        target_output = target(
            **batch, output_hidden_states=True, use_cache=False
        )
        logits_list, loss, _ = speculator(
            input_ids=batch["input_ids"],
            hidden_states=target_output.hidden_states[-1],
            attention_mask=None,
        )
    print(
        json.dumps(
            {
                "gate": "PASS",
                "target_architecture": type(target).__name__,
                "connector_shape": list(target_output.hidden_states[-1].shape),
                "mtp_steps": len(logits_list),
                "mtp_logits_shapes": [list(values.shape) for values in logits_list],
                "loss_finite": bool(torch.isfinite(loss).item()),
                "expected_steps": steps,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.probe_tokens % args.num_speculative_steps:
        raise ValueError("probe-tokens must be divisible by num-speculative-steps")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    root = Path(__file__).resolve().parents[2]
    dataset = _resolve(root, args.dataset)
    output_dir = _resolve(root, args.output_dir)
    converted = _resolve(root, args.converted_speculator)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)

    target_snapshot = snapshot_download(
        repo_id=args.target_model,
        revision=args.target_revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(target_snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _ensure_converted_speculator(
        target_snapshot, converted, args.num_speculative_steps
    )
    if args.compatibility_only:
        compatibility_check(
            target_snapshot,
            converted,
            tokenizer,
            device,
            args.num_speculative_steps,
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
        "MTP verifier member SFT",
    )
    save_adapter(target, output_dir / "adapter_target")
    speculator = _add_mtp_lora(
        _load_mtp(converted, device),
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    mtp_loss = adapt_mtp_on_auxiliary(
        speculator,
        target,
        auxiliary,
        tokenizer,
        device,
        args.mtp_adapt_steps,
        args.mtp_lr,
        args.seed + 20,
    )
    speculator.save_pretrained(output_dir / "adapter_mtp_aux_adapted")
    observations, diagnostics = extract_observations(
        speculator,
        target,
        candidates,
        tokenizer,
        device,
        args.probe_tokens,
        args.num_speculative_steps,
    )
    audit = evaluate_protocol_audit(
        observations,
        labels,
        calibration,
        test,
        args.num_speculative_steps,
        args.bootstrap_repeats,
        args.detector_seeds,
        args.seed + 100,
    )
    artifact = {
        "material_passport": {
            "experiment_id": f"mtp-qwen3.5-4b-{args.seed}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_HIDDEN_CONNECTOR_SFT",
        },
        "config": {
            **vars(args),
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "converted_speculator": str(args.converted_speculator),
        },
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": {
            "target_sft_loss": target_loss,
            "mtp_auxiliary_adaptation_loss": mtp_loss,
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
