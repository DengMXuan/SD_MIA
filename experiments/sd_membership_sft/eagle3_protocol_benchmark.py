from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import SFTRecord, collate_sft, make_sft_example, records_metadata
from .draver_activation import make_audit_split
from .protocol_features import ProtocolObservations, evaluate_protocol_audit
from .public_data import build_public_snapshot_split
from .training import add_lora, load_causal_lm, save_adapter, set_seed, sft_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hidden-connector EAGLE-3 membership-safety validation."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-revision", required=True)
    parser.add_argument(
        "--speculator-model", default="RedHatAI/Qwen3-8B-speculator.eagle3"
    )
    parser.add_argument("--speculator-revision", required=True)
    parser.add_argument("--speculators-overlay", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-adapter", type=Path)
    parser.add_argument("--speculator-adapter", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--n-per-class", type=int, default=256)
    parser.add_argument("--n-aux", type=int, default=256)
    parser.add_argument("--audit-train-per-class", type=int, default=64)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--target-batch-size", type=int, default=2)
    parser.add_argument("--target-grad-accum", type=int, default=4)
    parser.add_argument("--target-lr", type=float, default=2e-4)
    parser.add_argument("--speculator-distill-steps", type=int, default=256)
    parser.add_argument("--speculator-batch-size", type=int, default=1)
    parser.add_argument("--speculator-lr", type=float, default=2e-4)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--probe-tokens", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--transcript-repeats", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--detector-seeds", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--compatibility-only", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


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


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _target_layer_ids(target: torch.nn.Module) -> list[int]:
    config = target.get_base_model().config if hasattr(target, "get_base_model") else target.config
    layers = int(config.num_hidden_layers)
    return [2, layers // 2, layers - 3]


def _base_speculator(model: torch.nn.Module) -> torch.nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _state_summary(state: torch.Tensor) -> torch.Tensor:
    values = state.float()
    quantiles = torch.quantile(values, torch.tensor([0.25, 0.5, 0.75], device=values.device), dim=-1)
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
        ],
    )
    return get_peft_model(model, config)


def _mapped_teacher_logits(
    target_logits: torch.Tensor, speculator: torch.nn.Module
) -> torch.Tensor:
    base = _base_speculator(speculator)
    return target_logits.index_select(-1, base.d2t.long())


def distill_speculator(
    speculator: torch.nn.Module,
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
        indices = rng.integers(0, len(examples), size=batch_size)
        batch = collate_sft(
            [examples[int(index)] for index in indices], int(tokenizer.pad_token_id)
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.no_grad(), _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            layers = _target_layer_ids(target)
            connector = torch.cat(
                [target_output.hidden_states[index] for index in layers], dim=-1
            )
            teacher = _mapped_teacher_logits(target_output.logits, speculator)

        captured: list[torch.Tensor] = []
        base = _base_speculator(speculator)
        handle = base.norm.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        with _autocast(device):
            speculator(
                input_ids=batch["input_ids"],
                hidden_states=connector,
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            student = base.lm_head(captured[-1]).float()
            valid = batch["labels"].ne(-100)
            student_selected = student[valid]
            teacher_selected = teacher[valid].float()
            loss = F.kl_div(
                F.log_softmax(student_selected / temperature, dim=-1),
                F.softmax(teacher_selected / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
        handle.remove()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(speculator.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 4) == 0:
            print(
                f"EAGLE-3 auxiliary KD step {step + 1}/{steps}: loss={history[-1]:.5f}",
                flush=True,
            )
        del target_output, connector, teacher, student
    speculator.eval()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return history


def _select_mapped_bottom_k(
    q_logp: np.ndarray, mapped: np.ndarray, count: int
) -> np.ndarray:
    selected = np.empty((len(q_logp), count), dtype=np.int64)
    for row in range(len(q_logp)):
        candidates = np.flatnonzero(mapped[row] & np.isfinite(q_logp[row]))
        if len(candidates) < count:
            raise RuntimeError(
                f"record {row} has only {len(candidates)} EAGLE draft-vocab tokens"
            )
        local = candidates[np.argsort(q_logp[row, candidates])[:count]]
        selected[row] = np.sort(local)
    return selected


def _gather(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values))[:, None], selected]


def _simulate_block_transcript(
    alpha: np.ndarray, block_size: int, repeats: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if alpha.shape[1] % block_size:
        raise ValueError("probe token count must be divisible by block size")
    blocks = alpha.reshape(len(alpha), -1, block_size)
    visible_count = np.zeros_like(blocks, dtype=np.int32)
    accepted_count = np.zeros_like(blocks, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        live = np.ones(blocks.shape[:2], dtype=bool)
        for depth in range(block_size):
            visible_count[:, :, depth] += live
            draw = rng.random(blocks.shape[:2]) < blocks[:, :, depth]
            accepted_count[:, :, depth] += live & draw
            live &= draw
    observed = (accepted_count + 0.5) / (visible_count + 1.0)
    visible = visible_count > 0
    return observed.reshape(alpha.shape).astype(np.float32), visible.reshape(alpha.shape)


@torch.no_grad()
def extract_observations(
    speculator: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    probe_tokens: int,
    block_size: int,
    transcript_repeats: int,
    seed: int,
) -> tuple[ProtocolObservations, dict[str, float]]:
    target.eval()
    speculator.eval()
    base = _base_speculator(speculator)
    target_to_draft = np.full(int(target.config.vocab_size), -1, dtype=np.int64)
    d2t = base.d2t.detach().cpu().numpy().astype(np.int64)
    target_to_draft[d2t] = np.arange(len(d2t))

    p_batches: list[np.ndarray] = []
    q_batches: list[np.ndarray] = []
    mapped_batches: list[np.ndarray] = []
    local_batches: list[np.ndarray] = []
    connector_batches: list[np.ndarray] = []
    for batch in _loader(records, tokenizer, batch_size):
        batch = {name: value.to(device) for name, value in batch.items()}
        with _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        layer_ids = _target_layer_ids(target)
        connector_layers = [target_output.hidden_states[index] for index in layer_ids]
        connector = torch.cat(connector_layers, dim=-1)
        captured: list[torch.Tensor] = []
        handle = base.norm.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        with _autocast(device):
            speculator(
                input_ids=batch["input_ids"],
                hidden_states=connector,
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
        handle.remove()
        draft_logits = base.lm_head(captured[-1]).float()
        target_logits = target_output.logits.float()
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        safe_target = labels.clamp_min(0)
        mapping = torch.from_numpy(target_to_draft).to(device)
        draft_ids = mapping[safe_target]
        mapped = draft_ids.ge(0)
        safe_draft = draft_ids.clamp_min(0)
        p_logp = target_logits[:, :-1].log_softmax(dim=-1).gather(
            -1, safe_target.unsqueeze(-1)
        ).squeeze(-1)
        q_logp = draft_logits[:, :-1].log_softmax(dim=-1).gather(
            -1, safe_draft.unsqueeze(-1)
        ).squeeze(-1)

        response_count = int(valid.sum(dim=1).min().item())
        batch_p = np.full((len(labels), response_count), np.nan, dtype=np.float32)
        batch_q = np.full_like(batch_p, np.nan)
        batch_mapped = np.zeros_like(batch_p, dtype=bool)
        local_features = _state_summary(captured[-1][:, :-1])
        connector_features = torch.cat(
            [_state_summary(state[:, :-1]) for state in connector_layers], dim=-1
        )
        batch_local = np.zeros(
            (len(labels), response_count, local_features.shape[-1]), dtype=np.float32
        )
        batch_connector = np.zeros(
            (len(labels), response_count, connector_features.shape[-1]),
            dtype=np.float32,
        )
        for row in range(len(labels)):
            positions = valid[row].nonzero(as_tuple=False).flatten()[:response_count]
            batch_p[row] = p_logp[row, positions].cpu().numpy()
            batch_q[row] = q_logp[row, positions].cpu().numpy()
            batch_mapped[row] = mapped[row, positions].cpu().numpy()
            batch_local[row] = local_features[row, positions].cpu().numpy()
            batch_connector[row] = connector_features[row, positions].cpu().numpy()
        p_batches.append(batch_p)
        q_batches.append(batch_q)
        mapped_batches.append(batch_mapped)
        local_batches.append(batch_local)
        connector_batches.append(batch_connector)
        del target_output, connector, captured, draft_logits, target_logits

    p_logp = np.concatenate(p_batches)
    q_logp = np.concatenate(q_batches)
    mapped = np.concatenate(mapped_batches)
    local = np.concatenate(local_batches)
    connector = np.concatenate(connector_batches)
    selected = _select_mapped_bottom_k(q_logp, mapped, probe_tokens)
    selected_p = _gather(p_logp, selected)
    selected_q = _gather(q_logp, selected)
    alpha = np.minimum(1.0, np.exp(np.clip(selected_p - selected_q, -50, 50)))
    transcript, visible = _simulate_block_transcript(
        alpha, block_size, transcript_repeats, seed
    )
    observations = ProtocolObservations(
        local_state=_gather(local, selected),
        connector_state=_gather(connector, selected),
        acceptance=transcript,
        q_logp=selected_q,
        confidence=None,
        visible=visible,
        depth=np.tile(np.arange(block_size), probe_tokens // block_size),
    )
    diagnostics = {
        "mapped_response_fraction": float(mapped.mean()),
        "mean_exact_acceptance": float(alpha.mean()),
        "mean_observed_acceptance": float(transcript[visible].mean()),
        "visible_fraction": float(visible.mean()),
    }
    return observations, diagnostics


def _load_speculator(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    import sys

    overlay = str(args.speculators_overlay.resolve())
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    from speculators import SpeculatorModel, SpeculatorModelConfig

    SpeculatorModel.registry.pop("eagle3", None)
    SpeculatorModelConfig.registry.pop("eagle3", None)
    snapshot = snapshot_download(
        repo_id=args.speculator_model,
        revision=args.speculator_revision,
        local_files_only=True,
    )
    implementation_path = Path(snapshot) / "eagle3.py"
    module_spec = importlib.util.spec_from_file_location(
        "sd_mia_eagle3_remote", implementation_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load EAGLE-3 implementation: {implementation_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config = module.Eagle3SpeculatorConfig.from_pretrained(
        snapshot, local_files_only=True
    )
    model = module.Eagle3Speculator.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        verifier_attachment_mode="detached",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    return model.to(device)


@torch.no_grad()
def compatibility_check(
    args: argparse.Namespace, tokenizer: Any, device: torch.device
) -> None:
    target = load_causal_lm(
        args.target_model,
        device,
        revision=args.target_revision,
        local_files_only=True,
    )
    speculator = _load_speculator(args, device)
    encoded = tokenizer("EAGLE compatibility check.", return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with _autocast(device):
        target_output = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        layer_ids = _target_layer_ids(target)
        connector = torch.cat(
            [target_output.hidden_states[index] for index in layer_ids], dim=-1
        )
        speculator_output = speculator(
            input_ids=input_ids,
            hidden_states=connector,
            attention_mask=attention_mask,
            return_dict=True,
        )
    print(
        json.dumps(
            {
                "gate": "PASS",
                "target_layers": layer_ids,
                "connector_shape": list(connector.shape),
                "speculator_logits_shape": list(speculator_output.logits.shape),
                "mapped_vocab": int(len(_base_speculator(speculator).d2t)),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.probe_tokens % args.block_size:
        raise ValueError("probe-tokens must be divisible by block-size")
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
    tokenizer = AutoTokenizer.from_pretrained(target_snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if args.compatibility_only:
        compatibility_check(args, tokenizer, device)
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
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    calibration, test = make_audit_split(
        len(members),
        len(nonmembers),
        args.audit_train_per_class,
        args.split_seed + 30,
    )
    started = time.time()

    if target_adapter is None:
        target = add_lora(
            load_causal_lm(
                args.target_model,
                device,
                revision=args.target_revision,
                local_files_only=True,
            ),
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
            "EAGLE verifier member SFT",
        )
        save_adapter(target, output_dir / "adapter_target")
        speculator = _add_speculator_lora(
            _load_speculator(args, device),
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        speculator_loss = distill_speculator(
            speculator,
            target,
            auxiliary,
            tokenizer,
            device,
            args.speculator_distill_steps,
            args.speculator_batch_size,
            args.speculator_lr,
            args.distill_temperature,
            args.seed + 20,
        )
        speculator.save_pretrained(output_dir / "adapter_speculator_aux_distilled")
        reused_adapters = False
    else:
        target = PeftModel.from_pretrained(
            load_causal_lm(
                args.target_model,
                device,
                revision=args.target_revision,
                local_files_only=True,
            ),
            str(target_adapter),
        ).to(device)
        speculator = PeftModel.from_pretrained(
            _load_speculator(args, device), str(speculator_adapter)
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
        args.speculator_batch_size,
        args.probe_tokens,
        args.block_size,
        args.transcript_repeats,
        args.seed + 40,
    )
    audit = evaluate_protocol_audit(
        observations,
        labels,
        calibration,
        test,
        args.block_size,
        args.bootstrap_repeats,
        args.detector_seeds,
        args.seed + 100,
    )
    artifact = {
        "material_passport": {
            "experiment_id": f"eagle3-qwen3-8b-{args.seed}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_HIDDEN_CONNECTOR_SFT",
        },
        "config": {**vars(args), "dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": {
            "target_sft_loss": target_loss,
            "speculator_aux_distill_loss": speculator_loss,
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
    print(json.dumps({"output_dir": str(output_dir), "runtime": artifact["runtime"], "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
