"""Plain small-model draft: fine-tune the target and its draft LMs on post-cutoff pool data.

Draft approach 1 of 3 (see :mod:`drafts`): the deployment adapts a plain
small causal LM (default Qwen3-1.7B-Base) rather than a specialized
drafter head. The pipeline builds the controlled split from a frozen
pool, fine-tunes the target on member records, distills an
auxiliary-only draft (member-blind, deployment-aligned), optionally
fine-tunes a member-data draft (boundary condition), and saves
everything under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..data import records_metadata
from ..splits import build_split, pool_path
from ..training import (
    add_lora,
    distill_on_auxiliary,
    load_causal_lm,
    load_tokenizer,
    save_trained_model,
    set_seed,
    sft_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for name, kind in [
        ("gpu", int),
        ("seed", int),
        ("data-seed", int),
        ("target-epochs", int),
        ("n-per-class", int),
        ("n-aux", int),
        ("target-batch-size", int),
        ("target-grad-accum", int),
        ("draft-batch-size", int),
        ("draft-grad-accum", int),
        ("distill-steps", int),
        ("target-model", str),
        ("draft-model", str),
        ("target-revision", str),
        ("draft-revision", str),
    ]:
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=kind)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--target-lr", type=float)
    parser.add_argument("--draft-lr", type=float)
    parser.add_argument(
        "--benchmark",
        choices=["wikitection", "newstection", "arxivtection"],
    )
    parser.add_argument("--trainer", choices=["lora", "full"])
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        "--skip-training",
        dest="resume",
        action="store_true",
        default=None,
        help="reuse every complete checkpoint and train only missing stages",
    )
    parser.add_argument("--skip-trained-drafts", action="store_true", default=None)
    parser.add_argument("--no-save-adapters", action="store_true", default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    values = Config().as_dict()
    if args.config:
        with args.config.open("rb") as handle:
            values.update(tomllib.load(handle))
    for key, value in vars(args).items():
        if key in {
            "config",
            "skip_trained_drafts",
            "no_save_adapters",
            "resume",
        } or value is None:
            continue
        values[key] = value
    if args.skip_trained_drafts:
        values["run_auxiliary_draft"] = False
        values["run_member_draft"] = False
    if args.no_save_adapters:
        values["save_adapters"] = False
    if values["seed"] != values["data_seed"]:
        raise ValueError(
            "controlled model-pair conditions require --seed and --data-seed "
            "to be identical"
        )
    values["output_dir"] = Path(values["output_dir"])
    return Config(**values)


def render_markdown(
    cfg: Config,
    metadata: dict[str, Any],
    training: dict[str, Any],
) -> str:
    def trace(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{values[0]:.4f} → {values[-1]:.4f}"

    lines = [
        "# Full-Parameter SFT Condition"
        if cfg.trainer == "full"
        else "# LoRA SFT Condition",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{cfg.benchmark}-sft-{cfg.seed}-epoch{cfg.target_epochs}`",
        "- Status: COMPLETED",
        f"- Training objective: {'full-parameter' if cfg.trainer == 'full' else 'LoRA'} "
        f"instruction SFT ({cfg.optimizer}); prompt labels masked with `-100`",
        f"- Benchmark: {cfg.benchmark}",
        "- Raw text persisted: pool only (public post-cutoff corpus)",
        "",
        "## Model and SFT setting",
        "",
        f"- Target: `{cfg.target_model}`",
        f"- Draft: `{cfg.draft_model}`",
        f"- Target revision: `{cfg.target_revision or 'default'}`",
        f"- Draft revision: `{cfg.draft_revision or 'default'}`",
        f"- Target SFT epochs: {cfg.target_epochs}; "
        + (
            f"LoRA rank: {cfg.lora_r}"
            if cfg.trainer == "lora"
            else "full-parameter, lr 2e-5, effective batch 16"
        ),
        f"- Member records: {cfg.n_per_class}; nonmember records: {cfg.n_per_class}",
        f"- Auxiliary distillation records: {cfg.n_aux}",
        f"- SFT response: full document continuation ({cfg.benchmark} token band) plus EOS",
        "- SFT prompt: fixed instruction prompt with topic line; only document tokens contribute loss",
        "",
        "## Data controls",
        "",
    ]
    window = metadata.get("creation_interval_inclusive") or {}
    lines.extend(
        [
            f"- Frozen pool: `{metadata['pool_path']}` "
            f"(sha256 {metadata['pool_sha256'][:16]}, {metadata['pool_records']} documents)",
            f"- Creation window: {window.get('start')} .. {window.get('end')}; "
            f"{metadata.get('timestamp_semantics')}",
            f"- Token band: {metadata['token_band']['min_tokens']}.."
            f"{metadata['token_band']['max_tokens']} tokens per document; "
            f"{metadata.get('band_dropped_documents')} dropped below band",
            f"- License: {metadata.get('license')}",
            f"- Provenance: {metadata.get('provenance')}",
            "- Member/nonmember/auxiliary allocation is shuffled and hash-deduplicated",
            f"- Condition seed: {cfg.seed}",
            f"- Cross-split 13-gram gate: "
            f"{metadata['cross_split_ngram_audit']['gate']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Training trace",
            "",
            f"- Target SFT loss: {trace(training['target_sft_loss'])}",
            f"- Auxiliary draft distillation loss: {trace(training['aux_distill_loss'])}",
            f"- Member-data draft SFT loss: {trace(training['member_draft_sft_loss'])}",
            f"- Peak allocated GPU memory: {training['peak_gpu_memory_gib']:.2f} GiB",
            "",
            "## Interpretation boundary",
            "",
            "This is a controlled instruction-SFT condition on a public post-cutoff "
            "corpus; membership is defined by the training assignment, not by "
            "pretraining exposure.",
            (
                "LoRA is used to make the 8B target fit on one A100; results therefore "
                "describe adapter-based SFT."
                if cfg.trainer == "lora"
                else "Full-parameter fine-tuning (lr 2e-5, effective batch 16) "
                "with a bitsandbytes paged 8-bit AdamW on one A100-80GB."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _offload_model(model: torch.nn.Module, device: torch.device) -> None:
    """Release a trained model's CUDA allocations before loading the next model."""
    model.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _checkpoint_complete(path: Path) -> bool:
    """Return whether a PEFT adapter or full checkpoint finished saving."""
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").is_file():
        return (path / "adapter_model.safetensors").is_file() or (
            path / "adapter_model.bin"
        ).is_file()
    if not (path / "config.json").is_file():
        return False
    return any(
        (path / filename).is_file()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")

    root = Path(__file__).resolve().parents[3]
    output_dir = cfg.output_dir if cfg.output_dir.is_absolute() else root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_adapters:
        (output_dir / "adapters").mkdir(exist_ok=True)

    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(cfg.seed)
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)

    tokenizer = load_tokenizer(cfg.draft_model, revision=cfg.draft_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pool = cfg.pool_path if cfg.pool_path is not None else pool_path(cfg.benchmark)
    if not pool.is_absolute():
        pool = root / pool
    members, nonmembers, auxiliary, data_metadata = build_split(
        cfg.benchmark,
        pool,
        tokenizer,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    full_finetune = cfg.trainer == "full"
    checkpoint_dir = "checkpoints" if full_finetune else "adapters"

    started = time.time()
    resume = bool(args.resume)
    target_ckpt = output_dir / checkpoint_dir / "target"
    if resume and _checkpoint_complete(target_ckpt):
        from ..generalization import load_finetuned_model

        target = load_finetuned_model(output_dir, cfg.target_model, device)
        target.eval()
        target_sft_loss: list[float] = []
        print(f"resumed target from {target_ckpt}", flush=True)
    else:
        target = load_causal_lm(
            cfg.target_model, device, revision=cfg.target_revision
        )
        if not full_finetune:
            target = add_lora(
                target,
                cfg.lora_r,
                cfg.lora_alpha,
                cfg.lora_dropout,
            )
        target_sft_loss = sft_train(
            target,
            members,
            tokenizer,
            device,
            cfg.target_epochs,
            cfg.target_batch_size,
            cfg.target_grad_accum,
            cfg.target_lr,
            cfg.seed + 10,
            "target SFT",
            optimizer_name=cfg.optimizer,
        )
        if cfg.save_adapters:
            save_trained_model(target, output_dir / checkpoint_dir / "target")
            if not _checkpoint_complete(target_ckpt):
                raise RuntimeError(f"target checkpoint did not finish saving: {target_ckpt}")

    aux_distill_loss: list[float] = []
    member_draft_sft_loss: list[float] = []
    if cfg.run_auxiliary_draft:
        _offload_model(target, device)
        aux_ckpt = output_dir / checkpoint_dir / "draft_auxiliary_distilled"
        if resume and _checkpoint_complete(aux_ckpt):
            aux_distill_loss = []
            print(f"resumed auxiliary draft from {aux_ckpt}", flush=True)
        else:
            auxiliary_draft = load_causal_lm(
                cfg.draft_model, device, revision=cfg.draft_revision
            )
            if not full_finetune:
                auxiliary_draft = add_lora(
                    auxiliary_draft,
                    cfg.lora_r,
                    cfg.lora_alpha,
                    cfg.lora_dropout,
                )
            target.to(device)
            aux_distill_loss = distill_on_auxiliary(
                auxiliary_draft,
                target,
                auxiliary,
                tokenizer,
                device,
                cfg.distill_steps,
                cfg.draft_batch_size,
                cfg.draft_lr,
                cfg.distill_temperature,
                cfg.seed + 20,
                optimizer_name=cfg.optimizer,
            )
            if cfg.save_adapters:
                save_trained_model(
                    auxiliary_draft,
                    output_dir / checkpoint_dir / "draft_auxiliary_distilled",
                )
                if not _checkpoint_complete(aux_ckpt):
                    raise RuntimeError(
                        f"auxiliary draft checkpoint did not finish saving: {aux_ckpt}"
                    )
            _offload_model(target, device)
            del auxiliary_draft
        gc.collect()
        torch.cuda.empty_cache()

    del target
    gc.collect()
    torch.cuda.empty_cache()

    if cfg.run_member_draft:
        member_ckpt = output_dir / checkpoint_dir / "draft_member_sft"
        if resume and _checkpoint_complete(member_ckpt):
            member_draft_sft_loss = []
            print(f"resumed member draft from {member_ckpt}", flush=True)
        else:
            member_draft = load_causal_lm(
                cfg.draft_model, device, revision=cfg.draft_revision
            )
            if not full_finetune:
                member_draft = add_lora(
                    member_draft,
                    cfg.lora_r,
                    cfg.lora_alpha,
                    cfg.lora_dropout,
                )
            member_draft_sft_loss = sft_train(
                member_draft,
                members,
                tokenizer,
                device,
                cfg.target_epochs,
                cfg.draft_batch_size,
                cfg.draft_grad_accum,
                cfg.draft_lr,
                cfg.seed + 21,
                "member-data draft SFT",
                optimizer_name=cfg.optimizer,
            )
            if cfg.save_adapters:
                save_trained_model(
                    member_draft, output_dir / checkpoint_dir / "draft_member_sft"
                )
                if not _checkpoint_complete(member_ckpt):
                    raise RuntimeError(
                        f"member draft checkpoint did not finish saving: {member_ckpt}"
                    )
            del member_draft
        gc.collect()
        torch.cuda.empty_cache()

    training = {
        "target_sft_loss": target_sft_loss,
        "aux_distill_loss": aux_distill_loss,
        "member_draft_sft_loss": member_draft_sft_loss,
        "duration_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "device": torch.cuda.get_device_name(device),
    }
    artifact = {
        "material_passport": {
            "experiment_id": (
                f"{cfg.benchmark}-sft-{cfg.seed}-epoch{cfg.target_epochs}"
            ),
            "status": "COMPLETED",
            "verification_status": "COMPLETED_CONTROLLED_SFT_CONDITION",
            "benchmark": cfg.benchmark,
            "trainer": cfg.trainer,
            "optimizer": cfg.optimizer,
        },
        "config": cfg.as_dict(),
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": training,
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(cfg, data_metadata, training),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "training": training},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
