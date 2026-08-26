from __future__ import annotations

import argparse
import gc
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .audit import run_audit
from .config import Config
from .data import build_controlled_split, records_metadata
from .training import (
    add_lora,
    distill_on_auxiliary,
    extract_features,
    load_causal_lm,
    save_adapter,
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
        ("audit-seed", int),
        ("target-epochs", int),
        ("n-per-class", int),
        ("n-aux", int),
        ("audit-train-per-class", int),
        ("response-tokens", int),
        ("target-batch-size", int),
        ("target-grad-accum", int),
        ("draft-batch-size", int),
        ("draft-grad-accum", int),
        ("distill-steps", int),
        ("bootstrap-repeats", int),
    ]:
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=kind)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-trained-drafts", action="store_true", default=None)
    parser.add_argument("--no-save-adapters", action="store_true", default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    values = Config().as_dict()
    if args.config:
        with args.config.open("rb") as handle:
            values.update(tomllib.load(handle))
    for key, value in vars(args).items():
        if key in {"config", "skip_trained_drafts", "no_save_adapters"} or value is None:
            continue
        values[key] = value
    if args.skip_trained_drafts:
        values["run_auxiliary_draft"] = False
        values["run_member_draft"] = False
    if args.no_save_adapters:
        values["save_adapters"] = False
    values["output_dir"] = Path(values["output_dir"])
    return Config(**values)


def render_markdown(
    cfg: Config,
    metadata: dict[str, Any],
    training: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    budget: dict[str, float],
) -> str:
    def trace(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{values[0]:.4f} → {values[-1]:.4f}"

    lines = [
        "# Qwen3 Instruction-SFT Edge–Cloud SD Membership Audit",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `qwen3-sft-{cfg.seed}-epoch{cfg.target_epochs}`",
        "- Status: COMPLETED",
        "- Verification status: single-seed controlled experiment",
        "- Training objective: instruction SFT; prompt labels masked with `-100`",
        "- Signal boundary: draft white-box features plus intended verifier feedback",
        "- Raw PDF text persisted: no",
        "",
        "## Model and SFT setting",
        "",
        f"- Target: `{cfg.target_model}`",
        f"- Draft: `{cfg.draft_model}`",
        f"- Target SFT epochs: {cfg.target_epochs}; LoRA rank: {cfg.lora_r}",
        f"- Member records: {cfg.n_per_class}; nonmember records: {cfg.n_per_class}",
        f"- Auxiliary distillation records: {cfg.n_aux}",
        f"- Assistant response length: {cfg.response_tokens} source tokens plus EOS",
        "- SFT response: source paragraph chunk; prompt: source title and fixed technical-writing instruction",
        "",
        "## Data controls",
        "",
        f"- Source PDFs: {metadata['source_pdf_count']}",
        f"- Unique response chunks: {metadata['unique_chunk_count']}",
        "- Member/nonmember allocation is source-stratified, shuffled, and globally hash-deduplicated",
        "- Auxiliary records are disjoint from both audit classes",
        "",
        "## Training trace",
        "",
        f"- Target SFT loss: {trace(training['target_sft_loss'])}",
        f"- Auxiliary draft distillation loss: {trace(training['aux_distill_loss'])}",
        f"- Member-data draft SFT loss: {trace(training['member_draft_sft_loss'])}",
        f"- Peak allocated GPU memory: {training['peak_gpu_memory_gib']:.2f} GiB",
        "",
        "## Held-out membership-audit results",
        "",
        "Higher AUC is better; 0.5 is random. Learned probes use only the audit-calibration split.",
        "",
        "| Signal | AUC (95% bootstrap CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(metrics.items()):
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] "
            f"| {row['tpr_at_1pct_fpr']:.3f} | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Transcript query budget",
            "",
            f"- q-min selection: median {budget['qmin_median_bits']:.0f} bits, "
            f"P95 {budget['qmin_p95_bits']:.0f}",
            f"- random selection: median {budget['random_median_bits']:.0f} bits, "
            f"P95 {budget['random_p95_bits']:.0f}",
            "",
            "## Interpretation boundary",
            "",
            "This is a controlled SFT-membership experiment, not a claim about Qwen3 pretraining-data membership.",
            "LoRA is used to make the 8B target fit on one A100; results therefore describe adapter-based SFT.",
            "The transcript is a local semantic verifier simulation, not a deployment measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    if cfg.audit_train_per_class >= cfg.n_per_class:
        raise ValueError("audit_train_per_class must be smaller than n_per_class")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")

    root = Path(__file__).resolve().parents[2]
    output_dir = cfg.output_dir if cfg.output_dir.is_absolute() else root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_adapters:
        (output_dir / "adapters").mkdir(exist_ok=True)

    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(cfg.seed)
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    members, nonmembers, auxiliary, data_metadata = build_controlled_split(
        root,
        tokenizer,
        cfg.response_tokens,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    candidates = members + nonmembers

    started = time.time()
    target = add_lora(
        load_causal_lm(cfg.target_model, device),
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
    )
    if cfg.save_adapters:
        save_adapter(target, output_dir / "adapters" / "target")
    target_features = extract_features(
        target, candidates, tokenizer, device, cfg.target_batch_size
    )

    base_draft = load_causal_lm(cfg.draft_model, device)
    base_features = extract_features(
        base_draft, candidates, tokenizer, device, cfg.draft_batch_size
    )
    del base_draft
    gc.collect()
    torch.cuda.empty_cache()

    draft_features = {"base_draft": base_features}
    aux_distill_loss: list[float] = []
    member_draft_sft_loss: list[float] = []
    if cfg.run_auxiliary_draft:
        auxiliary_draft = add_lora(
            load_causal_lm(cfg.draft_model, device),
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout,
        )
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
        )
        if cfg.save_adapters:
            save_adapter(
                auxiliary_draft,
                output_dir / "adapters" / "draft_auxiliary_distilled",
            )
        draft_features["aux_distilled_draft"] = extract_features(
            auxiliary_draft, candidates, tokenizer, device, cfg.draft_batch_size
        )
        del auxiliary_draft
        gc.collect()
        torch.cuda.empty_cache()

    if cfg.run_member_draft:
        member_draft = add_lora(
            load_causal_lm(cfg.draft_model, device),
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
        )
        if cfg.save_adapters:
            save_adapter(member_draft, output_dir / "adapters" / "draft_member_sft")
        draft_features["member_sft_draft"] = extract_features(
            member_draft, candidates, tokenizer, device, cfg.draft_batch_size
        )
        del member_draft
        gc.collect()
        torch.cuda.empty_cache()

    metrics, budget = run_audit(
        members,
        nonmembers,
        target_features,
        draft_features,
        cfg,
    )
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
            "experiment_id": f"qwen3-sft-{cfg.seed}-epoch{cfg.target_epochs}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_SINGLE_SEED_CONTROLLED_SFT",
        },
        "config": cfg.as_dict(),
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": training,
        "metrics": metrics,
        "query_budget": budget,
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(cfg, data_metadata, training, metrics, budget), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "training": training, "budget": budget},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
