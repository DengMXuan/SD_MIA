from __future__ import annotations

import argparse
import gc
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .audit import make_audit_split, run_audit
from .config import Config
from .data import build_controlled_split, records_metadata
from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    paired_bootstrap_delta,
)
from .nart_data import build_nart_split, pool_path as nart_pool_path
from .training import (
    add_lora,
    distill_on_auxiliary,
    extract_features,
    load_causal_lm,
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
        ("selected-token-cap", int),
        ("target-model", str),
        ("draft-model", str),
    ]:
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=kind)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--target-lr", type=float)
    parser.add_argument("--draft-lr", type=float)
    parser.add_argument(
        "--benchmark",
        choices=["legacy", "wikitection", "newstection", "arxivtection"],
    )
    parser.add_argument("--trainer", choices=["lora", "full"])
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"])
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


def build_baseline_comparison(
    cfg: Config,
    metrics: dict[str, dict[str, float]],
    raw_scores: dict[str, np.ndarray],
    draver_result: dict[str, Any] | None,
    labels: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Direct-verifier MIA baselines vs SD-scenario signals, same test split.

    Baselines (score-based attacks on the fine-tuned target, no draft or
    protocol signal) and the SD transcript/DraVer-Act detectors all score the
    shared audit test records, so paired bootstrap deltas are valid.
    """
    prefix = next(
        (
            key.split("/")[0]
            for key in metrics
            if key.endswith("/honest_accept_rate_selected")
        ),
        None,
    )
    sd_signals = [
        ("Verifier transcript: accept rate (624-bit)", f"{prefix}/honest_accept_rate_selected"),
        ("Verifier transcript: acceptance tomography", f"{prefix}/acceptance_tomography_qmin"),
    ]
    if draver_result is not None:
        sd_signals.extend(
            [
                ("Transcript-only detector (triplet)", "draver_act/transcript_only_triplet"),
                ("DraVer-Act (residual, proposed)", "draver_act/draver_act_residual_triplet"),
            ]
        )
    baselines = [
        ("Min-K% Prob (k=0.2)", "verifier_direct/min_k_prob_k20"),
        ("WBC (w=2..40, |W|=10)", "verifier_direct/window_based_comparison"),
        ("Reference loss-diff (global)", "verifier_direct/reference_loss_diff"),
    ]

    rows = []
    for group, entries in (("SD", sd_signals), ("direct baseline", baselines)):
        for label, key in entries:
            row = metrics.get(key)
            if row is None:
                continue
            rows.append({"group": group, "signal": label, "key": key, **row})

    deltas: dict[str, dict[str, float]] = {}
    if draver_result is not None:
        labels_test = labels[test_idx]
        headline = (
            ("DraVer-Act", draver_result["scores"]["draver_act_residual_triplet"]),
            ("Transcript-only (triplet)", draver_result["scores"]["transcript_only_triplet"]),
        )
        for sd_label, sd_score in headline:
            for baseline_label, baseline_key in baselines:
                baseline_score = raw_scores.get(baseline_key)
                if baseline_score is None:
                    continue
                deltas[f"{sd_label} - {baseline_label}"] = paired_bootstrap_delta(
                    labels_test,
                    sd_score,
                    baseline_score,
                    cfg.bootstrap_repeats,
                    cfg.audit_seed + 900,
                )
    return {
        "rows": rows,
        "paired_deltas": deltas,
        "draver_act_available": draver_result is not None,
    }


def render_markdown(
    cfg: Config,
    metadata: dict[str, Any],
    training: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    budget: dict[str, float],
    comparison: dict[str, Any] | None = None,
) -> str:
    def trace(values: list[float]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{values[0]:.4f} → {values[-1]:.4f}"

    lines = [
        (
            "# NART-Style Full-Parameter SFT Edge–Cloud SD Membership Audit"
            if cfg.benchmark != "legacy" and cfg.trainer == "full"
            else "# Qwen3 Instruction-SFT Edge–Cloud SD Membership Audit"
        ),
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{'qwen3' if cfg.benchmark == 'legacy' else cfg.benchmark}-sft-{cfg.seed}-epoch{cfg.target_epochs}`",
        "- Status: COMPLETED",
        "- Verification status: single-seed controlled experiment",
        f"- Training objective: {'full-parameter' if cfg.trainer == 'full' else 'LoRA'} "
        f"instruction SFT ({cfg.optimizer}); prompt labels masked with `-100`",
        f"- Benchmark: {cfg.benchmark}",
        f"- Transcript position cap: {cfg.selected_token_cap} "
        f"(budget {cfg.selected_token_cap * cfg.transcript_repeats} bits/record at "
        f"{cfg.transcript_repeats} repeats)",
        "- Signal boundary: draft white-box features plus intended verifier feedback",
        f"- Raw text persisted: {'pool only (public post-cutoff corpus)' if cfg.benchmark != 'legacy' else 'no'}",
        "",
        "## Model and SFT setting",
        "",
        f"- Target: `{cfg.target_model}`",
        f"- Draft: `{cfg.draft_model}`",
        f"- Target SFT epochs: {cfg.target_epochs}; "
        + (
            f"LoRA rank: {cfg.lora_r}"
            if cfg.trainer == "lora"
            else "full-parameter, lr 2e-5, effective batch 16"
        ),
        f"- Member records: {cfg.n_per_class}; nonmember records: {cfg.n_per_class}",
        f"- Auxiliary distillation records: {cfg.n_aux}",
        *(
            [
                f"- Assistant response length: {cfg.response_tokens} source tokens plus EOS",
                "- SFT response: source paragraph chunk; prompt: source title and fixed technical-writing instruction",
            ]
            if cfg.benchmark == "legacy"
            else [
                f"- SFT response: full document continuation ({cfg.benchmark} token band) plus EOS",
                "- SFT prompt: NART fixed prompt with topic line; only document tokens contribute loss",
            ]
        ),
        "",
        "## Data controls",
        "",
    ]
    if cfg.benchmark == "legacy":
        lines.extend(
            [
                f"- Source PDFs: {metadata['source_pdf_count']}",
                f"- Unique response chunks: {metadata['unique_chunk_count']}",
                "- Member/nonmember allocation is source-stratified, shuffled, and "
                "globally hash-deduplicated",
                "- Auxiliary records are disjoint from both audit classes",
            ]
        )
    else:
        window = metadata.get("creation_interval_inclusive") or {}
        lines.extend(
            [
                f"- Frozen pool: `{metadata['pool_path']}` "
                f"(sha256 {metadata['pool_sha256'][:16]}, {metadata['pool_records']} documents)",
                f"- Creation window: {window.get('start')} .. {window.get('end')}; "
                f"{metadata.get('timestamp_semantics')}",
                f"- Token band: {metadata['token_band']['min_tokens']}.."
                f"{metadata['token_band']['max_tokens']} tokens per document; "
                f"{metadata['band_dropped_documents']} dropped below band",
                f"- License: {metadata.get('license')}",
                f"- Provenance: {metadata.get('provenance')}",
                "- Member/nonmember/auxiliary allocation is shuffled and hash-deduplicated",
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
        "## Held-out membership-audit results",
        "",
        "Higher AUC is better; 0.5 is random. Learned probes use only the audit-calibration split.",
        "",
        "| Signal | AUC (95% bootstrap CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    )
    for name, row in sorted(metrics.items()):
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] "
            f"| {row['tpr_at_1pct_fpr']:.3f} | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    if comparison is not None and comparison["rows"]:
        lines.extend(
            [
                "",
                "## Direct-verifier baselines vs SD-scenario signals",
                "",
                "All signals score the same audit test records. Direct baselines "
                "attack the fine-tuned verifier without any draft or protocol "
                "signal (Min-K% Prob: Shi et al. ICLR 2024; WBC: Chen et al. "
                "USENIX Security 2026, w=2..40, |W|=10; reference loss-diff: "
                "the global-average baseline WBC compares against). SD signals "
                "additionally use speculative-decoding information.",
                "",
                "| Group | Signal | AUC (95% CI) | TPR@1%FPR | TPR@5%FPR |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in comparison["rows"]:
            lines.append(
                f"| {row['group']} | {row['signal']} "
                f"| {row['auc']:.4f} [{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}] "
                f"| {row['tpr_at_1pct_fpr']:.4f} | {row['tpr_at_5pct_fpr']:.4f} |"
            )
        if comparison["paired_deltas"]:
            lines.extend(
                [
                    "",
                    "Paired AUC deltas (positive = the SD signal beats the direct "
                    "baseline on the same test records):",
                    "",
                    "| SD signal | Baseline | Delta AUC (95% CI) |",
                    "|---|---|---:|",
                ]
            )
            for name, delta in comparison["paired_deltas"].items():
                sd_label, baseline_label = name.split(" - ", 1)
                lines.append(
                    f"| {sd_label} | {baseline_label} "
                    f"| {delta['delta_auc']:+.4f} [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
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
            (
                "LoRA is used to make the 8B target fit on one A100; results therefore "
                "describe adapter-based SFT."
                if cfg.trainer == "lora"
                else "Full-parameter NART-style fine-tuning (lr 2e-5, effective batch 16) "
                "with a bitsandbytes paged 8-bit AdamW on one A100-80GB."
            ),
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
    if cfg.benchmark == "legacy":
        members, nonmembers, auxiliary, data_metadata = build_controlled_split(
            root,
            tokenizer,
            cfg.response_tokens,
            cfg.n_per_class,
            cfg.n_aux,
            cfg.data_seed,
        )
    else:
        pool = cfg.pool_path if cfg.pool_path is not None else nart_pool_path(cfg.benchmark)
        if not pool.is_absolute():
            pool = root / pool
        members, nonmembers, auxiliary, data_metadata = build_nart_split(
            cfg.benchmark,
            pool,
            tokenizer,
            cfg.n_per_class,
            cfg.n_aux,
            cfg.data_seed,
        )
    candidates = members + nonmembers
    full_finetune = cfg.trainer == "full"
    checkpoint_dir = "checkpoints" if full_finetune else "adapters"

    started = time.time()
    target = load_causal_lm(cfg.target_model, device)
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
    draft_activation_features = None
    aux_distill_loss: list[float] = []
    member_draft_sft_loss: list[float] = []
    if cfg.run_auxiliary_draft:
        auxiliary_draft = load_causal_lm(cfg.draft_model, device)
        if not full_finetune:
            auxiliary_draft = add_lora(
                auxiliary_draft,
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
            optimizer_name=cfg.optimizer,
        )
        if cfg.save_adapters:
            save_trained_model(
                auxiliary_draft,
                output_dir / checkpoint_dir / "draft_auxiliary_distilled",
            )
        draft_features["aux_distilled_draft"] = extract_features(
            auxiliary_draft, candidates, tokenizer, device, cfg.draft_batch_size
        )
        if cfg.benchmark != "legacy":
            draft_activation_features = extract_draft_activation_outputs(
                auxiliary_draft, candidates, tokenizer, device, cfg.draft_batch_size
            )
        del auxiliary_draft
        gc.collect()
        torch.cuda.empty_cache()

    if cfg.run_member_draft:
        member_draft = load_causal_lm(cfg.draft_model, device)
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
        draft_features["member_sft_draft"] = extract_features(
            member_draft, candidates, tokenizer, device, cfg.draft_batch_size
        )
        del member_draft
        gc.collect()
        torch.cuda.empty_cache()

    # Direct-verifier baselines need the pre-fine-tuning target's per-token
    # losses as the reference signal (WBC, reference loss-diff).
    base_target_model = load_causal_lm(cfg.target_model, device)
    base_target_features = extract_features(
        base_target_model, candidates, tokenizer, device, cfg.target_batch_size
    )
    del base_target_model
    gc.collect()
    torch.cuda.empty_cache()

    audit_split = make_audit_split(
        len(members), len(nonmembers), cfg.audit_train_per_class, cfg.audit_seed
    )
    audit_labels = np.concatenate(
        [
            np.ones(len(members), dtype=np.int64),
            np.zeros(len(nonmembers), dtype=np.int64),
        ]
    )
    metrics, budget, raw_scores = run_audit(
        members,
        nonmembers,
        target_features,
        draft_features,
        cfg,
        selected_token_cap=cfg.selected_token_cap,
        base_target_features=base_target_features,
        split=audit_split,
    )

    draver_result = None
    if draft_activation_features is not None:
        draver_result = evaluate_activation_audit(
            draft_activation_features,
            target_features,
            audit_labels,
            calibration=audit_split[0],
            test=audit_split[1],
            min_k_fraction=cfg.min_k_fraction,
            transcript_repeats=cfg.transcript_repeats,
            bootstrap_repeats=cfg.bootstrap_repeats,
            detector_seeds=3,
            seed=cfg.audit_seed,
            few_shot_per_class=(),
            selected_token_cap=cfg.selected_token_cap,
        )
        for name, row in draver_result["metrics"].items():
            metrics[f"draver_act/{name}"] = row

    comparison = build_baseline_comparison(
        cfg, metrics, raw_scores, draver_result, audit_labels, audit_split[1]
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
            "experiment_id": (
                f"{'qwen3' if cfg.benchmark == 'legacy' else cfg.benchmark}"
                f"-sft-{cfg.seed}-epoch{cfg.target_epochs}"
            ),
            "status": "COMPLETED",
            "verification_status": "ANALYZED_SINGLE_SEED_CONTROLLED_SFT",
            "benchmark": cfg.benchmark,
            "trainer": cfg.trainer,
            "optimizer": cfg.optimizer,
            "selected_token_cap": cfg.selected_token_cap,
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
        "baseline_comparison": comparison,
        "query_budget": budget,
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(cfg, data_metadata, training, metrics, budget, comparison),
        encoding="utf-8",
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
