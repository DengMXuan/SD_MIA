from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .audit import metric_row
from .data import records_metadata
from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    extract_pair_alignment_outputs,
    extract_target_token_outputs,
    make_audit_split,
    triplet_ensemble_scores,
)
from .public_data import build_public_snapshot_split
from .training import (
    add_lora,
    distill_on_auxiliary,
    load_causal_lm,
    save_adapter,
    set_seed,
    sft_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deployment-aligned public controlled-SFT validation for DraVer-Act."
        )
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("experiments/data/fineweb_cc-main-2025-26.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Keep the public/audit split fixed while varying the training seed.",
    )
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--n-per-class", type=int, default=160)
    parser.add_argument("--n-aux", type=int, default=160)
    parser.add_argument("--audit-train-per-class", type=int, default=48)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--target-batch-size", type=int, default=4)
    parser.add_argument("--target-grad-accum", type=int, default=4)
    parser.add_argument("--target-lr", type=float, default=2e-4)
    parser.add_argument("--draft-batch-size", type=int, default=4)
    parser.add_argument("--draft-grad-accum", type=int, default=4)
    parser.add_argument("--draft-epochs", type=int)
    parser.add_argument("--draft-lr", type=float, default=2e-4)
    parser.add_argument("--distill-steps", type=int, default=80)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--draft-adaptation",
        choices=("none", "aux_distill", "member_sft", "both"),
        default="both",
        help=(
            "Use auxiliary-only target-to-draft distillation, same-member SFT, "
            "both deployment conditions, or retain only the mismatch control."
        ),
    )
    parser.add_argument(
        "--pair-diagnostic-batch-size",
        type=int,
        default=1,
        help="Batch size for exact sum_v min(p, q) alignment diagnostics.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-k-fraction", type=float, default=0.20)
    parser.add_argument("--transcript-repeats", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--detector-seeds", type=int, default=3)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="Bound CPU parallelism for the small triplet encoders.",
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_local_tokenizer(model_id: str, revision: str) -> Any:
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=True,
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def _mean_token_logp(values: np.ndarray) -> np.ndarray:
    return np.nanmean(values, axis=1)


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])
    ):
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def _check_tokenizers(draft: Any, target: Any) -> dict[str, Any]:
    draft_vocab = draft.get_vocab()
    target_vocab = target.get_vocab()
    special_names = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    draft_special = {name: getattr(draft, name) for name in special_names}
    target_special = {name: getattr(target, name) for name in special_names}
    result = {
        "draft_vocab_entries": len(draft_vocab),
        "target_vocab_entries": len(target_vocab),
        "draft_tokenizer_length": len(draft),
        "target_tokenizer_length": len(target),
        "draft_vocab_sha256": _tokenizer_fingerprint(draft),
        "target_vocab_sha256": _tokenizer_fingerprint(target),
        "draft_special_token_ids": draft_special,
        "target_special_token_ids": target_special,
        "mapping_identical": draft_vocab == target_vocab,
        "special_ids_identical": draft_special == target_special,
    }
    if (
        not result["mapping_identical"]
        or len(draft) != len(target)
        or not result["special_ids_identical"]
    ):
        raise ValueError(
            "Classic speculative sampling requires identical draft/target token-id "
            "mappings and special-token IDs; this model pair does not satisfy "
            "that requirement"
        )
    return result


def _endpoint_diagnostics(
    target_outputs: dict[str, np.ndarray], labels: np.ndarray, test: np.ndarray
) -> dict[str, float]:
    scores = _mean_token_logp(target_outputs["token_logp"])[test]
    test_labels = labels[test]
    member = scores[test_labels == 1]
    nonmember = scores[test_labels == 0]
    return {
        "mean_token_logp_member": float(member.mean()),
        "mean_token_logp_nonmember": float(nonmember.mean()),
        "member_minus_nonmember": float(member.mean() - nonmember.mean()),
    }


def _pair_alignment_diagnostics(
    pair_outputs: dict[str, np.ndarray], labels: np.ndarray, test: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    test_labels = labels[test]
    for name, raw_values in pair_outputs.items():
        values = np.asarray(raw_values)[test]
        member = values[test_labels == 1]
        nonmember = values[test_labels == 0]
        result[f"{name}_overall"] = float(values.mean())
        result[f"{name}_member"] = float(member.mean())
        result[f"{name}_nonmember"] = float(nonmember.mean())
        result[f"{name}_member_minus_nonmember"] = float(
            member.mean() - nonmember.mean()
        )
    return result


def _alignment_recovery(
    diagnostics: dict[str, dict[str, float]], aligned_names: list[str]
) -> dict[str, dict[str, Any]]:
    if "target_only_mismatch" not in diagnostics:
        return {}
    mismatch = diagnostics["target_only_mismatch"]
    base = diagnostics["base_pair_unadapted"]
    denominator = (
        base["exact_acceptance_overall"]
        - mismatch["exact_acceptance_overall"]
    )
    result: dict[str, dict[str, Any]] = {}
    for name in aligned_names:
        aligned = diagnostics[name]
        acceptance_gain = (
            aligned["exact_acceptance_overall"]
            - mismatch["exact_acceptance_overall"]
        )
        rmse_reduction = (
            mismatch["candidate_logp_rmse_overall"]
            - aligned["candidate_logp_rmse_overall"]
        )
        result[name] = {
            "exact_acceptance_gain_vs_target_only": float(acceptance_gain),
            "base_acceptance_loss_restored_fraction": (
                float(acceptance_gain / denominator)
                if denominator > 1e-8
                else None
            ),
            "top1_agreement_gain_vs_target_only": float(
                aligned["top1_agreement_overall"]
                - mismatch["top1_agreement_overall"]
            ),
            "candidate_logp_rmse_reduction_vs_target_only": float(rmse_reduction),
            "alignment_gate": (
                "RESTORED"
                if acceptance_gain > 0.0 and rmse_reduction > 0.0
                else "NOT_RESTORED"
            ),
        }
    return result


def _blind_controls(
    records: list[Any],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    bootstrap_repeats: int,
    detector_seeds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    timestamp_values: list[float] = []
    for record in records:
        try:
            timestamp_values.append(
                datetime.fromisoformat(
                    record.source_timestamp.replace("Z", "+00:00")
                ).timestamp()
            )
        except (TypeError, ValueError):
            timestamp_values.append(0.0)
    timestamp = np.asarray(timestamp_values)
    features = np.column_stack(
        [
            np.log1p([record.source_char_count for record in records]),
            timestamp,
            [record.source_revision for record in records],
            [len(record.topic or "") for record in records],
        ]
    ).astype(np.float32)
    seeds = tuple(seed + offset for offset in range(detector_seeds))
    learned = triplet_ensemble_scores(features, labels, calibration, test, seeds)
    hash_uniform = np.asarray(
        [
            int(hashlib.sha256(record.source.encode()).hexdigest()[:12], 16)
            for record in records
        ],
        dtype=np.float64,
    )[test]
    return {
        "source_metadata_triplet": metric_row(
            labels[test], learned, bootstrap_repeats, seed + 100
        ),
        "record_hash_uniform": metric_row(
            labels[test], hash_uniform, bootstrap_repeats, seed + 101
        ),
    }


def _metric_table(result: dict[str, Any]) -> list[str]:
    wanted = [
        "draft_nart_stat_last_triplet",
        "draft_selected_activation_triplet",
        "transcript_only_triplet",
        "naive_activation_transcript_concat_triplet",
        "draver_act_residual_triplet",
        "control_qbin_shuffled_draver_act_triplet",
        "draft_min_k_logp",
        "verifier_mean_acceptance",
    ]
    lines = [
        "| Signal | AUC (95% bootstrap CI) | TPR@5%FPR |",
        "|---|---:|---:|",
    ]
    for name in wanted:
        row = result["metrics"][name]
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, "
            f"{row['auc_ci95_high']:.3f}] | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    return lines


def _trace(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{values[0]:.4f} → {values[-1]:.4f}"


def render_markdown(artifact: dict[str, Any]) -> str:
    cfg = artifact["config"]
    lines = [
        "# Deployment-aligned public controlled-SFT DraVer-Act validation",
        "",
        "## Material passport",
        "",
        f"- Experiment ID: `{artifact['material_passport']['experiment_id']}`",
        "- Status: COMPLETED",
        f"- Target: `{cfg['target_model']}@{cfg['target_revision']}`",
        f"- Draft: `{cfg['draft_model']}@{cfg['draft_revision']}`",
        f"- Dataset snapshot SHA-256: `{artifact['data']['snapshot_sha256']}`",
        f"- Target adaptation: member-only LoRA SFT, {cfg['target_epochs']} epochs",
        f"- Draft adaptation matrix: `{cfg['draft_adaptation']}`",
        f"- Calibration/test: {cfg['audit_train_per_class']}/{cfg['audit_test_per_class']} per class",
        f"- Device: {artifact['runtime']['device']}; peak {artifact['runtime']['peak_gpu_memory_gib']:.2f} GiB",
        f"- Duration: {artifact['runtime']['duration_seconds']:.1f} seconds",
        "",
        "## Deployment-alignment diagnostics",
        "",
        "`Exact acceptance` is the teacher-forced mean of $\\sum_v \\min(p(v), q(v))=1-TV(p,q)$ on response positions; it is the expected one-token acceptance under ordinary speculative sampling.",
        "",
        "| Pair endpoint | Exact acceptance | Member | Nonmember | Top-1 agreement | Candidate-logp RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in artifact["pair_alignment"].items():
        lines.append(
            f"| `{name}` | {row['exact_acceptance_overall']:.3f} | "
            f"{row['exact_acceptance_member']:.3f} | "
            f"{row['exact_acceptance_nonmember']:.3f} | "
            f"{row['top1_agreement_overall']:.3f} | "
            f"{row['candidate_logp_rmse_overall']:.3f} |"
        )
    if artifact["alignment_recovery"]:
        lines.extend(
            [
                "",
                "| Aligned endpoint | Acceptance gain vs target-only | Top-1 gain | Logp-RMSE reduction | Gate |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for name, row in artifact["alignment_recovery"].items():
            lines.append(
                f"| `{name}` | {row['exact_acceptance_gain_vs_target_only']:+.3f} | "
                f"{row['top1_agreement_gain_vs_target_only']:+.3f} | "
                f"{row['candidate_logp_rmse_reduction_vs_target_only']:+.3f} | "
                f"{row['alignment_gate']} |"
            )
    lines.extend(
        [
            "",
            "## Training trace",
            "",
            f"- Target SFT loss: {_trace(artifact['training']['target_sft_loss'])}",
            f"- Auxiliary-only target→draft distillation loss: {_trace(artifact['training']['aux_distill_loss'])}",
            f"- Same-member draft SFT loss: {_trace(artifact['training']['member_draft_sft_loss'])}",
            "",
            "## Leakage and distribution controls",
            "",
            f"- Member, nonmember, and auxiliary records are document-disjoint random partitions of one `{artifact['data']['dataset']}` snapshot.",
            "- The auxiliary-distilled draft never reads member records; the same-member SFT draft is reported separately as a deployment boundary condition.",
            "- Representation-training, support, and final test subsets are disjoint.",
            "- The base/base endpoint is a negative control; target-only is retained as a deployment-mismatch control.",
            "- The q-bin shuffle preserves draft difficulty strata while breaking tokenwise verifier–activation alignment.",
        ]
    )
    for name, endpoint in artifact["endpoints"].items():
        lines.extend(["", f"## Endpoint: `{name}`", ""])
        lines.extend(_metric_table(endpoint))
        lines.extend(
            [
                "",
                "### Paired AUC deltas for DraVer-Act residual",
                "",
                "| Comparison | Delta AUC (95% paired bootstrap CI) |",
                "|---|---:|",
            ]
        )
        for comparison, row in endpoint["paired_auc_deltas"].items():
            lines.append(
                f"| `{comparison}` | {row['delta_auc']:+.3f} "
                f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] |"
            )
    lines.extend(
        [
            "",
            "## Blind source controls",
            "",
            "| Control | AUC (95% bootstrap CI) |",
            "|---|---:|",
        ]
    )
    for name, row in artifact["blind_controls"].items():
        lines.append(
            f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, "
            f"{row['auc_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This validates controlled fine-tuning membership at a fixed-candidate L3 edge–cloud SD interface.",
            "The exact-acceptance table diagnoses whether each draft/target pair remains useful for ordinary SD; it is separate from the fixed-candidate membership signal.",
            "The auxiliary-only condition isolates target-side membership better than same-member SFT, because its draft has no direct access to member documents.",
            "Base-pretraining overlap is balanced by random same-pool assignment rather than ruled out item by item.",
            "Results do not establish pretraining membership or passive-production transcript risk.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with approved host GPU access")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if args.pair_diagnostic_batch_size < 1:
        raise ValueError("pair-diagnostic-batch-size must be positive")
    if args.draft_epochs is None:
        args.draft_epochs = args.target_epochs
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    if args.audit_train_per_class >= args.n_per_class:
        raise ValueError("audit-train-per-class must be smaller than n-per-class")
    run_aux_distill = args.draft_adaptation in {"aux_distill", "both"}
    run_member_sft = args.draft_adaptation in {"member_sft", "both"}
    if run_aux_distill and args.n_aux < 1:
        raise ValueError("aux_distill requires at least one auxiliary record")
    if run_aux_distill and args.distill_steps < 1:
        raise ValueError("aux_distill requires distill-steps to be positive")
    if run_aux_distill and args.distill_temperature <= 0:
        raise ValueError("distill-temperature must be positive")
    if run_member_sft and args.draft_epochs < 1:
        raise ValueError("member_sft requires draft-epochs to be positive")

    root = Path(__file__).resolve().parents[2]
    dataset_path = _resolve(root, args.dataset)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)

    tokenizer = _load_local_tokenizer(args.draft_model, args.draft_revision)
    target_tokenizer = _load_local_tokenizer(
        args.target_model, args.target_revision
    )
    tokenizer_compatibility = _check_tokenizers(tokenizer, target_tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    split_seed = args.seed if args.split_seed is None else args.split_seed
    members, nonmembers, auxiliary, data_metadata = build_public_snapshot_split(
        dataset_path,
        tokenizer,
        args.response_tokens,
        args.n_per_class,
        args.n_aux,
        split_seed,
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
        split_seed + 30,
    )
    started = time.time()

    target_base = load_causal_lm(
        args.target_model,
        device,
        revision=args.target_revision,
        local_files_only=True,
    )
    draft_base = load_causal_lm(
        args.draft_model,
        device,
        revision=args.draft_revision,
        local_files_only=True,
    )
    draft_outputs: dict[str, dict[str, np.ndarray]] = {
        "base": extract_draft_activation_outputs(
            draft_base, candidates, tokenizer, device, args.draft_batch_size
        )
    }
    target_epoch0 = extract_target_token_outputs(
        target_base, candidates, tokenizer, device, args.target_batch_size
    )
    pair_outputs: dict[str, dict[str, np.ndarray]] = {
        "base_pair_unadapted": extract_pair_alignment_outputs(
            draft_base,
            target_base,
            candidates,
            tokenizer,
            device,
            args.pair_diagnostic_batch_size,
        )
    }

    target = add_lora(
        target_base, args.lora_r, args.lora_alpha, args.lora_dropout
    )
    target_sft_loss = sft_train(
        target,
        members,
        tokenizer,
        device,
        args.target_epochs,
        args.target_batch_size,
        args.target_grad_accum,
        args.target_lr,
        args.seed + 10,
        "target member-only SFT",
    )
    target_adapted = extract_target_token_outputs(
        target, candidates, tokenizer, device, args.target_batch_size
    )
    pair_outputs["target_only_mismatch"] = extract_pair_alignment_outputs(
        draft_base,
        target,
        candidates,
        tokenizer,
        device,
        args.pair_diagnostic_batch_size,
    )
    save_adapter(target, output_dir / "adapter_target")
    del draft_base
    gc.collect()
    torch.cuda.empty_cache()

    aux_distill_loss: list[float] = []
    member_draft_sft_loss: list[float] = []
    adapter_paths: dict[str, str] = {"target": "adapter_target"}
    aligned_endpoint_names: list[str] = []

    if run_aux_distill:
        aux_draft = add_lora(
            load_causal_lm(
                args.draft_model,
                device,
                revision=args.draft_revision,
                local_files_only=True,
            ),
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        aux_distill_loss = distill_on_auxiliary(
            aux_draft,
            target,
            auxiliary,
            tokenizer,
            device,
            args.distill_steps,
            args.draft_batch_size,
            args.draft_lr,
            args.distill_temperature,
            args.seed + 20,
        )
        draft_outputs["aux_distilled"] = extract_draft_activation_outputs(
            aux_draft, candidates, tokenizer, device, args.draft_batch_size
        )
        endpoint_name = "deployment_aligned_aux_distilled"
        pair_outputs[endpoint_name] = extract_pair_alignment_outputs(
            aux_draft,
            target,
            candidates,
            tokenizer,
            device,
            args.pair_diagnostic_batch_size,
        )
        save_adapter(aux_draft, output_dir / "adapter_draft_aux_distilled")
        adapter_paths["draft_aux_distilled"] = "adapter_draft_aux_distilled"
        aligned_endpoint_names.append(endpoint_name)
        del aux_draft
        gc.collect()
        torch.cuda.empty_cache()

    if run_member_sft:
        member_draft = add_lora(
            load_causal_lm(
                args.draft_model,
                device,
                revision=args.draft_revision,
                local_files_only=True,
            ),
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )
        member_draft_sft_loss = sft_train(
            member_draft,
            members,
            tokenizer,
            device,
            args.draft_epochs,
            args.draft_batch_size,
            args.draft_grad_accum,
            args.draft_lr,
            args.seed + 21,
            "same-member draft SFT",
        )
        draft_outputs["member_sft"] = extract_draft_activation_outputs(
            member_draft, candidates, tokenizer, device, args.draft_batch_size
        )
        endpoint_name = "deployment_aligned_member_sft"
        pair_outputs[endpoint_name] = extract_pair_alignment_outputs(
            member_draft,
            target,
            candidates,
            tokenizer,
            device,
            args.pair_diagnostic_batch_size,
        )
        save_adapter(member_draft, output_dir / "adapter_draft_member_sft")
        adapter_paths["draft_member_sft"] = "adapter_draft_member_sft"
        aligned_endpoint_names.append(endpoint_name)
        del member_draft
        gc.collect()
        torch.cuda.empty_cache()

    del target, target_base
    gc.collect()
    torch.cuda.empty_cache()

    endpoint_specs: list[
        tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = [
        ("base_pair_unadapted", draft_outputs["base"], target_epoch0),
        ("target_only_mismatch", draft_outputs["base"], target_adapted),
    ]
    if run_aux_distill:
        endpoint_specs.append(
            (
                "deployment_aligned_aux_distilled",
                draft_outputs["aux_distilled"],
                target_adapted,
            )
        )
    if run_member_sft:
        endpoint_specs.append(
            (
                "deployment_aligned_member_sft",
                draft_outputs["member_sft"],
                target_adapted,
            )
        )

    endpoints: dict[str, Any] = {}
    target_diagnostics: dict[str, Any] = {}
    for offset, (name, current_draft, current_target) in enumerate(endpoint_specs):
        endpoints[name] = evaluate_activation_audit(
            current_draft,
            current_target,
            labels,
            calibration,
            test,
            args.min_k_fraction,
            args.transcript_repeats,
            args.bootstrap_repeats,
            args.detector_seeds,
            args.seed + 10000 + offset * 1000,
        )
        target_diagnostics[name] = _endpoint_diagnostics(
            current_target, labels, test
        )

    pair_alignment = {
        name: _pair_alignment_diagnostics(outputs, labels, test)
        for name, outputs in pair_outputs.items()
    }
    primary_deployment_endpoint = (
        "deployment_aligned_aux_distilled"
        if run_aux_distill
        else (
            "deployment_aligned_member_sft"
            if run_member_sft
            else "target_only_mismatch"
        )
    )
    config = {
        **vars(args),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "audit_test_per_class": args.n_per_class - args.audit_train_per_class,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "primary_deployment_endpoint": primary_deployment_endpoint,
        "resolved_split_seed": split_seed,
    }
    artifact = {
        "material_passport": {
            "experiment_id": (
                f"public-draver-aligned-{args.draft_model.split('/')[-1]}-to-"
                f"{args.target_model.split('/')[-1]}-{args.seed}"
            ),
            "status": "COMPLETED",
            "verification_status": "ANALYZED_PUBLIC_DEPLOYMENT_ALIGNED_SFT",
        },
        "config": config,
        "tokenizer_compatibility": tokenizer_compatibility,
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": {
            "target_sft_loss": target_sft_loss,
            "aux_distill_loss": aux_distill_loss,
            "member_draft_sft_loss": member_draft_sft_loss,
        },
        "adapter_paths": adapter_paths,
        "blind_controls": _blind_controls(
            candidates,
            labels,
            calibration,
            test,
            args.bootstrap_repeats,
            args.detector_seeds,
            args.seed + 20000,
        ),
        "target_diagnostics": target_diagnostics,
        "pair_alignment": pair_alignment,
        "alignment_recovery": _alignment_recovery(
            pair_alignment, aligned_endpoint_names
        ),
        "endpoints": endpoints,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
            / (1024**3),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        render_markdown(artifact), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runtime": artifact["runtime"],
                "training": artifact["training"],
                "alignment_recovery": artifact["alignment_recovery"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
