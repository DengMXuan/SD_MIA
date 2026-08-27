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
from transformers import AutoTokenizer

from .audit import metric_row
from .data import records_metadata
from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    make_audit_split,
    triplet_ensemble_scores,
)
from .public_data import build_public_snapshot_split
from .training import add_lora, load_causal_lm, save_adapter, set_seed, sft_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NART-inspired public controlled-SFT validation for DraVer-Act."
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
    parser.add_argument("--response-tokens", type=int, default=128)
    parser.add_argument("--n-per-class", type=int, default=160)
    parser.add_argument("--n-aux", type=int, default=160)
    parser.add_argument("--audit-train-per-class", type=int, default=48)
    parser.add_argument("--target-epochs", type=int, default=3)
    parser.add_argument("--target-batch-size", type=int, default=4)
    parser.add_argument("--target-grad-accum", type=int, default=4)
    parser.add_argument("--target-lr", type=float, default=2e-4)
    parser.add_argument("--draft-batch-size", type=int, default=4)
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


def _mean_token_logp(values: np.ndarray) -> np.ndarray:
    return np.nanmean(values, axis=1)


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


def _blind_controls(
    records: list[Any],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    bootstrap_repeats: int,
    detector_seeds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    timestamp = np.asarray(
        [
            datetime.fromisoformat(
                record.source_timestamp.replace("Z", "+00:00")
            ).timestamp()
            for record in records
        ]
    )
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


def render_markdown(artifact: dict[str, Any]) -> str:
    cfg = artifact["config"]
    lines = [
        "# Public controlled-SFT DraVer-Act validation",
        "",
        "## Material passport",
        "",
        f"- Experiment ID: `{artifact['material_passport']['experiment_id']}`",
        "- Status: COMPLETED",
        f"- Target: `{cfg['target_model']}@{cfg['target_revision']}`",
        f"- Draft: `{cfg['draft_model']}@{cfg['draft_revision']}`",
        f"- Dataset snapshot SHA-256: `{artifact['data']['snapshot_sha256']}`",
        f"- Target adaptation: member-only LoRA SFT, {cfg['target_epochs']} epochs",
        f"- Calibration/test: {cfg['audit_train_per_class']}/{cfg['audit_test_per_class']} per class",
        f"- Device: {artifact['runtime']['device']}; peak {artifact['runtime']['peak_gpu_memory_gib']:.2f} GiB",
        f"- Duration: {artifact['runtime']['duration_seconds']:.1f} seconds",
        "",
        "## Leakage and distribution controls",
        "",
        "- Member, nonmember, and auxiliary records are document-disjoint random partitions of one post-release FineWeb crawl snapshot.",
        "- Only members update the cloud target; the on-device draft remains the untouched public checkpoint.",
        "- Representation-training, support, and final test subsets are disjoint.",
        "- The unadapted target endpoint is a negative control using the identical records and labels.",
        "- The q-bin shuffle preserves draft difficulty strata while breaking tokenwise verifier–activation alignment.",
        "",
        "## Endpoint 0: unadapted target negative control",
        "",
    ]
    lines.extend(_metric_table(artifact["endpoints"]["epoch_0_unadapted"]))
    lines.extend(
        [
            "",
            f"## Endpoint {cfg['target_epochs']}: member-only adapted target",
            "",
        ]
    )
    trained = artifact["endpoints"][f"epoch_{cfg['target_epochs']}_adapted"]
    lines.extend(_metric_table(trained))
    lines.extend(
        [
            "",
            "### Paired AUC deltas for the proposed residual feature",
            "",
            "| Comparison | Delta AUC (95% paired bootstrap CI) |",
            "|---|---:|",
        ]
    )
    for name, row in trained["paired_auc_deltas"].items():
        lines.append(
            f"| `{name}` | {row['delta_auc']:+.3f} [{row['ci95_low']:+.3f}, "
            f"{row['ci95_high']:+.3f}] |"
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
    diagnostics = artifact["endpoint_diagnostics"]
    adapted_name = f"epoch_{cfg['target_epochs']}_adapted"
    lines.extend(
        [
            "",
            "## Target likelihood diagnostic on the held-out audit test",
            "",
            f"- Unadapted member-minus-nonmember mean token logp: {diagnostics['epoch_0_unadapted']['member_minus_nonmember']:+.4f}",
            f"- Adapted member-minus-nonmember mean token logp: {diagnostics[adapted_name]['member_minus_nonmember']:+.4f}",
            f"- Target SFT loss: {artifact['training']['target_sft_loss'][0]:.4f} → {artifact['training']['target_sft_loss'][-1]:.4f}",
            "",
            "## Interpretation boundary",
            "",
        "This validates controlled fine-tuning membership at a fixed-candidate L3 edge–cloud SD interface.",
            "The FineWeb timestamp is a crawl time, not original publication time; base-pretraining overlap is therefore balanced by random same-pool assignment rather than ruled out item by item.",
            "It does not establish pretraining membership or passive-production transcript risk.",
            "A DraVer-Act advantage is claimed only against the separated draft-only, transcript-only, naive-concatenation, and q-bin-shuffled controls with paired uncertainty.",
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
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    if args.audit_train_per_class >= args.n_per_class:
        raise ValueError("audit-train-per-class must be smaller than n-per-class")
    root = Path(__file__).resolve().parents[2]
    dataset_path = _resolve(root, args.dataset)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model, revision=args.draft_revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    members, nonmembers, auxiliary, data_metadata = build_public_snapshot_split(
        dataset_path,
        tokenizer,
        args.response_tokens,
        args.n_per_class,
        args.n_aux,
        args.seed,
    )
    candidates = members + nonmembers
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    calibration, test = make_audit_split(
        len(members),
        len(nonmembers),
        args.audit_train_per_class,
        args.seed + 30,
    )
    started = time.time()

    draft = load_causal_lm(
        args.draft_model,
        device,
        revision=args.draft_revision,
        local_files_only=True,
    )
    draft_outputs = extract_draft_activation_outputs(
        draft, candidates, tokenizer, device, args.draft_batch_size
    )
    del draft
    gc.collect()
    torch.cuda.empty_cache()

    target_base = load_causal_lm(
        args.target_model,
        device,
        revision=args.target_revision,
        local_files_only=True,
    )
    target_epoch0 = extract_target_token_outputs(
        target_base, candidates, tokenizer, device, args.target_batch_size
    )
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
    save_adapter(target, output_dir / "adapter_target")
    del target, target_base
    gc.collect()
    torch.cuda.empty_cache()

    endpoint_outputs = {
        "epoch_0_unadapted": target_epoch0,
        f"epoch_{args.target_epochs}_adapted": target_adapted,
    }
    endpoints: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for offset, (name, outputs) in enumerate(endpoint_outputs.items()):
        endpoints[name] = evaluate_activation_audit(
            draft_outputs,
            outputs,
            labels,
            calibration,
            test,
            args.min_k_fraction,
            args.transcript_repeats,
            args.bootstrap_repeats,
            args.detector_seeds,
            args.seed + 10000 + offset * 1000,
        )
        diagnostics[name] = _endpoint_diagnostics(outputs, labels, test)

    config = {
        **vars(args),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "audit_test_per_class": args.n_per_class - args.audit_train_per_class,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    artifact = {
        "material_passport": {
            "experiment_id": (
                f"public-draver-{args.draft_model.split('/')[-1]}-to-"
                f"{args.target_model.split('/')[-1]}-{args.seed}"
            ),
            "status": "COMPLETED",
            "verification_status": "ANALYZED_PUBLIC_CONTROLLED_SFT",
        },
        "config": config,
        "data": data_metadata,
        "records": {
            "members": records_metadata(members),
            "nonmembers": records_metadata(nonmembers),
            "auxiliary": records_metadata(auxiliary),
        },
        "training": {"target_sft_loss": target_sft_loss},
        "blind_controls": _blind_controls(
            candidates,
            labels,
            calibration,
            test,
            args.bootstrap_repeats,
            args.detector_seeds,
            args.seed + 20000,
        ),
        "endpoint_diagnostics": diagnostics,
        "endpoints": endpoints,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
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
                "target_sft_loss": target_sft_loss,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
