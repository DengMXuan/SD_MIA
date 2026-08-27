from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .data import build_controlled_split
from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    make_audit_split,
)
from .training import load_causal_lm, set_seed


VARIANT_ADAPTERS = {
    "base": None,
    "auxiliary_distilled": "draft_auxiliary_distilled",
    "member_sft": "draft_member_sft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate draft-verifier conditioned all-layer activation features."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path(
            "experiments/results/qwen3_sft/qwen3_1p7b_to_8b_epoch1/results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/qwen3_sft/p2_draver_act_epoch1"),
    )
    parser.add_argument(
        "--draft-variant",
        action="append",
        choices=sorted(VARIANT_ADAPTERS),
        help="Repeat for multiple variants; default: base and auxiliary_distilled.",
    )
    parser.add_argument("--detector-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _validate_reconstructed_records(
    source: dict[str, Any], members: list[Any], nonmembers: list[Any]
) -> None:
    for key, records in (("members", members), ("nonmembers", nonmembers)):
        expected = [row["response_hash"] for row in source["records"][key]]
        actual = [record.response_hash for record in records]
        if actual != expected:
            raise RuntimeError(
                f"Reconstructed {key} do not match the adapter's source experiment"
            )


def _render_variant(name: str, result: dict[str, Any]) -> list[str]:
    lines = [
        f"## Draft variant: `{name}`",
        "",
        "| Signal | AUC (95% bootstrap CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for metric_name, row in sorted(result["metrics"].items()):
        lines.append(
            f"| `{metric_name}` | {row['auc']:.3f} "
            f"[{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] | "
            f"{row['tpr_at_1pct_fpr']:.3f} | {row['tpr_at_5pct_fpr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Paired AUC deltas for DraVer-Act residual",
            "",
            "| Comparison | delta AUC (95% paired bootstrap CI) |",
            "|---|---:|",
        ]
    )
    for comparison, row in sorted(result["paired_auc_deltas"].items()):
        lines.append(
            f"| `{comparison}` | {row['delta_auc']:+.3f} "
            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "### Detector-seed stability",
            "",
            "| Learned signal | Mean single-seed AUC | SD | Range |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric_name, row in sorted(result["detector_stability"].items()):
        lines.append(
            f"| `{metric_name}` | {row['auc_mean']:.3f} | {row['auc_std']:.3f} | "
            f"{row['auc_min']:.3f}–{row['auc_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Explicit few-shot calibration",
            "",
            "| Labeled calibration per class | Signal | AUC | TPR@5%FPR |",
            "|---:|---|---:|---:|",
        ]
    )
    for size, rows in result["few_shot_per_class"].items():
        for metric_name, row in sorted(rows.items()):
            lines.append(
                f"| {size} | `{metric_name}` | {row['auc']:.3f} | "
                f"{row['tpr_at_5pct_fpr']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"- Selected tokens/record: {result['protocol']['selected_tokens_per_record']}",
            f"- Transcript bits/record: {result['protocol']['bits_per_record']}",
            f"- Draft transformer layers: {result['feature_schema']['layers']}",
            "",
        ]
    )
    return lines


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# P2: Draft–Verifier Conditioned Activation Audit",
        "",
        "## Material Passport",
        "",
        f"- Experiment ID: `{artifact['material_passport']['experiment_id']}`",
        "- Status: COMPLETED",
        "- Verification status: controlled adapter-SFT pilot",
        f"- Source experiment: `{artifact['source_experiment']}`",
        f"- Device: {artifact['runtime']['device']}",
        f"- Peak allocated GPU memory: {artifact['runtime']['peak_gpu_memory_gib']:.2f} GiB",
        f"- Duration: {artifact['runtime']['duration_seconds']:.1f} seconds",
        "",
        "## Leakage controls",
        "",
        "- Target adapter and candidate hashes are inherited from, and checked against, the source experiment.",
        "- Activation normalization is fit only on the representation-training subset.",
        "- Triplet representation samples and labeled support samples are disjoint; final test labels are never used for fitting.",
        "- Verifier nuisance residualization is cross-fitted on calibration data without membership labels.",
        "- The q-bin shuffled control preserves draft-probability strata but breaks verifier/activation alignment.",
        "",
    ]
    for name, result in artifact["variants"].items():
        lines.extend(_render_variant(name, result))
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This reuses a controlled Qwen3 adapter-SFT target and a local semantic implementation of stochastic speculative verification.",
            "It evaluates an L3 fixed-candidate interface and does not establish passive-L2 or pretraining-membership risk.",
            "DraVer-Act is supported only if it improves over draft-only, transcript-only, naive concatenation, and the q-bin shuffled control with uncertainty bounds excluding zero.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with approved host GPU access")
    if args.detector_seeds < 1:
        raise ValueError("detector-seeds must be positive")

    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source["config"]
    seed = int(config["seed"])
    data_seed = int(config.get("data_seed", seed))
    audit_seed = int(config.get("audit_seed", seed))
    bootstrap_repeats = int(
        args.bootstrap_repeats or config.get("bootstrap_repeats", 500)
    )
    variants = args.draft_variant or ["base", "auxiliary_distilled"]

    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config["draft_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    members, nonmembers, _, data_metadata = build_controlled_split(
        root,
        tokenizer,
        int(config["response_tokens"]),
        int(config["n_per_class"]),
        int(config["n_aux"]),
        data_seed,
    )
    _validate_reconstructed_records(source, members, nonmembers)
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
        int(config["audit_train_per_class"]),
        audit_seed + 30,
    )

    started = time.time()
    adapter_root = source_path.parent / "adapters"
    target = PeftModel.from_pretrained(
        load_causal_lm(config["target_model"], device),
        adapter_root / "target",
        is_trainable=False,
    )
    target_outputs = extract_target_token_outputs(
        target,
        candidates,
        tokenizer,
        device,
        int(config["target_batch_size"]),
    )
    del target
    gc.collect()
    torch.cuda.empty_cache()

    variant_results: dict[str, Any] = {}
    for offset, variant in enumerate(variants):
        draft = load_causal_lm(config["draft_model"], device)
        adapter_name = VARIANT_ADAPTERS[variant]
        if adapter_name is not None:
            adapter_path = adapter_root / adapter_name
            if not (adapter_path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"Missing adapter weights: {adapter_path}")
            draft = PeftModel.from_pretrained(
                draft, adapter_path, is_trainable=False
            )
        draft_outputs = extract_draft_activation_outputs(
            draft,
            candidates,
            tokenizer,
            device,
            int(config["draft_batch_size"]),
        )
        del draft
        gc.collect()
        torch.cuda.empty_cache()
        variant_results[variant] = evaluate_activation_audit(
            draft_outputs,
            target_outputs,
            labels,
            calibration,
            test,
            float(config["min_k_fraction"]),
            int(config["transcript_repeats"]),
            bootstrap_repeats,
            args.detector_seeds,
            seed + 10000 + offset * 1000,
        )
        del draft_outputs
        gc.collect()

    artifact = {
        "material_passport": {
            "experiment_id": f"p2-draver-act-{seed}-epoch{config['target_epochs']}",
            "status": "COMPLETED",
            "verification_status": "ANALYZED_CONTROLLED_ADAPTER_SFT_PILOT",
        },
        "source_experiment": str(source_path.relative_to(root)),
        "config": {
            "gpu": args.gpu,
            "target_model": config["target_model"],
            "draft_model": config["draft_model"],
            "target_epochs": config["target_epochs"],
            "n_per_class": config["n_per_class"],
            "audit_train_per_class": config["audit_train_per_class"],
            "audit_test_per_class": int(config["n_per_class"])
            - int(config["audit_train_per_class"]),
            "min_k_fraction": config["min_k_fraction"],
            "transcript_repeats": config["transcript_repeats"],
            "bootstrap_repeats": bootstrap_repeats,
            "detector_seeds": args.detector_seeds,
            "draft_variants": variants,
        },
        "data": {
            "source_pdf_count": data_metadata["source_pdf_count"],
            "unique_chunk_count": data_metadata["unique_chunk_count"],
            "record_hashes_checked": True,
            "raw_activations_persisted": False,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "variants": variant_results,
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
                "variants": list(variant_results),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
