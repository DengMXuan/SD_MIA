from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from .audit import bottom_k_indices
from .draver_activation import (
    evaluate_activation_audit,
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    gather_tokens,
    make_audit_split,
)
from .public_data import build_public_snapshot_split
from .training import load_causal_lm, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested verifier-feedback budget sweep for a public benchmark adapter."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 4, 24])
    parser.add_argument(
        "--draft-endpoint",
        choices=("auto", "base", "aux_distilled", "member_sft"),
        default="auto",
        help="Draft checkpoint to pair with the adapted target.",
    )
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


def _validate_records(source: dict[str, Any], members: list[Any], nonmembers: list[Any]) -> None:
    for name, records in (("members", members), ("nonmembers", nonmembers)):
        expected = [row["response_hash"] for row in source["records"][name]]
        actual = [record.response_hash for record in records]
        if expected != actual:
            raise RuntimeError(f"Reconstructed {name} do not match source adapter records")


def _select_draft_endpoint(
    source: dict[str, Any], source_path: Path, requested: str
) -> tuple[str, Path | None]:
    if requested == "auto":
        primary = source.get("config", {}).get("primary_deployment_endpoint", "")
        if primary == "deployment_aligned_aux_distilled":
            requested = "aux_distilled"
        elif primary == "deployment_aligned_member_sft":
            requested = "member_sft"
        else:
            requested = "base"
    if requested == "base":
        return requested, None
    key = f"draft_{requested}"
    configured = source.get("adapter_paths", {}).get(key)
    adapter_path = (
        source_path.parent / configured
        if configured
        else source_path.parent / f"adapter_draft_{requested}"
    )
    if not adapter_path.is_dir():
        raise FileNotFoundError(
            f"Requested draft endpoint {requested!r} has no adapter at {adapter_path}"
        )
    return requested, adapter_path


def _nested_acceptance(
    target_logp: np.ndarray,
    draft_logp: np.ndarray,
    selected: np.ndarray,
    budgets: list[int],
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    target = gather_tokens(target_logp, selected).astype(np.float64)
    draft = gather_tokens(draft_logp, selected).astype(np.float64)
    if not np.isfinite(target).all() or not np.isfinite(draft).all():
        target_bad = int((~np.isfinite(target)).sum())
        draft_bad = int((~np.isfinite(draft)).sum())
        raise ValueError(
            "non-finite selected token logp: "
            f"target={target_bad}/{target.size}, draft={draft_bad}/{draft.size}"
        )
    alpha = np.minimum(1.0, np.exp(np.clip(target - draft, -50.0, 50.0)))
    maximum = max(budgets)
    uniforms = np.random.default_rng(seed).random((*alpha.shape, maximum))
    bits = uniforms < alpha[:, :, None]
    observed = {
        repeats: ((bits[:, :, :repeats].sum(axis=-1) + 0.5) / (repeats + 1.0)).astype(
            np.float32
        )
        for repeats in budgets
    }
    return observed, alpha.astype(np.float32)


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Nested verifier-feedback budget sweep",
        "",
        f"- Source: `{artifact['source_experiment']}`",
        f"- Model pair: `{artifact['config']['draft_model']}` → `{artifact['config']['target_model']}`",
        "- Each larger budget reuses the exact prefix of the same per-token Bernoulli transcript.",
        f"- Draft endpoint: `{artifact['config']['draft_endpoint']}`.",
        "- No model training is performed in this sweep; source target/draft adapters are reused.",
        "",
        "| Bits/record | Repeats/token | DraVer-Act AUC | Transcript-only AUC | q-bin shuffle AUC | Delta vs transcript (95% CI) | Delta vs shuffle (95% CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for repeats_text, result in artifact["budgets"].items():
        repeats = int(repeats_text)
        metrics = result["metrics"]
        deltas = result["paired_auc_deltas"]
        transcript_delta = deltas["draver_act_residual_minus_transcript_only_triplet"]
        shuffle_delta = deltas[
            "draver_act_residual_minus_control_qbin_shuffled_draver_act_triplet"
        ]
        lines.append(
            f"| {result['protocol']['bits_per_record']} | {repeats} | "
            f"{metrics['draver_act_residual_triplet']['auc']:.3f} | "
            f"{metrics['transcript_only_triplet']['auc']:.3f} | "
            f"{metrics['control_qbin_shuffled_draver_act_triplet']['auc']:.3f} | "
            f"{transcript_delta['delta_auc']:+.3f} "
            f"[{transcript_delta['ci95_low']:+.3f}, {transcript_delta['ci95_high']:+.3f}] | "
            f"{shuffle_delta['delta_auc']:+.3f} "
            f"[{shuffle_delta['ci95_low']:+.3f}, {shuffle_delta['ci95_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "Interpretation is limited to the fixed-candidate L3 simulator and one target-training seed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(min(4, args.cpu_threads))
    budgets = sorted(set(args.repeats))
    if not budgets or budgets[0] < 1:
        raise ValueError("repeats must be positive")
    root = Path(__file__).resolve().parents[2]
    source_path = _resolve(root, args.source_results)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source["config"]
    draft_endpoint, draft_adapter_path = _select_draft_endpoint(
        source, source_path, args.draft_endpoint
    )
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(config["seed"])
    set_seed(seed)

    tokenizer_snapshot = snapshot_download(
        repo_id=config["draft_model"],
        revision=config["draft_revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset_path = _resolve(root, Path(config["dataset"]))
    members, nonmembers, _, data_metadata = build_public_snapshot_split(
        dataset_path,
        tokenizer,
        int(config["response_tokens"]),
        int(config["n_per_class"]),
        int(config["n_aux"]),
        seed,
    )
    _validate_records(source, members, nonmembers)
    candidates = members + nonmembers
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    calibration, test = make_audit_split(
        len(members),
        len(nonmembers),
        int(config["audit_train_per_class"]),
        seed + 30,
    )
    started = time.time()

    draft_base = load_causal_lm(
        config["draft_model"],
        device,
        revision=config["draft_revision"],
        local_files_only=True,
    )
    draft = (
        PeftModel.from_pretrained(
            draft_base,
            draft_adapter_path,
            is_trainable=False,
        )
        if draft_adapter_path is not None
        else draft_base
    )
    draft_outputs = extract_draft_activation_outputs(
        draft, candidates, tokenizer, device, int(config["draft_batch_size"])
    )
    del draft, draft_base
    gc.collect()
    torch.cuda.empty_cache()

    target = PeftModel.from_pretrained(
        load_causal_lm(
            config["target_model"],
            device,
            revision=config["target_revision"],
            local_files_only=True,
        ),
        source_path.parent / "adapter_target",
        is_trainable=False,
    )
    target_outputs = extract_target_token_outputs(
        target, candidates, tokenizer, device, int(config["target_batch_size"])
    )
    del target
    gc.collect()
    torch.cuda.empty_cache()

    selected = bottom_k_indices(draft_outputs["token_logp"], float(config["min_k_fraction"]))
    observed, exact_alpha = _nested_acceptance(
        target_outputs["token_logp"],
        draft_outputs["token_logp"],
        selected,
        budgets,
        seed + 50000,
    )
    results: dict[str, Any] = {}
    for repeats in budgets:
        results[str(repeats)] = evaluate_activation_audit(
            draft_outputs,
            target_outputs,
            labels,
            calibration,
            test,
            float(config["min_k_fraction"]),
            repeats,
            args.bootstrap_repeats,
            args.detector_seeds,
            seed + 60000 + repeats * 1000,
            few_shot_per_class=(),
            acceptance_override=observed[repeats],
            exact_alpha_override=exact_alpha,
            metric_family_names=(
                "transcript_only_triplet",
                "draver_act_residual_triplet",
                "control_qbin_shuffled_draver_act_triplet",
            ),
            include_direct_scores=False,
            comparison_baselines=(
                "transcript_only_triplet",
                "control_qbin_shuffled_draver_act_triplet",
            ),
        )

    artifact = {
        "material_passport": {
            "status": "COMPLETED",
            "verification_status": "ANALYZED_NESTED_TRANSCRIPT_BUDGET_SWEEP",
        },
        "source_experiment": str(source_path.relative_to(root)),
        "config": {
            "target_model": config["target_model"],
            "target_revision": config["target_revision"],
            "draft_model": config["draft_model"],
            "draft_revision": config["draft_revision"],
            "draft_endpoint": draft_endpoint,
            "draft_adapter": (
                str(draft_adapter_path.relative_to(root))
                if draft_adapter_path is not None
                else None
            ),
            "budgets_repeats_per_token": budgets,
            "selected_tokens_per_record": int(selected.shape[1]),
            "bootstrap_repeats": args.bootstrap_repeats,
            "detector_seeds": args.detector_seeds,
            "cpu_threads": args.cpu_threads,
            "nested_transcript_seed": seed + 50000,
        },
        "data": data_metadata,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "duration_seconds": time.time() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        },
        "budgets": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(render_markdown(artifact), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "runtime": artifact["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
