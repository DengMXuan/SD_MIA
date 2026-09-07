"""Draft–verifier acceptance comparison before and after target fine-tuning.

For one NART-track run directory this script replays the run's data split
(same pool, tokenizer, sizes, and data seed), samples a fixed number of
records per class, and measures the exact speculative-decoding acceptance
rate ``sum_v min(p(v), q(v)) = 1 - TV(p, q)`` (plus greedy top-1 agreement)
for every (draft variant, verifier) pair:

- drafts: base draft, auxiliary-distilled draft, member-SFT draft;
- verifiers: the pre-fine-tuning base target and the run's fine-tuned target.

``extract_pair_alignment_outputs`` teacher-forces the response (document)
tokens, so every pair is scored on identical positions. Per-record values
are persisted in ``acceptance_comparison.npz``; means with bootstrap CIs and
paired tuned-minus-base deltas are written to ``acceptance_comparison.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .draver_activation import extract_pair_alignment_outputs
from .generalization import (
    load_draft_model,
    load_finetuned_model,
    load_run_config,
)
from .nart_data import build_nart_split, pool_path as nart_pool_path
from .training import load_causal_lm, set_seed

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BATCH = {"wikitection": 8, "newstection": 8, "arxivtection": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def sample_records(
    classes: dict[str, list[Any]], per_class: int, seed: int
) -> tuple[list[Any], np.ndarray]:
    """Seeded per-class subsample; returns records and class indices 0..k-1."""
    rng = np.random.default_rng(seed)
    records: list[Any] = []
    labels: list[int] = []
    for class_index, (name, pool_records) in enumerate(classes.items()):
        take = rng.permutation(len(pool_records))[:per_class]
        for index in take:
            records.append(pool_records[int(index)])
            labels.append(class_index)
        print(f"class {name}: sampled {len(take)} of {len(pool_records)}", flush=True)
    return records, np.asarray(labels, dtype=np.int64)


def bootstrap_mean_ci(
    values: np.ndarray, repeats: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "count": int(len(values)),
    }


def summarize_pair(
    values: dict[str, np.ndarray], labels: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("exact_acceptance", "top1_agreement"):
        rows = {"overall": bootstrap_mean_ci(values[metric], 500, 7)}
        for class_index, name in enumerate(class_names):
            mask = labels == class_index
            rows[name] = bootstrap_mean_ci(values[metric][mask], 500, 8 + class_index)
        summary[metric] = rows
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    cfg = load_run_config(run_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run with the approved host GPU access")

    seed = args.seed if args.seed is not None else cfg.data_seed + 13
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pool = (
        args.pool_path
        if args.pool_path is not None
        else (
            cfg.pool_path
            if cfg.pool_path is not None
            else nart_pool_path(cfg.benchmark)
        )
    )
    if not pool.is_absolute():
        pool = ROOT / pool
    members, nonmembers, auxiliary, _metadata = build_nart_split(
        cfg.benchmark,
        pool,
        tokenizer,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    classes = {"member": members, "nonmember": nonmembers, "auxiliary": auxiliary}
    class_names = list(classes)
    records, labels = sample_records(classes, args.per_class, seed)
    batch_size = args.batch_size or DEFAULT_BATCH[cfg.benchmark]
    print(
        f"evaluating {len(records)} records x 6 pairs, batch {batch_size}",
        flush=True,
    )

    verifiers = {
        "base_target": load_causal_lm(cfg.target_model, device),
        "tuned_target": load_finetuned_model(run_dir, cfg.target_model, device),
    }
    pair_values: dict[str, dict[str, np.ndarray]] = {}
    for draft_name, checkpoint in (
        ("base_draft", None),
        ("aux_distilled_draft", "draft_auxiliary_distilled"),
        ("member_sft_draft", "draft_member_sft"),
    ):
        if checkpoint is None:
            draft = load_causal_lm(cfg.draft_model, device)
        else:
            draft = load_draft_model(run_dir, cfg.draft_model, checkpoint, device)
        for verifier_name, verifier in verifiers.items():
            key = f"{draft_name}__vs__{verifier_name}"
            started = time.time()
            pair_values[key] = extract_pair_alignment_outputs(
                draft, verifier, records, tokenizer, device, batch_size
            )
            print(
                f"{key}: exact_acceptance={pair_values[key]['exact_acceptance'].mean():.4f} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
        del draft
        gc.collect()
        torch.cuda.empty_cache()
    for verifier in verifiers.values():
        del verifier
    gc.collect()
    torch.cuda.empty_cache()

    results: dict[str, Any] = {}
    for draft_name in ("base_draft", "aux_distilled_draft", "member_sft_draft"):
        base = pair_values[f"{draft_name}__vs__base_target"]
        tuned = pair_values[f"{draft_name}__vs__tuned_target"]
        rows = {
            "vs_base_target": summarize_pair(base, labels, class_names),
            "vs_tuned_target": summarize_pair(tuned, labels, class_names),
        }
        deltas: dict[str, Any] = {}
        for metric in ("exact_acceptance", "top1_agreement"):
            delta_rows = {"overall": bootstrap_mean_ci(
                tuned[metric] - base[metric], args.bootstrap_repeats, 9
            )}
            for class_index, name in enumerate(class_names):
                mask = labels == class_index
                delta_rows[name] = bootstrap_mean_ci(
                    tuned[metric][mask] - base[metric][mask],
                    args.bootstrap_repeats,
                    10 + class_index,
                )
            deltas[metric] = delta_rows
        rows["delta_tuned_minus_base"] = deltas
        results[draft_name] = rows

    artifact = {
        "run_dir": str(run_dir),
        "benchmark": cfg.benchmark,
        "target_model": cfg.target_model,
        "draft_model": cfg.draft_model,
        "target_epochs": cfg.target_epochs,
        "protocol": {
            "per_class": args.per_class,
            "records": len(records),
            "batch_size": batch_size,
            "seed": seed,
            "data_seed": cfg.data_seed,
            "pool_path": str(pool),
            "positions": "teacher-forced response (document) tokens incl. EOS",
            "acceptance_definition": "sum_v min(p(v), q(v)) = 1 - TV(p, q)",
        },
        "results": results,
    }
    (run_dir / "acceptance_comparison.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "acceptance_comparison.npz",
        labels=labels,
        class_names=np.asarray(class_names),
        **{
            f"{key}__{metric}": values[metric]
            for key, values in pair_values.items()
            for metric in ("exact_acceptance", "top1_agreement")
        },
    )
    print(json.dumps({
        "run_dir": str(run_dir),
        "headline": {
            draft: {
                "base": rows["vs_base_target"]["exact_acceptance"]["overall"]["mean"],
                "tuned": rows["vs_tuned_target"]["exact_acceptance"]["overall"]["mean"],
                "delta": rows["delta_tuned_minus_base"]["exact_acceptance"]["overall"]["mean"],
            }
            for draft, rows in results.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
