"""Standalone baseline command: resolve inputs, execute and persist results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.shared.training.generalization import load_finetuned_model, load_run_config
from experiments.shared.training.training import set_seed
from experiments.baseline.runtime import RunProgress
from experiments.baseline.costs import CostMeter, cost_protocol, write_cost_report
from experiments.baseline import METHODS

from .engine import score_methods
from .scorer import TargetScorer
from .types import AuditRecord
from .data import _resolve, _target_tokenizer, load_audit_records
from .reporting import render_report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--pretraining-manifest", type=Path, help="frozen MIMIR evaluation manifest")
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--cost-warmup-records", type=int, default=1, help="untimed forward warmup records before each method")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="seconds between progress updates")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated method names, or 'all' (default: all)",
    )
    parser.add_argument("--k-percent", type=float, default=20.0)
    parser.add_argument("--recall-shots", type=int, default=4)
    parser.add_argument("--icp-top-k", type=int, default=5)
    parser.add_argument("--icp-aggregation", choices=("min", "mean", "max"), default="min")
    parser.add_argument("--sead-samples", type=int, default=50)
    parser.add_argument(
        "--sead-temperature",
        type=float,
        default=1.0,
        help="SEAD sampling temperature; 1.0 matches the official frequency estimator",
    )
    parser.add_argument("--samia-samples", type=int, default=10)
    parser.add_argument("--prefix-ratio", type=float, default=0.5)
    parser.add_argument("--perturbation-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--batch-size", type=int, default=1, help="reserved for future batched forward passes")
    parser.add_argument(
        "--record-start",
        type=int,
        default=0,
        help="inclusive audit-record index; useful for parallel generation runs",
    )
    parser.add_argument(
        "--record-end",
        type=int,
        help="exclusive audit-record index; defaults to the end of the audit set",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="batch size for generation-based baselines",
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/audits/baseline_v1/tasks"),
    )
    return parser.parse_args()



def _normalise_methods(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return METHODS
    requested = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not requested:
        raise ValueError("at least one baseline method is required")
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {', '.join(METHODS)}")
    return tuple(dict.fromkeys(requested))



def _run(args: argparse.Namespace, progress: RunProgress) -> None:
    methods = _normalise_methods(args.methods)
    if not 0.0 < args.prefix_ratio < 1.0:
        raise ValueError("--prefix-ratio must be in (0, 1)")
    if args.sead_samples <= 0 or args.samia_samples <= 0:
        raise ValueError("sampling counts must be positive")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")

    pretraining = getattr(args, "pretraining_manifest", None)
    output_dir = _resolve(args.output_dir).resolve()
    progress.event("loading tokenizer and audit records")
    if pretraining is not None:
        if args.pool_path is not None:
            raise ValueError("--pool-path cannot override a frozen pretraining manifest")
        from experiments.pretraining.data import load_evaluation
        evaluation = load_evaluation(_resolve(pretraining))
        run_dir = evaluation.manifest_path.parent
        cfg = evaluation.config
        tokenizer = evaluation.tokenizer
        members = [AuditRecord(record, 1) for record in evaluation.members]
        nonmembers = [AuditRecord(record, 0) for record in evaluation.nonmembers]
        auxiliary = evaluation.auxiliary
        split_metadata = evaluation.manifest
    else:
        run_dir = _resolve(args.run_dir).resolve()
        cfg = load_run_config(run_dir)
        tokenizer = _target_tokenizer(run_dir, cfg)
        members, nonmembers, auxiliary, split_metadata = load_audit_records(
            run_dir, cfg, tokenizer, args.pool_path
        )
    complete_records = members + nonmembers
    if not 0 <= args.record_start <= len(complete_records):
        raise ValueError("--record-start must be within the audit-record range")
    record_end = len(complete_records) if args.record_end is None else args.record_end
    if not args.record_start <= record_end <= len(complete_records):
        raise ValueError("--record-end must satisfy start <= end <= audit-record count")
    all_records = complete_records[args.record_start:record_end]
    labels = np.asarray([row.label for row in all_records], dtype=np.int64)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)
    progress.event("loading target model")
    if pretraining is not None:
        from experiments.pretraining.data import load_model
        model = load_model(evaluation.manifest["models"]["target"], device, args.attn_implementation, len(tokenizer))
    else:
        model = load_finetuned_model(
            run_dir, cfg.target_model, device, attn_implementation=args.attn_implementation
        )
    protocol = {
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "run_dir": str(run_dir),
        "benchmark": cfg.benchmark,
        "target_model": cfg.target_model,
        "target_checkpoint": str(
            next(
                (run_dir / directory / "target"
                 for directory in ("checkpoints", "adapters")
                 if (run_dir / directory / "target").exists()),
            run_dir / "checkpoints" / "target",
        )
        ),
        "methods": list(methods),
        "n_member": int(np.sum(labels == 1)),
        "n_nonmember": int(np.sum(labels == 0)),
        "n_auxiliary": len(auxiliary),
        "record_start": args.record_start,
        "record_end": record_end,
        "complete_n_member": len(members),
        "complete_n_nonmember": len(nonmembers),
        "k_percent": args.k_percent,
        "recall_shots": args.recall_shots,
        "icp_top_k": args.icp_top_k,
        "sead_samples": args.sead_samples,
        "sead_temperature": args.sead_temperature,
        "samia_samples": args.samia_samples,
        "generation_batch_size": args.generation_batch_size,
        "prefix_ratio": args.prefix_ratio,
        "split_metadata": split_metadata,
        "restrictions": {
            "reference_model": False,
            "unfinetuned_target_scored": False,
            "draft_model_loaded": False,
            "petal_calibration": "fine-tuned target on auxiliary records",
            "sead_estimator": "target-only Monte Carlo frequency density",
        },
    }
    if pretraining is not None:
        protocol.update(training_regime="pretraining", target_checkpoint=evaluation.manifest["models"]["target"],
                        token_contract=evaluation.manifest["token_contract"])
        protocol["restrictions"].update(unfinetuned_target_scored=True,
            petal_calibration="pretrained target on disjoint MIMIR auxiliary nonmembers")
    else:
        protocol["training_regime"] = "controlled_sft"
    progress.configure(protocol, labels, [row.record.record_id for row in all_records])
    progress.event("audit ready", records=len(all_records), methods=list(methods))

    warmup_records = getattr(args, "cost_warmup_records", 1)
    if warmup_records < 0 or not all_records:
        raise ValueError("warmup must be nonnegative and audit must be nonempty")
    protocol["cost_measurement"] = cost_protocol(model, device, min(warmup_records, len(all_records)), args.generation_batch_size)
    scores, costs = {}, {}
    reference_cache: dict[str, list[str]] = {}
    for method in methods:
        progress.active_method = method
        scorer = TargetScorer(model, tokenizer, device, args.sead_samples, args.sead_temperature, args.seed)
        try:
            progress.event("method warmup", method=method)
            for row in all_records[:warmup_records]:
                scorer.stats(row.record)
            set_seed(args.seed)
            meter = CostMeter(device, len(all_records))
            scorer.cost_meter = meter
            progress.event("method started", method=method)
            reused_reference = method in ("ws", "rs", "bt") and "texts" in reference_cache
            with meter.measure():
                values = score_methods(args, progress, scorer, all_records, auxiliary, tokenizer,
                                        (method,), reference_cache=reference_cache)
            scores[method] = values[method]
            measured_cost = meter.result()
            if method in ("ws", "rs", "bt"):
                if reused_reference:
                    costs[method] = {"cost_basis": "physical_incremental", "reference_reused": True,
                                     **{"physical_incremental_" + key: value
                                        for key, value in measured_cost.items()}}
                else:
                    costs[method] = {**measured_cost, "cost_basis": "standalone_measured",
                                     "reference_reused": False}
            else:
                costs[method] = measured_cost
            progress.save_method(method, scores[method], cost=costs[method])
        finally:
            scorer._probe_vector.cache_clear()
            del scorer

    write_cost_report(output_dir, protocol, costs)
    # A parallel generation shard may intentionally contain only one class.
    # Preserve its raw scores for the parent process instead of attempting an
    # AUC/TPR report that has no meaningful negative (or positive) examples.
    if np.unique(labels).size >= 2:
        render_report(output_dir, protocol, scores, labels, costs=costs)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "baseline_metrics.json").write_text(
            json.dumps(
                {"protocol": protocol, "metrics": {}, "scores": scores, "costs": costs},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    np.savez_compressed(
        output_dir / "baseline_scores.npz",
        labels=labels,
        record_ids=np.asarray([row.record.record_id for row in all_records]),
        **{name: np.asarray(values, dtype=np.float32) for name, values in scores.items()},
    )
    print(json.dumps({"output_dir": str(output_dir), "methods": list(scores)}, ensure_ascii=False))



def main() -> None:
    args = parse_args()
    with RunProgress(_resolve(args.output_dir), args.progress_interval) as progress:
        _run(args, progress)
