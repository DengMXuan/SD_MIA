"""Fit and evaluate the accept-only detector on the independent 600 role.

The target and draft language models are already fine-tuned and remain frozen.
Their accept-only observations are supplied as one explicit four-role archive:
600 trusted audit nonmembers, 2,000 members, and 2,000 nonmembers. The small
difficulty TCN uses 320/80 audit records for train/validation and the remaining
200 only for operating-point calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

import numpy as np
import torch

from experiments.sd_membership_sft.core.audit_metrics import conformal_tail_pvalues, membership_metrics
from experiments.sd_membership_sft.core.audit_partitions import deployment_partitions
from experiments.sd_membership_sft.core.audit_runtime import _write_json
from experiments.sd_membership_sft.methods.conditional_accept_only import load_observations, observable_inputs
from experiments.sd_membership_sft.core.data_contract import DEFAULT_DATA_CONTRACT
from experiments.sd_membership_sft.core.deployment_archive import validate_deployment_archive
from experiments.sd_membership_sft.methods.difficulty_accept_only import RESULTS, aggregate, fit, predict, save_fit


OUTPUT = RESULTS / "deployment_validation_independent_nm600"
OBSERVATIONS = RESULTS / "deployment_observations"
CONDITIONS = tuple(itertools.product(
    ("wikitection", "newstection", "arxivtection"),
    (1, 3),
    (20260914, 20260915, 20260916),
))
RATES = (0.01, 0.05, 0.10)


def _load_archive(path: Path):
    provenance = validate_deployment_archive(path)
    observations, labels, record_ids = load_observations(path)
    with np.load(path, allow_pickle=False) as archive:
        record_roles = np.asarray(archive["record_roles"]).astype(str)
        draft_features = np.asarray(archive["draft_features"], dtype=np.float32)
    if len(record_roles) != len(labels):
        raise ValueError("record_roles are not aligned with records")
    if draft_features.ndim != 2 or len(draft_features) != len(observations.logq):
        raise ValueError("draft_features must align with candidate-token observations")
    if not np.isfinite(draft_features).all():
        raise ValueError("draft_features contain nonfinite values")
    return (
        observations,
        labels,
        record_ids,
        record_roles,
        draft_features,
        provenance,
    )


def _report_matches_source(output: Path, provenance: dict[str, object]) -> bool:
    report_path = output / "REPORT.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = report.get("source", {})
    return (
        source.get("sha256") == provenance.get("archive_sha256")
        and source.get("sidecar_sha256") == provenance.get("sidecar_sha256")
    )


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _decisions(pvalues: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, object]]:
    result = {}
    for rate in RATES:
        hits = pvalues <= rate
        members = labels == 1
        nonmembers = labels == 0
        member_hits = int(hits[members].sum())
        nonmember_hits = int(hits[nonmembers].sum())
        result[f"{int(rate * 100)}%"] = {
            "tpr": member_hits / int(members.sum()),
            "tpr_95ci": _wilson(member_hits, int(members.sum())),
            "actual_fpr": nonmember_hits / int(nonmembers.sum()),
            "actual_fpr_95ci": _wilson(nonmember_hits, int(nonmembers.sum())),
            "member_hits": member_hits,
            "nonmember_hits": nonmember_hits,
        }
    return result


def evaluate(
    benchmark: str,
    epoch: int,
    seed: int,
    device: str = "cpu",
    observation_path: Path | None = None,
) -> None:
    output = OUTPUT / f"{benchmark}_epoch{epoch}" / f"seed{seed}"
    observation_path = observation_path or (
        OBSERVATIONS / f"{benchmark}_epoch{epoch}" / f"seed{seed}.npz"
    )
    (
        observations,
        labels,
        record_ids,
        record_roles,
        extra,
        provenance,
    ) = _load_archive(observation_path)
    expected_source = {
        "benchmark": benchmark,
        "target_epochs": epoch,
        "acceptance_seed": seed,
    }
    for name, expected in expected_source.items():
        if provenance.get(name) != expected:
            raise ValueError(
                f"observation provenance {name}={provenance.get(name)!r} "
                f"does not match requested {expected!r}"
            )
    if _report_matches_source(output, provenance):
        return
    parts = deployment_partitions(labels, record_ids, record_roles)
    base_features, counts, _ = observable_inputs(observations, 2, False)
    features = np.column_stack((base_features, extra))
    model, mean, scale, history, best_epoch = fit(
        features,
        counts,
        observations.lengths,
        parts,
        seed=seed,
        device=device,
    )
    logpmf = predict(model, (features - mean) / scale, counts, observations.lengths, device)
    scores = aggregate(logpmf, counts, observations.lengths, sparse=True)
    test = parts["test"]
    calibration = parts["calibration"]
    pvalues = conformal_tail_pvalues(scores[test], scores[calibration])
    raw_metrics = membership_metrics(scores, labels, calibration, test)
    decisions = _decisions(pvalues, labels[test])

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "difficulty.pt"
    training = save_fit(
        checkpoint, model, mean, scale, history, best_epoch
    )
    np.savez_compressed(
        output / "scores.npz", labels=labels, record_ids=record_ids,
        record_roles=record_roles, **parts,
        difficulty_sparse=scores, test_pvalues=pvalues,
    )
    counts_by_role = {
        "train_nonmembers": len(parts["train"]),
        "validation_nonmembers": len(parts["validation"]),
        "calibration_nonmembers": len(calibration),
        "auxiliary_nonmembers_total": len(parts["train"]) + len(parts["validation"]) + len(calibration),
        "test_members": int((labels[test] == 1).sum()),
        "test_nonmembers": int((labels[test] == 0).sum()),
    }
    report = {
        "benchmark": benchmark,
        "epoch": epoch,
        "seed": seed,
        "method": "difficulty_tcn_sparse_pooled200",
        "language_models_frozen": True,
        "detector_frozen_during_scoring": True,
        "new_detector_fits": 1,
        "training_member_count": 0,
        "synthetic_member_count": 0,
        "counts": counts_by_role,
        "training": training,
        "raw_metrics": raw_metrics,
        "calibrated_decisions": decisions,
        "calibration_min_pvalue": 1.0 / (len(calibration) + 1.0),
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "source": {
            "observation_archive": str(observation_path),
            "sha256": provenance["archive_sha256"],
            "sidecar": provenance["sidecar"],
            "sidecar_sha256": provenance["sidecar_sha256"],
            "producer": provenance["producer"],
            "protocol": provenance["protocol"],
            "run_manifest_sha256": provenance["run_manifest_sha256"],
            "draft_checkpoint": provenance["draft_checkpoint"],
            "target_checkpoint": provenance["target_checkpoint"],
        },
        "query_costs": {
            "per_record": "2L accept decisions for L candidate tokens",
            "calibration_total": int(2 * observations.lengths[calibration].sum()),
            "test_total": int(2 * observations.lengths[test].sum()),
        },
        "protocol": (
            "independent four-role data; feature-consistent q; B=2 fixed-candidate "
            "accept-only observations; pooled 200 nonmember calibration"
        ),
    }
    _write_json(output / "REPORT.json", report)
    print(json.dumps({"completed": str(output), "auc": raw_metrics["auc"], "decisions": decisions}), flush=True)


def matrix(args) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    gpu_ids = args.gpus.split(",") if args.gpus else []
    devices: queue.Queue[str] = queue.Queue()
    for gpu in gpu_ids:
        devices.put(gpu)
    # Every subprocess validates the source hash before deciding whether its
    # report is reusable. Presence of REPORT.json alone is not a completion
    # condition because an observation archive may have been regenerated.
    jobs = list(CONDITIONS)
    print(json.dumps({
        "scheduled": len(jobs),
        "auxiliary_nonmembers": DEFAULT_DATA_CONTRACT.audit_auxiliary,
        "test_per_class": DEFAULT_DATA_CONTRACT.members,
    }), flush=True)

    def run(case):
        benchmark, epoch, seed = case
        gpu = devices.get() if gpu_ids else None
        environment = os.environ.copy()
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable, "-m", "experiments.sd_membership_sft.deployment_accept_only", "evaluate",
            "--benchmark", benchmark, "--epoch", str(epoch), "--seed", str(seed),
            "--device", "cuda:0" if gpu is not None else "cpu",
        ]
        try:
            log = OUTPUT / "logs" / f"{benchmark}_epoch{epoch}_seed{seed}.log"
            with log.open("w") as handle:
                subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"completed": case}), flush=True)
        finally:
            if gpu is not None:
                devices.put(gpu)

    workers = min(args.jobs, len(gpu_ids)) if gpu_ids else args.jobs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))


def summarize() -> None:
    rows = []
    for benchmark, epoch, seed in CONDITIONS:
        path = OUTPUT / f"{benchmark}_epoch{epoch}" / f"seed{seed}" / "REPORT.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text())
        one = report["calibrated_decisions"]["1%"]
        rows.append({
            "benchmark": benchmark, "epoch": epoch, "seed": seed,
            "auc": report["raw_metrics"]["auc"],
            "pauc_0_10": report["raw_metrics"]["pauc_0_10"],
            "tpr_1": one["tpr"], "actual_fpr_1": one["actual_fpr"],
        })
    aggregate = {name: float(np.mean([row[name] for row in rows]))
                 for name in ("auc", "pauc_0_10", "tpr_1", "actual_fpr_1")}
    result = {
        "method": "difficulty_tcn_sparse_pooled200",
        "run_count": len(rows),
        "counts_per_run": {
            "auxiliary_nonmembers": DEFAULT_DATA_CONTRACT.audit_auxiliary,
            "test_members": DEFAULT_DATA_CONTRACT.members,
            "test_nonmembers": DEFAULT_DATA_CONTRACT.nonmembers,
        },
        "macro_average": aggregate,
        "runs": rows,
    }
    _write_json(OUTPUT / "DEPLOYMENT_REPORT.json", result)
    contract = DEFAULT_DATA_CONTRACT
    lines = [
        f"# Accept-only deployment evaluation: {contract.audit_auxiliary} trusted nonmembers",
        "",
        (
            "Frozen language models; difficulty TCN + sparse score; "
            f"{contract.detector_train} train, "
            f"{contract.detector_validation} validation, "
            f"{contract.calibration} pooled calibration; test "
            f"{contract.members:,} members + {contract.nonmembers:,} nonmembers."
        ),
        "",
        "| AUC | pAUC@10% | TPR@nominal 1% | Actual FPR |",
        "|---:|---:|---:|---:|",
        f"| {aggregate['auc']:.4f} | {aggregate['pauc_0_10']:.4f} | {aggregate['tpr_1']:.4f} | {aggregate['actual_fpr_1']:.4f} |",
        "",
        "Macro-average over six conditions and three verifier seeds. Repeated seeds/checkpoints sharing records are not independent datasets.",
    ]
    (OUTPUT / "DEPLOYMENT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "matrix", "summarize"))
    parser.add_argument("--benchmark", choices=("wikitection", "newstection", "arxivtection"))
    parser.add_argument("--epoch", type=int, choices=(1, 3))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--observations", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "evaluate":
        if args.benchmark is None or args.epoch is None:
            parser.error("benchmark and epoch required")
        evaluate(
            args.benchmark,
            args.epoch,
            args.seed,
            args.device,
            args.observations,
        )
    elif args.command == "matrix":
        matrix(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
