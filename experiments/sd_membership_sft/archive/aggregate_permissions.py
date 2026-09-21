"""Aggregate Stat-Delta or Accept-Only condition reports.

All comparisons are record-paired within each benchmark/epoch condition. The
macro table averages the six registered conditions; paired bootstrap deltas
reuse the same member/nonmember resample for both methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sd_membership_sft.archive.full_delta_mia import (_operating_point)
from experiments.sd_membership_sft.core.audit_metrics import (partial_auc, rank_auc)

from experiments.paths import ROOT
CONDITIONS = tuple((benchmark, epoch) for benchmark in ("wikitection", "newstection", "arxivtection") for epoch in (1, 3))


def _metric(scores: np.ndarray, labels: np.ndarray, partitions: dict[str, list[str]], ids: np.ndarray) -> dict[str, float]:
    lookup = {str(value): index for index, value in enumerate(ids)}
    idx = {name: np.asarray([lookup[str(value)] for value in values], dtype=np.int64) for name, values in partitions.items()}
    test_member = idx["T"][labels[idx["T"]] == 1]
    test_nonmember = idx["T"][labels[idx["T"]] == 0]
    calibration = idx["C"][labels[idx["C"]] == 0]
    test_scores = np.r_[scores[test_member], scores[test_nonmember]]
    test_labels = np.r_[np.ones(len(test_member), dtype=np.int64), np.zeros(len(test_nonmember), dtype=np.int64)]
    p1 = _operating_point(scores[test_member], scores[test_nonmember], scores[calibration], 0.01)
    p10 = _operating_point(scores[test_member], scores[test_nonmember], scores[calibration], 0.10)
    return {
        "auc": rank_auc(scores[test_member], scores[test_nonmember]),
        "pauc_0_10": partial_auc(test_scores, test_labels),
        "tpr_1": p1["tpr"],
        "tpr_10": p10["tpr"],
        "actual_fpr_1": p1["actual_fpr"],
        "actual_fpr_10": p10["actual_fpr"],
    }


def _score_path(report: dict[str, Any], method: str, condition_dir: Path) -> Path:
    result = report["methods"][method]
    if result.get("scores_path"):
        return Path(result["scores_path"])
    normalized = method.lower().replace("-", "_")
    candidates = sorted(condition_dir.glob(f"scores_{normalized}*.npz"))
    if not candidates:
        candidates = sorted(condition_dir.glob(f"scores_{method.lower()}*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no score file for {method} in {condition_dir}")
    return candidates[0]


def _paired_delta(left: dict[str, Any], right: dict[str, Any], left_scores: np.ndarray, right_scores: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("auc", "pauc_0_10", "tpr_1", "tpr_10")}
    for _ in range(repeats):
        labels = left["labels"]
        partitions = left["partitions"]
        ids = left["ids"]
        lookup = {str(value): index for index, value in enumerate(ids)}
        idx = {name: np.asarray([lookup[str(value)] for value in rows], dtype=np.int64) for name, rows in partitions.items()}
        tm = idx["T"][labels[idx["T"]] == 1]
        tn = idx["T"][labels[idx["T"]] == 0]
        c = idx["C"][labels[idx["C"]] == 0]
        tm = tm[rng.integers(0, len(tm), len(tm))]
        tn = tn[rng.integers(0, len(tn), len(tn))]
        c = c[rng.integers(0, len(c), len(c))]
        lmember, lnon, lcal = left_scores[tm], left_scores[tn], left_scores[c]
        rmember, rnon, rcal = right_scores[tm], right_scores[tn], right_scores[c]
        values["auc"].append(rank_auc(lmember, lnon) - rank_auc(rmember, rnon))
        values["pauc_0_10"].append(partial_auc(np.r_[lmember, lnon], np.r_[np.ones(len(lmember)), np.zeros(len(lnon))]) - partial_auc(np.r_[rmember, rnon], np.r_[np.ones(len(rmember)), np.zeros(len(rnon))]))
        values["tpr_1"].append(_operating_point(lmember, lnon, lcal, 0.01)["tpr"] - _operating_point(rmember, rnon, rcal, 0.01)["tpr"])
        values["tpr_10"].append(_operating_point(lmember, lnon, lcal, 0.10)["tpr"] - _operating_point(rmember, rnon, rcal, 0.10)["tpr"])
    return {key: {"point": float(np.mean(value)), "ci95_low": float(np.quantile(value, 0.025)), "ci95_high": float(np.quantile(value, 0.975))} for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stat_delta", "accept_only"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition-suffix",
        default="",
        help="suffix appended to draft_auxiliary_distilled, e.g. _no_eos",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="report filename; defaults to the kind-specific report name",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    records: list[dict[str, Any]] = []
    for benchmark, epoch in CONDITIONS:
        condition_dir = root / f"{benchmark}_epoch{epoch}" / f"draft_auxiliary_distilled{args.condition_suffix}"
        report_name = args.report_name or ("STAT_DELTA_REPORT.json" if args.kind == "stat_delta" else "ACCEPT_ONLY_REPORT.json")
        report_path = condition_dir / report_name
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        methods = report["methods"]
        for method in methods:
            score_path = _score_path(report, method, condition_dir)
            with np.load(score_path, allow_pickle=False) as data:
                scores = np.asarray(data["scores"], dtype=np.float64).mean(axis=1)
                labels = np.asarray(data["labels"], dtype=np.int64)
                ids = np.asarray(data["record_ids"])
            partitions = report["partitions"]
            metric = _metric(scores, labels, partitions, ids)
            records.append({"benchmark": benchmark, "epoch": epoch, "method": method, "metric": metric, "scores": scores, "labels": labels, "ids": ids, "partitions": partitions})
    if not records:
        raise RuntimeError(f"no completed {args.kind} condition reports under {root}")
    methods = sorted({row["method"] for row in records})
    by_method: dict[str, list[dict[str, Any]]] = {method: [row for row in records if row["method"] == method] for method in methods}
    summary: dict[str, Any] = {"kind": args.kind, "conditions_completed": sorted({f"{row['benchmark']}_epoch{row['epoch']}" for row in records}), "methods": {}}
    for method, rows in by_method.items():
        summary["methods"][method] = {key: float(np.mean([row["metric"][key] for row in rows])) for key in ("auc", "pauc_0_10", "tpr_1", "tpr_10", "actual_fpr_1", "actual_fpr_10")}
    if args.baseline in by_method:
        paired: dict[str, Any] = {}
        base_rows = {(row["benchmark"], row["epoch"]): row for row in by_method[args.baseline]}
        for method, rows in by_method.items():
            if method == args.baseline:
                continue
            deltas = []
            for row in rows:
                key = (row["benchmark"], row["epoch"])
                if key not in base_rows:
                    continue
                base = base_rows[key]
                left = {"labels": row["labels"], "ids": row["ids"], "partitions": row["partitions"]}
                right = {"labels": base["labels"], "ids": base["ids"], "partitions": base["partitions"]}
                deltas.append(_paired_delta(left, right, row["scores"], base["scores"], args.bootstrap_repeats, 20261050 + len(deltas)))
            if deltas:
                paired[method] = {key: {bound: float(np.mean([delta[key][bound] for delta in deltas])) for bound in ("point", "ci95_low", "ci95_high")} for key in deltas[0]}
        summary["paired_vs_baseline"] = {"baseline": args.baseline, "delta_is_method_minus_baseline": True, "by_method": paired}
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.kind.upper()}_AGGREGATE.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {args.kind} aggregate", "", f"- Completed conditions: {len(summary['conditions_completed'])}/6", "", "| Method | AUC | pAUC[0,0.1] | TPR@1% | actual FPR@1% | TPR@10% | actual FPR@10% |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, metric in summary["methods"].items():
        lines.append(f"| {method} | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | {metric['tpr_1']:.4f} | {metric['actual_fpr_1']:.4f} | {metric['tpr_10']:.4f} | {metric['actual_fpr_10']:.4f} |")
    (output_dir / f"{args.kind.upper()}_AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "conditions": len(summary["conditions_completed"]), "methods": len(methods)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
