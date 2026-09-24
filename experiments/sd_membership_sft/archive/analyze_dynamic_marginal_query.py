"""Paired record-bootstrap deltas for dynamic marginal query allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.shared.core.audit_metrics import conformal_tail_pvalues
from experiments.shared.core.audit_metrics import rank_auc
from experiments.shared.core.audit_runtime import split_indices
from experiments.shared.core.audit_runtime import ROOT, SPLIT_SEED, _write_json


COMPARISONS = (
    ("fixed50_active", "uniform_active"),
    ("dynamic_active", "uniform_active"),
    ("dynamic_capped_active", "uniform_active"),
    ("fixed50_fusion", "uniform_fusion"),
    ("dynamic_fusion", "uniform_fusion"),
    ("dynamic_capped_fusion", "uniform_fusion"),
)


def _partial_auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.10) -> float:
    """Vectorized equivalent of the canonical tie-aware partial AUC."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered_scores = np.asarray(scores, dtype=np.float64)[order]
    ordered_labels = np.asarray(labels, dtype=np.int64)[order]
    ends = np.r_[
        np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]),
        len(ordered_scores) - 1,
    ]
    positives = max(1, int(np.sum(ordered_labels == 1)))
    negatives = max(1, int(np.sum(ordered_labels == 0)))
    cumulative_positive = np.cumsum(ordered_labels == 1)
    cumulative_negative = np.cumsum(ordered_labels == 0)
    fpr = np.r_[0.0, cumulative_negative[ends] / negatives]
    tpr = np.r_[0.0, cumulative_positive[ends] / positives]
    left = int(np.searchsorted(fpr, max_fpr, side="right") - 1)
    clipped_fpr = fpr[: left + 1]
    clipped_tpr = tpr[: left + 1]
    if clipped_fpr[-1] < max_fpr:
        fraction = (max_fpr - fpr[left]) / (fpr[left + 1] - fpr[left])
        boundary_tpr = tpr[left] + fraction * (tpr[left + 1] - tpr[left])
        clipped_fpr = np.r_[clipped_fpr, max_fpr]
        clipped_tpr = np.r_[clipped_tpr, boundary_tpr]
    return float(np.trapezoid(clipped_tpr, clipped_fpr) / max_fpr)


def _metric(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    member = test[labels[test] == 1]
    nonmember = test[labels[test] == 0]
    calibration = calibration[labels[calibration] == 0]
    ordered_scores = np.r_[scores[member], scores[nonmember]]
    ordered_labels = np.r_[
        np.ones(len(member), dtype=np.int64),
        np.zeros(len(nonmember), dtype=np.int64),
    ]
    member_p = conformal_tail_pvalues(scores[member], scores[calibration])
    return np.asarray(
        [
            rank_auc(scores[member], scores[nonmember]),
            _partial_auc(ordered_scores, ordered_labels),
            np.mean(member_p <= 0.01),
            np.mean(member_p <= 0.10),
        ],
        dtype=np.float64,
    )


def analyze_budget(root: Path, pilot: int, budget: int, repeats: int, seed: int) -> dict[str, Any]:
    paths = sorted((root / "conditions").glob(f"*_p{pilot}_b{budget}/scores_seed_*.npz"))
    archives = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            archives.append({name: np.asarray(data[name]) for name in data.files})
    if len(archives) != 18:
        raise ValueError(f"expected 18 archives for p{pilot}/b{budget}, found {len(archives)}")
    names = ("auc", "pauc_0_10", "tpr_1", "tpr_10")
    pairs = list(COMPARISONS)
    if pilot == 1 and budget == 2:
        pairs.append(("uniform_fusion", "uniform_normal_k2_shadow"))
    condition = {pair: [] for pair in pairs}
    for archive in archives:
        labels = archive["labels"].astype(np.int64)
        partitions = split_indices(labels, SPLIT_SEED)
        for left, right in pairs:
            condition[(left, right)].append(
                _metric(archive[left], labels, partitions["C"], partitions["T"])
                - _metric(archive[right], labels, partitions["C"], partitions["T"])
            )
    rng = np.random.default_rng(seed + 101 * pilot + budget)
    bootstrap = {
        pair: np.empty((repeats, len(names)), dtype=np.float64) for pair in pairs
    }
    for repeat in range(repeats):
        accumulated = {pair: np.zeros(len(names), dtype=np.float64) for pair in pairs}
        for archive in archives:
            labels = archive["labels"].astype(np.int64)
            partitions = split_indices(labels, SPLIT_SEED)
            cal = partitions["C"][labels[partitions["C"]] == 0]
            member = partitions["T"][labels[partitions["T"]] == 1]
            nonmember = partitions["T"][labels[partitions["T"]] == 0]
            sampled_cal = cal[rng.integers(0, len(cal), len(cal))]
            sampled_test = np.r_[
                member[rng.integers(0, len(member), len(member))],
                nonmember[rng.integers(0, len(nonmember), len(nonmember))],
            ]
            for left, right in pairs:
                accumulated[(left, right)] += _metric(
                    archive[left], labels, sampled_cal, sampled_test
                ) - _metric(archive[right], labels, sampled_cal, sampled_test)
        for pair in pairs:
            bootstrap[pair][repeat] = accumulated[pair] / len(archives)
    comparisons: dict[str, Any] = {}
    for pair in pairs:
        values = np.asarray(condition[pair])
        comparisons[f"{pair[0]}_minus_{pair[1]}"] = {
            name: {
                "point": float(np.mean(values[:, index])),
                "ci95_low": float(np.quantile(bootstrap[pair][:, index], 0.025)),
                "ci95_high": float(np.quantile(bootstrap[pair][:, index], 0.975)),
                "wins": int(np.sum(values[:, index] > 0.0)),
                "total": len(values),
            }
            for index, name in enumerate(names)
        }
    return {
        "pilot_k": pilot,
        "total_budget": budget,
        "archives": len(archives),
        "bootstrap_repeats": repeats,
        "comparisons": comparisons,
    }


def write_markdown(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Paired Analysis: Dynamic Marginal Query Allocation",
        "",
        "Positive deltas favor the left method; CIs use paired record resampling.",
    ]
    for report in reports:
        lines.extend(
            [
                "",
                f"## Pilot K={report['pilot_k']}, mean total K={report['total_budget']}",
                "",
                "| Comparison | Delta AUC [95% CI] | Delta pAUC [95% CI] | pAUC wins |",
                "|---|---:|---:|---:|",
            ]
        )
        for comparison, metrics in report["comparisons"].items():
            auc, pauc = metrics["auc"], metrics["pauc_0_10"]
            auc_ci = (
                f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}]"
                if "ci95_low" in auc
                else "—"
            )
            pauc_ci = (
                f"[{pauc['ci95_low']:+.4f}, {pauc['ci95_high']:+.4f}]"
                if "ci95_low" in pauc
                else "—"
            )
            lines.append(
                f"| `{comparison}` | {auc['point']:+.4f} {auc_ci} | "
                f"{pauc['point']:+.4f} {pauc_ci} | {pauc['wins']}/{pauc['total']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/dynamic_marginal_query",
    )
    args = parser.parse_args()
    root = args.input_dir.resolve()
    reports = [
        analyze_budget(root, pilot, budget, args.repeats, args.seed)
        for pilot, budget in ((1, 2), (2, 8))
    ]
    _write_json(root / "PAIRED_ANALYSIS.json", reports)
    write_markdown(reports, root / "PAIRED_ANALYSIS.md")
    print(json.dumps({"reports": len(reports), "repeats": args.repeats}, indent=2))


if __name__ == "__main__":
    main()
