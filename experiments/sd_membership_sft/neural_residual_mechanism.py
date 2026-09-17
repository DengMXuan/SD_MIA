"""Diagnose whether neural residual gains are tie-breaking or broad reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_runtime import (_deterministic_subset)
from .audit_metrics import (membership_metrics)
from .audit_runtime import (split_indices)
from .audit_runtime import (BENCHMARKS, EPOCHS, N_CAL, N_REF, REPLAY_SEEDS, ROOT, SPLIT_SEED, _write_json)
from .neural_adaptive_accept_only import (_standardize)


LAMBDAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
BANDS = (0.10, 0.25, 0.50)


def lexicographic_score(base: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    if base.shape != secondary.shape or base.ndim != 1:
        raise ValueError("base and secondary must be aligned vectors")
    unique = np.unique(base)
    gaps = np.diff(unique)
    positive = gaps[gaps > 0]
    if len(positive) == 0:
        return secondary.copy()
    order = np.argsort(secondary, kind="stable")
    rank = np.empty(len(order), dtype=np.float64)
    rank[order] = np.linspace(-0.5, 0.5, len(order), dtype=np.float64)
    return base + float(np.min(positive)) * 0.49 * rank


def banded_tie_break_score(
    base: np.ndarray, secondary: np.ndarray, *, width: float
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    if base.shape != secondary.shape or width <= 0.0:
        raise ValueError("invalid banded tie-break input")
    groups = np.floor((base - float(np.min(base))) / width + 1e-12).astype(np.int64)
    score = groups.astype(np.float64)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        order = indices[np.argsort(secondary[indices], kind="stable")]
        score[order] += np.linspace(0.0, 0.9, len(order), dtype=np.float64)
    return score


def _pair_decomposition(
    base: np.ndarray,
    neural: np.ndarray,
    blend: np.ndarray,
    member: np.ndarray,
    nonmember: np.ndarray,
) -> dict[str, float | int]:
    base_difference = base[member, None] - base[nonmember][None, :]
    neural_difference = neural[member, None] - neural[nonmember][None, :]
    blend_difference = blend[member, None] - blend[nonmember][None, :]
    tie = np.isclose(base_difference, 0.0, atol=1e-12, rtol=0.0)
    correct, wrong = base_difference > 0.0, base_difference < 0.0
    return {
        "pairs": int(base_difference.size),
        "base_ties": int(np.sum(tie)),
        "tie_fraction": float(np.mean(tie)),
        "ties_neural_correct": int(np.sum(tie & (neural_difference > 0.0))),
        "ties_neural_wrong": int(np.sum(tie & (neural_difference < 0.0))),
        "base_correct_broken_by_blend": int(np.sum(correct & (blend_difference <= 0.0))),
        "base_wrong_repaired_by_blend": int(np.sum(wrong & (blend_difference > 0.0))),
        "strict_pairs_changed": int(
            np.sum((correct & (blend_difference <= 0.0)) | (wrong & (blend_difference > 0.0)))
        ),
    }


def _load_scores(benchmark: str, epoch: int, seed: int) -> dict[str, np.ndarray]:
    path = (
        ROOT
        / "experiments/results/sft_runs/accept_only_active_v2/neural_adaptive/conditions"
        / f"{benchmark}_epoch{epoch}"
        / f"scores_seed_{seed}.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def run() -> dict[str, Any]:
    rows = []
    for benchmark in BENCHMARKS:
        for epoch in EPOCHS:
            for seed in REPLAY_SEEDS:
                data = _load_scores(benchmark, epoch, seed)
                partitions = split_indices(data["labels"], SPLIT_SEED)
                d_nm = partitions["D"][data["labels"][partitions["D"]] == 0]
                c_nm = partitions["C"][data["labels"][partitions["C"]] == 0]
                reference = _deterministic_subset(d_nm, N_REF, SPLIT_SEED + N_REF)
                calibration = _deterministic_subset(c_nm, N_CAL, SPLIT_SEED + N_CAL + 1000)
                test = partitions["T"]
                member = test[data["labels"][test] == 1]
                nonmember = test[data["labels"][test] == 0]
                base = _standardize(data["lowq_k2"], reference)
                neural = _standardize(data["neural_score_k2"], reference)
                scores = {
                    f"lambda_{value:g}": base + value * neural for value in LAMBDAS
                }
                scores["exact_tie_break"] = lexicographic_score(base, neural)
                for width in BANDS:
                    scores[f"band_{width:g}"] = banded_tie_break_score(
                        base, neural, width=width
                    )
                metrics = {
                    name: membership_metrics(value, data["labels"], calibration, test)
                    for name, value in scores.items()
                }
                rows.append(
                    {
                        "benchmark": benchmark,
                        "epoch": epoch,
                        "seed": seed,
                        "metrics": metrics,
                        "pairs": _pair_decomposition(
                            base, neural, scores["lambda_0.25"], member, nonmember
                        ),
                    }
                )
    methods = tuple(rows[0]["metrics"])
    summary = {
        method: {
            "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
            "pauc_0_10": float(
                np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])
            ),
            "tpr_1": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows]
                )
            ),
        }
        for method in methods
    }
    pair_keys = tuple(rows[0]["pairs"])
    pair_summary = {
        key: float(np.sum([row["pairs"][key] for row in rows]))
        if key not in ("tie_fraction",)
        else float(np.mean([row["pairs"][key] for row in rows]))
        for key in pair_keys
    }
    return {"experiment": "neural residual mechanism", "rows": rows, "summary": summary, "pairs": pair_summary}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Neural Residual Mechanism",
        "",
        "| Method | AUC | pAUC | TPR@1% |",
        "|---|---:|---:|---:|",
    ]
    for method, metric in report["summary"].items():
        lines.append(
            f"| `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | {metric['tpr_1']:.4f} |"
        )
    pairs = report["pairs"]
    lines.extend(
        [
            "",
            "## Pair decomposition",
            "",
            f"- Exact Low-q tie fraction: {pairs['tie_fraction']:.2%}",
            f"- Tied pairs resolved correctly/wrongly by neural score: {int(pairs['ties_neural_correct'])}/{int(pairs['ties_neural_wrong'])}",
            f"- Strictly correct Low-q pairs broken by lambda=0.25: {int(pairs['base_correct_broken_by_blend'])}",
            f"- Strictly wrong Low-q pairs repaired by lambda=0.25: {int(pairs['base_wrong_repaired_by_blend'])}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/neural_residual_mechanism",
    )
    args = parser.parse_args()
    report = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "REPORT.json", report)
    write_markdown(report, args.output_dir / "REPORT.md")
    print(json.dumps({"output": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
