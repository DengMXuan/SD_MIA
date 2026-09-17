"""Paired record-bootstrap analysis for equal-budget query allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..audit_metrics import (membership_metrics)
from ..audit_runtime import (split_indices)
from ..audit_runtime import (ROOT, SPLIT_SEED, _write_json)


COMPARISONS = (
    ("full_ai_one_shot", "uniform"),
    ("full_ai_sequential", "uniform"),
    ("hybrid", "uniform"),
    ("oracle", "full_ai_one_shot"),
)
BRANCHES = ("active", "fusion")
METRICS = ("auc", "pauc_0_10", "tpr_1", "fpr_1", "tpr_10", "fpr_10")


def _metric_vector(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    value = membership_metrics(scores, labels, calibration, test)
    return np.asarray(
        (
            value["auc"],
            value["pauc_0_10"],
            value["tpr_at_fpr"]["1%"]["tpr"],
            value["tpr_at_fpr"]["1%"]["actual_fpr"],
            value["tpr_at_fpr"]["10%"]["tpr"],
            value["tpr_at_fpr"]["10%"]["actual_fpr"],
        ),
        dtype=np.float64,
    )


def _resampled_partitions(
    labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    partitions = split_indices(labels, SPLIT_SEED)
    calibration = partitions["C"][labels[partitions["C"]] == 0]
    member = partitions["T"][labels[partitions["T"]] == 1]
    nonmember = partitions["T"][labels[partitions["T"]] == 0]
    sampled_calibration = calibration[rng.integers(0, len(calibration), len(calibration))]
    sampled_member = member[rng.integers(0, len(member), len(member))]
    sampled_nonmember = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
    return sampled_calibration, np.r_[sampled_member, sampled_nonmember]


def _summary(
    point: float,
    bootstrap: np.ndarray,
    condition_deltas: np.ndarray,
    *,
    improvement_sign: float = 1.0,
) -> dict[str, Any]:
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_improvement": float(
            np.mean(improvement_sign * bootstrap > 0.0)
        ),
        "condition_seed_wins": int(np.sum(improvement_sign * condition_deltas > 0.0)),
        "condition_seed_ties": int(np.sum(condition_deltas == 0.0)),
        "condition_seed_total": int(len(condition_deltas)),
    }


def analyze(
    root: Path, *, repeats: int = 1000, seed: int = 20260922
) -> dict[str, Any]:
    paths = sorted((root / "conditions").glob("*/scores_seed_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no score archives under {root}")
    archives: list[dict[str, np.ndarray]] = []
    points: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            archive = {name: np.asarray(data[name]) for name in data.files}
        required = {
            "labels",
            *(f"{method}_{branch}" for pair in COMPARISONS for method in pair for branch in BRANCHES),
        }
        missing = required.difference(archive)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        labels = np.asarray(archive["labels"], dtype=np.int64)
        partitions = split_indices(labels, SPLIT_SEED)
        point = {
            name: _metric_vector(value, labels, partitions["C"], partitions["T"])
            for name, value in archive.items()
            if name not in ("labels", "record_ids")
        }
        archives.append(archive)
        points.append(point)

    rng = np.random.default_rng(seed)
    boot = {
        (left, right, branch): np.empty((repeats, len(METRICS)), dtype=np.float64)
        for left, right in COMPARISONS
        for branch in BRANCHES
    }
    for repeat in range(repeats):
        accum = {key: np.zeros(len(METRICS), dtype=np.float64) for key in boot}
        for archive in archives:
            labels = np.asarray(archive["labels"], dtype=np.int64)
            calibration, test = _resampled_partitions(labels, rng)
            for left, right in COMPARISONS:
                for branch in BRANCHES:
                    key = (left, right, branch)
                    accum[key] += _metric_vector(
                        archive[f"{left}_{branch}"], labels, calibration, test
                    ) - _metric_vector(
                        archive[f"{right}_{branch}"], labels, calibration, test
                    )
        for key in boot:
            boot[key][repeat] = accum[key] / len(archives)

    comparisons: dict[str, Any] = {}
    for left, right in COMPARISONS:
        for branch in BRANCHES:
            key = (left, right, branch)
            condition = np.stack(
                [point[f"{left}_{branch}"] - point[f"{right}_{branch}"] for point in points]
            )
            comparisons[f"{left}_minus_{right}_{branch}"] = {
                metric: _summary(
                    float(np.mean(condition[:, index])),
                    boot[key][:, index],
                    condition[:, index],
                    improvement_sign=-1.0 if metric.startswith("fpr_") else 1.0,
                )
                for index, metric in enumerate(METRICS)
            }

    raw_paths = sorted((root / "conditions").glob("*/RAW_RESULTS.json"))
    rows = [row for path in raw_paths for row in json.loads(path.read_text())["rows"]]
    rmse: dict[str, Any] = {}
    rmse_rng = np.random.default_rng(seed + 1)
    for left, right in COMPARISONS:
        values = np.asarray(
            [row["delta_rmse"][left] - row["delta_rmse"][right] for row in rows],
            dtype=np.float64,
        )
        sampled = np.mean(
            values[rmse_rng.integers(0, len(values), size=(10000, len(values)))], axis=1
        )
        rmse[f"{left}_minus_{right}"] = _summary(
            float(np.mean(values)), sampled, values, improvement_sign=-1.0
        )
        rmse[f"{left}_minus_{right}"]["condition_seed_improvements"] = int(np.sum(values < 0.0))

    return {
        "experiment": "paired analysis of equal-budget query allocation",
        "score_archives": len(archives),
        "record_bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "sign_convention": "positive AUC/pAUC/TPR deltas favor the left method; negative FPR/RMSE deltas favor the left method",
        "comparisons": comparisons,
        "delta_rmse": rmse,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Paired Analysis: Equal-Budget Query Allocation",
        "",
        f"Macro-average over {report['score_archives']} condition/seed score archives; "
        f"95% intervals use {report['record_bootstrap_repeats']} paired record resamples.",
        "",
        "Positive score deltas favor the left method. Intervals excluding zero are marked `yes`.",
        "",
        "| Comparison | Metric | Delta | 95% CI | Wins | Excludes zero |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for comparison, metrics in report["comparisons"].items():
        for metric in ("auc", "pauc_0_10", "tpr_1", "tpr_10"):
            value = metrics[metric]
            excludes = value["ci95_low"] > 0.0 or value["ci95_high"] < 0.0
            lines.append(
                f"| `{comparison}` | {metric} | {value['point']:+.4f} | "
                f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}] | "
                f"{value['condition_seed_wins']}/{value['condition_seed_total']} | "
                f"{'yes' if excludes else 'no'} |"
            )
    lines.extend(
        [
            "",
            "Negative RMSE deltas favor the left method.",
            "",
            "| Comparison | Delta RMSE | 95% CI | Improvements |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, value in report["delta_rmse"].items():
        lines.append(
            f"| `{comparison}` | {value['point']:+.4f} | "
            f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}] | "
            f"{value['condition_seed_improvements']}/{value['condition_seed_total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/full_ai_allocation",
    )
    args = parser.parse_args()
    report = analyze(args.input_dir.resolve(), repeats=args.repeats, seed=args.seed)
    _write_json(args.input_dir.resolve() / "PAIRED_ANALYSIS.json", report)
    write_markdown(report, args.input_dir.resolve() / "PAIRED_ANALYSIS.md")
    print(json.dumps({"score_archives": report["score_archives"]}, indent=2))


if __name__ == "__main__":
    main()
