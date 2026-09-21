"""Combine the best K=2 shadow scale gate with saved K=8 active scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sd_membership_sft.core.audit_runtime import (_deterministic_subset)
from experiments.sd_membership_sft.core.audit_metrics import (membership_metrics)
from experiments.sd_membership_sft.core.audit_runtime import (split_indices)
from experiments.sd_membership_sft.core.audit_runtime import (N_CAL, N_REF, ROOT, SPLIT_SEED, _write_json)
from experiments.sd_membership_sft.archive.neural_adaptive_accept_only import (_standardize)


ACTIVE_METHODS = ("uniform", "hybrid", "full_ai_one_shot", "full_ai_sequential")


def evaluate_pair(shadow_path: Path, active_path: Path) -> dict[str, Any]:
    with np.load(shadow_path, allow_pickle=False) as shadow_data:
        shadow = {name: np.asarray(shadow_data[name]) for name in shadow_data.files}
    with np.load(active_path, allow_pickle=False) as active_data:
        active = {name: np.asarray(active_data[name]) for name in active_data.files}
    if not np.array_equal(shadow["labels"], active["labels"]):
        raise ValueError("shadow and active labels differ")
    if not np.array_equal(shadow["record_ids"], active["record_ids"]):
        raise ValueError("shadow and active record IDs differ")
    labels = np.asarray(shadow["labels"], dtype=np.int64)
    partitions = split_indices(labels, SPLIT_SEED)
    d_nm = partitions["D"][labels[partitions["D"]] == 0]
    c_nm = partitions["C"][labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nm, N_REF, SPLIT_SEED + N_REF)
    calibration = _deterministic_subset(c_nm, N_CAL, SPLIT_SEED + N_CAL + 1000)
    backbone = np.asarray(shadow["shadow_gate"], dtype=np.float64)
    scores = {"shadow_k2": backbone}
    standardized_backbone = _standardize(backbone, reference)
    for method in ACTIVE_METHODS:
        active_score = np.asarray(active[f"{method}_active"], dtype=np.float64)
        scores[f"shadow_plus_{method}"] = standardized_backbone + 0.25 * _standardize(
            active_score, reference
        )
    scores["old_backbone_plus_full_ai_one_shot"] = np.asarray(
        active["full_ai_one_shot_fusion"], dtype=np.float64
    )
    return {
        name: membership_metrics(value, labels, calibration, partitions["T"])
        for name, value in scores.items()
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(rows[0]["metrics"])
    metrics = {}
    for method in methods:
        metrics[method] = {
            "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
            "pauc_0_10": float(
                np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])
            ),
            "tpr_1": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows]
                )
            ),
            "actual_fpr_1": float(
                np.mean(
                    [
                        row["metrics"][method]["tpr_at_fpr"]["1%"]["actual_fpr"]
                        for row in rows
                    ]
                )
            ),
            "tpr_10": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["10%"]["tpr"] for row in rows]
                )
            ),
        }
    baseline = "shadow_plus_uniform"
    deltas = {
        method: {
            "pauc_0_10": float(
                np.mean(
                    [
                        row["metrics"][method]["pauc_0_10"]
                        - row["metrics"][baseline]["pauc_0_10"]
                        for row in rows
                    ]
                )
            ),
            "condition_seed_wins": int(
                np.sum(
                    [
                        row["metrics"][method]["pauc_0_10"]
                        > row["metrics"][baseline]["pauc_0_10"]
                        for row in rows
                    ]
                )
            ),
            "condition_seed_total": len(rows),
        }
        for method in methods
        if method != baseline
    }
    return {"rows": len(rows), "metrics": metrics, "pauc_deltas_vs_shadow_uniform": deltas}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Shadow-Gate and Active-Query Combination",
        "",
        "| Method | AUC | pAUC | TPR@1% / FPR | TPR@10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, value in report["metrics"].items():
        lines.append(
            f"| `{method}` | {value['auc']:.4f} | {value['pauc_0_10']:.4f} | "
            f"{value['tpr_1']:.4f} / {value['actual_fpr_1']:.4f} | {value['tpr_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Method vs shadow+uniform | Delta pAUC | Wins |",
            "|---|---:|---:|",
        ]
    )
    for method, value in report["pauc_deltas_vs_shadow_uniform"].items():
        lines.append(
            f"| `{method}` | {value['pauc_0_10']:+.4f} | "
            f"{value['condition_seed_wins']}/{value['condition_seed_total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/shadow_scale_gate",
    )
    parser.add_argument(
        "--active-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/full_ai_allocation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/shadow_active_combination",
    )
    args = parser.parse_args()
    shadow_paths = sorted((args.shadow_dir / "conditions").glob("*/scores_seed_*.npz"))
    rows = []
    for shadow_path in shadow_paths:
        condition = shadow_path.parent.name
        active_path = args.active_dir / "conditions" / condition / shadow_path.name
        if not active_path.exists():
            raise FileNotFoundError(active_path)
        metrics = evaluate_pair(shadow_path, active_path)
        rows.append(
            {
                "condition": condition,
                "seed": int(shadow_path.stem.rsplit("_", 1)[1]),
                "metrics": metrics,
            }
        )
    report = aggregate(rows)
    report["condition_rows"] = rows
    output = args.output_dir.resolve()
    _write_json(output / "AGGREGATE.json", report)
    write_markdown(report, output / "AGGREGATE.md")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
