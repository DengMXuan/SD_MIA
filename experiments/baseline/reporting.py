"""Render standalone baseline scores and explicit physical/standalone costs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from experiments.baseline.costs import cost_table
from experiments.baseline.methods import rank_auc, upper_tail_tpr

def render_report(
    output_dir: Path,
    protocol: dict[str, Any],
    scores: dict[str, list[float]],
    labels: np.ndarray,
    costs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    members = labels == 1
    nonmembers = labels == 0
    metrics: dict[str, Any] = {}
    for name, values in scores.items():
        values_array = np.asarray(values, dtype=np.float64)
        member = values_array[members]
        nonmember = values_array[nonmembers]
        tpr10, fpr10, threshold10 = upper_tail_tpr(member, nonmember, 0.10)
        tpr01, fpr01, threshold01 = upper_tail_tpr(member, nonmember, 0.01)
        metrics[name] = {
            "auc": rank_auc(member, nonmember),
            "tpr@10%fpr": tpr10,
            "actual_fpr@10%": fpr10,
            "threshold@10%": threshold10,
            "tpr@1%fpr": tpr01,
            "actual_fpr@1%": fpr01,
            "threshold@1%": threshold01,
            "member_mean": float(np.mean(member)),
            "nonmember_mean": float(np.mean(nonmember)),
        }
    artifact = {"protocol": protocol, "metrics": metrics, "scores": scores}
    if costs is not None:
        artifact["costs"] = costs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Target-only membership-inference baselines",
        "",
        f"- Run: `{protocol['run_dir']}`",
        "- Model source: pretrained target checkpoint." if protocol.get("training_regime") == "pretraining" else "- Model source: saved fine-tuned target checkpoint only.",
        "- Reference model: none. Draft model: not loaded.",
        "",
        "| Method | AUC | TPR@10%FPR | actual FPR | TPR@1%FPR | actual FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        lines.append(
            f"| `{name}` | {result['auc']:.4f} | {result['tpr@10%fpr']:.4f} "
            f"| {result['actual_fpr@10%']:.4f} | {result['tpr@1%fpr']:.4f} "
            f"| {result['actual_fpr@1%']:.4f} |"
        )
    lines.extend(cost_table(costs))
    (output_dir / "BASELINE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact
