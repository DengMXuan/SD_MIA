"""Aggregate the complete Qwen3-8B baseline matrix into a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import METHODS


CONDITIONS = (
    ("arxivtection", 1),
    ("arxivtection", 3),
    ("newstection", 1),
    ("newstection", 3),
    ("wikitection", 1),
    ("wikitection", 3),
)
DISPLAY_NAMES = {
    "loss": "Loss",
    "min_k_prob": "Min-K% Prob",
    "min_k_pp": "Min-K%++",
    "recall": "ReCaLL",
    "icp_mia": "ICP-MIA",
    "petal": "PETAL",
    "sead": "SEAD",
    "ws": "WS",
    "rs": "RS",
    "bt": "BT",
    "samia": "SaMIA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def _condition_key(benchmark: str, epoch: int) -> str:
    return f"{benchmark}_epoch{epoch}"


def _load_condition(results_root: Path, benchmark: str, epoch: int) -> dict[str, Any]:
    condition = _condition_key(benchmark, epoch)
    directory = results_root / f"{condition}_parallel"
    metrics_path = directory / "baseline_metrics.json"
    scores_path = directory / "baseline_scores.npz"
    if not metrics_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"incomplete result for {condition}: {directory}")
    artifact = json.loads(metrics_path.read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    metrics = artifact["metrics"]
    expected = set(METHODS)
    if set(metrics) != expected:
        raise ValueError(f"{condition} methods differ: {sorted(set(metrics) ^ expected)}")
    scores = np.load(scores_path, allow_pickle=False)
    if set(scores.files) != {"labels", "record_ids", *METHODS}:
        raise ValueError(f"unexpected arrays in {scores_path}: {scores.files}")
    labels = np.asarray(scores["labels"])
    if len(labels) != 4000 or int(np.sum(labels == 1)) != 2000 or int(np.sum(labels == 0)) != 2000:
        raise ValueError(f"invalid labels in {scores_path}")
    observed_direction: dict[str, bool] = {}
    for method in METHODS:
        values = np.asarray(scores[method])
        if len(values) != len(labels):
            raise ValueError(f"{condition}/{method} has {len(values)} scores")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{condition}/{method} contains non-finite scores")
        observed_direction[method] = bool(
            float(np.mean(values[labels == 1]))
            > float(np.mean(values[labels == 0]))
        )
    return {
        "condition": condition,
        "benchmark": benchmark,
        "epoch": epoch,
        "directory": str(directory),
        "protocol": protocol,
        "metrics": metrics,
        "validated": {
            "n_scores": len(labels),
            "n_member": int(np.sum(labels == 1)),
            "n_nonmember": int(np.sum(labels == 0)),
            "member_mean_greater_than_nonmember_mean": observed_direction,
        },
    }


def _metric_table(conditions: list[dict[str, Any]], key: str) -> list[str]:
    headers = ["Condition", *(DISPLAY_NAMES[method] for method in METHODS)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for condition in conditions:
        values = [
            f"{condition['metrics'][method][key]:.6f}" for method in METHODS
        ]
        lines.append("| " + " | ".join([condition["condition"], *values]) + " |")
    return lines


def main() -> None:
    args = parse_args()
    conditions = [
        _load_condition(args.results_root, benchmark, epoch)
        for benchmark, epoch in CONDITIONS
    ]
    summary = []
    for method in METHODS:
        values = [condition["metrics"][method] for condition in conditions]
        summary.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "mean_auc": float(np.mean([value["auc"] for value in values])),
                "mean_tpr@10%fpr": float(
                    np.mean([value["tpr@10%fpr"] for value in values])
                ),
                "mean_tpr@1%fpr": float(
                    np.mean([value["tpr@1%fpr"] for value in values])
                ),
            }
        )
    summary.sort(key=lambda value: value["mean_auc"], reverse=True)

    report = {
        "benchmark_matrix": "3 datasets x epoch 1/3",
        "score_direction": "larger means more likely member",
        "methods": list(METHODS),
        "display_names": DISPLAY_NAMES,
        "conditions": conditions,
        "mean_ranking": summary,
        "restrictions": {
            "reference_model": False,
            "unfinetuned_target_scored": False,
            "draft_model_loaded": False,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Qwen3-8B Fine-tuning Membership-Inference Baseline Report",
        "",
        "## Executive summary",
        "",
        "本报告覆盖 WikiTection、NewsTection、ArXivTection 三个数据集，以及 epoch 1/3 两个已微调 Qwen3-8B target，共 6 个 target × 11 个 baseline。每个条件使用 2,000 members、2,000 non-members 和 2,000 auxiliary records。",
        "",
        "所有结果统一采用 member-positive 方向：分数越大，越可能属于 member。",
        "",
        "## Cross-condition mean ranking",
        "",
        "| Rank | Method | Mean AUC | Mean TPR@10%FPR | Mean TPR@1%FPR |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, value in enumerate(summary, 1):
        lines.append(
            f"| {rank} | {value['display_name']} | {value['mean_auc']:.6f} | "
            f"{value['mean_tpr@10%fpr']:.6f} | {value['mean_tpr@1%fpr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## AUC by condition",
            "",
            *_metric_table(conditions, "auc"),
            "",
            "## TPR@10%FPR by condition",
            "",
            *_metric_table(conditions, "tpr@10%fpr"),
            "",
            "## TPR@1%FPR by condition",
            "",
            *_metric_table(conditions, "tpr@1%fpr"),
            "",
            "## Protocol and interpretation",
            "",
            "- All six runs score only the saved fine-tuned target checkpoint for that condition. No reference model, unfine-tuned target, or draft model is loaded or scored.",
            "- Loss, Min-K% Prob, Min-K%++, and PETAL use their member-positive definitions without sign inversion. The machine-readable report records the observed member/non-member mean ordering for every condition; low-performing methods are retained rather than discarded.",
            "- TPR uses a threshold from the upper tail of non-member scores with strict `>` comparison. With 2,000 non-members, the realized FPR is 0.0995 at the nominal 10% point and 0.0095 at the nominal 1% point, except where ties make it lower.",
            "- SEAD uses 50 target-only Monte Carlo samples per token. SaMIA uses 10 target-only generation probes. ICP-MIA uses the target's own input-embedding space for auxiliary retrieval, and PETAL calibrates on auxiliary records scored by the same fine-tuned target.",
            "- ArXivTection uses SDPA attention and generation batch size 8 to fit its 2,048-token responses; the other conditions use the same SDPA/chunked implementation and batch size for comparability.",
            "",
            "## Reproducibility artifacts",
            "",
            f"- Results root: `{args.results_root}`",
            f"- Machine-readable report: `{args.json}`",
            "- Each condition directory contains `baseline_metrics.json` and `baseline_scores.npz`.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "json": str(args.json)}))


if __name__ == "__main__":
    main()
