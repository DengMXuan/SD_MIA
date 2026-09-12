"""Conformal evaluation and reporting for the frozen M2 method matrix.

Reads ``frozen_scores.npz`` produced by :mod:`m2_fit`, calibrates every
method on the C-partition nonmembers with the registered inclusive-tie
split-conformal rule, and reports the T-partition metrics once:

- test AUC and pAUC[0, 0.05];
- TPR / actual FPR at eta in {10%, 5%, 1%} with Wilson binomial intervals and
  the one-sided 95% upper bound for zero false positives;
- 2000 calibration/test resamples shared across methods;
- paired record-bootstrap deltas for the preregistered comparisons;
- per-condition and aggregate Markdown reports.

Usage:

    uv run --no-sync python -m experiments.sd_membership_sft.m2_evaluate \
        --benchmark wikitection --epoch 1 --role draft_auxiliary_distilled
    uv run --no-sync python -m experiments.sd_membership_sft.m2_evaluate --aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .directional_mia import (
    conformal_tail_pvalues,
    paired_bootstrap_method_delta,
    rank_auc,
)
from .m1_fit import _wilson_interval, make_partitions, partial_auc
from .scoring_common import ROOT

ETA_RATES = (0.10, 0.05, 0.01)
BOOTSTRAP_REPEATS = 2000
PAIR_SEED = 20260919
RESAMPLE_SEED = 20260929

MAIN_ORDER = (
    [f"B0__{name}" for name in ("p_mean_logp", "mean_abs_delta", "window_sign_16", "window_sign_multiscale")]
    + ["B1", "B2", "B2_mlp", "BQ", "H_only", "Direct", "M2_F"]
    + ["BQ_noM", "M2_F_noM", "M2_F_nodiff", "M2_F_plus", "M2_F_minus"]
    + ["P_G_zero", "P_G_wide", "M2_U", "M2_G", "M2_G_T2"]
)
MECHANISM_FAMILIES = ("M2_F_wshuffle", "M2_F_hshuffle", "M2_F_noise")
PAIRED_PAIRS = (
    ("M2_F", "Direct"),
    ("M2_F", "B2"),
    ("M2_G", "P_G_zero"),
    ("M2_G", "P_G_wide"),
    ("M2_G", "M2_U"),
    ("M2_G", "B2"),
    ("M2_U", "P_G_zero"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--role", default="draft_auxiliary_distilled")
    parser.add_argument("--condition-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def condition_dir(benchmark: str, epoch: int, role: str) -> Path:
    return (
        ROOT
        / "experiments/results/sft_runs/m2_activation_pooling"
        / f"{benchmark}_epoch{epoch}"
        / role
    )


def _zero_fp_bound(nonmember_n: int) -> float:
    return float(1.0 - 0.05 ** (1.0 / nonmember_n)) if nonmember_n > 0 else float("nan")


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    partitions: Any,
    bootstrap_indices: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    calibration = partitions["calibration"]
    test = partitions["test"]
    cal_nonmember = calibration[labels[calibration] == 0]
    test_member = test[labels[test] == 1]
    test_nonmember = test[labels[test] == 0]
    member_scores = scores[test_member]
    nonmember_scores = scores[test_nonmember]
    cal_scores = scores[cal_nonmember]

    result: dict[str, Any] = {
        "test_auc": float(rank_auc(member_scores, nonmember_scores)),
        "test_pauc_0_05": float(
            partial_auc(
                np.concatenate((member_scores, nonmember_scores)),
                np.concatenate(
                    (
                        np.ones(len(member_scores), dtype=np.int64),
                        np.zeros(len(nonmember_scores), dtype=np.int64),
                    )
                ),
            )
        ),
        "calibrated": {},
    }
    for eta in ETA_RATES:
        p_values_member = conformal_tail_pvalues(member_scores, cal_scores)
        p_values_nonmember = conformal_tail_pvalues(nonmember_scores, cal_scores)
        member_hits = int(np.sum(p_values_member <= eta))
        nonmember_hits = int(np.sum(p_values_nonmember <= eta))
        tpr_low, tpr_high = _wilson_interval(member_hits, len(member_scores))
        fpr_low, fpr_high = _wilson_interval(nonmember_hits, len(nonmember_scores))
        result["calibrated"][f"{int(eta * 100)}%"] = {
            "tpr": member_hits / len(member_scores),
            "fpr": nonmember_hits / len(nonmember_scores),
            "member_hits": member_hits,
            "member_n": int(len(member_scores)),
            "nonmember_hits": nonmember_hits,
            "nonmember_n": int(len(nonmember_scores)),
            "tpr_ci95_wilson": [tpr_low, tpr_high],
            "fpr_ci95_wilson": [fpr_low, fpr_high],
            "zero_fp_one_sided95_upper": (
                _zero_fp_bound(len(nonmember_scores)) if nonmember_hits == 0 else None
            ),
        }

    if bootstrap_indices is not None:
        cal_resamples = bootstrap_indices["calibration_nonmember"]
        member_resamples = bootstrap_indices["test_member"]
        nonmember_resamples = bootstrap_indices["test_nonmember"]
        cal_sorted_all = np.sort(cal_scores[cal_resamples], axis=1)
        member_sampled = member_scores[member_resamples]
        nonmember_sampled = nonmember_scores[nonmember_resamples]
        boot: dict[str, dict[str, list[float]]] = {
            f"{int(eta * 100)}%": {"tpr": [], "fpr": []} for eta in ETA_RATES
        }
        n_cal = cal_sorted_all.shape[1]
        for repeat in range(cal_sorted_all.shape[0]):
            ordered = cal_sorted_all[repeat]
            member_p = (1.0 + n_cal - np.searchsorted(ordered, member_sampled[repeat], side="left")) / (n_cal + 1.0)
            nonmember_p = (1.0 + n_cal - np.searchsorted(ordered, nonmember_sampled[repeat], side="left")) / (n_cal + 1.0)
            for eta in ETA_RATES:
                boot[f"{int(eta * 100)}%"]["tpr"].append(float(np.mean(member_p <= eta)))
                boot[f"{int(eta * 100)}%"]["fpr"].append(float(np.mean(nonmember_p <= eta)))
        result["bootstrap"] = {
            rate: {
                "tpr_ci95": [
                    float(np.quantile(values["tpr"], 0.025)),
                    float(np.quantile(values["tpr"], 0.975)),
                ],
                "fpr_ci95": [
                    float(np.quantile(values["fpr"], 0.025)),
                    float(np.quantile(values["fpr"], 0.975)),
                ],
            }
            for rate, values in boot.items()
        }
    return result


def make_bootstrap_indices(
    n_cal: int, n_member: int, n_nonmember: int, repeats: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "calibration_nonmember": rng.integers(0, n_cal, size=(repeats, n_cal)),
        "test_member": rng.integers(0, n_member, size=(repeats, n_member)),
        "test_nonmember": rng.integers(0, n_nonmember, size=(repeats, n_nonmember)),
    }


def evaluate_condition(directory: Path, bootstrap_repeats: int) -> dict[str, Any]:
    score_path = directory / "frozen_scores.npz"
    artifacts_path = directory / "m2_fit_artifacts.json"
    if not score_path.exists() or not artifacts_path.exists():
        raise FileNotFoundError(f"incomplete M2 condition under {directory}")
    data = np.load(score_path, allow_pickle=False)
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    labels = np.asarray(data["labels"], dtype=np.int64)
    feature_dir = Path(artifacts["protocol"]["feature_dir"])
    partitions = make_partitions(
        labels,
        np.asarray(data["record_ids"]),
        frozen_manifest_path=feature_dir / "partition_manifest.json",
    )
    calibration = partitions["calibration"]
    test = partitions["test"]
    bootstrap_indices = make_bootstrap_indices(
        int(np.sum(labels[calibration] == 0)),
        int(np.sum(labels[test] == 1)),
        int(np.sum(labels[test] == 0)),
        bootstrap_repeats,
        RESAMPLE_SEED,
    )

    methods: dict[str, Any] = {}
    for key in data.files:
        if key in ("labels", "record_ids"):
            continue
        methods[key] = evaluate_scores(
            np.asarray(data[key], dtype=np.float64), labels, partitions, bootstrap_indices
        )

    paired: dict[str, Any] = {}
    for first, second in PAIRED_PAIRS:
        if first in data.files and second in data.files:
            paired[f"{first}_vs_{second}"] = paired_bootstrap_method_delta(
                np.asarray(data[first], dtype=np.float64),
                np.asarray(data[second], dtype=np.float64),
                labels,
                partitions,
                ETA_RATES,
                bootstrap_repeats,
                PAIR_SEED,
            )

    return {
        "protocol": artifacts["protocol"],
        "methods": methods,
        "paired_deltas": paired,
        "fit_metadata": artifacts.get("methods", {}),
        "bootstrap_repeats": bootstrap_repeats,
    }


def _format_point(value: float) -> str:
    return f"{value:.4f}"


def render_condition_report(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    methods = report["methods"]
    lines = [
        "# M2 condition report",
        "",
        f"- Benchmark: `{protocol['benchmark']}`; epoch: `{protocol['epoch']}`; role: `{protocol['role']}`",
        f"- Records: {protocol['n_member']} member + {protocol['n_nonmember']} nonmember; "
        "conformal calibration on the 400 C nonmembers, test reported once; "
        f"{report['bootstrap_repeats']} shared resamples.",
        "- Conformal rule: `(1 + count(C0 score >= s)) / (n+1) <= eta`, inclusive ties; "
        "raw logits to avoid sigmoid saturation ties.",
        "- delta scale: "
        f"{protocol['delta_scale']:.6g}",
        "",
        "| Method | Test AUC | Test pAUC | TPR@10% (FPR) | TPR@5% (FPR) | TPR@1% (FPR) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    shown = [name for name in MAIN_ORDER if name in methods]
    for name in shown:
        result = methods[name]
        row = [
            f"`{name}`",
            f"{result['test_auc']:.4f}",
            f"{result['test_pauc_0_05']:.4f}",
        ]
        for rate in ("10%", "5%", "1%"):
            point = result["calibrated"][rate]
            row.append(f"{point['tpr']:.4f} ({point['fpr']:.4f})")
        lines.append("| " + " | ".join(row) + " |")
    for family in MECHANISM_FAMILIES:
        seeds = [name for name in methods if name.startswith(family)]
        if not seeds:
            continue
        row = [
            f"`{family}` (5 seeds)",
            f"{np.mean([methods[name]['test_auc'] for name in seeds]):.4f}",
            f"{np.mean([methods[name]['test_pauc_0_05'] for name in seeds]):.4f}",
        ]
        for rate in ("10%", "5%", "1%"):
            tprs = [methods[name]["calibrated"][rate]["tpr"] for name in seeds]
            fprs = [methods[name]["calibrated"][rate]["fpr"] for name in seeds]
            row.append(f"{np.mean(tprs):.4f} [{np.min(tprs):.3f},{np.max(tprs):.3f}] ({np.mean(fprs):.4f})")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Preregistered paired deltas", "", "| Comparison | ΔAUC | ΔTPR@1% | ΔFPR@1% |", "|---|---:|---:|---:|"])
    for comparison, result in report["paired_deltas"].items():
        tpr = result["tpr_at_calibrated_fpr"]["1%"]["tpr_delta"]
        fpr = result["tpr_at_calibrated_fpr"]["1%"]["fpr_delta"]
        auc = result["auc"]
        lines.append(
            f"| `{comparison}` | {auc['point']:.4f} [{auc['ci95_low']:.4f}, {auc['ci95_high']:.4f}] "
            f"| {tpr['point']:.4f} [{tpr['ci95_low']:.4f}, {tpr['ci95_high']:.4f}] "
            f"| {fpr['point']:.4f} [{fpr['ci95_low']:.4f}, {fpr['ci95_high']:.4f}] |"
        )

    lines.extend(["", "## Shared-resample 95% intervals at 1% FPR", "", "| Method | TPR@1% | actual FPR |", "|---|---:|---:|"])
    for name in shown:
        result = methods[name]
        boot = result.get("bootstrap", {}).get("1%")
        point = result["calibrated"]["1%"]
        if boot is None:
            continue
        lines.append(
            f"| `{name}` | {point['tpr']:.4f} [{boot['tpr_ci95'][0]:.4f}, {boot['tpr_ci95'][1]:.4f}] "
            f"| {point['fpr']:.4f} [{boot['fpr_ci95'][0]:.4f}, {boot['fpr_ci95'][1]:.4f}] |"
        )
    lines.append("")
    return "\n".join(lines)


def render_aggregate(reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# M2 aggregate report (four P0 conditions, q_aux)",
        "",
        "Independent M2-F / M2-G with the registered control matrix; conformal "
        "calibration on C nonmembers; test reported once per condition.",
        "",
        "## Test TPR@1% (actual FPR) by condition",
        "",
    ]
    condition_names = list(reports)
    header = "| Method | " + " | ".join(condition_names) + " |"
    lines.append(header)
    lines.append("|---" * (len(condition_names) + 1) + "|")
    shown = [name for name in MAIN_ORDER if all(name in reports[c]["methods"] for c in condition_names)]
    for name in shown:
        cells = [f"`{name}`"]
        for condition in condition_names:
            point = reports[condition]["methods"][name]["calibrated"]["1%"]
            cells.append(f"{point['tpr']:.4f} ({point['fpr']:.4f})")
        lines.append(" | ".join(cells) + " |")
    for family in MECHANISM_FAMILIES:
        if all(
            any(name.startswith(family) for name in reports[c]["methods"]) for c in condition_names
        ):
            cells = [f"`{family}` (5-seed mean)"]
            for condition in condition_names:
                seeds = [name for name in reports[condition]["methods"] if name.startswith(family)]
                tpr = np.mean([reports[condition]["methods"][name]["calibrated"]["1%"]["tpr"] for name in seeds])
                fpr = np.mean([reports[condition]["methods"][name]["calibrated"]["1%"]["fpr"] for name in seeds])
                cells.append(f"{tpr:.4f} ({fpr:.4f})")
            lines.append(" | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Development gate: validation macro pAUC[0, 0.05]",
            "",
            "Average over the four conditions of the V pAUC of the selected config.",
            "",
            "| Method | " + " | ".join(condition_names) + " | macro mean |",
            "|---" * (len(condition_names) + 2) + "|",
        ]
    )
    gate_methods = ["B2", "B2_mlp", "BQ", "Direct", "M2_F", "P_G_zero", "P_G_wide", "M2_U", "M2_G", "M2_G_T2"]
    for name in gate_methods:
        values = []
        cells = [f"`{name}`"]
        for condition in condition_names:
            meta = reports[condition]["fit_metadata"].get(name)
            if meta is None or "selected" not in meta:
                cells.append("-")
                continue
            value = meta["selected"]["validation_pauc_0_05"]
            values.append(value)
            cells.append(f"{value:.4f}")
        cells.append(f"{np.mean(values):.4f}" if values else "-")
        lines.append(" | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Preregistered paired deltas (ΔTPR@1% point [95% CI])",
            "",
            "| Comparison | " + " | ".join(condition_names) + " |",
            "|---" * (len(condition_names) + 1) + "|",
        ]
    )
    for comparison in PAIRED_PAIRS:
        key = f"{comparison[0]}_vs_{comparison[1]}"
        cells = [f"`{key}`"]
        for condition in condition_names:
            result = reports[condition]["paired_deltas"].get(key)
            if result is None:
                cells.append("-")
                continue
            delta = result["tpr_at_calibrated_fpr"]["1%"]["tpr_delta"]
            cells.append(f"{delta['point']:+.4f} [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}]")
        lines.append(" | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.aggregate:
        root = ROOT / "experiments/results/sft_runs/m2_activation_pooling"
        reports: dict[str, dict[str, Any]] = {}
        for condition_path in sorted(root.glob("*_epoch*")):
            for role_path in sorted(condition_path.iterdir()):
                if not role_path.is_dir():
                    continue
                if not (role_path / "frozen_scores.npz").exists():
                    continue
                key = f"{condition_path.name}/{role_path.name}"
                reports[key] = evaluate_condition(role_path, args.bootstrap_repeats)
        if not reports:
            raise RuntimeError("no completed M2 conditions found")
        (root / "m2_metrics_aggregate.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (root / "M2_RESULTS.md").write_text(render_aggregate(reports), encoding="utf-8")
        print(json.dumps({"aggregate_conditions": list(reports)}, indent=2))
        return

    if args.condition_dir is not None:
        directory = args.condition_dir if args.condition_dir.is_absolute() else ROOT / args.condition_dir
    else:
        if not args.benchmark or args.epoch is None:
            raise SystemExit("--benchmark/--epoch required without --condition-dir")
        directory = condition_dir(args.benchmark, args.epoch, args.role)
    report = evaluate_condition(directory, args.bootstrap_repeats)
    (directory / "m2_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "M2_RESULTS.md").write_text(render_condition_report(report), encoding="utf-8")
    print(json.dumps({"condition_dir": str(directory)}, indent=2))


if __name__ == "__main__":
    main()
