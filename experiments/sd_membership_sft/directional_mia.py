"""Offline directional/window membership metrics from saved p/q log-probabilities.

The model-facing pass is intentionally kept in ``pq_gap_mia``.  This module
reuses its exact teacher-forced response-token log-probabilities and evaluates
the preregistered directional and local-window scores from the 2026-09-08
design note.  It never refits the target/draft models or changes the stored
member assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOWS = (4, 8, 16, 32, 64)


def rank_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    values = np.concatenate([member, nonmember])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = ranks[inverse]
    n_member = len(member)
    return float(
        (ranks[:n_member].sum() - n_member * (n_member + 1) / 2.0)
        / (n_member * len(nonmember))
    )


def threshold_metrics(
    member: np.ndarray,
    nonmember: np.ndarray,
    fpr: float,
    threshold: float | None = None,
    calibration_nonmember: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a conservative split-conformal upper-tail rule.

    The calibration p-value is ``(1 + # {z: z >= score}) / (n + 1)``.  This
    deliberately counts ties in the nonmember tail.  The corresponding
    order-statistic boundary is retained for reporting, but hits are computed
    from the p-value so ties can never be broken to manufacture the requested
    FPR.
    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    calibration = nonmember if calibration_nonmember is None else calibration_nonmember
    if len(calibration) == 0:
        raise ValueError("calibration_nonmember must not be empty")
    if threshold is None:
        threshold = order_statistic_threshold(calibration, fpr)
    member_pvalues = conformal_tail_pvalues(member, calibration)
    nonmember_pvalues = conformal_tail_pvalues(nonmember, calibration)
    member_hits = int(np.sum(member_pvalues <= fpr))
    nonmember_hits = int(np.sum(nonmember_pvalues <= fpr))
    return {
        "threshold": float(threshold),
        "tpr": float(member_hits / len(member)),
        "fpr": float(nonmember_hits / len(nonmember)),
        "member_hits": member_hits,
        "member_n": int(len(member)),
        "nonmember_hits": nonmember_hits,
        "nonmember_n": int(len(nonmember)),
        "calibration_n_nonmember": int(len(calibration)),
        "threshold_rule": "(1 + count(calibration_score >= score))/(n+1) <= target_fpr",
    }


def conformal_tail_pvalues(
    scores: np.ndarray, calibration_nonmember: np.ndarray
) -> np.ndarray:
    """Return conservative upper-tail p-values with inclusive tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    calibration_nonmember = np.asarray(calibration_nonmember, dtype=np.float64)
    if calibration_nonmember.size == 0:
        raise ValueError("calibration_nonmember must not be empty")
    # With sorted calibration values, ``searchsorted(..., side='left')`` gives
    # the number of calibration values strictly below each score.  Subtracting
    # from n therefore counts values ``>= score`` exactly, including ties,
    # without constructing an O(n_scores * n_calibration) matrix.
    ordered = np.sort(calibration_nonmember)
    count_ge = len(ordered) - np.searchsorted(ordered, scores, side="left")
    return (1.0 + count_ge) / (len(calibration_nonmember) + 1.0)


def order_statistic_threshold(
    calibration_nonmember: np.ndarray, fpr: float
) -> float:
    """Boundary for the inclusive-tie conformal rule.

    The actual decision is p-value <= fpr.  For distinct values it is
    equivalent to ``score > threshold`` where threshold is the (k+1)-th largest
    calibration score; using p-values above also handles ties correctly.
    """
    calibration_nonmember = np.asarray(calibration_nonmember, dtype=np.float64)
    if not 0.0 < fpr < 1.0 or calibration_nonmember.size == 0:
        raise ValueError("invalid FPR or empty calibration set")
    max_calibration_tail = int(np.ceil(fpr * (len(calibration_nonmember) + 1.0))) - 1
    order_from_largest = max(1, max_calibration_tail + 1)
    ordered = np.sort(calibration_nonmember)[::-1]
    if order_from_largest > len(ordered):
        return float("-inf")
    return float(ordered[order_from_largest - 1])


def bootstrap_auc(
    member: np.ndarray,
    nonmember: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        m = member[rng.integers(0, len(member), size=len(member))]
        n = nonmember[rng.integers(0, len(nonmember), size=len(nonmember))]
        values[index] = rank_auc(m, n)
    return {
        "point": rank_auc(member, nonmember),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _ci_summary(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def bootstrap_operating_points(
    member: np.ndarray,
    nonmember: np.ndarray,
    calibration_nonmember: np.ndarray,
    rates: tuple[float, ...],
    repeats: int,
    seed: int,
    calibration_is_nonmember: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Bootstrap TPR and actual FPR, resampling records not tokens."""
    rng = np.random.default_rng(seed)
    samples = {
        rate: {"tpr": np.empty(repeats), "fpr": np.empty(repeats)}
        for rate in rates
    }
    for repeat in range(repeats):
        member_sample = member[rng.integers(0, len(member), size=len(member))]
        nonmember_indices = rng.integers(0, len(nonmember), size=len(nonmember))
        nonmember_sample = nonmember[nonmember_indices]
        if calibration_is_nonmember:
            calibration_sample = nonmember_sample
        else:
            calibration_sample = calibration_nonmember[
                rng.integers(
                    0, len(calibration_nonmember), size=len(calibration_nonmember)
                )
            ]
        for rate in rates:
            point = threshold_metrics(
                member_sample,
                nonmember_sample,
                rate,
                calibration_nonmember=calibration_sample,
            )
            samples[rate]["tpr"][repeat] = point["tpr"]
            samples[rate]["fpr"][repeat] = point["fpr"]
    return {
        f"{int(rate * 100)}%": {
            "tpr": _ci_summary(
                threshold_metrics(
                    member,
                    nonmember,
                    rate,
                    calibration_nonmember=calibration_nonmember,
                )["tpr"],
                values["tpr"],
            ),
            "fpr": _ci_summary(
                threshold_metrics(
                    member,
                    nonmember,
                    rate,
                    calibration_nonmember=calibration_nonmember,
                )["fpr"],
                values["fpr"],
            ),
        }
        for rate, values in samples.items()
    }


def _delta_summary(point: float, values: np.ndarray) -> dict[str, float]:
    return _ci_summary(point, values)


def paired_bootstrap_method_delta(
    values: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    rates: tuple[float, ...],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Paired record bootstrap for calibrated-test method increments."""
    calibration = partitions["calibration"]
    test = partitions["test"]
    calibration_nonmember = calibration[labels[calibration] == 0]
    test_member = test[labels[test] == 1]
    test_nonmember = test[labels[test] == 0]
    point_auc = rank_auc(values[test_member], values[test_nonmember])
    point_baseline_auc = rank_auc(baseline[test_member], baseline[test_nonmember])
    auc_deltas = np.empty(repeats)
    tpr_deltas = {rate: np.empty(repeats) for rate in rates}
    fpr_deltas = {rate: np.empty(repeats) for rate in rates}
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        member_indices = rng.integers(0, len(test_member), size=len(test_member))
        nonmember_indices = rng.integers(0, len(test_nonmember), size=len(test_nonmember))
        calibration_indices = rng.integers(
            0, len(calibration_nonmember), size=len(calibration_nonmember)
        )
        member_a = values[test_member][member_indices]
        member_b = baseline[test_member][member_indices]
        nonmember_a = values[test_nonmember][nonmember_indices]
        nonmember_b = baseline[test_nonmember][nonmember_indices]
        calibration_a = values[calibration_nonmember][calibration_indices]
        calibration_b = baseline[calibration_nonmember][calibration_indices]
        auc_deltas[repeat] = rank_auc(member_a, nonmember_a) - rank_auc(
            member_b, nonmember_b
        )
        for rate in rates:
            point_a = threshold_metrics(
                member_a, nonmember_a, rate, calibration_nonmember=calibration_a
            )
            point_b = threshold_metrics(
                member_b, nonmember_b, rate, calibration_nonmember=calibration_b
            )
            tpr_deltas[rate][repeat] = point_a["tpr"] - point_b["tpr"]
            fpr_deltas[rate][repeat] = point_a["fpr"] - point_b["fpr"]
    result: dict[str, Any] = {
        "auc": _delta_summary(point_auc - point_baseline_auc, auc_deltas),
        "tpr_at_calibrated_fpr": {},
    }
    for rate in rates:
        point_a = threshold_metrics(
            values[test_member], values[test_nonmember], rate,
            calibration_nonmember=values[calibration_nonmember],
        )
        point_b = threshold_metrics(
            baseline[test_member], baseline[test_nonmember], rate,
            calibration_nonmember=baseline[calibration_nonmember],
        )
        result["tpr_at_calibrated_fpr"][f"{int(rate * 100)}%"] = {
            "tpr_delta": _delta_summary(
                point_a["tpr"] - point_b["tpr"], tpr_deltas[rate]
            ),
            "fpr_delta": _delta_summary(
                point_a["fpr"] - point_b["fpr"], fpr_deltas[rate]
            ),
        }
    return result


def window_sign(delta: np.ndarray, width: int) -> float:
    if len(delta) < width:
        return float(np.mean(delta > 0.0))
    sums = np.convolve(delta, np.ones(width, dtype=np.float64), mode="valid")
    return float(np.mean(sums > 0.0))


def record_features(
    target: np.ndarray,
    draft: np.ndarray,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> dict[str, float]:
    delta = target - draft
    features = {
        "p_mean_logp": float(np.mean(target)),
        "q_mean_logq": float(np.mean(draft)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "mean_signed_delta": float(np.mean(delta)),
        "mean_negative_delta": float(np.mean(np.minimum(delta, 0.0))),
        "mean_positive_delta": float(np.mean(np.maximum(delta, 0.0))),
        "positive_fraction": float(np.mean(delta > 0.0)),
        "mean_alpha": float(np.mean(np.clip(np.exp(delta), 0.0, 1.0))),
        "median_delta": float(np.median(delta)),
        "delta_q10": float(np.quantile(delta, 0.10)),
        "delta_q25": float(np.quantile(delta, 0.25)),
        "delta_q75": float(np.quantile(delta, 0.75)),
        "delta_q90": float(np.quantile(delta, 0.90)),
    }
    for width in windows:
        features[f"window_sign_{width}"] = window_sign(delta, width)
    features["window_sign_multiscale"] = float(
        np.mean([features[f"window_sign_{width}"] for width in windows])
    )
    return features


def load_records(
    path: Path, scores_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    if scores_path is None:
        scores_path = path.with_name("pq_gap_scores.npz")
    score_data = np.load(scores_path, allow_pickle=False)
    labels = np.asarray(score_data["labels"], dtype=np.int64)
    lengths = np.asarray(data["lengths"], dtype=np.int64)
    target = np.asarray(data["target"], dtype=np.float64)
    roles = {
        key: np.asarray(data[key], dtype=np.float64)
        for key in data.files
        if key.startswith("draft_")
    }
    if not roles:
        raise ValueError(f"No draft arrays found in {path}")
    if len(lengths) != len(labels):
        raise ValueError("lengths and labels disagree")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    if offsets[-1] != len(target):
        raise ValueError("target token array does not match lengths")
    for name, values in roles.items():
        if len(values) != len(target):
            raise ValueError(f"{name} token array does not match target")
    return labels, lengths, {"target": target, **roles}


def split_indices(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Fixed 800/400/400/400 per class when 2000 records are available."""
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {name: [] for name in ("train", "validation", "calibration", "test")}
    for label in (1, 0):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        counts = (800, 400, 400, len(indices) - 1600)
        start = 0
        for name, count in zip(result, counts):
            result[name].extend(indices[start : start + count].tolist())
            start += count
    return {name: np.asarray(sorted(indices), dtype=np.int64) for name, indices in result.items()}


def evaluate_score(
    values: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    member = values[labels == 1]
    nonmember = values[labels == 0]
    rates = (0.10, 0.05, 0.01)
    full = {
        "auc": bootstrap_auc(member, nonmember, bootstrap_repeats, seed),
        "tpr_at_fpr": {},
    }
    full_bootstrap = bootstrap_operating_points(
        member,
        nonmember,
        nonmember,
        rates,
        bootstrap_repeats,
        seed + 101,
        calibration_is_nonmember=True,
    )
    for rate in rates:
        point = threshold_metrics(
            member, nonmember, rate, calibration_nonmember=nonmember
        )
        interval = full_bootstrap[f"{int(rate * 100)}%"]
        point.update(
            {
                "tpr_ci95_low": interval["tpr"]["ci95_low"],
                "tpr_ci95_high": interval["tpr"]["ci95_high"],
                "fpr_ci95_low": interval["fpr"]["ci95_low"],
                "fpr_ci95_high": interval["fpr"]["ci95_high"],
            }
        )
        full["tpr_at_fpr"][f"{int(rate * 100)}%"] = point
    calibration = partitions["calibration"]
    test = partitions["test"]
    cal_nonmember = values[calibration][labels[calibration] == 0]
    test_member = values[test][labels[test] == 1]
    test_nonmember = values[test][labels[test] == 0]
    calibrated_bootstrap = bootstrap_operating_points(
        test_member,
        test_nonmember,
        cal_nonmember,
        rates,
        bootstrap_repeats,
        seed + 202,
    )
    calibrated_points: dict[str, Any] = {}
    for rate in rates:
        point = threshold_metrics(
            test_member,
            test_nonmember,
            rate,
            calibration_nonmember=cal_nonmember,
        )
        interval = calibrated_bootstrap[f"{int(rate * 100)}%"]
        point.update(
            {
                "tpr_ci95_low": interval["tpr"]["ci95_low"],
                "tpr_ci95_high": interval["tpr"]["ci95_high"],
                "fpr_ci95_low": interval["fpr"]["ci95_low"],
                "fpr_ci95_high": interval["fpr"]["ci95_high"],
            }
        )
        calibrated_points[f"{int(rate * 100)}%"] = point
    split = {
        "calibration_n_member": int(np.sum(labels[calibration] == 1)),
        "calibration_n_nonmember": int(np.sum(labels[calibration] == 0)),
        "test_n_member": int(len(test_member)),
        "test_n_nonmember": int(len(test_nonmember)),
        "test_tpr_at_calibrated_fpr": calibrated_points,
        "test_auc": bootstrap_auc(
            test_member, test_nonmember, bootstrap_repeats, seed + 991
        ),
    }
    return {"full_pool": full, "calibrated_test": split}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="Optional pq_gap_scores.npz containing labels; defaults to the sibling file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.scores
    if scores_path is not None and not scores_path.is_absolute():
        scores_path = ROOT / scores_path
    labels, lengths, arrays = load_records(input_path, scores_path)
    partitions = split_indices(labels, args.seed)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    report: dict[str, Any] = {
        "protocol": {
            "benchmark": args.benchmark,
            "target_epochs": args.epoch,
            "input": str(input_path),
            "n_member": int(np.sum(labels == 1)),
            "n_nonmember": int(np.sum(labels == 0)),
            "windows": list(DEFAULT_WINDOWS),
            "split": "fixed per-class 800 train / 400 validation / 400 calibration / 400 test",
            "threshold_rule": (
                "split-conformal upper-tail p-value: "
                "(1 + count(calibration_nonmember_score >= score))/(n+1) <= target FPR; "
                "ties are counted inclusively"
            ),
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
        },
        "roles": {},
        "method_deltas": {},
    }
    npz_values: dict[str, np.ndarray] = {"labels": labels, "lengths": lengths}
    for role in sorted(key for key in arrays if key != "target"):
        feature_rows = []
        for index in range(len(labels)):
            start, end = int(offsets[index]), int(offsets[index + 1])
            feature_rows.append(record_features(arrays["target"][start:end], arrays[role][start:end]))
        names = list(feature_rows[0])
        feature_matrix = np.asarray([[row[name] for name in names] for row in feature_rows], dtype=np.float64)
        results: dict[str, Any] = {}
        for column, name in enumerate(names):
            results[name] = evaluate_score(
                feature_matrix[:, column], labels, partitions, args.bootstrap_repeats, args.seed + column
            )
        report["roles"][role] = results
        npz_values[f"{role}__features"] = feature_matrix.astype(np.float32)
        report["method_deltas"][role] = {}
        baselines = ("mean_abs_delta", "window_sign_16")
        for baseline_name in baselines:
            if baseline_name not in names:
                continue
            baseline = feature_matrix[:, names.index(baseline_name)]
            for column, name in enumerate(names):
                if name == baseline_name:
                    continue
                report["method_deltas"][role][f"{name}_vs_{baseline_name}"] = (
                    paired_bootstrap_method_delta(
                        feature_matrix[:, column],
                        baseline,
                        labels,
                        partitions,
                        (0.10, 0.05, 0.01),
                        args.bootstrap_repeats,
                        args.seed + 10000 + column + 100 * baselines.index(baseline_name),
                    )
                )
    np.savez_compressed(output_dir / "directional_features.npz", **npz_values)
    (output_dir / "directional_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Directional and local-window SD membership metrics",
        "",
        f"- Benchmark: `{args.benchmark}`; target SFT epoch: `{args.epoch}`",
        f"- Input: `{input_path}`",
        "- p/q source: saved teacher-forced response-token log probabilities from Qwen3-8B target and Qwen3-1.7B drafts.",
        "- Full-pool results are exploratory. The calibrated-test section uses a fixed 800/400/400/400 per-class split; thresholds use only calibration nonmembers.",
        "- Thresholds use the split-conformal upper-tail p-value `(1 + count(calibration score >= score))/(n+1)`; ties are counted inclusively and actual test FPR is reported.",
        "- TPR/FPR intervals are record-level percentile bootstrap intervals; calibration and test records are resampled separately. No token bootstrap is used.",
        "",
    ]
    for role, role_results in report["roles"].items():
        lines.extend([f"## {role}", "", "| Score | Full AUC | Full TPR@1% | Test AUC | Test TPR@1% | Test FPR@1% |", "|---|---:|---:|---:|---:|---:|"])
        for name, result in role_results.items():
            full = result["full_pool"]
            test = result["calibrated_test"]["test_tpr_at_calibrated_fpr"]["1%"]
            lines.append(
                f"| `{name}` | {full['auc']['point']:.4f} | {full['tpr_at_fpr']['1%']['tpr']:.4f} "
                f"| {result['calibrated_test']['test_auc']['point']:.4f} "
                f"| {test['tpr']:.4f} [{test['tpr_ci95_low']:.4f}, {test['tpr_ci95_high']:.4f}] "
                f"| {test['fpr']:.4f} [{test['fpr_ci95_low']:.4f}, {test['fpr_ci95_high']:.4f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "## Paired calibrated-test method increments",
            "",
            "The deltas below use the same resampled records for both methods; positive TPR means the first method is better.",
            "",
            "| Comparison | ΔAUC | ΔTPR@1% | ΔFPR@1% |",
            "|---|---:|---:|---:|",
        ]
    )
    for role, comparisons in report["method_deltas"].items():
        for comparison, result in comparisons.items():
            tpr = result["tpr_at_calibrated_fpr"]["1%"]["tpr_delta"]
            fpr = result["tpr_at_calibrated_fpr"]["1%"]["fpr_delta"]
            auc = result["auc"]
            lines.append(
                f"| `{role}: {comparison}` | {auc['point']:.4f} [{auc['ci95_low']:.4f}, {auc['ci95_high']:.4f}] "
                f"| {tpr['point']:.4f} [{tpr['ci95_low']:.4f}, {tpr['ci95_high']:.4f}] "
                f"| {fpr['point']:.4f} [{fpr['ci95_low']:.4f}, {fpr['ci95_high']:.4f}] |"
            )
    lines.append("")
    (output_dir / "DIRECTIONAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
