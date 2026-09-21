"""Compare frozen conditional detectors with paired, equal-budget baselines.

Record bootstrap intervals condition on the fitted models. Repeated seeds of
the same condition reuse the SAME resampled records to avoid treating them
as independent datasets. No result feeds back into training or model choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.sd_membership_sft.core.audit_metrics import (membership_metrics)
from experiments.sd_membership_sft.core.audit_runtime import (ROOT, _write_json)


def attach_legacy(archive: dict[str, np.ndarray], path: Path) -> None:
    with np.load(path, allow_pickle=False) as legacy:
        for key in ("labels", "record_ids"):
            if not np.array_equal(archive[key], legacy[key]):
                raise ValueError(f"legacy {key} alignment failed: {path}")
        if not np.allclose(archive["lowq"], legacy["lowq_k2"], atol=1e-10, rtol=1e-10):
            raise ValueError("legacy baseline differs: token scope, replay or normalization mismatch")
        archive["legacy_neural_fusion"] = legacy["lowq_plus_neural_k2"]


def ranking(scores, member, nonmember):
    """Vectorized tie-aware ROC, equivalent to the registered rank/pAUC code."""
    values = np.r_[scores[member], scores[nonmember]]
    labels = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    tp = np.cumsum(labels[order])[ends]
    fp = ends + 1 - tp
    x, y = np.r_[0., fp / len(nonmember)], np.r_[0., tp / len(member)]
    auc = float(np.trapezoid(y, x))
    select = (x[:-1] < .1) & (x[1:] > x[:-1])
    left, right = x[:-1][select], np.minimum(x[1:][select], .1)
    yleft = y[:-1][select]
    yright = yleft + (right - left) / (x[1:][select] - left) * (y[1:][select] - yleft)
    pauc = float(np.sum((right - left) * (yleft + yright) / 2) / .1)
    return np.array([auc, pauc])


def analyze(root: Path, budget: int, legacy_root: Path | None, repeats: int = 500, scope: str = "cached"):
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    if scope not in ("cached", "paired"):
        raise ValueError("unknown observation scope")
    paths = sorted(root.glob(f"*/b{budget}_seed*/REPORT.json"))
    if not paths:
        raise FileNotFoundError(f"no budget-{budget} reports under {root}")
    rows, archives, groups = [], [], {}
    for path in paths:
        report = json.loads(path.read_text())
        source = json.loads((path.parent / "SOURCE.json").read_text())
        # Whole-document cache scores and fixed-suffix paired scores must not
        # silently enter the same macro average.
        if ("benchmark" in source) != (scope == "cached"):
            continue
        with np.load(path.parent / "scores.npz", allow_pickle=False) as data:
            archive = dict(data)
        metrics = dict(report["metrics"])
        if legacy_root is not None and budget == 2 and "benchmark" in source:
            legacy = legacy_root / f"{source['benchmark']}_epoch{source['epoch']}" / f"scores_seed_{report['seed']}.npz"
            attach_legacy(archive, legacy)
            metrics["legacy_neural_fusion"] = membership_metrics(archive["legacy_neural_fusion"], archive["labels"], archive["calibration"], archive["test"])
        # The archive hash groups paired runs; original replay groups by condition.
        group = (source["benchmark"], source["epoch"]) if "benchmark" in source else (source["candidate_scope_sha256"],)
        for previous in groups.get(group, []):
            for key in ("labels", "record_ids", "test"):
                if not np.array_equal(archive[key], archives[previous][key]):
                    raise ValueError("same-condition seeds use different record partitions")
        groups.setdefault(group, []).append(len(archives))
        archives.append(archive)
        rows.append({"path": str(path), "condition": path.parent.parent.name, "seed": report["seed"], "metrics": metrics})
    if not rows:
        raise FileNotFoundError(f"no {scope} observation reports under {root}")
    methods = sorted(set.intersection(*(set(row["metrics"]) for row in rows)))
    def condition_mean(values):
        return np.mean([np.mean([values[i] for i in indices], axis=0) for indices in groups.values()], axis=0)
    means = {}
    for method in methods:
        values = []
        for row in rows:
            metric = row["metrics"][method]
            tail = metric["tpr_at_fpr"]["1%"]
            values.append([metric["auc"], metric["pauc_0_10"], tail["tpr"], tail["actual_fpr"]])
        means[method] = dict(zip(("auc", "pauc_0_10", "tpr_1", "fpr_1"), condition_mean(values).tolist()))
    pairs = [(left, right) for left in ("original_global", "original_span", "original_fusion", "paired_global", "paired_span", "paired_fusion")
             for right in ("lowq", "legacy_neural_fusion") if left in methods and right in methods]
    if "paired_span" in methods:
        pairs.extend([("paired_span", "original_span"), ("paired_fusion", "original_fusion")])
    boot = {pair: np.empty((repeats, 2)) for pair in pairs}
    used_methods = sorted({method for pair in pairs for method in pair})
    cohorts = {group: (tuple(archives[indices[0]]["record_ids"][archives[indices[0]]["test"]].tolist()),
                       tuple(archives[indices[0]]["labels"][archives[indices[0]]["test"]].tolist()))
               for group, indices in groups.items()}
    rng = np.random.default_rng(20260917)
    for repeat in range(repeats):
        sampled, cohort_draws = {}, {}
        for group, indices in groups.items():
            archive = archives[indices[0]]
            test, labels = archive["test"], archive["labels"]
            member, nonmember = test[labels[test] == 1], test[labels[test] == 0]
            cohort = cohorts[group]
            if cohort not in cohort_draws:
                cohort_draws[cohort] = (rng.integers(len(member), size=len(member)), rng.integers(len(nonmember), size=len(nonmember)))
            mi, ni = cohort_draws[cohort]
            sampled[group] = (member[mi], nonmember[ni])
        ranked = {}
        for group, indices in groups.items():
            member, nonmember = sampled[group]
            for i in indices:
                ranked[i] = {method: ranking(archives[i][method], member, nonmember) for method in used_methods}
        for pair in pairs:
            group_values = []
            for group, indices in groups.items():
                group_values.append(np.mean([ranked[i][pair[0]] - ranked[i][pair[1]] for i in indices], axis=0))
            boot[pair][repeat] = np.mean(group_values, axis=0)
    comparisons = {}
    for pair, samples in boot.items():
        comparisons[f"{pair[0]} - {pair[1]}"] = {
            key: {"delta": means[pair[0]][key] - means[pair[1]][key], "ci95": np.quantile(samples[:, j], [0.025, 0.975]).tolist()}
            for j, key in enumerate(("auc", "pauc_0_10"))}
    return {"budget": budget, "scope": scope, "condition_count": len(groups), "run_count": len(rows), "rows": rows,
            "means": means, "comparisons": comparisons, "bootstrap_repeats": repeats,
            "interval_scope": "paired test-record resampling shared across seeds and conditions with identical test records; conditional on fitted models, no retraining uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/results/sft_runs/conditional_accept_only")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--scope", choices=("cached", "paired"), default="cached")
    parser.add_argument("--legacy-root", type=Path, help="optional existing neural_adaptive/conditions directory; strict alignment required")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    args = parser.parse_args()
    report = analyze(args.root, args.budget, args.legacy_root, args.bootstrap_repeats, args.scope)
    output = args.root / f"COMPARISON{'_paired' if args.scope == 'paired' else ''}_b{args.budget}"
    _write_json(output.with_suffix(".json"), report)
    lines = ["# Conditional accept-only comparison", "", f"{report['condition_count']} conditions; {report['run_count']} frozen runs; budget {args.budget} per candidate token.",
             "", "| Method | AUC | pAUC@10% | TPR@1% | Actual FPR |", "|---|---:|---:|---:|---:|"]
    for name, values in report["means"].items():
        lines.append(f"| {name} | {values['auc']:.4f} | {values['pauc_0_10']:.4f} | {values['tpr_1']:.4f} | {values['fpr_1']:.4f} |")
    lines.extend(["", "Intervals: " + report["interval_scope"], "", "| Comparison | Delta AUC [95% CI] | Delta pAUC [95% CI] |", "|---|---:|---:|"])
    for name, values in report["comparisons"].items():
        cells = [f"{values[key]['delta']:+.4f} [{values[key]['ci95'][0]:+.4f}, {values[key]['ci95'][1]:+.4f}]" for key in ("auc", "pauc_0_10")]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
