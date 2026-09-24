"""Paired summaries for the registered frozen-model direction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.shared.core.audit_metrics import membership_metrics
from experiments.sd_membership_sft.analysis.analyze_conditional_accept_only import attach_legacy, ranking
from experiments.shared.core.audit_runtime import ROOT, _write_json


def summarize(paths, comparisons, *, repeats=500, legacy_root=None):
    archives, reports, groups = [], [], {}
    for path in sorted(paths):
        report = json.loads(path.read_text())
        with np.load(path.parent / "scores.npz", allow_pickle=False) as source:
            archive = dict(source)
        if legacy_root is not None:
            source = json.loads((path.parent / "SOURCE.json").read_text())
            legacy = legacy_root / f"{source['benchmark']}_epoch{source['epoch']}" / f"scores_seed_{report['seed']}.npz"
            attach_legacy(archive, legacy)
            report["metrics"]["legacy_neural_fusion"] = membership_metrics(archive["legacy_neural_fusion"], archive["labels"], archive["calibration"], archive["test"])
        condition = report.get("benchmark", "") + str(report.get("epoch", ""))
        if not condition:
            condition = path.parent.parent.name
        for other in groups.get(condition, []):
            for key in ("labels", "record_ids", "test", "calibration"):
                if not np.array_equal(archive[key], archives[other][key]):
                    raise ValueError("cannot pool seeds with different records or splits")
        groups.setdefault(condition, []).append(len(archives))
        archives.append(archive)
        reports.append(report)
    if not archives:
        raise ValueError("no completed runs")
    methods = sorted({name for pair in comparisons for name in pair})
    def macro(values):
        return np.mean([np.mean([values[i] for i in indices], axis=0) for indices in groups.values()], axis=0)
    means = {}
    per_condition = {}
    for method in methods:
        values = []
        for report in reports:
            m = report["metrics"][method]
            tail = m["tpr_at_fpr"]["1%"]
            values.append([m["auc"], m["pauc_0_10"], tail["tpr"], tail["actual_fpr"]])
        means[method] = dict(zip(("auc", "pauc", "tpr_1", "fpr_1"), macro(values).tolist()))
        per_condition[method] = {condition: np.mean([values[i] for i in indices], axis=0).tolist() for condition, indices in groups.items()}
    rng = np.random.default_rng(20260917)
    draws = {pair: np.empty((repeats, 2)) for pair in comparisons}
    cohorts = {condition: (tuple(archives[indices[0]]["record_ids"][archives[indices[0]]["test"]].tolist()),
                           tuple(archives[indices[0]]["labels"][archives[indices[0]]["test"]].tolist()))
               for condition, indices in groups.items()}
    for repeat in range(repeats):
        values = {pair: [] for pair in comparisons}
        cohort_draws = {}
        for condition, indices in groups.items():
            first = archives[indices[0]]
            test, labels = first["test"], first["labels"]
            member, nonmember = test[labels[test] == 1], test[labels[test] == 0]
            cohort = cohorts[condition]
            if cohort not in cohort_draws:
                cohort_draws[cohort] = (rng.integers(len(member), size=len(member)), rng.integers(len(nonmember), size=len(nonmember)))
            mi, ni = cohort_draws[cohort]
            member, nonmember = member[mi], nonmember[ni]
            scored = {i: {method: ranking(archives[i][method], member, nonmember) for method in methods} for i in indices}
            for pair in comparisons:
                values[pair].append(np.mean([scored[i][pair[0]] - scored[i][pair[1]] for i in indices], axis=0))
        for pair in comparisons:
            draws[pair][repeat] = np.mean(values[pair], axis=0)
    changes = {}
    for pair, samples in draws.items():
        changes[f"{pair[0]} - {pair[1]}"] = {
            name: {"delta": means[pair[0]][name] - means[pair[1]][name],
                   "ci95": np.quantile(samples[:, index], [.025, .975]).tolist(),
                   "positive_conditions": sum(per_condition[pair[0]][group][index] > per_condition[pair[1]][group][index] for group in groups)}
            for index, name in enumerate(("auc", "pauc"))}
    return {"runs": len(reports), "conditions": len(groups), "means": means,
            "comparisons": changes, "per_condition": per_condition,
            "bootstrap_repeats": repeats,
            "interval_scope": "paired record resampling shared across seeds and checkpoints with identical test records; fixed models"}


def render(name, report):
    lines = [f"## {name}", "", f"{report['conditions']} conditions, {report['runs']} runs.", "",
             "| Method | AUC | pAUC@10% | TPR@nominal 1% | Actual FPR |", "|---|---:|---:|---:|---:|"]
    for method, m in report["means"].items():
        lines.append(f"| {method} | {m['auc']:.4f} | {m['pauc']:.4f} | {m['tpr_1']:.4f} | {m['fpr_1']:.4f} |")
    lines += ["", "| Comparison | Delta AUC [95% CI] | Delta pAUC [95% CI] | AUC-positive conditions |", "|---|---:|---:|---:|"]
    for comparison, m in report["comparisons"].items():
        values = [f"{m[key]['delta']:+.4f} [{m[key]['ci95'][0]:+.4f}, {m[key]['ci95'][1]:+.4f}]" for key in ("auc", "pauc")]
        lines.append(f"| {comparison} | {' | '.join(values)} | {m['auc']['positive_conditions']}/{report['conditions']} |")
    lines += ["", "### Per-condition AUC", "",
              "| Method | " + " | ".join(next(iter(report["per_condition"].values()))) + " |",
              "|---|" + "---:|" * report["conditions"]]
    for method, values in report["per_condition"].items():
        lines.append(f"| {method} | " + " | ".join(f"{value[0]:.4f}" for value in values.values()) + " |")
    return lines


def protocol_costs(root):
    active = [json.loads(path.read_text()) for path in sorted((root / "active").glob("*/seed*/REPORT.json"))]
    stopping = {}
    if active:
        stopping = {level: {key: float(np.mean([report["positive_stopping"][level][key] for report in active]))
                            for key in ("tpr", "actual_fpr", "mean_decisions", "max_decisions")}
                    for level in ("0.01", "0.05")}
    return {"joint_positive_stopping_macro": stopping,
            "active_reference_queries_per_condition_seed": 400 * 64 * 10,
            "active_fixed_decisions_per_test_record": {"B2": 128, "B8": 512},
            "note": "active stopping is checked only at registered B=1,2,4,8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "experiments/results/sft_runs/directions_validation")
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    reports, completeness = {}, {}
    phases = [("conditional_b2", sorted((args.root.parent / "conditional_accept_only").glob("*/b2_seed*/REPORT.json")), 18,
               [(name, baseline) for name in ("original_global", "original_span", "original_fusion") for baseline in ("lowq", "legacy_neural_fusion")]),
              ("active", sorted((args.root / "active").glob("*/seed*/REPORT.json")), 18,
               [(f"{method}_b{b}", f"uniform_ladder_b{b}") for b in (2, 8) for method in ("fixed_q", "adaptive_q", "adaptive_position", "joint")])]
    for budget in (2, 8):
        phases.append((f"paired_b{budget}", sorted((args.root / "paired").glob(f"*/b{budget}_seed*/REPORT.json")), 18,
                       [(f"paired_{kind}", f"original_{kind}") for kind in ("global", "span", "fusion")]))
    for name, paths, expected, comparisons in phases:
        completeness[name] = {"expected": expected, "completed": len(paths)}
        if len(paths) != expected and not args.allow_incomplete:
            raise ValueError(f"{name}: expected {expected} runs, found {len(paths)}")
        if paths:
            legacy = args.root.parent / "accept_only_active_v2/neural_adaptive/conditions" if name == "conditional_b2" else None
            reports[name] = summarize(paths, comparisons, repeats=args.bootstrap_repeats, legacy_root=legacy)
    costs = protocol_costs(args.root)
    report = {"models_frozen": True, "training_member_count": 0, "phases": reports,
              "complete": all(value["expected"] == value["completed"] for value in completeness.values()),
              "completeness": completeness, "costs": costs,
              "warning": "candidate scopes/protocols differ across phases; compare within phase only"}
    _write_json(args.root / "DIRECTIONS_REPORT.json", report)
    lines = ["# Frozen-model direction validation", "", "Target/draft weights remain frozen. Only real trusted nonmembers fit and select detectors.",
             "Do not compare absolute AUC across whole-document and suffix scopes.",
             "Intervals condition on fitted models; nominal low-FPR TPR must be read with actual FPR.",
             "Paired 95% intervals are exploratory and not adjusted for multiple comparisons.",
             "pAUC@10% is ROC area over FPR 0–0.10 divided by 0.10; random ranking has expected value 0.05.", ""]
    lines += [f"Status: {'COMPLETE' if report['complete'] else 'PARTIAL'}.", ""]
    for name, phase in reports.items():
        lines += render(name, phase) + [""]
    lines += ["## Query costs", "",
              "Active reference fit/selection: 256,000 decisions per condition/seed. Test budgets: 128 or 512 decisions per record.",
              "Joint positive stopping uses nonmember calibration maxima over B=1,2,4,8.", "",
              "| Nominal FPR | TPR | Actual FPR | Mean decisions | Maximum decisions |", "|---|---:|---:|---:|---:|"]
    for level, value in costs["joint_positive_stopping_macro"].items():
        lines.append(f"| {float(level):.0%} | {value['tpr']:.4f} | {value['actual_fpr']:.4f} | {value['mean_decisions']:.2f} | {value['max_decisions']:.0f} |")
    (args.root / "DIRECTIONS_REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.root / "DIRECTIONS_REPORT.md")


if __name__ == "__main__":
    main()
