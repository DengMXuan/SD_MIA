"""Aggregate paired E3 replay seeds and evaluate Gate 2 on short fragments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sd_membership_sft.analysis.token_signal_anatomy import ROOT, build_roles, fast_partial_auc


SHORT_BENCHMARKS = ("wikitection", "newstection")
EPOCHS = (1, 3)
COMPARISONS = {
    "corrected_uniform_vs_fixed": ("active_q_corrected_uniform", "fixed_q"),
    "fixed_lowq_score_vs_fixed_window": ("fixed_q_lowq_score", "fixed_q"),
    "fixed_lowq_multiscale_vs_fixed_window": ("fixed_q_lowq_multiscale", "fixed_q"),
    "active_lowq_multiscale_vs_fixed_lowq_multiscale": (
        "active_q_corrected_uniform_lowq_multiscale", "fixed_q_lowq_multiscale"
    ),
    "lowq_score_uniform_active_vs_fixed": ("active_q_corrected_uniform_lowq_score", "fixed_q_lowq_score"),
    "q_low_importance_vs_fixed": ("q_low_importance_active_q", "fixed_q_lowq_score"),
    "q_low_importance_vs_uniform": ("q_low_importance_active_q", "active_q_corrected_uniform_lowq_score"),
    "oracle_window_vs_fixed": ("oracle_importance_active_q", "fixed_q"),
    "corrected_uniform_vs_uncorrected": ("active_q_corrected_uniform", "active_q_uncorrected"),
}


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pauc(values: np.ndarray, member: np.ndarray, nonmember: np.ndarray) -> float:
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    return fast_partial_auc(np.r_[values[member], values[nonmember]], labels)


def _load(root: Path, seed: int, benchmark: str, epoch: int) -> dict[str, Any]:
    directory = root / f"seed_{seed}" / f"{benchmark}_epoch{epoch}"
    report_path, scores_path = directory / "E3_OFFLINE_REPORT.json", directory / "scores.npz"
    if not report_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"incomplete E3 replay output: {directory}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(scores_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return {"report": report, "scores": arrays}


def aggregate(
    input_root: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    if not seeds or bootstrap_repeats <= 0:
        raise ValueError("seeds and bootstrap repeats must be nonempty/positive")
    runs = {
        (seed, benchmark, epoch): _load(input_root, seed, benchmark, epoch)
        for seed in seeds
        for benchmark in SHORT_BENCHMARKS
        for epoch in EPOCHS
    }
    first = runs[(seeds[0], SHORT_BENCHMARKS[0], EPOCHS[0])]
    budgets = tuple(map(int, first["report"]["protocol"]["budgets"]))
    for run in runs.values():
        if tuple(map(int, run["report"]["protocol"]["budgets"])) != budgets:
            raise ValueError("budget grids differ across replay runs")
        if run["report"]["protocol"]["score_kind"] != "window_sign_8":
            raise ValueError("Gate-2 aggregate expects the E1-frozen window_sign_8 score")
        if run["report"]["protocol"]["oracle_signal"] != "window_boundary":
            raise ValueError("Gate-2 aggregate expects the E1-aligned window-boundary oracle")

    roles: dict[str, Any] = {}
    for benchmark in SHORT_BENCHMARKS:
        reference = runs[(seeds[0], benchmark, EPOCHS[0])]["scores"]
        roles[benchmark] = build_roles(reference["labels"], 20260824)
        for seed in seeds:
            for epoch in EPOCHS:
                scores = runs[(seed, benchmark, epoch)]["scores"]
                if not np.array_equal(scores["labels"], reference["labels"]):
                    raise ValueError(f"label alignment failed for {benchmark}")
                if not np.array_equal(scores["record_ids"], reference["record_ids"]):
                    raise ValueError(f"record alignment failed for {benchmark}")

    rng = np.random.default_rng(bootstrap_seed)
    draws: dict[tuple[str, str], np.ndarray] = {}
    for benchmark in SHORT_BENCHMARKS:
        for role_name in ("t_member", "t_nonmember"):
            values = getattr(roles[benchmark], role_name)
            draws[(benchmark, role_name)] = values[
                rng.integers(0, len(values), size=(bootstrap_repeats, len(values)))
            ]
    seed_draws = rng.integers(0, len(seeds), size=(bootstrap_repeats, len(seeds)))

    comparisons: dict[str, Any] = {}
    gate2_budgets: list[int] = []
    uniform_effective_budgets: list[int] = []
    for comparison_name, (method, baseline) in COMPARISONS.items():
        comparisons[comparison_name] = {}
        for budget in budgets:
            condition_differences: dict[str, float] = {}
            method_points, baseline_points = [], []
            for benchmark in SHORT_BENCHMARKS:
                role = roles[benchmark]
                for epoch in EPOCHS:
                    per_seed_method, per_seed_baseline = [], []
                    for seed in seeds:
                        arrays = runs[(seed, benchmark, epoch)]["scores"]
                        per_seed_method.append(_pauc(arrays[f"{method}_keq{budget}"], role.t_member, role.t_nonmember))
                        per_seed_baseline.append(_pauc(arrays[f"{baseline}_keq{budget}"], role.t_member, role.t_nonmember))
                    method_point = float(np.mean(per_seed_method))
                    baseline_point = float(np.mean(per_seed_baseline))
                    method_points.append(method_point)
                    baseline_points.append(baseline_point)
                    condition_differences[f"{benchmark}_epoch{epoch}"] = method_point - baseline_point

            samples = np.zeros(bootstrap_repeats, dtype=np.float64)
            for repeat in range(bootstrap_repeats):
                total = 0.0
                for sampled_seed_index in seed_draws[repeat]:
                    seed = seeds[int(sampled_seed_index)]
                    for benchmark in SHORT_BENCHMARKS:
                        member = draws[(benchmark, "t_member")][repeat]
                        nonmember = draws[(benchmark, "t_nonmember")][repeat]
                        for epoch in EPOCHS:
                            arrays = runs[(seed, benchmark, epoch)]["scores"]
                            total += _pauc(arrays[f"{method}_keq{budget}"], member, nonmember)
                            total -= _pauc(arrays[f"{baseline}_keq{budget}"], member, nonmember)
                samples[repeat] = total / (len(seeds) * len(SHORT_BENCHMARKS) * len(EPOCHS))

            point_difference = float(np.mean(method_points) - np.mean(baseline_points))
            low, high = np.quantile(samples, (0.025, 0.975))
            # Measurement RMSE is already token-aggregated within each run.
            method_rmse, baseline_rmse = [], []
            for seed in seeds:
                for benchmark in SHORT_BENCHMARKS:
                    for epoch in EPOCHS:
                        methods = runs[(seed, benchmark, epoch)]["report"]["methods"]
                        method_rmse.append(methods[method][str(budget)]["measurement"]["delta0_rmse"])
                        baseline_rmse.append(methods[baseline][str(budget)]["measurement"]["delta0_rmse"])
            mean_method_rmse = float(np.mean(method_rmse))
            mean_baseline_rmse = float(np.mean(baseline_rmse))
            relative_rmse_reduction = 1.0 - mean_method_rmse / mean_baseline_rmse
            result = {
                "method": method,
                "baseline": baseline,
                "short_macro_method_pauc": float(np.mean(method_points)),
                "short_macro_baseline_pauc": float(np.mean(baseline_points)),
                "short_macro_pauc_difference": {
                    "point": point_difference,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                },
                "condition_mean_differences": condition_differences,
                "positive_conditions": int(sum(value > 0.0 for value in condition_differences.values())),
                "short_macro_delta0_rmse": mean_method_rmse,
                "short_macro_baseline_delta0_rmse": mean_baseline_rmse,
                "relative_rmse_reduction": relative_rmse_reduction,
            }
            if comparison_name == "oracle_window_vs_fixed":
                result["passes_gate2_at_budget"] = bool(
                    relative_rmse_reduction >= 0.25
                    or (point_difference >= 0.03 and low > 0.0)
                )
                if result["passes_gate2_at_budget"]:
                    gate2_budgets.append(budget)
            if comparison_name == "corrected_uniform_vs_fixed":
                result["active_measurement_effective_at_budget"] = bool(
                    relative_rmse_reduction >= 0.25
                    or (point_difference >= 0.03 and low > 0.0)
                )
                if result["active_measurement_effective_at_budget"]:
                    uniform_effective_budgets.append(budget)
            comparisons[comparison_name][str(budget)] = result

    report = {
        "experiment": "E3 short-fragment paired replay aggregate",
        "protocol": {
            "input_root": str(input_root.resolve()),
            "seeds": list(seeds),
            "conditions": [f"{b}_epoch{e}" for b in SHORT_BENCHMARKS for e in EPOCHS],
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap": "record-paired across methods/epochs plus replay-seed resampling",
            "score_kind": "window_sign_8 (frozen by E1)",
            "oracle_signal": "exact window-boundary influence; nondeployable",
            "status": "offline position-locked verifier replay; exploratory cached V/T",
        },
        "gate2_passed": bool(gate2_budgets),
        "gate2_passing_budgets": sorted(set(gate2_budgets)),
        "all_position_active_measurement_effective": bool(uniform_effective_budgets),
        "all_position_active_effective_budgets": sorted(set(uniform_effective_budgets)),
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "E3_AGGREGATE.json", report)
    lines = [
        "# E3 Offline Replay Aggregate",
        "",
        "> Offline position-locked verifier replay on cached exact p/q; not a real remote API run.",
        "",
        f"Gate 2 passed: **{report['gate2_passed']}**; passing budgets: `{report['gate2_passing_budgets']}`",
        "",
        f"All-position q-corrected measurement effective: **{report['all_position_active_measurement_effective']}**; budgets: `{report['all_position_active_effective_budgets']}`",
        "",
        "| Comparison | K_eq | Method pAUC | Baseline pAUC | ΔpAUC [95% CI] | ΔRMSE | Positive cond. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, by_budget in comparisons.items():
        for budget in budgets:
            row = by_budget[str(budget)]
            delta = row["short_macro_pauc_difference"]
            lines.append(
                f"| `{name}` | {budget} | {row['short_macro_method_pauc']:.4f} | "
                f"{row['short_macro_baseline_pauc']:.4f} | {delta['point']:+.4f} "
                f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] | "
                f"{row['relative_rmse_reduction']:+.1%} | {row['positive_conditions']}/4 |"
            )
    (output_dir / "E3_AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "experiments/results/sft_runs/accept_only_active_v2/e3_offline"
    parser.add_argument("--input-root", type=Path, default=default)
    parser.add_argument("--output-dir", type=Path, default=default)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260914, 20260915, 20260916, 20260917, 20260918])
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = aggregate(
        args.input_root.resolve(), args.output_dir.resolve(), tuple(args.seeds),
        args.bootstrap_repeats, args.bootstrap_seed,
    )
    print(json.dumps({
        "gate2_passed": report["gate2_passed"],
        "passing_budgets": report["gate2_passing_budgets"],
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
