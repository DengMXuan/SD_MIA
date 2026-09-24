"""Paired bootstrap for nonmember-only neural accept-only experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.shared.core.audit_metrics import rank_auc
from experiments.shared.core.audit_runtime import split_indices
from experiments.shared.core.audit_runtime import BENCHMARKS, EPOCHS, REPLAY_SEEDS, ROOT, _write_json
from experiments.sd_membership_sft.analysis.token_signal_anatomy import fast_partial_auc


COMPARISONS = {
    "neural_only_k1_vs_lowq": ("neural_score_k1", "lowq_k1"),
    "residual_k1_vs_lowq": ("lowq_plus_neural_k1", "lowq_k1"),
    "neural_only_k2_vs_lowq": ("neural_score_k2", "lowq_k2"),
    "residual_k2_vs_lowq": ("lowq_plus_neural_k2", "lowq_k2"),
    "member_mask_selector_vs_uniform": ("neural_active_k8", "uniform_active_k8"),
    "measurement_selector_vs_uniform": ("measurement_neural_active_k8", "uniform_active_k8"),
    "measurement_fusion_vs_uniform_fusion": ("measurement_fusion_k8", "uniform_fusion_k8"),
    "uniform_fusion_vs_residual_k2": ("uniform_fusion_k8", "lowq_plus_neural_k2"),
    "measurement_fusion_vs_residual_k2": ("measurement_fusion_k8", "lowq_plus_neural_k2"),
    "measurement_fusion_vs_uniform_active": ("measurement_fusion_k8", "uniform_active_k8"),
}


def _metrics(values: np.ndarray, member: np.ndarray, nonmember: np.ndarray) -> tuple[float, float]:
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    scores = np.r_[values[member], values[nonmember]]
    return rank_auc(values[member], values[nonmember]), fast_partial_auc(scores, labels)


def _load(input_root: Path) -> tuple[dict[tuple[str, int, int], dict[str, np.ndarray]], dict[tuple[str, int, int], dict[str, Any]]]:
    arrays: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        for epoch in EPOCHS:
            directory = input_root / "conditions" / f"{benchmark}_epoch{epoch}"
            report = json.loads((directory / "RAW_RESULTS.json").read_text(encoding="utf-8"))
            by_seed = {int(row["seed"]): row for row in report["rows"]}
            for seed in REPLAY_SEEDS:
                with np.load(directory / f"scores_seed_{seed}.npz", allow_pickle=False) as archive:
                    arrays[(benchmark, epoch, seed)] = {
                        name: np.asarray(archive[name]) for name in archive.files
                    }
                rows[(benchmark, epoch, seed)] = by_seed[seed]
    return arrays, rows


def aggregate(input_root: Path, repeats: int, seed: int) -> dict[str, Any]:
    arrays, rows = _load(input_root)
    roles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for benchmark in BENCHMARKS:
        first = arrays[(benchmark, EPOCHS[0], REPLAY_SEEDS[0])]
        partition = split_indices(first["labels"], 20260824)["T"]
        roles[benchmark] = (
            partition[first["labels"][partition] == 1],
            partition[first["labels"][partition] == 0],
        )
    rng = np.random.default_rng(seed)
    member_draws = {
        benchmark: member[rng.integers(0, len(member), size=(repeats, len(member)))]
        for benchmark, (member, _) in roles.items()
    }
    nonmember_draws = {
        benchmark: nonmember[rng.integers(0, len(nonmember), size=(repeats, len(nonmember)))]
        for benchmark, (_, nonmember) in roles.items()
    }
    seed_draws = rng.integers(0, len(REPLAY_SEEDS), size=(repeats, len(REPLAY_SEEDS)))
    results: dict[str, Any] = {}
    for name, (method, baseline) in COMPARISONS.items():
        condition_differences: dict[str, dict[str, float]] = {}
        method_auc, baseline_auc, method_pauc, baseline_pauc = [], [], [], []
        for benchmark in BENCHMARKS:
            member, nonmember = roles[benchmark]
            for epoch in EPOCHS:
                condition_method, condition_baseline = [], []
                condition_method_auc, condition_baseline_auc = [], []
                for replay_seed in REPLAY_SEEDS:
                    data = arrays[(benchmark, epoch, replay_seed)]
                    ma, mp = _metrics(data[method], member, nonmember)
                    ba, bp = _metrics(data[baseline], member, nonmember)
                    condition_method.append(mp)
                    condition_baseline.append(bp)
                    condition_method_auc.append(ma)
                    condition_baseline_auc.append(ba)
                method_pauc.append(float(np.mean(condition_method)))
                baseline_pauc.append(float(np.mean(condition_baseline)))
                method_auc.append(float(np.mean(condition_method_auc)))
                baseline_auc.append(float(np.mean(condition_baseline_auc)))
                condition_differences[f"{benchmark}_epoch{epoch}"] = {
                    "auc": float(np.mean(condition_method_auc) - np.mean(condition_baseline_auc)),
                    "pauc_0_10": float(np.mean(condition_method) - np.mean(condition_baseline)),
                }
        samples = np.zeros((repeats, 2), dtype=np.float64)
        for repeat in range(repeats):
            total_auc, total_pauc, count = 0.0, 0.0, 0
            for sampled_seed in seed_draws[repeat]:
                replay_seed = REPLAY_SEEDS[int(sampled_seed)]
                for benchmark in BENCHMARKS:
                    member = member_draws[benchmark][repeat]
                    nonmember = nonmember_draws[benchmark][repeat]
                    for epoch in EPOCHS:
                        data = arrays[(benchmark, epoch, replay_seed)]
                        ma, mp = _metrics(data[method], member, nonmember)
                        ba, bp = _metrics(data[baseline], member, nonmember)
                        total_auc += ma - ba
                        total_pauc += mp - bp
                        count += 1
            samples[repeat] = (total_auc / count, total_pauc / count)
        auc_ci = np.quantile(samples[:, 0], (0.025, 0.975))
        pauc_ci = np.quantile(samples[:, 1], (0.025, 0.975))
        results[name] = {
            "method": method,
            "baseline": baseline,
            "auc": {
                "method": float(np.mean(method_auc)),
                "baseline": float(np.mean(baseline_auc)),
                "difference": float(np.mean(method_auc) - np.mean(baseline_auc)),
                "ci95": auc_ci.tolist(),
            },
            "pauc_0_10": {
                "method": float(np.mean(method_pauc)),
                "baseline": float(np.mean(baseline_pauc)),
                "difference": float(np.mean(method_pauc) - np.mean(baseline_pauc)),
                "ci95": pauc_ci.tolist(),
            },
            "positive_pauc_conditions": int(
                sum(value["pauc_0_10"] > 0 for value in condition_differences.values())
            ),
            "condition_differences": condition_differences,
        }
    uniform_rmse = np.asarray(
        [row["active_delta_rmse"]["uniform_active_k8"] for row in rows.values()]
    )
    measurement_rmse = np.asarray(
        [row["active_delta_rmse"]["measurement_neural_active_k8"] for row in rows.values()]
    )
    indices = rng.integers(0, len(uniform_rmse), size=(10_000, len(uniform_rmse)))
    reduction = 1.0 - np.mean(measurement_rmse[indices], axis=1) / np.mean(
        uniform_rmse[indices], axis=1
    )
    return {
        "experiment": "paired neural adaptive accept-only confirmation",
        "status": "exploratory cached-T offline verifier replay",
        "bootstrap": {
            "repeats": repeats,
            "seed": seed,
            "unit": "record-paired across methods/epochs plus replay-seed resampling",
        },
        "comparisons": results,
        "measurement_rmse": {
            "uniform": float(np.mean(uniform_rmse)),
            "learned": float(np.mean(measurement_rmse)),
            "relative_reduction": float(1.0 - np.mean(measurement_rmse) / np.mean(uniform_rmse)),
            "ci95": np.quantile(reduction, (0.025, 0.975)).tolist(),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Neural Adaptive Accept-Only Paired Analysis",
        "",
        "> Exploratory cached-T replay; intervals are paired across methods.",
        "",
        "| Comparison | ΔAUC [95% CI] | ΔpAUC [95% CI] | Positive conditions |",
        "|---|---:|---:|---:|",
    ]
    for name, value in report["comparisons"].items():
        auc, pauc = value["auc"], value["pauc_0_10"]
        lines.append(
            f"| `{name}` | {auc['difference']:+.4f} [{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] | "
            f"{pauc['difference']:+.4f} [{pauc['ci95'][0]:+.4f}, {pauc['ci95'][1]:+.4f}] | "
            f"{value['positive_pauc_conditions']}/6 |"
        )
    rmse = report["measurement_rmse"]
    lines.extend(
        [
            "",
            "## Measurement value controller",
            "",
            f"Uniform RMSE: `{rmse['uniform']:.4f}`; learned RMSE: `{rmse['learned']:.4f}`; "
            f"relative reduction: `{rmse['relative_reduction']:.2%}` "
            f"[{rmse['ci95'][0]:.2%}, {rmse['ci95'][1]:.2%}].",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/neural_adaptive",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    args = parser.parse_args()
    report = aggregate(args.input_root.resolve(), args.bootstrap_repeats, args.bootstrap_seed)
    _write_json(args.input_root / "PAIRED_ANALYSIS.json", report)
    write_markdown(report, args.input_root / "PAIRED_ANALYSIS.md")
    print(json.dumps({"output": str(args.input_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
