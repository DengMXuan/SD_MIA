"""Aggregate M1 condition reports and run the CPU probability baselines.

The per-condition fitting command writes ``m1_metrics.json``.  This module
does not reselect a method from C/T: it only collects already-frozen reports
and computes the preregistered resource-gate summary from V.  With
``--baseline-cache`` it also runs B0/B1/B2 directly from the existing p/q
cache, which is the CPU-first step of the M1 execution order.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.sd_membership_sft.analysis.directional_mia import (record_features)
from experiments.sd_membership_sft.archive.m1_fit import (CALIBRATION_RATES, FIT_VERSION, M1Data, _jsonable, evaluate_score_vector, fixed_probability_baseline_scores, fit_detector, make_partitions, probability_b2_values)
from experiments.sd_membership_sft.analysis.m1_features import (aggregate_matrix)


from experiments.paths import ROOT
DEFAULT_ROOT = ROOT / "experiments/results/sft_runs/m1_conditional"
DEFAULT_OUTPUT = DEFAULT_ROOT / "M1_RESULTS.md"


def experiment_spec() -> dict[str, Any]:
    return {
        "spec_version": "m1-2026-09-09-v1",
        "source": "research/M1草稿激活条件校准_Qwen3已微调模型实验方案_2026-09-09.md",
        "status": "implementation_and_development_experiment",
        "target_model": "Qwen/Qwen3-8B-Base",
        "draft_model": "Qwen/Qwen3-1.7B-Base",
        "benchmarks": ["wikitection", "newstection", "arxivtection"],
        "epochs": [1, 3],
        "roles": ["draft_auxiliary_distilled", "draft_member_sft"],
        "q_features": [
            "candidate log q",
            "normalized entropy",
            "candidate rank",
            "top1 margin",
            "response relative position",
            "log length",
        ],
        "activation_blocks_zero_based": [6, 13, 20, 27],
        "activation_definition": "decoder block output residual stream before final RMSNorm",
        "activation_statistics": [
            "mean",
            "population std",
            "RMS",
            "mean absolute value",
            "min",
            "max",
            "q10",
            "q50",
            "q90",
            "positive fraction",
        ],
        "partitions": {
            "nuisance_fit": "400 nonmembers",
            "nuisance_location": "300 nonmembers",
            "nuisance_scale": "100 nonmembers",
            "detector_fit": "800 members + 400 nonmembers",
            "validation": "400 members + 400 nonmembers",
            "calibration": "400 members + 400 nonmembers; only nonmembers calibrate",
            "test": "400 members + 400 nonmembers",
        },
        "split_seed": 20260824,
        "nuisance_seed": 20260909,
        "detector_seeds": [20260909, 20260910, 20260911],
        "conditional_l2": [1e-3, 1e-2, 1e-1],
        "detector_l2": [1e-3, 1e-2, 1e-1, 1.0],
        "scale_floor_fractions": [0.1, 0.25],
        "calibration_rates": list(CALIBRATION_RATES),
        "bootstrap_repeats": 2000,
        "pauc": "integral TPR(f) df over [0,.05] / .05; tied scores are one threshold group",
        "threshold": "(1 + count(calibration_score >= score))/(n+1) <= eta; inclusive ties",
        "development_gate": "four difficult conditions: positive V pAUC versus B2, approximately +5 percentage points auxiliary V TPR@nominal 1%, stable three detector seeds, no obvious FPR deterioration",
    }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run_probability_baseline(
    probability_dir: Path,
    output_dir: Path,
    role: str,
    bootstrap_repeats: int,
    partition_manifest: Path,
) -> dict[str, Any]:
    """Run B0/B1/B2 using only an existing probability archive."""

    probability_dir = probability_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = np.load(probability_dir / "pq_gap_scores.npz", allow_pickle=False)
    tokens = np.load(probability_dir / "pq_gap_token_logps.npz", allow_pickle=False)
    labels = np.asarray(scores["labels"], dtype=np.int64)
    record_ids = np.asarray(scores["record_ids"])
    lengths = np.asarray(tokens["lengths"], dtype=np.int64)
    target = np.asarray(tokens["target"], dtype=np.float64)
    draft = np.asarray(tokens[role], dtype=np.float64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    rows = [
        record_features(target[int(start) : int(end)], draft[int(start) : int(end)])
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    names = tuple(rows[0])
    f19 = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    # A zero activation matrix is sufficient for this probability-only dummy
    # container; B0/B1/B2 never read H. q[:,0] remains the exact draft log q
    # so the identity of the reconstructed F19 is auditable.
    q = np.zeros((len(target), 6), dtype=np.float32)
    q[:, 0] = draft.astype(np.float32)
    h = np.zeros((len(target), 40), dtype=np.float32)
    data = M1Data(
        feature_dir=probability_dir,
        probability_dir=probability_dir,
        role=role,
        labels=labels,
        record_ids=record_ids,
        lengths=lengths,
        offsets=offsets,
        q=q,
        h=h,
        target_logp=target,
        draft_logq=draft,
        delta=target - draft,
        f19=f19,
        f19_names=names,
        eos_mask=np.zeros(len(target), dtype=bool),
        feature_manifest={"benchmark": "unknown", "epoch": "unknown", "eos_included": True},
        probability_manifest=None,
    )
    partitions = make_partitions(
        labels,
        record_ids,
        frozen_manifest_path=partition_manifest,
    )
    fixed = fixed_probability_baseline_scores(f19, names)
    delta = target - draft
    b2_values = probability_b2_values(f19, aggregate_matrix(delta, lengths))
    detector = fit_detector("B2/logistic", b2_values, data, partitions, "logistic")
    b1_detector = fit_detector("B1/logistic", f19, data, partitions, "logistic")
    method_scores = dict(fixed)
    method_scores["B1/logistic"] = b1_detector.scores(f19, 20260909)
    method_scores["B2/logistic"] = detector.scores(b2_values, 20260909)
    bootstrap = None
    if bootstrap_repeats > 0:
        from experiments.sd_membership_sft.archive.m1_fit import (make_bootstrap_indices)

        bootstrap = make_bootstrap_indices(
            int(np.sum(labels[partitions["test"]] == 1)),
            int(np.sum(labels[partitions["test"]] == 0)),
            int(np.sum(labels[partitions["calibration"]] == 0)),
            bootstrap_repeats,
            20260909 + 50_000,
        )
    report = {
        "protocol": {
            "kind": "probability_only_cpu_baseline",
            "probability_dir": str(probability_dir),
            "role": role,
            "partition_manifest": str(partition_manifest.resolve()),
            "bootstrap_repeats": bootstrap_repeats,
        },
        "partitions": {
            name: {"n": int(len(index)), "members": int(np.sum(labels[index] == 1)), "nonmembers": int(np.sum(labels[index] == 0))}
            for name, index in partitions.indices.items()
        },
        "methods": {
            name: evaluate_score_vector(values, data, partitions, bootstrap, bootstrap_repeats)
            for name, values in method_scores.items()
        },
        "detector_models": {
            "B1/logistic": {"regularization": b1_detector.regularization, "selection": b1_detector.selection},
            "B2/logistic": {"regularization": detector.regularization, "selection": detector.selection},
        },
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# M1 CPU probability baselines",
        "",
        f"- Probability cache: `{probability_dir}`",
        f"- Role: `{role}`; B1/B2 are trained on D and selected on V; C/T are withheld until evaluation.",
        "",
        "| Method | Test AUC | pAUC[0,.05] | TPR@1% | actual FPR@1% |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["methods"].items():
        test = result["test"]
        one = test["calibrated"]["1%"]
        lines.append(
            f"| `{name}` | {test['auc']['point']:.4f} | {test['pauc_0_05']['point']:.4f} | "
            f"{one['tpr']:.4f} ({one['member_hits']}/{one['member_n']}) | "
            f"{one['fpr']:.4f} ({one['nonmember_hits']}/{one['nonmember_n']}) |"
        )
    (output_dir / "BASELINE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def aggregate_reports(input_root: Path) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(input_root.glob("**/m1_metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not _report_matches_feature_provenance(report):
            skipped.append({"path": str(path.resolve()), "reason": "feature/probability provenance mismatch"})
            continue
        feature_dir_text = report.get("protocol", {}).get("feature_dir")
        expected_output = Path(str(feature_dir_text)).resolve() / path.parent.name
        if path.parent.resolve() != expected_output:
            skipped.append({"path": str(path.resolve()), "reason": "report is not under its feature directory"})
            continue
        report["_path"] = str(path.resolve())
        reports.append(report)
    return {"reports": reports, "count": len(reports), "skipped_reports": skipped}


def _report_matches_feature_provenance(report: dict[str, Any]) -> bool:
    """Exclude stale reports whose feature/EOS/probability chain no longer agrees."""

    protocol = report.get("protocol", {})
    feature_dir_text = protocol.get("feature_dir")
    if not feature_dir_text:
        # Keep legacy probability-only reports out of the M1 condition aggregate.
        return False
    feature_dir = Path(str(feature_dir_text)).resolve()
    manifest_path = feature_dir / "feature_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_eos = bool(protocol.get("eos_included", True))
        manifest_eos = bool(manifest.get("eos_included", True))
        if report_eos != manifest_eos:
            return False
        probability_dir = Path(str(protocol.get("probability_dir", ""))).resolve()
        if report_eos:
            expected = Path(str(manifest["probability_cache"]["path"])).resolve().parent
        else:
            expected_text = manifest.get("probability_cache_alignment", {}).get(
                "reconciled_probability_dir"
            )
            if not expected_text:
                return False
            expected = Path(str(expected_text)).resolve()
        return probability_dir == expected
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _report_variant(report: dict[str, Any]) -> str:
    path = report.get("_path")
    return Path(str(path)).parent.name if path else "unknown"


def _primary_row(
    report: dict[str, Any], method: str
) -> tuple[str, str, str, str, float, float, float, float] | None:
    result = report.get("methods", {}).get(method)
    if result is None:
        return None
    test = result["test"]
    one = test["calibrated"]["1%"]
    return (
        str(report["protocol"].get("benchmark")),
        str(report["protocol"].get("epoch")),
        _report_variant(report),
        method,
        float(test["auc"]["point"]),
        float(test["pauc_0_05"]["point"]),
        float(one["tpr"]),
        float(one["fpr"]),
    )


def render_aggregate(root: Path, aggregate: dict[str, Any], output_path: Path) -> None:
    rows: list[tuple[str, str, str, str, float, float, float, float]] = []
    for report in aggregate["reports"]:
        for method in ("B1/logistic", "B2/logistic", "MQ/logistic", "MQH/logistic", "B1/mlp", "B2/mlp", "MQ/mlp", "MQH/mlp"):
            row = _primary_row(report, method)
            if row is not None:
                rows.append(row)
    lines = [
        "# M1 results aggregate",
        "",
        f"- Input root: `{root}`",
        f"- Valid per-condition reports found: {aggregate['count']}",
        f"- Stale/inconsistent reports excluded by provenance checks: {len(aggregate.get('skipped_reports', []))}",
        "- This file aggregates frozen per-condition reports; it does not select a method using C/T.",
        "- Test uncertainty and zero-FP one-sided bounds are retained in each `m1_metrics.json`.",
        "",
        "## Core methods",
        "",
        "| Benchmark | Epoch | Variant | Method | Test AUC | pAUC[0,.05] | TPR@1% | actual FPR@1% |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for benchmark, epoch, variant, method, auc, pauc, tpr, fpr in rows:
        lines.append(f"| {benchmark} | {epoch} | `{variant}` | `{method}` | {auc:.4f} | {pauc:.4f} | {tpr:.4f} | {fpr:.4f} |")
    lines.extend(["", "## Development gate snapshot", ""])
    difficult = {
        ("wikitection", "1"),
        ("wikitection", "3"),
        ("newstection", "1"),
        ("arxivtection", "1"),
    }
    gate_rows: list[tuple[str, str, float, float, float, float]] = []
    for report in aggregate["reports"]:
        # Only the four pre-registered real-H linear/logistic reports define
        # the resource gate.  Ablations and MLPs are diagnostics, not extra
        # opportunities to average or select the gate.
        if _report_variant(report) != "fit_linear_logistic":
            continue
        key = (str(report["protocol"].get("benchmark")), str(report["protocol"].get("epoch")))
        if key not in difficult:
            continue
        b2 = report.get("methods", {}).get("B2/logistic")
        mqh = report.get("methods", {}).get("MQH/logistic")
        if b2 is None or mqh is None:
            continue
        b2_v = b2["validation"]["pauc_0_05"]
        mqh_v = mqh["validation"]["pauc_0_05"]
        b2_t = b2["validation"]["tpr_at_nominal_1pct"]
        mqh_t = mqh["validation"]["tpr_at_nominal_1pct"]
        gate_rows.append((key[0], key[1], b2_v, mqh_v, b2_t, mqh_t))
    lines.extend(["| Benchmark | Epoch | B2 V pAUC | MQH V pAUC | B2 V TPR@1% | MQH V TPR@1% |", "|---|---:|---:|---:|---:|---:|"])
    for row in gate_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} | {row[5]:.4f} |")
    if gate_rows:
        mean_delta_pauc = float(np.mean([row[3] - row[2] for row in gate_rows]))
        mean_delta_tpr = float(np.mean([row[5] - row[4] for row in gate_rows]))
        lines.extend(["", f"Difficult-condition V mean ΔpAUC (MQH−B2): `{mean_delta_pauc:+.4f}`; mean ΔTPR@1%: `{mean_delta_tpr:+.4f}`."])
    else:
        lines.extend(["", "No complete difficult-condition pair is available yet; the resource gate is not assessed."])
    lines.extend(["", "## Artefacts", "", "Each condition directory contains `partition_manifest.json`, `m1_metrics.json`, `M1_RESULTS.md`, `frozen_scores.npz`, and `models/`; extraction directories contain `feature_manifest.json`, `q.npy`, `h.npy`, and position metadata.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_ablation_report(root: Path, aggregate: dict[str, Any], output_path: Path) -> None:
    """Write an explicit point-estimate report for mechanism diagnostics."""

    detailed_variants = {
        "fit_mlp_all": "conditional MLP + real H",
        "fit_noise_linear_logistic": "linear + Gaussian noise H",
        "fit_no_eos_linear_logistic": "linear + no appended EOS",
    }
    detailed: list[tuple[str, str, str, float, float, float, float, float]] = []
    permutation: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for report in aggregate["reports"]:
        variant = _report_variant(report)
        b2 = report.get("methods", {}).get("B2/logistic")
        mqh = report.get("methods", {}).get("MQH/logistic")
        if b2 is None or mqh is None:
            continue
        condition = f"{report['protocol'].get('benchmark')}_epoch{report['protocol'].get('epoch')}"
        v_delta_pauc = float(mqh["validation"]["pauc_0_05"] - b2["validation"]["pauc_0_05"])
        v_delta_tpr = float(mqh["validation"]["tpr_at_nominal_1pct"] - b2["validation"]["tpr_at_nominal_1pct"])
        m_test = mqh["test"]
        b_test = b2["test"]
        test_delta_pauc = float(m_test["pauc_0_05"]["point"] - b_test["pauc_0_05"]["point"])
        test_delta_tpr = float(m_test["calibrated"]["1%"]["tpr"] - b_test["calibrated"]["1%"]["tpr"])
        test_delta_fpr = float(m_test["calibrated"]["1%"]["fpr"] - b_test["calibrated"]["1%"]["fpr"])
        if variant in detailed_variants:
            detailed.append((condition, detailed_variants[variant], "-", v_delta_pauc, v_delta_tpr, test_delta_pauc, test_delta_tpr, test_delta_fpr))
        elif variant.startswith("fit_permute_linear_logistic_seed"):
            seed = int(variant.rsplit("seed", 1)[1])
            permutation.setdefault(condition, []).append((seed, v_delta_pauc, v_delta_tpr, test_delta_pauc, test_delta_tpr))

    lines = [
        "# M1 mechanism and capacity ablations",
        "",
        "These are development point estimates. The linear/logistic real-H main line has the registered 2000-bootstrap intervals in each `fit_linear_logistic/m1_metrics.json`; these diagnostic runs used `--no-bootstrap` and were not used to reopen method selection.",
        "",
        "## MLP, noise, and no-EOS diagnostics",
        "",
        "| Condition | Variant | Seed | ΔV pAUC (MQH−B2) | ΔV TPR@1% | Δtest pAUC | Δtest TPR@1% | Δtest FPR@1% |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(detailed):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:+.4f} | {row[4]:+.4f} | {row[5]:+.4f} | {row[6]:+.4f} | {row[7]:+.4f} |")
    complete_permutation = {
        condition: values for condition, values in permutation.items() if len(values) == 5
    }
    lines.extend(["", "## Within-bucket permutation (five fixed seeds)", ""])
    if complete_permutation:
        lines.extend(["| Condition | Seeds | mean ΔV pAUC | sd ΔV pAUC | mean ΔV TPR@1% | sd ΔV TPR@1% | mean Δtest pAUC | mean Δtest TPR@1% |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for condition in sorted(complete_permutation):
            values = complete_permutation[condition]
            v_pauc = np.asarray([row[1] for row in values], dtype=np.float64)
            v_tpr = np.asarray([row[2] for row in values], dtype=np.float64)
            t_pauc = np.asarray([row[3] for row in values], dtype=np.float64)
            t_tpr = np.asarray([row[4] for row in values], dtype=np.float64)
            seed_text = ", ".join(str(row[0]) for row in sorted(values))
            lines.append(f"| {condition} | {seed_text} | {v_pauc.mean():+.4f} | {v_pauc.std(ddof=0):.4f} | {v_tpr.mean():+.4f} | {v_tpr.std(ddof=0):.4f} | {t_pauc.mean():+.4f} | {t_tpr.mean():+.4f} |")
    else:
        lines.append("Permutation results are not reported: the five-seed ablation was intentionally stopped before completion.")
    lines.extend(["", "Interpretation rule: a permutation result near the real-H gain weakens an activation-specific claim; noise near the real-H result weakens a capacity/dimension claim. The resource gate is assessed only from the real-H linear/logistic table in `M1_RESULTS.md`.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_cost_report(root: Path, output_path: Path) -> None:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("**/feature_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("role") == "draft_auxiliary_distilled":
            manifests.append((path, manifest))
    lines = [
        "# M1 extraction cost and provenance",
        "",
        "Costs below are measured from completed feature manifests; the original p scoring budget remains a reused input and is not counted as zero queries.",
        "",
        "| Benchmark | Epoch | Role | EOS | Records | Scoring tokens | Q/H bytes | Seconds | Peak GPU GiB |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    total_tokens = 0
    total_bytes = 0
    for _path, manifest in manifests:
        total_tokens += int(manifest.get("total_tokens", 0))
        arrays = manifest.get("arrays", {})
        feature_dir = _path.parent
        q_path = feature_dir / "q.npy"
        h_path = feature_dir / "h.npy"
        qh_bytes = (q_path.stat().st_size if q_path.exists() else 0) + (h_path.stat().st_size if h_path.exists() else 0)
        total_bytes += qh_bytes
        peak = manifest.get("peak_gpu_memory_bytes")
        peak_gib = float(peak) / (1024**3) if peak is not None else float("nan")
        eos = "yes" if manifest.get("eos_included", True) else "no"
        lines.append(f"| {manifest.get('benchmark')} | {manifest.get('epoch')} | {manifest.get('role')} | {eos} | {manifest.get('records')} | {manifest.get('total_tokens')} | {qh_bytes / 1e9:.3f} GB | {float(manifest.get('seconds', 0.0)):.1f} | {peak_gib:.2f} |")
    lines.extend(["", f"- Total listed scoring positions: `{total_tokens:,}`.", f"- Total listed Q/H array volume: `{total_bytes / 1e9:.3f} GB`.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-spec", action="store_true")
    parser.add_argument("--baseline-cache", type=Path, default=None)
    parser.add_argument("--baseline-role", choices=("draft_auxiliary_distilled", "draft_member_sft"), default="draft_auxiliary_distilled")
    parser.add_argument("--baseline-output", type=Path, default=None)
    parser.add_argument(
        "--baseline-partition-manifest",
        type=Path,
        default=None,
        help="Frozen record_id partition manifest for --baseline-cache.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve(args.input_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.write_spec:
        (root / "experiment_spec.json").write_text(
            json.dumps(experiment_spec(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.baseline_cache is not None:
        cache = _resolve(args.baseline_cache)
        baseline_output = _resolve(args.baseline_output) if args.baseline_output is not None else root / "cpu_baseline"
        baseline_partition = (
            _resolve(args.baseline_partition_manifest)
            if args.baseline_partition_manifest is not None
            else baseline_output / "partition_manifest.json"
        )
        run_probability_baseline(
            cache,
            baseline_output,
            args.baseline_role,
            args.bootstrap_repeats,
            baseline_partition,
        )
    aggregate = aggregate_reports(root)
    render_aggregate(root, aggregate, _resolve(args.output))
    render_ablation_report(root, aggregate, root / "M1_ABLATIONS.md")
    render_cost_report(root, root / "M1_COSTS.md")
    (root / "aggregate_metrics.json").write_text(
        json.dumps(
            {"fit_version": FIT_VERSION, "generated_at": time.time(), "python": platform.python_version(), **_jsonable(aggregate)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"reports": aggregate["count"], "output": str(_resolve(args.output))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
