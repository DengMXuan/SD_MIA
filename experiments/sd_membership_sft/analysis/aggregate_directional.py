"""Aggregate the six directional/window SD membership reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paths import ROOT
RUNS = (
    ("wikitection", 1),
    ("wikitection", 3),
    ("newstection", 1),
    ("newstection", 3),
    ("arxivtection", 1),
    ("arxivtection", 3),
)
ROLES = ("draft_auxiliary_distilled", "draft_member_sft")
SCORES = (
    "p_mean_logp",
    "q_mean_logq",
    "mean_abs_delta",
    "mean_signed_delta",
    "mean_negative_delta",
    "mean_alpha",
    "window_sign_4",
    "window_sign_8",
    "window_sign_16",
    "window_sign_multiscale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/results/sft_runs/pq_directional"),
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    reports = {}
    for benchmark, epoch in RUNS:
        label = f"{benchmark}_epoch{epoch}"
        path = root / label / "directional_metrics.json"
        reports[label] = json.loads(path.read_text(encoding="utf-8"))

    aggregate = {
        "protocol": {
            "runs": [f"{benchmark}_epoch{epoch}" for benchmark, epoch in RUNS],
            "roles": list(ROLES),
            "scores": list(SCORES),
            "source": "experiments/results/sft_runs/pq_directional/*/directional_metrics.json",
            "interpretation": "full_pool is exploratory; calibrated_test thresholds use only 400 calibration nonmembers per run",
        },
        "runs": reports,
    }
    (root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Directional/window SD membership experiment",
        "",
        "## Scope",
        "",
        "- Models: the saved, controlled-SFT Qwen3-8B target and the saved Qwen3-1.7B auxiliary-distilled / member-SFT drafts.",
        "- Runs: WikiTection, NewsTection, and ArXivTection, each at target SFT epoch 1 and 3; 2,000 member + 2,000 nonmember records per run.",
        "- Token protocol: teacher-forced response tokens with the run's fixed prompt, prompt loss masked, EOS included; δ = log p − log q.",
        "- Scores: p-only, q-only, absolute gap, signed gap, negative part, α, fixed local sign windows (4/8/16), and the fixed multi-scale mean over {4, 8, 16, 32, 64}.",
        "- Full-pool metrics are exploratory because the score choice was motivated by the same benchmark family. Calibrated-test thresholds use only the 400 calibration nonmembers in the fixed 800/400/400/400 per-class partition; test has 400 members and 400 nonmembers.",
        "- Threshold rule: split-conformal upper-tail p-value `(1 + count(calibration score >= score))/(n+1) <= target FPR`; ties are counted inclusively and actual test FPR is retained.",
        "- TPR/FPR intervals use record-level percentile bootstrap with calibration and test records resampled separately; method increments use paired record bootstrap. No token bootstrap is used.",
        "",
        "## Main low-FPR table",
        "",
        "Values are `full-pool AUC / full-pool TPR@1% / calibrated-test AUC / calibrated-test TPR@1% / actual test FPR@1%`.",
        "",
        "| Run | Draft | signed | negative | α | S16 | multi-scale |",
        "|---|---|---|---|---|---|---|",
    ]
    for benchmark, epoch in RUNS:
        label = f"{benchmark}_epoch{epoch}"
        for role in ROLES:
            result = reports[label]["roles"][role]
            cells = []
            for score in ("mean_signed_delta", "mean_negative_delta", "mean_alpha", "window_sign_16", "window_sign_multiscale"):
                full = result[score]["full_pool"]
                test = result[score]["calibrated_test"]
                point = test["test_tpr_at_calibrated_fpr"]["1%"]
                cells.append(
                    "/".join(
                        fmt(value)
                        for value in (
                            full["auc"]["point"],
                            full["tpr_at_fpr"]["1%"]["tpr"],
                            test["test_auc"]["point"],
                            point["tpr"],
                            point["fpr"],
                        )
                    )
                )
            lines.append(f"| {label} | {role} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
        "## Calibrated-test low-FPR intervals",
        "",
        "Entries are `point [95% CI]`; the threshold, hit counts, sample sizes, and actual FPR are retained in each run's `directional_metrics.json`.",
        "",
        "| Run | Draft | Score | TPR@10% | FPR@10% | TPR@5% | FPR@5% | TPR@1% | FPR@1% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark, epoch in RUNS:
        label = f"{benchmark}_epoch{epoch}"
        for role in ROLES:
            result = reports[label]["roles"][role]
            for score in ("mean_signed_delta", "mean_negative_delta", "mean_alpha", "window_sign_16", "window_sign_multiscale"):
                points = result[score]["calibrated_test"]["test_tpr_at_calibrated_fpr"]
                cells = []
                for rate in ("10%", "5%", "1%"):
                    point = points[rate]
                    cells.extend(
                        [
                            f"{point['tpr']:.4f} [{point['tpr_ci95_low']:.4f}, {point['tpr_ci95_high']:.4f}]",
                            f"{point['fpr']:.4f} [{point['fpr_ci95_low']:.4f}, {point['fpr_ci95_high']:.4f}]",
                        ]
                    )
                lines.append(f"| {label} | {role} | `{score}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Full score inventory",
        "",
        "| Run | Draft | Score | Full AUC | Full TPR@10% | Full TPR@5% | Full TPR@1% | Test AUC | Test TPR@1% | Test FPR@1% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark, epoch in RUNS:
        label = f"{benchmark}_epoch{epoch}"
        for role in ROLES:
            result = reports[label]["roles"][role]
            for score in SCORES:
                full = result[score]["full_pool"]
                test = result[score]["calibrated_test"]
                point = test["test_tpr_at_calibrated_fpr"]["1%"]
                lines.append(
                    f"| {label} | {role} | `{score}` | {fmt(full['auc']['point'])} | "
                    f"{fmt(full['tpr_at_fpr']['10%']['tpr'])} | {fmt(full['tpr_at_fpr']['5%']['tpr'])} | "
                    f"{fmt(full['tpr_at_fpr']['1%']['tpr'])} | {fmt(test['test_auc']['point'])} | "
                    f"{fmt(point['tpr'])} | {fmt(point['fpr'])} |"
                )
    lines.extend(
        [
            "",
            "## Paired method increments",
            "",
            "Entries are `first method − baseline`, reported on the calibrated test split; intervals are paired record-bootstrap 95% CIs.",
            "",
            "| Run | Draft | Comparison | ΔAUC | ΔTPR@10% | ΔTPR@5% | ΔTPR@1% | ΔFPR@1% |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark, epoch in RUNS:
        label = f"{benchmark}_epoch{epoch}"
        for role in ROLES:
            comparisons = reports[label].get("method_deltas", {}).get(role, {})
            for comparison, result in comparisons.items():
                rate_points = result["tpr_at_calibrated_fpr"]
                tpr10 = rate_points["10%"]["tpr_delta"]
                tpr5 = rate_points["5%"]["tpr_delta"]
                tpr1 = rate_points["1%"]["tpr_delta"]
                fpr1 = rate_points["1%"]["fpr_delta"]
                auc = result["auc"]
                def format_ci(point: dict[str, float]) -> str:
                    return f"{point['point']:.4f} [{point['ci95_low']:.4f}, {point['ci95_high']:.4f}]"
                lines.append(
                    f"| {label} | {role} | `{comparison}` | {format_ci(auc)} | "
                    f"{format_ci(tpr10)} | {format_ci(tpr5)} | {format_ci(tpr1)} | {format_ci(fpr1)} |"
                )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Each run's model-facing archives are `target.npz`, `draft_auxiliary_distilled.npz`, and `draft_member_sft.npz`; merged token log-probabilities are `pq_gap_token_logps.npz`. The offline reports are generated with:",
            "",
            "```bash",
            ".venv/bin/python -m experiments.sd_membership_sft.directional_mia \\",
            "  --input experiments/results/sft_runs/pq_directional/<run>/pq_gap_token_logps.npz \\",
            "  --output-dir experiments/results/sft_runs/pq_directional/<run> \\",
            "  --benchmark <benchmark> --epoch <epoch> --bootstrap-repeats 500",
            "```",
            "",
            "The corrected model-facing pass uses FP32 vocabulary logsumexp in sequence chunks (64 positions) while retaining the BF16 model weights; role-isolated scoring uses per-batch CUDA cache cleanup. Offline calibration uses the inclusive-tie split-conformal rule above.",
            "",
            "## Interpretation boundary",
            "",
            "These are controlled SFT membership-audit results, not claims about Qwen3 pretraining membership. The teacher-forced p/q pass assumes the service can legally expose target probability for the audited prefix/candidate; it is not a passive normal-SD transcript measurement. The fixed test split is within the already-public benchmark pool, so it is not an independent new-data confirmation.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"results": str(root / "RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
