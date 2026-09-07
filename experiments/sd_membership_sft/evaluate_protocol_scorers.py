"""Evaluate the offline P0 protocol-scorer package on replay dumps.

Consumes ``.npz`` dumps produced by ``transcript_replay`` and computes, for
every (selection strategy x verifier budget) cell, the P0 scorer battery plus
attacker-observable diagnostics, then reports:

- AUC with bootstrap CI, TPR@0.1%/1%/5% FPR and an 80%-power minimal
  detectable AUC effect (2.8 x bootstrap SE) per scorer;
- paired bootstrap dAUC against ``mean_acceptance`` (the repo's headline
  transcript statistic) on the same test records with the same selection and
  budget;
- cross-seed aggregates for conditions replayed under multiple training seeds.

Everything is offline numpy: one replay dump serves all selections, budgets
(first-R' bit slices) and scorers. The threat model is draft white-box plus
protocol feedback only; no score reads the target's logits directly (alpha is
the R->inf limit of the acceptance channel, attacker-observable in
expectation).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit import (
    bootstrap_auc_ci,
    tpr_at_fpr,
    auc_rank,
)
from .draver_activation import paired_bootstrap_delta
from .protocol_scorers import (
    NullRates,
    acceptance_lira,
    fit_null_rates,
    gather_selection,
    honest_alpha,
    mean_acceptance,
    min_a_percent,
    qbin_conditional_means,
    saturation_rate,
    standardize_qbin,
    surp_sd,
    wbc_sd,
)

SELECTIONS = ("mink20", "entropy_low", "random_k", "all")
BUDGETS = (1, 4, 24)
BOOTSTRAP = 200


def select_positions(
    strategy: str,
    draft_logp: np.ndarray,
    draft_entropy: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    fraction: float = 0.2,
) -> np.ndarray:
    rows, width = valid.shape
    positions = np.zeros((rows, width), dtype=np.int64)
    for row in range(rows):
        valid_positions = np.flatnonzero(valid[row])
        if strategy == "all":
            chosen = valid_positions
        else:
            k = max(1, min(len(valid_positions), int(math.ceil(width * fraction))))
            if strategy == "mink20":
                order = np.argsort(
                    np.where(valid[row], draft_logp[row], np.inf), kind="mergesort"
                )
                chosen = order[:k]
            elif strategy == "entropy_low":
                order = np.argsort(
                    np.where(valid[row], draft_entropy[row], np.inf), kind="mergesort"
                )
                chosen = order[:k]
            elif strategy == "random_k":
                chosen = rng.choice(valid_positions, size=k, replace=False)
            else:
                raise ValueError(f"unknown selection strategy: {strategy}")
        positions[row, : len(chosen)] = chosen
        if len(chosen) < width:
            positions[row, len(chosen) :] = chosen[-1]
    return positions


def selection_masks(
    positions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row first-k selection validity: columns beyond a row's k are invalid."""
    rows, width = positions.shape
    counts = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        unique = len(np.unique(positions[row]))
        counts[row] = unique
    columns = np.arange(width)[None, :] < counts[:, None]
    gathered_valid = np.take_along_axis(valid, np.where(columns, positions, 0), axis=1)
    return columns & gathered_valid, columns


def scorer_battery(
    q_prob_sel: np.ndarray,
    counts_sel: np.ndarray,
    entropy_sel: np.ndarray,
    positions_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    null: NullRates,
    calibration_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    edges = null.edges
    means, bin_mask = qbin_conditional_means(q_prob_sel, counts_sel, repeats, valid_sel, edges)
    battery = {
        "mean_acceptance": mean_acceptance(counts_sel, repeats, valid_sel),
        "lira_mean": acceptance_lira(
            q_prob_sel, counts_sel, repeats, valid_sel, null, reduction="mean"
        ),
        "lira_sum": acceptance_lira(
            q_prob_sel, counts_sel, repeats, valid_sel, null, reduction="sum"
        ),
        "min_a20": min_a_percent(counts_sel, repeats, valid_sel, fraction=0.2),
        "saturation_rate": saturation_rate(counts_sel, repeats, valid_sel),
        "wbc_sd": wbc_sd(positions_sel, q_prob_sel, counts_sel, repeats, valid_sel, null),
        "qbin_conditional_z": standardize_qbin(means, bin_mask, calibration_rows),
        "surp_sd": surp_sd(entropy_sel, counts_sel, repeats, valid_sel),
    }
    return battery


def metric_row(y: np.ndarray, scores: np.ndarray, seed: int) -> dict[str, float]:
    finite = np.isfinite(scores)
    y_finite = y[finite]
    s_finite = scores[finite]
    low, high = bootstrap_auc_ci(y_finite, s_finite, BOOTSTRAP, seed)
    rng = np.random.default_rng(seed + 3)
    pos = np.flatnonzero(y_finite == 1)
    neg = np.flatnonzero(y_finite == 0)
    if len(pos) == 0 or len(neg) == 0:
        se = float("nan")
    else:
        bootstrap = []
        for _ in range(BOOTSTRAP):
            index = np.concatenate(
                [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
            )
            bootstrap.append(auc_rank(y_finite[index], s_finite[index]))
        se = float(np.std(bootstrap, ddof=1))
    return {
        "auc": float(auc_rank(y_finite, s_finite)),
        "auc_ci95_low": float(low),
        "auc_ci95_high": float(high),
        "tpr_at_0p1pct_fpr": float(tpr_at_fpr(y_finite, s_finite, 0.001)),
        "tpr_at_1pct_fpr": float(tpr_at_fpr(y_finite, s_finite, 0.01)),
        "tpr_at_5pct_fpr": float(tpr_at_fpr(y_finite, s_finite, 0.05)),
        # 80%-power minimal detectable AUC effect at alpha=0.05 (two-sided).
        "mde80_auc": float(2.8 * se) if np.isfinite(se) else float("nan"),
    }


def evaluate_dump(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    draft_logp = data["draft_logp"]
    draft_entropy = data["draft_entropy"]
    target_logp = data["target_logp"]
    bits = data["accept_bits"]
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    response_ids = data["response_ids"]

    repeats = int(meta["transcript_repeats"])
    alpha = honest_alpha(target_logp, draft_logp)
    valid = np.isfinite(alpha)
    y_test = labels[test_idx]
    calibration_rows = train_idx
    seed = int(meta["audit_seed"]) + int(meta["seed"]) % 1000

    results: dict = {
        "meta": meta,
        "source": str(path),
        "cells": [],
        "diagnostics": {},
    }

    # Attacker-observable diagnostics: draft white-box min-k plus the exact
    # acceptance expectation (R->inf limit of the protocol channel).
    draft_selected = np.argsort(
        np.where(valid, draft_logp, np.inf), axis=1, kind="mergesort"
    )[:, : max(1, int(math.ceil(valid.shape[1] * 0.2)))]
    direct = {
        "draft_only_min_k20": np.nanmean(
            np.take_along_axis(draft_logp, draft_selected, axis=1), axis=1
        ),
        "oracle_mean_alpha_all_positions": np.nanmean(alpha, axis=1),
    }
    for name, score in direct.items():
        results["diagnostics"][name] = metric_row(y_test, score[test_idx], seed)

    counts_full = bits.sum(axis=-1).astype(np.int64)
    selection_seeds = {"mink20": 11, "entropy_low": 12, "random_k": 13, "all": 14}
    for selection in SELECTIONS:
        rng = np.random.default_rng(seed + selection_seeds[selection])
        positions = select_positions(
            selection, draft_logp, draft_entropy, valid, rng
        )
        valid_sel, sel_mask = selection_masks(positions, valid)
        q_prob = np.exp(np.nan_to_num(draft_logp, nan=-np.inf))
        for budget in BUDGETS:
            if budget > repeats:
                continue
            counts = bits[..., :budget].sum(axis=-1).astype(np.int64)
            q_sel, q_valid = gather_selection(q_prob, positions, valid_sel)
            c_sel, c_valid = gather_selection(counts.astype(np.float64), positions, sel_mask)
            e_sel, _ = gather_selection(draft_entropy, positions, sel_mask)
            p_sel, _ = gather_selection(positions.astype(np.float64), positions, sel_mask)
            null = fit_null_rates(
                q_sel[calibration_rows], c_sel[calibration_rows], budget, c_valid[calibration_rows]
            )
            battery = scorer_battery(
                q_sel, c_sel.astype(np.int64), e_sel, p_sel, budget, c_valid, null, calibration_rows
            )
            baseline = battery["mean_acceptance"]
            for name, score in battery.items():
                row = metric_row(y_test, score[test_idx], seed)
                delta = paired_bootstrap_delta(
                    y_test, score[test_idx], baseline[test_idx], BOOTSTRAP, seed + 7
                )
                row["paired_delta_auc_vs_mean_acceptance"] = delta["delta_auc"]
                row["paired_delta_ci95"] = [delta["ci95_low"], delta["ci95_high"]]
                results["cells"].append(
                    {
                        "selection": selection,
                        "budget_repeats": budget,
                        "scorer": name,
                        **row,
                    }
                )
    return results


def aggregate_across_seeds(results: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in results:
        meta = item["meta"]
        groups[(meta["target_epochs"], meta["n_per_class"])].append(item)
    output = []
    for (epochs, n_per_class), items in sorted(groups.items()):
        by_cell: dict[tuple, list[dict]] = defaultdict(list)
        for item in items:
            for cell in item["cells"]:
                by_cell[(cell["selection"], cell["budget_repeats"], cell["scorer"])].append(cell)
        for (selection, budget, scorer), cells in sorted(by_cell.items()):
            aucs = [cell["auc"] for cell in cells]
            deltas = [cell["paired_delta_auc_vs_mean_acceptance"] for cell in cells]
            positive_ci = sum(
                1
                for cell in cells
                if cell["paired_delta_ci95"][0] > 0.0
            )
            output.append(
                {
                    "epochs": epochs,
                    "n_per_class": n_per_class,
                    "n_seeds": len(cells),
                    "selection": selection,
                    "budget_repeats": budget,
                    "scorer": scorer,
                    "auc_mean": float(np.mean(aucs)),
                    "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_positive_ci_seeds": positive_ci,
                }
            )
    return output


def render_markdown(per_file: list[dict], aggregates: list[dict]) -> str:
    lines = [
        "# P0 protocol-scorer package on replayed transcripts",
        "",
        "All rows are computed offline from one replay dump per condition;",
        "deltas are paired against `mean_acceptance` on the same test records.",
        "",
    ]
    for item in per_file:
        meta = item["meta"]
        lines.extend(
            [
                f"## {Path(item['source']).parent.name} — epochs={meta['target_epochs']} "
                f"seed={meta['seed']} n/class={meta['n_per_class']}",
                "",
                "### Attacker-observable diagnostics",
                "",
                "| Diagnostic | AUC [95% CI] | TPR@1%FPR |",
                "|---|---|---:|",
            ]
        )
        for name, row in item["diagnostics"].items():
            lines.append(
                f"| `{name}` | {row['auc']:.3f} [{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}] "
                f"| {row['tpr_at_1pct_fpr']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Protocol cells (min-k selection; all budgets)",
                "",
                "| Selection | R | Scorer | AUC [95% CI] | dAUC vs mean-accept [95% CI] | TPR@1%FPR |",
                "|---|---:|---|---|---|---:|",
            ]
        )
        for cell in item["cells"]:
            if cell["selection"] != "mink20":
                continue
            lines.append(
                f"| {cell['selection']} | {cell['budget_repeats']} | `{cell['scorer']}` "
                f"| {cell['auc']:.3f} [{cell['auc_ci95_low']:.3f}, {cell['auc_ci95_high']:.3f}] "
                f"| {cell['paired_delta_auc_vs_mean_acceptance']:+.3f} "
                f"[{cell['paired_delta_ci95'][0]:+.3f}, {cell['paired_delta_ci95'][1]:+.3f}] "
                f"| {cell['tpr_at_1pct_fpr']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Selection comparison at full budget (R=24)",
                "",
                "| Selection | Scorer | AUC | dAUC vs mean-accept |",
                "|---|---|---:|---:|",
            ]
        )
        for cell in item["cells"]:
            if cell["budget_repeats"] != 24:
                continue
            lines.append(
                f"| {cell['selection']} | `{cell['scorer']}` | {cell['auc']:.3f} "
                f"| {cell['paired_delta_auc_vs_mean_acceptance']:+.3f} |"
            )
        lines.append("")
    if aggregates:
        lines.extend(
            [
                "## Cross-seed aggregates (epoch-1 matrix)",
                "",
                "| epochs | n/class | seeds | Selection | R | Scorer | AUC mean (SD) | dAUC mean | #seeds ΔCI>0 |",
                "|---:|---:|---:|---|---:|---|---|---:|---:|",
            ]
        )
        for row in aggregates:
            if row["epochs"] != 1 or row["budget_repeats"] != 24:
                continue
            lines.append(
                f"| {row['epochs']} | {row['n_per_class']} | {row['n_seeds']} "
                f"| {row['selection']} | {row['budget_repeats']} | `{row['scorer']}` "
                f"| {row['auc_mean']:.3f} ({row['auc_sd']:.3f}) "
                f"| {row['delta_mean']:+.3f} | {row['delta_positive_ci_seeds']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_file = [evaluate_dump(path) for path in args.npz]
    aggregates = aggregate_across_seeds(per_file)
    (args.output_dir / "protocol_scorers.json").write_text(
        json.dumps({"per_file": per_file, "aggregates": aggregates}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        render_markdown(per_file, aggregates), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
