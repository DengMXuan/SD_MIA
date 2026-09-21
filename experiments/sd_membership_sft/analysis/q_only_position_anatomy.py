"""E1b: test whether draft-q alone can locate membership-informative tokens.

This is an explicitly exploratory follow-up to E1.  Token selection reads only
the white-box draft log-probability; exact delta is used only to score the
selected positions and establish an offline upper diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sd_membership_sft.core.replay_cache import (load_delta_data)
from experiments.sd_membership_sft.analysis.token_signal_anatomy import (ROOT, build_roles, fast_partial_auc)
from experiments.sd_membership_sft.core.replay_cache import (drop_final_cached_token, load_paired_logps)


FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
SHORT = ("wikitection", "newstection")
LONG = ("arxivtection",)
EPOCHS = (1, 3)


def q_selected_positive_fraction_scores(
    delta: np.ndarray,
    logq: np.ndarray,
    lengths: np.ndarray,
    fractions: tuple[float, ...] = FRACTIONS,
    random_repeats: int = 5,
    seed: int = 20260920,
) -> dict[str, np.ndarray]:
    """Return low-q/high-q/random selected positive-delta fractions per record."""
    delta = np.asarray(delta, dtype=np.float64)
    logq = np.asarray(logq, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if delta.shape != logq.shape or len(delta) != int(np.sum(lengths)):
        raise ValueError("delta, logq, and lengths are not aligned")
    if random_repeats <= 0 or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("fractions and random_repeats must be valid")
    result = {"global_positive_fraction": np.empty(len(lengths), dtype=np.float64)}
    for fraction in fractions:
        tag = f"{round(100 * fraction):02d}pct"
        for selector in ("lowq", "highq", "random"):
            result[f"{selector}_positive_fraction_{tag}"] = np.empty(len(lengths), dtype=np.float64)
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        values = delta[int(start):int(end)]
        q = logq[int(start):int(end)]
        positive = values > 0.0
        result["global_positive_fraction"][index] = float(np.mean(positive))
        order = np.argsort(q, kind="stable")
        random_sums = np.zeros(len(fractions), dtype=np.float64)
        for repeat in range(random_repeats):
            permutation = np.random.default_rng(
                np.random.SeedSequence([seed, index, repeat])
            ).permutation(len(values))
            prefix = np.cumsum(positive[permutation], dtype=np.float64)
            for fraction_index, fraction in enumerate(fractions):
                count = max(1, int(math.ceil(fraction * len(values))))
                random_sums[fraction_index] += prefix[count - 1] / count
        for fraction_index, fraction in enumerate(fractions):
            count = max(1, int(math.ceil(fraction * len(values))))
            tag = f"{round(100 * fraction):02d}pct"
            result[f"lowq_positive_fraction_{tag}"][index] = float(np.mean(positive[order[:count]]))
            result[f"highq_positive_fraction_{tag}"][index] = float(np.mean(positive[order[-count:]]))
            result[f"random_positive_fraction_{tag}"][index] = random_sums[fraction_index] / random_repeats
    return result


def _pauc(values: np.ndarray, member: np.ndarray, nonmember: np.ndarray) -> float:
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    return fast_partial_auc(np.r_[values[member], values[nonmember]], labels)


def _write_json(path: Path, value: Any) -> None:
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


def run(output_dir: Path, repeats: int = 1000, random_repeats: int = 5) -> dict[str, Any]:
    conditions: dict[tuple[str, int], dict[str, Any]] = {}
    result_root = ROOT / "experiments/results/sft_runs"
    for benchmark in SHORT + LONG:
        for epoch in EPOCHS:
            name = f"{benchmark}_epoch{epoch}"
            data = drop_final_cached_token(load_delta_data(
                result_root / "full_delta" / name / "draft_auxiliary_distilled/full_delta.npz"
            ))
            _, logq = load_paired_logps(
                result_root / "pq_directional" / name / "pq_gap_token_logps.npz", data, True
            )
            conditions[(benchmark, epoch)] = {
                "labels": data.labels,
                "record_ids": data.record_ids.astype(str),
                "roles": build_roles(data.labels, 20260824),
                "scores": q_selected_positive_fraction_scores(
                    data.delta, logq, data.lengths, random_repeats=random_repeats
                ),
            }
    for benchmark in SHORT + LONG:
        first, second = conditions[(benchmark, 1)], conditions[(benchmark, 3)]
        if not np.array_equal(first["record_ids"], second["record_ids"]):
            raise ValueError(f"record alignment failed across {benchmark} epochs")

    rng = np.random.default_rng(20260921)
    draws: dict[tuple[str, str], np.ndarray] = {}
    for benchmark in SHORT + LONG:
        roles = conditions[(benchmark, 1)]["roles"]
        for role_name in ("t_member", "t_nonmember"):
            values = getattr(roles, role_name)
            draws[(benchmark, role_name)] = values[
                rng.integers(0, len(values), size=(repeats, len(values)))
            ]

    rows: dict[str, Any] = {}
    baseline_name = "global_positive_fraction"
    for fraction in FRACTIONS:
        tag = f"{round(100 * fraction):02d}pct"
        for selector in ("lowq", "highq", "random"):
            method = f"{selector}_positive_fraction_{tag}"
            short_points, long_points = [], []
            short_boot = np.zeros(repeats)
            long_boot = np.zeros(repeats)
            per_condition: dict[str, Any] = {}
            for benchmark in SHORT + LONG:
                roles = conditions[(benchmark, 1)]["roles"]
                for epoch in EPOCHS:
                    scores = conditions[(benchmark, epoch)]["scores"]
                    x, baseline = scores[method], scores[baseline_name]
                    point_x = _pauc(x, roles.t_member, roles.t_nonmember)
                    point_b = _pauc(baseline, roles.t_member, roles.t_nonmember)
                    per_condition[f"{benchmark}_epoch{epoch}"] = {
                        "method_pauc": point_x,
                        "baseline_pauc": point_b,
                        "difference": point_x - point_b,
                    }
                    sample = np.empty(repeats)
                    for repeat in range(repeats):
                        member = draws[(benchmark, "t_member")][repeat]
                        nonmember = draws[(benchmark, "t_nonmember")][repeat]
                        sample[repeat] = _pauc(x, member, nonmember) - _pauc(baseline, member, nonmember)
                    if benchmark in SHORT:
                        short_points.append(point_x - point_b)
                        short_boot += sample / (len(SHORT) * len(EPOCHS))
                    else:
                        long_points.append(point_x - point_b)
                        long_boot += sample / (len(LONG) * len(EPOCHS))
            rows[method] = {
                "selector_uses": "draft q only",
                "conditions": per_condition,
                "short_macro_difference": {
                    "point": float(np.mean(short_points)),
                    "ci95_low": float(np.quantile(short_boot, 0.025)),
                    "ci95_high": float(np.quantile(short_boot, 0.975)),
                },
                "short_positive_conditions": int(sum(value["difference"] > 0 for key, value in per_condition.items() if not key.startswith("arxiv"))),
                "arxiv_macro_difference": {
                    "point": float(np.mean(long_points)),
                    "ci95_low": float(np.quantile(long_boot, 0.025)),
                    "ci95_high": float(np.quantile(long_boot, 0.975)),
                },
            }
    ranked = sorted(rows, key=lambda name: rows[name]["short_macro_difference"]["point"], reverse=True)
    report = {
        "experiment": "E1b draft-q-only position anatomy",
        "protocol": {
            "status": "post-hoc exploratory follow-up prompted by E1 q-bin monotonicity",
            "selection_permission": "token subset uses draft logq only; exact delta is used only for offline scoring",
            "fractions": list(FRACTIONS),
            "random_repeats": random_repeats,
            "bootstrap_repeats": repeats,
            "bootstrap": "record-paired across epochs within each domain",
            "short_conditions": [f"{b}_epoch{e}" for b in SHORT for e in EPOCHS],
            "arxiv_conditions": [f"{b}_epoch{e}" for b in LONG for e in EPOCHS],
        },
        "ranked_methods": ranked,
        "methods": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "E1B_Q_ONLY_REPORT.json", report)
    lines = [
        "# E1b Draft-q-Only Position Anatomy",
        "",
        "> Post-hoc exploratory analysis. q selects positions; exact delta only evaluates them.",
        "",
        "| Method | Short macro ΔpAUC | 95% CI | Positive | ArXiv ΔpAUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ranked:
        row = rows[method]
        short, long = row["short_macro_difference"], row["arxiv_macro_difference"]
        lines.append(
            f"| `{method}` | {short['point']:+.4f} | [{short['ci95_low']:+.4f}, {short['ci95_high']:+.4f}] | "
            f"{row['short_positive_conditions']}/4 | {long['point']:+.4f} |"
        )
    (output_dir / "E1B_Q_ONLY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "experiments/results/sft_runs/accept_only_active_v2/e1_q_only_positions"
    parser.add_argument("--output-dir", type=Path, default=default)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-repeats", type=int, default=5)
    args = parser.parse_args()
    report = run(args.output_dir.resolve(), args.bootstrap_repeats, args.random_repeats)
    print(json.dumps({"best": report["ranked_methods"][0], "output": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
