"""Aggregate E1 exact-delta diagnostics and evaluate preregistered Gate 1.

The short-fragment macro is Wiki/News x epoch 1/3.  Within a domain, every
bootstrap draw reuses the same record resample for both epochs, because the
underlying records and role assignments are identical.  ArXiv is reported
separately and never contributes to Gate 1.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .token_signal_anatomy import (
    ROOT,
    build_roles,
    fast_partial_auc,
    matching_global_baseline,
)


SHORT_BENCHMARKS = ("wikitection", "newstection")
LONG_BENCHMARKS = ("arxivtection",)
EPOCHS = (1, 3)


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


def _score_pauc(values: np.ndarray, member: np.ndarray, nonmember: np.ndarray) -> float:
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    return fast_partial_auc(np.r_[values[member], values[nonmember]], labels)


def _candidate(name: str) -> bool:
    if name.startswith("top_keep_"):
        return not name.endswith("_100pct")
    if name.startswith("window_max_"):
        return not name.endswith("_shuffled")
    return name.startswith("window_positive_rate_") and not name.endswith("_shuffled")


def load_condition(root: Path, benchmark: str, epoch: int, split_seed: int) -> dict[str, Any]:
    path = root / f"{benchmark}_epoch{epoch}" / "scores.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing completed E1 scores: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"labels", "record_ids", "lengths"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path} is missing {sorted(required - set(archive.files))}")
        labels = np.asarray(archive["labels"], dtype=np.int64)
        record_ids = np.asarray(archive["record_ids"]).astype(str)
        scores = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name not in required
        }
    return {
        "path": path,
        "labels": labels,
        "record_ids": record_ids,
        "roles": build_roles(labels, split_seed),
        "scores": scores,
    }


def _validate_epoch_alignment(conditions: dict[tuple[str, int], dict[str, Any]], benchmark: str) -> None:
    first, second = conditions[(benchmark, EPOCHS[0])], conditions[(benchmark, EPOCHS[1])]
    if not np.array_equal(first["labels"], second["labels"]):
        raise ValueError(f"labels differ across epochs for {benchmark}")
    if not np.array_equal(first["record_ids"], second["record_ids"]):
        raise ValueError(f"record IDs differ across epochs for {benchmark}")
    if set(first["scores"]) != set(second["scores"]):
        raise ValueError(f"score sets differ across epochs for {benchmark}")


def _interval(point: float, samples: np.ndarray) -> dict[str, float]:
    return {
        "point": float(point),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def aggregate(
    input_root: Path,
    output_dir: Path,
    split_seed: int = 20260824,
    bootstrap_repeats: int = 1000,
    bootstrap_seed: int = 20260917,
) -> dict[str, Any]:
    if bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive")
    conditions = {
        (benchmark, epoch): load_condition(input_root, benchmark, epoch, split_seed)
        for benchmark in SHORT_BENCHMARKS + LONG_BENCHMARKS
        for epoch in EPOCHS
    }
    for benchmark in SHORT_BENCHMARKS + LONG_BENCHMARKS:
        _validate_epoch_alignment(conditions, benchmark)

    score_sets = [set(condition["scores"]) for condition in conditions.values()]
    common = set.intersection(*score_sets)
    candidates = sorted(name for name in common if _candidate(name))
    if not candidates:
        raise ValueError("no common preregistered top/window statistics found")

    rng = np.random.default_rng(bootstrap_seed)
    # Reuse a record resample across epoch1/3 within a domain.
    draws: dict[tuple[str, str], np.ndarray] = {}
    for benchmark in SHORT_BENCHMARKS + LONG_BENCHMARKS:
        roles = conditions[(benchmark, EPOCHS[0])]["roles"]
        for role_name in ("t_member", "t_nonmember"):
            role = getattr(roles, role_name)
            draws[(benchmark, role_name)] = role[
                rng.integers(0, len(role), size=(bootstrap_repeats, len(role)))
            ]

    rows: dict[str, Any] = {}
    for method in candidates:
        baseline = matching_global_baseline(method)
        condition_points: dict[str, dict[str, float]] = {}
        short_samples = np.zeros(bootstrap_repeats, dtype=np.float64)
        long_samples = np.zeros(bootstrap_repeats, dtype=np.float64)
        short_points: list[float] = []
        long_points: list[float] = []
        for benchmark in SHORT_BENCHMARKS + LONG_BENCHMARKS:
            roles = conditions[(benchmark, EPOCHS[0])]["roles"]
            for epoch in EPOCHS:
                condition = conditions[(benchmark, epoch)]
                method_values = condition["scores"][method]
                baseline_values = condition["scores"][baseline]

                def difference(member: np.ndarray, nonmember: np.ndarray) -> float:
                    return _score_pauc(method_values, member, nonmember) - _score_pauc(
                        baseline_values, member, nonmember
                    )

                point = difference(roles.t_member, roles.t_nonmember)
                name = f"{benchmark}_epoch{epoch}"
                condition_points[name] = {
                    "method_pauc": _score_pauc(method_values, roles.t_member, roles.t_nonmember),
                    "baseline_pauc": _score_pauc(baseline_values, roles.t_member, roles.t_nonmember),
                    "difference": point,
                }
                sample = np.empty(bootstrap_repeats, dtype=np.float64)
                member_draws = draws[(benchmark, "t_member")]
                nonmember_draws = draws[(benchmark, "t_nonmember")]
                for repeat in range(bootstrap_repeats):
                    sample[repeat] = difference(member_draws[repeat], nonmember_draws[repeat])
                if benchmark in SHORT_BENCHMARKS:
                    short_points.append(point)
                    short_samples += sample / (len(SHORT_BENCHMARKS) * len(EPOCHS))
                else:
                    long_points.append(point)
                    long_samples += sample / (len(LONG_BENCHMARKS) * len(EPOCHS))

        short = _interval(float(np.mean(short_points)), short_samples)
        long = _interval(float(np.mean(long_points)), long_samples)
        positive_short = sum(
            condition_points[f"{benchmark}_epoch{epoch}"]["difference"] > 0.0
            for benchmark in SHORT_BENCHMARKS
            for epoch in EPOCHS
        )
        rows[method] = {
            "matching_global_baseline": baseline,
            "conditions": condition_points,
            "short_fragment_macro_difference": short,
            "short_positive_conditions": positive_short,
            "long_fragment_macro_difference": long,
            "passes_gate1": bool(
                short["point"] >= 0.03
                and positive_short >= 3
                and short["ci95_low"] > 0.0
            ),
        }

    ranked = sorted(rows, key=lambda name: rows[name]["short_fragment_macro_difference"]["point"], reverse=True)
    passing = [name for name in ranked if rows[name]["passes_gate1"]]
    report = {
        "experiment": "E1 cross-condition Gate 1 aggregate",
        "protocol": {
            "input_root": str(input_root.resolve()),
            "short_fragment_conditions": [
                f"{benchmark}_epoch{epoch}" for benchmark in SHORT_BENCHMARKS for epoch in EPOCHS
            ],
            "long_fragment_conditions": [
                f"{benchmark}_epoch{epoch}" for benchmark in LONG_BENCHMARKS for epoch in EPOCHS
            ],
            "bootstrap_unit": "record",
            "paired_across_epochs_within_domain": True,
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_seed": bootstrap_seed,
            "comparison": "each sparse statistic versus its like-for-like global directional mean",
            "gate1_rule": "short macro >= +0.03, >=3/4 positive conditions, paired CI lower bound > 0",
            "status": "exploratory; existing V/T have been inspected before",
        },
        "gate1_passed": bool(passing),
        "passing_methods": passing,
        "ranked_methods": ranked,
        "methods": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "E1_AGGREGATE.json", report)
    lines = [
        "# E1 Cross-Condition Gate 1",
        "",
        "> Exploratory result. ArXiv is a separate 1024–2048-token generalization test and does not enter Gate 1.",
        "",
        f"Gate 1 passed: **{report['gate1_passed']}**",
        "",
        "| Method | Matching mean | Short macro ΔpAUC | 95% CI | Positive | ArXiv macro ΔpAUC | Gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in ranked:
        row = rows[method]
        short = row["short_fragment_macro_difference"]
        long = row["long_fragment_macro_difference"]
        lines.append(
            f"| `{method}` | `{row['matching_global_baseline']}` | {short['point']:+.4f} | "
            f"[{short['ci95_low']:+.4f}, {short['ci95_high']:+.4f}] | "
            f"{row['short_positive_conditions']}/4 | {long['point']:+.4f} | {row['passes_gate1']} |"
        )
    (output_dir / "E1_AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = ROOT / "experiments/results/sft_runs/accept_only_active_v2/e1_token_anatomy"
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_root)
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = aggregate(
        args.input_root.resolve(),
        args.output_dir.resolve(),
        args.split_seed,
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )
    print(json.dumps({
        "gate1_passed": report["gate1_passed"],
        "passing_methods": report["passing_methods"],
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
