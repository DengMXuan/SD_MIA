"""Compare equal-budget uniform, hybrid, full-AI, sequential, and oracle probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..active_importance_replay import (acceptance_probabilities, estimate_corrected_delta0, fragment_score, oracle_action_levels)
from ..replay_cache import (load_replay_data)
from ..audit_runtime import (_deterministic_subset)
from .adaptive_window_accept_only import (_token_mask, fit_nonmember_model, token_features)
from ..audit_metrics import (membership_metrics)
from ..audit_runtime import (split_indices)
from .neural_adaptive_accept_only import (ACTIVE_BUDGET, ProbeValueExamples, _standardize, fit_probe_value_model, nonmember_probe_value_targets, predict_evidence)
from ..audit_runtime import (BENCHMARKS, EPOCHS, N_CAL, N_REF, REPLAY_SEEDS, ROOT, SPLIT_SEED, _paths, _write_json)


METHODS = ("uniform", "hybrid", "full_ai_one_shot", "full_ai_sequential", "oracle")


def _top_indices(priority: np.ndarray, fraction: float) -> np.ndarray:
    priority = np.asarray(priority, dtype=np.float64)
    if priority.ndim != 1 or len(priority) == 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid priority/fraction")
    count = max(1, int(math.ceil(fraction * len(priority))))
    return np.argsort(-priority, kind="stable")[:count]


def uniform_counts(length: int, budget: int = ACTIVE_BUDGET) -> np.ndarray:
    if length <= 0 or budget < 2:
        raise ValueError("length/budget must preserve two pilots")
    return np.full(length, budget - 2, dtype=np.int64)


def hybrid_counts(priority: np.ndarray, budget: int = ACTIVE_BUDGET) -> np.ndarray:
    """Current 6/10 hybrid: four active everywhere and four extra on half."""
    priority = np.asarray(priority, dtype=np.float64)
    counts = np.full(len(priority), budget - 4, dtype=np.int64)
    remaining = (budget - 2) * len(priority) - int(np.sum(counts))
    selected = _top_indices(priority, 0.50)
    base, remainder = divmod(remaining, len(selected))
    counts[selected] += base
    counts[selected[:remainder]] += 1
    if int(np.sum(counts)) != (budget - 2) * len(priority):
        raise RuntimeError("hybrid allocation violated the exact budget")
    return counts


def full_one_shot_counts(priority: np.ndarray, budget: int = ACTIVE_BUDGET) -> np.ndarray:
    """Spend every post-pilot query on the learned top half."""
    priority = np.asarray(priority, dtype=np.float64)
    counts = np.zeros(len(priority), dtype=np.int64)
    selected = _top_indices(priority, 0.50)
    remaining = (budget - 2) * len(priority)
    base, remainder = divmod(remaining, len(selected))
    counts[selected] = base
    counts[selected[:remainder]] += 1
    if int(np.sum(counts)) != remaining:
        raise RuntimeError("one-shot allocation violated the exact budget")
    return counts


def sequential_round_counts(priority: np.ndarray, selected_fraction: float = 0.25) -> np.ndarray:
    """Allocate exactly one record length of decisions to the current top set."""
    priority = np.asarray(priority, dtype=np.float64)
    selected = _top_indices(priority, selected_fraction)
    counts = np.zeros(len(priority), dtype=np.int64)
    base, remainder = divmod(len(priority), len(selected))
    counts[selected] = base
    counts[selected[:remainder]] += 1
    if int(np.sum(counts)) != len(priority):
        raise RuntimeError("sequential round violated the exact budget")
    return counts


def _record_uniforms(seed: int, record: int, length: int, width: int = 40) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, record, 77123])).random(
        (length, width), dtype=np.float64
    )


def _pilot_observations(data: Any, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rates = np.empty(len(data.logp), dtype=np.float64)
    uniforms: list[np.ndarray] = []
    for record in range(len(data.lengths)):
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        draws = _record_uniforms(seed, record, end - start)
        alpha0 = acceptance_probabilities(
            data.logp[start:end], data.logq0[start:end]
        )[:, 0]
        rates[start:end] = np.mean(draws[:, :2] < alpha0[:, None], axis=1)
        uniforms.append(draws)
    return rates, uniforms


def _initial_counts(rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = len(rate)
    accepts = np.zeros((length, 5), dtype=np.int64)
    trials = np.zeros((length, 5), dtype=np.int64)
    accepts[:, 0] = np.rint(2.0 * rate).astype(np.int64)
    trials[:, 0] = 2
    return accepts, trials


def _apply_extras(
    alpha: np.ndarray,
    accepts: np.ndarray,
    trials: np.ndarray,
    uniforms: np.ndarray,
    allocated: np.ndarray,
    new_counts: np.ndarray,
    oracle_levels: np.ndarray | None = None,
) -> None:
    for token in np.flatnonzero(new_counts > 0):
        count = int(new_counts[token])
        for _ in range(count):
            offset = int(allocated[token])
            level = int(oracle_levels[token]) if oracle_levels is not None else 1 + offset % 4
            draw_column = 2 + offset
            trials[token, level] += 1
            accepts[token, level] += int(uniforms[token, draw_column] < alpha[token, level])
            allocated[token] += 1


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    result = np.empty(len(order), dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float64)
    return result


def _sequential_priority(
    base_priority: np.ndarray,
    estimate: np.ndarray,
    censoring: np.ndarray,
    allocated: np.ndarray,
) -> np.ndarray:
    learned = _rank01(base_priority)
    unresolved = 1.0 / np.sqrt(1.0 + allocated)
    unresolved *= 1.0 + 0.5 * (censoring != 0)
    width = min(8, len(estimate))
    local = np.convolve(unresolved, np.ones(width), mode="same") / np.convolve(
        np.ones(len(estimate)), np.ones(width), mode="same"
    )
    return (0.25 + learned) * (0.5 * unresolved + 0.5 * local)


def _allocation_summary(all_counts: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate(all_counts).astype(np.float64) + 2.0
    sorted_values = np.sort(values)
    cumulative = np.cumsum(sorted_values)
    gini = 0.0
    if cumulative[-1] > 0:
        n = len(values)
        gini = float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)
    return {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
        "pilot_only_fraction": float(np.mean(values == 2.0)),
        "gini": gini,
    }


def _simulate_method(
    data: Any,
    pilot_rate: np.ndarray,
    uniforms: list[np.ndarray],
    priority: np.ndarray,
    method: str,
) -> tuple[np.ndarray, float, dict[str, float]]:
    scores = np.empty(len(data.lengths), dtype=np.float64)
    squared_error, token_count = 0.0, 0
    allocation_rows: list[np.ndarray] = []
    for record in range(len(data.lengths)):
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        logq0 = data.logq0[start:end]
        truth = data.logp[start:end] - logq0
        alpha = acceptance_probabilities(data.logp[start:end], logq0)
        accepts, trials = _initial_counts(pilot_rate[start:end])
        allocated = np.zeros(end - start, dtype=np.int64)
        local_priority = priority[start:end]
        if method == "uniform":
            _apply_extras(
                alpha, accepts, trials, uniforms[record], allocated, uniform_counts(end - start)
            )
        elif method == "hybrid":
            _apply_extras(
                alpha, accepts, trials, uniforms[record], allocated, hybrid_counts(local_priority)
            )
        elif method == "full_ai_one_shot":
            _apply_extras(
                alpha,
                accepts,
                trials,
                uniforms[record],
                allocated,
                full_one_shot_counts(local_priority),
            )
        elif method in ("full_ai_sequential", "oracle"):
            oracle_levels = oracle_action_levels(data.logp[start:end], logq0) if method == "oracle" else None
            for _ in range(ACTIVE_BUDGET - 2):
                estimate = estimate_corrected_delta0(
                    logq0, accepts, trials, bisection_steps=24
                )
                if method == "oracle":
                    error = np.square(estimate.delta0 - truth)
                    width = min(8, len(error))
                    current = 0.5 * error + 0.5 * np.convolve(
                        error, np.ones(width), mode="same"
                    ) / np.convolve(np.ones(len(error)), np.ones(width), mode="same")
                else:
                    current = _sequential_priority(
                        local_priority, estimate.delta0, estimate.censoring, allocated
                    )
                new_counts = sequential_round_counts(current)
                _apply_extras(
                    alpha,
                    accepts,
                    trials,
                    uniforms[record],
                    allocated,
                    new_counts,
                    oracle_levels=oracle_levels,
                )
        else:
            raise ValueError(f"unknown allocation method {method!r}")
        estimate = estimate_corrected_delta0(logq0, accepts, trials, bisection_steps=24)
        scores[record] = fragment_score(estimate.delta0, "window_sign_8")
        squared_error += float(np.sum(np.square(estimate.delta0 - truth)))
        token_count += end - start
        allocation_rows.append(allocated.copy())
    return scores, math.sqrt(squared_error / token_count), _allocation_summary(allocation_rows)


def _load_backbone(benchmark: str, epoch: int, seed: int) -> np.ndarray:
    path = (
        ROOT
        / "experiments/results/sft_runs/accept_only_active_v2/neural_adaptive/conditions"
        / f"{benchmark}_epoch{epoch}"
        / f"scores_seed_{seed}.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["lowq_plus_neural_k2"], dtype=np.float64)


def evaluate_condition_seed(
    benchmark: str,
    epoch: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    full, pq = _paths(benchmark, epoch)
    data = load_replay_data(full, pq)
    partitions = split_indices(data.labels, SPLIT_SEED)
    d_nonmember = partitions["D"][data.labels[partitions["D"]] == 0]
    c_nonmember = partitions["C"][data.labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nonmember, N_REF, SPLIT_SEED + N_REF)
    calibration = _deterministic_subset(c_nonmember, N_CAL, SPLIT_SEED + N_CAL + 1000)
    shuffled = np.random.default_rng(SPLIT_SEED + seed).permutation(reference)
    train_records, validation_records = np.sort(shuffled[:320]), np.sort(shuffled[320:])
    static_raw = token_features(data.logq0, data.lengths)
    token_train = _token_mask(train_records, data.offsets)
    mean, scale = np.mean(static_raw[token_train], axis=0), np.std(static_raw[token_train], axis=0)
    static = np.asarray((static_raw - mean) / np.where(scale < 1e-6, 1.0, scale), dtype=np.float32)
    pilot_rate, uniforms = _pilot_observations(data, seed)
    null_model = fit_nonmember_model(
        static_raw,
        pilot_rate,
        token_train,
        trials_per_token=2,
        seed=seed + 2,
    )
    predicted = null_model.predict(static_raw)
    targets = nonmember_probe_value_targets(
        data, reference, pilot_rate, pilot_k=2, replay_seed=seed
    )
    fit = fit_probe_value_model(
        ProbeValueExamples(
            static, pilot_rate, predicted, targets, data.offsets, train_records, k=2
        ),
        ProbeValueExamples(
            static, pilot_rate, predicted, targets, data.offsets, validation_records, k=2
        ),
        input_dim=static.shape[1] + 6,
        seed=seed + 9002,
        device=device,
    )
    _, priority = predict_evidence(
        fit.model, static, pilot_rate, predicted, data.offsets, k=2, device=device
    )
    backbone = _load_backbone(benchmark, epoch, seed)
    scores: dict[str, np.ndarray] = {}
    rmse: dict[str, float] = {}
    allocation: dict[str, dict[str, float]] = {}
    for method in METHODS:
        value, error, summary = _simulate_method(
            data, pilot_rate, uniforms, priority, method
        )
        scores[f"{method}_active"] = value
        scores[f"{method}_fusion"] = _standardize(backbone, reference) + 0.25 * _standardize(
            value, reference
        )
        rmse[method] = error
        allocation[method] = summary
    metrics = {
        name: membership_metrics(value, data.labels, calibration, partitions["T"])
        for name, value in scores.items()
    }
    row = {
        "benchmark": benchmark,
        "epoch": epoch,
        "seed": seed,
        "controller_validation_objective": fit.validation_auc,
        "metrics": metrics,
        "delta_rmse": rmse,
        "allocation": allocation,
    }
    return row, {"labels": data.labels, "record_ids": data.record_ids, **scores}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(rows[0]["metrics"])
    metrics = {}
    for method in methods:
        metrics[method] = {
            "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
            "pauc_0_10": float(np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])),
            "tpr_1": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows])),
            "actual_fpr_1": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["1%"]["actual_fpr"] for row in rows])),
            "tpr_10": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["10%"]["tpr"] for row in rows])),
            "actual_fpr_10": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["10%"]["actual_fpr"] for row in rows])),
        }
    return {
        "experiment": "full post-pilot AI allocation",
        "rows": len(rows),
        "metrics": metrics,
        "delta_rmse": {
            method: float(np.mean([row["delta_rmse"][method] for row in rows]))
            for method in METHODS
        },
        "allocation": {
            method: {
                key: float(np.mean([row["allocation"][method][key] for row in rows]))
                for key in rows[0]["allocation"][method]
            }
            for method in METHODS
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Full Post-Pilot AI Query Allocation",
        "",
        "| Method | Branch | AUC | pAUC | TPR@1% / FPR | TPR@10% / FPR | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for branch in ("active", "fusion"):
            metric = summary["metrics"][f"{method}_{branch}"]
            lines.append(
                f"| `{method}` | {branch} | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
                f"{metric['tpr_1']:.4f} / {metric['actual_fpr_1']:.4f} | "
                f"{metric['tpr_10']:.4f} / {metric['actual_fpr_10']:.4f} | "
                f"{summary['delta_rmse'][method]:.4f} |"
            )
    lines.extend(
        [
            "",
            "| Method | min | p10 | median | p90 | max | pilot-only | Gini |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        value = summary["allocation"][method]
        lines.append(
            f"| `{method}` | {value['minimum']:.1f} | {value['p10']:.1f} | {value['median']:.1f} | "
            f"{value['p90']:.1f} | {value['maximum']:.1f} | {value['pilot_only_fraction']:.2%} | {value['gini']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epoch", choices=EPOCHS, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/full_ai_allocation",
    )
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.aggregate_existing:
        paths = sorted((output / "conditions").glob("*/RAW_RESULTS.json"))
        rows = [row for path in paths for row in json.loads(path.read_text())["rows"]]
        summary = aggregate(rows)
        _write_json(output / "AGGREGATE.json", summary)
        write_markdown(summary, output / "AGGREGATE.md")
        print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))
        return
    if args.benchmark is None or args.epoch is None:
        parser.error("--benchmark and --epoch are required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    condition = output / "conditions" / f"{args.benchmark}_epoch{args.epoch}"
    rows = []
    for seed in args.seeds:
        row, arrays = evaluate_condition_seed(
            args.benchmark, args.epoch, seed, device
        )
        rows.append(row)
        condition.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(condition / f"scores_seed_{seed}.npz", **arrays)
        print(json.dumps({"condition": condition.name, "seed": seed}), flush=True)
    report = {"experiment": "full post-pilot AI allocation", "rows": rows}
    _write_json(condition / "RAW_RESULTS.json", report)
    summary = aggregate(rows)
    _write_json(condition / "AGGREGATE.json", summary)
    write_markdown(summary, condition / "AGGREGATE.md")


if __name__ == "__main__":
    main()
