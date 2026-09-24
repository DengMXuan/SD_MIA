"""Evaluate nonmember-conditioned windows and coverage-constrained active queries.

This is one compact follow-up to the frozen low-q multiscale replay.  It uses
only draft features and accept/reject bits on the detector side.  Exact p is
read solely by the offline verifier simulator and for delta-RMSE diagnostics.

The experiment has two parts:

* N1/N2: fit a low-capacity nonmember acceptance model, turn token surprises
  into multiscale window scores, and test smaller N_ref/N_cal subsets;
* N3/N4: spend a fixed fraction of the post-pilot budget on windows selected
  from draft q plus pilot surprises, while retaining an all-position coverage
  floor and correcting every observation back to delta0=log(p)-log(q0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import genpareto

from experiments.sd_membership_sft.analysis.active_importance_replay import BUDGETS, LAMBDAS, acceptance_probabilities, estimate_corrected_delta0, fragment_score, sample_schedule, uniform_schedule
from experiments.shared.core.replay_cache import ReplayData, load_replay_data
from experiments.shared.core.audit_metrics import conformal_tail_pvalues
from experiments.shared.core.audit_metrics import partial_auc, rank_auc
from experiments.shared.core.audit_runtime import split_indices

from experiments.shared.core.audit_runtime import _write_json, _record_uniforms, _deterministic_subset
from experiments.shared.methods.lowq_baseline import standardized_max
from experiments.shared.core.audit_metrics import membership_metrics


from experiments.paths import ROOT
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
REPLAY_SEEDS = (20260914, 20260915, 20260916)
LOWQ_FRACTIONS = (0.10, 0.20, 0.50)
WINDOWS = (4, 8, 16, 32)
SAMPLE_SETTINGS = ((100, 100), (400, 200), (800, 400))
# The one-seed screen rejected K=4/32 and 50% adaptive allocation.  The
# multi-seed follow-up is deliberately restricted to the two positive point
# estimates instead of spending confirmation compute on rejected settings.
ACTIVE_BUDGETS = (8, 16)
ADAPTIVE_FRACTIONS = (0.25,)
RATES = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class RidgeLogistic:
    """Small conditional acceptance model fitted only on nonmember tokens."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        design = np.c_[np.ones(len(values), dtype=np.float64), values]
        logits = np.clip(design @ self.coefficients, -30.0, 30.0)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-4, 1.0 - 1e-4)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value




def _local_moments(values: np.ndarray, width: int = 8) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    width = min(width, len(values))
    kernel = np.ones(width, dtype=np.float64)
    sums = np.convolve(values, kernel, mode="same")
    squares = np.convolve(values * values, kernel, mode="same")
    counts = np.convolve(np.ones(len(values)), kernel, mode="same")
    mean = sums / counts
    variance = np.maximum(0.0, squares / counts - mean * mean)
    return mean, np.sqrt(variance)


def token_features(logq0: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Build q-only/context features without token identity or target values."""
    logq0 = np.asarray(logq0, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(logq0) != int(np.sum(lengths)) or np.any(lengths <= 0):
        raise ValueError("logq0 and lengths are not aligned")
    result = np.empty((len(logq0), 8), dtype=np.float64)
    offset = 0
    for length in lengths:
        end = offset + int(length)
        surprise = np.clip(-logq0[offset:end], 0.0, 30.0)
        position = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
        local_mean, local_std = _local_moments(surprise)
        order = np.argsort(surprise, kind="stable")
        rank = np.empty(int(length), dtype=np.float64)
        rank[order] = np.linspace(0.0, 1.0, int(length), dtype=np.float64)
        result[offset:end] = np.column_stack(
            (
                surprise,
                np.sqrt(surprise),
                position,
                position * position,
                local_mean,
                local_std,
                rank,
                surprise * position,
            )
        )
        offset = end
    return result


def _token_mask(indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(offsets[-1]), dtype=bool)
    for index in np.asarray(indices, dtype=np.int64):
        mask[int(offsets[index]) : int(offsets[index + 1])] = True
    return mask


def fit_nonmember_model(
    features: np.ndarray,
    accept_rate: np.ndarray,
    token_mask: np.ndarray,
    *,
    trials_per_token: int,
    ridge: float = 1.0,
    max_tokens: int = 200_000,
    seed: int = 0,
    iterations: int = 20,
) -> RidgeLogistic:
    """Fit ridge-logistic IRLS to possibly fractional binomial responses."""
    if trials_per_token <= 0 or iterations <= 0 or ridge < 0.0:
        raise ValueError("invalid fit parameters")
    rows = np.flatnonzero(token_mask)
    if len(rows) == 0:
        raise ValueError("nonmember training mask is empty")
    if len(rows) > max_tokens:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_tokens, replace=False))
    x = np.asarray(features[rows], dtype=np.float64)
    y = np.asarray(accept_rate[rows], dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    design = np.c_[np.ones(len(x)), (x - mean) / scale]
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 1e-6
    for _ in range(iterations):
        logits = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = trials_per_token * (design.T @ (y - probability)) - penalty @ coefficients
        hessian = trials_per_token * (design.T @ (variance[:, None] * design)) + penalty
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return RidgeLogistic(mean=mean, scale=scale, coefficients=coefficients)




def fixed_q_observations(data: ReplayData, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return token acceptance rates and all-accept indicators at normal q0."""
    if k <= 0:
        raise ValueError("k must be positive")
    alpha = np.exp(np.minimum(0.0, data.logp - data.logq0))
    rates = np.empty(len(alpha), dtype=np.float64)
    all_accept = np.empty(len(alpha), dtype=np.float64)
    for record_index in range(len(data.lengths)):
        start, end = int(data.offsets[record_index]), int(data.offsets[record_index + 1])
        bits = _record_uniforms(seed, record_index, end - start, k) < alpha[start:end, None]
        rates[start:end] = np.mean(bits, axis=1)
        all_accept[start:end] = np.all(bits, axis=1)
    return rates, all_accept


def raw_fragment_scores(
    all_accept: np.ndarray,
    predicted_accept: np.ndarray,
    logq0: np.ndarray,
    lengths: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """Compute fixed low-q and learned one-sided window candidates."""
    predicted_all = np.clip(predicted_accept, 1e-4, 1.0 - 1e-4) ** k
    scale = np.sqrt(np.maximum(predicted_all * (1.0 - predicted_all), 0.02))
    residual = (np.asarray(all_accept, dtype=np.float64) - predicted_all) / scale
    result = {
        **{f"lowq_{int(100*fraction)}": np.empty(len(lengths)) for fraction in LOWQ_FRACTIONS},
        "residual_global": np.empty(len(lengths)),
        **{f"window_{width}": np.empty(len(lengths)) for width in WINDOWS},
    }
    offset = 0
    for record_index, length_value in enumerate(lengths):
        length = int(length_value)
        end = offset + length
        q = logq0[offset:end]
        bits = all_accept[offset:end]
        values = residual[offset:end]
        q_order = np.argsort(q, kind="stable")
        for fraction in LOWQ_FRACTIONS:
            count = max(1, int(math.ceil(fraction * length)))
            result[f"lowq_{int(100*fraction)}"][record_index] = np.mean(bits[q_order[:count]])
        result["residual_global"][record_index] = np.mean(np.maximum(values, 0.0))
        for width in WINDOWS:
            actual = min(width, length)
            means = np.convolve(values, np.ones(actual) / actual, mode="valid")
            result[f"window_{width}"][record_index] = np.mean(means > 0.0)
        offset = end
    return result






def evt_membership_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    tail_fraction: float = 0.20,
) -> dict[str, Any]:
    """Fit a generalized-Pareto nonmember tail instead of an empirical cutoff."""
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    member = test[labels[test] == 1]
    nonmember = test[labels[test] == 0]
    calibration = calibration[labels[calibration] == 0]
    values = np.asarray(scores[calibration], dtype=np.float64)
    boundary = float(np.quantile(values, 1.0 - tail_fraction))
    excess = values[values > boundary] - boundary
    if len(excess) < 10 or np.allclose(excess, excess[0]):
        raise ValueError("too few distinct calibration tail values for EVT")
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    scale = max(float(scale), np.finfo(np.float64).eps)
    y = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    result = {
        "auc": rank_auc(scores[member], scores[nonmember]),
        "pauc_0_10": partial_auc(np.r_[scores[member], scores[nonmember]], y),
        "threshold_model": {
            "family": "generalized_pareto",
            "tail_fraction": tail_fraction,
            "tail_count": len(excess),
            "shape": float(shape),
            "scale": scale,
        },
        "tpr_at_fpr": {},
    }
    for rate in RATES:
        conditional_tail = min(1.0, rate / tail_fraction)
        threshold = boundary + float(
            genpareto.ppf(1.0 - conditional_tail, shape, loc=0.0, scale=scale)
        )
        result["tpr_at_fpr"][f"{int(rate*100)}%"] = {
            "tpr": float(np.mean(scores[member] > threshold)),
            "actual_fpr": float(np.mean(scores[nonmember] > threshold)),
            "threshold": threshold,
        }
    return result




def evaluate_learned_windows(
    data: ReplayData,
    features: np.ndarray,
    *,
    k: int,
    replay_seed: int,
    n_ref: int,
    n_cal: int,
    split_seed: int = 20260824,
    observations: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    partitions = split_indices(data.labels, split_seed)
    d_nonmember = partitions["D"][data.labels[partitions["D"]] == 0]
    c_nonmember = partitions["C"][data.labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nonmember, n_ref, split_seed + n_ref)
    calibration = _deterministic_subset(c_nonmember, n_cal, split_seed + n_cal + 1000)
    rates, all_accept = (
        observations
        if observations is not None
        else fixed_q_observations(data, k, replay_seed)
    )
    model = fit_nonmember_model(
        features,
        rates,
        _token_mask(reference, data.offsets),
        trials_per_token=k,
        seed=replay_seed + n_ref,
    )
    predicted = model.predict(features)
    raw = raw_fragment_scores(all_accept, predicted, data.logq0, data.lengths, k)
    lowq_names = tuple(f"lowq_{int(100*fraction)}" for fraction in LOWQ_FRACTIONS)
    window_names = tuple(f"window_{width}" for width in WINDOWS)
    scores = {
        "lowq_multiscale": standardized_max(raw, lowq_names, reference),
        "learned_window_multiscale": standardized_max(raw, window_names, reference),
        "learned_global": standardized_max(raw, ("residual_global",), reference),
        "lowq_plus_window": standardized_max(raw, lowq_names + window_names, reference),
    }
    metrics = {
        name: membership_metrics(values, data.labels, calibration, partitions["T"])
        for name, values in scores.items()
    }
    metrics["lowq_multiscale_evt"] = evt_membership_metrics(
        scores["lowq_multiscale"],
        data.labels,
        calibration,
        partitions["T"],
    )
    return {
        "n_ref": n_ref,
        "n_cal": n_cal,
        "k": k,
        "metrics": metrics,
    }


def window_priority(
    pilot: np.ndarray,
    predicted: np.ndarray,
    logq0: np.ndarray,
    width: int = 8,
) -> np.ndarray:
    """Score positions by low-q difficulty, pilot surprise, and window support."""
    pilot = np.asarray(pilot, dtype=np.float64)
    predicted = np.clip(np.asarray(predicted, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    residual = (pilot - predicted) / np.sqrt(np.maximum(predicted * (1.0 - predicted), 0.02))
    actual = min(width, len(residual))
    window = np.convolve(residual, np.ones(actual) / actual, mode="valid")
    support = np.zeros(len(residual), dtype=np.float64)
    for start, value in enumerate(np.maximum(window, 0.0)):
        support[start : start + actual] += value
    q_order = np.argsort(logq0, kind="stable")
    lowq = np.empty(len(logq0), dtype=np.float64)
    lowq[q_order] = np.linspace(1.0, 0.0, len(logq0), dtype=np.float64)
    uncertainty = predicted * (1.0 - predicted)
    for values in (support, uncertainty):
        span = float(np.max(values) - np.min(values))
        if span > 1e-12:
            values -= np.min(values)
            values /= span
    return support + 0.5 * lowq + 0.25 * uncertainty


def coverage_constrained_schedule(
    length: int,
    budget: int,
    priority: np.ndarray,
    adaptive_fraction: float,
    selected_fraction: float = 0.50,
) -> np.ndarray:
    """Keep a uniform active-q floor and allocate remaining decisions by priority."""
    if length <= 0 or budget <= 1 or not 0.0 < adaptive_fraction <= 1.0:
        raise ValueError("invalid schedule arguments")
    if len(priority) != length or not 0.0 < selected_fraction <= 1.0:
        raise ValueError("priority/selection mismatch")
    extra_total = (budget - 1) * length
    adaptive_total = int(round(adaptive_fraction * extra_total))
    uniform_rounds = (extra_total - adaptive_total) // length
    uniform_total = uniform_rounds * length
    adaptive_total = extra_total - uniform_total
    selected_count = max(1, int(math.ceil(selected_fraction * length)))
    selected = np.argsort(-priority, kind="stable")[:selected_count]
    base, remainder = divmod(adaptive_total, selected_count)
    counts = np.full(selected_count, base, dtype=np.int64)
    counts[:remainder] += 1
    width = 1 + uniform_rounds + int(np.max(counts, initial=0))
    schedule = np.full((length, width), -1, dtype=np.int8)
    schedule[:, 0] = 0
    ladder = np.asarray([1, 2, 3, 4], dtype=np.int8)
    for column in range(uniform_rounds):
        schedule[:, 1 + column] = ladder[column % len(ladder)]
    for rank, position in enumerate(selected):
        start = 1 + uniform_rounds
        schedule[position, start : start + counts[rank]] = np.resize(ladder, counts[rank])
    if int(np.sum(schedule >= 0)) != budget * length:
        raise RuntimeError("coverage schedule violated equal decision budget")
    if not np.all(schedule[:, 0] == 0):
        raise RuntimeError("coverage schedule lost its normal-q pilot")
    return schedule


def active_pilot_context(
    data: ReplayData,
    features: np.ndarray,
    replay_seed: int,
    split_seed: int = 20260824,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    partitions = split_indices(data.labels, split_seed)
    reference = partitions["D"][data.labels[partitions["D"]] == 0]
    pilot, _ = fixed_q_observations(data, 1, replay_seed)
    model = fit_nonmember_model(
        features,
        pilot,
        _token_mask(reference, data.offsets),
        trials_per_token=1,
        seed=replay_seed,
    )
    predicted = model.predict(features)
    pilot_raw_base = raw_fragment_scores(pilot, predicted, data.logq0, data.lengths, 1)
    pilot_raw = {
        **{
            f"pilot_lowq_{int(100*f)}": pilot_raw_base[f"lowq_{int(100*f)}"]
            for f in LOWQ_FRACTIONS
        },
        **{
            f"pilot_window_{width}": pilot_raw_base[f"window_{width}"]
            for width in WINDOWS
        },
    }
    priority = np.empty(len(data.logq0), dtype=np.float64)
    for record_index in range(len(data.lengths)):
        start, end = int(data.offsets[record_index]), int(data.offsets[record_index + 1])
        priority[start:end] = window_priority(
            pilot[start:end], predicted[start:end], data.logq0[start:end]
        )
    return predicted, pilot, priority, pilot_raw


def _active_raw_scores(
    data: ReplayData,
    priority: np.ndarray,
    pilot_raw: dict[str, np.ndarray],
    budget: int,
    replay_seed: int,
    adaptive_fraction: float,
) -> tuple[dict[str, np.ndarray], float]:
    raw = {"delta_window": np.empty(len(data.lengths), dtype=np.float64)}
    raw.update({name: np.asarray(values).copy() for name, values in pilot_raw.items()})
    squared_error, token_count = 0.0, 0
    for record_index in range(len(data.lengths)):
        start, end = int(data.offsets[record_index]), int(data.offsets[record_index + 1])
        logp, logq0 = data.logp[start:end], data.logq0[start:end]
        alpha = acceptance_probabilities(logp, logq0)
        maximum_width = max(2 * budget + 2, budget)
        uniforms = _record_uniforms(replay_seed, record_index, end - start, maximum_width)
        schedule = coverage_constrained_schedule(
            end - start, budget, priority[start:end], adaptive_fraction
        )
        accepts, trials = sample_schedule(alpha, schedule, uniforms)
        estimate = estimate_corrected_delta0(
            logq0, accepts, trials, bisection_steps=24
        ).delta0
        truth = logp - logq0
        squared_error += float(np.sum((estimate - truth) ** 2))
        token_count += len(truth)
        raw["delta_window"][record_index] = fragment_score(estimate, "window_sign_8")
    return raw, math.sqrt(squared_error / token_count)


def evaluate_active_windows(
    data: ReplayData,
    features: np.ndarray,
    *,
    budget: int,
    replay_seed: int,
    adaptive_fraction: float,
    split_seed: int = 20260824,
    pilot_context: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    partitions = split_indices(data.labels, split_seed)
    reference = partitions["D"][data.labels[partitions["D"]] == 0]
    calibration = partitions["C"][data.labels[partitions["C"]] == 0]
    if pilot_context is None:
        pilot_context = active_pilot_context(data, features, replay_seed, split_seed)
    _, _, priority, pilot_raw = pilot_context
    raw, rmse = _active_raw_scores(
        data, priority, pilot_raw, budget, replay_seed, adaptive_fraction
    )
    lowq_names = tuple(f"pilot_lowq_{int(100*f)}" for f in LOWQ_FRACTIONS)
    window_names = tuple(f"pilot_window_{width}" for width in WINDOWS)
    scores = {
        "active_delta_window": standardized_max(raw, ("delta_window",), reference),
        "active_fusion": standardized_max(
            raw, ("delta_window",) + lowq_names + window_names, reference
        ),
    }
    return {
        "budget": budget,
        "adaptive_fraction": adaptive_fraction,
        "delta_rmse": rmse,
        "metrics": {
            name: membership_metrics(values, data.labels, calibration, partitions["T"])
            for name, values in scores.items()
        },
    }


def _default_full_delta(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled/full_delta.npz"


def _default_pq(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/pq_directional" / f"{benchmark}_epoch{epoch}" / "pq_gap_token_logps.npz"


def run_condition_rows(
    benchmark: str,
    epoch: int,
    seeds: tuple[int, ...] = REPLAY_SEEDS,
    include_active: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = load_replay_data(_default_full_delta(benchmark, epoch), _default_pq(benchmark, epoch))
    features = token_features(data.logq0, data.lengths)
    for seed in seeds:
        observation_cache = {
            k: fixed_q_observations(data, k, seed) for k in (1, 2)
        }
        for k in (1, 2):
            for n_ref, n_cal in SAMPLE_SETTINGS:
                result = evaluate_learned_windows(
                    data,
                    features,
                    k=k,
                    replay_seed=seed,
                    n_ref=n_ref,
                    n_cal=n_cal,
                    observations=observation_cache[k],
                )
                rows.append({"experiment": "learned_windows", "benchmark": benchmark, "epoch": epoch, "seed": seed, **result})
        if include_active:
            pilot_context = active_pilot_context(data, features, seed)
            for budget in ACTIVE_BUDGETS:
                for adaptive_fraction in ADAPTIVE_FRACTIONS:
                    result = evaluate_active_windows(
                        data,
                        features,
                        budget=budget,
                        replay_seed=seed,
                        adaptive_fraction=adaptive_fraction,
                        pilot_context=pilot_context,
                    )
                    rows.append({"experiment": "active_windows", "benchmark": benchmark, "epoch": epoch, "seed": seed, **result})
        print(json.dumps({"condition": f"{benchmark}_epoch{epoch}", "seed": seed}), flush=True)
    return rows


def run_all(output_dir: Path, seeds: tuple[int, ...] = REPLAY_SEEDS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        for epoch in EPOCHS:
            rows.extend(run_condition_rows(benchmark, epoch, seeds))
    report = {
        "experiment": "nonmember-conditioned windowed active querying",
        "status": "exploratory offline position-locked replay",
        "seeds": list(seeds),
        "sample_settings": [list(value) for value in SAMPLE_SETTINGS],
        "active_budgets": list(ACTIVE_BUDGETS),
        "adaptive_fractions": list(ADAPTIVE_FRACTIONS),
        "rows": rows,
    }
    _write_json(output_dir / "RAW_RESULTS.json", report)
    return report


def aggregate(report: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in report["rows"]:
        if row["experiment"] == "learned_windows":
            key = ("learned_windows", row["k"], row["n_ref"], row["n_cal"])
        else:
            key = ("active_windows", row["budget"], row["adaptive_fraction"])
        groups.setdefault(key, []).append(row)
    summary: dict[str, Any] = {"learned_windows": {}, "active_windows": {}}
    for key, rows in groups.items():
        experiment = key[0]
        method_names = rows[0]["metrics"]
        metrics: dict[str, Any] = {}
        for method in method_names:
            metrics[method] = {
                "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
                "pauc_0_10": float(np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])),
                "tpr_1": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows])),
                "actual_fpr_1": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["1%"]["actual_fpr"] for row in rows])),
                "tpr_10": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["10%"]["tpr"] for row in rows])),
                "actual_fpr_10": float(np.mean([row["metrics"][method]["tpr_at_fpr"]["10%"]["actual_fpr"] for row in rows])),
            }
        if experiment == "learned_windows":
            summary[experiment][f"k{key[1]}_nref{key[2]}_ncal{key[3]}"] = {"metrics": metrics}
        else:
            summary[experiment][f"k{key[1]}_adaptive{key[2]:.2f}"] = {
                "metrics": metrics,
                "delta_rmse": float(np.mean([row["delta_rmse"] for row in rows])),
            }
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Nonmember-Conditioned Windowed Active Querying",
        "",
        "> Exploratory offline position-locked replay; not a real remote verifier run.",
        "",
        "## Learned nonmember windows and sample efficiency",
        "",
        "| Setting | Method | AUC | pAUC | TPR@1% | actual FPR | TPR@10% | actual FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for setting, value in summary["learned_windows"].items():
        for method, metric in value["metrics"].items():
            lines.append(
                f"| `{setting}` | `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
                f"{metric['tpr_1']:.4f} | {metric['actual_fpr_1']:.4f} | "
                f"{metric['tpr_10']:.4f} | {metric['actual_fpr_10']:.4f} |"
            )
    lines.extend([
        "",
        "## Coverage-constrained active querying",
        "",
        "| Setting | Method | AUC | pAUC | TPR@1% | actual FPR | TPR@10% | actual FPR | Delta RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for setting, value in summary["active_windows"].items():
        for method, metric in value["metrics"].items():
            lines.append(
                f"| `{setting}` | `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
                f"{metric['tpr_1']:.4f} | {metric['actual_fpr_1']:.4f} | "
                f"{metric['tpr_10']:.4f} | {metric['actual_fpr_10']:.4f} | {value['delta_rmse']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/adaptive_window",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epoch", choices=EPOCHS, type=int)
    parser.add_argument("--skip-active", action="store_true")
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Aggregate condition RAW_RESULTS.json files already under output-dir.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.aggregate_existing:
        paths = sorted((output_dir / "conditions").glob("*/RAW_RESULTS.json"))
        if not paths:
            raise RuntimeError("no condition results to aggregate")
        rows = []
        for path in paths:
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        report = {
            "experiment": "nonmember-conditioned windowed active querying",
            "status": "exploratory offline position-locked replay",
            "seeds": list(args.seeds),
            "rows": rows,
        }
    elif args.benchmark is not None or args.epoch is not None:
        if args.benchmark is None or args.epoch is None:
            parser.error("--benchmark and --epoch must be supplied together")
        rows = run_condition_rows(
            args.benchmark,
            args.epoch,
            tuple(args.seeds),
            include_active=not args.skip_active,
        )
        report = {
            "experiment": "nonmember-conditioned windowed active querying",
            "status": "exploratory offline position-locked replay",
            "seeds": list(args.seeds),
            "rows": rows,
        }
        condition_dir = output_dir / "conditions" / f"{args.benchmark}_epoch{args.epoch}"
        _write_json(condition_dir / "RAW_RESULTS.json", report)
        condition_summary = aggregate(report)
        _write_json(condition_dir / "AGGREGATE.json", condition_summary)
        write_markdown(condition_summary, condition_dir / "AGGREGATE.md")
        print(json.dumps({"output": str(condition_dir), "rows": len(rows)}, indent=2))
        return
    else:
        report = run_all(output_dir, tuple(args.seeds))
    summary = aggregate(report)
    _write_json(output_dir / "AGGREGATE.json", summary)
    write_markdown(summary, output_dir / "AGGREGATE.md")
    print(json.dumps({"output": str(output_dir), "rows": len(report["rows"])}, indent=2))


if __name__ == "__main__":
    main()
