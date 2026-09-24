"""Shadow-trained marginal-value allocation with closed-loop query updates.

Every token first receives ``pilot_k`` normal-q verifier decisions.  A small
MLP trained only on a legitimate local shadow model predicts token-level
positive-delta anomaly value from the currently observable state.  Remaining
decisions are allocated by diminishing-return water filling, observations are
incorporated, and values are recomputed each round.  No fixed selected-token
fraction is used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.sd_membership_sft.analysis.active_importance_replay import acceptance_probabilities, estimate_corrected_delta0, fragment_score
from experiments.shared.core.replay_cache import load_replay_data
from experiments.shared.core.audit_runtime import _deterministic_subset
from experiments.shared.core.audit_metrics import membership_metrics
from experiments.sd_membership_sft.archive.adaptive_window_accept_only import token_features
from experiments.shared.core.audit_runtime import split_indices
from experiments.sd_membership_sft.archive.interpretable_scale_gate import _fit_gate, _predict_gate, fragment_q_summaries, scale_accept_scores
from experiments.shared.core.audit_runtime import BENCHMARKS, EPOCHS, N_CAL, N_REF, REPLAY_SEEDS, ROOT, SPLIT_SEED, _paths, _write_json
from experiments.sd_membership_sft.archive.neural_adaptive_accept_only import _standardize


BUDGETS = ((1, 2), (2, 8))
METHODS = ("uniform", "fixed50", "dynamic", "dynamic_capped")
STATE_DIM = 24


@dataclass(frozen=True)
class ShadowCache:
    labels: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    logp: np.ndarray
    logq0: np.ndarray


class MarginalValueMLP(nn.Module):
    def __init__(self, input_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(values).squeeze(-1))


@dataclass(frozen=True)
class PolicyFit:
    model: MarginalValueMLP
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_scale: float
    validation_loss: float


def _load_shadow(path: Path) -> ShadowCache:
    with np.load(path, allow_pickle=False) as archive:
        lengths = np.asarray(archive["lengths"], dtype=np.int64)
        return ShadowCache(
            labels=np.asarray(archive["labels"], dtype=np.int64),
            lengths=lengths,
            offsets=np.r_[0, np.cumsum(lengths, dtype=np.int64)],
            logp=np.asarray(archive["logp"], dtype=np.float64),
            logq0=np.asarray(archive["logq0"], dtype=np.float64),
        )


def _record_token_mask(records: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    mask = np.zeros(int(offsets[-1]), dtype=bool)
    for record in np.asarray(records, dtype=np.int64):
        mask[int(offsets[record]) : int(offsets[record + 1])] = True
    return mask


def _local_mean(values: np.ndarray, lengths: np.ndarray, width: int = 8) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    offset = 0
    for length_value in lengths:
        length = int(length_value)
        end = offset + length
        actual = min(width, length)
        kernel = np.ones(actual, dtype=np.float64)
        output[offset:end] = np.convolve(values[offset:end], kernel, mode="same") / np.convolve(
            np.ones(length, dtype=np.float64), kernel, mode="same"
        )
        offset = end
    return output


def _shadow_teacher(
    shadow: ShadowCache,
    static: np.ndarray,
    train_records: np.ndarray,
) -> np.ndarray:
    train_nonmember = train_records[shadow.labels[train_records] == 0]
    mask = _record_token_mask(train_nonmember, shadow.offsets)
    x = np.asarray(static, dtype=np.float64)
    mean, scale = np.mean(x[mask], axis=0), np.std(x[mask], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    design = np.c_[np.ones(np.sum(mask)), (x[mask] - mean) / scale]
    target = (shadow.logp - shadow.logq0)[mask]
    ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-2
    ridge[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    predicted = np.c_[np.ones(len(x)), (x - mean) / scale] @ coefficients
    residual_scale = max(1e-4, float(np.std(target - predicted[mask])))
    positive = np.maximum((shadow.logp - shadow.logq0 - predicted) / residual_scale, 0.0)
    squared = np.square(np.clip(positive, 0.0, 20.0))
    return 0.5 * squared + 0.5 * _local_mean(squared, shadow.lengths)


def state_features(
    static: np.ndarray,
    logq0: np.ndarray,
    accepts: np.ndarray,
    trials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deployable state features and expose estimate/censoring."""
    estimate = estimate_corrected_delta0(logq0, accepts, trials, bisection_steps=20)
    total = np.sum(trials, axis=1).astype(np.float64)
    rates = np.divide(
        accepts,
        trials,
        out=np.full(accepts.shape, 0.5, dtype=np.float64),
        where=trials > 0,
    )
    fractions = trials / total[:, None]
    censor = np.column_stack(
        (estimate.censoring == -1, estimate.censoring == 0, estimate.censoring == 1)
    ).astype(np.float64)
    features = np.column_stack(
        (
            np.asarray(static, dtype=np.float64),
            np.clip(estimate.delta0, -8.0, 8.0),
            censor,
            np.log1p(total),
            1.0 / np.sqrt(total),
            rates,
            fractions,
        )
    )
    if features.shape[1] != STATE_DIM:
        raise RuntimeError(f"unexpected dynamic state width {features.shape[1]}")
    return np.asarray(features, dtype=np.float32), estimate.delta0, estimate.censoring


def _simulate_counts(
    alpha: np.ndarray,
    pilot_k: int,
    extra_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    extra_counts = np.asarray(extra_counts, dtype=np.int64)
    accepts = np.zeros((len(alpha), 5), dtype=np.int64)
    trials = np.zeros((len(alpha), 5), dtype=np.int64)
    trials[:, 0] = pilot_k
    accepts[:, 0] = np.sum(
        rng.random((len(alpha), pilot_k)) < alpha[:, 0, None], axis=1
    )
    for level in range(1, 5):
        count = extra_counts // 4 + (extra_counts % 4 >= level)
        trials[:, level] = count
        if np.any(count):
            maximum = int(np.max(count))
            draws = rng.random((len(alpha), maximum))
            accepts[:, level] = np.sum(
                (np.arange(maximum)[None, :] < count[:, None])
                & (draws < alpha[:, level, None]),
                axis=1,
            )
    return accepts, trials


def _sample_policy_examples(
    shadow: ShadowCache,
    static: np.ndarray,
    teacher: np.ndarray,
    records: np.ndarray,
    *,
    pilot_k: int,
    total_budget: int,
    seed: int,
    tokens_per_record: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    max_extra = max(1, min(16, 2 * (total_budget - pilot_k) + 2))
    for record_value in np.asarray(records, dtype=np.int64):
        record = int(record_value)
        start, end = int(shadow.offsets[record]), int(shadow.offsets[record + 1])
        length = end - start
        count = min(length, tokens_per_record)
        top_count = count // 2
        local_teacher = teacher[start:end]
        top = np.argsort(-local_teacher, kind="stable")[:top_count]
        remaining = np.setdiff1d(np.arange(length), top, assume_unique=False)
        rng = np.random.default_rng(np.random.SeedSequence([seed, record, pilot_k]))
        random_count = count - len(top)
        random = rng.choice(remaining, random_count, replace=False) if random_count else np.empty(0, int)
        chosen = np.sort(np.r_[top, random])
        indices = start + chosen
        alpha = acceptance_probabilities(shadow.logp[indices], shadow.logq0[indices])
        for state in range(3):
            if state == 0:
                extras = np.zeros(count, dtype=np.int64)
            elif state == 1:
                extras = rng.integers(0, max_extra + 1, size=count, dtype=np.int64)
            else:
                probability = local_teacher[chosen] + 0.05
                probability = probability / np.max(probability)
                extras = rng.binomial(max_extra, np.clip(probability, 0.0, 1.0)).astype(np.int64)
            accepts, trials = _simulate_counts(alpha, pilot_k, extras, rng)
            features, _, _ = state_features(
                static[indices], shadow.logq0[indices], accepts, trials
            )
            feature_rows.append(features)
            target_rows.append(np.log1p(local_teacher[chosen]).astype(np.float32))
    return np.concatenate(feature_rows), np.concatenate(target_rows)


def fit_shadow_policy(
    shadow: ShadowCache,
    *,
    pilot_k: int,
    total_budget: int,
    seed: int,
    device: torch.device,
) -> PolicyFit:
    rng = np.random.default_rng(seed + 991)
    by_label = [rng.permutation(np.flatnonzero(shadow.labels == label)) for label in (0, 1)]
    train_records = np.sort(np.r_[by_label[0][:160], by_label[1][:160]])
    validation_records = np.sort(np.r_[by_label[0][160:], by_label[1][160:]])
    static = token_features(shadow.logq0, shadow.lengths)
    teacher = _shadow_teacher(shadow, static, train_records)
    train_x, train_y = _sample_policy_examples(
        shadow,
        static,
        teacher,
        train_records,
        pilot_k=pilot_k,
        total_budget=total_budget,
        seed=seed + 100,
    )
    val_x, val_y = _sample_policy_examples(
        shadow,
        static,
        teacher,
        validation_records,
        pilot_k=pilot_k,
        total_budget=total_budget,
        seed=seed + 200,
    )
    mean, scale = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    scale = np.where(scale < 1e-5, 1.0, scale)
    train_x = np.asarray((train_x - mean) / scale, dtype=np.float32)
    val_x = np.asarray((val_x - mean) / scale, dtype=np.float32)
    target_scale = max(1e-4, float(np.quantile(train_y, 0.95)))
    train_y = np.asarray(train_y / target_scale, dtype=np.float32)
    val_y = np.asarray(val_y / target_scale, dtype=np.float32)

    torch.manual_seed(seed)
    model = MarginalValueMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x_t = torch.from_numpy(train_x).to(device)
    y_t = torch.from_numpy(train_y).to(device)
    xv_t = torch.from_numpy(val_x).to(device)
    yv_t = torch.from_numpy(val_y).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, stale = np.inf, None, 0
    batch_size = 4096
    for _ in range(30):
        model.train()
        permutation = torch.randperm(len(x_t), generator=generator)
        for start in range(0, len(permutation), batch_size):
            rows = permutation[start : start + batch_size].to(device)
            prediction = model(x_t[rows])
            weights = 1.0 + 2.0 * torch.clamp(y_t[rows], 0.0, 2.0)
            loss = torch.mean(weights * torch.square(prediction - y_t[rows]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            losses = []
            for start in range(0, len(xv_t), 32768):
                prediction = model(xv_t[start : start + 32768])
                losses.append(
                    torch.mean(torch.square(prediction - yv_t[start : start + 32768])).cpu()
                )
            validation_loss = float(torch.stack(losses).mean())
        if validation_loss < best_loss - 1e-5:
            best_loss, stale = validation_loss, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("marginal-value policy failed to train")
    model.load_state_dict(best_state)
    model.eval()
    return PolicyFit(
        model=model,
        feature_mean=np.asarray(mean, dtype=np.float32),
        feature_scale=np.asarray(scale, dtype=np.float32),
        target_scale=target_scale,
        validation_loss=best_loss,
    )


@torch.inference_mode()
def predict_utility(
    fit: PolicyFit,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    normalized = np.asarray(
        (features - fit.feature_mean) / fit.feature_scale, dtype=np.float32
    )
    output = np.empty(len(normalized), dtype=np.float64)
    for start in range(0, len(normalized), batch_size):
        values = torch.from_numpy(normalized[start : start + batch_size]).to(device)
        prediction = fit.model(values).cpu().numpy()
        output[start : start + len(prediction)] = np.expm1(
            np.clip(prediction * fit.target_scale, 0.0, 8.0)
        )
    return output


def waterfill_counts(
    utility: np.ndarray,
    current_queries: np.ndarray,
    budget: int,
    *,
    max_add: int,
    remaining_cap: np.ndarray | None = None,
    decay_power: float = 0.5,
) -> np.ndarray:
    """Select the globally best diminishing marginal query units exactly."""
    utility = np.asarray(utility, dtype=np.float64)
    current = np.asarray(current_queries, dtype=np.float64)
    if (
        len(utility) == 0
        or current.shape != utility.shape
        or budget < 0
        or max_add <= 0
        or decay_power <= 0.0
    ):
        raise ValueError("invalid water-filling inputs")
    cap = (
        np.full(len(utility), max_add, dtype=np.int64)
        if remaining_cap is None
        else np.minimum(np.asarray(remaining_cap, dtype=np.int64), max_add)
    )
    if cap.shape != utility.shape or np.any(cap < 0):
        raise ValueError("remaining cap must align with utility")
    if budget > int(np.sum(cap)):
        raise ValueError("round budget exceeds the per-token allocation cap")
    if budget == 0:
        return np.zeros(len(utility), dtype=np.int64)
    steps = np.arange(1, max_add + 1, dtype=np.float64)
    marginal = (np.maximum(utility, 0.0)[:, None] + 1e-12) / np.power(
        current[:, None] + steps[None, :], decay_power
    )
    marginal[steps[None, :] > cap[:, None]] = -np.inf
    flat = marginal.ravel()
    chosen = np.argpartition(flat, len(flat) - budget)[-budget:]
    return np.bincount(chosen // max_add, minlength=len(utility)).astype(np.int64)


def _record_uniforms(seed: int, record: int, length: int, width: int) -> np.ndarray:
    return np.random.default_rng(
        np.random.SeedSequence([seed, record, 99173])
    ).random((length, width), dtype=np.float64)


def _pilot_state(
    alpha: np.ndarray, draws: np.ndarray, pilot_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepts = np.zeros((len(alpha), 5), dtype=np.int64)
    trials = np.zeros((len(alpha), 5), dtype=np.int64)
    accepts[:, 0] = np.sum(draws[:, :pilot_k] < alpha[:, 0, None], axis=1)
    trials[:, 0] = pilot_k
    return accepts, trials, np.zeros(len(alpha), dtype=np.int64)


def _apply_counts(
    alpha: np.ndarray,
    accepts: np.ndarray,
    trials: np.ndarray,
    draws: np.ndarray,
    allocated: np.ndarray,
    counts: np.ndarray,
    pilot_k: int,
) -> None:
    for step in range(int(np.max(counts, initial=0))):
        rows = np.flatnonzero(counts > step)
        if len(rows) == 0:
            break
        previous = allocated[rows] + step
        levels = 1 + previous % 4
        columns = pilot_k + previous
        if int(np.max(columns)) >= draws.shape[1]:
            raise RuntimeError("query draw width is too small")
        trials[rows, levels] += 1
        accepts[rows, levels] += (
            draws[rows, columns] < alpha[rows, levels]
        ).astype(np.int64)
    allocated += counts


def _round_budgets(length: int, extra_per_token: int, rounds: int) -> list[int]:
    total = length * extra_per_token
    base, remainder = divmod(total, rounds)
    return [base + int(index < remainder) for index in range(rounds)]


def _priority_with_context(
    predicted: np.ndarray,
    censoring: np.ndarray,
    length: int,
) -> np.ndarray:
    base = (0.05 + np.maximum(predicted, 0.0)) * (1.0 + 0.5 * (censoring != 0))
    width = min(8, length)
    local = np.convolve(base, np.ones(width), mode="same") / np.convolve(
        np.ones(length), np.ones(width), mode="same"
    )
    return 0.65 * base + 0.35 * local


def _simulate_method(
    data: Any,
    static: np.ndarray,
    fit: PolicyFit,
    *,
    pilot_k: int,
    total_budget: int,
    seed: int,
    method: str,
    device: torch.device,
    record_chunk: int = 64,
) -> tuple[np.ndarray, float, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(method)
    extra_per_token = total_budget - pilot_k
    rounds = 4 if (pilot_k, total_budget) == (1, 2) else extra_per_token
    max_add = 4 if (pilot_k, total_budget) == (1, 2) else 8
    draw_width = pilot_k + rounds * max_add
    scores = np.empty(len(data.lengths), dtype=np.float64)
    all_queries = np.empty(len(data.logq0), dtype=np.int16)
    squared_error = 0.0
    oracle_sum = oracle_count = lowq_sum = lowq_count = 0.0

    for first_record in range(0, len(data.lengths), record_chunk):
        last_record = min(len(data.lengths), first_record + record_chunk)
        chunk_start = int(data.offsets[first_record])
        chunk_end = int(data.offsets[last_record])
        local_offsets = data.offsets[first_record : last_record + 1] - chunk_start
        logp = data.logp[chunk_start:chunk_end]
        logq0 = data.logq0[chunk_start:chunk_end]
        alpha = acceptance_probabilities(logp, logq0)
        draws = np.concatenate(
            [
                _record_uniforms(
                    seed,
                    record,
                    int(data.lengths[record]),
                    draw_width,
                )
                for record in range(first_record, last_record)
            ],
            axis=0,
        )
        accepts, trials, allocated = _pilot_state(alpha, draws, pilot_k)

        if method == "uniform":
            counts = np.full(len(logq0), extra_per_token, dtype=np.int64)
            _apply_counts(alpha, accepts, trials, draws, allocated, counts, pilot_k)
        elif method == "fixed50":
            features, _, censoring = state_features(
                static[chunk_start:chunk_end], logq0, accepts, trials
            )
            predicted = predict_utility(fit, features, device)
            counts = np.zeros(len(logq0), dtype=np.int64)
            for row in range(last_record - first_record):
                start, end = int(local_offsets[row]), int(local_offsets[row + 1])
                priority = _priority_with_context(predicted[start:end], censoring[start:end], end - start)
                selected_count = max(1, int(math.ceil(0.50 * (end - start))))
                selected = np.argsort(-priority, kind="stable")[:selected_count]
                budget = extra_per_token * (end - start)
                base, remainder = divmod(budget, selected_count)
                counts[start + selected] = base
                counts[start + selected[:remainder]] += 1
            _apply_counts(alpha, accepts, trials, draws, allocated, counts, pilot_k)
        else:
            budgets = [
                _round_budgets(
                    int(data.lengths[record]), extra_per_token, rounds
                )
                for record in range(first_record, last_record)
            ]
            for round_index in range(rounds):
                features, _, censoring = state_features(
                    static[chunk_start:chunk_end], logq0, accepts, trials
                )
                predicted = predict_utility(fit, features, device)
                counts = np.zeros(len(logq0), dtype=np.int64)
                current = np.sum(trials, axis=1)
                for row in range(last_record - first_record):
                    start, end = int(local_offsets[row]), int(local_offsets[row + 1])
                    priority = _priority_with_context(
                        predicted[start:end], censoring[start:end], end - start
                    )
                    capped = method == "dynamic_capped"
                    maximum_total = 5 if (pilot_k, total_budget) == (1, 2) else 20
                    remaining_cap = (
                        np.maximum(0, maximum_total - current[start:end])
                        if capped
                        else None
                    )
                    counts[start:end] = waterfill_counts(
                        priority,
                        current[start:end],
                        budgets[row][round_index],
                        max_add=max_add,
                        remaining_cap=remaining_cap,
                        decay_power=1.0 if capped else 0.5,
                    )
                _apply_counts(alpha, accepts, trials, draws, allocated, counts, pilot_k)

        estimate = estimate_corrected_delta0(logq0, accepts, trials, bisection_steps=24)
        truth = logp - logq0
        squared_error += float(np.sum(np.square(estimate.delta0 - truth)))
        query_counts = np.sum(trials, axis=1)
        all_queries[chunk_start:chunk_end] = query_counts
        for row, record in enumerate(range(first_record, last_record)):
            start, end = int(local_offsets[row]), int(local_offsets[row + 1])
            scores[record] = fragment_score(estimate.delta0[start:end], "window_sign_8")
            length = end - start
            count = max(1, int(math.ceil(0.20 * length)))
            oracle = np.argsort(-np.maximum(truth[start:end], 0.0), kind="stable")[:count]
            lowq = np.argsort(logq0[start:end], kind="stable")[:count]
            oracle_sum += float(np.sum(query_counts[start:end][oracle]))
            oracle_count += len(oracle)
            lowq_sum += float(np.sum(query_counts[start:end][lowq]))
            lowq_count += len(lowq)

    values = all_queries.astype(np.float64)
    ordered = np.sort(values)
    cumulative = np.cumsum(ordered)
    n = len(values)
    gini = float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)
    allocation = {
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
        "pilot_only_fraction": float(np.mean(values == pilot_k)),
        "gini": gini,
        "positive_delta_top20_mean_queries": oracle_sum / oracle_count,
        "lowq20_mean_queries": lowq_sum / lowq_count,
    }
    return scores, math.sqrt(squared_error / len(data.logq0)), allocation


def _pilot_all_accept(data: Any, pilot_k: int, seed: int) -> np.ndarray:
    output = np.empty(len(data.logq0), dtype=np.float64)
    for record in range(len(data.lengths)):
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        alpha = acceptance_probabilities(data.logp[start:end], data.logq0[start:end])[:, 0]
        draws = _record_uniforms(seed, record, end - start, pilot_k)
        output[start:end] = np.all(draws < alpha[:, None], axis=1)
    return output


def _shadow_gate_examples_k(
    shadow: ShadowCache,
    *,
    pilot_k: int,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    alpha = acceptance_probabilities(shadow.logp, shadow.logq0)[:, 0]
    rng = np.random.default_rng(seed)
    all_accept = np.all(
        rng.random((len(alpha), pilot_k)) < alpha[:, None], axis=1
    ).astype(np.float64)
    raw = scale_accept_scores(all_accept, shadow.logq0, shadow.lengths)
    groups = [rng.permutation(np.flatnonzero(shadow.labels == label)) for label in (0, 1)]
    train = np.sort(np.r_[groups[0][:160], groups[1][:160]])
    validation = np.sort(np.r_[groups[0][160:], groups[1][160:]])
    nm = train[shadow.labels[train] == 0]
    center, scale = np.mean(raw[nm], axis=0), np.std(raw[nm], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    evidence = np.asarray((raw - center) / scale, dtype=np.float32)
    summary = fragment_q_summaries(shadow.logq0, shadow.lengths)
    summary_mean, summary_scale = np.mean(summary[train], axis=0), np.std(summary[train], axis=0)
    summary_scale = np.where(summary_scale < 1e-6, 1.0, summary_scale)
    summary = np.asarray((summary - summary_mean) / summary_scale, dtype=np.float32)

    def take(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return summary[rows], evidence[rows], shadow.labels[rows].astype(np.float32)

    return take(train), take(validation)


def _target_shadow_gate_score(
    data: Any,
    shadow: ShadowCache,
    reference: np.ndarray,
    train_records: np.ndarray,
    *,
    pilot_k: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    all_accept = _pilot_all_accept(data, pilot_k, seed)
    raw = scale_accept_scores(all_accept, data.logq0, data.lengths)
    center, scale = np.mean(raw[train_records], axis=0), np.std(raw[train_records], axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    evidence = np.asarray((raw - center) / scale, dtype=np.float32)
    summary = fragment_q_summaries(data.logq0, data.lengths)
    mean, summary_scale = np.mean(summary[train_records], axis=0), np.std(
        summary[train_records], axis=0
    )
    summary_scale = np.where(summary_scale < 1e-6, 1.0, summary_scale)
    summary = np.asarray((summary - mean) / summary_scale, dtype=np.float32)
    train, validation = _shadow_gate_examples_k(
        shadow, pilot_k=pilot_k, seed=seed + 2000
    )
    model, _ = _fit_gate(
        train,
        validation,
        accept_aware=False,
        seed=seed + 20,
        device=device,
    )
    score, _ = _predict_gate(model, summary, evidence, device)
    return np.asarray(score, dtype=np.float64)


def evaluate_condition_seed(
    benchmark: str,
    epoch: int,
    seed: int,
    pilot_k: int,
    total_budget: int,
    policy: PolicyFit,
    shadow: ShadowCache,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    full, pq = _paths(benchmark, epoch)
    data = load_replay_data(full, pq)
    partitions = split_indices(data.labels, SPLIT_SEED)
    d_nm = partitions["D"][data.labels[partitions["D"]] == 0]
    c_nm = partitions["C"][data.labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nm, N_REF, SPLIT_SEED + N_REF)
    calibration = _deterministic_subset(c_nm, N_CAL, SPLIT_SEED + N_CAL + 1000)
    shuffled = np.random.default_rng(SPLIT_SEED + seed).permutation(reference)
    train_records = np.sort(shuffled[:320])
    static = token_features(data.logq0, data.lengths)
    backbone = _target_shadow_gate_score(
        data,
        shadow,
        reference,
        train_records,
        pilot_k=pilot_k,
        seed=seed,
        device=device,
    )
    scores: dict[str, np.ndarray] = {"backbone": backbone}
    rmse: dict[str, float] = {}
    allocation: dict[str, dict[str, float]] = {}
    for method in METHODS:
        active, error, summary = _simulate_method(
            data,
            static,
            policy,
            pilot_k=pilot_k,
            total_budget=total_budget,
            seed=seed,
            method=method,
            device=device,
        )
        scores[f"{method}_active"] = active
        scores[f"{method}_fusion"] = _standardize(
            backbone, reference
        ) + 0.25 * _standardize(active, reference)
        rmse[method] = error
        allocation[method] = summary
    if (pilot_k, total_budget) == (1, 2):
        existing = (
            ROOT
            / "experiments/results/sft_runs/accept_only_active_v2/shadow_scale_gate/conditions"
            / f"{benchmark}_epoch{epoch}"
            / f"scores_seed_{seed}.npz"
        )
        with np.load(existing, allow_pickle=False) as archive:
            scores["uniform_normal_k2_shadow"] = np.asarray(
                archive["shadow_gate"], dtype=np.float64
            )
    metrics = {
        name: membership_metrics(value, data.labels, calibration, partitions["T"])
        for name, value in scores.items()
    }
    row = {
        "benchmark": benchmark,
        "epoch": epoch,
        "seed": seed,
        "pilot_k": pilot_k,
        "total_budget": total_budget,
        "policy_validation_loss": policy.validation_loss,
        "metrics": metrics,
        "delta_rmse": rmse,
        "allocation": allocation,
    }
    return row, {"labels": data.labels, "record_ids": data.record_ids, **scores}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = tuple(rows[0]["metrics"])
    metrics = {
        method: {
            "auc": float(np.mean([row["metrics"][method]["auc"] for row in rows])),
            "pauc_0_10": float(
                np.mean([row["metrics"][method]["pauc_0_10"] for row in rows])
            ),
            "tpr_1": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["1%"]["tpr"] for row in rows]
                )
            ),
            "actual_fpr_1": float(
                np.mean(
                    [
                        row["metrics"][method]["tpr_at_fpr"]["1%"]["actual_fpr"]
                        for row in rows
                    ]
                )
            ),
            "tpr_10": float(
                np.mean(
                    [row["metrics"][method]["tpr_at_fpr"]["10%"]["tpr"] for row in rows]
                )
            ),
        }
        for method in methods
    }
    comparisons = {}
    for branch in ("active", "fusion"):
        baseline = f"uniform_{branch}"
        for method in ("fixed50", "dynamic", "dynamic_capped"):
            name = f"{method}_{branch}"
            differences = np.asarray(
                [
                    row["metrics"][name]["pauc_0_10"]
                    - row["metrics"][baseline]["pauc_0_10"]
                    for row in rows
                ]
            )
            comparisons[f"{name}_minus_{baseline}"] = {
                "pauc_0_10": float(np.mean(differences)),
                "wins": int(np.sum(differences > 0.0)),
                "total": len(rows),
            }
    return {
        "rows": len(rows),
        "pilot_k": rows[0]["pilot_k"],
        "total_budget": rows[0]["total_budget"],
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
        "comparisons": comparisons,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dynamic Marginal-Value Query Allocation",
        "",
        f"Pilot K={summary['pilot_k']}; mean total K={summary['total_budget']}.",
        "",
        "| Method | AUC | pAUC | TPR@1% / FPR | TPR@10% | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metric in summary["metrics"].items():
        base = next(
            (
                candidate
                for candidate in METHODS
                if method in (f"{candidate}_active", f"{candidate}_fusion")
            ),
            None,
        )
        error = summary["delta_rmse"].get(base) if base is not None else None
        lines.append(
            f"| `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
            f"{metric['tpr_1']:.4f} / {metric['actual_fpr_1']:.4f} | "
            f"{metric['tpr_10']:.4f} | {'—' if error is None else f'{error:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Delta pAUC | Wins |",
            "|---|---:|---:|",
        ]
    )
    for name, value in summary["comparisons"].items():
        lines.append(
            f"| `{name}` | {value['pauc_0_10']:+.4f} | {value['wins']}/{value['total']} |"
        )
    lines.extend(
        [
            "",
            "| Allocation | min | p10 | median | p90 | max | pilot-only | Gini | true-positive-delta top20% | low-q20% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, value in summary["allocation"].items():
        lines.append(
            f"| `{method}` | {value['minimum']:.1f} | {value['p10']:.1f} | "
            f"{value['median']:.1f} | {value['p90']:.1f} | {value['maximum']:.1f} | "
            f"{value['pilot_only_fraction']:.2%} | {value['gini']:.3f} | "
            f"{value['positive_delta_top20_mean_queries']:.2f} | {value['lowq20_mean_queries']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epochs", nargs="+", choices=EPOCHS, type=int, default=list(EPOCHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/local_shadow",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/dynamic_marginal_query",
    )
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.aggregate_existing:
        all_rows = [
            row
            for path in sorted((output / "conditions").glob("*/RAW_RESULTS.json"))
            for row in json.loads(path.read_text())["rows"]
        ]
        for pilot_k, total_budget in BUDGETS:
            rows = [
                row
                for row in all_rows
                if row["pilot_k"] == pilot_k and row["total_budget"] == total_budget
            ]
            summary = aggregate(rows)
            name = f"p{pilot_k}_b{total_budget}"
            _write_json(output / f"AGGREGATE_{name}.json", summary)
            write_markdown(summary, output / f"AGGREGATE_{name}.md")
        return
    if args.benchmark is None:
        parser.error("--benchmark is required")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    shadow = _load_shadow(args.shadow_root.resolve() / args.benchmark / "shadow_pq.npz")
    policies: dict[tuple[int, int, int], PolicyFit] = {}
    for seed in args.seeds:
        for pilot_k, total_budget in BUDGETS:
            policies[(pilot_k, total_budget, seed)] = fit_shadow_policy(
                shadow,
                pilot_k=pilot_k,
                total_budget=total_budget,
                seed=seed + 7000,
                device=device,
            )
    for epoch in args.epochs:
        for pilot_k, total_budget in BUDGETS:
            condition = (
                output
                / "conditions"
                / f"{args.benchmark}_epoch{epoch}_p{pilot_k}_b{total_budget}"
            )
            rows = []
            for seed in args.seeds:
                row, arrays = evaluate_condition_seed(
                    args.benchmark,
                    epoch,
                    seed,
                    pilot_k,
                    total_budget,
                    policies[(pilot_k, total_budget, seed)],
                    shadow,
                    device,
                )
                rows.append(row)
                condition.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(condition / f"scores_seed_{seed}.npz", **arrays)
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "seed": seed,
                            "policy_validation_loss": row["policy_validation_loss"],
                        }
                    ),
                    flush=True,
                )
            report = {"experiment": "dynamic marginal-value allocation", "rows": rows}
            _write_json(condition / "RAW_RESULTS.json", report)
            summary = aggregate(rows)
            _write_json(condition / "AGGREGATE.json", summary)
            write_markdown(summary, condition / "AGGREGATE.md")


if __name__ == "__main__":
    main()
