"""Learn nonmember-only neural scores and adaptive probe allocation.

This is an exploratory offline verifier replay.  The detector is trained only
on trusted nonmembers plus preregistered synthetic positive-delta alternatives.
Cached exact target probabilities are used only by the verifier simulator that
returns accept/reject bits and by the explicitly reported delta-RMSE diagnostic.
They are never model inputs or synthetic-training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.sd_membership_sft.core.replay_cache import (ReplayData, load_replay_data)
from experiments.sd_membership_sft.analysis.active_importance_replay import (acceptance_probabilities, estimate_corrected_delta0, fragment_score, sample_schedule, uniform_schedule)
from experiments.sd_membership_sft.core.audit_runtime import (_deterministic_subset)
from experiments.sd_membership_sft.archive.adaptive_window_accept_only import (_token_mask, fit_nonmember_model, fixed_q_observations, raw_fragment_scores, token_features, window_priority)
from experiments.sd_membership_sft.core.audit_metrics import (membership_metrics)
from experiments.sd_membership_sft.methods.lowq_baseline import (standardized_max)
from experiments.sd_membership_sft.core.audit_metrics import (rank_auc)
from experiments.sd_membership_sft.core.audit_runtime import (split_indices)

from experiments.sd_membership_sft.core.audit_runtime import (ROOT, BENCHMARKS, EPOCHS, REPLAY_SEEDS, N_REF, N_CAL, SPLIT_SEED, _jsonable, _write_json, _paths)


PILOT_QUERIES = (1, 2)
ACTIVE_BUDGET = 8






def pseudo_member_acceptance(
    base_acceptance: np.ndarray,
    logq0: np.ndarray,
    *,
    seed: int,
    family: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a sparse positive log-ratio effect without real member labels."""
    base = np.clip(np.asarray(base_acceptance, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    logq0 = np.asarray(logq0, dtype=np.float64)
    if base.ndim != 1 or base.shape != logq0.shape or len(base) == 0:
        raise ValueError("base acceptance and logq0 must be aligned nonempty vectors")
    if family not in (0, 1, 2):
        raise ValueError("pseudo-member family must be 0, 1, or 2")
    rng = np.random.default_rng(seed)
    length = len(base)
    selected = np.zeros(length, dtype=bool)
    if family == 0:
        # Sparse low-q positions, but with randomized density so the network
        # cannot recover the label from a fixed 10/20/50 percent boundary.
        fraction = float(rng.uniform(0.05, 0.25))
        pool_count = max(1, int(math.ceil(0.60 * length)))
        pool = np.argsort(logq0, kind="stable")[:pool_count]
        count = min(pool_count, max(1, int(math.ceil(fraction * length))))
        chosen = rng.choice(pool, count, replace=False)
        selected[chosen] = True
    elif family == 1:
        # One or two contiguous regions with an unconstrained effective width.
        widths = np.asarray([4, 8, 16, 32, 64], dtype=np.int64)
        for _ in range(int(rng.integers(1, 3))):
            width = min(length, int(rng.choice(widths)))
            start = int(rng.integers(0, length - width + 1))
            selected[start : start + width] = True
    else:
        # Multiple q-biased local regions approximate dispersed memorization.
        rank = np.empty(length, dtype=np.float64)
        rank[np.argsort(logq0, kind="stable")] = np.linspace(1.0, 0.0, length)
        centers = rng.choice(length, min(3, length), replace=False, p=(rank + 0.1) / np.sum(rank + 0.1))
        for center in centers:
            radius = int(rng.choice(np.asarray([2, 4, 8, 16])))
            selected[max(0, center - radius) : min(length, center + radius + 1)] = True
    amplitude = float(rng.uniform(0.20, 1.25))
    boosted = base.copy()
    boosted[selected] = np.minimum(1.0 - 1e-7, base[selected] * math.exp(amplitude))
    return boosted, selected


def build_evidence_features(
    static: np.ndarray,
    accept_rate: np.ndarray,
    expected_acceptance: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Combine q/context features with the currently observable bit state."""
    static = np.asarray(static, dtype=np.float64)
    rate = np.asarray(accept_rate, dtype=np.float64)
    expected = np.clip(np.asarray(expected_acceptance, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    if static.ndim != 2 or len(static) != len(rate) or rate.shape != expected.shape or k <= 0:
        raise ValueError("misaligned evidence features")
    all_accept = np.isclose(rate, 1.0).astype(np.float64)
    variance = np.maximum(expected * (1.0 - expected) / k, 0.02 / k)
    residual = (rate - expected) / np.sqrt(variance)
    expected_all = expected**k
    all_residual = (all_accept - expected_all) / np.sqrt(
        np.maximum(expected_all * (1.0 - expected_all), 0.02)
    )
    query_scale = np.full(len(rate), 1.0 / math.sqrt(k), dtype=np.float64)
    dynamic = np.column_stack((rate, all_accept, expected, residual, expected_all, all_residual))
    # Encode K without changing the public feature width: scale the residual
    # columns by an explicit deterministic query-count factor.
    dynamic[:, 3] *= query_scale
    dynamic[:, 5] *= query_scale
    return np.asarray(np.column_stack((static, dynamic)), dtype=np.float32)


class _MaskedTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.dropout(F.gelu(self.norm(self.conv(values)))) + values
        return output * mask.unsqueeze(1).to(output.dtype)


class EvidenceTCN(nn.Module):
    """Multi-scale scorer whose token logits also define probe priority."""

    def __init__(self, input_dim: int, channels: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [_MaskedTCNBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8, 16)]
        )
        self.importance = nn.Conv1d(channels, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError("values must be batch-by-token-by-feature with an aligned mask")
        valid = mask.unsqueeze(1).to(values.dtype)
        hidden = self.projection(values.transpose(1, 2)) * valid
        for block in self.blocks:
            hidden = block(hidden, mask)
        token_logits = self.importance(hidden).squeeze(1).masked_fill(~mask, -torch.inf)
        attention = torch.softmax(token_logits, dim=1).unsqueeze(1)
        weighted = (hidden * attention).sum(dim=2)
        mean = (hidden * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=2)
        fragment = self.head(torch.cat((weighted, mean, maximum), dim=1)).squeeze(1)
        return fragment, token_logits


class EvidenceExamples(Dataset[tuple[np.ndarray, float, np.ndarray]]):
    """Real/simulated nonmembers and label-free pseudo-member alternatives."""

    def __init__(
        self,
        static: np.ndarray,
        actual_rate: np.ndarray,
        expected: np.ndarray,
        logq0: np.ndarray,
        offsets: np.ndarray,
        records: np.ndarray,
        *,
        k: int,
        seed: int,
    ) -> None:
        self.static = static
        self.actual_rate = actual_rate
        self.expected = expected
        self.logq0 = logq0
        self.offsets = offsets
        self.records = np.asarray(records, dtype=np.int64)
        self.k = int(k)
        self.seed = int(seed)
        # actual NM, simulated NM, and three alternative families
        self.variants = 5

    def __len__(self) -> int:
        return len(self.records) * self.variants

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, np.ndarray]:
        record = int(self.records[index // self.variants])
        variant = int(index % self.variants)
        start, end = int(self.offsets[record]), int(self.offsets[record + 1])
        expected = self.expected[start:end]
        importance = np.zeros(end - start, dtype=np.float32)
        if variant == 0:
            rate = self.actual_rate[start:end]
            label = 0.0
        else:
            rng_seed = int(np.random.SeedSequence([self.seed, record, variant]).generate_state(1)[0])
            alpha = expected
            label = float(variant >= 2)
            if variant >= 2:
                alpha, selected = pseudo_member_acceptance(
                    expected,
                    self.logq0[start:end],
                    seed=rng_seed,
                    family=variant - 2,
                )
                importance[selected] = 1.0
            rng = np.random.default_rng(rng_seed + 17)
            rate = np.mean(rng.random((end - start, self.k)) < alpha[:, None], axis=1)
        features = build_evidence_features(
            self.static[start:end], rate, expected, k=self.k
        )
        return features, label, importance


class ProbeValueExamples(Dataset[tuple[np.ndarray, float, np.ndarray]]):
    """Nonmember-only token targets measuring where extra probes reduce error."""

    def __init__(
        self,
        static: np.ndarray,
        rate: np.ndarray,
        expected: np.ndarray,
        value: np.ndarray,
        offsets: np.ndarray,
        records: np.ndarray,
        *,
        k: int,
    ) -> None:
        self.static = static
        self.rate = rate
        self.expected = expected
        self.value = value
        self.offsets = offsets
        self.records = np.asarray(records, dtype=np.int64)
        self.k = int(k)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, np.ndarray]:
        record = int(self.records[index])
        start, end = int(self.offsets[record]), int(self.offsets[record + 1])
        features = build_evidence_features(
            self.static[start:end], self.rate[start:end], self.expected[start:end], k=self.k
        )
        return features, 0.0, np.asarray(self.value[start:end], dtype=np.float32)


def _collate(
    batch: list[tuple[np.ndarray, float, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty evidence batch")
    width = max(len(item[0]) for item in batch)
    feature_dim = batch[0][0].shape[1]
    values = torch.zeros((len(batch), width, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), width), dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.float32)
    targets = torch.zeros((len(batch), width), dtype=torch.float32)
    for row, (features, label, importance) in enumerate(batch):
        length = len(features)
        values[row, :length] = torch.from_numpy(features)
        mask[row, :length] = True
        labels[row] = label
        targets[row, :length] = torch.from_numpy(importance)
    return values, mask, labels, targets


@dataclass(frozen=True)
class FitResult:
    model: EvidenceTCN
    best_epoch: int
    validation_auc: float


@torch.inference_mode()
def _synthetic_auc(
    model: EvidenceTCN, loader: DataLoader, device: torch.device
) -> float:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for values, mask, batch_labels, _ in loader:
        logits, _ = model(values.to(device), mask.to(device))
        scores.append(logits.cpu().numpy())
        labels.append(batch_labels.numpy())
    score = np.concatenate(scores)
    label = np.concatenate(labels).astype(np.int64)
    return rank_auc(score[label == 1], score[label == 0])


def fit_evidence_model(
    train: EvidenceExamples,
    validation: EvidenceExamples,
    *,
    input_dim: int,
    seed: int,
    device: torch.device,
    max_epochs: int = 24,
    patience: int = 5,
    batch_size: int = 24,
) -> FitResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    model = EvidenceTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )
    best_auc, best_epoch, stale = -np.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for values, mask, labels, target in train_loader:
            values, mask = values.to(device), mask.to(device)
            labels, target = labels.to(device), target.to(device)
            fragment, token_logits = model(values, mask)
            classification = F.binary_cross_entropy_with_logits(fragment, labels)
            positive = target.sum(dim=1) > 0
            if torch.any(positive):
                log_weights = F.log_softmax(token_logits[positive], dim=1)
                distribution = target[positive] / target[positive].sum(dim=1, keepdim=True)
                importance = -torch.sum(
                    distribution * log_weights.masked_fill(~mask[positive], 0.0), dim=1
                ).mean()
            else:
                importance = fragment.new_zeros(())
            loss = classification + 0.15 * importance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        validation_auc = _synthetic_auc(model, validation_loader, device)
        if validation_auc > best_auc + 1e-4:
            best_auc, best_epoch, stale = validation_auc, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("neural evidence training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return FitResult(model=model, best_epoch=best_epoch, validation_auc=float(best_auc))


def fit_probe_value_model(
    train: ProbeValueExamples,
    validation: ProbeValueExamples,
    *,
    input_dim: int,
    seed: int,
    device: torch.device,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 24,
) -> FitResult:
    """Imitate high-query nonmember measurement value with a listwise loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = EvidenceTCN(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    def listwise_loss(loader: DataLoader, update: bool) -> float:
        model.train(update)
        total, batches = 0.0, 0
        for values, mask, _, target in loader:
            values, mask, target = values.to(device), mask.to(device), target.to(device)
            _, token_logits = model(values, mask)
            target = torch.where(mask, target.clamp_min(0.0) + 1e-6, 0.0)
            distribution = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
            log_weights = F.log_softmax(token_logits, dim=1).masked_fill(~mask, 0.0)
            loss = -torch.sum(distribution * log_weights, dim=1).mean()
            if update:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        return total / max(1, batches)

    best_loss, best_epoch, stale = np.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, max_epochs + 1):
        listwise_loss(train_loader, True)
        with torch.inference_mode():
            validation_loss = listwise_loss(validation_loader, False)
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("probe-value training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    # validation_auc stores a direction-preserving convenience score here.
    return FitResult(model=model, best_epoch=best_epoch, validation_auc=float(-best_loss))


@torch.inference_mode()
def predict_evidence(
    model: EvidenceTCN,
    static: np.ndarray,
    rates: np.ndarray,
    expected: np.ndarray,
    offsets: np.ndarray,
    *,
    k: int,
    device: torch.device,
    batch_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(offsets) - 1, dtype=np.float64)
    token_values = np.empty(len(rates), dtype=np.float64)
    for batch_start in range(0, len(scores), batch_size):
        batch_end = min(len(scores), batch_start + batch_size)
        items = []
        for record in range(batch_start, batch_end):
            start, end = int(offsets[record]), int(offsets[record + 1])
            features = build_evidence_features(
                static[start:end], rates[start:end], expected[start:end], k=k
            )
            items.append((features, 0.0, np.zeros(end - start, dtype=np.float32)))
        values, mask, _, _ = _collate(items)
        fragment, token = model(values.to(device), mask.to(device))
        scores[batch_start:batch_end] = fragment.cpu().numpy()
        token = token.cpu().numpy()
        for row, record in enumerate(range(batch_start, batch_end)):
            start, end = int(offsets[record]), int(offsets[record + 1])
            token_values[start:end] = token[row, : end - start]
    return scores, token_values


def learned_pilot_schedule(
    priority: np.ndarray,
    *,
    budget: int = ACTIVE_BUDGET,
    selected_fraction: float = 0.50,
) -> np.ndarray:
    """Two normal-q pilots, an active coverage floor, then learned extras."""
    priority = np.asarray(priority, dtype=np.float64)
    length = len(priority)
    if length <= 0 or budget < 4 or not 0.0 < selected_fraction <= 1.0:
        raise ValueError("invalid learned schedule arguments")
    selected_count = max(1, int(math.ceil(selected_fraction * length)))
    selected = np.argsort(-priority, kind="stable")[:selected_count]
    remaining = (budget - 2) * length
    uniform_rounds = max(1, budget - 4)
    uniform_total = uniform_rounds * length
    adaptive_total = remaining - uniform_total
    if adaptive_total < 0:
        raise ValueError("budget is too small for the coverage floor")
    base, remainder = divmod(adaptive_total, selected_count)
    counts = np.full(selected_count, base, dtype=np.int64)
    counts[:remainder] += 1
    width = 2 + uniform_rounds + int(np.max(counts, initial=0))
    schedule = np.full((length, width), -1, dtype=np.int8)
    schedule[:, :2] = 0
    ladder = np.asarray([1, 2, 3, 4], dtype=np.int8)
    schedule[:, 2 : 2 + uniform_rounds] = np.resize(ladder, uniform_rounds)
    for rank, position in enumerate(selected):
        start = 2 + uniform_rounds
        schedule[position, start : start + counts[rank]] = np.resize(ladder, counts[rank])
    if int(np.sum(schedule >= 0)) != budget * length:
        raise RuntimeError("learned schedule violated the exact decision budget")
    return schedule


def _uniform_pilot_schedule(length: int, budget: int = ACTIVE_BUDGET) -> np.ndarray:
    if length <= 0 or budget < 2:
        raise ValueError("invalid uniform pilot schedule")
    levels = np.r_[np.zeros(2, dtype=np.int8), np.resize(np.asarray([1, 2, 3, 4], dtype=np.int8), budget - 2)]
    return np.broadcast_to(levels, (length, budget)).copy()


def _record_uniforms(seed: int, record: int, length: int, width: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, record, 8801])).random(
        (length, width), dtype=np.float64
    )


def nonmember_probe_value_targets(
    data: ReplayData,
    records: np.ndarray,
    pilot_rate: np.ndarray,
    *,
    pilot_k: int,
    replay_seed: int,
    teacher_budget: int = 32,
) -> np.ndarray:
    """Build API-realizable high-query teacher targets on nonmembers only."""
    records = np.asarray(records, dtype=np.int64)
    if pilot_k <= 0 or teacher_budget <= pilot_k or np.any(data.labels[records] != 0):
        raise ValueError("probe-value teachers require only trusted nonmembers and a larger budget")
    values = np.zeros(len(data.logq0), dtype=np.float64)
    for record in records:
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        length = end - start
        pilot_accepts = np.zeros((length, 5), dtype=np.int64)
        pilot_trials = np.zeros((length, 5), dtype=np.int64)
        pilot_accepts[:, 0] = np.rint(pilot_rate[start:end] * pilot_k).astype(np.int64)
        pilot_trials[:, 0] = pilot_k
        pilot_delta = estimate_corrected_delta0(
            data.logq0[start:end], pilot_accepts, pilot_trials, bisection_steps=24
        ).delta0
        schedule = uniform_schedule(length, teacher_budget, active=True)
        alpha = acceptance_probabilities(data.logp[start:end], data.logq0[start:end])
        uniforms = _record_uniforms(
            replay_seed + 31001, int(record), length, schedule.shape[1]
        )
        accepts, trials = sample_schedule(alpha, schedule, uniforms)
        teacher_delta = estimate_corrected_delta0(
            data.logq0[start:end], accepts, trials, bisection_steps=24
        ).delta0
        error = np.square(teacher_delta - pilot_delta)
        width = min(8, length)
        kernel = np.ones(width, dtype=np.float64)
        local = np.convolve(error, kernel, mode="same") / np.convolve(
            np.ones(length, dtype=np.float64), kernel, mode="same"
        )
        values[start:end] = 0.5 * error + 0.5 * local
    return values


def active_delta_scores(
    data: ReplayData,
    priority: np.ndarray | None,
    *,
    replay_seed: int,
    budget: int = ACTIVE_BUDGET,
) -> tuple[np.ndarray, float]:
    scores = np.empty(len(data.lengths), dtype=np.float64)
    squared_error, token_count = 0.0, 0
    for record in range(len(data.lengths)):
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        if priority is None:
            schedule = _uniform_pilot_schedule(end - start, budget)
        else:
            schedule = learned_pilot_schedule(priority[start:end], budget=budget)
        alpha = acceptance_probabilities(data.logp[start:end], data.logq0[start:end])
        uniforms = _record_uniforms(replay_seed, record, end - start, schedule.shape[1])
        accepts, trials = sample_schedule(alpha, schedule, uniforms)
        estimate = estimate_corrected_delta0(
            data.logq0[start:end], accepts, trials, bisection_steps=24
        ).delta0
        truth = data.logp[start:end] - data.logq0[start:end]
        squared_error += float(np.sum((estimate - truth) ** 2))
        token_count += end - start
        scores[record] = fragment_score(estimate, "window_sign_8")
    return scores, math.sqrt(squared_error / token_count)


def _standardize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    fit = np.asarray(values, dtype=np.float64)[reference]
    scale = max(float(np.std(fit)), 1e-8)
    return (np.asarray(values, dtype=np.float64) - float(np.mean(fit))) / scale




def evaluate_condition_seed(
    data: ReplayData,
    static_raw: np.ndarray,
    *,
    benchmark: str,
    epoch: int,
    replay_seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    partitions = split_indices(data.labels, SPLIT_SEED)
    d_nonmember = partitions["D"][data.labels[partitions["D"]] == 0]
    c_nonmember = partitions["C"][data.labels[partitions["C"]] == 0]
    reference = _deterministic_subset(d_nonmember, N_REF, SPLIT_SEED + N_REF)
    calibration = _deterministic_subset(c_nonmember, N_CAL, SPLIT_SEED + N_CAL + 1000)
    rng = np.random.default_rng(SPLIT_SEED + replay_seed)
    shuffled = rng.permutation(reference)
    train_records, validation_records = np.sort(shuffled[:320]), np.sort(shuffled[320:])
    static_mean = np.mean(static_raw[_token_mask(train_records, data.offsets)], axis=0)
    static_scale = np.std(static_raw[_token_mask(train_records, data.offsets)], axis=0)
    static_scale = np.where(static_scale < 1e-6, 1.0, static_scale)
    static = np.asarray((static_raw - static_mean) / static_scale, dtype=np.float32)

    observations = {
        k: fixed_q_observations(data, k, replay_seed) for k in PILOT_QUERIES
    }
    scores: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    neural_priority: np.ndarray | None = None
    predicted_k2: np.ndarray | None = None
    for k in PILOT_QUERIES:
        rates, all_accept = observations[k]
        null_model = fit_nonmember_model(
            static_raw,
            rates,
            _token_mask(train_records, data.offsets),
            trials_per_token=k,
            seed=replay_seed + k,
        )
        predicted = null_model.predict(static_raw)
        train = EvidenceExamples(
            static,
            rates,
            predicted,
            data.logq0,
            data.offsets,
            train_records,
            k=k,
            seed=replay_seed + 100 * k,
        )
        validation = EvidenceExamples(
            static,
            rates,
            predicted,
            data.logq0,
            data.offsets,
            validation_records,
            k=k,
            seed=replay_seed + 1000 + 100 * k,
        )
        fit = fit_evidence_model(
            train,
            validation,
            input_dim=static.shape[1] + 6,
            seed=replay_seed + 10 * k,
            device=device,
        )
        neural, priority = predict_evidence(
            fit.model,
            static,
            rates,
            predicted,
            data.offsets,
            k=k,
            device=device,
        )
        raw = raw_fragment_scores(all_accept, predicted, data.logq0, data.lengths, k)
        names = ("lowq_10", "lowq_20", "lowq_50")
        lowq = standardized_max(raw, names, reference)
        scores[f"lowq_k{k}"] = lowq
        scores[f"neural_score_k{k}"] = neural
        scores[f"lowq_plus_neural_k{k}"] = _standardize(lowq, reference) + 0.25 * _standardize(neural, reference)
        metadata[f"k{k}"] = {
            "best_epoch": fit.best_epoch,
            "synthetic_validation_auc": fit.validation_auc,
        }
        if k == 2:
            neural_priority = priority
            predicted_k2 = predicted

    if neural_priority is None or predicted_k2 is None:
        raise RuntimeError("K=2 model is required for adaptive probing")
    rates_k2, _ = observations[2]
    probe_targets = nonmember_probe_value_targets(
        data,
        reference,
        rates_k2,
        pilot_k=2,
        replay_seed=replay_seed,
    )
    probe_fit = fit_probe_value_model(
        ProbeValueExamples(
            static,
            rates_k2,
            predicted_k2,
            probe_targets,
            data.offsets,
            train_records,
            k=2,
        ),
        ProbeValueExamples(
            static,
            rates_k2,
            predicted_k2,
            probe_targets,
            data.offsets,
            validation_records,
            k=2,
        ),
        input_dim=static.shape[1] + 6,
        seed=replay_seed + 9002,
        device=device,
    )
    _, measurement_priority = predict_evidence(
        probe_fit.model,
        static,
        rates_k2,
        predicted_k2,
        data.offsets,
        k=2,
        device=device,
    )
    metadata["probe_value"] = {
        "best_epoch": probe_fit.best_epoch,
        "validation_listwise_objective": probe_fit.validation_auc,
        "teacher_budget": 32,
        "teacher_member_count": 0,
    }
    rule_priority = np.empty_like(neural_priority)
    for record in range(len(data.lengths)):
        start, end = int(data.offsets[record]), int(data.offsets[record + 1])
        rule_priority[start:end] = window_priority(
            rates_k2[start:end], predicted_k2[start:end], data.logq0[start:end]
        )
    uniform_active, uniform_rmse = active_delta_scores(
        data, None, replay_seed=replay_seed
    )
    rule_active, rule_rmse = active_delta_scores(
        data, rule_priority, replay_seed=replay_seed
    )
    neural_active, neural_rmse = active_delta_scores(
        data, neural_priority, replay_seed=replay_seed
    )
    measurement_active, measurement_rmse = active_delta_scores(
        data, measurement_priority, replay_seed=replay_seed
    )
    scores.update(
        {
            "uniform_active_k8": uniform_active,
            "rule_active_k8": rule_active,
            "neural_active_k8": neural_active,
            "measurement_neural_active_k8": measurement_active,
            "canp_fusion_k8": (
                _standardize(scores["lowq_k2"], reference)
                + 0.25 * _standardize(scores["neural_score_k2"], reference)
                + 0.25 * _standardize(neural_active, reference)
            ),
            "uniform_fusion_k8": (
                _standardize(scores["lowq_k2"], reference)
                + 0.25 * _standardize(scores["neural_score_k2"], reference)
                + 0.25 * _standardize(uniform_active, reference)
            ),
            "measurement_fusion_k8": (
                _standardize(scores["lowq_k2"], reference)
                + 0.25 * _standardize(scores["neural_score_k2"], reference)
                + 0.25 * _standardize(measurement_active, reference)
            ),
        }
    )
    metrics = {
        name: membership_metrics(values, data.labels, calibration, partitions["T"])
        for name, values in scores.items()
    }
    row = {
        "benchmark": benchmark,
        "epoch": epoch,
        "seed": replay_seed,
        "n_ref": N_REF,
        "n_cal": N_CAL,
        "training_member_count": 0,
        "pseudo_families": ["sparse_low_q", "contiguous", "dispersed_local"],
        "model": metadata,
        "active_delta_rmse": {
            "uniform_active_k8": uniform_rmse,
            "rule_active_k8": rule_rmse,
            "neural_active_k8": neural_rmse,
            "measurement_neural_active_k8": measurement_rmse,
        },
        "metrics": metrics,
    }
    arrays = {
        "labels": data.labels,
        "record_ids": data.record_ids,
        **scores,
    }
    return row, arrays


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate no neural adaptive rows")
    methods = tuple(rows[0]["metrics"])
    metrics: dict[str, Any] = {}
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
        "experiment": "nonmember-only neural evidence and adaptive probing",
        "status": "exploratory cached-T offline verifier replay",
        "conditions": sorted({f"{row['benchmark']}_epoch{row['epoch']}" for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "rows": len(rows),
        "metrics": metrics,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Nonmember-Only Neural Evidence and Adaptive Probing",
        "",
        "> Exploratory cached-T offline verifier replay; exact p is simulator-only.",
        "",
        "| Method | AUC | pAUC | TPR@1% | actual FPR | TPR@10% | actual FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metric in summary["metrics"].items():
        lines.append(
            f"| `{method}` | {metric['auc']:.4f} | {metric['pauc_0_10']:.4f} | "
            f"{metric['tpr_1']:.4f} | {metric['actual_fpr_1']:.4f} | "
            f"{metric['tpr_10']:.4f} | {metric['actual_fpr_10']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS)
    parser.add_argument("--epoch", choices=EPOCHS, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REPLAY_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/results/sft_runs/accept_only_active_v2/neural_adaptive",
    )
    parser.add_argument("--aggregate-existing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.aggregate_existing:
        paths = sorted((output / "conditions").glob("*/RAW_RESULTS.json"))
        if not paths:
            raise RuntimeError("no condition results found")
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        summary = aggregate(rows)
        _write_json(output / "AGGREGATE.json", summary)
        write_markdown(summary, output / "AGGREGATE.md")
        print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))
        return
    if args.benchmark is None or args.epoch is None:
        parser.error("--benchmark and --epoch are required unless --aggregate-existing is used")
    full, pq = _paths(args.benchmark, args.epoch)
    data = load_replay_data(full, pq)
    static = token_features(data.logq0, data.lengths)
    device = _resolve_device(args.device)
    condition = output / "conditions" / f"{args.benchmark}_epoch{args.epoch}"
    rows = []
    for seed in args.seeds:
        row, arrays = evaluate_condition_seed(
            data,
            static,
            benchmark=args.benchmark,
            epoch=args.epoch,
            replay_seed=seed,
            device=device,
        )
        rows.append(row)
        condition.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(condition / f"scores_seed_{seed}.npz", **arrays)
        print(json.dumps({"condition": condition.name, "seed": seed, "device": str(device)}), flush=True)
    report = {
        "experiment": "nonmember-only neural evidence and adaptive probing",
        "protocol": {
            "training": "trusted nonmembers plus preregistered synthetic positive-delta alternatives",
            "training_member_count": 0,
            "n_ref": N_REF,
            "n_cal": N_CAL,
            "pilot_queries": list(PILOT_QUERIES),
            "active_budget": ACTIVE_BUDGET,
            "threshold": "split conformal on independent real nonmembers",
            "exact_p_policy": "offline verifier simulation and delta-RMSE only",
        },
        "rows": rows,
    }
    _write_json(condition / "RAW_RESULTS.json", report)
    summary = aggregate(rows)
    _write_json(condition / "AGGREGATE.json", summary)
    write_markdown(summary, condition / "AGGREGATE.md")
    print(json.dumps({"output": str(condition), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
