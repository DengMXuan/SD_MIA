"""Stat-Delta membership audit.

The detector-side input in this module is restricted to fixed statistics of
``delta = logp - logq``.  It deliberately does not load p/q or activation
features.  The command operates on one benchmark/epoch condition so six
conditions can be scheduled independently.

Example::

    .venv/bin/python -m experiments.sd_membership_sft.stat_delta_mia \
        --benchmark wikitection --epoch 1 --device cpu
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn

from experiments.shared.core.audit_runtime import DEFAULT_SPLIT_SEED, split_indices
from experiments.sd_membership_sft.archive.full_delta_mia import DEFAULT_TRAINING_SEEDS, _bootstrap_metrics, _metric_point, fit_row_standardizer
from experiments.shared.core.audit_metrics import partial_auc

from experiments.shared.core.replay_cache import DeltaData, load_delta_data, sliding_means

from experiments.paths import ROOT
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
WINDOWS = (4, 8, 16, 32, 64)
FEATURE_PACKS = ("S11", "S18", "S22")
EPS = 1e-8








def _s11(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(values, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    negative = np.minimum(values, 0.0)
    row = np.asarray(
        [
            np.mean(values),
            np.std(values),
            np.mean(np.abs(values)),
            np.mean(positive),
            np.mean(negative),
            np.mean(values > 0.0),
            *np.quantile(values, (0.10, 0.25, 0.50, 0.75, 0.90)),
        ],
        dtype=np.float64,
    )
    names = (
        "delta_mean",
        "delta_std",
        "delta_abs_mean",
        "delta_positive_part_mean",
        "delta_negative_part_mean",
        "delta_positive_fraction",
        "delta_q10",
        "delta_q25",
        "delta_q50",
        "delta_q75",
        "delta_q90",
    )
    return row, names


def _mean_alpha(values: np.ndarray) -> float:
    # The clip before exp prevents overflow while preserving the acceptance
    # rule to float precision for all values in the saved cache.
    return float(np.mean(np.minimum(1.0, np.exp(np.clip(values, -80.0, 80.0)))))


def _window_sign(values: np.ndarray, width: int) -> float:
    means = sliding_means(values, width)
    return float(np.mean(means > 0.0))


def _s18(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    base, names = _s11(values)
    signs = np.asarray([_window_sign(values, width) for width in WINDOWS], dtype=np.float64)
    row = np.concatenate((base, [_mean_alpha(values)], signs, [float(np.mean(signs))]))
    return row, names + (
        "mean_alpha",
        "window_sign_4",
        "window_sign_8",
        "window_sign_16",
        "window_sign_32",
        "window_sign_64",
        "window_sign_multiscale",
    )


def _s22(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    base, names = _s11(values)
    local: list[float] = []
    local_names: list[str] = []
    for width in WINDOWS:
        means = sliding_means(values, width)
        local.extend((float(np.quantile(means, 0.10)), float(np.quantile(means, 0.90))))
        local_names.extend((f"window_mean_{width}_q10", f"window_mean_{width}_q90"))
    row = np.concatenate((base, [_mean_alpha(values)], np.asarray(local, dtype=np.float64)))
    return row, names + ("mean_alpha",) + tuple(local_names)


def feature_row(values: np.ndarray, pack: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if pack == "S11":
        return _s11(values)
    if pack == "S18":
        return _s18(values)
    if pack == "S22":
        return _s22(values)
    raise ValueError(f"unknown feature pack {pack!r}")


def document_features(data: DeltaData, pack: str) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for start, end in zip(data.offsets[:-1], data.offsets[1:]):
        row, row_names = feature_row(data.delta[int(start) : int(end)], pack)
        rows.append(row)
        names = row_names
    matrix = np.asarray(rows, dtype=np.float64)
    if names is None or matrix.shape[1] != len(names):
        raise RuntimeError("feature construction produced no rows or wrong dimension")
    return matrix, names


def chunk_features(data: DeltaData, pack: str = "S22", slots: int = 4) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct fixed K position slots without exposing a length feature."""
    if slots <= 0:
        raise ValueError("slots must be positive")
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for start, end in zip(data.offsets[:-1], data.offsets[1:]):
        pieces = np.array_split(data.delta[int(start) : int(end)], slots)
        if any(len(piece) == 0 for piece in pieces):
            raise ValueError("a record is shorter than the requested fixed slot count")
        piece_rows: list[np.ndarray] = []
        for slot, piece in enumerate(pieces):
            row, row_names = feature_row(piece, pack)
            piece_rows.append(row)
            if names is None:
                names = tuple(f"slot{slot}_{name}" for name in row_names)
        if names is not None:
            names = tuple(
                f"slot{slot_index}_{name}"
                for slot_index in range(slots)
                for name in row_names
            )
        rows.append(np.stack(piece_rows, axis=0))
    matrix = np.stack(rows, axis=0)
    if names is None:
        raise RuntimeError("chunk feature construction produced no rows")
    return matrix, names


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def fit_logistic(values: np.ndarray, labels: np.ndarray, train: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    x = np.asarray(values[train], dtype=np.float64)
    y = np.asarray(labels[train], dtype=np.float64)
    class_counts = {label: max(1, int(np.sum(y == label))) for label in (0.0, 1.0)}
    weights = np.where(y == 1.0, 1.0 / (2.0 * class_counts[1.0]), 1.0 / (2.0 * class_counts[0.0]))

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        score = x @ params[:-1] + params[-1]
        probability = _sigmoid(score)
        gradient_score = weights * (probability - y)
        loss = float(
            np.sum(weights * np.logaddexp(0.0, score) - weights * y * score)
            + l2 * np.sum(params[:-1] ** 2)
        )
        gradient = np.r_[x.T @ gradient_score + 2.0 * l2 * params[:-1], np.sum(gradient_score)]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


class RegressionTree:
    def __init__(self, depth: int, min_leaf: int, max_thresholds: int = 16) -> None:
        self.depth = int(depth)
        self.min_leaf = int(min_leaf)
        self.max_thresholds = int(max_thresholds)
        self.root: tuple[Any, ...] | None = None

    def _build(self, x: np.ndarray, residual: np.ndarray, indices: np.ndarray, depth: int) -> tuple[Any, ...]:
        value = float(np.mean(residual[indices]))
        if depth <= 0 or len(indices) < 2 * self.min_leaf:
            return ("leaf", value)
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        parent_mean = value
        parent_loss = float(np.sum((residual[indices] - parent_mean) ** 2))
        for feature in range(x.shape[1]):
            column = x[indices, feature]
            finite = column[np.isfinite(column)]
            if len(finite) == 0 or float(np.min(finite)) == float(np.max(finite)):
                continue
            quantiles = np.linspace(0.05, 0.95, self.max_thresholds)
            thresholds = np.unique(np.quantile(finite, quantiles))
            for threshold in thresholds:
                left = indices[column <= threshold]
                right = indices[column > threshold]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                loss = float(np.sum((residual[left] - np.mean(residual[left])) ** 2) + np.sum((residual[right] - np.mean(residual[right])) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, feature, float(threshold), left, right)
        if best is None or best[0] >= parent_loss - 1e-12:
            return ("leaf", value)
        _, feature, threshold, left, right = best
        return (
            "node",
            feature,
            threshold,
            self._build(x, residual, left, depth - 1),
            self._build(x, residual, right, depth - 1),
        )

    def fit(self, x: np.ndarray, residual: np.ndarray) -> "RegressionTree":
        self.root = self._build(x, residual, np.arange(len(x), dtype=np.int64), self.depth)
        return self

    def _predict_one(self, row: np.ndarray, node: tuple[Any, ...]) -> float:
        if node[0] == "leaf":
            return float(node[1])
        return self._predict_one(row, node[3] if row[int(node[1])] <= float(node[2]) else node[4])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("tree is not fitted")
        return np.asarray([self._predict_one(row, self.root) for row in x], dtype=np.float64)


class GradientBoostedTrees:
    def __init__(self, estimators: int, depth: int, learning_rate: float, min_leaf: int) -> None:
        self.estimators = int(estimators)
        self.depth = int(depth)
        self.learning_rate = float(learning_rate)
        self.min_leaf = int(min_leaf)
        self.initial = 0.0
        self.trees: list[RegressionTree] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        self.initial = float(np.log(np.mean(y) / max(EPS, 1.0 - np.mean(y))))
        scores = np.full(len(y), self.initial, dtype=np.float64)
        self.trees = []
        for _ in range(self.estimators):
            residual = y - _sigmoid(scores)
            tree = RegressionTree(self.depth, self.min_leaf).fit(x, residual)
            scores += self.learning_rate * tree.predict(x)
            self.trees.append(tree)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = np.full(len(x), self.initial, dtype=np.float64)
        for tree in self.trees:
            scores += self.learning_rate * tree.predict(x)
        return scores


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden:
            layers.extend((nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)))
            current = width
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ChunkAttention(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.weight = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token(x)
        weights = torch.softmax(self.weight(h).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head((h * weights).sum(dim=1)).squeeze(-1)


class ChunkTCN(nn.Module):
    def __init__(self, dim: int, channels: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, channels, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat((h.mean(dim=2), h.amax(dim=2)), dim=1)).squeeze(-1)


class ChunkTransformer(nn.Module):
    def __init__(self, dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.projection(x)).mean(dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _candidate_grid(family: str) -> list[dict[str, Any]]:
    if family == "logistic":
        return [{"l2": value} for value in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)]
    if family == "mlp":
        return [
            {"hidden": hidden, "dropout": dropout, "lr": lr, "weight_decay": wd}
            for hidden in ((32,), (64, 32), (128, 64))
            for dropout in (0.0, 0.1)
            for lr in (3e-4, 1e-3)
            for wd in (1e-4, 1e-3)
        ]
    if family == "gbdt":
        return [
            {"estimators": n, "depth": depth, "learning_rate": lr, "min_leaf": leaf}
            for n in (50, 100)
            for depth in (1, 2)
            for lr in (0.03, 0.1)
            for leaf in (10, 30)
        ]
    if family == "triplet":
        return [
            {"embedding": embedding, "margin": margin, "lr": lr, "weight_decay": wd}
            for embedding in (8, 16, 32)
            for margin in (0.2, 0.5)
            for lr in (3e-4, 1e-3)
            for wd in (1e-4, 1e-3)
        ]
    if family == "attention":
        return [
            {"hidden": hidden, "dropout": dropout, "lr": lr, "weight_decay": wd}
            for hidden in (32, 64)
            for dropout in (0.0, 0.1)
            for lr in (3e-4, 1e-3)
            for wd in (1e-4, 1e-3)
        ]
    if family == "tcn":
        return [
            {"channels": channels, "kernel": kernel, "dropout": dropout, "lr": lr, "weight_decay": wd}
            for channels in (16, 32)
            for kernel in (3, 5)
            for dropout in (0.0, 0.1)
            for lr in (3e-4, 1e-3)
            for wd in (1e-4, 1e-3)
        ]
    if family == "transformer":
        return [
            {"hidden": hidden, "layers": layers, "dropout": dropout, "lr": lr, "weight_decay": wd}
            for hidden in (32, 64)
            for layers in (1, 2)
            for dropout in (0.0, 0.1)
            for lr in (3e-4, 1e-3)
            for wd in (1e-4, 1e-3)
        ]
    raise ValueError(f"unknown family {family!r}")


def _train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    spec: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    set_seed(seed)
    model = MLP(x.shape[1], tuple(int(v) for v in spec["hidden"]), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y.astype(np.float32)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best = -np.inf
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        logits = model(x_tensor[train])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_scores = model(x_tensor[validation]).cpu().numpy()
        score = partial_auc(val_scores, y[validation])
        if score > best + 1e-10:
            best = score
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scores = model(x_tensor).cpu().numpy().astype(np.float64)
    return scores, float(best), int(best_epoch)


class TripletEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, embedding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


def _train_triplet(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    spec: dict[str, Any],
    seed: int,
    max_epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    set_seed(seed)
    train_member = train[y[train] == 1]
    train_nonmember = train[y[train] == 0]
    if len(train_member) < 3 or len(train_nonmember) < 3:
        raise ValueError("triplet training requires both classes")
    model = TripletEncoder(x.shape[1], int(spec["embedding"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    loss_fn = nn.TripletMarginLoss(margin=float(spec["margin"]), p=2)
    x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
    rng = np.random.default_rng(seed)
    best = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        n = min(len(train_member), len(train_nonmember))
        member = rng.choice(train_member, size=n, replace=False)
        nonmember = rng.choice(train_nonmember, size=n, replace=False)
        member_positive = rng.choice(train_member, size=n, replace=True)
        nonmember_positive = rng.choice(train_nonmember, size=n, replace=True)
        anchor = np.concatenate((member, nonmember))
        positive = np.concatenate((member_positive, nonmember_positive))
        negative = np.concatenate((nonmember, member))
        model.train()
        loss = loss_fn(model(x_tensor[anchor]), model(x_tensor[positive]), model(x_tensor[negative]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            embeddings = model(x_tensor).cpu().numpy().astype(np.float64)
        weight, bias = fit_logistic(embeddings, y, train, 1e-2)
        val_score = embeddings[validation] @ weight + bias
        score = partial_auc(val_score, y[validation])
        if score > best + 1e-10:
            best = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("triplet training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        embeddings = model(x_tensor).cpu().numpy().astype(np.float64)
    weight, bias = fit_logistic(embeddings, y, train, 1e-2)
    return embeddings @ weight + bias, float(best), int(best_epoch)


def _fit_chunk_neural(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    family: str,
    spec: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, float, int]:
    set_seed(seed)
    dim = values.shape[2]
    if family == "attention":
        model: nn.Module = ChunkAttention(dim, int(spec["hidden"]), float(spec["dropout"]))
    elif family == "tcn":
        model = ChunkTCN(dim, int(spec["channels"]), int(spec["kernel"]), float(spec["dropout"]))
    elif family == "transformer":
        model = ChunkTransformer(dim, int(spec["hidden"]), int(spec["layers"]), float(spec["dropout"]))
    else:
        raise ValueError(f"unknown chunk family {family!r}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    x_tensor = torch.from_numpy(values.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    best = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        logits = model(x_tensor[train])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_scores = model(x_tensor[validation]).cpu().numpy()
        score = partial_auc(validation_scores, labels[validation])
        if score > best + 1e-10:
            best = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("chunk model did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scores = model(x_tensor).cpu().numpy().astype(np.float64)
    return scores, float(best), int(best_epoch)


def _standardize_document(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = fit_row_standardizer(values, train)
    transformed = scaler.transform(values)
    return transformed, {"mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "fit_split": "D"}


def _standardize_chunks(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    flat = values.reshape(-1, values.shape[-1])
    train_flat = values[train].reshape(-1, values.shape[-1])
    mean = train_flat.mean(axis=0)
    std = np.where(train_flat.std(axis=0) < 1e-8, 1.0, train_flat.std(axis=0))
    transformed = (flat - mean) / std
    return transformed.reshape(values.shape), {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "D"}


def _run_method(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    partitions: dict[str, np.ndarray],
    family: str,
    training_seeds: tuple[int, ...],
    device: torch.device,
    max_epochs: int,
    patience: int,
    bootstrap_repeats: int = 1000,
) -> dict[str, Any]:
    train, validation = partitions["D"], partitions["V"]
    is_sequence = values.ndim == 3
    transformed, scaler = _standardize_chunks(values, train) if is_sequence else _standardize_document(values, train)
    candidates = _candidate_grid(family)
    candidate_table: list[dict[str, Any]] = []
    selected_scores: dict[int, np.ndarray] = {}
    for index, spec in enumerate(candidates):
        seed_results: list[dict[str, Any]] = []
        seed_scores: dict[int, np.ndarray] = {}
        for seed in training_seeds:
            started = time.perf_counter()
            if family == "logistic":
                flat = transformed.reshape(len(transformed), -1)
                weight, bias = fit_logistic(flat, labels, train, float(spec["l2"]))
                scores = flat @ weight + bias
                val = partial_auc(scores[validation], labels[validation])
                epoch = 0
            elif family == "mlp":
                flat = transformed.reshape(len(transformed), -1)
                scores, val, epoch = _train_mlp(flat, labels, train, validation, spec, seed, max_epochs, patience, device)
            elif family == "gbdt":
                flat = transformed.reshape(len(transformed), -1) if is_sequence else transformed
                model = GradientBoostedTrees(**{key: spec[key] for key in ("estimators", "depth", "learning_rate", "min_leaf")}).fit(flat[train], labels[train].astype(np.float64))
                scores = model.decision_function(flat)
                val = partial_auc(scores[validation], labels[validation])
                epoch = int(spec["estimators"])
            elif family == "triplet":
                flat = transformed.reshape(len(transformed), -1)
                scores, val, epoch = _train_triplet(flat, labels, train, validation, spec, seed, max_epochs, device)
            elif family in ("attention", "tcn", "transformer"):
                scores, val, epoch = _fit_chunk_neural(transformed, labels, train, validation, family, spec, seed, max_epochs, patience, device)
            else:
                raise ValueError(f"unsupported family {family!r}")
            seed_results.append({"seed": seed, "validation_pauc_0_10": float(val), "best_epoch": int(epoch), "seconds": time.perf_counter() - started})
            seed_scores[seed] = scores
        mean_val = float(np.mean([row["validation_pauc_0_10"] for row in seed_results]))
        std_val = float(np.std([row["validation_pauc_0_10"] for row in seed_results]))
        candidate_table.append({"candidate_index": index, "config": spec, "seed_results": seed_results, "validation_pauc_mean": mean_val, "validation_pauc_std": std_val})
        selected_scores[index] = np.column_stack([seed_scores[seed] for seed in training_seeds])
        print(f"{name} candidate {index + 1}/{len(candidates)} V-pAUC={mean_val:.4f}±{std_val:.4f}", flush=True)
    selected = max(candidate_table, key=lambda row: (row["validation_pauc_mean"], -row["validation_pauc_std"], -int(row["candidate_index"])))
    selected_index = int(selected["candidate_index"])
    score_matrix = selected_scores[selected_index]
    seed_metrics: dict[str, Any] = {}
    for column, seed in enumerate(training_seeds):
        seed_metrics[str(seed)] = {
            "validation_pauc_0_10": float(selected["seed_results"][column]["validation_pauc_0_10"]),
            "best_epoch": int(selected["seed_results"][column]["best_epoch"]),
            "metrics": _bootstrap_metrics(score_matrix[:, column], labels, partitions, bootstrap_repeats, 20261000 + int(seed) % 1000),
        }
    return {
        "method": name,
        "permission_label": "strict delta-stat-only",
        "family": family,
        "feature_dim": list(values.shape[1:]),
        "selected_config": selected["config"],
        "candidate_count": len(candidates),
        "candidate_table": candidate_table,
        "seed_metrics": seed_metrics,
        "scores": score_matrix,
        "standardizer": scaler,
        "training_seeds": list(training_seeds),
    }


def _default_input(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled" / "full_delta.npz"


def _default_output(benchmark: str, epoch: int, eos: str) -> Path:
    suffix = "" if eos == "with_eos" else "_no_eos"
    return ROOT / "experiments/results/sft_runs/stat_delta" / f"{benchmark}_epoch{epoch}" / f"draft_auxiliary_distilled{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--epoch", choices=EPOCHS, type=int, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--training-seeds", default=",".join(str(v) for v in DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--chunk-slots", default="4,8,16")
    parser.add_argument("--drop-eos", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    return parser.parse_args()


def _drop_eos(data: DeltaData) -> DeltaData:
    if np.any(data.lengths <= 1):
        raise ValueError("cannot drop EOS from a one-token record")
    pieces = [data.delta[int(start) : int(end) - 1] for start, end in zip(data.offsets[:-1], data.offsets[1:])]
    lengths = data.lengths - 1
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return DeltaData(data.labels, data.record_ids, lengths, offsets, np.concatenate(pieces).astype(np.float32))


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    training_seeds = tuple(int(value) for value in args.training_seeds.split(",") if value.strip())
    input_path = args.input or _default_input(args.benchmark, args.epoch)
    input_path = input_path if input_path.is_absolute() else ROOT / input_path
    data = load_delta_data(input_path)
    if args.drop_eos:
        data = _drop_eos(data)
    partitions = split_indices(data.labels, args.split_seed)
    output_dir = args.output_dir or _default_output(args.benchmark, args.epoch, "no_eos" if args.drop_eos else "with_eos")
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    started = time.perf_counter()
    feature_cache: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    for pack in FEATURE_PACKS:
        feature_cache[pack], names = document_features(data, pack)
        feature_names[pack] = list(names)
    slots = [int(value) for value in args.chunk_slots.split(",") if value.strip()]
    for count in slots:
        values, names = chunk_features(data, "S22", count)
        feature_cache[f"chunk_S22_K{count}"] = values
        feature_names[f"chunk_S22_K{count}"] = list(names)
    np.savez_compressed(output_dir / "features.npz", labels=data.labels, record_ids=data.record_ids, lengths=data.lengths, **feature_cache)
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "input": str(input_path.resolve()),
                "records": len(data.labels),
                "tokens": int(len(data.delta)),
                "split_seed": args.split_seed,
                "feature_contract": "strict delta-stat-only; no p/q/token-id/activation/length input",
                "eos_policy": "drop final cached token" if args.drop_eos else "as cached (main)",
                "packs": {key: {"dim": list(value.shape[1:]), "names": feature_names[key]} for key, value in feature_cache.items()},
                "standardization": "D only",
                "feature_seconds": time.perf_counter() - started,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    methods: dict[str, dict[str, Any]] = {}
    # All three packs get the linear baseline.  The nonlinear families are
    # run on every pack as preregistered, not selected after inspecting T.
    family_by_pack = {
        pack: ("logistic", "mlp", "gbdt", "triplet") for pack in FEATURE_PACKS
    }
    for pack, families in family_by_pack.items():
        for family in families:
            name = f"{pack}-{family}"
            result = _run_method(name, feature_cache[pack], data.labels, partitions, family, training_seeds, device, args.max_epochs, args.patience, args.bootstrap_repeats)
            methods[name] = result
            scores = result.pop("scores")
            score_path = output_dir / f"scores_{name.lower()}.npz"
            np.savez_compressed(score_path, labels=data.labels, record_ids=data.record_ids, seeds=np.asarray(training_seeds), scores=scores)
            result["scores_path"] = str(score_path.resolve())

    if not args.skip_chunks:
        for count in slots:
            key = f"chunk_S22_K{count}"
            result = _run_method(f"{key}-uniform-logistic", feature_cache[key].mean(axis=1), data.labels, partitions, "logistic", training_seeds, device, args.max_epochs, args.patience, args.bootstrap_repeats)
            methods[result["method"]] = result
            scores = result.pop("scores")
            score_path = output_dir / f"scores_{key.lower()}_uniform_logistic.npz"
            np.savez_compressed(score_path, labels=data.labels, record_ids=data.record_ids, seeds=np.asarray(training_seeds), scores=scores)
            result["scores_path"] = str(score_path.resolve())
            for family in ("attention", "tcn", "transformer"):
                name = f"{key}-{family}"
                result = _run_method(name, feature_cache[key], data.labels, partitions, family, training_seeds, device, args.max_epochs, args.patience, args.bootstrap_repeats)
                methods[name] = result
                scores = result.pop("scores")
                score_path = output_dir / f"scores_{name.lower()}.npz"
                np.savez_compressed(score_path, labels=data.labels, record_ids=data.record_ids, seeds=np.asarray(training_seeds), scores=scores)
                result["scores_path"] = str(score_path.resolve())

    for name, result in methods.items():
        result["metrics"] = result["seed_metrics"][str(training_seeds[0])]["metrics"]
    report = {
        "protocol": {
            "benchmark": args.benchmark,
            "epoch": args.epoch,
            "role": "draft_auxiliary_distilled",
            "split_seed": args.split_seed,
            "training_seeds": list(training_seeds),
            "bootstrap_repeats": args.bootstrap_repeats,
            "selection_metric": "mean validation pAUC[0,0.1] over registered detector seeds",
            "test_usage": "T is read only after configuration selection; C supplies thresholds",
            "permission": "strict Stat-Delta; detector receives fixed delta statistics only",
            "legacy_baseline": "B1/B2 are not included; they are P/Q-stat controls",
        },
        "partitions": {name: [str(data.record_ids[index]) for index in indices] for name, indices in partitions.items()},
        "methods": methods,
    }
    (output_dir / "STAT_DELTA_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Stat-Delta membership audit",
        "",
        f"- Condition: `{args.benchmark}_epoch{args.epoch}`",
        "- Permission: strict delta-stat-only; no p/q, token ID, activation or length input.",
        "- Feature standardization: D only. Split: record-level D/V/C/T with split seed 20260824.",
        "",
        "| Method | V pAUC | T AUC | T pAUC | TPR@1% | TPR@10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in methods.items():
        metric = result["metrics"]
        test = metric["test"]
        lines.append(
            f"| {name} | {metric['validation_pauc_0_10']:.4f} | {test['auc']['point']:.4f} | {test['pauc_0_10']['point']:.4f} | {test['tpr_at_fpr']['1%']['tpr']['point']:.4f} | {test['tpr_at_fpr']['10%']['tpr']['point']:.4f} |"
        )
    (output_dir / "STAT_DELTA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "methods": len(methods)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
