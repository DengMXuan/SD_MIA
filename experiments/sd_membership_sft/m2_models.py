"""M2 model zoo: class-balanced logistic, doc MLP and the gated token family.

Registered training budget (M2 plan section 8): logistic L2 grid with
unpenalized bias; small networks use AdamW lr 1e-3, weight decay
{1e-3, 1e-2}, dropout {0, 0.1}, at most 100 document epochs with early
stopping on validation pAUC[0, 0.05] (patience 10), gradient-norm clip 1.0
and class-balanced effective batches of 16 documents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from .m1_fit import partial_auc

LOGISTIC_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0)
NET_WEIGHT_DECAYS = (1e-3, 1e-2)
NET_DROPOUTS = (0.0, 0.1)
DOC_BATCH = 16
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
GRAD_CLIP = 1.0
LONG_DOC_TOKENS = 1024
MICROBATCH_DOCS = 4


def balanced_doc_weights(labels: np.ndarray, doc_indices: np.ndarray) -> np.ndarray:
    """Weight each D document ``1 / (2 |D_m|)`` so class means weight one half."""

    labels = np.asarray(labels, dtype=np.int64)
    doc_indices = np.asarray(doc_indices, dtype=np.int64)
    counts = {label: int(np.sum(labels[doc_indices] == label)) for label in (0, 1)}
    weights = np.empty(len(doc_indices), dtype=np.float64)
    for row, doc in enumerate(doc_indices):
        weights[row] = 1.0 / (2.0 * counts[int(labels[doc])])
    return weights


def fit_doc_standardizer(
    values: np.ndarray, labels: np.ndarray, doc_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the D-only doc-level mean/std with class-balanced weights."""

    matrix = np.asarray(values, dtype=np.float64)
    weights = balanced_doc_weights(labels, doc_indices)
    subset = matrix[doc_indices]
    mean = (weights[:, None] * subset).sum(axis=0)
    variance = (weights[:, None] * np.square(subset - mean)).sum(axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


@dataclass
class LogisticModel:
    weight: np.ndarray
    bias: float

    def scores(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) @ self.weight + self.bias

    def parameter_count(self) -> int:
        return int(self.weight.size) + 1


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def fit_class_balanced_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    doc_indices: np.ndarray,
    l2: float,
) -> LogisticModel:
    """Class-balanced BCE with L2 on the weight vector only (bias free)."""

    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    doc_indices = np.asarray(doc_indices, dtype=np.int64)
    x = matrix[doc_indices]
    y = labels[doc_indices].astype(np.float64)
    weights = balanced_doc_weights(labels, doc_indices)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        score = x @ parameters[:-1] + parameters[-1]
        probability = _sigmoid(score)
        gradient_score = weights * (probability - y)
        loss = float(
            np.sum(weights * np.logaddexp(0.0, score))
            - np.sum(weights * y * score)
            + l2 * float(parameters[:-1] @ parameters[:-1])
        )
        gradient_weight = x.T @ gradient_score + 2.0 * l2 * parameters[:-1]
        gradient_bias = float(gradient_score.sum())
        return loss, np.concatenate((gradient_weight, [gradient_bias]))

    result = minimize(
        objective,
        np.zeros(matrix.shape[1] + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    return LogisticModel(weight=result.x[:-1].copy(), bias=float(result.x[-1]))


def fit_logistic_grid(
    standardized: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    l2_grid: tuple[float, ...] = LOGISTIC_L2_GRID,
) -> tuple[LogisticModel, list[dict[str, Any]]]:
    """Fit the L2 grid on already-standardized features, select on V pAUC."""

    labels = np.asarray(labels, dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    fitted: list[LogisticModel] = []
    for l2 in l2_grid:
        model = fit_class_balanced_logistic(standardized, labels, train_indices, l2)
        validation_scores = model.scores(standardized[validation_indices])
        pauc = partial_auc(validation_scores, labels[validation_indices])
        candidates.append(
            {
                "l2": float(l2),
                "validation_pauc_0_05": float(pauc),
                "parameter_count": model.parameter_count(),
            }
        )
        fitted.append(model)
    best = max(
        range(len(candidates)),
        key=lambda i: (candidates[i]["validation_pauc_0_05"], candidates[i]["l2"]),
    )
    return fitted[best], candidates


class DocMLP(nn.Module):
    """F41 + 64-wide one-hidden-layer detector (M1 detector spec)."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class TokenGatedNet(nn.Module):
    """M2-G token network and readout.

    ``use_token_h`` selects the 131-dim input (z, b+z, b- z present) or the
    11-dim probability-only input; the readout consumes
    ``[B_Q, G, U_v]`` where ``G`` is attention-pooled and ``U_v`` the uniform
    mean.  ``uniform_attention`` fixes ``a = 1/n`` (M2-U).
    """

    def __init__(
        self,
        bq_dim: int,
        hidden: int = 64,
        use_token_h: bool = True,
        uniform_attention: bool = False,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_token_h = use_token_h
        self.uniform_attention = uniform_attention
        self.temperature = float(temperature)
        token_in = 6 + 1 + 4 + (3 * 40 if use_token_h else 0)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden, 1)
        self.readout = nn.Sequential(
            nn.Linear(bq_dim + 2 * hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        token_x: torch.Tensor,
        mask: torch.Tensor,
        bq: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask.to(token_x.dtype)
        hidden = self.token_mlp(token_x)
        if self.uniform_attention:
            attention = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            logits = self.attention(hidden).squeeze(-1) / self.temperature
            logits = logits.masked_fill(~mask, float("-inf"))
            attention = torch.softmax(logits, dim=1)
        pooled = (attention.unsqueeze(-1) * hidden).sum(dim=1)
        uniform = (hidden * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return self.readout(torch.cat((bq, pooled, uniform), dim=1)).squeeze(-1)

    def attention_stats(self, token_x: torch.Tensor, mask: torch.Tensor) -> dict[str, np.ndarray]:
        with torch.inference_mode():
            hidden = self.token_mlp(token_x)
            logits = self.attention(hidden).squeeze(-1) / self.temperature
            logits = logits.masked_fill(~mask, float("-inf"))
            attention = torch.softmax(logits, dim=1)
        valid_counts = mask.sum(dim=1).clamp_min(1.0).to(torch.float64)
        attention = attention.to(torch.float64)
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = entropy / valid_counts.log()
        effective = attention.exp().sum(dim=1)
        return {
            "mean_normalized_entropy": normalized_entropy.cpu().numpy(),
            "mean_max_attention": attention.max(dim=1).values.cpu().numpy(),
            "effective_tokens": effective.cpu().numpy(),
            "valid_tokens": valid_counts.cpu().numpy(),
        }

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())


@dataclass
class TokenBatcher:
    """Assemble padded per-token input tensors for one condition."""

    lengths: np.ndarray
    offsets: np.ndarray
    q_std: np.ndarray
    h_std: np.ndarray
    l_std: np.ndarray
    evidence: np.ndarray
    b_plus: np.ndarray
    b_minus: np.ndarray
    use_token_h: bool = True
    zero_h: bool = False

    def encode(self, doc_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Padded ``[docs, max_len, dim]`` inputs and validity mask."""

        doc_indices = np.asarray(doc_indices, dtype=np.int64)
        size = len(doc_indices)
        lengths = self.lengths[doc_indices]
        max_len = int(lengths.max())
        token_dim = 6 + 1 + 4 + (3 * 40 if self.use_token_h else 0)
        inputs = np.zeros((size, max_len, token_dim), dtype=np.float32)
        mask = np.zeros((size, max_len), dtype=bool)
        for row, doc in enumerate(doc_indices):
            start, end = int(self.offsets[doc]), int(self.offsets[doc + 1])
            n = end - start
            parts = [
                self.q_std[start:end],
                self.l_std[start:end, None],
                self.evidence[start:end],
            ]
            if self.use_token_h:
                if self.zero_h:
                    h = np.zeros((n, self.h_std.shape[1]), dtype=np.float32)
                else:
                    h = self.h_std[start:end]
                parts.extend((h, self.b_plus[start:end, None] * h, self.b_minus[start:end, None] * h))
            inputs[row, :n] = np.concatenate(parts, axis=1)
            mask[row, :n] = True
        return inputs, mask


def _doc_chunks(members: np.ndarray, nonmembers: np.ndarray, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    member_order = rng.permutation(members)
    nonmember_order = rng.permutation(nonmembers)
    chunks = []
    member_chunks = [member_order[i : i + DOC_BATCH // 2] for i in range(0, len(member_order), DOC_BATCH // 2)]
    nonmember_chunks = [
        nonmember_order[i : i + DOC_BATCH // 2]
        for i in range(0, len(nonmember_order), DOC_BATCH // 2)
    ]
    steps = max(len(member_chunks), len(nonmember_chunks))
    for step in range(steps):
        member_chunk = member_chunks[step % len(member_chunks)] if member_chunks else np.empty(0, dtype=np.int64)
        nonmember_chunk = (
            nonmember_chunks[step % len(nonmember_chunks)] if nonmember_chunks else np.empty(0, dtype=np.int64)
        )
        chunks.append((member_chunk, nonmember_chunk))
    return chunks


def train_small_net(
    model: nn.Module,
    loss_batch: Callable[[np.ndarray, np.ndarray], torch.Tensor],
    score_fn: Callable[[nn.Module, np.ndarray], np.ndarray],
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    validation_indices: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    seed: int,
    weight_decay: float,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    """Shared AdamW/early-stopping loop for doc-level and token networks."""

    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    members = train_indices[train_labels[train_indices] == 1]
    nonmembers = train_indices[train_labels[train_indices] == 0]
    rng = np.random.default_rng(seed)
    best_pauc = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(max_epochs):
        model.train()
        for member_chunk, nonmember_chunk in _doc_chunks(members, nonmembers, rng):
            optimizer.zero_grad(set_to_none=True)
            # loss_batch owns forward + backward (so it can microbatch internally).
            loss_batch(model, member_chunk, nonmember_chunk)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        model.eval()
        validation_scores = score_fn(model, validation_indices)
        pauc = partial_auc(validation_scores, validation_labels)
        history.append({"epoch": epoch, "validation_pauc_0_05": float(pauc)})
        if pauc > best_pauc + 1e-12:
            best_pauc = pauc
            best_epoch = epoch
            bad = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "best_validation_pauc_0_05": float(best_pauc),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "history": history,
    }


def search_small_net(
    build_model: Callable[[dict[str, Any]], nn.Module],
    loss_batch: Callable[[np.ndarray, np.ndarray], torch.Tensor],
    score_fn: Callable[[nn.Module, np.ndarray], np.ndarray],
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    validation_indices: np.ndarray,
    validation_labels: np.ndarray,
    device: torch.device,
    seed: int,
    config_grid: list[dict[str, Any]],
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]]]:
    """Train the 4-config grid, select on V pAUC (ties: stronger wd, less dropout)."""

    candidates: list[dict[str, Any]] = []
    fitted: list[tuple[nn.Module, dict[str, Any]]] = []
    for config in config_grid:
        model = build_model(config)
        summary = train_small_net(
            model,
            loss_batch,
            score_fn,
            train_indices,
            train_labels,
            validation_indices,
            validation_labels,
            device,
            seed,
            weight_decay=config["weight_decay"],
        )
        candidates.append(
            {
                **config,
                "validation_pauc_0_05": summary["best_validation_pauc_0_05"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "parameter_count": sum(int(p.numel()) for p in model.parameters()),
            }
        )
        fitted.append((model, summary))
    best = max(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["validation_pauc_0_05"],
            candidates[i]["weight_decay"],
            -candidates[i]["dropout"],
        ),
    )
    model, summary = fitted[best]
    return model, {**candidates[best], "training": summary}, candidates
