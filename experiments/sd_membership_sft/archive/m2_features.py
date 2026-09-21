"""M2 feature construction: probability-evidence weights and doc pooling.

Implements the preregistered preprocessing of
``research/M2概率证据引导激活聚合_Qwen3已微调模型实验方案_2026-09-09.md``:

- token weights ``1 / (2 |D_m| n_i)`` so documents and classes are balanced;
- ``s_delta = max(1e-3, E_D|delta|)`` with delta scaled but never centered;
- ``b± = min(max(±u, 0), c) / c`` with fixed ``c = 4`` and
  ``alpha = exp(min(delta, 0))``;
- the 4-dim evidence vector ``e = [clip(u, -4, 4)/4, b+, b-, alpha]``;
- per-dimension token standardization of Q/H/log p fitted on D, clipped to
  [-8, 8], with dims of std < 1e-6 zeroed;
- document pooling ``U(z)=[mean, std]``, ``W(z)=[H+, H-, H+-H-]`` and the
  6-dim evidence-quality vector ``M``;
- the shared 77-dim probability branch ``B_Q = [F41, U(Q), W(Q), M]``.

All accumulations use float64.  Zero-weight directions return exact zero
vectors (no 0/0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.sd_membership_sft.analysis.m1_features import (aggregate_matrix)

EVIDENCE_CLIP = 4.0
STD_CLIP = 8.0
STD_FLOOR = 1e-6
DELTA_FLOOR = 1e-3

TOKEN_CHUNK = 1_000_000


@dataclass
class M2Features:
    """Doc-level matrices and flat token-level arrays for one condition."""

    labels: np.ndarray
    record_ids: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    f19: np.ndarray
    f19_names: tuple[str, ...]
    a22: np.ndarray
    f41: np.ndarray
    u_q: np.ndarray
    w_q: np.ndarray
    quality: np.ndarray
    bq: np.ndarray
    u_h: np.ndarray
    w_h: np.ndarray
    log_n: np.ndarray
    delta_scale: float
    token: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def doc_matrix(self, name: str) -> np.ndarray:
        """Assemble the registered doc-level input matrices by method name."""

        if name == "B1":
            return self.f19
        if name == "B2":
            return self.f41
        if name == "BQ":
            return self.bq
        if name == "BQ_noM":
            return np.concatenate((self.f41, self.u_q, self.w_q), axis=1)
        if name == "H_only":
            return np.concatenate((self.u_h, self.log_n), axis=1)
        if name == "Direct":
            return np.concatenate((self.bq, self.u_h), axis=1)
        if name == "M2_F":
            return np.concatenate((self.bq, self.u_h, self.w_h), axis=1)
        if name == "M2_F_nodiff":
            plus, minus, _ = _split_w(self.w_h, self.u_h.shape[1] // 2)
            return np.concatenate((self.bq, self.u_h, plus, minus), axis=1)
        if name == "M2_F_plus":
            plus, _minus, _ = _split_w(self.w_h, self.u_h.shape[1] // 2)
            return np.concatenate((self.bq, self.u_h, plus), axis=1)
        if name == "M2_F_minus":
            _plus, minus, _ = _split_w(self.w_h, self.u_h.shape[1] // 2)
            return np.concatenate((self.bq, self.u_h, minus), axis=1)
        if name == "M2_F_noM":
            return np.concatenate(
                (self.f41, self.u_q, self.w_q, self.u_h, self.w_h), axis=1
            )
        raise KeyError(f"unknown doc matrix {name!r}")


def _split_w(w: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus, minus, diff = w[:, :dim], w[:, dim : 2 * dim], w[:, 2 * dim :]
    return plus, minus, diff


def doc_token_weights(
    labels: np.ndarray, lengths: np.ndarray, doc_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-doc normalization constant and per-doc weight sums.

    Each token of D document ``i`` in class ``m`` carries weight
    ``1 / (2 |D_m| n_i)``; the weights over all D tokens sum to one.
    """

    labels = np.asarray(labels, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    doc_indices = np.asarray(doc_indices, dtype=np.int64)
    class_counts = {
        label: int(np.sum(labels[doc_indices] == label)) for label in (0, 1)
    }
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise RuntimeError("D must contain both classes for token weighting")
    per_doc = np.empty(len(doc_indices), dtype=np.float64)
    for row, doc in enumerate(doc_indices):
        per_doc[row] = 1.0 / (2.0 * class_counts[int(labels[doc])] * lengths[doc])
    return per_doc, lengths[doc_indices].astype(np.int64)


def weighted_mean_std(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    total = float(weights.sum())
    mean = (weights[:, None] * values).sum(axis=0) / total
    variance = (weights[:, None] * np.square(values - mean)).sum(axis=0) / total
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _standardize_tokens(
    values: np.ndarray,
    token_weights: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit weighted mean/std on ``fit_mask`` tokens, transform all tokens."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.zeros(len(values), dtype=np.float64)
    weights[fit_mask] = token_weights
    mean, std = weighted_mean_std(values[fit_mask], token_weights)
    zero_dims = std < STD_FLOOR
    safe_std = np.where(zero_dims, 1.0, std)
    standardized = (values - mean) / safe_std
    standardized[:, zero_dims] = 0.0
    clipped = np.clip(standardized, -STD_CLIP, STD_CLIP)
    touch_fraction = float(np.mean(np.any(np.abs(np.asarray(standardized)) > STD_CLIP, axis=1)))
    diagnostics = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "zero_dims": zero_dims.tolist(),
        "clip_touch_fraction": touch_fraction,
    }
    return clipped, diagnostics


def build_m2_features(data: Any, partitions: Any) -> M2Features:
    """Build every registered M2 input from an :class:`M1Data` cache."""

    labels = np.asarray(data.labels, dtype=np.int64)
    lengths = np.asarray(data.lengths, dtype=np.int64)
    offsets = np.asarray(data.offsets, dtype=np.int64)
    detector_fit = np.asarray(partitions["detector_fit"], dtype=np.int64)
    per_doc_weight, doc_lengths = doc_token_weights(labels, lengths, detector_fit)

    token_doc = np.repeat(np.arange(len(labels)), lengths)
    fit_docs = np.zeros(len(labels), dtype=bool)
    fit_docs[detector_fit] = True
    fit_mask = fit_docs[token_doc]
    token_weights = np.zeros(int(lengths.sum()), dtype=np.float64)
    token_weights[fit_mask] = np.repeat(per_doc_weight, doc_lengths)

    delta = np.asarray(data.delta, dtype=np.float64)
    delta_scale = max(DELTA_FLOOR, float(np.sum(token_weights * np.abs(delta))))
    u = delta / delta_scale
    b_plus = np.minimum(np.maximum(u, 0.0), EVIDENCE_CLIP) / EVIDENCE_CLIP
    b_minus = np.minimum(np.maximum(-u, 0.0), EVIDENCE_CLIP) / EVIDENCE_CLIP
    alpha = np.exp(np.minimum(delta, 0.0))
    evidence = np.stack(
        (np.clip(u, -EVIDENCE_CLIP, EVIDENCE_CLIP) / EVIDENCE_CLIP, b_plus, b_minus, alpha),
        axis=1,
    )

    logp = np.asarray(data.target_logp, dtype=np.float64)
    q_values = np.asarray(data.q, dtype=np.float64)
    h_values = np.asarray(data.h, dtype=np.float64)

    q_std, q_diag = _standardize_tokens(q_values, token_weights[fit_mask], fit_mask)
    h_std, h_diag = _standardize_tokens(h_values, token_weights[fit_mask], fit_mask)
    logp2d = logp[:, None]
    l_std2d, l_diag = _standardize_tokens(logp2d, token_weights[fit_mask], fit_mask)
    l_std = l_std2d[:, 0]

    a22 = aggregate_matrix(delta, lengths)
    f19 = np.asarray(data.f19, dtype=np.float64)
    f41 = np.concatenate((f19, a22), axis=1)

    n_docs = len(labels)
    u_q = np.empty((n_docs, 2 * q_std.shape[1]), dtype=np.float64)
    u_h = np.empty((n_docs, 2 * h_std.shape[1]), dtype=np.float64)
    w_q = np.empty((n_docs, 3 * q_std.shape[1]), dtype=np.float64)
    w_h = np.empty((n_docs, 3 * h_std.shape[1]), dtype=np.float64)
    quality = np.empty((n_docs, 6), dtype=np.float64)
    weight_sums = np.empty((n_docs, 2), dtype=np.float64)
    log_n = np.log(lengths.astype(np.float64))[:, None]

    for index in range(n_docs):
        start, end = int(offsets[index]), int(offsets[index + 1])
        n = end - start
        plus = b_plus[start:end]
        minus = b_minus[start:end]
        q_doc = q_std[start:end]
        h_doc = h_std[start:end]
        u_q[index] = _u_pool(q_doc)
        u_h[index] = _u_pool(h_doc)
        w_q[index] = _w_pool(q_doc, plus, minus)
        w_h[index] = _w_pool(h_doc, plus, minus)
        quality[index] = _quality(plus, minus, n)
        weight_sums[index] = (float(plus.sum()), float(minus.sum()))

    bq = np.concatenate((f41, u_q, w_q, quality), axis=1)
    features = M2Features(
        labels=labels,
        record_ids=np.asarray(data.record_ids),
        lengths=lengths,
        offsets=offsets,
        f19=f19,
        f19_names=tuple(data.f19_names),
        a22=a22,
        f41=f41,
        u_q=u_q,
        w_q=w_q,
        quality=quality,
        bq=bq,
        u_h=u_h,
        w_h=w_h,
        log_n=log_n,
        delta_scale=delta_scale,
        token={
            "q_std": q_std.astype(np.float32),
            "h_std": h_std.astype(np.float32),
            "l_std": l_std.astype(np.float32),
            "e": evidence.astype(np.float32),
            "b_plus": b_plus.astype(np.float32),
            "b_minus": b_minus.astype(np.float32),
        },
        diagnostics={
            "delta_scale": delta_scale,
            "q_standardization": q_diag,
            "h_standardization": h_diag,
            "logp_standardization": l_diag,
            "weight_sums_plus_mean": float(weight_sums[:, 0].mean()),
            "weight_sums_minus_mean": float(weight_sums[:, 1].mean()),
            "zero_plus_weight_docs": int(np.sum(weight_sums[:, 0] == 0.0)),
            "zero_minus_weight_docs": int(np.sum(weight_sums[:, 1] == 0.0)),
        },
    )
    return features


def _u_pool(z: np.ndarray) -> np.ndarray:
    return np.concatenate((z.mean(axis=0), z.std(axis=0, ddof=0)))


def _w_pool(z: np.ndarray, plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    sum_plus = float(plus.sum())
    sum_minus = float(minus.sum())
    if sum_plus > 0.0:
        h_plus = (plus[:, None] * z).sum(axis=0) / sum_plus
    else:
        h_plus = np.zeros(z.shape[1], dtype=np.float64)
    if sum_minus > 0.0:
        h_minus = (minus[:, None] * z).sum(axis=0) / sum_minus
    else:
        h_minus = np.zeros(z.shape[1], dtype=np.float64)
    return np.concatenate((h_plus, h_minus, h_plus - h_minus))


def _quality(plus: np.ndarray, minus: np.ndarray, n: int) -> np.ndarray:
    sum_plus = float(plus.sum())
    sum_minus = float(minus.sum())
    m_plus, m_minus = sum_plus / n, sum_minus / n
    f_plus = float(np.mean(plus > 0.0))
    f_minus = float(np.mean(minus > 0.0))
    denom_plus = float(np.square(plus).sum())
    denom_minus = float(np.square(minus).sum())
    ess_plus = (sum_plus * sum_plus / denom_plus) if denom_plus > 0.0 else 0.0
    ess_minus = (sum_minus * sum_minus / denom_minus) if denom_minus > 0.0 else 0.0
    return np.array([m_plus, m_minus, f_plus, f_minus, ess_plus / n, ess_minus / n])
