"""Offline membership scorers computed from raw SD-protocol transcripts.

P0 statistical package from
``research/端云SD可借鉴思想与实验改进_文献综合.md`` §2. Every scorer consumes
per-position acceptance counts ``K_t`` out of ``R`` verifier repeats for the
candidate token ``x_t`` proposed with draft confidence ``q_t``, i.e. the L2/L3
observation ``K_t ~ Binomial(R, min(1, p_t/q_t))``. Null acceptance rates are
estimated per draft-confidence bin from audit-calibration tokens with labels
unused, matching the repo's label-free nuisance-fit convention
(``draver_activation`` verifier surprise residual).

Conventions follow ``audit.transcript_tomography``: the honest acceptance
probability uses ``q`` clipped to ``[1e-7, 0.95]`` and observed rates are
Jeffreys-smoothed ``(K + 0.5) / (R + 1)``. Reduced verifier budgets are
evaluated by slicing the stored per-repeat accept bits (first ``R'`` repeats),
not by thinning, so a budget sweep reuses one replay dump.

Scorers (translation -> literature source):

- ``acceptance_lira``: one-sided binomial likelihood ratio against the q-bin
  null; saturated tokens contribute ``R * log(1/a)`` instead of being dropped
  (Hayes et al., NeurIPS 2025 -> acceptance channel).
- ``min_a_percent``: mean log Jeffreys acceptance over the lowest-k% positions
  (Shi et al., ICLR 2024 Min-K% -> acceptance channel).
- ``saturation_rate``: share of fully accepted probed positions; the censoring
  complement of Min-A%.
- ``wbc_sd``: geometric-window sign ensemble over the null-calibrated
  acceptance residual (Chen et al., USENIX Security 2026 WBC -> acceptance
  channel).
- ``qbin_conditional_z``: per-record acceptance averaged inside draft-confidence
  bins, standardized against calibration (CAMIA-style stratified statistic).
- ``surp_sd``: double-condition score on low-draft-entropy positions with low
  observed acceptance (Zhang & Wu SURP -> acceptance channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Q_CLIP_LOW = 1e-7
Q_CLIP_HIGH = 0.95


def honest_alpha(
    target_logp: np.ndarray,
    draft_logp: np.ndarray,
    q_clip_low: float = Q_CLIP_LOW,
    q_clip_high: float = Q_CLIP_HIGH,
) -> np.ndarray:
    """Acceptance probability min(1, p/q) with the repo's q clipping."""
    p = np.exp(np.nan_to_num(target_logp, nan=-np.inf))
    q0 = np.exp(np.nan_to_num(draft_logp, nan=-np.inf))
    q = np.clip(q0, q_clip_low, q_clip_high)
    alpha = np.minimum(1.0, p / q)
    return np.where(np.isfinite(target_logp) & np.isfinite(draft_logp), alpha, np.nan)


def jeffreys_rates(counts: np.ndarray, repeats: int) -> np.ndarray:
    return (counts + 0.5) / (repeats + 1.0)


@dataclass
class NullRates:
    """Per-draft-confidence-bin null acceptance rates from calibration."""

    edges: np.ndarray
    rates: np.ndarray

    def bin_index(self, q_prob: np.ndarray) -> np.ndarray:
        inner = np.clip(np.searchsorted(self.edges, q_prob, side="right") - 1, 0, len(self.rates) - 1)
        return np.where(np.isfinite(q_prob), inner, 0)

    def rate_at(self, q_prob: np.ndarray) -> np.ndarray:
        return self.rates[self.bin_index(q_prob)]


def fit_null_rates(
    q_prob: np.ndarray,
    counts: np.ndarray,
    repeats: int,
    valid: np.ndarray,
    n_bins: int = 12,
) -> NullRates:
    """Pooled Jeffreys null rate per q-probability quantile bin.

    ``q_prob``, ``counts`` and ``valid`` are [records, positions]; only
    calibration records must be passed (labels unused, matching the label-free
    nuisance-fit convention).
    """
    flat_q = q_prob[valid]
    flat_c = counts[valid]
    if flat_q.size == 0:
        raise ValueError("no valid calibration tokens for null fitting")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(flat_q, quantiles))
    if edges.size < 2:
        edges = np.array([0.0, 1.0], dtype=np.float64)
    bin_id = np.clip(
        np.searchsorted(edges, flat_q, side="right") - 1, 0, edges.size - 2
    )
    rates = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        mask = bin_id == b
        n = int(mask.sum())
        if n == 0:
            continue
        rates[b] = (flat_c[mask].sum() + 0.5) / (repeats * n + 1.0)
    # Empty bins inherit the nearest populated rate so lookups stay finite.
    filled = np.where(np.isnan(rates))[0]
    for b in filled:
        neighbours = [i for i in range(len(rates)) if not np.isnan(rates[i])]
        if neighbours:
            rates[b] = rates[min(neighbours, key=lambda i: abs(i - b))]
        else:
            rates[b] = 0.5
    return NullRates(edges=edges, rates=rates)


def gather_selection(
    values: np.ndarray, positions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slice [records, all_positions] arrays down to a selection.

    ``positions`` is [records, k_cols]; ``valid`` is a *column mask* aligned
    with ``positions``' columns (True = this column is a real selected
    position for that record), NOT a mask over token positions. Positions in
    masked-out columns are mapped to 0 so every scorer can rely on aligned
    shapes. Returns the gathered values and the same column mask.
    """
    safe = np.where(valid, positions, 0)
    gathered = np.take_along_axis(values, safe, axis=1)
    return gathered, valid


def acceptance_lira(
    q_prob_sel: np.ndarray,
    counts_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    null: NullRates,
    reduction: str = "mean",
) -> np.ndarray:
    """One-sided binomial LLR of the observed counts against the q-bin null.

    ``b_hat = max(a, K/R)`` is the constrained MLE under the one-sided
    alternative (SFT exposure only raises p, hence acceptance); when
    ``K == R`` the statistic reduces to ``R * log(1/a)`` so saturated tokens
    keep contributing evidence. Positions in saturated null bins (a >= 1) or
    invalid positions contribute zero.
    """
    a = null.rate_at(q_prob_sel)
    alpha_mle = counts_sel / repeats
    b_hat = np.maximum(a, alpha_mle)
    # Guard 0 * log(0): when the token is saturated (K == R, b_hat == 1) the
    # reject term vanishes instead of being 0 * (-inf) = NaN; positions with a
    # degenerate null (a >= 1) are masked out below.
    with np.errstate(divide="ignore", invalid="ignore"):
        accept_term = counts_sel * np.log(b_hat / a)
        log_reject = np.log((1.0 - b_hat) / (1.0 - a))
        reject_term = (repeats - counts_sel) * log_reject
    accept_term = np.where(counts_sel > 0, accept_term, 0.0)
    reject_term = np.where(counts_sel < repeats, reject_term, 0.0)
    llr = np.nan_to_num(accept_term, nan=0.0) + np.nan_to_num(reject_term, nan=0.0, posinf=0.0)
    keep = valid_sel & (a < 1.0) & (a > 0.0)
    llr = np.where(keep, llr, 0.0)
    counts_keep = keep.sum(axis=1)
    if reduction == "sum":
        return llr.sum(axis=1)
    safe_total = np.maximum(counts_keep, 1)
    return llr.sum(axis=1) / safe_total


def min_a_percent(
    counts_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    fraction: float = 0.2,
) -> np.ndarray:
    """Mean log Jeffreys acceptance over the lowest ``fraction`` positions."""
    alpha_hat = jeffreys_rates(counts_sel, repeats)
    log_alpha = np.log(alpha_hat)
    rows = log_alpha.shape[0]
    scores = np.full(rows, np.nan)
    for row in range(rows):
        mask = valid_sel[row]
        n = int(mask.sum())
        if n == 0:
            continue
        values = log_alpha[row, mask]
        k = max(1, min(n, int(math.ceil(n * fraction))))
        lowest = np.partition(values, k - 1)[:k]
        scores[row] = float(lowest.mean())
    return scores


def saturation_rate(
    counts_sel: np.ndarray, repeats: int, valid_sel: np.ndarray
) -> np.ndarray:
    keep = valid_sel.sum(axis=1) > 0
    saturated = ((counts_sel == repeats) & valid_sel).sum(axis=1) / np.maximum(
        valid_sel.sum(axis=1), 1
    )
    return np.where(keep, saturated, np.nan)


def wbc_sd(
    positions_sel: np.ndarray,
    q_prob_sel: np.ndarray,
    counts_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    null: NullRates,
    w_min: int = 2,
    w_max: int = 40,
    n_window_sizes: int = 10,
) -> np.ndarray:
    """WBC-style geometric-window sign ensemble on the acceptance residual.

    Residuals ``d_t = alpha_hat_t - a_bin(t)`` are placed in document order
    (positions sorted ascending); for each window size the statistic is the
    share of sliding windows with positive residual sum, averaged over sizes.
    """
    alpha_hat = jeffreys_rates(counts_sel, repeats)
    residual = alpha_hat - null.rate_at(q_prob_sel)
    residual = np.where(valid_sel, residual, 0.0)
    rows, width = residual.shape
    span = max(2, w_max)
    sizes = sorted(
        {
            int(round(w_min * (span / w_min) ** (k / max(1, n_window_sizes - 1))))
            for k in range(n_window_sizes)
        }
    )
    scores = np.full(rows, np.nan)
    for row in range(rows):
        order = np.argsort(np.where(valid_sel[row], positions_sel[row], np.iinfo(np.int64).max))
        n = int(valid_sel[row].sum())
        if n == 0:
            continue
        values = residual[row, order][:n]
        cumsum = np.concatenate([[0.0], np.cumsum(values)])
        statistics = []
        for w in sizes:
            if w > n:
                continue
            window_sums = cumsum[w:] - cumsum[:-w]
            statistics.append(float(np.mean(window_sums > 0.0)))
        if statistics:
            scores[row] = float(np.mean(statistics))
    return scores


def qbin_conditional_means(
    q_prob_sel: np.ndarray,
    counts_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-record mean Jeffreys acceptance inside shared q-probability bins.

    Returns ``(means, mask)`` with shape [records, n_bins]; mask marks bins the
    record actually probed. Standardization against calibration happens in the
    evaluation harness so calibration rows never touch test rows.
    """
    alpha_hat = jeffreys_rates(counts_sel, repeats)
    n_bins = max(1, edges.size - 1)
    bin_id = np.clip(np.searchsorted(edges, q_prob_sel, side="right") - 1, 0, n_bins - 1)
    rows = alpha_hat.shape[0]
    means = np.full((rows, n_bins), np.nan)
    mask = np.zeros((rows, n_bins), dtype=bool)
    for row in range(rows):
        for b in range(n_bins):
            cell = valid_sel[row] & (bin_id[row] == b)
            if cell.any():
                means[row, b] = float(alpha_hat[row, cell].mean())
                mask[row, b] = True
    return means, mask


def standardize_qbin(
    means: np.ndarray, mask: np.ndarray, calibration_rows: np.ndarray
) -> np.ndarray:
    """Z-score each bin against calibration rows, then average available bins.

    Bins whose calibration mean acceptance is saturated (>= 0.95) or degenerate
    (<= 0.02) are dropped: their z-noise dominates the average without carrying
    membership signal. Sigma is floored at 0.05 so a near-constant calibration
    bin cannot amplify noise arbitrarily.
    """
    with np.errstate(invalid="ignore"):
        cal = np.where(mask[calibration_rows], means[calibration_rows], np.nan)
        mu = np.nanmean(cal, axis=0)
        sigma = np.nanstd(cal, axis=0, ddof=1)
    informative = np.isfinite(mu) & (mu < 0.95) & (mu > 0.02)
    sigma = np.where((~np.isfinite(sigma)) | (sigma < 0.05), 0.05, sigma)
    z = (means - mu) / sigma
    usable = mask & informative[None, :] & np.isfinite(z)
    z = np.where(usable, z, 0.0)
    denom = np.maximum(usable.sum(axis=1), 1)
    return z.sum(axis=1) / denom


def surp_sd(
    entropy_sel: np.ndarray,
    counts_sel: np.ndarray,
    repeats: int,
    valid_sel: np.ndarray,
    entropy_quantile: float = 0.4,
    alpha_upper: float = 0.9,
) -> np.ndarray:
    """SURP-style double condition: low draft entropy AND low acceptance.

    A position qualifies when its draft entropy is within the record's lowest
    ``entropy_quantile`` among probed positions and its Jeffreys acceptance is
    below ``alpha_upper``; the score is the mean log acceptance over qualifying
    positions (0.0 when none qualify, i.e. no evidence either way).
    """
    alpha_hat = jeffreys_rates(counts_sel, repeats)
    log_alpha = np.log(alpha_hat)
    rows = log_alpha.shape[0]
    scores = np.full(rows, np.nan)
    for row in range(rows):
        mask = valid_sel[row]
        n = int(mask.sum())
        if n == 0:
            continue
        ent = entropy_sel[row, mask]
        threshold = np.quantile(ent, entropy_quantile)
        low_entropy = ent <= threshold
        accepted_low = low_entropy & (alpha_hat[row, mask] < alpha_upper)
        if accepted_low.any():
            scores[row] = float(log_alpha[row, mask][accepted_low].mean())
        else:
            scores[row] = 0.0
    return scores


def mean_acceptance(
    counts_sel: np.ndarray, repeats: int, valid_sel: np.ndarray
) -> np.ndarray:
    """Verifier mean acceptance: the repo's headline transcript statistic."""
    alpha_hat = jeffreys_rates(counts_sel, repeats)
    keep = valid_sel.sum(axis=1) > 0
    return np.where(keep, np.nanmean(np.where(valid_sel, alpha_hat, np.nan), axis=1), np.nan)
