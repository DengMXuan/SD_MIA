"""Pure scoring primitives used by :mod:`experiments.baseline.run`.

The functions in this module deliberately do not load a model.  This makes the
direction of every score testable and keeps the target-only restrictions in one
place: all scores are member-positive (larger means more likely member).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def _as_1d(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("a non-empty one-dimensional sequence is required")
    return result


def mean_log_likelihood(token_logp: Iterable[float]) -> float:
    """The target-only Loss baseline, oriented so members score higher."""

    return float(np.mean(_as_1d(token_logp)))


def _lowest(values: Iterable[float], k_percent: float) -> np.ndarray:
    values = _as_1d(values)
    if not 0.0 < k_percent <= 100.0:
        raise ValueError("k_percent must be in (0, 100]")
    count = max(1, int(values.size * k_percent / 100.0))
    return np.partition(values, count - 1)[:count]


def min_k_prob_score(token_logp: Iterable[float], k_percent: float = 20.0) -> float:
    """Mean log-probability of the lowest-probability tokens.

    Log-probabilities are negative, so a high-likelihood member has the
    larger (less negative) member-positive score.
    """

    return float(np.mean(_lowest(token_logp, k_percent)))


def min_k_plus_plus_score(
    token_logp: Iterable[float],
    expected_logp: Iterable[float],
    variance_logp: Iterable[float],
    k_percent: float = 20.0,
    variance_floor: float = 1e-12,
) -> float:
    """Min-K%++ with the target model's own vocabulary moments.

    The official implementation standardizes each observed token log-probability
    by the mean and standard deviation of the model's complete next-token
    distribution, then averages the lowest K%.  The result is already
    member-positive: members have the larger (less negative) standardized
    low-tail mean.
    """

    token_logp = _as_1d(token_logp)
    expected_logp = _as_1d(expected_logp)
    variance_logp = _as_1d(variance_logp)
    if not (token_logp.size == expected_logp.size == variance_logp.size):
        raise ValueError("Min-K%++ inputs must have equal lengths")
    standardized = (token_logp - expected_logp) / np.sqrt(
        np.maximum(variance_logp, variance_floor)
    )
    return float(np.mean(_lowest(standardized, k_percent)))


def recall_score(base_ll: float, conditional_ll: float) -> float:
    """ReCaLL's relative conditional log-likelihood (LL(P+x) / LL(x))."""

    if not np.isfinite(base_ll) or not np.isfinite(conditional_ll):
        return float("nan")
    if abs(base_ll) < 1e-12:
        return float(conditional_ll - base_ll)
    return float(conditional_ll / base_ll)


def icp_score(base_ll: float, conditional_ll: float) -> float:
    """ICP-MIA's optimization-gap proxy: LL(x) - LL(probe+x)."""

    return float(base_ll - conditional_ll)


def geometric_windows(
    minimum: int = 2, maximum: int = 40, count: int = 10
) -> tuple[int, ...]:
    if minimum <= 0 or maximum < minimum or count <= 0:
        raise ValueError("invalid geometric window parameters")
    # The published WBC configuration uses this rounded set (rather than the
    # exact floating-point values of numpy.geomspace).
    if (minimum, maximum, count) == (2, 40, 10):
        return (2, 3, 4, 6, 9, 13, 18, 25, 32, 40)
    if count == 1:
        return (minimum,)
    values = np.geomspace(minimum, maximum, count)
    return tuple(sorted(set(int(round(value)) for value in values)))


def window_sign_score(delta: Iterable[float], windows: Iterable[int]) -> float:
    """WBC-style window sign aggregation over a target-only token sequence."""

    delta = _as_1d(delta)
    scores: list[float] = []
    for width in windows:
        width = int(width)
        if width <= 0:
            raise ValueError("window widths must be positive")
        if delta.size < width:
            continue
        sums = np.convolve(delta, np.ones(width), mode="valid")
        scores.append(float(np.mean(sums > 0.0)))
    if not scores:
        return float(np.mean(delta > 0.0))
    return float(np.mean(scores))


def _words(text: str) -> list[str]:
    return [word for word in text.split() if word]


def rouge1_recall(candidate: str, reference: str) -> float:
    """The surface-overlap statistic used by the official SaMIA code."""

    candidate_words = _words(candidate)
    reference_words = _words(reference)
    if not candidate_words or not reference_words:
        return 0.0
    overlap = Counter(candidate_words) & Counter(reference_words)
    return float(sum(overlap.values()) / len(reference_words))


def petal_score(
    log_similarity: Iterable[float],
    calibration_logp: Iterable[float],
    slope: float | None = None,
    intercept: float | None = None,
) -> tuple[float, float, float]:
    """Target-only PETAL calibration and member-positive score.

    PETAL's published artifact fits a similarity-to-log-probability line on a
    separate surrogate.  The repository cannot load that surrogate by design,
    so this implementation fits the same line on auxiliary records scored by
    the fine-tuned target.  ``slope`` and ``intercept`` may be supplied after a
    pooled calibration pass.
    """

    x = _as_1d(log_similarity)
    y = _as_1d(calibration_logp)
    if x.size != y.size:
        raise ValueError("PETAL calibration arrays must have equal lengths")
    if slope is None or intercept is None:
        if np.ptp(x) < 1e-12:
            slope, intercept = 0.0, float(np.mean(y))
        else:
            slope, intercept = (float(value) for value in np.polyfit(x, y, 1))
    # The artifact writes an NLL-style negative score.  This package's public
    # contract is the opposite orientation (larger means member), so return
    # the estimated mean log-probability itself.
    estimate = float(np.mean(float(slope) * x + float(intercept)))
    return estimate, float(slope), float(intercept)


def sead_score(
    sampled_token_ids: np.ndarray,
    target_token_ids: Iterable[int],
    embedding_matrix: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> tuple[float, float]:
    """SEAD frequency-density score from Monte Carlo token samples.

    ``sampled_token_ids`` has shape ``[tokens, samples]``.  If an embedding
    matrix is provided, cosine similarity of the target model's own input
    embeddings is used as an optional target-only semantic proxy.  Without it
    this is the exact surrogate-free frequency estimator from the paper and
    is the default used by the runner.
    """

    samples = np.asarray(sampled_token_ids)
    targets = np.asarray(list(target_token_ids), dtype=np.int64)
    if samples.ndim != 2 or samples.shape[0] != targets.size:
        raise ValueError("SEAD samples must have shape [tokens, samples]")
    lexical = np.mean(samples == targets[:, None], axis=1)
    if embedding_matrix is None:
        density = lexical
    else:
        embeddings = np.asarray(embedding_matrix, dtype=np.float64)
        target_vectors = embeddings[targets]
        target_norm = np.linalg.norm(target_vectors, axis=1, keepdims=True)
        sampled_vectors = embeddings[samples]
        sampled_norm = np.linalg.norm(sampled_vectors, axis=2)
        cosine = np.sum(sampled_vectors * target_vectors[:, None, :], axis=2)
        cosine /= np.maximum(sampled_norm * target_norm, epsilon)
        similarity = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
        density = np.mean(similarity, axis=1)
        # Exact equality must retain full mass, as in SEAD's exact-match term.
        density = np.maximum(density, lexical)
    return float(np.mean(np.log(np.maximum(density, epsilon)))), float(np.mean(lexical))


def rank_auc(member: Iterable[float], nonmember: Iterable[float]) -> float:
    """Member-positive rank AUC without a scikit-learn dependency."""

    member = _as_1d(member)
    nonmember = _as_1d(nonmember)
    values = np.concatenate([member, nonmember])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = ranks[inverse]
    return float(
        (np.sum(ranks[: member.size]) - member.size * (member.size + 1) / 2.0)
        / (member.size * nonmember.size)
    )


def upper_tail_tpr(
    member: Iterable[float], nonmember: Iterable[float], fpr: float
) -> tuple[float, float, float]:
    """Return TPR, realized FPR, and the conservative nonmember threshold."""

    if not 0.0 < fpr < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    member = _as_1d(member)
    nonmember = _as_1d(nonmember)
    order = max(1, int(np.ceil(fpr * (nonmember.size + 1))) - 1)
    threshold = float(np.sort(nonmember)[::-1][min(order - 1, nonmember.size - 1)])
    return (
        float(np.mean(member > threshold)),
        float(np.mean(nonmember > threshold)),
        threshold,
    )
