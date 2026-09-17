"""Ranking and nonmember tail calibration shared by current and archived audits."""
from __future__ import annotations
from typing import Any
import numpy as np

RATES = (0.01, 0.05, 0.10)

def conformal_tail_pvalues(
    scores: np.ndarray, calibration_nonmember: np.ndarray
) -> np.ndarray:
    """Return conservative upper-tail p-values with inclusive tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    calibration_nonmember = np.asarray(calibration_nonmember, dtype=np.float64)
    if calibration_nonmember.size == 0:
        raise ValueError("calibration_nonmember must not be empty")
    # With sorted calibration values, ``searchsorted(..., side='left')`` gives
    # the number of calibration values strictly below each score.  Subtracting
    # from n therefore counts values ``>= score`` exactly, including ties,
    # without constructing an O(n_scores * n_calibration) matrix.
    ordered = np.sort(calibration_nonmember)
    count_ge = len(ordered) - np.searchsorted(ordered, scores, side="left")
    return (1.0 + count_ge) / (len(calibration_nonmember) + 1.0)


def order_statistic_threshold(
    calibration_nonmember: np.ndarray, fpr: float
) -> float:
    """Boundary for the inclusive-tie conformal rule.

    The actual decision is p-value <= fpr.  For distinct values it is
    equivalent to ``score > threshold`` where threshold is the (k+1)-th largest
    calibration score; using p-values above also handles ties correctly.
    """
    calibration_nonmember = np.asarray(calibration_nonmember, dtype=np.float64)
    if not 0.0 < fpr < 1.0 or calibration_nonmember.size == 0:
        raise ValueError("invalid FPR or empty calibration set")
    max_calibration_tail = int(np.ceil(fpr * (len(calibration_nonmember) + 1.0))) - 1
    order_from_largest = max(1, max_calibration_tail + 1)
    ordered = np.sort(calibration_nonmember)[::-1]
    if order_from_largest > len(ordered):
        return float("-inf")
    return float(ordered[order_from_largest - 1])


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores, ordered_labels = scores[order], labels[order]
    starts = np.flatnonzero(np.r_[True, ordered_scores[1:] != ordered_scores[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))
    tp = fp = 0
    fpr, tpr = [0.0], [0.0]
    for start, end in zip(starts, ends):
        group = ordered_labels[start:end]
        tp += int(np.sum(group == 1))
        fp += int(np.sum(group == 0))
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
    return np.asarray(fpr), np.asarray(tpr)


def rank_auc(member: np.ndarray, nonmember: np.ndarray) -> float:
    combined = np.concatenate((member, nonmember))
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = ranks[inverse]
    return float((ranks[: len(member)].sum() - len(member) * (len(member) + 1) / 2.0) / (len(member) * len(nonmember)))


def partial_auc(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.10) -> float:
    fpr, tpr = _roc_points(scores, labels)
    area = 0.0
    for left in range(len(fpr) - 1):
        x0, x1 = float(fpr[left]), float(fpr[left + 1])
        if x0 >= max_fpr:
            break
        right = min(x1, max_fpr)
        if right <= x0:
            continue
        fraction = (right - x0) / max(1e-12, x1 - x0)
        y_right = tpr[left] + fraction * (tpr[left + 1] - tpr[left])
        area += (right - x0) * (tpr[left] + y_right) / 2.0
        if x1 >= max_fpr:
            break
    return float(area / max_fpr)


def membership_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    member = test[labels[test] == 1]
    nonmember = test[labels[test] == 0]
    calibration = calibration[labels[calibration] == 0]
    y = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    result = {
        "auc": rank_auc(scores[member], scores[nonmember]),
        "pauc_0_10": partial_auc(np.r_[scores[member], scores[nonmember]], y),
        "tpr_at_fpr": {},
    }
    member_p = conformal_tail_pvalues(scores[member], scores[calibration])
    nonmember_p = conformal_tail_pvalues(scores[nonmember], scores[calibration])
    for rate in RATES:
        result["tpr_at_fpr"][f"{int(rate*100)}%"] = {
            "tpr": float(np.mean(member_p <= rate)),
            "actual_fpr": float(np.mean(nonmember_p <= rate)),
        }
    return result

