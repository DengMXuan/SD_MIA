"""One definition of ranking and deployment metrics for the Qwen audit matrix."""
from __future__ import annotations

import numpy as np

from experiments.sd_membership_sft.core.audit_metrics import rank_auc, partial_auc, conformal_tail_pvalues
from experiments.sd_membership_sft.methods.deployment_accept_only import _wilson


def roc_operating_point(scores, labels, fpr):
    """Highest attainable TPR with empirical FPR <= cap; never split a tie."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if not 0 < fpr < 1 or not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("finite scores, two test classes and a valid FPR cap required")
    order = np.argsort(-scores, kind="stable")
    values, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(y) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, np.cumsum(1 - y)[ends]]
    allowed = np.flatnonzero(fp / (labels == 0).sum() <= fpr + 1e-12)
    index = allowed[-1]
    return float(tp[index] / (labels == 1).sum()), float(fp[index] / (labels == 0).sum())


def metrics(scores, labels, calibration, test, *, seed=20260914, bootstrap=200):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    calibration, test = np.asarray(calibration, int), np.asarray(test, int)
    if (scores.shape != labels.shape or scores.ndim != 1 or not np.isfinite(scores).all()
            or calibration.ndim != 1 or test.ndim != 1
            or not len(calibration) or not len(test)
            or (calibration < 0).any() or (test < 0).any()
            or (calibration >= len(scores)).any() or (test >= len(scores)).any()
            or len(np.unique(calibration)) != len(calibration)
            or len(np.unique(test)) != len(test)
            or np.intersect1d(calibration, test).size
            or (labels[calibration] != 0).any()
            or set(np.unique(labels[test])) != {0, 1}):
        raise ValueError("invalid scores or independent calibration/test partitions")
    y, values = labels[test], scores[test]
    member, nonmember = values[y == 1], values[y == 0]
    normalized = partial_auc(values, y, .1)
    result = {"auc": rank_auc(member, nonmember), "pauc_10_raw": normalized * .1,
              "pauc_10_normalized": normalized, "n_test_member": len(member),
              "n_test_nonmember": len(nonmember), "n_calibration": len(calibration)}
    if bootstrap:
        rng = np.random.default_rng(seed)
        samples = [rank_auc(rng.choice(member, len(member)), rng.choice(nonmember, len(nonmember)))
                   for _ in range(bootstrap)]
        result["auc_ci_low"], result["auc_ci_high"] = np.quantile(samples, [.025, .975]).tolist()
    pvalues = conformal_tail_pvalues(values, scores[calibration])
    for rate, suffix in ((.1, "10"), (.01, "1")):
        tpr, actual = roc_operating_point(values, y, rate)
        result[f"roc_tpr_at_{suffix}pct_fpr"] = tpr
        result[f"roc_actual_fpr_at_{suffix}pct"] = actual
        for mask, name in ((y == 1, "tpr"), (y == 0, "actual_fpr")):
            hits = int((pvalues[mask] <= rate).sum())
            total = int(mask.sum())
            field = f"calibrated_{name}_at_{suffix}pct"
            result[field] = hits / total
            result[field + "_ci_low"], result[field + "_ci_high"] = _wilson(hits, total)
    return result


METRIC_CONVENTIONS = {
    "orientation": "larger score means more likely member; no test-label orientation selection",
    "roc_tpr": "largest attainable TPR at empirical FPR <= cap; tied scores stay together",
    "pauc": "trapezoidal ROC area on [0,0.10]; normalized = raw / 0.10, not chance-corrected",
    "deployment": "inclusive-tie conformal p <= nominal alpha using independent nonmember calibration",
    "uncertainty": "document-stratified bootstrap AUC; Wilson deployment rates conditional on calibration",
}
