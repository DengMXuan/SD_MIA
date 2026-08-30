from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import SFTRecord


def bottom_k_indices(token_logp: np.ndarray, fraction: float) -> np.ndarray:
    finite = np.isfinite(token_logp)
    width = token_logp.shape[1]
    k = max(1, min(width, int(math.ceil(width * fraction))))
    safe = np.where(finite, token_logp, np.inf)
    return np.argpartition(safe, kth=k - 1, axis=1)[:, :k]


def cap_selected_positions(
    selected: np.ndarray, draft_logp: np.ndarray, cap: int | None
) -> np.ndarray:
    """Keep only the ``cap`` least-likely positions among the min-k selection.

    With long NART documents the min-k selection alone would grow the verifier
    transcript budget with document length; the cap pins it to
    ``cap * repeats`` bits per record (26 * 24 = 624 by default).
    """
    if cap is None or cap >= selected.shape[1]:
        return selected
    gathered = gather_positions(draft_logp, selected)
    order = np.argsort(gathered, axis=1, kind="mergesort")[:, :cap]
    return np.take_along_axis(selected, order, axis=1)


def make_audit_split(
    n_members: int, n_nonmembers: int, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Audit calibration/test split shared by every audit path.

    Uses ``seed + 30`` so existing runs keep the exact split they were
    produced with. Direct-verifier baselines, the SD transcript scores, and
    the DraVer-Act detector must all evaluate on the same test records for
    paired comparisons to be valid.
    """
    rng = np.random.default_rng(seed + 30)
    positive = rng.permutation(n_members)
    negative = rng.permutation(n_nonmembers) + n_members
    train = np.concatenate([positive[:per_class], negative[:per_class]])
    test = np.concatenate([positive[per_class:], negative[per_class:]])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def window_based_comparison(
    target_logp: np.ndarray,
    reference_logp: np.ndarray,
    w_min: int = 2,
    w_max: int = 40,
    n_window_sizes: int = 10,
) -> np.ndarray:
    """WBC attack score (Chen et al., USENIX Security 2026).

    Reimplemented from the paper's Equations 10 and 12; official code is
    github.com/Stry233/WBC. With token losses defined as l = -logp, the
    per-token loss difference is Delta_j = l_R_j - l_T_j = logp_T_j -
    logp_R_j. For each geometrically spaced window size w, the statistic is
    the fraction of sliding windows whose Delta sum is positive; the final
    score averages the statistic across sizes. Hyperparameters follow the
    paper's ablation optimum: w_min=2, w_max=40, |W|=10, stride 1.

    ``target_logp`` and ``reference_logp`` are [records, positions] arrays
    with NaN padding; NaN positions are excluded via a zero-padded delta
    cumsum over each row's valid prefix.
    """
    delta = target_logp - reference_logp
    rows, width = delta.shape
    scores = np.full(rows, np.nan, dtype=np.float64)
    sizes = [
        int(round(w_min * (w_max / w_min) ** (k / (n_window_sizes - 1))))
        for k in range(n_window_sizes)
    ]
    for row in range(rows):
        valid = np.isfinite(delta[row])
        count = int(valid.sum())
        if count == 0:
            continue
        values = np.where(valid, delta[row], 0.0)[:count]
        cumsum = np.concatenate([[0.0], np.cumsum(values)])
        statistics = []
        for w in sizes:
            if w > count:
                continue
            window_sums = cumsum[w:] - cumsum[:-w]
            statistics.append(float(np.mean(window_sums > 0.0)))
        if statistics:
            scores[row] = float(np.mean(statistics))
    return scores


def min_k_prob(target_logp: np.ndarray, fraction: float) -> np.ndarray:
    """Min-K% Prob attack score (Shi et al., ICLR 2024).

    Reimplemented per the paper (official code github.com/swj0419/detect-
    pretrain-code): mean log-probability of the k% least-likely tokens.
    """
    selected = bottom_k_indices(target_logp, fraction)
    return np.nanmean(gather_positions(target_logp, selected), axis=1)


def reference_loss_diff(
    target_logp: np.ndarray, reference_logp: np.ndarray
) -> np.ndarray:
    """Reference-based global loss-difference score.

    The standard fine-tuned-MIA baseline the WBC paper compares against:
    mean over document tokens of (reference loss - target loss) = mean
    Delta_j, i.e. WBC's signal without windowing or sign aggregation.
    """
    delta = target_logp - reference_logp
    return np.nanmean(np.where(np.isfinite(delta), delta, np.nan), axis=1)


def gather_positions(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, positions, axis=1)


def transcript_tomography(
    target_logp: np.ndarray,
    draft_logp: np.ndarray,
    selected: np.ndarray,
    repeats: int,
    levels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate target token probabilities from simulated accept/reject bits."""
    rng = np.random.default_rng(seed)
    p = np.exp(gather_positions(target_logp, selected)).astype(np.float64)
    q0 = np.exp(gather_positions(draft_logp, selected)).astype(np.float64)
    estimates = np.empty_like(p)
    query_bits = np.zeros(p.shape[0], dtype=np.int64)
    honest_accept = np.empty_like(p)

    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            pv = float(p[row, col])
            q_base = float(np.clip(q0[row, col], 1e-7, 0.95))
            honest_alpha = min(1.0, pv / q_base)
            honest_count = rng.binomial(repeats, honest_alpha)
            honest_accept[row, col] = (honest_count + 0.5) / (repeats + 1.0)

            chosen_estimate = q_base
            best_distance = float("inf")
            for level in range(levels):
                q_probe = float(min(0.95, q_base * (2.0**level)))
                alpha = min(1.0, pv / q_probe)
                count = int(rng.binomial(repeats, alpha))
                query_bits[row] += repeats
                smoothed = (count + 0.5) / (repeats + 1.0)
                estimate = q_probe * smoothed
                distance = abs(smoothed - 0.5)
                if distance < best_distance:
                    best_distance = distance
                    chosen_estimate = estimate
                if 0 < count < repeats and smoothed <= 0.80:
                    chosen_estimate = estimate
                    break
                if q_probe >= 0.95:
                    chosen_estimate = estimate
                    break
            estimates[row, col] = float(np.clip(chosen_estimate, 1e-12, 1.0))
    return np.log(estimates), honest_accept, query_bits


def auc_rank(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    positive = np.asarray(scores)[np.asarray(y) == 1]
    negative = np.sort(np.asarray(scores)[np.asarray(y) == 0])[::-1]
    allowed_fp = int(math.floor(target_fpr * len(negative)))
    if allowed_fp >= len(negative):
        threshold = -np.inf
    else:
        threshold = np.nextafter(negative[allowed_fp], np.inf)
    return float(np.mean(positive >= threshold))


def bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    values: list[float] = []
    for _ in range(repeats):
        index = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        values.append(auc_rank(y[index], scores[index]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    steps: int = 1200,
    lr: float = 0.08,
    l2: float = 0.02,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    train = np.concatenate([(x_train - mean) / std, np.ones((len(x_train), 1))], axis=1)
    test = np.concatenate([(x_test - mean) / std, np.ones((len(x_test), 1))], axis=1)
    weights = np.zeros(train.shape[1], dtype=np.float64)
    y_float = y_train.astype(np.float64)
    for _ in range(steps):
        logits = np.clip(train @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = train.T @ (probabilities - y_float) / len(train)
        gradient[:-1] += l2 * weights[:-1]
        weights -= lr * gradient
    return 1.0 / (1.0 + np.exp(-np.clip(test @ weights, -30, 30)))


def hashed_bow(ids: Any, width: int = 2048) -> np.ndarray:
    """Hashed bag-of-words over variable-length token id sequences."""
    result = np.zeros((len(ids), width), dtype=np.float32)
    for row, sequence in enumerate(ids):
        tokens = np.asarray(sequence, dtype=np.int64)
        bins = np.mod(tokens * 2654435761, width)
        np.add.at(result[row], bins, 1.0)
    result /= np.maximum(result.sum(axis=1, keepdims=True), 1.0)
    return result


def random_project_hidden(hidden: np.ndarray, seed: int, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0, 1.0 / math.sqrt(width), size=(hidden.shape[1], width)
    ).astype(np.float32)
    return hidden @ projection


def metric_row(
    y: np.ndarray, scores: np.ndarray, bootstrap_repeats: int, seed: int
) -> dict[str, float]:
    low, high = bootstrap_auc_ci(y, scores, bootstrap_repeats, seed)
    return {
        "auc": auc_rank(y, scores),
        "auc_ci95_low": low,
        "auc_ci95_high": high,
        "tpr_at_1pct_fpr": tpr_at_fpr(y, scores, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(y, scores, 0.05),
    }


def add_draft_metrics(
    prefix: str,
    draft: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    min_k_fraction: float,
    transcript_repeats: int,
    transcript_levels: int,
    bootstrap_repeats: int,
    seed_offset: int,
    results: dict[str, dict[str, float]],
    selected_token_cap: int | None = None,
) -> dict[str, np.ndarray]:
    selected = cap_selected_positions(
        bottom_k_indices(draft["token_logp"], min_k_fraction),
        draft["token_logp"],
        selected_token_cap,
    )
    draft_mean = np.nanmean(draft["token_logp"], axis=1)
    draft_min = np.nanmean(gather_positions(draft["token_logp"], selected), axis=1)
    draft_entropy = -np.nanmean(draft["entropy"], axis=1)
    draft_grad = -np.nanmean(gather_positions(draft["grad_proxy"], selected), axis=1)
    greedy = np.nanmean(gather_positions(target["top1_match"], selected), axis=1)
    oracle_qselected = np.nanmean(gather_positions(target["token_logp"], selected), axis=1)
    target_selected = bottom_k_indices(target["token_logp"], min_k_fraction)
    true_target_min = np.nanmean(
        gather_positions(target["token_logp"], target_selected), axis=1
    )

    tomo_logp, honest_accept, query_bits = transcript_tomography(
        target["token_logp"],
        draft["token_logp"],
        selected,
        transcript_repeats,
        transcript_levels,
        seed_offset,
    )
    tomo = np.nanmean(tomo_logp, axis=1)
    honest = np.nanmean(honest_accept, axis=1)

    rng = np.random.default_rng(seed_offset + 1)
    random_selected = np.vstack(
        [
            rng.choice(target["token_logp"].shape[1], selected.shape[1], replace=False)
            for _ in range(len(labels))
        ]
    )
    random_tomo_logp, _, random_query_bits = transcript_tomography(
        target["token_logp"],
        draft["token_logp"],
        random_selected,
        transcript_repeats,
        transcript_levels,
        seed_offset + 2,
    )
    random_tomo = np.nanmean(random_tomo_logp, axis=1)

    scores = {
        f"{prefix}/draft_mean_logp": draft_mean,
        f"{prefix}/draft_min20_logp": draft_min,
        f"{prefix}/draft_negative_entropy": draft_entropy,
        f"{prefix}/draft_negative_grad_proxy": draft_grad,
        f"{prefix}/greedy_verifier_match_selected": greedy,
        f"{prefix}/honest_accept_rate_selected": honest,
        f"{prefix}/acceptance_tomography_random": random_tomo,
        f"{prefix}/acceptance_tomography_qmin": tomo,
        f"{prefix}/oracle_target_qselected": oracle_qselected,
        f"{prefix}/oracle_target_true_min20": true_target_min,
    }
    y_test = labels[test_idx]
    for metric_idx, (name, score) in enumerate(scores.items()):
        results[name] = metric_row(
            y_test,
            score[test_idx],
            bootstrap_repeats,
            seed_offset + metric_idx + 10,
        )

    hidden = random_project_hidden(draft["hidden_mean"], seed_offset + 3)
    hidden_test = fit_logistic(hidden[train_idx], labels[train_idx], hidden[test_idx])
    results[f"{prefix}/whitebox_hidden_probe"] = metric_row(
        y_test, hidden_test, bootstrap_repeats, seed_offset + 40
    )

    joint = np.column_stack(
        [draft_mean, draft_min, draft_entropy, draft_grad, greedy, honest, tomo]
    )
    joint_test = fit_logistic(joint[train_idx], labels[train_idx], joint[test_idx])
    results[f"{prefix}/joint_whitebox_transcript"] = metric_row(
        y_test, joint_test, bootstrap_repeats, seed_offset + 41
    )
    raw_scores = {
        name: np.asarray(score)[test_idx] for name, score in scores.items()
    }
    raw_scores[f"{prefix}/joint_whitebox_transcript"] = joint_test
    return {
        "query_bits": query_bits,
        "random_query_bits": random_query_bits,
        "raw_scores": raw_scores,
    }


def run_audit(
    members: list[SFTRecord],
    nonmembers: list[SFTRecord],
    target_features: dict[str, np.ndarray],
    draft_features: dict[str, dict[str, np.ndarray]],
    config: Any,
    selected_token_cap: int | None = None,
    base_target_features: dict[str, np.ndarray] | None = None,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, np.ndarray]]:
    candidates = members + nonmembers
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)]
    )
    ids = [record.response_ids for record in candidates]
    if split is None:
        train_idx, test_idx = make_audit_split(
            len(members), len(nonmembers), config.audit_train_per_class, config.audit_seed
        )
    else:
        train_idx, test_idx = split

    results: dict[str, dict[str, float]] = {}
    raw_scores: dict[str, np.ndarray] = {}

    def register(name: str, score: np.ndarray, preindexed: bool = False) -> None:
        test_score = score if preindexed else score[test_idx]
        results[name] = metric_row(
            labels[test_idx], test_score, config.bootstrap_repeats, config.audit_seed + 500
        )
        raw_scores[name] = test_score

    bow = hashed_bow(ids)
    bow_test = fit_logistic(bow[train_idx], labels[train_idx], bow[test_idx])
    register("control/model_less_hashed_bow", bow_test, preindexed=True)

    if base_target_features is not None:
        # Direct verifier MIA baselines: score-based attacks on the fine-tuned
        # target with (WBC, reference loss-diff) or without (Min-K%) the
        # pre-fine-tuning reference model. No draft or protocol signal used.
        register(
            "verifier_direct/min_k_prob_k20",
            min_k_prob(target_features["token_logp"], config.min_k_fraction),
        )
        register(
            "verifier_direct/window_based_comparison",
            window_based_comparison(
                target_features["token_logp"],
                base_target_features["token_logp"],
            ),
        )
        register(
            "verifier_direct/reference_loss_diff",
            reference_loss_diff(
                target_features["token_logp"],
                base_target_features["token_logp"],
            ),
        )

    query_arrays: list[np.ndarray] = []
    random_query_arrays: list[np.ndarray] = []
    for index, (prefix, features) in enumerate(draft_features.items()):
        extra = add_draft_metrics(
            prefix,
            features,
            target_features,
            labels,
            train_idx,
            test_idx,
            config.min_k_fraction,
            config.transcript_repeats,
            config.transcript_levels,
            config.bootstrap_repeats,
            config.audit_seed + 600 + index * 100,
            results,
            selected_token_cap=selected_token_cap,
        )
        query_arrays.append(extra["query_bits"])
        random_query_arrays.append(extra["random_query_bits"])
        raw_scores.update(extra["raw_scores"])

    query_bits = np.concatenate(query_arrays)
    random_bits = np.concatenate(random_query_arrays)
    budget = {
        "qmin_median_bits": float(np.median(query_bits)),
        "qmin_p95_bits": float(np.quantile(query_bits, 0.95)),
        "random_median_bits": float(np.median(random_bits)),
        "random_p95_bits": float(np.quantile(random_bits, 0.95)),
        "audit_train_size": int(len(train_idx)),
        "audit_test_size": int(len(test_idx)),
    }
    return results, budget, raw_scores
