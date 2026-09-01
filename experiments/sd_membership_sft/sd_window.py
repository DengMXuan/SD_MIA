"""SD-Window: windowed robust aggregation of speculative-decoding feedback.

Second-generation SD attack family. Where the first-generation methods
(``audit.py`` transcript scores, ``draver_activation.py`` DraVer-Act) pool
the verifier acceptance transcript over the whole record, this module
re-aggregates the SAME per-position acceptance observations with the
windowed robust statistics of WBC (Chen et al., USENIX Security 2026):

- window definitions: (1) speculative-decoding block structure (accepted
  depth per block, SD-native), (2) adjacency clusters of the probed
  min-k positions (memorization locality), (3) WBC geometric window sizes
  over position order;
- the visibility mask is honoured everywhere: positions after a block's
  first rejection are excluded from every window statistic. In the current
  fixed-candidate transcript simulation every probed position is visible,
  so the mask is all-True; it becomes binding only for passive L2 traces;
- two per-window statistics: the sign vote (fraction of windows whose mean
  acceptance exceeds the nonmember baseline) and the Wilcoxon-style
  signed-rank statistic over window means (uses acceptance magnitude,
  rank-robust). Both are bounded to [-1, 1];
- records with no admissible window fall back to their OWN unwindowed
  acceptance mean, never to a global constant, so windowing never destroys
  the information of unclustered records;
- the 2x2 method matrix closes by conditioning draft activations on window
  membership (window-pooled activation features) on top of the windowed
  transcript scores.

Pure post-processing: every function consumes arrays already produced by the
feature-extraction layer, so existing checkpoints can be re-scored without
touching any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audit import metric_row
from .draver_activation import (
    paired_bootstrap_delta,
    transcript_summary,
    weighted_pool,
)


@dataclass(frozen=True)
class WindowResult:
    """Per-record windowed statistics for one window definition."""

    # [records, max_windows] window mean acceptance, NaN padding
    window_means: np.ndarray
    # [records] number of valid windows per record
    window_counts: np.ndarray


def _validate_inputs(
    acceptance: np.ndarray, visible: np.ndarray | None, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acceptance = np.asarray(acceptance, dtype=np.float64)
    if acceptance.ndim != 2:
        raise ValueError("acceptance must be [records, probed_positions]")
    records, width = acceptance.shape
    positions = np.asarray(positions)
    if positions.shape != (records, width):
        raise ValueError("positions must match acceptance shape")
    if visible is None:
        visible = np.isfinite(acceptance)
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != acceptance.shape:
        raise ValueError("visible must match acceptance shape")
    return acceptance, visible, positions


def _record_means(
    acceptance: np.ndarray, visible: np.ndarray | None, baseline: float
) -> np.ndarray:
    """Per-record mean over visible finite positions; baseline if none."""
    acceptance, visible, _ = _validate_inputs(acceptance, visible, _dummy_positions(acceptance))
    usable = visible & np.isfinite(acceptance)
    counts = np.maximum(usable.sum(axis=1), 1)
    sums = np.where(usable, acceptance, 0.0).sum(axis=1)
    means = sums / counts
    return np.where(usable.any(axis=1), means, baseline)


def _dummy_positions(acceptance: np.ndarray) -> np.ndarray:
    return np.zeros_like(acceptance, dtype=np.int64)


def _cluster_bounds(sorted_positions: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[start, end) index bounds of runs whose consecutive position gaps are <= gap."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(sorted_positions) + 1):
        if (
            index == len(sorted_positions)
            or sorted_positions[index] - sorted_positions[index - 1] > gap
        ):
            bounds.append((start, index))
            start = index
    return bounds


# ---------------------------------------------------------------------------
# Window definitions
# ---------------------------------------------------------------------------


def block_windows(
    acceptance: np.ndarray,
    visible: np.ndarray | None,
    positions: np.ndarray,
    block_size: int = 4,
) -> WindowResult:
    """Group probed positions by their speculative-decoding block index.

    SD-native windows: the block id of a probed position is its original
    document position // block_size. Blocks with no visible probed position
    yield NaN. This is the only definition whose windows exist
    independently of the probe selection.
    """
    acceptance, visible, positions = _validate_inputs(acceptance, visible, positions)
    records, width = acceptance.shape
    block_ids = positions // max(1, block_size)
    max_windows = 1 + (int(positions.max()) if positions.size else 0) // max(1, block_size)
    means = np.full((records, max_windows), np.nan)
    counts = np.zeros(records, dtype=np.int64)
    for row in range(records):
        for block in np.unique(block_ids[row]):
            mask = (block_ids[row] == block) & visible[row] & np.isfinite(acceptance[row])
            if not mask.any():
                continue
            means[row, int(block)] = float(acceptance[row][mask].mean())
            counts[row] += 1
    return WindowResult(means, counts)


def adjacency_windows(
    acceptance: np.ndarray,
    visible: np.ndarray | None,
    positions: np.ndarray,
    gap: int = 4,
    min_size: int = 2,
) -> WindowResult:
    """Cluster probed positions by adjacency in the original document.

    Consecutive probed positions whose document indices differ by at most
    ``gap`` form one cluster (memorization tends to be local, so min-k hits
    clump). Clusters shorter than ``min_size`` are dropped; a record with no
    cluster has zero valid windows and falls back to its own record mean
    downstream.
    """
    acceptance, visible, positions = _validate_inputs(acceptance, visible, positions)
    records, width = acceptance.shape
    max_clusters = max(1, width // max(1, min_size))
    means = np.full((records, max_clusters), np.nan)
    counts = np.zeros(records, dtype=np.int64)
    for row in range(records):
        order = np.argsort(positions[row])
        sorted_positions = positions[row][order]
        sorted_ok = visible[row][order] & np.isfinite(acceptance[row][order])
        for cluster_start, cluster_end in _cluster_bounds(sorted_positions, gap):
            members = [
                index
                for index in range(cluster_start, cluster_end)
                if sorted_ok[index]
            ]
            if len(members) < min_size or counts[row] >= max_clusters:
                continue
            values = acceptance[row][order][members]
            means[row, counts[row]] = float(values.mean())
            counts[row] += 1
    return WindowResult(means, counts)


def geometric_windows(
    acceptance: np.ndarray,
    visible: np.ndarray | None,
    positions: np.ndarray,
    w_min: int = 2,
    w_max: int = 13,
    n_sizes: int = 4,
) -> WindowResult:
    """WBC geometric window sizes over the probed positions in document order.

    Windows are consecutive runs of ``w`` probed positions (not document
    tokens, which the probe selection does not cover densely). A window is
    valid only if every one of its positions is visible; ``window_counts``
    reports the number of valid window means (across all sizes), keeping the
    sign statistic on the same [0, 1] scale as the other definitions.
    """
    acceptance, visible, positions = _validate_inputs(acceptance, visible, positions)
    records, width = acceptance.shape
    sizes = [
        max(1, int(round(w_min * (w_max / w_min) ** (k / (n_sizes - 1)))))
        for k in range(n_sizes)
    ]
    total = sum(max(0, width - w + 1) for w in sizes)
    means = np.full((records, total), np.nan)
    counts = np.zeros(records, dtype=np.int64)
    for row in range(records):
        order = np.argsort(positions[row])
        values = acceptance[row][order]
        ok = visible[row][order] & np.isfinite(values)
        offset = 0
        for w in sizes:
            for start in range(0, width - w + 1):
                if ok[start : start + w].all():
                    means[row, offset + start] = float(values[start : start + w].mean())
            offset += max(0, width - w + 1)
        counts[row] = int(np.isfinite(means[row]).sum())
    return WindowResult(means, counts)


WINDOW_DEFINITIONS = {
    "block": block_windows,
    "adjacency": adjacency_windows,
    "geometric": geometric_windows,
}


# ---------------------------------------------------------------------------
# Per-window statistics
# ---------------------------------------------------------------------------


def sign_scores(window_result: WindowResult, baseline: float = 0.5) -> np.ndarray:
    """WBC-style sign vote: fraction of valid windows with mean > baseline.

    The denominator is the record's number of valid window means, so the
    score stays in [0, 1] for every window definition.
    """
    means = window_result.window_means
    finite = np.isfinite(means)
    counts = np.maximum(finite.sum(axis=1), 1)
    votes = np.where(finite, means > baseline, 0.0).sum(axis=1) / counts
    votes[finite.sum(axis=1) == 0] = np.nan
    return votes


def wilcoxon_scores(window_result: WindowResult, baseline: float = 0.5) -> np.ndarray:
    """Signed-rank statistic: |mean - baseline| weighted by rank direction.

    Uses acceptance magnitude around the nonmember baseline while remaining
    robust through the rank transform; zero-deviation windows are dropped.
    Ranks are ordinal (ties are negligible for continuous acceptance rates);
    the score is bounded to [-1, 1].
    """
    means = window_result.window_means
    deviation = means - baseline
    valid = np.isfinite(deviation) & (np.abs(deviation) > 1e-12)
    scores = np.full(len(means), np.nan)
    for row in range(len(means)):
        rows = deviation[row][valid[row]]
        if rows.size == 0:
            continue
        ranks = np.argsort(np.argsort(np.abs(rows))).astype(np.float64) + 1.0
        scores[row] = float(np.sum(np.sign(rows) * ranks) / np.sum(ranks))
    return scores


# ---------------------------------------------------------------------------
# Window-conditioned activation pooling (2x2 matrix: transcript x activations)
# ---------------------------------------------------------------------------


def _window_weights(
    acceptance: np.ndarray,
    positions: np.ndarray,
    visible: np.ndarray | None,
    gap: int = 4,
    min_size: int = 2,
) -> np.ndarray:
    """Per-record pooling weights: acceptance inside adjacency clusters, 0 outside.

    A record with no admissible cluster falls back to its unwindowed visible
    acceptance weights, so the conditioned family never loses a record.
    """
    acceptance, visible, positions = _validate_inputs(acceptance, visible, positions)
    records, width = acceptance.shape
    weights = np.zeros((records, width), dtype=np.float64)
    for row in range(records):
        order = np.argsort(positions[row])
        sorted_positions = positions[row][order]
        sorted_ok = visible[row][order] & np.isfinite(acceptance[row][order])
        placed = False
        for cluster_start, cluster_end in _cluster_bounds(sorted_positions, gap):
            members = [
                index
                for index in range(cluster_start, cluster_end)
                if sorted_ok[index]
            ]
            if len(members) < min_size:
                continue
            placed = True
            for index in members:
                weights[row, order[index]] = acceptance[row][order[index]]
        if not placed:
            weights[row] = np.where(
                visible[row] & np.isfinite(acceptance[row]), acceptance[row], 0.0
            )
    return weights


def _window_conditioned_activations(
    selected_activations: np.ndarray,
    acceptance: np.ndarray,
    positions: np.ndarray,
    visible: np.ndarray | None,
    gap: int = 4,
) -> np.ndarray:
    """Pool draft activations restricted to adjacency-window support.

    Within each record, only positions inside adjacency clusters carry
    weight: the accepted pool is weighted by cluster acceptance, the
    rejected pool by (1 - acceptance) over the same support, and the
    covariance block by the cluster-centred acceptance. Records with no
    cluster fall back to unwindowed conditioning on all visible positions.
    The global transcript summary is appended so the family is a strict
    superset of the first generation.
    """
    acceptance, visible, positions = _validate_inputs(acceptance, visible, positions)
    records, width = acceptance.shape
    if selected_activations.shape[0] != records or selected_activations.shape[1] != width:
        raise ValueError(
            "selected_activations must align with acceptance "
            f"({selected_activations.shape[:2]} vs {acceptance.shape})"
        )
    weights = _window_weights(acceptance, positions, visible, gap)
    in_cluster = weights > 0
    acc64 = np.nan_to_num(acceptance, nan=0.0)

    accepted_weights = np.where(in_cluster, acc64, 0.0)
    rejected_weights = np.where(in_cluster, 1.0 - acc64, 0.0)
    # cluster-centred acceptance: mean over the record's in-cluster positions
    sums = np.where(in_cluster, acc64, 0.0).sum(axis=1)
    counts = np.maximum(in_cluster.sum(axis=1), 1)
    centres = (sums / counts)[:, None]
    centred = np.where(in_cluster, acc64 - centres, 0.0)
    # records with no cluster fall back to unwindowed conditioning
    fallback = ~in_cluster.any(axis=1)
    if fallback.any():
        accepted_weights[fallback] = acc64[fallback]
        rejected_weights[fallback] = 1.0 - acc64[fallback]
        centred[fallback] = acc64[fallback] - acc64[fallback].mean(axis=1, keepdims=True)

    summary = transcript_summary(acceptance.astype(np.float32))
    blocks = [
        weighted_pool(selected_activations, accepted_weights.astype(np.float32)),
        weighted_pool(selected_activations, rejected_weights.astype(np.float32)),
        weighted_pool(selected_activations, accepted_weights.astype(np.float32))
        - weighted_pool(selected_activations, rejected_weights.astype(np.float32)),
        np.mean(
            selected_activations * centred[:, :, None, None], axis=1
        ),
    ]
    flattened = [block.reshape(len(block), -1) for block in blocks]
    return np.column_stack([*flattened, summary]).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature-family registration for the audit
# ---------------------------------------------------------------------------


def build_window_feature_families(
    acceptance: np.ndarray,
    positions: np.ndarray,
    visible: np.ndarray | None = None,
    baseline: float = 0.5,
) -> dict[str, np.ndarray]:
    """Direct scores for every (window definition x statistic) combination.

    Records with no admissible window fall back to their own unwindowed
    acceptance mean over visible positions (the first-generation statistic),
    never to a global constant.
    """
    record_means = _record_means(acceptance, visible, baseline)
    families: dict[str, np.ndarray] = {}
    for name, builder in WINDOW_DEFINITIONS.items():
        result = builder(acceptance, visible, positions)
        for statistic_name, statistic in (
            ("sign", sign_scores(result, baseline)),
            ("wilcoxon", wilcoxon_scores(result, baseline)),
        ):
            statistic = np.asarray(statistic, dtype=np.float64)
            statistic = np.where(np.isfinite(statistic), statistic, record_means)
            families[f"sd_window_{name}_{statistic_name}"] = statistic
    return families


def evaluate_window_audit(
    acceptance: np.ndarray,
    positions: np.ndarray,
    selected_activations: np.ndarray | None,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    bootstrap_repeats: int,
    seed: int,
    visible: np.ndarray | None = None,
    detector_seeds: int = 3,
) -> dict[str, Any]:
    """Score every window family on the shared audit split.

    Direct scores (sign/Wilcoxon) are threshold-free. The conditioned
    activation family reuses the metric-learning triplet detector so all
    rows of the 2x2 matrix are evaluated under one protocol. Paired deltas
    are reported against the unwindowed acceptance mean.
    """
    from .draver_activation import triplet_scores_by_seed

    families = build_window_feature_families(acceptance, positions, visible)
    results: dict[str, dict[str, float]] = {}
    all_scores: dict[str, np.ndarray] = {}
    test_labels = labels[test]

    transcript_summary_values = transcript_summary(acceptance.astype(np.float32))
    reference = transcript_summary_values.mean(axis=1).astype(np.float64)

    def register(name: str, score: np.ndarray, test_indexed: bool = False) -> None:
        values = np.asarray(score, dtype=np.float64)
        if not test_indexed:
            values = values[test]
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name}: NaN/inf scores survived the record-mean fallback; "
                "refusing to fill with a global constant"
            )
        all_scores[name] = values
        results[name] = metric_row(test_labels, values, bootstrap_repeats, seed)

    register("sd_window/acceptance_mean_reference", reference)
    for name, score in families.items():
        register(f"sd_window/{name.removeprefix('sd_window_')}", score)

    if selected_activations is not None:
        window_conditioned = _window_conditioned_activations(
            selected_activations, acceptance, positions, visible
        )
        seeds = tuple(seed + 1000 + offset for offset in range(detector_seeds))
        score = triplet_scores_by_seed(
            window_conditioned, labels, calibration, test, seeds
        ).mean(axis=0)
        register("sd_window/conditioned_activation_triplet", score, test_indexed=True)

    deltas = {}
    for name, score in all_scores.items():
        if name == "sd_window/acceptance_mean_reference":
            continue
        deltas[f"{name} minus unwindowed_mean"] = paired_bootstrap_delta(
            test_labels,
            score,
            all_scores["sd_window/acceptance_mean_reference"],
            bootstrap_repeats,
            seed + 7,
        )
    return {
        "metrics": results,
        "scores": all_scores,
        "paired_auc_deltas": deltas,
        "protocol": {
            "window_definitions": sorted(WINDOW_DEFINITIONS),
            "visibility_mode": (
                "first-rejection mask applied"
                if visible is not None and not np.all(visible)
                else "fixed-candidate simulation (all probed positions visible)"
            ),
        },
    }
