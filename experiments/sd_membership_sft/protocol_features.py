from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .audit import auc_rank, metric_row
from .draver_activation import paired_bootstrap_delta, triplet_scores_by_seed


@dataclass(frozen=True)
class ProtocolObservations:
    """Token-aligned observations available to an on-device speculator.

    ``local_state`` is produced by the white-box draft/speculator.  For
    hidden-conditioned protocols, ``connector_state`` is the cloud-exported
    verifier representation consumed by the speculator.  ``visible`` prevents
    positions after a block's first rejection from leaking into features.
    """

    local_state: np.ndarray
    acceptance: np.ndarray
    q_logp: np.ndarray
    visible: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray | None = None
    connector_state: np.ndarray | None = None


def first_rejection_visibility(
    accepted: np.ndarray, block_size: int
) -> np.ndarray:
    """Expose the accepted prefix and its first rejected boundary per block."""
    values = np.asarray(accepted, dtype=bool)
    if values.ndim != 2 or block_size < 1 or values.shape[1] % block_size:
        raise ValueError("accepted must be [records, blocks*block_size]")
    blocks = values.reshape(values.shape[0], -1, block_size)
    visible = np.ones_like(blocks, dtype=bool)
    if block_size > 1:
        visible[:, :, 1:] = np.cumprod(
            blocks[:, :, :-1].astype(np.int8), axis=2
        ).astype(bool)
    return visible.reshape(values.shape)


def _validate(observations: ProtocolObservations) -> tuple[int, int]:
    local = np.asarray(observations.local_state)
    if local.ndim != 3:
        raise ValueError("local_state must have shape [records, positions, features]")
    records, positions, _ = local.shape
    expected = (records, positions)
    for name, values in (
        ("acceptance", observations.acceptance),
        ("q_logp", observations.q_logp),
        ("visible", observations.visible),
    ):
        if np.asarray(values).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    depth = np.asarray(observations.depth)
    if depth.shape not in {(positions,), expected}:
        raise ValueError("depth must have shape [positions] or [records, positions]")
    if observations.confidence is not None:
        if np.asarray(observations.confidence).shape != expected:
            raise ValueError(f"confidence must have shape {expected}")
    if observations.connector_state is not None:
        connector = np.asarray(observations.connector_state)
        if connector.ndim != 3 or connector.shape[:2] != expected:
            raise ValueError(
                "connector_state must have shape [records, positions, features]"
            )
    visible = np.asarray(observations.visible, dtype=bool)
    if np.any(visible.sum(axis=1) == 0):
        raise ValueError("every record needs at least one visible position")
    for name, values in (
        ("local_state", local),
        ("acceptance", observations.acceptance),
        ("q_logp", observations.q_logp),
    ):
        array = np.asarray(values)
        if not np.all(np.isfinite(array[visible])):
            raise ValueError(f"{name} contains non-finite visible values")
    return records, positions


def _broadcast_depth(depth: np.ndarray, records: int) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim == 1:
        values = np.broadcast_to(values[None, :], (records, len(values)))
    return values


def _masked_summary(values: np.ndarray, visible: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, :, None]
    rows: list[np.ndarray] = []
    for row, mask in zip(array, visible, strict=True):
        selected = row[mask]
        quantiles = np.quantile(selected, (0.25, 0.50, 0.75), axis=0)
        rows.append(
            np.concatenate(
                [
                    selected.mean(axis=0),
                    selected.std(axis=0),
                    selected.min(axis=0),
                    quantiles.reshape(-1),
                    selected.max(axis=0),
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _prefix_summary(
    acceptance: np.ndarray, visible: np.ndarray, block_size: int
) -> np.ndarray:
    if block_size < 1 or acceptance.shape[1] % block_size:
        raise ValueError("positions must be divisible by block_size")
    accepted = np.asarray(acceptance) >= 0.5
    blocks = accepted.reshape(len(accepted), -1, block_size)
    masks = visible.reshape(len(visible), -1, block_size)
    rows: list[list[float]] = []
    for row, row_mask in zip(blocks, masks, strict=True):
        lengths: list[float] = []
        for block, mask in zip(row, row_mask, strict=True):
            observed = block[mask]
            rejected = np.flatnonzero(~observed)
            lengths.append(float(rejected[0]) if len(rejected) else float(len(observed)))
        rows.append(
            [
                float(np.mean(lengths)),
                float(np.std(lengths)),
                float(np.min(lengths)),
                float(np.quantile(lengths, 0.5)),
                float(np.max(lengths)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _correlation_feature(
    left: np.ndarray, right: np.ndarray, visible: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(left), 1), dtype=np.float32)
    for index, mask in enumerate(visible):
        x = np.asarray(left[index, mask], dtype=np.float64)
        y = np.asarray(right[index, mask], dtype=np.float64)
        if x.std() > 1e-8 and y.std() > 1e-8:
            result[index, 0] = float(np.corrcoef(x, y)[0, 1])
    return result


def transcript_features(
    observations: ProtocolObservations, block_size: int
) -> np.ndarray:
    """Summarize only cloud-verification feedback visible in the transcript.

    In particular, this baseline must not contain ``q_logp`` or a private
    speculator confidence head.  Those quantities are available because the
    client has white-box access to the speculator, not because they occur in a
    transcript.  Keeping the information boundary explicit is necessary for a
    paired comparison against ordinary transcript-only membership checks.
    """
    records, _ = _validate(observations)
    visible = np.asarray(observations.visible, dtype=bool)
    depth = _broadcast_depth(observations.depth, records)
    acceptance = np.asarray(observations.acceptance, dtype=np.float64)
    return np.column_stack(
        [
            _masked_summary(acceptance, visible),
            _masked_summary(depth, visible),
            _prefix_summary(acceptance, visible, block_size),
        ]
    ).astype(np.float32)


def speculator_difficulty_features(
    observations: ProtocolObservations,
) -> np.ndarray:
    """Summarize white-box proposal difficulty without verifier feedback."""
    _validate(observations)
    visible = np.asarray(observations.visible, dtype=bool)
    q_logp = np.asarray(observations.q_logp, dtype=np.float64)
    blocks = [_masked_summary(q_logp, visible)]
    if observations.confidence is not None:
        blocks.append(
            _masked_summary(
                np.asarray(observations.confidence, dtype=np.float64), visible
            )
        )
    return np.column_stack(blocks).astype(np.float32)


def _ridge_fit_predict(
    design: np.ndarray,
    response: np.ndarray,
    visible: np.ndarray,
    fitting: np.ndarray,
    predicting: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    fit_mask = visible[fitting]
    pred_mask = visible[predicting]
    x_train = design[fitting][fit_mask]
    y_train = response[fitting][fit_mask]
    x_predict = design[predicting][pred_mask]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    train = (x_train - mean) / std
    predict = (x_predict - mean) / std
    train = np.column_stack([train, np.ones(len(train))])
    predict = np.column_stack([predict, np.ones(len(predict))])
    penalty = np.eye(train.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    flat_prediction = predict @ weights
    result = np.zeros((len(predicting), design.shape[1]), dtype=np.float64)
    result[pred_mask] = flat_prediction
    return result


def cross_fitted_protocol_residual(
    observations: ProtocolObservations,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    folds: int = 4,
) -> np.ndarray:
    records, positions = _validate(observations)
    visible = np.asarray(observations.visible, dtype=bool)
    q = np.asarray(observations.q_logp, dtype=np.float64)
    confidence = (
        np.asarray(observations.confidence, dtype=np.float64)
        if observations.confidence is not None
        else np.exp(np.clip(q, -30.0, 0.0))
    )
    depth = _broadcast_depth(observations.depth, records)
    position = np.broadcast_to(
        np.linspace(0.0, 1.0, positions, dtype=np.float64), (records, positions)
    )
    design = np.stack(
        [q, q * q, confidence, confidence * confidence, q * confidence, depth, position],
        axis=-1,
    )
    clipped = np.clip(
        np.asarray(observations.acceptance, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    response = np.log(clipped) - np.log1p(-clipped)
    residual = np.zeros_like(response)
    rng = np.random.default_rng(seed)
    fold_records = np.array_split(
        rng.permutation(np.sort(calibration)), min(folds, len(calibration))
    )
    for held_out in fold_records:
        fitting = np.setdiff1d(calibration, held_out, assume_unique=False)
        prediction = _ridge_fit_predict(
            design, response, visible, fitting, held_out
        )
        residual[held_out] = response[held_out] - prediction
    test_prediction = _ridge_fit_predict(
        design, response, visible, calibration, test
    )
    residual[test] = response[test] - test_prediction
    residual[~visible] = 0.0
    return residual.astype(np.float32)


def stratified_protocol_shuffle(
    observations: ProtocolObservations,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    bins: int = 8,
) -> np.ndarray:
    records, _ = _validate(observations)
    q = np.asarray(observations.q_logp, dtype=np.float64)
    confidence = (
        np.asarray(observations.confidence, dtype=np.float64)
        if observations.confidence is not None
        else np.exp(np.clip(q, -30.0, 0.0))
    )
    depth = _broadcast_depth(observations.depth, records)
    visible = np.asarray(observations.visible, dtype=bool)
    shuffled = np.asarray(observations.acceptance, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    for raw_subset in (calibration, test):
        subset = np.sort(raw_subset)
        subset_visible = visible[subset]
        q_values = q[subset][subset_visible]
        c_values = confidence[subset][subset_visible]
        q_edges = np.unique(np.quantile(q_values, np.linspace(0, 1, bins + 1)))
        c_edges = np.unique(np.quantile(c_values, np.linspace(0, 1, bins + 1)))
        q_group = np.digitize(q_values, q_edges[1:-1], right=True)
        c_group = np.digitize(c_values, c_edges[1:-1], right=True)
        d_group = depth[subset][subset_visible]
        values = shuffled[subset][subset_visible]
        groups = np.column_stack([q_group, c_group, d_group])
        for group in np.unique(groups, axis=0):
            positions = np.flatnonzero(np.all(groups == group, axis=1))
            values[positions] = rng.permutation(values[positions])
        target = shuffled[subset]
        target[subset_visible] = values
        shuffled[subset] = target
    return shuffled.astype(np.float32)


def _residual_state_features(
    state: np.ndarray, residual: np.ndarray, visible: np.ndarray
) -> np.ndarray:
    mask = visible.astype(np.float64)
    denominator = np.maximum(mask.sum(axis=1), 1.0)[:, None]
    weighted = state * residual[:, :, None]
    absolute = state * np.abs(residual[:, :, None])
    delta = np.diff(state, axis=1, prepend=state[:, :1])
    blocks = [
        (weighted * mask[:, :, None]).sum(axis=1) / denominator,
        (absolute * mask[:, :, None]).sum(axis=1) / denominator,
        (delta * residual[:, :, None] * mask[:, :, None]).sum(axis=1)
        / denominator,
    ]
    return np.column_stack(blocks).astype(np.float32)


def build_protocol_feature_families(
    observations: ProtocolObservations,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    _validate(observations)
    visible = np.asarray(observations.visible, dtype=bool)
    local = np.asarray(observations.local_state, dtype=np.float64)
    connector = (
        np.asarray(observations.connector_state, dtype=np.float64)
        if observations.connector_state is not None
        else None
    )
    transcript = transcript_features(observations, block_size)
    difficulty_only = speculator_difficulty_features(observations)
    local_only = np.column_stack(
        [_masked_summary(local, visible), difficulty_only]
    ).astype(np.float32)
    connector_only = (
        _masked_summary(connector, visible)
        if connector is not None
        else np.zeros((len(local), 1), dtype=np.float32)
    )
    confidence_only = (
        _masked_summary(
            np.asarray(observations.confidence, dtype=np.float64), visible
        )
        if observations.confidence is not None
        else None
    )
    residual = cross_fitted_protocol_residual(
        observations, calibration, test, seed + 1
    )
    proposed_blocks = [
        transcript,
        local_only,
        connector_only,
        _residual_state_features(local, residual, visible),
    ]
    if connector is not None:
        proposed_blocks.append(
            _residual_state_features(connector, residual, visible)
        )

    shuffled_acceptance = stratified_protocol_shuffle(
        observations, calibration, test, seed + 2
    )
    shuffled_observations = ProtocolObservations(
        local_state=local,
        acceptance=shuffled_acceptance,
        q_logp=observations.q_logp,
        visible=visible,
        depth=observations.depth,
        confidence=observations.confidence,
        connector_state=connector,
    )
    shuffled_residual = cross_fitted_protocol_residual(
        shuffled_observations, calibration, test, seed + 3
    )
    shuffled_blocks = [
        transcript_features(shuffled_observations, block_size),
        local_only,
        connector_only,
        _residual_state_features(local, shuffled_residual, visible),
    ]
    if connector is not None:
        shuffled_blocks.append(
            _residual_state_features(connector, shuffled_residual, visible)
        )
    families = {
        "speculator_state_only_triplet": local_only,
        "connector_only_triplet": connector_only,
        "transcript_only_triplet": transcript,
        "naive_protocol_concat_triplet": np.column_stack(
            [local_only, connector_only, transcript]
        ).astype(np.float32),
        "draver_protocol_residual_triplet": np.column_stack(
            proposed_blocks
        ).astype(np.float32),
        "control_stratified_shuffle_triplet": np.column_stack(
            shuffled_blocks
        ).astype(np.float32),
    }
    if confidence_only is not None:
        families["confidence_only_triplet"] = confidence_only
    return families


def evaluate_protocol_audit(
    observations: ProtocolObservations,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    block_size: int,
    bootstrap_repeats: int,
    detector_seeds: int,
    seed: int,
) -> dict[str, Any]:
    families = build_protocol_feature_families(
        observations, calibration, test, seed + 10, block_size
    )
    seeds: Iterable[int] = (
        seed + 1000 + offset for offset in range(detector_seeds)
    )
    seeds = tuple(seeds)
    test_labels = labels[test]
    scores: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    stability: dict[str, dict[str, Any]] = {}
    for offset, (name, features) in enumerate(families.items()):
        by_seed = triplet_scores_by_seed(
            features, labels, calibration, test, seeds
        )
        score = by_seed.mean(axis=0)
        seed_aucs = [float(auc_rank(test_labels, row)) for row in by_seed]
        scores[name] = score
        metrics[name] = metric_row(
            test_labels, score, bootstrap_repeats, seed + 2000 + offset
        )
        stability[name] = {
            "auc_by_seed": seed_aucs,
            "auc_mean": float(np.mean(seed_aucs)),
            "auc_std": float(np.std(seed_aucs, ddof=1))
            if len(seed_aucs) > 1
            else 0.0,
        }
    proposed_name = "draver_protocol_residual_triplet"
    comparisons = {}
    baselines = [
        "transcript_only_triplet",
        "speculator_state_only_triplet",
        "connector_only_triplet",
        "naive_protocol_concat_triplet",
        "control_stratified_shuffle_triplet",
    ]
    if "confidence_only_triplet" in scores:
        baselines.append("confidence_only_triplet")
    for offset, baseline in enumerate(baselines):
        comparisons[f"{proposed_name}_minus_{baseline}"] = paired_bootstrap_delta(
            test_labels,
            scores[proposed_name],
            scores[baseline],
            bootstrap_repeats,
            seed + 3000 + offset,
        )
    return {
        "metrics": metrics,
        "detector_stability": stability,
        "paired_auc_deltas": comparisons,
        "protocol": {
            "block_size": block_size,
            "visible_fraction": float(np.mean(observations.visible)),
            "connector_exposed": observations.connector_state is not None,
            "confidence_exposed": observations.confidence is not None,
            "detector_seed_count": detector_seeds,
        },
    }
