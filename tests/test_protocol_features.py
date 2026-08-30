from __future__ import annotations

import numpy as np

from experiments.sd_membership_sft.protocol_features import (
    ProtocolObservations,
    build_protocol_feature_families,
    first_rejection_visibility,
    stratified_protocol_shuffle,
    transcript_features,
)
from experiments.sd_membership_sft.draver_activation import _transcript_summary


def _observations() -> ProtocolObservations:
    rng = np.random.default_rng(7)
    accepted = np.asarray(
        [
            [1, 1, 0, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    return ProtocolObservations(
        local_state=rng.normal(size=(8, 8, 3)).astype(np.float32),
        acceptance=accepted,
        q_logp=rng.normal(-2.0, 0.4, size=(8, 8)).astype(np.float32),
        confidence=rng.uniform(0.2, 0.9, size=(8, 8)).astype(np.float32),
        connector_state=rng.normal(size=(8, 8, 2)).astype(np.float32),
        visible=first_rejection_visibility(accepted, block_size=4),
        depth=np.tile(np.arange(4), 2),
    )


def test_visibility_stops_after_first_rejection() -> None:
    visible = first_rejection_visibility(
        np.asarray([[1, 1, 0, 1, 1, 0, 1, 1]]), block_size=4
    )
    np.testing.assert_array_equal(
        visible, np.asarray([[1, 1, 1, 0, 1, 1, 0, 0]], dtype=bool)
    )


def test_protocol_feature_families_include_sd_specific_controls() -> None:
    observations = _observations()
    families = build_protocol_feature_families(
        observations,
        calibration=np.asarray([0, 1, 2, 3]),
        test=np.asarray([4, 5, 6, 7]),
        seed=11,
        block_size=4,
    )
    assert set(families) == {
        "speculator_state_only_triplet",
        "connector_only_triplet",
        "transcript_only_triplet",
        "naive_protocol_concat_triplet",
        "draver_protocol_residual_triplet",
        "control_stratified_shuffle_triplet",
        "confidence_only_triplet",
    }
    assert all(values.shape[0] == 8 for values in families.values())
    assert all(np.isfinite(values).all() for values in families.values())


def test_stratified_shuffle_preserves_invisible_positions() -> None:
    observations = _observations()
    shuffled = stratified_protocol_shuffle(
        observations,
        calibration=np.asarray([0, 1, 2, 3]),
        test=np.asarray([4, 5, 6, 7]),
        seed=13,
        bins=2,
    )
    invisible = ~observations.visible
    np.testing.assert_array_equal(
        shuffled[invisible], observations.acceptance[invisible]
    )


def test_transcript_only_excludes_white_box_difficulty() -> None:
    observations = _observations()
    changed = ProtocolObservations(
        local_state=observations.local_state,
        acceptance=observations.acceptance,
        q_logp=observations.q_logp + 100.0,
        confidence=1.0 - observations.confidence,
        connector_state=observations.connector_state,
        visible=observations.visible,
        depth=observations.depth,
    )
    np.testing.assert_array_equal(
        transcript_features(observations, block_size=4),
        transcript_features(changed, block_size=4),
    )


def test_independent_draft_transcript_summary_has_no_q_input() -> None:
    acceptance = np.asarray(
        [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]], dtype=np.float32
    )
    features = _transcript_summary(acceptance)
    assert features.shape == (2, 12)
    assert np.isfinite(features).all()
