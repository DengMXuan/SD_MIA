import numpy as np
import pytest

from experiments.sd_membership_sft.method_probe import (
    QCOND_BINS,
    build_paraphrased_records,
    bottom_k_fraction_means,
    calibration_layer_statistics,
    histogram_features,
    projection_features,
    q_conditional_transcript,
    restrict_to_long_responses,
    top_k_fraction_means,
)
from experiments.sd_membership_sft.data import SFTRecord


def _record(index: int, length: int) -> SFTRecord:
    ids = tuple(range(100 + index, 100 + index + length))
    return SFTRecord(
        record_id=f"record-{index}",
        source=f"https://example.test/{index}",
        response_ids=ids,
        response_hash=str(index),
        topic="domain",
    )


def test_bottom_and_top_k_fraction_means_use_ceil() -> None:
    values = np.array([[5.0, 1.0, 3.0], [9.0, 7.0, 8.0]])
    bottoms = bottom_k_fraction_means(values, (0.3, 0.6))
    tops = top_k_fraction_means(values, (0.3, 0.6))
    assert bottoms.shape == (2, 2)
    assert tops.shape == (2, 2)
    assert bottoms[0, 0] == pytest.approx(1.0)
    assert bottoms[0, 1] == pytest.approx(2.0)
    assert bottoms[1, 0] == pytest.approx(7.0)
    assert tops[0, 0] == pytest.approx(5.0)
    assert tops[0, 1] == pytest.approx(4.0)
    assert tops[1, 1] == pytest.approx(8.5)


def test_q_conditional_transcript_matches_manual_bin_means() -> None:
    rng = np.random.default_rng(0)
    acceptance = rng.random((6, 8)).astype(np.float32)
    q_logp = rng.normal(size=(6, 8))
    calibration = np.array([0, 1, 2], dtype=np.int64)
    test = np.array([3, 4, 5], dtype=np.int64)
    features = q_conditional_transcript(acceptance, q_logp, calibration, test)
    assert features.shape == (6, 12 + QCOND_BINS + 10)

    edges = np.unique(
        np.quantile(q_logp[np.sort(calibration)].reshape(-1), np.linspace(0, 1, QCOND_BINS + 1))
    )
    n_bins = len(edges) - 1
    assert features.shape == (6, 12 + n_bins + 10)
    conditional = features[:, 12 : 12 + n_bins]
    first_bin = np.digitize(q_logp, edges[1:-1]) == 0
    manual = np.array(
        [
            acceptance[row][first_bin[row]].mean() if first_bin[row].any() else 0.0
            for row in range(6)
        ]
    )
    assert conditional[:, 0] == pytest.approx(manual)
    assert np.isfinite(features).all()


def test_q_conditional_transcript_handles_out_of_range_test_q() -> None:
    acceptance = np.full((4, 5), 0.5, dtype=np.float32)
    q_logp = np.array(
        [
            [-10.0, -9.0, -8.0, -7.0, -6.0],
            [-10.0, -9.0, -8.0, -7.0, -6.0],
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [10.0, 9.0, 8.0, 7.0, 6.0],
        ]
    )
    calibration = np.array([0, 1], dtype=np.int64)
    test = np.array([2, 3], dtype=np.int64)
    features = q_conditional_transcript(acceptance, q_logp, calibration, test)
    assert np.isfinite(features).all()


def test_calibration_layer_statistics_uses_calibration_rows_only() -> None:
    raw = np.arange(48, dtype=np.float32).reshape(4, 2, 3, 2)
    mean, std = calibration_layer_statistics(raw, np.array([0, 1], dtype=np.int64))
    subset = raw[[0, 1]]
    assert mean.shape == (1, 1, 3, 2)
    assert std.shape == (1, 1, 3, 2)
    assert mean == pytest.approx(subset.mean(axis=(0, 1), keepdims=True))
    assert std == pytest.approx(subset.std(axis=(0, 1), keepdims=True) + 1e-6)
    broadcast = (raw - mean) / std
    assert broadcast.shape == raw.shape
    assert np.isfinite(broadcast).all()


def test_histogram_features_sums_to_one_and_clips() -> None:
    states = np.zeros((3, 2, 2, 16), dtype=np.float32)
    states[:, :, :, :4] = 100.0
    states[:, :, :, 4:] = -100.0
    hist = histogram_features(states, bins=8, bounds=(-1.0, 1.0))
    assert hist.shape == (3, 2, 2, 8)
    assert hist.sum(axis=-1) == pytest.approx(np.ones((3, 2, 2)))
    assert hist[..., -1] == pytest.approx(np.full((3, 2, 2), 0.25))
    assert hist[..., 0] == pytest.approx(np.full((3, 2, 2), 0.75))


def test_projection_features_deterministic_and_shaped() -> None:
    states = np.random.default_rng(1).normal(size=(5, 3, 4, 32)).astype(np.float32)
    left = projection_features(states, dim=7, seed=3)
    right = projection_features(states, dim=7, seed=3)
    other = projection_features(states, dim=7, seed=4)
    assert left.shape == (5, 3, 4, 7)
    assert left == pytest.approx(right)
    assert not np.allclose(left, other)


def test_build_paraphrased_records_truncates_and_preserves_order() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False):
            return {"input_ids": [(ord(ch) * 257) % 40000 for ch in text[:200]]}

    records = [_record(0, 10), _record(1, 10), _record(2, 10), _record(3, 10)]
    texts = ["abcdefghijKLM", "xyz", "", "mnopqr"]
    with pytest.raises(RuntimeError):
        build_paraphrased_records(records, texts, FakeTokenizer(), max_response_tokens=8)
    texts = ["abcdefghijKLM", "xyz", "short", "mnopqr"]
    rebuilt, report = build_paraphrased_records(
        records, texts, FakeTokenizer(), max_response_tokens=8
    )
    assert [record.record_id for record in rebuilt] == [record.record_id for record in records]
    assert len(rebuilt[0].response_ids) == 8
    assert rebuilt[1].response_ids == tuple((ord(ch) * 257) % 40000 for ch in "xyz")
    assert all(
        record.response_hash != _record(i, 10).response_hash
        for i, record in enumerate(rebuilt)
    )
    assert report["records_truncated"] == 1
    assert set(report["paraphrased_token_counts"]) == {"members", "nonmembers"}


def test_restrict_to_long_responses_remaps_indices() -> None:
    records = [_record(i, 10) for i in range(6)]
    short = [_record(i, 3 if i == 2 else 10) for i in range(6)]
    labels = np.array([1, 1, 1, 0, 0, 0])
    calibration = np.array([2, 0, 4])
    test = np.array([3, 5, 1])
    kept_c, kept_p, kept_labels, cal, tst, report = restrict_to_long_responses(
        records, short, labels, calibration, test, min_tokens=8
    )
    assert report["records_dropped_short"] == 1
    assert report["dropped_members"] == 1
    assert report["dropped_nonmembers"] == 0
    assert len(kept_c) == len(kept_p) == 5
    assert list(cal) == [0, 3] and list(tst) == [2, 4, 1]
    assert list(kept_labels) == [1, 1, 0, 0, 0]
    assert all(len(record.response_ids) >= 8 for record in kept_p)


def test_restrict_to_long_responses_noop_when_all_long() -> None:
    records = [_record(i, 10) for i in range(4)]
    labels = np.array([1, 1, 0, 0])
    calibration = np.array([0, 2])
    test = np.array([1, 3])
    kept_c, kept_p, kept_labels, cal, tst, report = restrict_to_long_responses(
        records, records, labels, calibration, test, min_tokens=8
    )
    assert report["records_dropped_short"] == 0
    assert kept_c == records and kept_p == records
    assert list(cal) == [0, 2] and list(tst) == [1, 3]
