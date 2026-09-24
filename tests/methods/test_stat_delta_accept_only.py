import numpy as np

from experiments.sd_membership_sft.archive.accept_only_mia import BIT_STAT_NAMES, bit_statistics, fixed_position_bins, run_lengths, simulate_replay
from experiments.sd_membership_sft.archive.stat_delta_mia import DeltaData, chunk_features, document_features, feature_row


def _data() -> DeltaData:
    lengths = np.asarray([8, 12], dtype=np.int64)
    offsets = np.asarray([0, 8, 20], dtype=np.int64)
    delta = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    return DeltaData(
        labels=np.asarray([1, 0], dtype=np.int64),
        record_ids=np.asarray(["a", "b"]),
        lengths=lengths,
        offsets=offsets,
        delta=delta,
    )


def test_stat_pack_dimensions_are_fixed_and_s22_is_22() -> None:
    values = np.linspace(-1.0, 0.5, 40)
    assert feature_row(values, "S11")[0].shape == (11,)
    assert feature_row(values, "S18")[0].shape == (18,)
    assert feature_row(values, "S22")[0].shape == (22,)
    assert len(feature_row(values, "S22")[1]) == 22


def test_chunk_features_has_one_name_per_slot_and_feature() -> None:
    matrix, names = chunk_features(_data(), "S22", slots=4)
    assert matrix.shape == (2, 4, 22)
    assert len(names) == 4 * 22
    assert names[0].startswith("slot0_")
    assert names[-1].startswith("slot3_")


def test_accept_statistics_and_position_bins_use_only_bits() -> None:
    bits = np.asarray([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    assert run_lengths(bits, 1).tolist() == [2.0, 1.0, 2.0]
    assert bit_statistics(bits).shape == (len(BIT_STAT_NAMES),)
    assert np.all((bit_statistics(bits) >= 0.0) & np.isfinite(bit_statistics(bits)))
    assert fixed_position_bins(bits, 4).tolist() == [1.0, 0.0, 0.5, 1.0]


def test_accept_replay_is_deterministic_and_has_expected_shape() -> None:
    data = _data()
    first = simulate_replay(data.delta, data.lengths, data.offsets, 4, 20260909, 4)
    second = simulate_replay(data.delta, data.lengths, data.offsets, 4, 20260909, 4)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].shape == (2, 4, 17)
    assert first[1].shape == (2, 4, 4)


def test_document_features_are_record_aligned() -> None:
    features, names = document_features(_data(), "S11")
    assert features.shape == (2, 11)
    assert len(names) == 11
